import json
import datetime
import matplotlib.pyplot as plt
import optuna
import itertools
import logging
import field_codec_utils
import cv2
from logging_utils import root_logger
optuna.logging.set_verbosity(logging.WARNING)  

plt.rcParams['font.sans-serif'] = ['SimHei'] # 运行配置参数中的字体（font）为黑体（SimHei）

import common
from common import KB_DATA_PATH,MAX_NUMBER, model_op, conf_and_serv_info, service_info_dict
from common import MIN_DELAY,MAX_ACCURACY,MIN_RESOURCE
from AccuracyPrediction import AccuracyPrediction

#该文件作用是利用refer_kb_name里的区间知识库，来构造kb_name的区间知识库

          
# KnowledgeBaseUser可以基于建立好的知识库，提供冷启动所需的一系列策略
# 实际上它可以充分使用知识库
class  KnowledgeBaseUser():   
    
    #冷启动计划者，初始化时需要conf_names,serv_names,service_info_list,user_constraint一共四个量
    #在初始化过程中，会根据这些参数，制造conf_list，serv_ip_list，serv_cpu_list以及serv_meme_list
    def __init__(self, conf_names, serv_names, service_info_list, user_constraint, rsc_constraint, rsc_upper_bound, rsc_down_bound, work_condition, portrait_info, bandwidth_dict,kb_name,refer_kb_name):
        self.conf_names = conf_names
        self.serv_names = serv_names
        self.service_info_list = service_info_list
        self.user_constraint = user_constraint
        self.rsc_constraint = rsc_constraint
        self.rsc_upper_bound = rsc_upper_bound
        self.rsc_down_bound = rsc_down_bound
        self.work_condition = work_condition
        self.portrait_info = portrait_info
        self.bandwidth_dict = bandwidth_dict
        self.kb_name = kb_name
        self.refer_kb_name = refer_kb_name
        
        self.GD_prior_queue = list() #梯度下降时的优先队列，梯度下降时从中提取下一个有潜力的区间信息
        self.GD_used_set = set() #梯度下降时存储已经走过的区间

        # 工况和画像均为空，则此时知识库用于冷启动；否则用于正常调度
        self.cold_start_flag = True if not self.work_condition and not self.portrait_info else False
        
    # get_section_info
    # 用途：读取配置文件中的section_info信息
    def get_section_info(self):
                #获取section_info以供后续使用
        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name
        section_info={}
        with open(self.refer_kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
            #print('打开知识库section_info:',self.kb_name + '/' + dag_name + '/' + 'section_info.json')
            section_info=json.load(f) 
        f.close()


        # 获取self.refer_kb_name对应的section_info后（完整知识库），需要修改其中的section_ids部分，这样才能与kb_name相匹配
        section_info2={}
        with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
            #print('打开知识库section_info:',self.kb_name + '/' + dag_name + '/' + 'section_info.json')
            section_info2=json.load(f) 
        f.close()

        section_info['section_ids']=section_info2['section_ids']

        return section_info
    
    # judge_section
    # 用途：判断一个conf配置处于知识库中的哪一个分区，返回分区编号
    # 返回值：result，分别存储了分区编号，以及该分区当前有没有被存储到section_info之中
    '''
    result={
            'section_id':"reso=0-fps=0-encoder=0",
            'exist':True
        }
    '''
    def judge_section(self,conf,section_info):

        conf_sections=section_info['conf_sections']
        section_id=''
        for i in range(0,len(self.conf_names)):
            conf_name=self.conf_names[i]
            if i>0:
                section_id+='-'
            sec_choice='0'
            for key in conf_sections[conf_name].keys():
                if conf[conf_name] in conf_sections[conf_name][key]:
                    sec_choice=key
                    break
            section_id=section_id + self.conf_names[i]+'='+sec_choice
        #print('该配置所在区间id为',section_id)

        result={}
        result['section_id']=section_id

        result['exist']=False
        if section_id in section_info['section_ids']:
            result['exist']=True

        return result

    # get_pred_rsc_in_section
    # 用途：在指定分区中判断某一个配置的资源消耗
    # 方法：目前就是单纯计算边缘端上的资源消耗
    def get_pred_rsc_in_section(self, section_id, conf, flow_mapping, resource_limit, edge_cloud_cut_choice):
        edge_cpu_use=0
        for serv_name in self.serv_names:
            if flow_mapping[serv_name]['node_role']!='cloud':
                edge_cpu_use+=resource_limit[serv_name]['cpu_util_limit']
        edge_cpu_use=round(edge_cpu_use,2)
        return edge_cpu_use

    # get_pred_delay_in_section
    # 用途：在指定分区中判断某一个配置对应的时延
    # 方法：直接加载分区相关知识库，不需要section_info信息
    def get_pred_delay_in_section(self, section_id, conf, flow_mapping, resource_limit, edge_cloud_cut_choice):
        # 存储配置对应的各阶段时延，以及总时延
        pred_delay_list = []
        pred_delay_total = 0
        status = 0  # 为0表示字典中没有满足配置的存在


        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name

        # 对于service_info_list里的service_info依次评估性能
        for service_info in self.service_info_list:
            # （1）加载服务对应的性能评估器。注意使用refer_kb_name
            f = open(self.refer_kb_name+'/'+dag_name+'/'+service_info['name']+'/'+section_id+".json")  
            evaluator = json.load(f)
            f.close()
            # （2）获取service_info里对应的服务配置参数，从参数conf中获取该服务配置旋钮的需要的各个值，加上ip选择
            # 得到形如["360p"，"1","JPEG","114.212.81.11"]的conf_for_dict，用于从字典中获取性能指标
            service_conf = list(service_info['conf']) # 形如":["reso","fps","encoder"]
                # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
            conf_for_dict = []
                #形如：["360p"，"1","JPEG"]
            for service_conf_name in service_conf:
                conf_for_dict.append(str(conf[service_conf_name]))   
            
            # 完成以上操作后，conf_for_dict内还差ip地址,首先要判断当前评估器是不是针对传输阶段进行的：
            ip_for_dict_index = service_info['name'].find("_trans") 
            if ip_for_dict_index > 0:
                # 当前是trans，则去除服务名末尾的_trans，形如“face_detection”
                ip_for_dict_name = service_info['name'][0:ip_for_dict_index] 
            else: # 当前不是trans
                ip_for_dict_name = service_info['name']
            ip_for_dict = flow_mapping[ip_for_dict_name]['node_ip']
                
            cpu_for_dict = resource_limit[ip_for_dict_name]["cpu_util_limit"]
            mem_for_dict = resource_limit[ip_for_dict_name]["mem_util_limit"]
            
            conf_for_dict.append(str(ip_for_dict))  
            conf_for_dict.append(str(cpu_for_dict)) 
            conf_for_dict.append(str(mem_for_dict)) 

            service_conf.append(service_info['name']+'_ip')
            service_conf.append(service_info['name']+'_cpu_util_limit')
            service_conf.append(service_info['name']+'_mem_util_limit')
            # 形如["360p"，"1","JPEG","114.212.81.11","0.1"."0.1"]

            # （3）根据conf_for_dict，从性能评估器中提取该服务的评估时延
     
            dict_key=''
            for i in range(0, len(service_conf)):
                dict_key += service_conf[i] + '=' + conf_for_dict[i] + ' '

            
            if dict_key not in evaluator:
                #print('配置不存在',dict_key)
                return status, pred_delay_list, pred_delay_total
            
            pred_delay = evaluator[dict_key]
            
            #  (4) 如果pred_delay为0，意味着这一部分对应的性能估计结果不存在，该配置在知识库中没有找到合适的解。此时直接返回结果。
            if pred_delay == 0:
                return status, pred_delay_list, pred_delay_total
            
            #  (5)对预测出时延根据工况进行修改
            #  注意，此处获得的pred_delay_list和pred_delay_total里的时延都是单一工况下的，因此需要结合工况进行调整
            obj_n = 1
            if 'obj_n' in self.work_condition:
                obj_n = self.work_condition['obj_n']
            if service_info_dict[service_info['name']]["vary_with_obj_n"]:
                pred_delay = pred_delay * obj_n

            # （6）将预测的时延添加到列表中
            pred_delay_list.append(pred_delay)

        # 计算总处理时延
        for pred_delay in pred_delay_list:
            pred_delay_total += pred_delay
        
        
        ############ 在当前带宽状态下预估传输时延 ############
        if not self.cold_start_flag:  # 非冷启动时才可预测传输时延，因为冷启动时没有工况、画像、视频帧信息，无法预估传输的数据量，也无法预估传输时延
            index = edge_cloud_cut_choice
            edge_to_cloud_data = 0
            while index < len(self.serv_names):
                edge_to_cloud_data += self.portrait_info['data_trans_size'][self.serv_names[index]]  # 以上次执行时各服务之间数据的传输作为当前传输的数据量
                index += 1
            
            if edge_cloud_cut_choice == 0:  # 此时需要将图像传到云端，图像的数据量很大，需要较精确的计算
                pre_frame_str_size = len(self.portrait_info['frame'])
                pre_frame = field_codec_utils.decode_image(self.portrait_info['frame'])
                temp_width = common.resolution_wh[conf['reso']]['w']
                temp_height = common.resolution_wh[conf['reso']]['h']
                temp_frame = cv2.resize(pre_frame, (temp_width, temp_height))
                temp_frame_str_size = len(field_codec_utils.encode_image(temp_frame))
                
                edge_to_cloud_data -= pre_frame_str_size
                edge_to_cloud_data += temp_frame_str_size
            pred_delay_total += edge_to_cloud_data / (self.bandwidth_dict['kB/s'] * 1000)
        
        
        status = 1
        return status, pred_delay_list, pred_delay_total  # 返回各个部分的时延

    # get_pred_delay：
    # 用途：根据参数里指定的配置，根据知识库来预测对应性能，如果在知识库中找不到，则返回status为0；否则为1
    # 方法：先判断配置所在分区，然后再返回结果。需要section_info来判断某个分区是否存在。
    # 返回值：status,pred_delay_list,pred_delay_total，描述性能评估成果与否、预测的各阶段时延、预测的总时延
    def get_pred_delay(self, conf, flow_mapping, resource_limit, edge_cloud_cut_choice, section_info):

        pred_delay_list = []
        pred_delay_total = 0
        status = 0  # 为0表示字典中没有满足配置的存在

        result=self.judge_section(conf=conf,section_info=section_info)
        if not result['exist']: #根本不存在这样的分区
            return status, pred_delay_list, pred_delay_total
        
        # 如果存在这样的分区：
        section_id = result['section_id']

        status, pred_delay_list, pred_delay_total=self.get_pred_delay_in_section(section_id=section_id,conf=conf,flow_mapping=flow_mapping,\
                                                                                 resource_limit=resource_limit,edge_cloud_cut_choice=edge_cloud_cut_choice)
        return status, pred_delay_list, pred_delay_total
    
    # get_plan_description：
    # 用途：获取一个配置的精度、预测时延、违反资源约束程度。对于任意调度计划，直接使用这个函数就可以获得对这个计划的完整性能预测结果
    # 返回值：status, plan_description，status表示该计划在知识库中是否存在，任务描述形如：
    '''
    {
        'pred_delay_list': [0.044744093992091935, 0.0018706620685637382], 
		'pred_delay_total': 0.055844776060655674, 
        'task_accuracy': 0.7949547259749911,
		'deg_violate': 0, 
        'edge_cpu_use_total':0.27
    }
    '''
    def get_plan_description(self,conf,flow_mapping,resource_limit,edge_cloud_cut_choice,section_info):
        # 0、初始化
        plan_description={}
        plan_description['pred_delay_list']=[]
        plan_description['pred_delay_total']=0
        plan_description['deg_violate']=0
        plan_description['task_accuracy']=0

        # 1、计算时延
        status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf,
                                                                        flow_mapping=flow_mapping,
                                                                        resource_limit=resource_limit,
                                                                        edge_cloud_cut_choice=edge_cloud_cut_choice,
                                                                        section_info=section_info)
        if status==0:
           status=0 #没有配置可用
           return status, plan_description
        

        # 2、计算精度
        task_accuracy = 1.0
        acc_pre = AccuracyPrediction()
        obj_size = None if 'obj_size' not in self.work_condition else self.work_condition['obj_size']
        obj_speed = None if 'obj_speed' not in self.work_condition else self.work_condition['obj_speed']
        for serv_name in serv_names:
            if service_info_dict[serv_name]["can_seek_accuracy"]:
                task_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                    'fps':conf['fps'],
                    'reso':conf['reso']
                }, obj_size=obj_size, obj_speed=obj_speed)
  

        # 3、计算资源约束违反程度
        deg_violate = 0 #违反资源约束的程度
        for device_ip in self.rsc_constraint.keys():
            # 只针对非云设备计算违反资源约束程度
            if model_op[device_ip]['node_role'] != 'cloud':
                cpu_util = 0
                mem_util = 0
                for serv_name in resource_limit.keys():
                    if flow_mapping[serv_name]['node_ip'] == device_ip:
                        # 必须用round保留小数点，因为python对待浮点数不精确，0.35加上0.05会得到0.39999……
                        cpu_util = round(cpu_util + resource_limit[serv_name]['cpu_util_limit'], 2)
                        mem_util = round(mem_util + resource_limit[serv_name]['mem_util_limit'], 2)
                cpu_util_ratio = float(cpu_util) / float(self.rsc_constraint[device_ip]['cpu'])
                mem_util_ratio = float(mem_util) / float(self.rsc_constraint[device_ip]['mem'])
                
                if cpu_util_ratio > 1:
                    deg_violate += cpu_util_ratio
                if mem_util_ratio > 1:
                    deg_violate += mem_util_ratio
        

       # 4、计算边端资源总消耗量
        edge_cpu_use_total=0.0 #边端cpu资源消耗总量
        for serv_name in self.serv_names:
            if flow_mapping[serv_name]["node_role"] != "cloud": 
                edge_cpu_use_total+=resource_limit[serv_name]["cpu_util_limit"] 
        edge_cpu_use_total=round(edge_cpu_use_total,2)

        plan_description['pred_delay_list']=pred_delay_list
        plan_description['pred_delay_total']=pred_delay_total
        plan_description['task_accuracy']=task_accuracy
        plan_description['deg_violate']=deg_violate
        plan_description['edge_cpu_use_total']=edge_cpu_use_total
        
        
        status = 1 

        return status, plan_description

    # get_section_ids_in_cons：
    # 用途：根据当前的conf_and_serv_info里各个配置的范围，找出能够涵盖这些范围的区间，并分类为已经建立的区间和没有建立的区间
    #       需要section_info来进行查找
    # 返回值：1个列表section_ids_in_cons。列表里是已经采样完成的各个区间，并进行排序。
    def get_section_ids_in_cons(self,section_info,optimze_goal,section_ids_sub=None):

        section_ids=[]
        if section_ids_sub==None:
            section_ids = section_info['section_ids']
        else:
            section_ids = section_ids_sub
       
        # (1)筛选满足精度约束、时延约束、资源约束的分区，求出右上角精度、左下角最小时延、左下角最小资源,记录在section_ids_in_cons之中
        section_ids_info_in_cons=[]
        for section_id in section_ids:
            # 计算当前分区内下各个配置的取值范围
            sec_conf_range={}

            judge1=True #判断区间范围是否在conf_ande_serv_info之内
            judge2=True #右上角精度是否满足约束
            judge3=True #左下角资源充分时的最小时延是否满足约束
            # 以上三个条件都满足的时候，这个区间才能够进入section_ids_info_in_cons列表之中

            top_right_conf={}
            bottom_left_conf={}
            top_right_accuracy = 1.0
            bottom_left_accuracy = 1.0
            bottom_left_min_delay = MAX_NUMBER
            bottom_left_min_rsc = MAX_NUMBER

            #（1.1）第一轮审查：区间范围是否在conf_ande_serv_info之内
            for part in section_id.split('-'):
                # 使用 split('=') 分割键值对  
                conf_name, value = part.split('=')  
                # 将键值对添加到字典中  
                sec_conf_range[conf_name] = section_info['conf_sections'][conf_name][value]
                bottom_left_conf[conf_name] = sec_conf_range[conf_name][0]
                top_right_conf[conf_name] = sec_conf_range[conf_name][-1]
                

                set1 = set(conf_and_serv_info[conf_name])
                set2 = set(sec_conf_range[conf_name])
                # 检查交集是否为空  
                if not set1.intersection(set2):  
                    print("该区间不在conf_and_serv_info约束中") 
                    judge1=False
                    break
            
        

            #(1.2)第二轮审查：右上角是否大于等于精度约束
            if judge1: 
                acc_pre = AccuracyPrediction()
                obj_size = None if 'obj_size' not in self.work_condition else self.work_condition['obj_size']
                obj_speed = None if 'obj_speed' not in self.work_condition else self.work_condition['obj_speed']
                for serv_name in serv_names:
                    if service_info_dict[serv_name]["can_seek_accuracy"]:
                        # 计算右上角精度
                        top_right_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                            'fps': top_right_conf['fps'],
                            'reso':top_right_conf['reso']
                        }, obj_size=obj_size, obj_speed=obj_speed)
                        # 计算左下角精度，后续要用
                        bottom_left_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                            'fps': bottom_left_conf['fps'],
                            'reso':bottom_left_conf['reso']
                        }, obj_size=obj_size, obj_speed=obj_speed)
                # 最后根据右上角精度判断区间是否能满足精度约束
                if top_right_accuracy < self.user_constraint['accuracy']:
                    judge2 =False
            
            #(1.3)第三轮审查：计算左下角各个配置，求左下角所有配置对应的最小时延，以及最小资源消耗
            if judge2:
                bottom_left_plans = section_info[section_id]['bottom_left_plans']
                for plan in bottom_left_plans:
                    #print('当前plan',plan)

                    plan_delay=0
                    plan_rsc=0

                    conf=plan['conf']
                    flow_mapping=plan['flow_mapping']
                    resource_limit=plan['resource_limit']
                    edge_cloud_cut_choice=0
                    for key in flow_mapping.keys():
                        if flow_mapping[key]['node_role']=='cloud':
                            break
                        else:
                            edge_cloud_cut_choice+=1
                    #print('当前云边切分',edge_cloud_cut_choice)

                    status, pred_delay_list, plan_delay=self.get_pred_delay_in_section(section_id=section_id,conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit,edge_cloud_cut_choice=edge_cloud_cut_choice)
                    #print(status, pred_delay_list, plan_delay)
                    plan_rsc=self.get_pred_rsc_in_section(section_id=section_id,conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit,edge_cloud_cut_choice=edge_cloud_cut_choice)
                    if status!=0: # 配置可能不存在
                        bottom_left_min_delay = min (plan_delay,bottom_left_min_delay)
                        bottom_left_min_rsc = min (plan_rsc,bottom_left_min_rsc)
                # 完成以上操作后就得到了左下角最小时延和最小配置了，如果最小时延都不满足约束就说明区间完蛋了。
                if bottom_left_min_delay > self.user_constraint['delay'] or bottom_left_min_delay==MAX_NUMBER:
                    #如果资源充分的时候依然时延超标，或者所有plan在知识库里都查不到，就说明judeg3为False，审查不通过
                    #print('当前左下角最小时延是',bottom_left_min_delay)
                    judge3=False
            
            #(1.4)看看judge各个情况
            #print(section_id,judge1,judge2,judge3)
            # section_ids_info_in_cons=[]
            '''
            top_right_accuracy = 1.0
            bottom_left_accuracy = 1.0
            bottom_left_min_delay = MAX_NUMBER
            bottom_left_min_rsc = MAX_NUMBER
            '''
            if judge1 and judge2 and judge3:
                section_id_info={}
                section_id_info['section_id'] = section_id
                section_id_info['top_right_accuracy'] = top_right_accuracy
                section_id_info['bottom_left_accuracy'] =  bottom_left_accuracy
                section_id_info['bottom_left_min_delay'] = bottom_left_min_delay
                section_id_info['bottom_left_min_rsc'] = bottom_left_min_rsc
                section_ids_info_in_cons.append(section_id_info)

        # 最后根据优化目标排序
        if optimze_goal==MAX_ACCURACY:
            # 按照左下角精度从大到小排序
            sorted_list = sorted(section_ids_info_in_cons, key=lambda item: item['bottom_left_accuracy'], reverse=True) 
        elif optimze_goal==MIN_DELAY:
            # 按照左下角时延从小到大排序
            sorted_list = sorted(section_ids_info_in_cons, key=lambda item: item['bottom_left_min_delay'], reverse=False) 
        elif optimze_goal==MIN_RESOURCE:
            # 按照左下角资源从小到大排序
            sorted_list = sorted(section_ids_info_in_cons, key=lambda item: item['bottom_left_min_rsc'], reverse=False) 
        
        section_ids_in_cons=[section_id_info['section_id'] for section_id_info in section_ids_info_in_cons]

        return section_ids_in_cons
    
    # 由trial给出下一组要尝试的参数并返回，需要section_info来限制取值范围
    def get_next_params(self, trial,section_ids,section_info, optimize_goal):

        mul_objects = []

         #（1）开始计算当前的各种配置
        conf = {}
        flow_mapping = {}
        resource_limit = {}
        # 首先确定section_id
        if len(section_ids)==0:
            status=0 #没有配置可用
            return status, conf, flow_mapping, resource_limit, mul_objects

        section_id = trial.suggest_categorical('section_id',section_ids) 
        #print('选择分区是',section_id)
        #然后确定各个配置的取值。要选取当前conf_and_serv_info和区间conf_info里重合的部分
        no_conf=False # 为True表示无配置可选
        for conf_name in self.conf_names:
            # 将列表转换为集合  
            set1 = set(conf_and_serv_info[conf_name])  
            set2 = set(section_info[section_id]['conf_info'][self.serv_names[0]][conf_name])
            # 计算两个集合的交集，并转换回列表  
            conf_select_range = list(set1.intersection(set2))
            #print(conf_name,"采样范围是:", conf_select_range)
            if len(conf_select_range)>0:
                # optuna不允许相同名称的超参数拥有不同的取值范围，所以此处才需要在后续加上区间编号，区分不同的超参数
                conf[conf_name] = trial.suggest_categorical(conf_name + section_id,conf_select_range) 
            else:
                no_conf=True 
                break
        
        if no_conf:
            status=0 #没有配置可用
            return status, conf, flow_mapping, resource_limit, mul_objects
        
        conf_and_serv_info["edge_cloud_cut_point"]=[i for i in range(len(self.serv_names) + 1)]  #[0,1,2]
        choice_idx = trial.suggest_int('edge_cloud_cut_choice', 0, len(conf_and_serv_info["edge_cloud_cut_point"])-1) #[0,1,2]是左右包含的，所以减去1
        edge_cloud_cut_choice=conf_and_serv_info["edge_cloud_cut_point"][choice_idx]

        # 确定调度计划的资源限制
        for serv_name in self.serv_names:
            if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                flow_mapping[serv_name] = model_op[common.edge_ip]
            else:  # 服务索引大于等于云边协同切分点，在云执行
                flow_mapping[serv_name] = model_op[common.cloud_ip]

            serv_cpu_limit = serv_name + "_cpu_util_limit"
            serv_mem_limit = serv_name + "_mem_util_limit"

            resource_limit[serv_name] = {}
            if flow_mapping[serv_name]["node_role"] == "cloud":  
                resource_limit[serv_name]["cpu_util_limit"] = 1.0
                resource_limit[serv_name]["mem_util_limit"] = 1.0
            else:
                # 边缘端每一个服务的cpu资源
                device_ip = flow_mapping[serv_name]["node_ip"]
                cpu_upper_limit = min(self.rsc_constraint[device_ip]['cpu'], self.rsc_upper_bound[serv_name]['cpu_limit'])
                cpu_down_limit = max(0.0, self.rsc_down_bound[serv_name]['cpu_limit'])

                conf_info = section_info[section_id]['conf_info'][serv_name]
            
                cpu_choice_range = [item for item in conf_info[serv_cpu_limit] if item <= cpu_upper_limit and item >= cpu_down_limit]

                # 要防止资源约束导致取值范围为空的情况
                if len(cpu_choice_range) == 0:
                    cpu_choice_range = [item for item in conf_info[serv_cpu_limit]]

                resource_limit[serv_name]["cpu_util_limit"] = trial.suggest_categorical(serv_cpu_limit + section_id, cpu_choice_range)
                resource_limit[serv_name]["mem_util_limit"] = 1.0

            
        #（2）得到配置后，计算该配置对应的时延，精度，资源消耗，并根据optimze_goal，来确定四个优化目标
        #status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf, flow_mapping=flow_mapping, resource_limit=resource_limit, edge_cloud_cut_choice=edge_cloud_cut_choice)
        status, plan_description = self.get_plan_description(conf=conf,
                                                             flow_mapping=flow_mapping,
                                                             resource_limit=resource_limit,
                                                             edge_cloud_cut_choice=edge_cloud_cut_choice,
                                                             section_info=section_info)

        if status==0:
           status=0 #没有配置可用
           return status, conf, flow_mapping, resource_limit, mul_objects
        
        # (3) 根据约束满足情况制定三个约束,满足约束就取0，否则往大了走
        delay_constraint = self.user_constraint["delay"]
        accuracy_constraint = self.user_constraint["accuracy"]
        edge_cpu_constraint = self.rsc_constraint[common.edge_ip]['cpu']

        pred_delay_total = plan_description['pred_delay_total']
        task_accuracy = plan_description['task_accuracy']
        edge_cpu_use_total = plan_description['edge_cpu_use_total']

        if optimize_goal == MIN_DELAY: #目标为最小化时延的时候，最小化时延
            mul_objects.append(pred_delay_total)
        elif pred_delay_total <= 0.85 * delay_constraint: #否则，满足约束即可
            mul_objects.append(0)
        else: #不满足约束，那就越小越好，而且此时取值不会小于0
            mul_objects.append(pred_delay_total)
        
        if optimize_goal == MAX_ACCURACY: #目标为最大化精度，那精度越大越好
            mul_objects.append(1-task_accuracy) 
        elif task_accuracy >= accuracy_constraint: #否则，满足约束即可
            mul_objects.append(0)
        else: #满足约束，那就越大越好，而且此时取值不会小于0
            mul_objects.append(1-task_accuracy) #精度越大则差距越小
        
        if optimize_goal == MIN_RESOURCE: #目标为最小化资源，那资源消耗越少越好，需要一个计价函数。暂时设置为边缘端资源消耗量，此时会倾向于把任务卸载到云端，不知道意义何在
            mul_objects.append(edge_cpu_use_total)
        elif edge_cpu_use_total <= edge_cpu_constraint: #否则，满足约束即可
            mul_objects.append(0)
        else: #不满足约束，那就越小越好，而且此时取值不会小于0
            mul_objects.append(edge_cpu_use_total)

        status=1
        return status, conf, flow_mapping, resource_limit, mul_objects
    
    # get_next_params_in_sec
    # 用途：类似get_next_params，但是只从指定区间中进行一次贝叶斯优化，而且返回结果中包含对解的详细描述     
    # 返回值：status, ans_dict, ans_json, mul_objects，其中ans_dict格式如下,描述了当前所采样的配置方案的详细信息;ans_json将其转化为字符串形式
    '''
    {	
		'conf': {'reso': '480p', 'fps': 7, 'encoder': 'JPEG'}, 
		'flow_mapping': {
			'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
			'gender_classification': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}
		}, 
		'resource_limit': {
			'face_detection': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}, 
			'gender_classification': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}
		}, 
		'edge_cloud_cut_choice': 0, 
		'pred_delay_list': [0.044744093992091935, 0.0018706620685637382], 
		'pred_delay_total': 0.055844776060655674, 
		'deg_violate': 0, 
		'task_accuracy': 0.7949547259749911
	}
    '''
    def get_next_params_in_sec(self, trial, section_id, section_info, optimize_goal):

        mul_objects = []
        ans_dict = {}
        ans_json = ''

         #（1）开始计算当前的各种配置
        conf = {}
        flow_mapping = {}
        resource_limit = {}
        # 获取参数section_id中对应的各个conf
        no_conf=False # 为True表示无配置可选
        for conf_name in self.conf_names:
            # 将列表转换为集合  
            set1 = set(conf_and_serv_info[conf_name])  
            set2 = set(section_info[section_id]['conf_info'][self.serv_names[0]][conf_name])
            # 计算两个集合的交集，并转换回列表  
            conf_select_range = list(set1.intersection(set2))
            #print(conf_name,"采样范围是:", conf_select_range)
            if len(conf_select_range)>0:
                # optuna不允许相同名称的超参数拥有不同的取值范围，所以此处才需要在后续加上区间编号，区分不同的超参数
                conf[conf_name] = trial.suggest_categorical(conf_name, conf_select_range) 
            else:
                no_conf=True 
                break
        
        if no_conf:
            status=0 #没有配置可用
            return status, ans_dict, ans_json, mul_objects
        
        conf_and_serv_info["edge_cloud_cut_point"]=[i for i in range(len(self.serv_names) + 1)]  #[0,1,2]
        choice_idx = trial.suggest_int('edge_cloud_cut_choice', 0, len(conf_and_serv_info["edge_cloud_cut_point"])-1) #[0,1,2]是左右包含的，所以减去1
        edge_cloud_cut_choice=conf_and_serv_info["edge_cloud_cut_point"][choice_idx]

        # 求资源限制
        for serv_name in self.serv_names:
            if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                flow_mapping[serv_name] = model_op[common.edge_ip]
            else:  # 服务索引大于等于云边协同切分点，在云执行
                flow_mapping[serv_name] = model_op[common.cloud_ip]

            serv_cpu_limit = serv_name + "_cpu_util_limit"
            serv_mem_limit = serv_name + "_mem_util_limit"

            resource_limit[serv_name] = {}
            if flow_mapping[serv_name]["node_role"] == "cloud":  
                resource_limit[serv_name]["cpu_util_limit"] = 1.0
                resource_limit[serv_name]["mem_util_limit"] = 1.0
            else:
                # 边缘端每一个服务的cpu资源
                device_ip = flow_mapping[serv_name]["node_ip"]
                cpu_upper_limit = min(self.rsc_constraint[device_ip]['cpu'], self.rsc_upper_bound[serv_name]['cpu_limit'])
                cpu_down_limit = max(0.0, self.rsc_down_bound[serv_name]['cpu_limit'])

                conf_info = section_info[section_id]['conf_info'][serv_name]
            
                cpu_choice_range = [item for item in conf_info[serv_cpu_limit] if item <= cpu_upper_limit and item >= cpu_down_limit]

                # 要防止资源约束导致取值范围为空的情况
                if len(cpu_choice_range) == 0:
                    cpu_choice_range = [item for item in conf_info[serv_cpu_limit]]

                resource_limit[serv_name]["cpu_util_limit"] = trial.suggest_categorical(serv_cpu_limit, cpu_choice_range)
                resource_limit[serv_name]["mem_util_limit"] = 1.0

            
        #（2）得到配置后，计算该配置对应的时延，精度，资源消耗，并根据optimze_goal，来确定四个优化目标
        #status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf, flow_mapping=flow_mapping, resource_limit=resource_limit, edge_cloud_cut_choice=edge_cloud_cut_choice)
        status, plan_description = self.get_plan_description(conf=conf,
                                                             flow_mapping=flow_mapping,
                                                             resource_limit=resource_limit,
                                                             edge_cloud_cut_choice=edge_cloud_cut_choice,
                                                             section_info=section_info)

        if status == 0:
           status=0 #没有配置可用
           return status, ans_dict, ans_json, mul_objects
        
    
        # (3) 根据约束满足情况制定三个约束,满足约束就取0，否则往大了走
        delay_constraint = self.user_constraint["delay"]
        accuracy_constraint = self.user_constraint["accuracy"]
        edge_cpu_constraint = self.rsc_constraint[common.edge_ip]['cpu']

        pred_delay_total = plan_description['pred_delay_total']
        pred_delay_list = plan_description['pred_delay_list']
        task_accuracy = plan_description['task_accuracy']
        edge_cpu_use_total = plan_description['edge_cpu_use_total']
        deg_violate = plan_description['deg_violate']
    
        if optimize_goal == MIN_DELAY: #目标为最小化时延的时候，最小化时延
            mul_objects.append(pred_delay_total)
        elif pred_delay_total <= 0.85 * delay_constraint: #否则，满足约束即可
            mul_objects.append(0)
        else: #不满足约束，那就越小越好，而且此时取值不会小于0
            mul_objects.append(pred_delay_total)
        
        if optimize_goal == MAX_ACCURACY: #目标为最大化精度，那精度越大越好
            mul_objects.append(1-task_accuracy) 
        elif task_accuracy >= accuracy_constraint: #否则，满足约束即可
            mul_objects.append(0)
        else: #满足约束，那就越大越好，而且此时取值不会小于0
            mul_objects.append(1-task_accuracy) #精度越大则差距越小
        
        if optimize_goal == MIN_RESOURCE: #目标为最小化资源，那资源消耗越少越好，需要一个计价函数。暂时设置为边缘端资源消耗量，此时会倾向于把任务卸载到云端，不知道意义何在
            mul_objects.append(edge_cpu_use_total)
        elif edge_cpu_use_total <= edge_cpu_constraint: #否则，满足约束即可
            mul_objects.append(0)
        else: #不满足约束，那就越小越好，而且此时取值不会小于0
            mul_objects.append(edge_cpu_use_total)

        
        # 设置状态为1
        status=1

        # 但是还要计算ans_dict
        
        ans_dict['conf']=conf
        ans_dict['flow_mapping']=flow_mapping
        ans_dict['resource_limit']=resource_limit
        ans_dict['edge_cloud_cut_choice']=edge_cloud_cut_choice
        ans_dict['pred_delay_list']=pred_delay_list
        ans_dict['pred_delay_total']=pred_delay_total
        ans_dict['deg_violate']=deg_violate
        ans_dict['task_accuracy']=task_accuracy

        ans_json = json.dumps({
            'conf':ans_dict['conf'],
            'flow_mapping':ans_dict['flow_mapping'],
            'resource_limit':ans_dict['resource_limit'],
        })

        return status, ans_dict, ans_json, mul_objects
  
    # get_plan_in_cons_for_one_section
    # 用途：在特定区间内通过贝叶斯优化尝试获得最优解
    # 方法：专门建立一个优化器，特点是在计算优化目标的同时就完成对该配置的分类；
    #       按照ask_and_tell的方法，计算mul_objects和其他描述配置的内容，然后将其归类
    # 返回值：params_in_cons, params_out_cons。
    '''
    {	
		'conf': {'reso': '480p', 'fps': 7, 'encoder': 'JPEG'}, 
		'flow_mapping': {
			'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
			'gender_classification': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}
		}, 
		'resource_limit': {
			'face_detection': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}, 
			'gender_classification': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}
		}, 
		'edge_cloud_cut_choice': 0, 
		'pred_delay_list': [0.044744093992091935, 0.0018706620685637382], 
		'pred_delay_total': 0.055844776060655674, 
		'deg_violate': 0, 
		'task_accuracy': 0.7949547259749911
	}
    '''    
    def get_plan_in_cons_for_one_section(self, n_trials, optimize_goal, section_info, section_id):
        # 以下内容应该通过n_trials充分调用ask_and_tell，并设立终止条件，比如返回已经出现的结果，就立刻停止。或者，也可以在性能不断下降后终止

        params_in_cons = []
        params_out_cons = []
        params_set=set()
        study = optuna.create_study(directions=['minimize' for _ in range(3)], sampler=optuna.samplers.NSGAIISampler()) 
        n_num = 0
        while(n_num < n_trials):
            n_num+=1
            next_trial = study.ask()
            status, ans_dict, ans_json, mul_objects = self.get_next_params_in_sec(trial=next_trial,
                                                                        section_id=section_id,
                                                                        section_info=section_info,
                                                                        optimize_goal=optimize_goal)
            # 进行一次查询得到解后，分析其应该加入约束列表和中的哪一个
            if status!=0:
                # 得确保内容并没有重复,才能加入列表中
                if ans_json not in params_set:
                    params_set.add(ans_json)
                    if ans_dict['pred_delay_total'] <= 0.95*self.user_constraint["delay"] and ans_dict['task_accuracy']>=self.user_constraint["accuracy"] and ans_dict['deg_violate'] == 0 :
                            params_in_cons.append(ans_dict)
                    else:
                        params_out_cons.append(ans_dict)
        
        return params_in_cons, params_out_cons
            
    # get_plan_in_cons_from_sections：
    # 用途：从指定的一系列区间中获取能够满足约束的解，需要指定优化目标。
    #      理论上应该根据某些方式选出一系列区间后再使用它
    # 方法：通过贝叶斯优化，在有限轮数内选择一个最优的结果。需要section_info给get_next_params使用。
    # 返回值：满足约束的解；不满足约束的解；贝叶斯优化模型建议的下一组参数值
    def get_plan_in_cons_from_sections(self, n_trials, optimize_goal, section_info, section_ids_in_cons):
        # 贝叶斯权衡目标一共有3个，时延、精度、资源，根据优化目标，极致化其中一个，另外两个满足约束即可。

        # 使用ask_and_tell接口，首先进行贝叶斯优化。要实现3个目标（精度，时延，资源）之间的平衡
        study = optuna.create_study(directions=['minimize' for _ in range(3)], sampler=optuna.samplers.NSGAIISampler()) 

        n_num=0
        #print('开始',n_trials,'次采样')
        while(n_num < n_trials): #进行n_trials次采样
            n_num+=1
            '''
            if n_num%20==0:
                print('开始',n_num,'采样')
            '''
            next_trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.
            status, conf, flow_mapping, resource_limit, mul_objects = self.get_next_params(trial=next_trial,\
                                                                                           section_ids=section_ids_in_cons,\
                                                                                            section_info=section_info,\
                                                                                            optimize_goal=optimize_goal)
            if status == 0:
                continue
            #如果配置有意义，就继续采样
            study.tell(next_trial,mul_objects)
        #print('完成',n_trials,'次采样')

        # 完成以上全部操作后，就会得到最终的study，现在要从中获取帕累托最优解集
        ans_params = []
        trials = sorted(study.best_trials, key=lambda t: t.values) # 对字典best_trials按values从小到达排序

        # 此处的帕累托最优解可能重复，因此首先要提取内容，将其转化为字符串记录在ans_params中，然后从ans_params里删除重复项
        #print('开始处理帕累托配置')
        for trial in trials:
            conf = {}
            flow_mapping = {}
            resource_limit = {}
            section_id=trial.params['section_id']
            #print(trial.params)
           
            for conf_name in self.conf_names:   # conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
                # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
                conf[conf_name] = trial.params[conf_name+section_id]
            edge_cloud_cut_choice = trial.params['edge_cloud_cut_choice']
            
            for serv_name in self.serv_names:
                if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                    flow_mapping[serv_name] = model_op[common.edge_ip]
                else:  # 服务索引大于等于云边协同切分点，在云执行
                    flow_mapping[serv_name] = model_op[common.cloud_ip]

                serv_cpu_limit = serv_name + "_cpu_util_limit"
                serv_mem_limit = serv_name + "_mem_util_limit"
                resource_limit[serv_name] = {}

                if flow_mapping[serv_name]["node_role"] == "cloud":
                    resource_limit[serv_name]["cpu_util_limit"] = 1.0
                    resource_limit[serv_name]["mem_util_limit"] = 1.0
                else:
                    resource_limit[serv_name]["cpu_util_limit"] = trial.params[serv_cpu_limit+section_id]
                    resource_limit[serv_name]["mem_util_limit"] = 1.0
            
            #print('完成处理')
            
            ans_item = {}
            ans_item['conf'] = conf
            ans_item['flow_mapping'] = flow_mapping
            ans_item['resource_limit'] = resource_limit
            ans_item['edge_cloud_cut_choice'] = edge_cloud_cut_choice
            
            ans_params.append(json.dumps(ans_item))
            '''
            {   'reso': '720p', 'fps': 10, 'encoder': 'JPEG'}
            {   'face_detection': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}, 
                'face_alignment': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}}
            {   'face_detection': {'cpu_util_limit': 0.2, 'mem_util_limit': 0.45}, 
                'face_alignment': {'cpu_util_limit': 0.1, 'mem_util_limit': 0.45}}
            '''

        # 从ans_params里删除重复项，并选择真正status不为0的有效帕累托最优解返回。同时存储该配置对应的预估时延
        ans_params_set = list(set(ans_params))  

        # 接下来要咋办呢？帕累托最优解未必都满足约束。一共有三个约束要考虑，精度，时延，资源。
        # 将配置分为两大类，其一是三种约束都满足的，其二是不能满足全部约束的。排序什么的由调度器来决定。

        params_in_cons = []
        params_out_cons = []

        #print('进行配置分类')
        for param in ans_params_set:
            ans_dict = json.loads(param)
            conf = ans_dict['conf']
            flow_mapping = ans_dict['flow_mapping']
            resource_limit = ans_dict['resource_limit']
            edge_cloud_cut_choice = ans_dict['edge_cloud_cut_choice']

            # 获取该配置的任务描述，包括时延、精度、资源约束违反程度等
            status , plan_description = self.get_plan_description(conf=conf,
                                                                  flow_mapping=flow_mapping,
                                                                  resource_limit=resource_limit,
                                                                  edge_cloud_cut_choice=edge_cloud_cut_choice,
                                                                  section_info=section_info)
            # 配置存在的时候才会记录它
            if status != 0:
                ans_dict['pred_delay_list'] =  plan_description['pred_delay_list']
                ans_dict['pred_delay_total'] = plan_description['pred_delay_total']
                ans_dict['deg_violate'] =  plan_description['deg_violate']
                ans_dict['task_accuracy'] =  plan_description['task_accuracy']

                # 根据配置是否满足时延和精度约束，将其分为两类
                if  plan_description['pred_delay_total'] <= 0.95 * self.user_constraint["delay"] and \
                    plan_description['task_accuracy'] >= self.user_constraint["accuracy"] and \
                    plan_description['deg_violate'] == 0 :
                    
                    params_in_cons.append(ans_dict)
                else:
                    params_out_cons.append(ans_dict)
        
        # 构建贝叶斯优化模型建议的下一组参数值
        #print('获取下一组配置')
        next_trial = study.ask()
        status, next_conf, next_flow_mapping, next_resource_limit,mul_objects = self.get_next_params(trial=next_trial,\
                                                                                                     section_ids=section_ids_in_cons,\
                                                                                                     section_info=section_info,\
                                                                                                     optimize_goal=optimize_goal)
        # 为了防止陷入死循环，最多尝试10次
        try_num=0
        while(status==0 and try_num < 10):
            try_num+=1
            next_trial = study.ask()
            status, next_conf, next_flow_mapping, next_resource_limit,mul_objects = self.get_next_params(trial=next_trial,\
                                                                                                     section_ids=section_ids_in_cons,\
                                                                                                     section_info=section_info,\
                                                                                                     optimize_goal=optimize_goal)
       
        # 这下面next_plan用video_conf作为键，和前面的pramas_in_cons明显是不统一的，但是涉及到师兄在调度器里的接口，所以暂时不改了
        next_plan = {
            'video_conf': next_conf,
            'flow_mapping': next_flow_mapping,
            'resource_limit': next_resource_limit
        }
              
        return params_in_cons,params_out_cons,next_plan
 

    # 调用以上函数获取约束内配置
    def get_plan_in_cons(self,n_trials,optimze_goal,section_ids_sub=None):
        section_info = self.get_section_info()
        section_ids_in_cons=self.get_section_ids_in_cons(section_info=section_info,optimze_goal=optimze_goal,section_ids_sub=section_ids_sub)
        print('过滤后,能满足约束的区间有',len(section_ids_in_cons),'个')
        print(section_ids_in_cons)
        params_in_cons,params_out_cons,next_plan=self.get_plan_in_cons_from_sections(n_trials=n_trials,optimize_goal=optimze_goal,section_info=section_info,section_ids_in_cons=section_ids_in_cons)

        return params_in_cons,params_out_cons,next_plan

    # get_potential_score_for_param
    # 用途：为一个配置参数的潜力评分。用于指导梯度下降
    # 方法：判断参数是否在约束内，然后根据优化目标采取不同的评分方法。注意，约束内的得分一定要比约束外更高
    # 返回值：该配置的得分
    def get_potential_score_for_param(self,param, optimize_goal, if_in_cons):
        '''
        #  目前只考虑两种优化目标，最大化精度和最小化时延。
        #  满足约束时：根据其优化程度打分
        #  最大化精度的时候，打分为精度；
        #  最小化时延的时候，打分为1-当前时延/10。打分后加上1，确保一定比不满足约束的情况得分更高。如下实现中，大于1的就是打分结果。
        #  不满足约束是：根据满足约束的程度打分
        #  最大化精度的时候，时延、资源为约束，1 - [ max(0,实际时延-时延约束)/10 + 资源约束违反程度]/2, 越大越好
        #  最小化时延的时候，精度、资源为约束，1 - [ max(0,精度约束-实际精度) + 资源约束违反程度]/2, 越大越好
        '''
        potential_score = 0
        
        if if_in_cons: # 是否满足约束
            if optimize_goal == MAX_ACCURACY:
                potential_score = param['task_accuracy'] + 1
            elif optimize_goal == MIN_DELAY:
                potential_score = 1 - (param['pred_delay_total']/10.0) + 1
        else:
            if optimize_goal == MAX_ACCURACY:
                potential_score = 1 - (( max( 0 , param['pred_delay_total']-self.user_constraint['delay'] ) )/10.0 + param['deg_violate'] )/2.0
            elif optimize_goal == MIN_DELAY:
                potential_score = 1 - (( max( 0 , self.user_constraint['accuracy']-param['task_accuracy'] ) )      + param['deg_violate'] )/2.0
        
        return potential_score
    
    # get_sorted_potential_params
    # 用途：对params_in_cons,params_out_cons中的参数按照潜力进行排序，并合并为一个完整的列表
    def get_sorted_potential_params(self,params_in_cons,params_out_cons,optimize_goal):
        
        #评估各个配置的潜力并打分，放入列表中
        potential_params=[]
        for param in params_in_cons:
            param['potential_score'] = self.get_potential_score_for_param(param=param,
                                                                          optimize_goal=optimize_goal,
                                                                          if_in_cons=True)
            potential_params.append(param)
        for param in params_out_cons:
            param['potential_score'] = self.get_potential_score_for_param(param=param,
                                                                          optimize_goal=optimize_goal,
                                                                          if_in_cons=False)
            potential_params.append(param)
        
        #将列表按照潜力从大到小进行排序
        sorted_potential_params = sorted(potential_params, key=lambda x: x['potential_score'], reverse=True)  
        return sorted_potential_params
    
    # merge_sorted_potential_params
    # 用途：对已经完成排序的两个潜力参数列表进行合并，目的是为了减小排序代价，哎。
    def merge_sorted_potential_params(self,sorted_potential_params1,sorted_potential_params2):
        # 初始化一个空列表来保存合并后的结果  
        merged_list = []  
        # 使用两个指针分别指向列表1和列表2的开头  
        i = 0  
        j = 0  
        # 当两个列表都还有元素时  
        while i < len(sorted_potential_params1) and j < len(sorted_potential_params2):  
            # 将较大的元素添加到合并后的列表中  
            if sorted_potential_params1[i]['potential_score']>= sorted_potential_params2[j]['potential_score']:  
                merged_list.append(sorted_potential_params1[i])  
                i += 1  
            else:  
                merged_list.append(sorted_potential_params2[j])  
                j += 1  
        
        # 将列表1中剩余的元素添加到合并后的列表中（如果有的话）  
        while i < len(sorted_potential_params1):  
            merged_list.append(sorted_potential_params1[i])  
            i += 1  
        
        # 将列表2中剩余的元素添加到合并后的列表中（如果有的话）  
        while j < len(sorted_potential_params2):  
            merged_list.append(sorted_potential_params2[j])  
            j += 1  
        
        # 因为列表原本就是降序排列的，所以合并后的列表也是降序排列的  
        return merged_list  


    # split_potential_params
    # 用途：将各个潜在最优解按照是否满足约束进行分组
    # 方法：每一个解都有自己的分数.根据前面的设定，potential_score大于1的是满足约束的解，否则是不满足约束的
    #      注意，满足约束解至多max_in_cons个，不满足约束解至多max_out_cons个。
    def split_sorted_potential_params(self,max_in_cons,max_out_cons,sorted_potential_params):
        params_in_cons=[]
        params_out_cons=[]

        for param in sorted_potential_params:
            if param['potential_score'] > 1:
                if len(params_in_cons)==max_in_cons:
                    continue
                else:
                    params_in_cons.append(param)
            else:
                if len(params_out_cons)==max_out_cons:
                    break
                else:
                    params_out_cons.append(param)
        
        return params_in_cons, params_out_cons
        

    # get_distance_between_sections
    # 用途：计算两个区间之间的距离，比如"reso=0-fps=0-encoder=0"和"reso=8-fps=9-encoder=0"之间的距离就是(8*8+9*9)
    # 返回值：两个坐标之间的集合距离
    def get_distance_between_sections(self,section_id_1,section_id_2):
         
        pos1=[]
        for part in section_id_1.split('-'):
            # 使用 split('=') 分割键值对  
            conf_name, value = part.split('=')
            pos1.append(int(value))
        pos2=[]
        for part in section_id_2.split('-'):
            # 使用 split('=') 分割键值对  
            conf_name, value = part.split('=')
            pos2.append(int(value))
        distance = 0
        for i in range(0,len(pos1)):
            distance += (pos1[i]-pos2[i])**2
        
        return distance**(1/2)

    # judeg_potential_direction
    # 用途：判断param1到param2的相对方向，是否等于section_id1到section_id2的相对方向
    '''
    参数start_param形如:
    {
        'conf': {'reso': '480p', 'fps': 1, 'encoder': 'JPEG'}, 
		'flow_mapping': {
			'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'},
			'gender_classification': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}
		}, 
		'resource_limit': {
			'face_detection': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}, 
			'gender_classification': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}
		}, 
		'edge_cloud_cut_choice': 0, 
        'pred_delay_list': [0.044744093992091935, 0.0018706620685637382], 
        'pred_delay_total': 0.055844776060655674, 
        'deg_violate': 0, 
        'task_accuracy': 0.7949547259749911
    }
    '''
    def judeg_potential_direction(self, param1, param2, section_id_1, section_id_2):

        pos1=[]
        conf1=[]
        for part in section_id_1.split('-'):
            # 使用 split('=') 分割键值对  
            conf_name, value = part.split('=')
            pos1.append(int(value))
            conf1.append(conf_and_serv_info[conf_name].index(param1['conf'][conf_name]))
        pos2=[]
        conf2=[]
        for part in section_id_2.split('-'):
            # 使用 split('=') 分割键值对  
            conf_name, value = part.split('=')
            pos2.append(int(value))
            conf2.append(conf_and_serv_info[conf_name].index(param2['conf'][conf_name]))
        
        #print(pos1,pos2)
        #print(conf1,conf2)
        # 现在要看是不是都符合相同的大小关系
        for i in range(0,len(pos1)):
            if pos1[i] > pos2[i]:
                if conf1[i] <= conf2[i]:
                    return False
            elif pos1[i] < pos2[i]:
                if conf1[i] >= conf2[i]:
                    return False
            elif pos1[i] == pos2[i]:
                if conf1[i] != conf2[i]:
                    return False
        # 以上全避免了才能返回true,认为这
        return True

    # get_potential_section_id_GD
    # 用途: 在梯度下降中，根据一个区间section_id中的初始点start_param和潜力解potential_param，在section_ids_in_cons中寻找潜力解相对初始点对应的区间
    def get_potential_section_id_GD(self,section_id, section_info, start_param, potential_param, section_ids_in_cons):

        # 判断potential_param相对start_param在各个配置上的大小关系，可能是大于，小于，等于。寻找满足这三个关系的其他区间。
        # 我要筛选得到所有满足潜力方向的区间，然后再进行排序，从中选取最好的。主要不要和自身冲个了。
        filtered_section_ids_in_cons = [
                                        section_id_in_cons for section_id_in_cons in section_ids_in_cons \
                                            if (self.judeg_potential_direction(param1=start_param,
                                                                            param2=potential_param,
                                                                            section_id_1=section_id,
                                                                            section_id_2=section_id_in_cons) \
                                                and section_id != section_id_in_cons)
                                        ]
                                         
        
        # 完成筛选后按照到section_id的距离由小到大排序
        sorted_potential_section_ids = sorted(filtered_section_ids_in_cons,\
                                               key=lambda x: self.get_distance_between_sections(section_id_1=x, 
                                                                                                section_id_2=section_id)) 
        # 完成排序之后就可以取出其中的第一个区间编号作为结果了。
        # 注意，可能不存在这样的区间
        status = 0
        potential_section_id = ''
        if len(sorted_potential_section_ids)==0:
            return status, potential_section_id
        
        status = 1
        potential_section_id = sorted_potential_section_ids[0]
        return status, potential_section_id, 


        

    # get_potential_section_ids_info
    # 用途：梯度下降，从section_ids_in_cons里找到至多max_directions个潜力区间
    # 方法：目前靠一些粗暴方法，选择表现优秀的解，将这些解的得分作为相应区间的潜力
    # 返回值：potential_section_ids_info列表，内部元素如下。这里的potential_param是cur_section_id内引导了潜力区间的潜力点
    '''
    {
        "section_id":"reso=0-fps=0-encoder=0",
        "potential_score":0.7,
        "potential_param":
        {	
            'conf': {'reso': '480p', 'fps': 7, 'encoder': 'JPEG'}, 
            'flow_mapping': {
                'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
                'gender_classification': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}
            }, 
            'resource_limit': {
                'face_detection': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}, 
                'gender_classification': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}
            }, 
            'edge_cloud_cut_choice': 0, 
            'pred_delay_list': [0.044744093992091935, 0.0018706620685637382], 
            'pred_delay_total': 0.055844776060655674, 
            'deg_violate': 0, 
            'task_accuracy': 0.7949547259749911,
            'potential_score':0.7
        }
    }
    '''
    def get_potential_section_ids_info(self, section_info, section_id,sorted_potential_params,start_param,max_directions,optimize_goal,section_ids_in_cons):
        '''
        # 在区间section_id中，有一个初始点start_param，以及若干解sorted_potential_params
        # 需要从sorted_potential_params里找到比start_param性能更优的解，并由此从section_ids_in_cons中寻找有潜力的下一个区间
        # 有潜力的区间分为：原区间自身；保底区间；较优区间
        # 较优区间：sorted_potential_params中比起始点更优的解，且这些解能够引导相应区间。潜力为更优解的得分。
        # (未实现)保底区间：sorted_potential_params中比起始点更优的解，但相应方向不存在可用区间，此时寻找近似可用的最近区间，潜力为更优解的得分。
        # 补偿区间：找不到较优区间的时候，选择没有采样过的最近的区间
        # (未实现)原本区间：section_id自身，得分为初始点的分数。如果params_in_cons和params_out_cons中不存在比初始点更好的点，就只有原本区间一个。
        # 最终至少含有一个区间
        '''
        #1、初始化未来的返回值,删除可能存在于列表中的section_id
        potential_section_ids_info=[]
        '''
        if section_id in section_ids_in_cons:
            print('约束内区间')
            print(section_ids_in_cons)
            section_ids_in_cons.remove(section_id)
            if len(section_ids_in_cons)==0:
                print('没有可梯度下降的区间')
                assert 0
        '''
        
        #2、从完成潜力评分的sorted_potential_params中找出比start_param['potential_score']更高的配置。区间信息已经按照潜力得分从大到小排序了。
        filtered_sorted_potential_params = [param for param in sorted_potential_params if param['potential_score'] >= start_param['potential_score']] 
        
        #3、接下来要从完成过滤的参数中开始提取潜在的最优区间。这里要考虑区间重复的问题，提取的区间总数在达到max_directions后终止
        potential_section_ids_num = 0
        potential_section_id_set = set()
        for potential_param in filtered_sorted_potential_params:
            if potential_section_ids_num == max_directions:
                break
            # 根据当前section_id，当前区间中的start_param，以及这个潜力解potential_param，从section_ids_in_cons里寻找下一个较好的区间
            # 注意，可能得到较优区间，也可能得到保底区间。
            status, potential_section_id = self.get_potential_section_id_GD(section_id=section_id,
                                                                            section_info=section_info,
                                                                            start_param=start_param,
                                                                            potential_param=potential_param,
                                                                            section_ids_in_cons=section_ids_in_cons)
            # 现在得到了一个可能有用的区间。但是要防止重复。
            if status == 1:
                if potential_section_id not in potential_section_id_set :
                    potential_section_id_set.add(potential_section_id)
                    potential_section_ids_num+=1
                    # 然后制造新的info
                    potential_section_id_info={}
                    potential_section_id_info['section_id']=potential_section_id
                    potential_section_id_info['potential_score']=potential_param['potential_score']
                    potential_section_id_info['potential_param']=potential_param
                    potential_section_ids_info.append(potential_section_id_info)
        # 最后得到被填补了的potential_section_ids_info


        #4、考虑没有最优解的情况
        # 注意，一个区间中某种配置可能只有一个取值，这种情况下，潜力解不可能为这个配置指引方向（比如只有一种分辨率），此时怎么办？其实是有办法的，那就是随机尝试
        #       具体的，需要统计潜力解指向的潜力方向，相对原本配置是否有所突破；如果完全没有突破，说明区间内的潜力解不能起到指引作用；此时寻找距离自己最近的几个区间凑数，潜力分数设置为原始值
        #       当前方法选择没有使用过的且最近的区间,选择至多max_directions个，就像探索一样
        if potential_section_ids_num == 0: #没有区间可走
            sorted_section_ids = sorted(section_ids_in_cons,\
                                        key=lambda x: self.get_distance_between_sections(section_id_1=x, 
                                                                                        section_id_2=section_id)) 
            for potential_section_id in sorted_section_ids:
                if potential_section_ids_num == max_directions:
                    break
                if potential_section_id not in self.GD_used_set:#最好是之前没有去过的区间
                    potential_section_id_info={}
                    potential_section_id_info['section_id']=potential_section_id
                    potential_section_id_info['potential_score']=start_param['potential_score']
                    potential_section_id_info['potential_param']=start_param
                    potential_section_ids_info.append(potential_section_id_info)
            
            if potential_section_ids_num == 0:
                # 如果还是空的，那就直接走到较近的地方去吧
                potential_section_id_info={}
                potential_section_id_info['section_id']=sorted_section_ids[0]
                potential_section_id_info['potential_score']=start_param['potential_score']
                potential_section_id_info['potential_param']=start_param
                potential_section_ids_info.append(potential_section_id_info)
        
        # 最终返回的列表信息一定不为空
        return potential_section_ids_info



    # get_start_param_GD
    # 用途：在一个新区间完成贝叶斯优化采样后，为了进行梯度下降，需要从中寻找一个初始点start_parma
    # 方法：从该区间中已经按照潜力得分排序的解中，选择一个大于等于区间潜力的最小的解；如果没有这样的解，选择最大的。
    def get_start_param_GD(self, cur_section_id_info, cur_sorted_potential_params):
        # 潜力得分是从大到小排序的,所以我要从小到大寻找
        for i in range(len(cur_sorted_potential_params)-1,-1,-1):
            if cur_sorted_potential_params[i]['potential_score'] >= cur_section_id_info['potential_score']:
                return cur_sorted_potential_params[i]
        # 如果都不行，那就返回最大解了
        return cur_sorted_potential_params[0]
                                                                                    

    # get_plan_in_cons_from_sections_GD：
    # 用途：基于梯度下降的方法，从section_ids_in_cons中不断选择合适的section_id来尝试查找最优解.
    #       参数max_directions表示每次进行梯度下降的时候至多选择几个有潜力的区间
    # 方法：从初始位置start_param开始，将贝叶斯优化与梯度下降融合，用梯度下降指导贝叶斯优化的选择.
    # 返回值：满足时延和资源约束的解；满足时延约束不满足资源约束的解；不满足时延约束的解；贝叶斯优化模型建议的下一组参数值
    '''
    使用梯度下降法的基本思路：
    根据start_param作为初始,定位一个区间,然后在这个区间内进行贝叶斯优化采样得到结果,然后分析下一个需要采样的区间
    n_descent表示一共追查多少个这样的区间
    max_directions表示每一个区间内尝试多少个方向
    n_trials表示每一个区间内进行贝叶斯优化时采样多少次
    optimze_goal表示优化目标
    start_param表示梯度下降的起始点k,一般是上一个配置,这个配置对应的性能会因为工况的变化而变化
    start_param内含有conf,flow_mapping,resource_limit,edge_cloud_cut_choice。但是因为工况发生了变化,需要重新计算性能和资源约束违反程度
    
    为了后续使用，需要将其改造成如下形式（重新计算）：
        {	
            'conf': {'reso': '480p', 'fps': 7, 'encoder': 'JPEG'}, 
            'flow_mapping': {
                'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
                'gender_classification': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}
            }, 
            'resource_limit': {
                'face_detection': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}, 
                'gender_classification': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}
            }, 
            'edge_cloud_cut_choice': 0, 
            'pred_delay_list': [0.044744093992091935, 0.0018706620685637382], 
            'pred_delay_total': 0.055844776060655674, 
            'deg_violate': 0, 
            'task_accuracy': 0.7949547259749911
        }
    '''
    def get_plan_in_cons_from_sections_GD(self, n_descent, max_directions, n_trials, optimize_goal, section_info, section_ids_in_cons, start_param,max_in_cons,max_out_cons):

        # 1、初始化,设计后续作为返回值的列表，以及优先级队列prior_queue。
        params_in_cons = [] #最终返回值
        params_out_cons = [] #最终返回值
        cur_params_in_cons = [] #暂存贝叶斯优化在一个区间内得到的解
        cur_params_out_cons = [] #暂存贝叶斯优化在一个区间内得到的解
        sorted_potential_params = [] #存储每次找到的解,每次合并的时候都会在内部进行排序

        self.GD_prior_queue.clear() #梯度下降时的优先队列，梯度下降时从中提取下一个有潜力的区间信息
        self.GD_used_set.clear() #梯度下降时存储已经走过的区间

        #    文件描述符，记录每一次梯度下降的选择
        filename = 'kb_GD/' + self.kb_name + '/' +datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')+'_GD_record.json'
        f = open(filename, 'a', encoding='utf-8')
        

        # 1.1 start_param作为梯度下降的起点，需要判断其所在的起始区间start_section_id
        start_section_id_info = self.judge_section(conf=start_param['conf'],section_info=section_info)
        if not start_section_id_info['exist']:
            print('初始配置根本不存在于区间中，梯度下降无法进行')
            assert 0
        cur_section_id = start_section_id_info['section_id']
        
        # 1.2 计算start_param现在(当前工况下)的预测时延和性能，以便后续进行梯度下降时作为参照
        status, plan_description = self.get_plan_description(conf = start_param['conf'],
                                                             flow_mapping = start_param['flow_mapping'],
                                                             resource_limit = start_param['resource_limit'],
                                                             edge_cloud_cut_choice = start_param['edge_cloud_cut_choice'],
                                                             section_info=section_info)
        #     用预测结果来更新start_param
        start_param['pred_delay_list']=plan_description['pred_delay_list']
        start_param['pred_delay_total']=plan_description['pred_delay_total']
        start_param['deg_violate']=plan_description['deg_violate']
        start_param['task_accuracy']=plan_description['task_accuracy']
        start_param['potential_score']=self.get_potential_score_for_param(param=start_param,optimize_goal=optimize_goal,if_in_cons=False)



        # 2、获取以上信息后,在初始区间中使用贝叶斯优化获取一系列解，并进行排序和打分
        #    每一次贝叶斯优化采样之前，先把区间编号放入集合之中
        self.GD_used_set.add(cur_section_id)
        cur_params_in_cons, cur_params_out_cons = self.get_plan_in_cons_for_one_section(n_trials = n_trials, 
                                                                                        optimize_goal = optimize_goal, 
                                                                                        section_info = section_info,
                                                                                        section_id = cur_section_id)
        cur_sorted_potential_params = self.get_sorted_potential_params(params_in_cons = cur_params_in_cons,
                                                                       params_out_cons = cur_params_out_cons,
                                                                       optimize_goal = optimize_goal)
        # 将这个区间上已经完成排序的解合并起来，依旧保持排序（从大到小）
        sorted_potential_params=self.merge_sorted_potential_params(sorted_potential_params1=sorted_potential_params,
                                                                   sorted_potential_params2=cur_sorted_potential_params)
        
       
        
        # 3、进入循环，开始不断进行梯度下降。在循环中，利用上一次计算得到的完成排序的解，寻找有潜力的其他区间
        des_num = 0
        while des_num < n_descent:
            des_num +=1
            # 3.1 根据cur_section_id区间内找到的解ccur_sorted_potential_params，获取其他有潜力的区间
            potential_section_ids_info_list = self.get_potential_section_ids_info( section_info = section_info,
                                                                            section_id = cur_section_id, 
                                                                            sorted_potential_params = cur_sorted_potential_params,
                                                                            start_param = start_param,
                                                                            max_directions = max_directions,
                                                                            optimize_goal = optimize_goal,
                                                                            section_ids_in_cons = section_ids_in_cons)
            # 每一次循环都要记录最新的查找结果:
            new_GD_data={}
            new_GD_data['cur_section_id']=cur_section_id
            new_GD_data['start_param']=start_param
            new_GD_data['cur_sorted_potential_params']=cur_sorted_potential_params
            new_GD_data['potential_section_ids_info_list']=potential_section_ids_info_list  #记录一下本次采集到的潜力区间
            json.dump(new_GD_data,f,indent=4) 
            
            # 3.2 扩展优先队列，优先队列里存放有潜力的区间
            self.GD_prior_queue.extend(potential_section_ids_info_list)
            
            # 3.3 从优先队列next_section_ids_list中获得下一个要检索的区间，取出之后就删除
            cur_section_id_info = max(self.GD_prior_queue, key=lambda item: item['potential_score'])
            print('选择出下一个区间')
            print(cur_section_id_info)
            self.GD_prior_queue.remove(cur_section_id_info)
            cur_section_id = cur_section_id_info['section_id']


            # 3.4 在下一个要检索的区间中进行贝叶斯优化，对解进行排序和打分，然后放入cur_sorted_potential_params之中。
            self.GD_used_set.add(cur_section_id)
            cur_params_in_cons, cur_params_out_cons = self.get_plan_in_cons_for_one_section(n_trials = n_trials, 
                                                                                            optimize_goal = optimize_goal, 
                                                                                            section_info = section_info,
                                                                                            section_id = cur_section_id)
            cur_sorted_potential_params = self.get_sorted_potential_params(params_in_cons = cur_params_in_cons,
                                                                            params_out_cons = cur_params_out_cons,
                                                                            optimize_goal = optimize_goal)
            sorted_potential_params=self.merge_sorted_potential_params(sorted_potential_params1=sorted_potential_params,
                                                                        sorted_potential_params2=cur_sorted_potential_params)
            
            
            # 3.5 根据优化结果，计算出适合该区间的start_param，用于引导下一次梯度下降
            start_param = self.get_start_param_GD(cur_section_id_info, cur_sorted_potential_params)

        #关闭文件描述符
        f.close()

        # 最后从params_in_cons和params_out_cons中找到帕累托最优解，进行筛选。
        params_in_cons,params_out_cons = self.split_sorted_potential_params(max_in_cons = max_in_cons,
                                                                            max_out_cons = max_out_cons,
                                                                            sorted_potential_params = sorted_potential_params)

        return params_in_cons,params_out_cons
 

    # get_plan_in_cons_GD
    # 用途：以梯度下降的策略来获取合适的配置，而非嵌套的贝叶斯优化。
    # 方法：先通过筛选获取一系列可能含有最优解的区间，然后从上一次调度方案start_param开始进行梯度下降
    # 返回值：帕累托最优解中的满足约束解、不满足约束解、下一个计划
    '''
    参数中,n_descent表示梯度下降的时候最多下降多少次(尝试多少个区间)
    max_directions表示每一个区间内尝试多少个方向
    n_trials表示每一个区间内进行贝叶斯优化时采样多少次
    optimze_goal表示优化目标
    start_param表示梯度下降的起始点k,一般是上一个配置,这个配置对应的性能会因为工况的变化而变化
    
    参数start_param形如:
    {
        'conf': {'reso': '480p', 'fps': 1, 'encoder': 'JPEG'}, 
		'flow_mapping': {
			'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'},
			'gender_classification': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}
		}, 
		'resource_limit': {
			'face_detection': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}, 
			'gender_classification': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}
		}, 
		'edge_cloud_cut_choice': 0, 
    }
    
    '''
    def get_plan_in_cons_GD(self,n_descent,max_directions,n_trials,optimze_goal,start_param,max_in_cons,max_out_cons,section_ids_sub=None,):
        print('梯度下降，初始配置是')
        print(start_param)
        section_info = self.get_section_info()
        section_ids_in_cons=self.get_section_ids_in_cons(section_info=section_info,optimze_goal=optimze_goal,section_ids_sub=section_ids_sub)
        print('过滤后,能满足约束的区间有',len(section_ids_in_cons),'个')
        print(section_ids_in_cons)
        print('开始通过梯度下降求约束内解')
        params_in_cons,params_out_cons=self.get_plan_in_cons_from_sections_GD(n_descent=n_descent,
                                                                                max_directions=max_directions,
                                                                                n_trials=n_trials,
                                                                                optimize_goal=optimze_goal,
                                                                                section_info=section_info,
                                                                                section_ids_in_cons=section_ids_in_cons,
                                                                                start_param=start_param,
                                                                                max_in_cons=max_in_cons,
                                                                                max_out_cons=max_out_cons,
                                                                                )

        #return params_in_cons,params_out_cons,next_plan
        return params_in_cons,params_out_cons






    # update_kb
    # 知识库更新问题暂时不考虑                
    # 知识库更新接口，调度器使用此接口将前一调度周期内使用的调度策略与性能指标之间的关系更新入知识库
   
    
    



# 1、流水线信息
service_info_list=[
    {
        "name":'face_detection',
        "value":'face_detection_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    {
        "name":'gender_classification',
        "value":'gender_classification_proc_delay',
        "conf":["reso","fps","encoder"]
    },
]
conf_names=["reso","fps","encoder"]
serv_names=["face_detection","gender_classification"]   


# 2、工况
work_condition={
    "obj_n": 1,
    "obj_stable": True,
    "obj_size": 300,
    "delay": 0.2  #这玩意包含了传输时延，我不想看
}

# 3、数据传输量与带宽
'''
# edge_to_cloud_data += self.portrait_info['data_trans_size'][self.serv_names[index]]  # 以上次执行时各服务之间数据的传输作为当前传输的数据量
# pre_frame_str_size = len(self.portrait_info['frame'])
# pre_frame = field_codec_utils.decode_image(self.portrait_info['frame'])
# pred_delay_total += edge_to_cloud_data / (self.bandwidth_dict['kB/s'] * 1000)
        
'''
# 如果使用画像，需要知道['data_trans_size'][self.serv_names[index]]，以及self.portrait_info['frame']，以及self.bandwidth_dict['kB/s'] * 1000

portrait_info={}
frame = cv2.imread('video_frames/cold_start_4/frame_1080p.jpg')  
frame=field_codec_utils.encode_image(frame)
portrait_info['frame']=frame
portrait_info['data_trans_size']={}
portrait_info['data_trans_size']['face_detection']=1200000
portrait_info['data_trans_size']['gender_classification']=70000

bandwidth_dict={}
#bandwidth_dict['kB/s']=100000
bandwidth_dict['kB/s']=100


# 4、用户约束
#    一般情况下优化目标是在满足精度约束的时候最小化时延，此时精度不能小于约束，时延不能超过约束且需要最小化
#
user_constraint={  
    "delay": 0.2,
    "accuracy": 0.9
}



if __name__ == "__main__":

    
    from RuntimePortrait import RuntimePortrait
    myportrait=RuntimePortrait(pipeline=serv_names)
    rsc_upper_bound={}
    for serv_name in serv_names:
        serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
        rsc_upper_bound[serv_name]={}
        rsc_upper_bound[serv_name]['cpu_limit']=serv_rsc_cons['cpu']['edge']
        rsc_upper_bound[serv_name]['mem_limit']=serv_rsc_cons['mem']['edge']
    print("画像提供的资源上限")
    print(rsc_upper_bound)
    '''
    {
    'face_detection': {'cpu_limit': 0.258420012563467, 'mem_limit': 0.014812282925592597}, 
    'gender_classification': {'cpu_limit': 0.9544000817298889, 'mem_limit': 0.008211500913059324}
    }
    '''
    #设置资源阈值的下限
    rsc_down_bound={
        "face_detection": {"cpu_limit": 0.0, "mem_limit": 0.0}, 
        "gender_classification": {"cpu_limit": 0.0, "mem_limit": 0.0}
    }

    rsc_constraint={
            "114.212.81.11": {"cpu": 1.0, "mem": 1.0}, 
            "192.168.1.7": {"cpu": 1.0, "mem": 1.0}
        }


    kb_name='kb_data_90i90_no_clst-1'
    kb_user=KnowledgeBaseUser(conf_names=conf_names,
                                  serv_names=serv_names,
                                  service_info_list=service_info_list,
                                  user_constraint=user_constraint,
                                  rsc_constraint=rsc_constraint,
                                  rsc_upper_bound=rsc_upper_bound,
                                  rsc_down_bound=rsc_down_bound,
                                  work_condition=work_condition,
                                  portrait_info=portrait_info,
                                  bandwidth_dict=bandwidth_dict,
                                  kb_name=kb_name,
                                  refer_kb_name='kb_data_90i90_no_clst-1'
                                  )
     
    need_pred_delay=0

    need_to_cold_start=0

    need_to_search_GD=1

    if need_to_cold_start==1:
        n_trials=200
        optimze_goal=MIN_DELAY

        params_in_cons,params_out_cons,next_plan=kb_user.get_plan_in_cons(n_trials=n_trials,
                                                                          optimze_goal=optimze_goal)
        #section_ids_sub=[]
        #params_in_cons,params_out_cons,next_plan=kb_user.get_plan_in_cons(n_trials=n_trials,optimze_goal=optimze_goal,section_ids_sub=section_ids_sub)
        print('约束内配置')
        for param in params_in_cons:
            print(param)
        print('约束外配置')
        for param in params_out_cons:
            print(param)
    
    
    if need_to_search_GD == 1:

        n_descent = 10
        max_directions = 4
        n_trials = 20
        optimze_goal=MIN_DELAY
        start_param={
            'conf': {'reso': '480p', 'fps': 1, 'encoder': 'JPEG'}, 
            'flow_mapping': 
                {
                'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
                'gender_classification': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}
                }, 
            'resource_limit': 
                {
                'face_detection': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}, 
                'gender_classification': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}
                }, 
            'edge_cloud_cut_choice': 0, 
        }
        max_in_cons=3
        max_out_cons=3
        params_in_cons,params_out_cons = kb_user.get_plan_in_cons_GD(n_descent=n_descent,
                                                                    max_directions=max_directions,
                                                                    n_trials=n_trials,
                                                                    optimze_goal=optimze_goal,
                                                                    start_param=start_param,
                                                                    max_in_cons=max_in_cons,
                                                                    max_out_cons=max_out_cons,
                                                                    section_ids_sub=None)
        print('约束内配置')
        for param in params_in_cons:
            print(param)
        print('约束外配置')
        for param in params_out_cons:
            print(param)
    


    if need_pred_delay==1:
        conf=dict({"reso": "360p", "fps": 10, "encoder": "JPEG"})
        flow_mapping=dict({
            "face_detection": {"model_id": 0, "node_ip": "172.27.132.253", "node_role": "host"}, 
            "gender_classification": {"model_id": 0, "node_ip": "172.27.132.253", "node_role": "host"}
            })
        resource_limit=dict({
            "face_detection": {"cpu_util_limit": 0.2, "mem_util_limit": 0.004}, 
            "gender_classification": {"cpu_util_limit": 0.2, "mem_util_limit": 0.008}
            })
        import time
        time.sleep(1)
        print('开始查找')
        section_info=kb_user.get_section_info()
        status,pred_delay_list,pred_delay_total=kb_user.get_pred_delay(conf=conf,
                                                                       flow_mapping=flow_mapping,
                                                                       resource_limit=resource_limit,
                                                                       edge_cloud_cut_choice=0,
                                                                       section_info=section_info)
        print(status,pred_delay_list,pred_delay_total)
    
        

            
    exit()