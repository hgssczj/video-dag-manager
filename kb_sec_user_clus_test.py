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
    # 用途：在指定分区中判断资源消耗
    # 方法：目前就是单纯计算边缘端上的资源消耗
    def get_pred_rsc_in_section(self, section_id, conf, flow_mapping, resource_limit, edge_cloud_cut_choice):
        edge_cpu_use=0
        for serv_name in self.serv_names:
            if flow_mapping[serv_name]['node_role']!='cloud':
                edge_cpu_use+=resource_limit[serv_name]['cpu_util_limit']
        edge_cpu_use=round(edge_cpu_use,2)
        return edge_cpu_use


    # get_pred_delay_in_section
    # 用途：在指定分区中判断时延
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
    

    # 由trial给出下一组要尝试的参数并返回，需要section_info来限制取值范围
    def get_next_params(self, trial,section_ids,section_info, optimize_goal):
        conf = {}
        flow_mapping = {}
        resource_limit = {}
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

        # 求资源限制
        edge_cpu_use_total=0.0 #边端cpu资源消耗总量
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

                edge_cpu_use_total+=resource_limit[serv_name]["cpu_util_limit"] 
            
        #（2）得到配置后，计算该配置对应的时延，精度，资源消耗，并根据optimze_goal，来确定四个优化目标
        #status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf, flow_mapping=flow_mapping, resource_limit=resource_limit, edge_cloud_cut_choice=edge_cloud_cut_choice)
        
        # (2.1)计算时延
        status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf,
                                                                        flow_mapping=flow_mapping,
                                                                        resource_limit=resource_limit,
                                                                        edge_cloud_cut_choice=edge_cloud_cut_choice,
                                                                        section_info=section_info)
        if status==0:
           status=0 #没有配置可用
           return status, conf, flow_mapping, resource_limit, mul_objects

        # (2.2)计算精度
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
        # 最后task_accuracy就是精度

        # (2.3)计算资源消耗
        edge_cpu_use_total=round(edge_cpu_use_total,2)

        # (3.1) 根据约束满足情况制定三个约束,满足约束就取0，否则往大了走
        delay_constraint = self.user_constraint["delay"]
        accuracy_constraint = self.user_constraint["accuracy"]
        edge_cpu_constraint = self.rsc_constraint[common.edge_ip]['cpu']

    
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
    

    # get_plan_in_cons_from_sections：
    # 用途：从指定的一系列区间中获取能够满足约束的解，需要指定优化目标。
    #      理论上应该根据某些方式选出一系列区间后再使用它
    # 方法：通过贝叶斯优化，在有限轮数内选择一个最优的结果。需要section_info给get_next_params使用。
    # 返回值：满足时延和资源约束的解；满足时延约束不满足资源约束的解；不满足时延约束的解；贝叶斯优化模型建议的下一组参数值
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
            status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf,
                                                                            flow_mapping=flow_mapping,
                                                                            resource_limit=resource_limit,
                                                                            edge_cloud_cut_choice=edge_cloud_cut_choice,
                                                                            section_info=section_info)
            
            # 求出精度
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
                    
            if status != 0:  # 如果status不为0，才说明这个配置是有效的，否则是无效的
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
                    
                ans_dict['pred_delay_list'] = pred_delay_list
                ans_dict['pred_delay_total'] = pred_delay_total
                ans_dict['deg_violate'] = deg_violate
                ans_dict['task_accuracy'] = task_accuracy

                # 根据配置是否满足时延约束，将其分为两类
                if pred_delay_total <= 0.95*self.user_constraint["delay"] and task_accuracy>=self.user_constraint["accuracy"] and ans_dict['deg_violate'] == 0 :
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
       
        next_plan = {
            'video_conf': next_conf,
            'flow_mapping': next_flow_mapping,
            'resource_limit': next_resource_limit
        }
              
        return params_in_cons,params_out_cons,next_plan
 
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

            judge1=True #判断该区间是否有资格列入section_ids_in_cons之中
            judge2=True #右上角精度是否满足约束
            judge3=True #左下角资源充分时的最小时延是否满足约束

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
            
            #(1.3)第三轮审查：计算左下角各个配置，求最小时延，以及最小资源消耗
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
                    print('当前左下角最小时延是',bottom_left_min_delay)
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
    
    # 调用以上函数获取约束内配置
    def get_plan_in_cons(self,n_trials,optimze_goal,section_ids_sub=None):
        section_info = self.get_section_info()
        section_ids_in_cons=self.get_section_ids_in_cons(section_info=section_info,optimze_goal=optimze_goal,section_ids_sub=section_ids_sub)
        print('过滤后,能满足约束的区间是')
        print(section_ids_in_cons)
        params_in_cons,params_out_cons,next_plan=self.get_plan_in_cons_from_sections(n_trials=n_trials,optimize_goal=optimze_goal,section_info=section_info,section_ids_in_cons=section_ids_in_cons)

        return params_in_cons,params_out_cons,next_plan


    # update_kb
    # 知识库更新问题暂时不考虑                
    # 知识库更新接口，调度器使用此接口将前一调度周期内使用的调度策略与性能指标之间的关系更新入知识库
   
    
    




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


# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","gender_classification"]   


user_constraint={  #同时需要时延约束和资源约束
    "delay": 1.0,  #用户约束暂时设置为0.3
    "accuracy": 0.7
}

work_condition={
    "obj_n": 1,
    "obj_stable": True,
    "obj_size": 300,
    "delay": 0.2  #这玩意包含了传输时延，我不想看
}



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
bandwidth_dict['kB/s']=100000




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


    '''
    with open('static_data.json', 'r') as f:  
        static_data = json.load(f)
    
    rsc_constraint=static_data['rsc_constraint']

    '''
    rsc_constraint={
            "114.212.81.11": {"cpu": 1.0, "mem": 1.0}, 
            "192.168.1.7": {"cpu": 1.0, "mem": 1.0}
        }

    #设置资源阈值的上限
    '''
    rsc_upper_bound={
        "face_detection": {"cpu_limit": 0.25, "mem_limit": 0.012}, 
        "gender_classification": {"cpu_limit": 0.1, "mem_limit": 0.008}
    }
    '''
    
    conf_and_serv_info['reso']=['360p','480p','720p','1080p']
    conf_and_serv_info['fps']=[i + 1 for i in range(30)]
    conf_and_serv_info['face_detection_ip']=["192.168.1.7","114.212.81.11"]
    conf_and_serv_info['face_detection_cpu_util_limit']=[0.05,0.10,0.15,0.20,
                                                        0.25,0.30,0.35,0.40,
                                                        0.45,0.50,0.55,0.60,
                                                        0.65,0.70,0.75,0.80,
                                                        0.85,0.90,0.95,1.00]
    conf_and_serv_info['gender_classification_ip']=["192.168.1.7","114.212.81.11"]
    conf_and_serv_info['gender_classification_cpu_util_limit']=[0.05,0.10,0.15,0.20,
                                                                0.25,0.30,0.35,0.40,
                                                                0.45,0.50,0.55,0.60,
                                                                0.65,0.70,0.75,0.80,
                                                                0.85,0.90,0.95,1.00]
              
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
                                  kb_name=KB_DATA_PATH,
                                  refer_kb_name='kb_data_90i90_no_clst-1'
                                  )
    # def __init__(self, conf_names, serv_names, service_info_list, user_constraint, rsc_constraint, rsc_upper_bound, rsc_down_bound, work_condition, portrait_info, bandwidth_dict,kb_name,refer_kb_name):
   
    need_pred_delay=0

    need_to_test=1

 

    if need_to_test==1:
        n_trials=200
        optimze_goal=MIN_DELAY

        params_in_cons,params_out_cons,next_plan=kb_user.get_plan_in_cons(n_trials=n_trials,optimze_goal=optimze_goal)
        #section_ids_sub=[]
        #params_in_cons,params_out_cons,next_plan=kb_user.get_plan_in_cons(n_trials=n_trials,optimze_goal=optimze_goal,section_ids_sub=section_ids_sub)
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