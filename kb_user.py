import json
import datetime
import matplotlib.pyplot as plt
import optuna
import itertools
plt.rcParams['font.sans-serif'] = ['SimHei'] # 运行配置参数中的字体（font）为黑体（SimHei）

#试图以类的方式将整个独立建立知识库的过程模块化

MAX_NUMBER=999999


from kb_builder import model_op, conf_and_serv_info,kb_data_path

kb_plan_path='kb_plan'

          
# KnowledgeBaseUser可以基于建立好的知识库，提供冷启动所需的一系列策略
# 实际上它可以充分使用知识库
class  KnowledgeBaseUser():   
    
    #冷启动计划者，初始化时需要conf_names,serv_names,service_info_list,user_constraint一共四个量
    #在初始化过程中，会根据这些参数，制造conf_list，serv_ip_list，serv_cpu_list以及serv_meme_list
    def __init__(self,conf_names,serv_names,service_info_list,user_constraint,rsc_constraint,rsc_upper_bound):
        self.conf_names=conf_names
        self.serv_names=serv_names
        self.service_info_list=service_info_list
        self.user_constraint=user_constraint
        self.rsc_constraint=rsc_constraint
        self.rsc_upper_bound=rsc_upper_bound

        
        self.conf_list=[]
        for conf_name in conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip和资源限制
            self.conf_list.append(conf_and_serv_info[conf_name])
        # conf_list会包含各类配置参数取值范围，例如分辨率、帧率等

        self.serv_ip_list=[]
        for serv_name in serv_names:
            serv_ip=serv_name+"_ip"
            self.serv_ip_list.append(conf_and_serv_info[serv_ip])
        # serv_ip_list包含各个模型的ip的取值范围

        self.serv_cpu_list=[]
        for serv_name in serv_names:
            serv_cpu=serv_name+"_cpu_util_limit"
            self.serv_cpu_list.append(conf_and_serv_info[serv_cpu])
        # serv_cpu_list包含各个模型的cpu使用率的取值范围

        self.serv_mem_list=[]
        for serv_name in serv_names:
            serv_mem=serv_name+"_mem_util_limit"
            self.serv_mem_list.append(conf_and_serv_info[serv_mem])
        # serv_mem_list包含各个模型的mem使用率的取值范围
        
        #例如,可能的一组取值如下：
        # conf_names=["reso","fps","encoder"]
        # serv_names=["face_detection", "face_alignment"]
        # conf_list=[ ["360p", "480p", "720p", "1080p"],[1, 5, 10, 20, 30],["JPEG"]]
        # serv_ip_list=[["114.212.81.11","172.27.143.164","172.27.151.145"],["114.212.81.11","172.27.143.164","172.27.151.145"]]
        # serv_cpu_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]
        # serv_mem_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]

    # get_pred_delay：
    # 用途：根据参数里指定的配置，根据知识库来预测对应性能，如果在知识库中找不到，则返回status为0；否则为1
    # 方法：依次遍历评估性能所需的各个字典并根据参数设置从中获取性能评估结果
    # 返回值：status,pred_delay_list,pred_delay_total，描述性能评估成果与否、预测的各阶段时延、预测的总时延
    '''
    'video_conf':   
    {    'encoder': 'JPEG', 'fps': 1, 'reso': '360p'    }
    'flow_mapping': 
    {   
        'face_alignment': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
        'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'} 
    }
    resource_limit={
        "face_detection": {
            "cpu_util_limit": 0.1,
            "mem_util_limit": 0.2,
        },
        "face_alignment": {
            "cpu_util_limit": 0.1,
            "mem_util_limit": 0.2,
        }
    }

    '''
    def get_pred_delay(self,conf, flow_mapping, resource_limit):
        # 知识库所在目录名
        # 存储配置对应的各阶段时延，以及总时延
        pred_delay_list=[]
        pred_delay_total=0
        status=0  #为0表示字典中没有满足配置的存在
        # 对于service_info_list里的service_info依次评估性能
        for service_info in self.service_info_list:
            # （1）加载服务对应的性能评估器
            f=open(kb_data_path+'/'+service_info['name']+".json")  
            evaluator=json.load(f)
            f.close()
            # （2）获取service_info里对应的服务配置参数，从参数conf中获取该服务配置旋钮的需要的各个值，加上ip选择
            # 得到形如["360p"，"1","JPEG","114.212.81.11"]的conf_for_dict，用于从字典中获取性能指标
            service_conf=list(service_info['conf']) # 形如":["reso","fps","encoder"]
                # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
            conf_for_dict=[]
                #形如：["360p"，"1","JPEG"]
            for service_conf_name in service_conf:
                conf_for_dict.append(str(conf[service_conf_name]))   
            
            #完成以上操作后，conf_for_dict内还差ip地址,首先要判断当前评估器是不是针对传输阶段进行的：
            ip_for_dict_index=service_info['name'].find("_trans") 
            if ip_for_dict_index>0:
                # 当前是trans，则去除服务名末尾的_trans，形如“face_detection”
                ip_for_dict_name=service_info['name'][0:ip_for_dict_index] 
            else: #当前不是trans
                ip_for_dict_name=service_info['name']
            ip_for_dict=flow_mapping[ip_for_dict_name]['node_ip']
                
            cpu_for_dict=resource_limit[ip_for_dict_name]["cpu_util_limit"]
            mem_for_dict=resource_limit[ip_for_dict_name]["mem_util_limit"]
            conf_for_dict.append(str(ip_for_dict))  
            conf_for_dict.append(str(cpu_for_dict)) 
            conf_for_dict.append(str(mem_for_dict)) 
            # 形如["360p"，"1","JPEG","114.212.81.11","0.1"."0.1"]

            # （3）根据conf_for_dict，从性能评估器中提取该服务的评估时延
            sub_evaluator=evaluator
            for i in range(0,len(conf_for_dict)-1):
                if conf_for_dict[i] in sub_evaluator:
                    sub_evaluator=sub_evaluator[conf_for_dict[i]]
                else:
                    print('配置不存在')
                    return status,pred_delay_list,pred_delay_total
            
            pred_delay=sub_evaluator[conf_for_dict[len(conf_for_dict)-1]]
            
            #  (4) 如果pred_delay为0，意味着这一部分对应的性能估计结果不存在，该配置在知识库中没有找到合适的解。此时直接返回结果。
            if pred_delay==0:
                return status,pred_delay_list,pred_delay_total
            
            # （5）将预测的时延添加到列表中
            pred_delay_list.append(pred_delay)
        # 计算总时延
        for pred_delay in pred_delay_list:
            pred_delay_total+=pred_delay
        status=1
        return status,pred_delay_list,pred_delay_total  # 返回各个部分的时延
    

    # get_coldstart_plan_rotate：
    # 用途：遍历conf_list、ip、cpu和mem所有配置，选择一个能够满足约束的冷启动计划。
    # 方法：生成配置遍历的全排列，对每一种配置都调用get_pred_delay预测时延，选择较好的
    # 返回值：最好的conf, flow_mapping, resource_limit
    def get_coldstart_plan_rotate(self):
        
        #例如,可能的一组取值如下：
        # conf_names=["reso","fps","encoder"]
        # serv_names=["face_detection", "face_alignment"]
        # conf_list=[ ["360p", "480p", "720p", "1080p"],[1, 5, 10, 20, 30],["JPEG"]]
        # serv_ip_list=[["114.212.81.11","172.27.143.164","172.27.151.145"],["114.212.81.11","172.27.143.164","172.27.151.145"]]
        # serv_cpu_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]
        # serv_mem_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]
        
        # 获取时延约束
        delay_constraint = self.user_constraint["delay"]


        best_conf={}
        best_flow_mapping={}
        best_resource_limit={}
        best_pred_delay_list=[]
        best_pred_delay_total=-1

        min_conf={}
        min_flow_mapping={}
        min_resource_limit={}
        min_pred_delay_list=[]
        min_pred_delay_total=-1

        conf_combine=itertools.product(*self.conf_list)
        for conf_plan in conf_combine:
            serv_ip_combine=itertools.product(*self.serv_ip_list)
            for serv_ip_plan in serv_ip_combine:# 遍历所有配置和卸载策略组合
                serv_cpu_combind=itertools.product(*self.serv_cpu_list)
                for serv_cpu_plan in serv_cpu_combind: 
                    serv_mem_combind=itertools.product(*self.serv_mem_list)
                    for serv_mem_plan in serv_mem_combind: 
                        conf={}
                        flow_mapping={}
                        resource_limit={}
                        for i in range(0,len(conf_names)):
                            conf[conf_names[i]]=conf_plan[i]
                        for i in range(0,len(serv_names)):
                            flow_mapping[serv_names[i]]=model_op[serv_ip_plan[i]]
                            resource_limit[serv_names[i]]={}
                            resource_limit[serv_names[i]]["cpu_util_limit"]=serv_cpu_plan[i]
                            resource_limit[serv_names[i]]["mem_util_limit"]=serv_mem_plan[i]
                        # 右上，得到了conf，flow_mapping，以及resource_limit
                        status,pred_delay_list,pred_delay_total = self.get_pred_delay(conf=conf,
                                                                                flow_mapping=flow_mapping,
                                                                                resource_limit=resource_limit,
                                                                                )
                        if status == 0: #如果为0，意味着配置在字典中找不到对应的性能评估结果，知识库没有存储这种配置对应的估计结果
                            continue
                        if best_pred_delay_total<0:   #初始化最优配置和最小配置
                            best_conf=conf
                            best_flow_mapping=flow_mapping
                            best_resource_limit=resource_limit
                            best_pred_delay_list=pred_delay_list
                            best_pred_delay_total=pred_delay_total

                            min_conf=conf
                            min_flow_mapping=flow_mapping
                            min_resource_limit=resource_limit
                            min_pred_delay_list=pred_delay_list
                            min_pred_delay_total=pred_delay_total

                        elif pred_delay_total < delay_constraint*0.7 and pred_delay_total>best_pred_delay_total: #选出一个接近约束且比较大的
                            best_conf=conf
                            best_flow_mapping=flow_mapping
                            best_resource_limit=resource_limit
                            best_pred_delay_list=pred_delay_list
                            best_pred_delay_total=pred_delay_total
                        
                        elif pred_delay_total < best_pred_delay_total: #选出一个最小的
                            min_conf=conf
                            min_flow_mapping=flow_mapping
                            min_resource_limit=resource_limit
                            min_pred_delay_list=pred_delay_list
                            min_pred_delay_total=pred_delay_total
        
        # 完成遍历后，应该可以找到一个比较优秀的冷启动结果
        '''
        print("最优配置是：")
        print(best_conf)
        print(best_flow_mapping)
        print(best_resource_limit)
        print(best_pred_delay_list)
        print(best_pred_delay_total)
        print("最小配置是：")
        print(min_conf)
        print(min_flow_mapping)
        print(min_resource_limit)
        print(min_pred_delay_list)
        print(min_pred_delay_total)
        print("时延约束是",delay_constraint)
        '''       

        if min_pred_delay_total > delay_constraint:
            # print("约束过于严格，选择最小配置")
            return min_conf, min_flow_mapping, min_resource_limit
        else:
            # print("约束不算特别严格，选择最优策略")
            return best_conf, best_flow_mapping, best_resource_limit

    #该objective只能优化时延，不能优化资源约束，也不能限制资源约束
    def objective_old(self,trial):
        conf={}
        flow_mapping={}
        resource_limit={}

        for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
            # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
            conf[conf_name]=trial.suggest_categorical(conf_name,conf_and_serv_info[conf_name])

        #但是注意，每一种服务都有自己的专门取值范围
        for serv_name in self.serv_names:
            with open(kb_data_path+'/'+ serv_name+'_conf_info'+'.json', 'r') as f:  
                conf_info = json.load(f)  
                serv_ip=serv_name+"_ip"
                flow_mapping[serv_name]=model_op[trial.suggest_categorical(serv_ip,conf_info[serv_ip])]
                serv_cpu_limit=serv_name+"_cpu_util_limit"
                serv_mem_limit=serv_name+"_mem_util_limit"
                resource_limit[serv_name]={}
                if flow_mapping[serv_name]["node_role"] =="cloud":  #对于云端没必要研究资源约束下的情况
                    resource_limit[serv_name]["cpu_util_limit"]=1.0
                    resource_limit[serv_name]["mem_util_limit"]=1.0
                else:
                    resource_limit[serv_name]["cpu_util_limit"]=trial.suggest_categorical(serv_cpu_limit,conf_info[serv_cpu_limit])
                    resource_limit[serv_name]["mem_util_limit"]=trial.suggest_categorical(serv_mem_limit,conf_info[serv_mem_limit])
            

        status,pred_delay_list,pred_delay_total = self.get_pred_delay(conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit)

    #该objective是一个动态多目标优化过程，能够求出帕累托最优解
    def objective(self,trial):
        conf={}
        flow_mapping={}
        resource_limit={}

        for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
            # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
            conf[conf_name]=trial.suggest_categorical(conf_name,conf_and_serv_info[conf_name])

        #但是注意，每一种服务都有自己的专门取值范围
        cloud_ip='' #记录云端ip,判断流水线是否已经开始进入云端
        for serv_name in self.serv_names:
            with open(kb_data_path+'/'+ serv_name+'_conf_info'+'.json', 'r') as f:  
                conf_info = json.load(f)  
                serv_ip=serv_name+"_ip"
                if bool(cloud_ip): #如果已经记录了云端ip，说明流水线已经迁移到云端了，此时后续阶段只能选择云端
                    flow_mapping[serv_name]=model_op[cloud_ip]
                else:
                    flow_mapping[serv_name]=model_op[trial.suggest_categorical(serv_ip,conf_info[serv_ip])]
                serv_cpu_limit=serv_name+"_cpu_util_limit"
                serv_mem_limit=serv_name+"_mem_util_limit"
                resource_limit[serv_name]={}
                if flow_mapping[serv_name]["node_role"] =="cloud":  #对于云端没必要研究资源约束下的情况
                    cloud_ip=flow_mapping[serv_name]["node_ip"]
                    resource_limit[serv_name]["cpu_util_limit"]=1.0
                    resource_limit[serv_name]["mem_util_limit"]=1.0
                    # 根据flow_mapping中的node_ip可以从rsc_constraint中获得该设备上的资源约束，通过serv_name可以获得rsc_upper_bound上的约束，
                    # 二者取最小的，用于限制从conf_info中获取的可取值范围，从而限制查找值。
                else:
                    device_ip=flow_mapping[serv_name]["node_ip"]
                    cpu_limit=min(self.rsc_constraint[device_ip]['cpu'],self.rsc_upper_bound[serv_name]['cpu_limit'])
                    mem_limit=min(self.rsc_constraint[device_ip]['mem'],self.rsc_upper_bound[serv_name]['mem_limit'])
                    cpu_choice_range=[item for item in conf_info[serv_cpu_limit] if item <= cpu_limit]
                    mem_choice_range=[item for item in conf_info[serv_mem_limit] if item <= mem_limit]
                    # 要防止资源约束导致取值范围为空的情况
                    if len(cpu_choice_range)==0:
                        cpu_choice_range=[item for item in conf_info[serv_cpu_limit]]
                    if len(mem_choice_range)==0:
                        mem_choice_range=[item for item in conf_info[serv_mem_limit]]
                    resource_limit[serv_name]["cpu_util_limit"]=trial.suggest_categorical(serv_cpu_limit,cpu_choice_range)
                    resource_limit[serv_name]["mem_util_limit"]=trial.suggest_categorical(serv_mem_limit,mem_choice_range)
        '''
        
        user_constraint={
            "delay": 0.1,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }

        rsc_constraint={
            "114.212.81.11":{
                "cpu": 1.0,
                "mem":1.0
            },
            "172.27.143.164": {
                "cpu": 0.6,
                "mem": 0.7
            },
            "172.27.151.145": {
                "cpu": 0.5,
                "mem": 0.8 
            },
        }
        # 描述每一种服务所需的中资源阈值，它限制了贝叶斯优化的时候采取怎样的内存取值范围
        rsc_upper_bound={
            'face_detection':{
                'cpu_limit':0.4,
                'mem_limit':0.5,
            },
            'face_alignment':{
                'cpu_limit':0.4,
                'mem_limit':0.5,
            }
        }

        '''

        
        mul_objects=[]
        # 这是一个多目标优化问题，所以既要满足时延约束，又要满足精度约束。以下是获取时延约束的情况，要让最后的结果尽可能小。首先加入时延优化目标：
        status,pred_delay_list,pred_delay_total = self.get_pred_delay(conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit)
        
        if status==0:  #返回0说明相应配置压根不存在，此时返回MAX_NUMBER。贝叶斯优化的目标是让返回值尽可能小，这种MAX_NUMBER的情况自然会被尽量避免
            mul_objects.append(MAX_NUMBER)
        else:  #如果成功找到了一个可行的策略，按照如下方式计算返回值，目的是得到尽可能靠近约束时延0.7倍且小于约束时延的配置
            delay_constraint = self.user_constraint["delay"]
            if pred_delay_total <= 0.95*delay_constraint:
                mul_objects.append(delay_constraint - pred_delay_total)
            elif pred_delay_total > 0.95*delay_constraint:
                mul_objects.append(0.05*delay_constraint + pred_delay_total)
        
        # 然后加入各个设备上的优化目标：我需要获取每一个设备上的cpu约束，以及mem约束。
        
        for device_ip in self.rsc_constraint.keys():
            cpu_util=0
            mem_util=0
            # 检查该device_ip上消耗了多少资源
            for serv_name in resource_limit.keys():
                if flow_mapping[serv_name]['node_ip']==device_ip:
                    cpu_util+=resource_limit[serv_name]['cpu_util_limit']
                    mem_util+=resource_limit[serv_name]['mem_util_limit']
            
            # 如果cpu_util和mem_util取值都为0，说明用户根本没有在这个ip上消耗资源，此时显然是满足优化目标的，直接返回两个0即可
            if cpu_util==0 and mem_util==0:
                mul_objects.append(0)
                mul_objects.append(0)
            
            # 否则，说明在这个ip上消耗资源了。
            # 此时，如果是在云端，资源消耗根本不算什么,但是应该尽可能不选择云端才对。所以此处应该返回相对较大的结果。
            # 那么应该返回多少比较合适呢？先设置为1看看。
            elif model_op[device_ip]['node_role']=='cloud':
                mul_objects.append(1.0)
                mul_objects.append(1.0)

            # 只有在ip为边端且消耗了资源的情况下，才需要尽可能让资源最小化。这里的0.5是为了减小资源约束相对于时延的占比权重
            else: 
                mul_objects.append(0.5*cpu_util)
                mul_objects.append(0.5*mem_util)

        # 返回的总优化目标数量应该有：1+2*ips个
        return mul_objects

        
        '''
        rsc_constraint={
            "114.212.81.11":{
                "cpu": 1.0,
                "mem":1.0
            },
            "172.27.143.164": {
                "cpu": 0.6,
                "mem": 0.7
            },
            "172.27.151.145": {
                "cpu": 0.5,
                "mem": 0.8 
            },
        }
            'video_conf':   
            {    'encoder': 'JPEG', 'fps': 1, 'reso': '360p'    }
            'flow_mapping': 
            {   
                'face_alignment': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
                'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'} 
            }
            resource_limit={
                "face_detection": {
                    "cpu_util_limit": 0.1,
                    "mem_util_limit": 0.2,
                },
                "face_alignment": {
                    "cpu_util_limit": 0.1,
                    "mem_util_limit": 0.2,
                }
            }
        '''

            

    # get_coldstart_plan_bayes：
    # 用途：基于贝叶斯优化，调用多目标优化的贝叶斯模型，得到一系列帕累托最优解
    # 方法：通过贝叶斯优化，在有限轮数内选择一个最优的结果
    # 返回值：最好的conf, flow_mapping, resource_limit
    def get_coldstart_plan_bayes(self,n_trials):
        # 开始多目标优化
        study = optuna.create_study(directions=['minimize' for _ in range(1+2*len(rsc_constraint.keys()))])  
        study.optimize(self.objective,n_trials=n_trials)

        ans_params=[]
        print("开始获取帕累托最优解")
        trials = sorted(study.best_trials, key=lambda t: t.values) # 对字典best_trials按values从小到达排序  
        
        #此处的帕累托最优解可能重复，因此首先要提取内容，将其转化为字符串记录在ans_params中，然后从ans_params里删除重复项
        for trial in trials:
            conf={}
            flow_mapping={}
            resource_limit={}
            for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
                # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
                conf[conf_name]=trial.params[conf_name]
            cloud_ip='' #用于记录云端ip并判断是不是已经到达分界点了，如果是后续都是云端ip
            for serv_name in self.serv_names:
                serv_ip=serv_name+"_ip"
                if bool(cloud_ip):
                    flow_mapping[serv_name]=model_op[cloud_ip]
                else:
                    flow_mapping[serv_name]=model_op[trial.params[serv_ip]]

                serv_cpu_limit=serv_name+"_cpu_util_limit"
                serv_mem_limit=serv_name+"_mem_util_limit"
                resource_limit[serv_name]={}

                if flow_mapping[serv_name]["node_role"] =="cloud":
                    cloud_ip=flow_mapping[serv_name]["node_ip"]
                    resource_limit[serv_name]["cpu_util_limit"]=1.0
                    resource_limit[serv_name]["mem_util_limit"]=1.0
                else:
                    resource_limit[serv_name]["cpu_util_limit"]=trial.params[serv_cpu_limit]
                    resource_limit[serv_name]["mem_util_limit"]=trial.params[serv_mem_limit]
            
            ans_item={}
            ans_item['conf']=conf
            ans_item['flow_mapping']=flow_mapping
            ans_item['resource_limit']=resource_limit
            ans_params.append(json.dumps(ans_item))
            '''
            {   'reso': '720p', 'fps': 10, 'encoder': 'JPEG'}
            {   'face_detection': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}, 
                'face_alignment': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}}
            {   'face_detection': {'cpu_util_limit': 0.2, 'mem_util_limit': 0.45}, 
                'face_alignment': {'cpu_util_limit': 0.1, 'mem_util_limit': 0.45}}
            '''

        # 从ans_params里删除重复项，并选择真正status不为0的有效帕累托最优解返回。同时存储该配置对应的预估时延
        ans_params_set=list(set(ans_params))
        print("帕累托最优解总数（含重复）",len(ans_params_set))
        # 保存满足时延约束的解
        params_in_delay_cons=[]
        params_out_delay_cons=[]
        for param in ans_params_set:
            ans_dict=json.loads(param)
            #print(type(ans_dict))
            #print(ans_dict)
            conf=ans_dict['conf']
            flow_mapping=ans_dict['flow_mapping']
            resource_limit=ans_dict['resource_limit']
            status,pred_delay_list,pred_delay_total=self.get_pred_delay(conf=conf,
                                                                    flow_mapping=flow_mapping,
                                                                    resource_limit=resource_limit)
            if status!=0: #如果status不为0，才说明这个配置是有效的，否则是无效的
                # 首先附加时延

                
                # 然后计算违背资源约束的程度。由于使用云端的时候被计为100%，因此不用专门区分云和边了
                '''
                rsc_constraint={
                    "114.212.81.11":{
                        "cpu": 1.0,
                        "mem":1.0
                    },
                    "172.27.143.164": {
                        "cpu": 0.6,
                        "mem": 0.7
                    },
                    "172.27.151.145": {
                        "cpu": 0.5,
                        "mem": 0.8 
                    },
                '''
                
                num_cloud=0 #使用云端的服务的数量
                for serv_name in flow_mapping.keys():
                    if flow_mapping[serv_name]["node_role"] =="cloud":
                        num_cloud+=1

                deg_violate=0 #违反资源约束的程度
                #print("分析资源使用情况")
                for device_ip in self.rsc_constraint.keys():
                        # 只针对非云设备计算违反资源约束程度
                        if model_op[device_ip]['node_role']!='cloud':
                            cpu_util=0
                            mem_util=0
                            for serv_name in resource_limit.keys():
                                if flow_mapping[serv_name]['node_ip']==device_ip:
                                    # 必须用round保留小数点，因为python对待浮点数不精确，0.35加上0.05会得到0.39999……
                                    cpu_util=round(cpu_util+resource_limit[serv_name]['cpu_util_limit'],2)
                                    mem_util=round(mem_util+resource_limit[serv_name]['mem_util_limit'],2)
                            cpu_util_ratio=float(cpu_util)/float(self.rsc_constraint[device_ip]['cpu'])
                            mem_util_ratio=float(mem_util)/float(self.rsc_constraint[device_ip]['mem'])
                            #print("展示一下该设备下资源使用率情况")
                            #print(cpu_util,self.rsc_constraint[device_ip]['cpu'])
                            #print(mem_util,self.rsc_constraint[device_ip]['mem'])
                            if cpu_util_ratio>1:
                                deg_violate+=cpu_util_ratio
                            if mem_util_ratio>1:
                                deg_violate+=mem_util_ratio
                    
                ans_dict['pred_delay_list']=pred_delay_list
                ans_dict['pred_delay_total']=pred_delay_total
                ans_dict['num_cloud']=num_cloud
                ans_dict['deg_violate']=deg_violate


                # 根据配置是否满足时延约束，将其分为两类
                if pred_delay_total < self.user_constraint["delay"]:
                    params_in_delay_cons.append(ans_dict)
                else:
                    params_out_delay_cons.append(ans_dict)
                '''
                print(conf)
                print(flow_mapping) 
                print(resource_limit)
                print(status,pred_delay_list,pred_delay_total)
                print(status)
                '''

        return params_in_delay_cons,params_out_delay_cons
    
 

# 尝试进行探索

# 使用KnowledgeBaseBuilder需要提供以下参数：
# service_info_list描述了构成流水线的所有阶段的服务。下图表示face_detection和face_alignment构成的流水线
# 由于同时需要为数据处理时延和数据传输时延建模，因此还有face_detection_trans和face_alignment_trans。一共4个需要关注的服务。
# 每一个服务都有自己的value，用于从采样得到的csv文件里提取相应的时延；conf表示影响该服务性能的配置，
# 但是conf没有包括设备ip、cpu和mem资源约束，因为这是默认的。一个服务默认使用conf里的配置参数加上Ip和资源约束构建字典。
#'''
service_info_list=[
    {
        "name":'face_detection',
        "value":'face_detection_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    {
        "name":'face_alignment',
        "value":'face_alignment_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    {
        "name":'face_detection_trans',
        "value":'face_detection_trans_delay',
        "conf":["reso","fps","encoder"]
    },
    {
        "name":'face_alignment_trans',
        "value":'face_alignment_trans_delay',
        "conf":["reso","fps","encoder"]
    },
]





#以下是发出query请求时的内容。注意video_id。当前文件需要配合query_manager_v2.py运行，后者使用的调度器会根据video_id的取值判断是否会运行。
#建议将video_id设置为99，它对应的具体视频内容可以在camera_simulation里找到，可以自己定制。query_manager_v2.py的调度器发现query_id为99的时候，
#不会进行调度动作。因此，知识库建立者可以自由使用update_plan接口操控任务的调度方案，不会受到云端调度器的影响了。
'''
query_body = {
        "node_addr": "172.27.151.145:5001",
        "video_id": 99,   
        "pipeline": ["face_detection", "face_alignment"],#制定任务类型
        "user_constraint": {
            "delay": 0.6,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }  

service_info_list=[
    {
        "name":'car_detection',
        "value":'car_detection_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    {
        "name":'car_detection_trans',
        "value":'car_detection_trans_delay',
        "conf":["reso","fps","encoder"]
    },
]

# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["car_detection"]   
# 进行车辆检测
query_body = {
        "node_addr": "172.27.151.145:5001",
        "video_id": 101,   
        "pipeline": ["car_detection"],#制定任务类型
        "user_constraint": {
            "delay": 0.1,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }  
'''

# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","face_alignment"]   


user_constraint={
    "delay": 0.1,  #用户约束暂时设置为0.3
    "accuracy": 0.7
}

rsc_constraint={
    "114.212.81.11":{
        "cpu": 1.0,
        "mem":1.0
    },
    "172.27.143.164": {
        "cpu": 0.6,
        "mem": 0.7
    },
    "172.27.151.145": {
        "cpu": 0.5,
        "mem": 0.8 
    },
}
# 描述每一种服务所需的中资源阈值，它限制了贝叶斯优化的时候采取怎样的内存取值范围
rsc_upper_bound={
    'face_detection':{
        'cpu_limit':0.4,
        'mem_limit':0.5,
    },
    'face_alignment':{
        'cpu_limit':0.4,
        'mem_limit':0.5,
    }
}

if __name__ == "__main__":

              
    cold_starter=KnowledgeBaseUser(conf_names=conf_names,
                                  serv_names=serv_names,
                                  service_info_list=service_info_list,
                                  user_constraint=user_constraint,
                                  rsc_constraint=rsc_constraint,
                                  rsc_upper_bound=rsc_upper_bound
                                  )
    
    need_cold_start=0
    if need_cold_start==1:
        # 首先基于文件进行初始化:尝试在不同（或者相同）的约束之下，基于不同的n_trials，来看看基于不同方法检索到的最优配置是怎样的。
        cons_delay_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        n_trials_list=[100,200,300,400,500,600,700,800,900,1000]
        cons_delay_list=[0.4]
        n_trials_list=[100,100,100,100,100,100,100,100,100,100,
                       112,112,112,112,112,112,112,112,112,
                       125,125,125,125,125,125,125,125,
                       143,143,143,143,143,143,143,
                       167,167,167,167,167,167,
                       200,200,200,200,200,
                       250,250,250,250,
                       334,334,334,
                       500,500,
                       1000]
        n_trials_list=[10]
        record_cons_delay=[]
        
        record_n_trials=[]
        record_status=[]
        record_pred_delay=[]

        record_best_status=[]
        record_best_delay=[]

        for cons_delay in cons_delay_list:
            cold_starter.user_constraint["delay"]=cons_delay  #设置一个cons_delay时延
            # 首先，记录一下通过遍历能拿到的最优解
            '''
            conf, flow_mapping, resource_limit = cold_starter.get_coldstart_plan_rotate()
            status0,pred_delay_list0,pred_delay_total0=cold_starter.get_pred_delay(conf=conf,
                                                                flow_mapping=flow_mapping,
                                                                resource_limit=resource_limit)
            '''

            for n_trials in n_trials_list:
                conf, flow_mapping, resource_limit = cold_starter.get_coldstart_plan_bayes(n_trials=n_trials)
                status,pred_delay_list,pred_delay_total=cold_starter.get_pred_delay(conf=conf,
                                                                    flow_mapping=flow_mapping,
                                                                    resource_limit=resource_limit)

                record_cons_delay.append(cons_delay)
                record_n_trials.append(n_trials)
                record_status.append(status)
                record_pred_delay.append(pred_delay_total)
                # record_best_status.append(status0)
                # record_best_delay.append(pred_delay_total0)
        
        #现在要把cold_plan_list整个写入文件（查询到的若干计划）：
        with open(kb_plan_path+'/'+datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')+'cold_plan_list.txt', 'w') as file:
            file.writelines(str(['cons','n_trials','rotate_status','bayes_status','rotate_pred_delay','bayes_pred_delay'])+'\n')
            file.writelines(str(record_cons_delay)+'\n')
            file.writelines(str(record_n_trials)+'\n')
            file.writelines(str(record_best_status)+'\n')
            file.writelines(str(record_status)+'\n')
            file.writelines(str(record_best_delay)+'\n')
            file.writelines(str(record_pred_delay)+'\n')
            
           
        
        file.close()

    need_anylze_plan=0
    if need_anylze_plan==1:
        plan_file_name='20240127_19_51_37cold_plan_list.txt'
        plan_file_name='20240127_16_53_18cold_plan_list.txt'
        file = open(plan_file_name, "r")
        lines = file.readlines()
        file.close()
        print(lines)
        read_res_list=[]
        num=0


        for line in lines:
            
            # plan_item=list(line)
            new_line=line[:-1]
            #print(type(new_line))
            #print(new_line)
            new_line=new_line.replace("'",'')
            new_line=new_line.replace(',','')
            new_line=new_line.replace('[','')
            new_line=new_line.replace(']','')
            #print(type(new_line))
            #print(new_line)

            new_line=new_line.split(' ')
            #print(type(new_line))
            #print(new_line)

            if num>0: #第二行开始才是数字
                number_list=[float(item) for item in new_line]
                #print(type(number_list))
                #print(number_list)
                read_res_list.append(number_list)

            num+=1

        for read_res in read_res_list:
            print(read_res)
        
        cons_delay_list=read_res_list[0]
        n_trials_list=read_res_list[1]
        rotate_status_list=read_res_list[2]
        bayes_status_list=read_res_list[3]
        rotate_pred_delay=read_res_list[4]
        bayes_pred_delay=read_res_list[5]

        # 现在要将它们绘制到一幅图里。
        x_value=[]
        for i in range(0,len(n_trials_list)):
            x_value.append(i)
       

        # 现在的问题是如何在横轴上显示各个n_trials_list
        print(n_trials_list)

        
        # 现在需要绘制cons_delay_list
        fig=plt.figure()
        ax1=fig.add_subplot()
        ax1.plot(x_value,cons_delay_list,label="任务时延约束")
        
        ax1.plot(x_value,rotate_pred_delay,label="遍历查找冷所得启动时延")
        ax1.plot(x_value,bayes_pred_delay,label="贝叶斯查找所得冷启动时延")
        
        ax1.set_ylim(0,1.2)
        ax1.set_ylabel('时延$\mathrm{(s)}$',fontsize=15)
        ax1.set_xlabel('实验次数',fontsize=15)
        plt.yticks(fontsize=14)
        #plt.xticks([0,10,19,27,34,40,45,49,52,54],fontsize=14)
        plt.xticks(fontsize=14)

        #for x_line in [0,10,19,27,34,40,45,49,52,54]:
        #    plt.axvline(x=x_line, color='red')

        ax1.legend(loc='upper left',frameon=True,fontsize=14) #可以显示图例，标注label
        #plt.savefig('z_mypicture/infer_1.png', dpi=600)    # dpi     分辨率

        ax2=ax1.twinx()
        ax2.plot(x_value,n_trials_list,linestyle='--',label='贝叶斯优化采样次数')
        ax2.set_ylabel('贝叶斯优化采样次数n_trials',fontsize=15)
        ax2.set_ylim(0,10000)
        ax2.legend(loc='upper right',frameon=True,fontsize=14) #可以显示图例，标注label
        plt.yticks([100,500,1000,1500],fontsize=14)
        # plt.xticks(fontsize=14)
         # plt.yticks(fontsize=14)
        plt.show()

    exit()