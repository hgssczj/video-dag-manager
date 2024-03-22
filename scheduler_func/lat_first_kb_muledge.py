import sys  
import itertools
from kb_user import KnowledgeBaseUser,kb_plan_path,kb_data_path
from logging_utils import root_logger


import json

import copy

prev_conf = dict()
prev_flow_mapping = dict()
prev_resource_limit = dict()
prev_runtime_info = dict()


from kb_user import model_op,conf_and_serv_info,kb_data_path,kb_plan_path

# servcie_info_list应该根据流水线的需要,从service_info_dict中挑选，随时组建
service_info_dict={
    'face_detection':{
        "name":'face_detection',
        "value":'face_detection_proc_delay',
        "conf":["reso","fps","encoder"]
    },

    'face_detection_trans':{
        "name":'face_detection_trans',
        "value":'face_detection_trans_delay',
        "conf":["reso","fps","encoder"]
    },
    'gender_classification':{
        "name":'gender_classification',
        "value":'gender_classification_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    'gender_classification_trans':{
        "name":'gender_classification_trans',
        "value":'gender_classification_trans_delay',
        "conf":["reso","fps","encoder"]
    },
}


# 该函数的作用是在给定的约束下寻求最合适的一个解，该解本身未必能够满足约束
def get_plan_based_on_constraint(
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    rsc_upper_bound=None,
):
    #(1)建立冷启动器
    cold_starter=KnowledgeBaseUser( conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    user_constraint=user_constraint,
                                    rsc_constraint=rsc_constraint,
                                    rsc_upper_bound=rsc_upper_bound
                                    )

    # (2)依次尝试不同的n_trail，并用params_in_delay_cons_total和params_out_delay_cons_total两个列表，
    # 分别存储查到的满足约束和不满足约束的解。查找在找到一个绝佳解的时候停止。
    n_trials_range=[100,200,300,400]
    params_in_delay_cons_total=[]
    params_out_delay_cons_total=[]
    for n_trials in n_trials_range:
        
        params_in_delay_cons, params_out_delay_cons=cold_starter.get_coldstart_plan_bayes(n_trials=n_trials)
        params_in_delay_cons_total.extend(params_in_delay_cons)
        params_out_delay_cons_total.extend(params_out_delay_cons)
        sorted_params_temp=sorted(params_in_delay_cons,key=lambda item:(item['deg_violate'],item['num_cloud'],-item['pred_delay_total']))
        if sorted_params_temp[0]['deg_violate']==0 and sorted_params_temp[0]['num_cloud']==0:
            print("找到一个绝佳解，停止继续搜索")
            break
    
    # （3）按照以下排序方式对得到的解进行排序。理论上两个列表都不为空，可以找到一个最优解。当然也可能找不到最优解。
    '''
        (1)对于能满足时延约束的解
        按照资源约束违反程度从小到大排序；
        按照云端服务使用数量从小到大排序；
        按照时延从大到小排序；
        然后选取最优解——满足约束，且违反程度最小、对云端使用最少、最逼近约束的.

        (2)对于不能满足时延约束的解
        此时没有能满足时延约束的解，怎么办？那只能权衡一下三种指标了。
        按照某种权重，对所有不满足实验约束的解的配置进行排序，然后选择一个最优的，希望它在时延上较小且资源约束较好。

        (3)如果两个解都找不到
        那就直接返回说明确实查找失败了
    '''
    sorted_params=[]
    if len(params_in_delay_cons)>0:
        sorted_params=sorted(params_in_delay_cons,key=lambda item:(item['deg_violate'],item['num_cloud'],-item['pred_delay_total']))
    elif len(params_out_delay_cons)>0:
        sorted_params=sorted(params_out_delay_cons,key=lambda item:(item['deg_violate']+item['pred_delay_total']))
    else:
        #如果找不到最优解，说明知识库根本无法在该约束下找到一个合适的解
        ans_found=0
        conf={}
        flow_mapping={}
        resource_limit={}
        return ans_found, conf, flow_mapping, resource_limit

    # 如果可以找到最优解，现在就需要查看并进行一些修正了    
    print('排序后的解')
    for param_valid in sorted_params:
        for key in param_valid.keys():
            print(key,param_valid[key])
    
    best_params=sorted_params[0]
    print('初始最优解')
    for key in best_params.keys():
            print(key,best_params[key])
    
    ''' 最优解的格式形如：
    conf {'reso': '720p', 'fps': 1, 'encoder': 'JPEG'}
    flow_mapping {'face_detection': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}, 'face_alignment': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}}
    resource_limit {'face_detection': {'cpu_util_limit': 0.25, 'mem_util_limit': 0.3}, 'face_alignment': {'cpu_util_limit': 0.25, 'mem_util_limit': 0.3}}
    pred_delay_list [0.0020203223595251798, 0.01092090790088356, 0.0724544489689362, 0.027888460770631396]
    pred_delay_total 0.11328413999997633
    num_cloud 0
    deg_violate 0
    '''
    
    conf=best_params['conf']
    flow_mapping=best_params['flow_mapping']
    resource_limit=best_params['resource_limit']

    # (4)开始对找到的最优解进行检查，观察其是否满足约束。如果不满足约束，需要进行一系列修正。
    # 这个修正过程包括重新分配资源，以及将任务挪到云端。serv_offload_cloud_idx用于记录由于不满足约束导致的必须迁移到云端的起始索引

    serv_offload_cloud_idx=len(serv_names) #如果要将一些任务挪到云端，这个索引作为起始索引。初始为服务的总长，如果减小了，说明有必要挪到云端
    cloud_ip=''
    for device_ip in rsc_constraint.keys():
        # 只针对非云设备计算违反资源约束程度
        if model_op[device_ip]['node_role']=='cloud':
            cloud_ip=device_ip
        else:# 开始计算当前最优解是否在该设备上违反了资源约束;如果违反了约束且不可均分，将该设备上的服务挪到云端
            cpu_util=0
            mem_util=0
            move_to_cloud=0 #如果不得不挪到云端，move_to_cloud会等于0，这发生在现有资源连0.05粒度的均匀分配都做不到的时候
            device_serv_names=[]  #当前ip上所使用的所有服务
            
            # 计算该设备上的资源消耗量
            for serv_name in resource_limit.keys():
                if flow_mapping[serv_name]['node_ip']==device_ip:
                    device_serv_names.append(serv_name)
                    cpu_util=round(cpu_util+resource_limit[serv_name]['cpu_util_limit'],2)
                    mem_util=round(mem_util+resource_limit[serv_name]['mem_util_limit'],2)
            
            # 如果违反了资源约束，进行平分（要求均分后的结果是0.05的倍数）
            if cpu_util>rsc_constraint[device_ip]['cpu']:
                # 此时需要进行重新分配。
                print('cpu不满足约束,要重新分配')
                theoretical_share = round( rsc_constraint[device_ip]['cpu'] / len(device_serv_names),2)
                cpu_share = round(int(theoretical_share / 0.05) * 0.05,2)
                if cpu_share==0:
                    move_to_cloud=1
                else: #否则，是可分的,将该设备上所有服务的cpu限制都改为此cpu_share
                    for serv_name in device_serv_names:
                        resource_limit[serv_name]['cpu_util_limit']=cpu_share
                    
            # 如果违反了资源约束，进行平分（要求评分后的结果是0.05的倍数）
            if mem_util>rsc_constraint[device_ip]['mem']:
                # 此时需要进行重新分配。
                print('mem不满足约束,要重新分配')
                theoretical_share = round(rsc_constraint[device_ip]['mem'] / len(device_serv_names),2)
                mem_share = round(int(theoretical_share / 0.001) * 0.001,3)
                if mem_share==0:
                    move_to_cloud=1
                else: #否则，是可分的,将该设备上所有服务的mem限制都改为此mem_share
                    for serv_name in device_serv_names:
                        resource_limit[serv_name]['mem_util_limit']=mem_share
            
            # 如果move_to_cloud是1，就说明这个ip上的服务都要挪到云端去
            if move_to_cloud==1:
                for serv_name in device_serv_names:
                    serv_offload_cloud_idx=min(serv_offload_cloud_idx,serv_names.index(serv_name))
                    #开始重新定位起始被挪到云端的索引
    

    for idx in range(serv_offload_cloud_idx,len(serv_names)):
        #从这个索引开始的所有服务都要挪到云端
        serv_name=serv_names[idx]
        if flow_mapping[serv_name]['node_role']!='cloud':
            flow_mapping[serv_name]=model_op[cloud_ip]
            resource_limit[serv_name]['cpu_util_limit']=1.0
            resource_limit[serv_name]['mem_util_limit']=1.0


    # 最终会得到一个真正的最优解
    print("正式最优解")
    print('conf',conf)
    print('flow_mapping',flow_mapping)
    print('resource_limit',resource_limit)
    ans_found=1

    return ans_found, conf, flow_mapping, resource_limit




# -----------------
# ---- 调度入口 ----
def scheduler(
    job_uid=None,
    dag=None,
    system_status=None,
    work_condition=None,
    portrait_info=None,
    user_constraint=None,
):
   


    # 上面是实在没办法的测试
    # 现在根据scheduler的输入，要开始选择调度策略。
    assert job_uid, "should provide job_uid"

    global prev_conf, prev_flow_mapping,prev_resource_limit
    global conf_and_serv_info,service_info_dict

    # 调度器首先要根据dag来得到关于服务的各个信息
    # （1）构建serv_names:根据dag获取serv_names，涉及的各个服务名
    serv_names=dag['flow']
    #（2）构建service_info_list：根据serv_names，加上“_trans”构建传输阶段，从service_info_dict提取信息构建service_info_list
    service_info_list=[]  #建立service_info_list，需要处理阶段，也需要传输阶段
    for serv_name in serv_names:
        service_info_list.append(service_info_dict[serv_name])
        service_info_list.append(service_info_dict[serv_name+"_trans"])
    # (3) 构建conf_names：根据service_info_list里每一个服务的配置参数，得到conf_names，保存所有影响任务的配置参数（不含ip）
    conf=dict()
    for service_info in service_info_list:
        service_conf_names = service_info['conf']
        for service_conf_name in service_conf_names:
            if service_conf_name not in conf:
                conf[service_conf_name]=1
    conf_names=list(conf.keys())  #conf_names里保存所有配置参数的名字
    '''
    # conf_names=["reso","fps","encoder"]
    # serv_names=["face_detection", "face_alignment"]
    print("初始冷启动,获取任务信息：")
    print("conf_names:",conf_names)
    print("serv_names",serv_names)
    '''
    #（4）获取每一个设备的资源约束，以及每一个服务的资源上限

 #####################################
    # 测试部分——-负反馈测试器
   
    plan_conf={}
    rsc_constraint={}
    rsc_upper_bound={}
    with open('plan_conf.json', 'r') as f:  
        plan_conf = json.load(f) 
    with open('rsc_constraint.json', 'r') as f:  
        rsc_constraint = json.load(f) 
    with open('rsc_upper_bound.json', 'r') as f:  
        rsc_upper_bound = json.load(f)
    
    conf = plan_conf['conf']
    flow_mapping = plan_conf['flow_mapping']
    resource_limit = plan_conf['resource_limit']

    if_test=plan_conf['if_test']

    if if_test==1: #此时仅仅是读取文件中的配置并返还
        return conf, flow_mapping, resource_limit
    
    elif if_test==2 and portrait_info['if_overtime']: # #如果test==2且超时了，执行负反馈策略

        #此处假设可以通过调整资源来解决问题

        return conf, flow_mapping, resource_limit
    
    elif if_test==3 and portrait_info['if_overtime']: # 如果超时了，使用不依赖画像的查表法

        rsc_upper_bound={
            "face_detection": {"cpu_limit": 1.0, "mem_limit": 1.0}, 
            "gender_classification": {"cpu_limit": 1.0, "mem_limit": 1.0}
        }
         
        ans_found, conf, flow_mapping, resource_limit=get_plan_based_on_constraint(
                                                        conf_names=conf_names,
                                                        serv_names=serv_names,
                                                        service_info_list=service_info_list,
                                                        rsc_constraint=rsc_constraint,
                                                        user_constraint=user_constraint,
                                                        rsc_upper_bound=rsc_upper_bound,
                                                    )
        if ans_found==1:#如果确实找到了
            prev_conf[job_uid]=conf
            prev_flow_mapping[job_uid]=flow_mapping
            prev_resource_limit[job_uid]=prev_resource_limit
            return conf, flow_mapping, resource_limit
        else:
            print("冷启动查询失败")
            return None #后续这里应该改成别的，比如默认配置什么的


    else:
        print('未知状况，默认返回配置文件内容')
        return conf, flow_mapping, resource_limit





    ####################################





    # 一、初始情况下选择使用冷启动策略
    if not bool(work_condition) or not bool(user_constraint):
        root_logger.info("to get COLD start executation plan")

        rsc_constraint={}
        rsc_upper_bound={}
        with open('rsc_constraint.json', 'r') as f:  
            rsc_constraint = json.load(f) 
        with open('rsc_upper_bound.json', 'r') as f:  
            rsc_upper_bound = json.load(f)

        ans_found, conf, flow_mapping, resource_limit=get_plan_based_on_constraint(
                                                        conf_names=conf_names,
                                                        serv_names=serv_names,
                                                        service_info_list=service_info_list,
                                                        rsc_constraint=rsc_constraint,
                                                        user_constraint=user_constraint,
                                                        rsc_upper_bound=rsc_upper_bound,
                                                    )


        if ans_found==1:#如果确实找到了
            prev_conf[job_uid]=conf
            prev_flow_mapping[job_uid]=flow_mapping
            prev_resource_limit[job_uid]=prev_resource_limit
            return conf, flow_mapping, resource_limit
        else:
            print("冷启动查询失败")
            return None #后续这里应该改成别的，比如默认配置什么的
    

    # 二、如果不是冷启动，进行“负反馈调节”
    # ---- 若有负反馈结果，则进行负反馈调节 ----


    return prev_conf[job_uid], prev_flow_mapping[job_uid], prev_resource_limit[job_uid]  #沿用之前的配置
