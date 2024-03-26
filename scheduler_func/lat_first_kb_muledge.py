import sys  
import itertools
from kb_user import KnowledgeBaseUser
from logging_utils import root_logger

from common import model_op,service_info_dict,conf_and_serv_info,ip_range,reso_range,fps_range
import common
import json
from RuntimePortrait import RuntimePortrait
from AccuracyPrediction import AccuracyPrediction



prev_conf = dict()
prev_flow_mapping = dict()
prev_resource_limit = dict()
prev_runtime_info = dict()

# servcie_info_list应该根据流水线的需要,从service_info_dict中挑选，随时组建

# 冷启动专用的贝叶斯优化查找函数
# 该函数的作用是使用KnowledgeBaseUser基于多次贝叶斯优化，试图找到一个可行解。其得出的结果必然是可用的。
# 使用该函数的时候，会在给定约束范围内寻找一个能够使用的最优解，但这个最优解本身未必能够使性能最优，因此只可用于冷启动。
def get_coldstart_plan_bayes(
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    rsc_upper_bound=None,
    rsc_down_bound=None,
    work_condition=None
):
    #(1)建立冷启动器
    cold_starter=KnowledgeBaseUser( conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    user_constraint=user_constraint,
                                    rsc_constraint=rsc_constraint,
                                    rsc_upper_bound=rsc_upper_bound,
                                    rsc_down_bound=rsc_down_bound,
                                    work_condition=work_condition
                                    )

    # (2)依次尝试不同的n_trail，并用params_in_delay_cons_total和params_out_delay_cons_total两个列表，
    # 分别存储查到的满足约束和不满足约束的解。查找在找到一个绝佳解的时候停止。
    n_trials_range=[100,200,300,400,500]
    params_in_delay_in_rsc_cons_total=[]
    params_in_delay_out_rsc_cons_total=[]
    params_out_delay_cons_total=[]
    for n_trials in n_trials_range:
        
        params_in_delay_in_rsc_cons,params_in_delay_out_rsc_cons,params_out_delay_cons=cold_starter.get_plan_in_cons(n_trials=n_trials)
        params_in_delay_in_rsc_cons_total.extend(params_in_delay_in_rsc_cons)
        params_in_delay_out_rsc_cons_total.extend(params_in_delay_out_rsc_cons)
        params_out_delay_cons_total.extend(params_out_delay_cons)
        #看看完全满足约束的解中，是否能出现一个绝佳解。绝佳解暂定为不在云端执行且精度高于0.9的，所以先按照num_cloud从小到大排序，再按照精度从大到小排序
        sorted_params_temp=sorted(params_in_delay_in_rsc_cons,key=lambda item:(item['num_cloud'],-item['task_accuracy']))
        if len(sorted_params_temp)>0:
            if sorted_params_temp[0]['num_cloud']==0 and sorted_params_temp[0]['task_accuracy']>0.9:
                print("找到一个绝佳解，停止继续搜索")
                break
    
    # （3）按照以下排序方式对得到的解进行排序。理论上两个列表都不为空，可以找到一个最优解。当然也可能找不到最优解。
    '''
        (1)对于能满足时延和约束的解
        希望num_cloud尽可能小,task_accuracy尽可能大,也就是按照num_cloud-task_accuracy从小到大排序

        (2)对于能满足时延约束但不能满足资源约束的解
        选择资源约束违反程度最小的
        此时没有能满足时延约束的解，怎么办？那只能权衡一下三种指标了。
        按照某种权重，对所有不满足实验约束的解的配置进行排序，然后选择一个最优的，希望它在时延上较小且资源约束较好。

        (3)对于不能满足时延约束的解
        让资源违反程度和耗时相加，取最小的。找不到就算了。
    '''
    sorted_params=[]

    if len(params_in_delay_in_rsc_cons_total)>0:
        sorted_params=sorted(params_in_delay_in_rsc_cons_total,key=lambda item:(item['num_cloud']-item['task_accuracy']))

    elif len(params_in_delay_out_rsc_cons_total)>0:
        sorted_params=sorted(params_in_delay_out_rsc_cons_total,key=lambda item:(item['deg_violate']))

    elif len(params_out_delay_cons_total)>0:
        sorted_params=sorted(params_out_delay_cons_total,key=lambda item:(item['deg_violate']+item['pred_delay_total']))
    else:
        #如果找不到最优解，说明知识库根本无法在该约束下找到一个合适的解
        ans_found=0
        conf={}
        flow_mapping={}
        resource_limit={}
        #print("未能获取任何可用解")
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
    {
    'conf': {'reso': '1080p', 'fps': 29, 'encoder': 'JPEG'}, 
    'flow_mapping': {'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
                     'gender_classification': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}}, 
    'resource_limit': {'face_detection': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}, 
                       'gender_classification': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}}, 
    'pred_delay_list': [0.48846824169158937, 0.00422081947326656], 
    'pred_delay_total': 0.49268906116485595, 
    'num_cloud': 2, 
    'deg_violate': 0, 
    'task_accuracy': 0.9589738585430643
    }
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

# 进行一次非常快的查找，只在配置范围非常小的时候才适用
# 如果查找失败，那也没有办法。
def get_fast_judege_bayes(
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    rsc_upper_bound=None,
    rsc_down_bound=None,
    work_condition=None
):
    '''
    print('get_fast_judege_bayes面临的资源约束')
    print(rsc_upper_bound)
    print(rsc_down_bound)
    print('get_fast_judege_bayes面临的可选配置')
    print(conf_and_serv_info['reso'])
    print(conf_and_serv_info['fps'])
    for serv_name in serv_names:
        print(conf_and_serv_info[serv_name+'_ip'])
    '''
    #(1)建立冷启动器
    cold_starter=KnowledgeBaseUser( conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    user_constraint=user_constraint,
                                    rsc_constraint=rsc_constraint,
                                    rsc_upper_bound=rsc_upper_bound,
                                    rsc_down_bound=rsc_down_bound,
                                    work_condition=work_condition
                                    )

    # (2)依次尝试不同的n_trail，并用params_in_delay_cons_total和params_out_delay_cons_total两个列表，
    # 分别存储查到的满足约束和不满足约束的解。查找在找到一个绝佳解的时候停止。
    n_trials_range=[100,200,300]
    ans_found=0
    for n_trials in n_trials_range:
        params_in_delay_in_rsc_cons,params_in_delay_out_rsc_cons,params_out_delay_cons=cold_starter.get_plan_in_cons(n_trials=n_trials)
        if len(params_in_delay_in_rsc_cons)>0:
            sorted_params=sorted(params_in_delay_in_rsc_cons,key=lambda item:(item['num_cloud']-item['task_accuracy']))
            best_params=sorted_params[0]
            #'''
            print('获取fast_judege最新查找的可行解')
            for key in best_params.keys():
                print(key,best_params[key])
            #'''
            conf=best_params['conf']
            flow_mapping=best_params['flow_mapping']
            resource_limit=best_params['resource_limit']
            ans_found=1
            return ans_found, conf, flow_mapping, resource_limit
    conf={}
    flow_mapping={}
    resource_limit={}
    print("未能获取fast_judege获取任何可用解")
    return ans_found, conf, flow_mapping, resource_limit

# 努力寻找第一个满足约束解的贝叶斯优化函数
# 如果查找失败，那也没有办法。
def get_in_all_cons_bayes(
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    rsc_upper_bound=None,
    rsc_down_bound=None,
    work_condition=None
):
    #(1)建立冷启动器
    cold_starter=KnowledgeBaseUser( conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    user_constraint=user_constraint,
                                    rsc_constraint=rsc_constraint,
                                    rsc_upper_bound=rsc_upper_bound,
                                    rsc_down_bound=rsc_down_bound,
                                    work_condition=work_condition
                                    )

    # (2)依次尝试不同的n_trail，并用params_in_delay_cons_total和params_out_delay_cons_total两个列表，
    # 分别存储查到的满足约束和不满足约束的解。查找在找到一个绝佳解的时候停止。
    n_trials_range=[100,200,300,400]
    ans_found=0
    for n_trials in n_trials_range:
        params_in_delay_in_rsc_cons,params_in_delay_out_rsc_cons,params_out_delay_cons=cold_starter.get_plan_in_cons(n_trials=n_trials)
        if len(params_in_delay_in_rsc_cons)>0:
            sorted_params=sorted(params_in_delay_in_rsc_cons,key=lambda item:(item['num_cloud']-item['task_accuracy']))
            best_params=sorted_params[0]
            '''
            print('获取最新查找的可行解')
            for key in best_params.keys():
                print(key,best_params[key])
            '''
            conf=best_params['conf']
            flow_mapping=best_params['flow_mapping']
            resource_limit=best_params['resource_limit']
            ans_found=1
            return ans_found, conf, flow_mapping, resource_limit
    conf={}
    flow_mapping={}
    resource_limit={}
    #print("未能获取任何可用解")
    return ans_found, conf, flow_mapping, resource_limit

# 快速寻找满足约束，且精度更高的解的贝叶斯优化函数
# 如果查找失败，那也没有办法。
def get_better_accuracy_bayes(
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    rsc_upper_bound=None,
    rsc_down_bound=None,
    work_condition=None,
    old_accuracy=None
):
    #(1)建立冷启动器
    cold_starter=KnowledgeBaseUser( conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    user_constraint=user_constraint,
                                    rsc_constraint=rsc_constraint,
                                    rsc_upper_bound=rsc_upper_bound,
                                    rsc_down_bound=rsc_down_bound,
                                    work_condition=work_condition
                                    )

    # (2)依次尝试不同的n_trail，并用params_in_delay_cons_total和params_out_delay_cons_total两个列表，
    # 分别存储查到的满足约束和不满足约束的解。查找在找到一个绝佳解的时候停止。
    n_trials_range=[100,200,300,400]
    ans_found=0
    for n_trials in n_trials_range:
        params_in_delay_in_rsc_cons,params_in_delay_out_rsc_cons,params_out_delay_cons=cold_starter.get_plan_in_cons(n_trials=n_trials)
        if len(params_in_delay_in_rsc_cons)>0:
            sorted_params=sorted(params_in_delay_in_rsc_cons,key=lambda item:(item['num_cloud']-item['task_accuracy']))
            best_params=sorted_params[0]
            if best_params['task_accuracy']>old_accuracy:
                print('获取精度更大的可行解,旧精度',old_accuracy,'，新精度',best_params['task_accuracy'])
                for key in best_params.keys():
                    print(key,best_params[key])
                conf=best_params['conf']
                flow_mapping=best_params['flow_mapping']
                resource_limit=best_params['resource_limit']
                ans_found=1
                return ans_found, conf, flow_mapping, resource_limit
    conf={}
    flow_mapping={}
    resource_limit={}
    print("未能获取精度更大的可用解")
    return ans_found, conf, flow_mapping, resource_limit
    
# 微观查找
# 在更高的reso和fps空间中，ip不变
# 调用get_better_accuracy_bayes试图获取更高精度的解
# 该方法的数字记录为1
def micro_search_improve_accuracy(
    job_uid=None,
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    work_condition=None
):
    print('执行微观调度:尝试提高reso和fps来增加精度')
    #(1)获取旧配置，和旧配置下的精度
    old_accuracy=1.0
    global prev_conf
    global prev_flow_mapping
    old_conf=prev_conf[job_uid]
    old_flow_mapping=prev_flow_mapping[job_uid]
    acc_pre=AccuracyPrediction()
    for serv_name in serv_names:
        if service_info_dict[serv_name]["can_seek_accuracy"]:
            old_accuracy*=acc_pre.predict(service_name=serv_name,service_conf={
                'fps':old_conf['fps'],
                'reso':old_conf['reso']
            })
    
     #（2）根据old_confh和old_flowmapping计算提升配置时要考虑的新配置组合
    #conf_and_serv_info['reso']=['360p']
    #conf_and_serv_info['fps']=[5]
    # 查找时保持ip不变,reso和conf设置为更大的区间
            
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=[old_flow_mapping[serv_name]['node_ip']]
    bigger_reso_range=[item for item in reso_range if int(item[:-1])>=int(old_conf['reso'][:-1])]
    bigger_fps_range=[item for item in fps_range if item>=old_conf['fps']]
    conf_and_serv_info['reso']=bigger_reso_range
    conf_and_serv_info['fps']=bigger_fps_range
    
    
    #（3）获取该服务的画像，作为查找时的资源上界和下界（下界一般为0）
    myportrait=RuntimePortrait(pipeline=serv_names)
    rsc_upper_bound={}
    rsc_down_bound={}
    for serv_name in serv_names:
        serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
        rsc_upper_bound[serv_name]={}
        rsc_upper_bound[serv_name]['cpu_limit']=serv_rsc_cons['cpu']['edge']
        rsc_upper_bound[serv_name]['mem_limit']=serv_rsc_cons['mem']['edge']
        rsc_down_bound[serv_name]={}
        rsc_down_bound[serv_name]['cpu_limit']=0.0
        rsc_down_bound[serv_name]['mem_limit']=0.0

    ans_found, conf, flow_mapping, resource_limit=get_better_accuracy_bayes(
        conf_names=conf_names,
        serv_names=serv_names,
        service_info_list=service_info_list,
        rsc_constraint=rsc_constraint,
        user_constraint=user_constraint,
        rsc_upper_bound=rsc_upper_bound,
        rsc_down_bound=rsc_down_bound,
        work_condition=work_condition,
        old_accuracy=old_accuracy
    )

    #查询完毕之后复原
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=ip_range
    conf_and_serv_info['reso']=reso_range
    conf_and_serv_info['fps']=fps_range

    return ans_found, conf, flow_mapping, resource_limit

# 微观查找：
# 配置不变，ip不变，重新分配资源
# 调用get_in_all_cons_bayes尽快获取满足时延的解
# 该方法的数字记录为2
def micro_search_reassign_rsc(
    job_uid=None,
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    work_condition=None
):
    print('执行微观调度:尝试在配置和ip不变时重新分配资源来减小时延')
     #（1）根据old_confh和old_flowmapping计算提升配置时要考虑的新配置组合
    #conf_and_serv_info['reso']=['360p']
    #conf_and_serv_info['fps']=[5]
    # 查找时保持ip不变，reso和fps也不变。
    global prev_conf
    global prev_flow_mapping
    old_conf=prev_conf[job_uid]
    old_flow_mapping=prev_flow_mapping[job_uid]
    
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=[old_flow_mapping[serv_name]['node_ip']]
    conf_and_serv_info['reso']=[old_conf['reso']]
    conf_and_serv_info['fps']=[old_conf['fps']]
    
    
    #（2）获取该服务的画像，作为查找时的资源上界和下界（下界一般为0）
    myportrait=RuntimePortrait(pipeline=serv_names)
    rsc_upper_bound={}
    rsc_down_bound={}
    for serv_name in serv_names:
        serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
        rsc_upper_bound[serv_name]={}
        rsc_upper_bound[serv_name]['cpu_limit']=serv_rsc_cons['cpu']['edge']
        rsc_upper_bound[serv_name]['mem_limit']=serv_rsc_cons['mem']['edge']
        rsc_down_bound[serv_name]={}
        rsc_down_bound[serv_name]['cpu_limit']=0.0
        rsc_down_bound[serv_name]['mem_limit']=0.0

    ans_found, conf, flow_mapping, resource_limit=get_in_all_cons_bayes(
        conf_names=conf_names,
        serv_names=serv_names,
        service_info_list=service_info_list,
        rsc_constraint=rsc_constraint,
        user_constraint=user_constraint,
        rsc_upper_bound=rsc_upper_bound,
        rsc_down_bound=rsc_down_bound,
        work_condition=work_condition,
    )
    #查询完毕之后复原
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=ip_range
    conf_and_serv_info['reso']=reso_range
    conf_and_serv_info['fps']=fps_range

    return ans_found, conf, flow_mapping, resource_limit

# 微观查找：
# 在更低的reso和fps空间中，ip不变
# 调用get_in_all_cons_bayes尽快获取满足时延的解
# 该方法的数字记录为3
def micro_search_downgrade_conf(
    job_uid=None,
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    work_condition=None
):
    print('执行微观调度:尝试在更小的reso和fps中查找以减少时延')
     #（1）根据old_confh和old_flowmapping计算提升配置时要考虑的新配置组合
    #conf_and_serv_info['reso']=['360p']
    #conf_and_serv_info['fps']=[5]
    # 查找时保持ip不变，reso和fps也不变。
    global prev_conf
    global prev_flow_mapping
    old_conf=prev_conf[job_uid]
    old_flow_mapping=prev_flow_mapping[job_uid]
    
    
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=[old_flow_mapping[serv_name]['node_ip']]
    smaller_reso_range=[item for item in reso_range if int(item[:-1])<=int(old_conf['reso'][:-1])]
    smaller_fps_range=[item for item in fps_range if item<=old_conf['fps']]
    conf_and_serv_info['reso']=smaller_reso_range
    conf_and_serv_info['fps']=smaller_fps_range
    
    #（2）获取该服务的画像，作为查找时的资源上界和下界（下界一般为0）
    myportrait=RuntimePortrait(pipeline=serv_names)
    rsc_upper_bound={}
    rsc_down_bound={}
    for serv_name in serv_names:
        serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
        rsc_upper_bound[serv_name]={}
        rsc_upper_bound[serv_name]['cpu_limit']=serv_rsc_cons['cpu']['edge']
        rsc_upper_bound[serv_name]['mem_limit']=serv_rsc_cons['mem']['edge']
        rsc_down_bound[serv_name]={}
        rsc_down_bound[serv_name]['cpu_limit']=0.0
        rsc_down_bound[serv_name]['mem_limit']=0.0

    ans_found, conf, flow_mapping, resource_limit=get_in_all_cons_bayes(
        conf_names=conf_names,
        serv_names=serv_names,
        service_info_list=service_info_list,
        rsc_constraint=rsc_constraint,
        user_constraint=user_constraint,
        rsc_upper_bound=rsc_upper_bound,
        rsc_down_bound=rsc_down_bound,
        work_condition=work_condition,
    )
    #查询完毕之后复原
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=ip_range
    conf_and_serv_info['reso']=reso_range
    conf_and_serv_info['fps']=fps_range

    return ans_found, conf, flow_mapping, resource_limit

# 微观查找：
# 把流水线上一个任务挪到云端，其他配置不变
# 调用get_in_all_cons_bayes尽快获取满足时延的解
# 该方法的数字记录为4
def micro_search_move_cloud(
    job_uid=None,
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    work_condition=None
):
    print('执行微观调度:尝试将至少一个任务挪到云端以减少时延')
     #（1）根据old_confh和old_flowmapping计算提升配置时要考虑的新配置组合
    #conf_and_serv_info['reso']=['360p']
    #conf_and_serv_info['fps']=[5]
    # 查找时保持ip不变，reso和fps也不变。
    global prev_conf
    global prev_flow_mapping
    old_conf=prev_conf[job_uid]
    old_flow_mapping=prev_flow_mapping[job_uid]

    cloud_ip=''
    for device_ip in model_op.keys():
        if model_op[device_ip]['node_role']=='cloud':
            cloud_ip=device_ip
            break
    # 逆序遍历，从后往前查找
    for serv_name in serv_names[::-1]:
        if model_op[old_flow_mapping[serv_name]['node_ip']]['node_role']=='cloud':
            conf_and_serv_info[serv_name+'_ip']=[old_flow_mapping[serv_name]['node_ip']]
        else:
            conf_and_serv_info[serv_name+'_ip']=[cloud_ip]
            # 到此未知，只将一个任务挪到云端
            break 

    # 其他配置都不变
    conf_and_serv_info['reso']=[old_conf['reso']]
    conf_and_serv_info['fps']=[old_conf['fps']]
    
    #（2）获取该服务的画像，作为查找时的资源上界和下界（下界一般为0）
    myportrait=RuntimePortrait(pipeline=serv_names)
    rsc_upper_bound={}
    rsc_down_bound={}
    for serv_name in serv_names:
        serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
        rsc_upper_bound[serv_name]={}
        rsc_upper_bound[serv_name]['cpu_limit']=serv_rsc_cons['cpu']['edge']
        rsc_upper_bound[serv_name]['mem_limit']=serv_rsc_cons['mem']['edge']
        rsc_down_bound[serv_name]={}
        rsc_down_bound[serv_name]['cpu_limit']=0.0
        rsc_down_bound[serv_name]['mem_limit']=0.0
   
    ans_found, conf, flow_mapping, resource_limit=get_in_all_cons_bayes(
        conf_names=conf_names,
        serv_names=serv_names,
        service_info_list=service_info_list,
        rsc_constraint=rsc_constraint,
        user_constraint=user_constraint,
        rsc_upper_bound=rsc_upper_bound,
        rsc_down_bound=rsc_down_bound,
        work_condition=work_condition,
    )
    #查询完毕之后复原
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=ip_range
    conf_and_serv_info['reso']=reso_range
    conf_and_serv_info['fps']=fps_range

    return ans_found, conf, flow_mapping, resource_limit

def micro_search_extreme_case(
    serv_names=None,
):
    conf=dict({"reso": "360p", "fps": 1, "encoder": "JPEG"})
    flow_mapping=dict()
    resource_limit=dict()
    cloud_ip=''
    for device_ip in model_op.keys():
        if model_op[device_ip]['node_role']=='cloud':
            cloud_ip=device_ip
            break
    for serv_name in serv_names:
        flow_mapping[serv_name]=model_op[cloud_ip]
        resource_limit[serv_name]={"cpu_util_limit": 1.0, "mem_util_limit": 1.0}
    
    return conf,flow_mapping,resource_limit

# 判断是否可以在特定的配置之下，仅通过重新分配资源来完成任务
# 此处进行非常快速的判断，调用get_fast_judege_bayes在很小的资源约束范围内查找
# 会抽取服务对应设备的现有的资源约束值，以及任务的中资源阈值，取其最小者作为upper_bound，再基于upper_bound求down_bound
def access_to_reassign_only(
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    work_condition=None,
    cert_conf=None,
    cert_flow_mapping=None,   
):
    '''
      rsc_constraint={
            "114.212.81.11": {"cpu": 1.0, "mem": 1.0}, 
            "172.27.143.164": {"cpu": 1.0, "mem": 1.0}, 
            "172.27.151.145":{"cpu": 1.0, "mem": 1.0}
        },
    '''
    # print('执行微观调度:尝试在配置和ip不变时重新分配资源来减小时延')
     #（1）指定资源限制外的配置为特定配置
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=[cert_flow_mapping[serv_name]['node_ip']]
    #print('cert_conf',cert_conf)
    conf_and_serv_info['reso']=[cert_conf['reso']]
    conf_and_serv_info['fps']=[cert_conf['fps']]
    
    
    #（2）获取特定配置下该服务的画像，严格限制其资源
    myportrait=RuntimePortrait(pipeline=serv_names)
    rsc_upper_bound={}
    rsc_down_bound={}
    for serv_name in serv_names:
        task_info={}
        task_info['service_name']=serv_name
        task_info['fps']=cert_conf['fps']
        task_info['reso']=common.reso_2_index_dict[cert_conf['reso']]
        if 'obj_n' in work_condition:
            task_info['obj_num']=work_condition['obj_n']
        else:
            task_info['obj_num']=1
        rsc_threshold=myportrait.predict_resource_threshold(task_info=task_info)
        serv_ip=cert_flow_mapping[serv_name]["node_ip"]
        if model_op[serv_ip]['node_role']=='cloud':
            rsc_upper_bound[serv_name]={}
            rsc_upper_bound[serv_name]['cpu_limit']=1.0
            rsc_upper_bound[serv_name]['mem_limit']=1.0
            rsc_down_bound[serv_name]={}
            rsc_down_bound[serv_name]['cpu_limit']=1.0
            rsc_down_bound[serv_name]['mem_limit']=1.0
        else:
            rsc_upper_bound[serv_name]={}
            rsc_upper_bound[serv_name]['cpu_limit']=min(rsc_constraint[serv_ip]['cpu'],rsc_threshold['cpu']['edge']['upper_bound'])
            rsc_upper_bound[serv_name]['mem_limit']=min(rsc_constraint[serv_ip]['mem'],rsc_threshold['mem']['edge']['upper_bound'])
            rsc_down_bound[serv_name]={}
            rsc_down_bound[serv_name]['cpu_limit']=max(0.0,rsc_upper_bound[serv_name]['cpu_limit']-1.5)
            rsc_down_bound[serv_name]['mem_limit']=max(0.0,rsc_upper_bound[serv_name]['mem_limit']-0.005)

    ans_found, conf, flow_mapping, resource_limit=get_fast_judege_bayes(
        conf_names=conf_names,
        serv_names=serv_names,
        service_info_list=service_info_list,
        rsc_constraint=rsc_constraint,
        user_constraint=user_constraint,
        rsc_upper_bound=rsc_upper_bound,
        rsc_down_bound=rsc_down_bound,
        work_condition=work_condition,
    )

    #查询完毕之后复原
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=ip_range
    conf_and_serv_info['reso']=reso_range
    conf_and_serv_info['fps']=fps_range

    #看看能不能快速查到一个可行解
    return ans_found, conf, flow_mapping, resource_limit

# -----------------
# 对当前任务进行宏观判断
# ---- 调度入口 ----
def macro_judge (
    job_uid=None,
    work_condition=None,
    user_constraint=None,
    rsc_constraint=None,
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    all_proc_delay=None
):
    global prev_conf
    global prev_flow_mapping
    global prev_resource_limit

    old_conf=prev_conf[job_uid]
    old_flow_mapping=prev_flow_mapping[job_uid]
    old_resource_limit=prev_resource_limit[job_uid]
    

    #该函数作为宏观画像，用于为如何进行微观调度提供指导
    #首先需要判断当前是否满足时延约束，以及是否满足精度约束
    in_delay_cons=1  #是否满足时延约束

    if all_proc_delay > user_constraint['delay']:
        in_delay_cons=0  #超时
    
    #判断是否满足资源约束
    in_rsc_cons=1
    deg_violate=0.0
    for device_ip in rsc_constraint.keys():
        # 只针对非云设备计算违反资源约束程度
        if model_op[device_ip]['node_role']!='cloud':
            cpu_util=0
            mem_util=0
            for serv_name in old_resource_limit.keys():
                if old_flow_mapping[serv_name]['node_ip']==device_ip:
                    # 必须用round保留小数点，因为python对待浮点数不精确，0.35加上0.05会得到0.39999……
                    cpu_util=round(cpu_util+old_resource_limit[serv_name]['cpu_util_limit'],2)
                    mem_util=round(mem_util+old_resource_limit[serv_name]['mem_util_limit'],2)
            cpu_util_ratio=float(cpu_util)/float(rsc_constraint[device_ip]['cpu'])
            mem_util_ratio=float(mem_util)/float(rsc_constraint[device_ip]['mem'])

            if cpu_util_ratio>1:
                deg_violate+=cpu_util_ratio
            if mem_util_ratio>1:
                deg_violate+=mem_util_ratio
    if deg_violate>0:
        in_rsc_cons=0  #超出资源约束
    
    # 时延约束满足，资源约束也满足
    if in_delay_cons==1 and in_rsc_cons==1:
        print('满足约束，一切安好')
        #此时要追求更高的精度，除非精度已经达到上限
        if old_conf['reso']=='1080p' and old_conf['fps']==30:
            print('配置无法增加，精度无法增加')
            return []  #空表示什么都不需要做
        elif all_proc_delay<0.5*user_constraint['delay']:
            return [common.IMPROVE_ACCURACY] #建议使用micro_search_improve_accuracy
        else: #可以增加精度但是不值得增加精度
            return []
    
    # 如果时延约束和资源约束不能同时满足，就要思考该怎么办了。
    # 为了简化实现，我直接根据画像判断当前的资源充分情况。如果资源已经十分充分，就不考虑配置不变重新分配资源的事
    # 什么情况下不考虑“保持配置不变修改资源”？要么是资源已经非常充分了，要么是不可能在当前配置下满足时延了。
    else:
        print('约束不满足，开始进行宏观调度')

        # 首先判断当前任务是否已经都在云端
        cert_host_num=0
        for serv_name in old_flow_mapping.keys():
            if old_flow_mapping[serv_name]['node_role']!='cloud':
                cert_host_num+=1
            else:
                break
        
        if cert_host_num==0: #如果所有任务已经都在云端，边端没有任务，那就不考虑重新分配资源了
            print('宏观调度认为应该优先进行：降低配置')
            return [common.DOWNGRADE_CONF,common.EXTREME_CASE]

        #1、首先判断能否通过只进行新的资源分配来解决问题：
        ans_found, conf, flow_mapping, resource_limit=access_to_reassign_only(
                                                            conf_names=conf_names,
                                                            serv_names=serv_names,
                                                            service_info_list=service_info_list,
                                                            rsc_constraint=rsc_constraint,
                                                            user_constraint=user_constraint,
                                                            work_condition=work_condition,
                                                            cert_conf=old_conf,
                                                            cert_flow_mapping=old_flow_mapping,       
                                                        )
        if ans_found==1:
            #此时将该方法放在最开头
            print('宏观调度认为可以优先进行：配置不变下的资源重新分配')
            return [common.REASSIGN_RSC,common.DOWNGRADE_CONF,common.MOVE_CLOUD,common.EXTREME_CASE]
        
        #2、接着考虑能否通过3来解决问题，只要看最低配置下能不能找到可行解就够了
        min_conf=dict({"reso": "360p", "fps": 5, "encoder": "JPEG"})
        ans_found, conf, flow_mapping, resource_limit=access_to_reassign_only(
                                                            conf_names=conf_names,
                                                            serv_names=serv_names,
                                                            service_info_list=service_info_list,
                                                            rsc_constraint=rsc_constraint,
                                                            user_constraint=user_constraint,
                                                            work_condition=work_condition,
                                                            cert_conf=min_conf,
                                                            cert_flow_mapping=old_flow_mapping,       
                                                        )
        if ans_found==1:
            #此时将3作为优先采用的方法（降低配置）
            print('宏观调度认为应该优先进行：降低配置')
            return [common.DOWNGRADE_CONF,common.MOVE_CLOUD,common.EXTREME_CASE]
        
        #如果都不行，那就只能将4作为优先采用的方法了,也就是挪到云端。为此，首先要检查当前是否有可以挪到云端的服务
        for serv_name in old_flow_mapping.keys():
            if old_flow_mapping[serv_name]['node_role']!='cloud':
                print('宏观调度认为应该优先进行：推送一个任务到云端')
                return [common.MOVE_CLOUD,common.EXTREME_CASE]
        
        #如果已经全部挪到云端，就只能考虑使用极端策略了。计算策略就是配置全部到云端，然后配置全部都最低。
        print('宏观调度认为应该优先进行：采用极端状况预案')
        return [common.EXTREME_CASE]


# -----------------
# 该函数是使用了画像的真正调度器。本质上就是分了两个阶段分别进行宏观和微观调控
# ---- 调度入口 ----
def scheduler(
    job_uid=None,
    dag=None,
    system_status=None,
    work_condition=None,
    portrait_info=None,
    user_constraint=None,
    appended_result_list=None
):
    assert job_uid, "should provide job_uid"

    # 调度器首先要根据dag来得到关于服务的各个信息
    #（1）构建serv_names:根据dag获取serv_names，涉及的各个服务名
    #（2）构建service_info_list：根据serv_names，加上“_trans”构建传输阶段，从service_info_dict提取信息构建service_info_list
    # (3) 构建conf_names：根据service_info_list里每一个服务的配置参数，得到conf_names，保存所有影响任务的配置参数（不含ip）
    #（4）获取每一个设备的资源约束，以及每一个服务的资源上限
    serv_names=dag['flow']
    
    #获得service_info_list
    service_info_list=[]  #建立service_info_list，需要处理阶段，也需要传输阶段
    for serv_name in serv_names:
        service_info_list.append(service_info_dict[serv_name])
        # 不考虑传输时延了
        # service_info_list.append(service_info_dict[serv_name+"_trans"])
    
    #获得conf_name
    conf=dict()
    for service_info in service_info_list:
        service_conf_names = service_info['conf']
        for service_conf_name in service_conf_names:
            if service_conf_name not in conf:
                conf[service_conf_name]=1
    conf_names=list(conf.keys())  #conf_names里保存所有配置参数的名字

    #获得各阶段处理时延all_proc_delay
    all_proc_delay=0
    if appended_result_list!=None:
        #print("展示最新执行结果ext_runtime")
        #print(appended_result_list[-1]['ext_runtime'])
        serv_proc_delays=appended_result_list[-1]['ext_runtime']['plan_result']['process_delay']
        for serv_name in serv_proc_delays.keys():
            all_proc_delay+=serv_proc_delays[serv_name]
        print('感知到总处理时延是',all_proc_delay)
        print(serv_proc_delays)
    
    #获得设备的资源约束rsc_constraint
    with open('static_data.json', 'r') as f:  
        static_data = json.load(f)
    rsc_constraint=static_data['rsc_constraint']


    # 一、冷启动判断
    if not bool(work_condition) or not bool(user_constraint):
        myportrait=RuntimePortrait(pipeline=serv_names)
        rsc_upper_bound={}
        rsc_down_bound={}
        for serv_name in serv_names:
            serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
            rsc_upper_bound[serv_name]={}
            rsc_upper_bound[serv_name]['cpu_limit']=serv_rsc_cons['cpu']['edge']
            rsc_upper_bound[serv_name]['mem_limit']=serv_rsc_cons['mem']['edge']
            rsc_down_bound[serv_name]={}
            rsc_down_bound[serv_name]['cpu_limit']=0.0
            rsc_down_bound[serv_name]['mem_limit']=0.0
   

        ans_found, conf, flow_mapping, resource_limit=get_coldstart_plan_bayes(
                                                        conf_names=conf_names,
                                                        serv_names=serv_names,
                                                        service_info_list=service_info_list,
                                                        rsc_constraint=rsc_constraint,
                                                        user_constraint=user_constraint,
                                                        rsc_upper_bound=rsc_upper_bound,
                                                        rsc_down_bound=rsc_down_bound,
                                                        work_condition=work_condition
                                                    )

        if ans_found==1:#如果确实找到了，需要更新字典里之前保存的三大配置
            prev_conf[job_uid]=conf
            prev_flow_mapping[job_uid]=flow_mapping
            prev_resource_limit[job_uid]=resource_limit
        else:
            print("查表已经失败,使用极端配置")
            conf,flow_mapping,resource_limit=micro_search_extreme_case(serv_names=serv_names)
            prev_conf[job_uid]=conf
            prev_flow_mapping[job_uid]=flow_mapping
            prev_resource_limit[job_uid]=resource_limit
            print('最终采用:极端情况')
        
        # 为了测试方便，以下人为设置初始冷启动值，以便查看更多稳定可靠的效果。但是下面这一部分代码实际上不能作为真正的冷启动。
        conf=dict({
        "reso": "360p", "fps": 30, "encoder": "JPEG", 
        })
        flow_mapping=dict({
            "face_detection": {"model_id": 0, "node_ip": "172.27.143.164", "node_role": "host"}, 
            "gender_classification": {"model_id": 0, "node_ip": "172.27.143.164", "node_role": "host"}
        })
        resource_limit=dict({
            "face_detection": {"cpu_util_limit": 0.25, "mem_util_limit": 0.004}, 
            "gender_classification": {"cpu_util_limit": 0.75, "mem_util_limit": 0.008}
        })
        prev_conf[job_uid]=conf
        prev_flow_mapping[job_uid]=flow_mapping
        prev_resource_limit[job_uid]=resource_limit
        print('目前使用了默认配置代替冷启动过程')

    else:
        #获取宏观调控计划
        micro_plans=macro_judge (
                    job_uid=job_uid,
                    work_condition=work_condition,
                    user_constraint=user_constraint,
                    rsc_constraint=rsc_constraint,
                    conf_names=conf_names,
                    serv_names=serv_names,
                    service_info_list=service_info_list,
                    all_proc_delay=all_proc_delay
                )
        print('本次给出的宏观指导',micro_plans)
        # 按照宏观调控计划来完成任务
        if len(micro_plans)>0:
            #形如[common.REASSIGN_RSC,common.DOWNGRADE_CONF,common.MOVE_CLOUD,common.EXTREME_CASE]
            for micro_plan in micro_plans:
                # 极端情况
                if micro_plan==common.EXTREME_CASE:
                    conf,flow_mapping,resource_limit=micro_search_extreme_case(serv_names=serv_names)
                    prev_conf[job_uid]=conf
                    prev_flow_mapping[job_uid]=flow_mapping
                    prev_resource_limit[job_uid]=resource_limit
                    print('最终采用:极端情况')
                    break
                elif micro_plan==common.IMPROVE_ACCURACY:
                    ans_found,conf,flow_mapping,resource_limit=micro_search_improve_accuracy(  
                                                        job_uid=job_uid,
                                                        conf_names=conf_names,
                                                        serv_names=serv_names,
                                                        service_info_list=service_info_list,
                                                        rsc_constraint=rsc_constraint,
                                                        user_constraint=user_constraint,
                                                        work_condition=work_condition
                                                    )
                    if ans_found==1:
                        prev_conf[job_uid]=conf
                        prev_flow_mapping[job_uid]=flow_mapping
                        prev_resource_limit[job_uid]=resource_limit
                        print('最终采用:提升精度')
                        break
                elif micro_plan==common.REASSIGN_RSC:
                    ans_found,conf,flow_mapping,resource_limit=micro_search_reassign_rsc(  
                                                        job_uid=job_uid,
                                                        conf_names=conf_names,
                                                        serv_names=serv_names,
                                                        service_info_list=service_info_list,
                                                        rsc_constraint=rsc_constraint,
                                                        user_constraint=user_constraint,
                                                        work_condition=work_condition
                                                    )
                    if ans_found==1:
                        prev_conf[job_uid]=conf
                        prev_flow_mapping[job_uid]=flow_mapping
                        prev_resource_limit[job_uid]=resource_limit
                        print('最终采用:重新分配资源')
                        break
                elif micro_plan==common.DOWNGRADE_CONF:
                    ans_found,conf,flow_mapping,resource_limit=micro_search_downgrade_conf(  
                                                        job_uid=job_uid,
                                                        conf_names=conf_names,
                                                        serv_names=serv_names,
                                                        service_info_list=service_info_list,
                                                        rsc_constraint=rsc_constraint,
                                                        user_constraint=user_constraint,
                                                        work_condition=work_condition
                                                    )
                    if ans_found==1:
                        prev_conf[job_uid]=conf
                        prev_flow_mapping[job_uid]=flow_mapping
                        prev_resource_limit[job_uid]=resource_limit
                        print('最终采用:降低配置')
                        break
                elif micro_plan==common.MOVE_CLOUD:
                    ans_found,conf,flow_mapping,resource_limit=micro_search_move_cloud(  
                                                        job_uid=job_uid,
                                                        conf_names=conf_names,
                                                        serv_names=serv_names,
                                                        service_info_list=service_info_list,
                                                        rsc_constraint=rsc_constraint,
                                                        user_constraint=user_constraint,
                                                        work_condition=work_condition
                                                    )
                    if ans_found==1:
                        prev_conf[job_uid]=conf
                        prev_flow_mapping[job_uid]=flow_mapping
                        prev_resource_limit[job_uid]=resource_limit
                        print('最终采用:移到云端')
                        break
    print(prev_conf[job_uid])
    print(prev_flow_mapping[job_uid])
    print(prev_resource_limit[job_uid])

    return prev_conf[job_uid], prev_flow_mapping[job_uid], prev_resource_limit[job_uid]  #沿用之前的配置




# -----------------
# 该函数只使用get_coldstart_plan_bayes冷启动函数来解决问题，作用是作为无画像的baseline存在。
# ---- 调度入口 ----
def scheduler_only_cold(
    job_uid=None,
    dag=None,
    system_status=None,
    work_condition=None,
    portrait_info=None,
    user_constraint=None,
    appended_result_list=None
):


    assert job_uid, "should provide job_uid"
    ########################################################################################################
    # 调度器首先要根据dag来得到关于服务的各个信息
    #（1）构建serv_names:根据dag获取serv_names，涉及的各个服务名
    #（2）构建service_info_list：根据serv_names，加上“_trans”构建传输阶段，从service_info_dict提取信息构建service_info_list
    # (3) 构建conf_names：根据service_info_list里每一个服务的配置参数，得到conf_names，保存所有影响任务的配置参数（不含ip）
    #（4）获取每一个设备的资源约束，以及每一个服务的资源上限
    serv_names=dag['flow']
    
    service_info_list=[]  #建立service_info_list，需要处理阶段，也需要传输阶段
    for serv_name in serv_names:
        service_info_list.append(service_info_dict[serv_name])
        # 不考虑传输时延了
        # service_info_list.append(service_info_dict[serv_name+"_trans"])
    
    conf=dict()
    for service_info in service_info_list:
        service_conf_names = service_info['conf']
        for service_conf_name in service_conf_names:
            if service_conf_name not in conf:
                conf[service_conf_name]=1
    conf_names=list(conf.keys())  #conf_names里保存所有配置参数的名字
    ########################################################################################################

    ########################################################################################################
    #调度器要展示最新结果并获取执行时延，以及当前资源约束（资源约束可能随时变化）
    all_proc_delay=0
    if appended_result_list!=None:
        print("展示最新执行结果ext_runtime")
        print(appended_result_list[-1]['ext_runtime'])
        serv_proc_delays=appended_result_list[-1]['ext_runtime']['plan_result']['process_delay']
        
        for serv_name in serv_proc_delays.keys():
            all_proc_delay+=serv_proc_delays[serv_name]
    

    with open('static_data.json', 'r') as f:  
        static_data = json.load(f)
    
    #当前资源约束情况
    rsc_constraint=static_data['rsc_constraint']
    '''
    rsc_constraint:{
            "114.212.81.11": {"cpu": 1.0, "mem": 1.0}, 
            "172.27.143.164": {"cpu": 0.1, "mem": 0.1}, 
            "172.27.151.145":{"cpu": 0.1, "mem": 0.2}
        }
    '''
    ########################################################################################################

    ########################################################################################################

    need_seek_table=0 #需要查表
  
    # 一、初始情况下选择使用查表
    if not bool(work_condition) or not bool(user_constraint):
        print('初始冷启动')
        need_seek_table=1
    # 二、时延超出常规时需要查表
    elif all_proc_delay > user_constraint['delay']:
        print('时延不满足')
        need_seek_table==1

    ########################################################################################################

    
    # 如果需要查表
    if need_seek_table==1:
        #查找时用于剪枝的资源上限
        myportrait=RuntimePortrait(pipeline=serv_names)
        rsc_upper_bound={}
        for serv_name in serv_names:
            serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
            rsc_upper_bound[serv_name]={}
            rsc_upper_bound[serv_name]['cpu_limit']=serv_rsc_cons['cpu']['edge']
            rsc_upper_bound[serv_name]['mem_limit']=serv_rsc_cons['mem']['edge']
        print("画像提供的资源上限")
        print(rsc_upper_bound)

        rsc_down_bound={
            "face_detection": {"cpu_limit": 0.0, "mem_limit": 0.0}, 
            "gender_classification": {"cpu_limit": 0.0, "mem_limit": 0.0}
        }
    
        ans_found, conf, flow_mapping, resource_limit=get_coldstart_plan_bayes(
                                                        conf_names=conf_names,
                                                        serv_names=serv_names,
                                                        service_info_list=service_info_list,
                                                        rsc_constraint=rsc_constraint,
                                                        user_constraint=user_constraint,
                                                        rsc_upper_bound=rsc_upper_bound,
                                                        rsc_down_bound=rsc_down_bound,
                                                        work_condition=work_condition
                                                    )

        if ans_found==1:#如果确实找到了，需要更新字典里之前保存的三大配置
            prev_conf[job_uid]=conf
            prev_flow_mapping[job_uid]=flow_mapping
            prev_resource_limit[job_uid]=resource_limit
        else:
            print("查表已经失败，出现重大问题")
            return None #后续这里应该改成别的，比如默认配置什么的
    


    return prev_conf[job_uid], prev_flow_mapping[job_uid], prev_resource_limit[job_uid]  #沿用之前的配置


# -----------------
# 该函数仅仅是不断读取静态文件中存储的计划，用于验证任务的运行性能
# ---- 调度入口 ----
def scheduler_test(
    job_uid=None,
    dag=None,
    system_status=None,
    work_condition=None,
    portrait_info=None,
    user_constraint=None,
    appended_result_list=None
):
    assert job_uid, "should provide job_uid"


    # 测试部分——-负反馈测试器，利用静态数据测试实时变化
    all_proc_delay=0
    if appended_result_list!=None:
        print("展示最新执行结果ext_runtime")
        print(appended_result_list[-1]['ext_runtime'])
        serv_proc_delays=appended_result_list[-1]['ext_runtime']['plan_result']['process_delay']
        for serv_name in serv_proc_delays.keys():
            all_proc_delay+=serv_proc_delays[serv_name]

    all_proc_delay=0
    if appended_result_list!=None:
        #print("展示最新执行结果ext_runtime")
        #print(appended_result_list[-1]['ext_runtime'])
        serv_proc_delays=appended_result_list[-1]['ext_runtime']['plan_result']['process_delay']
        
        for serv_name in serv_proc_delays.keys():
            all_proc_delay+=serv_proc_delays[serv_name]
        print('感知到总处理时延是',all_proc_delay)
        print(serv_proc_delays)
   
    static_data={}

    with open('static_data.json', 'r') as f:  
        static_data = json.load(f) 

    
    conf = static_data['conf']
    flow_mapping = static_data['flow_mapping']
    resource_limit = static_data['resource_limit']

    if_test=static_data['if_test']

    if if_test==1: #此时仅仅是读取文件中的配置并返还
        return conf, flow_mapping, resource_limit
    
   

