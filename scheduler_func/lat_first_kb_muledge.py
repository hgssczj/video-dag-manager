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
    'face_alignment':{
        "name":'face_alignment',
        "value":'face_alignment_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    'face_detection_trans':{
        "name":'face_detection_trans',
        "value":'face_detection_trans_delay',
        "conf":["reso","fps","encoder"]
    },
    'face_alignment_trans':{
        "name":'face_alignment_trans',
        "value":'face_alignment_trans_delay',
        "conf":["reso","fps","encoder"]
    }
}


def get_cold_start_plan(
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    rsc_upper_bound=None,
):        

    cold_starter=KnowledgeBaseUser(conf_names=conf_names,
                                  serv_names=serv_names,
                                  service_info_list=service_info_list,
                                  user_constraint=user_constraint,
                                  rsc_constraint=rsc_constraint,
                                  rsc_upper_bound=rsc_upper_bound
                                  )
    n_trials=250
    ans_params_valid=cold_starter.get_coldstart_plan_bayes(n_trials=n_trials)
    for param_valid in ans_params_valid:
        for key in param_valid.keys():
            print(param_valid[key])

    return None




# -----------------
# ---- 调度入口 ----
def scheduler(
    job_uid=None,
    dag=None,
    resource_info=None,
    runtime_info=None,
    user_constraint=None,
):
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
    rsc_constraint={}
    rsc_upper_bound={}
  

    # （1）初始情况下选择使用冷启动策略
    if not bool(runtime_info) or not bool(user_constraint):
        root_logger.info("to get COLD start executation plan")
        # 首先获取当前设备上的资源限制，以及每一个服务的中资源阈值
        with open('rsc_constraint.json', 'r') as f:  
            rsc_constraint = json.load(f) 
        with open('rsc_upper_bound.json', 'r') as f:  
            rsc_upper_bound = json.load(f)

        # 基于前文开始利用知识库
        cold_starter=KnowledgeBaseUser( conf_names=conf_names,
                                        serv_names=serv_names,
                                        service_info_list=service_info_list,
                                        user_constraint=user_constraint,
                                        rsc_constraint=rsc_constraint,
                                        rsc_upper_bound=rsc_upper_bound
                                        )
        '''
         (1)对于能满足时延约束的解
		 按照资源约束违反程度从小到大排序；
		 按照云端服务使用数量从小到大排序；
		 按照时延从大到小排序；
		 然后选取最优解——满足约束，且违反程度最小、对云端使用最少、最逼近约束的.

        (2)对于不能满足时延约束的解
		此时没有能满足时延约束的解，怎么办？那只能权衡一下三种指标了。
		按照某种权重，对所有不满足实验约束的解的配置进行排序，然后选择一个最优的，希望它在时延上较小	且资源约束较好。
        '''
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
            
        sorted_params=[]
        if len(params_in_delay_cons)>0:
            sorted_params=sorted(params_in_delay_cons,key=lambda item:(item['deg_violate'],item['num_cloud'],-item['pred_delay_total']))
        else:
            sorted_params=sorted(params_out_delay_cons,key=lambda item:(item['deg_violate']+item['pred_delay_total']))

        print('排序后的解')
        for param_valid in sorted_params:
            for key in param_valid.keys():
                print(key,param_valid[key])
        best_params=sorted_params[0]
        ''' 形如：
        conf {'reso': '720p', 'fps': 1, 'encoder': 'JPEG'}
        flow_mapping {'face_detection': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}, 'face_alignment': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}}
        resource_limit {'face_detection': {'cpu_util_limit': 0.25, 'mem_util_limit': 0.3}, 'face_alignment': {'cpu_util_limit': 0.25, 'mem_util_limit': 0.3}}
        pred_delay_list [0.0020203223595251798, 0.01092090790088356, 0.0724544489689362, 0.027888460770631396]
        pred_delay_total 0.11328413999997633
        num_cloud 0
        deg_violate 0
        '''
        best_conf=best_params['conf']
        best_flow_mapping=best_params['flow_mapping']
        best_resource_limit=best_params['resource_limit']

        prev_conf[job_uid]=best_conf
        prev_flow_mapping[job_uid]=best_flow_mapping
        prev_resource_limit[job_uid]=best_resource_limit

        return best_conf, best_flow_mapping, best_resource_limit
    
    # （2）如果不是冷启动，现在应该怎么办？首先要看时延是否是满足的
    else:
            # 首先获取当前设备上的资源限制，以及每一个服务的中资源阈值
        with open('rsc_constraint.json', 'r') as f:  
            rsc_constraint = json.load(f) 
        with open('rsc_upper_bound.json', 'r') as f:  
            rsc_upper_bound = json.load(f)
        


    # ---- 若有负反馈结果，则进行负反馈调节 ----


    return prev_conf[job_uid], prev_flow_mapping[job_uid], prev_resource_limit[job_uid]  #沿用之前的配置
