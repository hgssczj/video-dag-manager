import sys  
import itertools
from kb_user import KnowledgeBaseUser,kb_plan_path,kb_data_path
from logging_utils import root_logger


import json

import copy

prev_conf = dict()
prev_flow_mapping = dict()
prev_runtime_info = dict()

init_conf = dict()
init_flow_mapping = dict()

cloud_ip="114.212.81.11"


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
    job_uid=None,
    dag=None,
    resource_info=None,
    runtime_info=None,
    user_constraint=None,
):
    assert job_uid, "should provide job_uid"

    global prev_conf, prev_flow_mapping
    global init_conf,init_flow_mapping
    global conf_and_serv_info,service_info_dict

    # 调度器首先要根据dag来得到关于服务的各个信息
    # （1）构建serv_names:根据dag获取serv_names，涉及的各个服务名
    serv_names=dag['flow']
    #（2）构建service_info_list：根据serv_names，加上“_trans”构建传输阶段，从service_info_dict提取信息构建service_info_list
    service_info_list=[]  #建立service_info_list，需要处理阶段，也需要传输阶段
    for serv_name in serv_names:
        service_info_list.append(service_info_dict[serv_name])
        service_info_list.append(service_info_dict[serv_name+"_trans"])
    # (3)构建conf_names：根据service_info_list里每一个服务的配置参数，得到conf_names，保存所有影响任务的配置参数（不含ip）
    conf=dict()
    for service_info in service_info_list:
        service_conf_names = service_info['conf']
        for service_conf_name in service_conf_names:
            if service_conf_name not in conf:
                conf[service_conf_name]=1
    conf_names=list(conf.keys())  #conf_names里保存所有配置参数的名字

    # conf_names=["reso","fps","encoder"]
    # serv_names=["face_detection", "face_alignment"]
    print("初始冷启动,获取任务信息：")
    print("conf_names:",conf_names)
    print("serv_names",serv_names)

    #还需要rsc_constraint，也就是资源约束才能进行正常的查询工作。

    rsc_constraint={
        "114.212.81.11":{
            "cpu": 1.0,
            "mem":1.0
        },
        "172.27.143.164": {
            "cpu": 1.0,
            "mem": 1.0
        },
        "172.27.151.145": {
            "cpu": 1.0,
            "mem": 1.0 
        },
    }
    # 描述每一种服务所需的中资源阈值，它限制了贝叶斯优化的时候采取怎样的内存取值范围
    rsc_upper_bound={
        'face_detection':{
            'cpu_limit':0.5,
            'mem_limit':0.5,
        },
        'face_alignment':{
            'cpu_limit':0.5,
            'mem_limit':0.5,
        }
    }
    cold_starter=KnowledgeBaseUser(conf_names=conf_names,
                                  serv_names=serv_names,
                                  service_info_list=service_info_list,
                                  user_constraint=user_constraint,
                                  rsc_constraint=rsc_constraint,
                                  rsc_upper_bound=rsc_upper_bound
                                  )
    n_trials=500
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
    # 初始情况下选择使用冷启动策略
    if not bool(runtime_info) or not bool(user_constraint):
        root_logger.info("to get COLD start executation plan")
        return get_cold_start_plan(
            job_uid=job_uid,
            dag=dag,
            resource_info=resource_info,
            runtime_info=runtime_info,
            user_constraint=user_constraint
        )

    # ---- 若有负反馈结果，则进行负反馈调节 ----
    global prev_conf, prev_flow_mapping

    return prev_conf[job_uid], prev_flow_mapping[job_uid]  #沿用之前的配置
