import sys  
sys.path.append('F:\\2022-0研究生相关\\2_个人工作\\工作1_新系统\\video南网云边协同稳定版\\video-dag-manager-no-render-demo\\video-dag-manager')
print(sys.path)
import itertools
from logging_utils import root_logger
import pandas as pd
import os
import time
import math
import requests
import json
import time
import copy

prev_conf = dict()
prev_flow_mapping = dict()
prev_runtime_info = dict()

init_conf = dict()
init_flow_mapping = dict()

cloud_ip="114.212.81.11"

knowledgebase_path="knowledgebase_rotate"  

conf_and_serv_info={  #各种配置参数的可选值
    "reso":["360p", "480p", "720p", "1080p"],
    "fps":[1, 5, 10, 20, 30],
    "encoder":["JPEG"],
    "face_alignment_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],
    "face_alignment_trans_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],
}

# 我人为以下的servcie_info_list实际上应该根据流水线的需要随时组建
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

# 以下是可能影响任务性能的可配置参数，用于指导模块化知识库的建立
model_op={  
    "114.212.81.11":{
        "model_id": 0,
        "node_ip": "114.212.81.11",
        "node_role": "cloud"
    },
    "172.27.143.164": {
        "model_id": 0,
        "node_ip": "172.27.143.164",
        "node_role": "host"  
    },
    "172.27.151.145": {
        "model_id": 0,
        "node_ip": "172.27.151.145",
        "node_role": "host"  
    },

}

''' # 根据流水线pipeline 可以从service_info_dict里提取信息构建service_info_list
service_info_list=[
    {
        "name":'face_detection',
        "value":'face_detection_proc_delay',
        "conf":["reso","fps"]
    },
    {
        "name":'face_alignment',
        "value":'face_alignment_proc_delay',
        "conf":["reso","fps"]
    },
    {
        "name":'face_detection_trans',
        "value":'face_detection_trans_delay',
        "conf":["reso","fps"]
    },
    {
        "name":'face_alignment_trans',
        "value":'face_alignment_trans_delay',
        "conf":["reso","fps"]
    },
]
'''

# 获取总预估的时延

'''
"flow_mapping": {
    "face_detection": {
        "model_id": 0,
        "node_ip": "114.212.81.11",
        "node_role": "cloud"
    },
    "face_alignment": {
        "model_id": 0,
        "node_ip": "114.212.81.11",
        "node_role": "cloud"
    }
},
"conf": {
    "reso": "360p"
    "fps": 1,
    "encoder": "JPEG",
}
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
'''

# 根据配置选择从各个服务的性能评估器中提取性能指标，给出对整个流水线的性能评估结果
def get_pred_delay(conf=None, flow_mapping=None,service_info_list=None):
    # 知识库所在目录名
    global knowledgebase_path
    # 存储配置对应的各阶段时延，以及总时延
    pred_delay_list=[]
    pred_delay_total=0
    status=0  #为0表示字典中没有满足配置的存在
    # 对于service_info_list里的service_info依次评估性能
    for service_info in service_info_list:
        # （1）加载服务对应的性能评估器
        f=open(knowledgebase_path+"/"+service_info['name']+".json")  
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
            # 现在还差ip地址,首先要判断当前评估器是不是针对传输阶段进行的：
        ip_for_dict_index=service_info['name'].find("_trans") 
        if ip_for_dict_index>0:
            # 去除服务名末尾的_trans，形如“face_detection”
            ip_for_dict_name=service_info['name'][0:ip_for_dict_index] 
        else: #当前不是trans
            ip_for_dict_name=service_info['name']
        ip_for_dict=flow_mapping[ip_for_dict_name]['node_ip']
            # 形如["360p"，"1","JPEG","114.212.81.11"]
        conf_for_dict.append(str(ip_for_dict))  
        # （3）根据conf_for_dict，从性能评估器中提取该服务的评估时延
        sub_evaluator=evaluator
        for i in range(0,len(conf_for_dict)-1):
            sub_evaluator=sub_evaluator[conf_for_dict[i]]
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
    # （4）构建conf_list：把conf_names里每一个配置的可选配置参数
    conf_list=[]
    for conf_name in conf_names:   
        conf_list.append(conf_and_serv_info[conf_name])
    # （5）构建serv_ip_list，包含每一个服务的可选ip
    serv_ip_list=[]
    for serv_name in serv_names:
        serv_ip=serv_name+"_ip"
        serv_ip_list.append(conf_and_serv_info[serv_ip])
    # conf_names=["reso","fps","encoder"]
    # serv_names=["face_detection", "face_alignment"]
    # conf_list=[ ["360p", "480p", "720p", "1080p"],[1, 5, 10, 20, 30],["JPEG"]]
    # serv_ip_list=[["114.212.81.11","172.27.143.164","172.27.151.145"],["114.212.81.11","172.27.143.164","172.27.151.145"]]

    print("初始冷启动,获取任务信息：")
    print("conf_names:",conf_names)
    print("serv_names",serv_names)
    print("conf_list",conf_list)
    print("serv_ip_list",serv_ip_list)

    # 获取时延约束
    delay_constraint = user_constraint["delay"]

    best_conf={}
    best_flow_mapping={}
    best_pred_delay_list=[]
    best_pred_delay_total=-1

    min_conf={}
    min_flow_mapping={}
    min_pred_delay_list=[]
    min_pred_delay_total=-1
    
    conf_combine=itertools.product(*conf_list)
    for conf_plan in conf_combine:
        serv_ip_combine=itertools.product(*serv_ip_list)
        for serv_ip_plan in serv_ip_combine:# 遍历所有配置和卸载策略组合
            conf={}
            flow_mapping={}
            for i in range(0,len(conf_names)):
                conf[conf_names[i]]=conf_plan[i]
            for i in range(0,len(serv_names)):
                flow_mapping[serv_names[i]]=model_op[serv_ip_plan[i]]
            # 经过遍历，得到一组conf额一组flow_mapping。现在需要根据conf和flow_mapping来从知识库中估计时延
            status,pred_delay_list,pred_delay_total = get_pred_delay(conf=conf,flow_mapping=flow_mapping,service_info_list=service_info_list)
            if status == 0: #如果为0，意味着配置在字典中找不到对应的性能评估结果，知识库没有存储这种配置对应的估计结果
                continue
            if best_pred_delay_total<0:   #初始化最优配置和最小配置
                best_conf=conf
                best_flow_mapping=flow_mapping
                best_pred_delay_list=pred_delay_list
                best_pred_delay_total=pred_delay_total

                min_conf=conf
                min_flow_mapping=flow_mapping
                min_pred_delay_list=pred_delay_list
                min_pred_delay_total=pred_delay_total

            elif pred_delay_total < delay_constraint*0.7 and pred_delay_total>best_pred_delay_total: #选出一个接近约束且比较大的
                best_conf=conf
                best_flow_mapping=flow_mapping
                best_pred_delay_list=pred_delay_list
                best_pred_delay_total=pred_delay_total
            
            elif pred_delay_total < best_pred_delay_total: #选出一个最小的
                min_conf=conf
                min_flow_mapping=flow_mapping
                min_pred_delay_list=pred_delay_list
                min_pred_delay_total=pred_delay_total
    
    # 完成遍历后，应该可以找到一个比较优秀的冷启动结果
    print("最优配置是：")
    print(best_conf)
    print(best_flow_mapping)
    print(best_pred_delay_list)
    print(best_pred_delay_total)
    print("最小配置是：")
    print(min_conf)
    print(min_flow_mapping)
    print(min_pred_delay_list)
    print(min_pred_delay_total)
    print("时延约束是",delay_constraint)

    if min_pred_delay_total > delay_constraint:
        print("约束过于严格，选择最小配置")
        prev_conf[job_uid] = min_conf
        prev_flow_mapping[job_uid] = min_flow_mapping
        init_conf[job_uid] = copy.deepcopy(min_conf)
        init_flow_mapping[job_uid] =copy.deepcopy(min_flow_mapping)
        return prev_conf[job_uid], prev_flow_mapping[job_uid]
    else:
        print("约束不算特别严格，选择最优策略")
        prev_conf[job_uid] = best_conf
        prev_flow_mapping[job_uid] = best_flow_mapping
        init_conf[job_uid] = copy.deepcopy(best_conf)
        init_flow_mapping[job_uid] =copy.deepcopy(best_flow_mapping)
        return prev_conf[job_uid], prev_flow_mapping[job_uid]




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


if __name__ == "__main__":

    

    dag={}
    dag["flow"]= ["face_detection", "face_alignment"]
    user_constraint={}
    user_constraint["delay"]= 0.5
    knowledgebase_path="knowledgebase_bayes"  

    get_cold_start_plan(
        job_uid=1,
        dag=dag,
        user_constraint=user_constraint,
    )