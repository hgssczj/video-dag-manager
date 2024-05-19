    
import requests
import time
import csv
import json
import copy
import os
import datetime
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.font_manager import FontProperties
import time
import optuna
import itertools
import random
import re
import common

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
matplotlib.use('TkAgg')

#试图以类的方式将整个独立建立知识库的过程模块化



from common import KB_DATA_PATH,NO_BAYES_GOAL,BEST_ALL_DELAY,BEST_STD_DELAY,model_op,conf_and_serv_info
from RuntimePortrait import RuntimePortrait

# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","gender_classification"]   

myportrait = RuntimePortrait(pipeline=serv_names)
#从画像里收集服务在边缘端的资源上限
# 描述每一种服务所需的中资源阈值，它限制了贝叶斯优化的时候采取怎样的内存取值范围
rsc_upper_bound = {}
for serv_name in serv_names:
    serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
    rsc_upper_bound[serv_name] = {}
    rsc_upper_bound[serv_name]['cpu_limit'] = serv_rsc_cons['cpu']['edge']
    rsc_upper_bound[serv_name]['mem_limit'] = serv_rsc_cons['mem']['edge']

available_resource={  
                '114.212.81.11': {  
                    'node_role': 'cloud',  
                    'available_cpu': 1.0,  
                    'available_mem': 1.0 
                },
                '192.168.1.7':{  
                    'node_role': 'host',  
                    'available_cpu': 0.7,  
                    'available_mem': 0.8 
                },
            }

# objective：
# 用途：作为贝叶斯优化采样时需要优化的目标函数。
# 方法：根据bayes_goal的取值，选择优化目标，可能是使当前采样结果最小，也可能是为了求平均值。
# 返回值：csv的文件名
def objective(trial):
    conf={}
    flow_mapping={}
    resource_limit={}

    for conf_name in conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
        # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
        conf[conf_name]=trial.suggest_categorical(conf_name,conf_and_serv_info[conf_name])
    
    # 确定配置后，接下来选择云边协同切分点，以及各个服务的资源使用量。此时要格外注意。
    # 流水线中索引小于edge_cloud_cut_point的将被留在边缘执行，因此edge_cloud_cut_point的最大取值是边缘所能负载的最多的服务个数
    # 在指定的配置之下，可以计算出其最低资源需求（cpu按照0.05，mem按照其最低资源需求）
    edge_serv_num=0
    edge_cpu_use=0.0
    edge_mem_use=0.0
    # 注意，需要保证cpu以0.05为最小单位时不会出现偏差小数点
    # 如果让0.6-0.05，那么会得到0.5499999999999999这种结果，此时只能通过round保留为小数点后两位解决
    # 因此以0.05为单位计算时要时刻使用round
    
    myportrait=RuntimePortrait(pipeline=serv_names)
    for serv_name in serv_names:
        # 该服务的最小cpu消耗量
        edge_cpu_use=round(edge_cpu_use+0.05,2)
        # 该服务的最小mem消耗量
        task_info = {
                'service_name': serv_name,
                'fps': conf['fps'],
                'reso': common.reso_2_index_dict[conf['reso']],
                'obj_num': 1 #因为采样用的视频中只有一张人脸
            }
        rsc_threshold = myportrait.predict_resource_threshold(task_info=task_info)
        edge_mem_use = edge_mem_use + rsc_threshold['mem']['edge']['lower_bound']

        if edge_cpu_use <= available_resource[common.edge_ip]['available_cpu'] and \
            edge_mem_use <= available_resource[common.edge_ip]['available_mem']:
            edge_serv_num+=1 #说明可转移到边缘端的服务又增加了一个
        else:
            break
    # 最终edge_serv_num就是切分点的上限
    conf_and_serv_info["edge_cloud_cut_point"]=[i for i in range(edge_serv_num + 1)]
    # 由于optuna本身的性质，同名的超参数不能具有相同的suggest_categorical取值范围。
    # 所以我改用suggest_int代替suggest_categorical
    choice_idx = trial.suggest_int('edge_cloud_cut_choice', 0, len(conf_and_serv_info["edge_cloud_cut_point"])-1)
    edge_cloud_cut_choice=conf_and_serv_info["edge_cloud_cut_point"][choice_idx]

    # 为了防止边缘端上给各个服务分配的cpu使用率被用光，每一个服务能够被分配的CPU使用率存在上限
    # 下限为0.05，上限为目前可用资源量减去剩下的服务数量乘以0.05

    edge_cpu_left=available_resource[common.edge_ip]['available_cpu']
    for i in range(1,len(serv_names)+1):
        serv_name = serv_names[i-1]
        
        if serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
            flow_mapping[serv_name] = model_op[common.edge_ip]
        else:  # 服务索引大于等于云边协同切分点，在云执行
            flow_mapping[serv_name] = model_op[common.cloud_ip]
        
        # 根据云边切分点选择flow_mapping的选择
        # 接下来进行cpu使用率的选择（但是不进行mem的选择，假设mem不变，在判断切分点的时候就已经确保了内存捕获超过限制）
        # 所以内存限制永远是1.0
        serv_cpu_limit=serv_name+"_cpu_util_limit"

        resource_limit[serv_name]={}
        resource_limit[serv_name]["mem_util_limit"]=1.0

        if flow_mapping[serv_name]["node_role"] =="cloud":  #对于云端没必要研究资源约束下的情况
            resource_limit[serv_name]["cpu_util_limit"]=1.0
            
        else: #边端服务会消耗cpu资源
            
            # cpu_upper_bound
            # edge_cloud_cut_choice实际上是要留在边缘的服务的数量，i表示当前是第几个留在边缘的服务，i>=1
            # edge_cloud_cut_choice-i表示还剩下多少服务要分配到边缘上。至少给它们每一个留下0.05的cpu资源
            cpu_upper_bound=edge_cpu_left-round((edge_cloud_cut_choice-i)*0.05,2)
            cpu_upper_bound=round(cpu_upper_bound,2)
            cpu_upper_bound=max(0.05,cpu_upper_bound) #分配量不得超过此值
            cpu_upper_bound=min(rsc_upper_bound[serv_name]['cpu_limit'],cpu_upper_bound) #同时也不应该超出其中资源上限
            # 最后给出当前服务可选的cpu
            cpu_select=[]
            cpu_select=[x for x in conf_and_serv_info[serv_cpu_limit] if x <= cpu_upper_bound]

            # 由于optuna本身的性质，同名的超参数不能具有相同的suggest_categorical取值范围。
            # 所以我改用suggest_int代替suggest_categorical
            choice_idx=trial.suggest_int(serv_cpu_limit,0,len(cpu_select)-1)
            resource_limit[serv_name]["cpu_util_limit"]=cpu_select[choice_idx]
        
            edge_cpu_left=round(edge_cpu_left-resource_limit[serv_name]["cpu_util_limit"],2)


    avg_delay=1
    # 使用此函数，目标是最优化采样得到的时延

    return avg_delay



# 以贝叶斯采样的方式获取离线知识库
#  kb_builder.sample_and_record_bayes(sample_bound=10,n_trials=80,bin_nums=100,bayes_goal=BEST_ALL_DELAY)
def sample_and_record_bayes(n_trials):

    study = optuna.create_study()
    study.optimize(objective,n_trials=n_trials)

    print(study.best_params)


if __name__ == "__main__":

    dag_name=''
    for serv_name in ["face_detection","gender_classification"]  :
        if len(dag_name)==0:
            dag_name+=serv_name
        else:
            dag_name+='-'+serv_name
    
    print(dag_name)

    
    print("画像提供的资源上限")
    print(rsc_upper_bound)
    #sample_and_record_bayes(n_trials=100)
