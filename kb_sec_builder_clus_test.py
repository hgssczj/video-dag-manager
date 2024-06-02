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
from AccuracyPrediction import AccuracyPrediction
import re  
from sklearn.cluster import KMeans  
import math
os.environ['OMP_NUM_THREADS'] = '1'
'''
设置以上环境变量是为了解决调用k-means时的警告:
UserWarning: KMeans is known to have a memory leak on Windows with MKL, 
when there are less chunks than available threads. 
You can avoid it by setting the environment variable OMP_NUM_THREADS=1
'''


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
matplotlib.use('TkAgg')

#试图以类的方式将整个独立建立知识库的过程模块化



from common import KB_DATA_PATH,model_op,conf_and_serv_info

# 每一个特定任务对应一个KnowledgeBaseBuilder类
class KnowledgeBaseBuilder():   
    
    #知识库建立者，每次运行的时候要指定本次实验名称，用于作为记录实验的文件
    def __init__(self,expr_name,node_ip,node_addr,query_addr,service_addr,query_body,conf_names,serv_names,service_info_list,rsc_upper_bound,kb_name,refer_kb_name):

        #(1)基础初始化：实验名称、ip、各类节点地址
        self.expr_name = expr_name #用于构建record_fname的实验名称
        self.node_ip = node_ip #指定一个ip,用于从resurece_info获取相应资源信息
        self.node_addr = node_addr #指定node地址，向该地址更新调度策略
        self.query_addr = query_addr #指定query地址，向该地址提出任务请求并获取结果
        self.service_addr = service_addr #指定service地址，从该地址获得全局资源信息


        #(2)任务初始化：待执行任务的query内容、任务相关的配置、涉及的服务名称、服务相关的信息、任务的id
        self.query_body = query_body
        self.conf_names = conf_names
        self.serv_names = serv_names
        self.service_info_list=service_info_list
        self.query_id = None #查询返回的id

        #(3)记录初始化：记录数据所需的文件描述符、防止重复的字典、写入文件对象、每一种配置采集的样本数
        self.fp=None
        self.written_n_loop = dict()
        self.writer = None
        self.sample_bound = None
        

        #(4)贝叶斯初始化：基于贝叶斯优化建立稀疏知识库时，为了测算采样的均匀程度，所需要的字典、区间划分数、bayes优化目标
        self.explore_conf_dict=dict()  # 用于记录字典查询过程中设计到的配置，不希望其重复
        self.explore_conf_dict['delay']={}
        self.explore_conf_dict['accuracy']={}

        self.bin_nums= None

        #(5)通信初始化：用于建立连接
        self.sess = requests.Session()  #客户端通信

        #(6)中资源阈值初始化：描述每一个服务所需的中资源阈值上限
        self.rsc_upper_bound = rsc_upper_bound

        #(7)可用资源初始化:描述云边目前可用的资源
        '''
        available_resource 
            {
            '114.212.81.11': 
                {'available_cpu': 1.0, 'available_mem': 1.0, 'node_role': 'cloud'}, 
            '192.168.1.7':
                {'available_cpu': 1.0, 'available_mem': 1.0, 'node_role': 'edge'}
            }
        '''
        self.available_resource={}
        self.available_resource[common.cloud_ip]={'available_cpu': 1.0, 'available_mem': 1.0, 'node_role': 'cloud'}
        self.available_resource[common.edge_ip]={'available_cpu': 1.0, 'available_mem': 1.0, 'node_role': 'edge'}

        # 知识库目录
        self.kb_name = kb_name

        # 用于测试参考用的知识库目录
        self.refer_kb_name = refer_kb_name

 
    def set_expr_name(self,expr_name):
        self.expr_name=expr_name

    def set_node_ip(self,node_ip):
        self.node_ip=node_ip
    
    def set_node_addr(self,node_addr):
        self.node_addr=node_addr
    
    def set_query_addr(self,query_addr):
        self.query_addr=query_addr
    
    def set_query_addr(self,sample_amount):
        self.sample_amount=sample_amount

    def set_bin_nums(self,bin_nums):
        self.bin_nums=bin_nums

   
    # allocate_resources:
    # 用途：edge_serv_cpu_needs是一个列表，记录了边缘端上各个服务的资源需求。该函数可以将总量为1.0的cpu资源以0.05的粒度按比例分配给这些服务，且不会超出总量，也不会为0
    # 方法：进行复杂的算法验证
    # 返回值：列表edge_serv_cpu_alloc，里面的元素与edge_serv_cpu_needs一一对应，其和可能超过1.0
    def allocate_resources(self,edge_serv_cpu_needs):  
        n = len(edge_serv_cpu_needs)  
        total_need = sum(edge_serv_cpu_needs)  
        edge_serv_cpu_alloc = [0.05] * n  # 初始化分配数组，每个任务先分配0.05的资源  
        allocated_so_far = n * 0.05  # 已经分配的资源总量  
    
        # 如果没有剩余需求，或者总需求原本就不足n*0.05，则直接返回  
        if total_need <= n * 0.05:  
            return edge_serv_cpu_alloc  
    
        # 计算剩余可分配的资源  
        remaining_resources = min(total_need - n * 0.05, 1 - n * 0.05)  
    
        # 分配剩余资源  
        for i, need in enumerate(edge_serv_cpu_needs):  
            # 按比例分配剩余的资源，并取0.05的倍数  
            allocation = round((need / (total_need - n * 0.05)) * remaining_resources / 0.05) * 0.05  
            edge_serv_cpu_alloc[i] += allocation  
            allocated_so_far += allocation  
            remaining_resources -= allocation  
    
            # 如果资源已经分配完，或者当前任务的分配量已经足够，则退出循环  
            if remaining_resources <= 0 or edge_serv_cpu_alloc[i] >= need:  
                break  
    
        # 如果还有剩余资源，并且总分配量小于1.0，则尝试将剩余资源分配给未满足需求的任务  
        while remaining_resources > 0 and allocated_so_far < 1.0:  
            # 找到未满足需求的任务  
            for i, alloc in enumerate(edge_serv_cpu_alloc):  
                if alloc < need:  
                    # 分配最小单位0.05，直到资源耗尽或任务需求满足  
                    extra_allocation = min(remaining_resources, 0.05 - (alloc % 0.05))  
                    edge_serv_cpu_alloc[i] += extra_allocation  
                    allocated_so_far += extra_allocation  
                    remaining_resources -= extra_allocation  
    
                    # 如果资源已经分配完或任务需求已经满足，则退出循环  
                    if remaining_resources <= 0 or edge_serv_cpu_alloc[i] >= need:  
                        break  
    
            # 如果所有任务都已满足需求或资源已耗尽，但总分配量仍小于1.0，则分配给第一个任务剩余资源  
            if remaining_resources > 0 and allocated_so_far < 1.0:  
                edge_serv_cpu_alloc[0] += remaining_resources  
                allocated_so_far += remaining_resources  
                remaining_resources = 0  
    
        # 如果总分配量超过1.0，则需要减少一些任务的分配量以符合限制  
        while allocated_so_far > 1.0:  
            # 从最后一个任务开始减少分配量  
            for i in range(n-1, -1, -1):  
                if edge_serv_cpu_alloc[i] > 0.05:  
                    # 减少最小单位0.05  
                    edge_serv_cpu_alloc[i] -= 0.05  
                    allocated_so_far -= 0.05  
    
                    # 如果总分配量已经满足要求，则退出循环  
                    if allocated_so_far <= 1.0:  
                        break  
        
        edge_serv_cpu_alloc = [round(x,2) for x in edge_serv_cpu_alloc]
        return edge_serv_cpu_alloc  
    
    # get_bottom_left_plans:
    # 用途：给出指定配置在多种原本协同切分点下的最优配置方案
    #       对于区间知识库的左下角配置，想要知道它在资源尽可能充分情况下需要的最小时延是多少。假设流水线上一共n个服务，为此最多需要考虑n+1种卸载方式，也就是给出n+1种plan
    #       资源约束按照obj_num为1的情况计算。由此导致的资源约束问题我觉得不是很重要。
    # 方法：调用上文函数进行复杂分配
    # 返回值：bottom_left_plans列表，里面全部都是该配置下资源最充分的分配方案
    def get_bottom_left_plans(self,conf):

        edge_serv_num=0
        edge_cpu_use=0.0
        edge_mem_use=0.0

        bottom_left_plans=[]
        #(1)计算该配置下最多能把几个服务放在边缘端上
        myportrait=RuntimePortrait(pipeline=self.serv_names)
        serv_cpu_edge_need=[]
        for serv_name in self.serv_names:
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
            serv_cpu_edge_need.append(rsc_threshold['cpu']['edge']['lower_bound'])

            if edge_cpu_use <= 1.0 and edge_mem_use <= 1.0:
                edge_serv_num+=1 #说明可转移到边缘端的服务又增加了一个
            else:
                break
        # 最终edge_serv_num就是切分点的上限,最多允许在边缘存放edge_serv_num个服务。


        # （2）为每一个云边协同切分点生成一个配置方案
        for edge_cloud_cut_choice in range(0,edge_serv_num+1):
            flow_mapping={}
            resource_limit={}
            
            #根据切分点，算出当前切分点下，边缘端上每一个服务的资源需求量，并计算出合适的资源分配量
            edge_serv_cpu_needs=serv_cpu_edge_need[0:edge_cloud_cut_choice]
            edge_serv_cpu_alloc=self.allocate_resources(edge_serv_cpu_needs)

            for i in range(0,len(self.serv_names)):
                serv_name = self.serv_names[i]
                if i < edge_cloud_cut_choice: #边端配置赋值
                    flow_mapping[serv_name] = model_op[common.edge_ip]
                    resource_limit[serv_name]={}
                    resource_limit[serv_name]["mem_util_limit"]=1.0
                    resource_limit[serv_name]["cpu_util_limit"]=edge_serv_cpu_alloc[i]
                else: #云端配置赋值
                    flow_mapping[serv_name] = model_op[common.cloud_ip]
                    resource_limit[serv_name]={}
                    resource_limit[serv_name]["mem_util_limit"]=1.0
                    resource_limit[serv_name]["cpu_util_limit"]=1.0
            
            #得到当前切分点下的配置方案
            new_plan={}
            new_plan['conf']=conf
            new_plan['flow_mapping']=flow_mapping
            new_plan['resource_limit']=resource_limit
            bottom_left_plans.append(new_plan)
    
        return bottom_left_plans

    # exmaine_explore_distribution:
    # 用途：获取贝叶斯优化采样过程中，采样结果的均匀程度，即方差（可考虑改成CoV）
    # 方法：将字典explore_conf_dict中的键值对的值提取出来构建数组，提取方差
    # 返回值：方差
    def exmaine_explore_distribution(self):
        #查看explore_conf_dict的分布,字典形如下，左边是配置，右边是对应的结果
        '''
        {
            '{"x0": 3, "x1": 0, "x2": 3, "x3": 3, "x4": 0, "x5": 5, "x6": 2, "x7": 8, "x8": 1, "x9": 1}': 1182503303,
            '{"x0": 3, "x1": 7, "x2": 8, "x3": 7, "x4": 2, "x5": 4, "x6": 3, "x7": 4, "x8": 7, "x9": 4}': 4743427873
        }
        '''
        assert self.bin_nums
        delay_value_list=[]
        accuracy_value_list=[]
        for conf_str in self.explore_conf_dict['delay']:
            delay_value_list.append(self.explore_conf_dict['delay'][conf_str])
        for conf_str in self.explore_conf_dict['accuracy']:
            accuracy_value_list.append(self.explore_conf_dict['accuracy'][conf_str])

        delay_value_list=sorted(np.array(delay_value_list))
        delay_bins = np.linspace(min(delay_value_list), max(delay_value_list), self.bin_nums+1) 
        delay_count_bins=np.bincount(np.digitize(delay_value_list,bins=delay_bins))

        accuracy_value_list=sorted(np.array(accuracy_value_list))
        accuracy_bins = np.linspace(min(accuracy_value_list), max(accuracy_value_list), self.bin_nums+1) 
        accuracy_count_bins=np.bincount(np.digitize(accuracy_value_list,bins=accuracy_bins))

        return delay_count_bins.std(), accuracy_count_bins.std() #返回方差

    # clear_explore_distribution:
    # 用途：还原self.explore_conf_dict为初始状况
    # 方法：将字典explore_conf_dict中的值重新恢复到以前
    # 返回值：方差
    def clear_explore_distribution(self,delay_range,accuracy_range):
        #查看explore_conf_dict的分布,字典形如下，左边是配置，右边是对应的结果
        '''
        {
            '{"x0": 3, "x1": 0, "x2": 3, "x3": 3, "x4": 0, "x5": 5, "x6": 2, "x7": 8, "x8": 1, "x9": 1}': 1182503303,
            '{"x0": 3, "x1": 7, "x2": 8, "x3": 7, "x4": 2, "x5": 4, "x6": 3, "x7": 4, "x8": 7, "x9": 4}': 4743427873
        }
        '''
        del self.explore_conf_dict
        self.explore_conf_dict=dict()
        self.explore_conf_dict['delay']={}
        self.explore_conf_dict['accuracy']={}
        self.explore_conf_dict["delay"]["min"]=delay_range["min"]
        self.explore_conf_dict["delay"]["max"]=delay_range["max"]
        self.explore_conf_dict["accuracy"]["min"]=accuracy_range["min"]
        self.explore_conf_dict["accuracy"]["max"]=accuracy_range["max"]

    
    # send_query_and_start：
    # 用途：发出查询，启动当前系统中的一个任务，并给予一个初始配置
    # 方法：向query_addr发出query_body，启动其中指定的任务；持续向query_addr发出新配置直到成功
    # 返回值：query_id，会被保存在当前知识库建立者的成员中

    # init_record_file：
    # 用途：初始化一个csv文件，fieldnames包括n_loop','frame_id','all_delay','edge_mem_ratio'，所有配置
    #      以及每一个服务（含中间传输阶段）的ip、时间、cpu和mem相关限制等。未来可能还需要再修改。
    # 方法：根据初始conf_names和serv_names获知所有配置参数和各个服务名
    # 返回值：该csv文件名，同时在当前目录下生成一个csv文件

    # write_in_file：
    # 用途：在csv文件中写入一条执行记录
    # 方法：参数r2r3r4分别表示资源信息、执行回应和runtime_info，从中提取可以填充csv文件的信息，利用字典把所有不重复的感知结果都保存在updatetd_result之中
    # 返回值：updatetd_result，保存了当前的运行时情境和执行结果


    # post_get_write：
    # 用途：更新调度计划，感知运行时情境，并调用write_in_file记录在csv文件中
    # 方法：使用update_plan接口将conf,flow_mapping,resource_limit应用在query_id指定的任务中
    #      依次获取运行时情境
    #      使用write_in_file方法将感知到的结果写入文件之中
    # 返回值：包含updated_result的键值对
  

    # get_write：
    # 用途：感知运行时情境，并调用write_in_file记录在csv文件中。相比post_get_write，不会修改调度计划。
    # 方法：依次获取运行时情境
    #      使用write_in_file方法将感知到的结果写入文件之中
    # 返回值：包含updated_result的键值对


    # collect_for_sample：
    # 用途：获取特定配置下的一系列采样结果，并将平均结果记录在explore_conf_dict贝叶斯字典中
    # 方法：在指定参数的配置下，反复执行post_get_write获取sample_bound个不重复的结果，并计算平均用时avg_delay
    #      之后将当前配置作为键、avg_delay作为值，以键值对的形式将该配置的采样结果保存在字典中，以供贝叶斯优化时的计算   
    # 返回值：当前conf,flow_mapping,resource_limit所对应的sample_bound个采样结果的平均时延
 

    # just_record：
    # 用途：不进行任何配置指定，单纯从当前正在执行的系统中进行record_num次采样。其作用和sample_and_rescord（采样并记录）相对。
    # 方法：初始化一个csv文件，进行record_num次调用get_write并记录当前运行时情境和执行结果，期间完全不控制系统的调度策略
    # 返回值：csv的文件名

    # sample_and_record：
    # 用途：遍历conf_list、ip、cpu和mem所有配置，对每一种配置进行采样并记录结果。和just_record相对。
    # 方法：初始化一个csv文件，然后生成配置遍历的全排列，对每一种配置都调用collect_for_sample进行采样和记录
    # 返回值：csv的文件名

    # objective：
    # 用途：作为贝叶斯优化采样时需要优化的目标函数。
    # 返回值：csv的文件名

    # 以贝叶斯采样的方式获取离线知识库
    # sample_and_record_bayes
    # 用途：在特定的各类.conf_names范围下（区间范围下）进行采样，一共采样,n_trials个点，每一种配置采样sample_bound次
    # 方法：调用一层贝叶斯优化
    # 返回值：在该区间上采样得到的文件名，以及表示本次采样最终性能的study.best_value
   

    # sample_and_record_bayes_for_section
    # 用途：以贝叶斯采样的方式获取离线知识库
    # 方法：相比前一种方法，这种方法首先要对分区的左下角进行若干次采样，然后才进行贝叶斯优化采样
    #       使用该函数之前，必须确保conf_and_serv_info里的内容已经完成调整
    
    
    
    
    
    # sample_and_record_bayes_for_section_test
    # 用途：根据用户指定的section_id，从self.refer_kb_name知识库中找到相应的区间，然后人为计算其上各个配置的时延标准差，以及精度标准差
    
    
    # sample_and_record_bayes_for_section_test
    # 用途：当开始对section_id进行采样的时候，从self.refer_kb_name指定的知识库中提取相应csv文件，并计算其时延和精度的标准差
    # 方法：从 refer_kb_name知识库中读取section_info，获取特定分区对应的csv文件路径
    # def sample_and_record_bayes_for_section(self,sample_bound,n_trials,bin_nums,delay_range,accuracy_range):
    def sample_and_record_bayes_for_section_test(self,conf_sections,section_id,sample_bound,n_trials, bin_nums, delay_range, accuracy_range):

        print('当前采样区间是',section_id)
        # 前期准备1：根据conf_sections和section_id重置conf_and_serv_info的内容
        # section_id形如"reso=0-fps=0-encoder=0",filename是以csv结尾的文件名
        for part in section_id.split('-'):
            # 使用 split('=') 分割键值对  
            conf_name, sec_choice = part.split('=') #value形如'0'
            conf_and_serv_info[conf_name]=conf_sections[conf_name][sec_choice]
            #print(conf_name,conf_and_serv_info[conf_name])
        
        # 前期准备2：清空已有的字典，重新设置好字典的初始值，用于后续计算整个区间的时延和精度标准差
        self.clear_explore_distribution(delay_range=delay_range,accuracy_range=accuracy_range)
        self.bin_nums=bin_nums
        self.explore_conf_dict["delay"]["min"]=delay_range["min"]
        self.explore_conf_dict["delay"]["max"]=delay_range["max"]
        self.explore_conf_dict["accuracy"]["min"]=accuracy_range["min"]
        self.explore_conf_dict["accuracy"]["max"]=accuracy_range["max"]

        # 前期准备3：获取左下角的一系列配置
        bottom_left_conf={}
        for conf_name in self.conf_names:
            bottom_left_conf[conf_name]=conf_and_serv_info[conf_name][0]  #取最小值
        #print('左下角配置:',bottom_left_conf)
        bottom_left_plans=self.get_bottom_left_plans(conf=bottom_left_conf)


        #(1)获取refer_kb_name知识库的section_info_refer，以及自身知识库的section_info，二者作用不同。
        # refer_kb_name
        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name
        
        section_info_refer={}
        with open(self.refer_kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
            section_info_refer=json.load(f) 
        f.close()
        
        section_info={}
        with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
            section_info=json.load(f) 
        f.close()


        #(2)从refer_kb_name的secion_info_refer中获取section_id对应的csv文件,读取其中的df内容
        filenames = section_info_refer[section_id]['filenames']
        filename = filenames[0]
        filepath = self.refer_kb_name + '/' + dag_name + '/' + 'record_data' + '/' +filename
        df = pd.read_csv(filepath) #读取数据内容
        # df形如：
        # n_loop,frame_id,all_delay,obj_n,bandwidth,reso,fps,encoder,face_detection_role,face_detection_ip,face_detection_proc_delay,face_detection_trans_ip,face_detection_trans_delay,face_detection_cpu_util_limit,face_detection_cpu_util_use,face_detection_mem_util_limit,face_detection_mem_util_use,gender_classification_role,gender_classification_ip,gender_classification_proc_delay,gender_classification_trans_ip,gender_classification_trans_delay,gender_classification_cpu_util_limit,gender_classification_cpu_util_use,gender_classification_mem_util_limit,gender_classification_mem_util_use
        # 1,25.0,0.005130281815162072,1,68207.89080843213,360p,1,JPEG,cloud,114.212.81.11,0.004311607434199407,114.212.81.11,0.002276484782879169,1.0,0.04270833333333333,1.0,9.112141805367775e-05,cloud,114.212.81.11,0.0008186743809626653,114.212.81.11,0.002720484366783729,1.0,0.6119791666666666,1.0,0.00024139821168189316
        
        #(3)遍历df中的每一行，将可用的配置记录到字典里
        #(3.1)准备工作，求出dag_conf，即需要考虑的配置项组合
        dag_conf = list(self.conf_names)
        for serv_name in self.serv_names:
            dag_conf.append(serv_name+'_ip')
            dag_conf.append(serv_name+'_cpu_util_limit')
            dag_conf.append(serv_name+'_mem_util_limit')
        #dag_conf形如['reso', 'fps', 'encoder', 'face_detection_ip', 'face_detection_cpu_util_limit', 'face_detection_mem_util_limit','gender_classification_ip', 'gender_classification_cpu_util_limit', 'gender_classification_mem_util_limit',]
        
        #(3.2)获取df的最低索引和最高索引, 开始遍历df中的每一行，获取这一行对应的conf等配置
        min_index=df.index.min()
        max_index=df.index.max()
        in_sec_num=0
        out_sec_num=0
        for index in range(min_index,max_index+1):

            #(3.2.1)计算这一行对应的配置取值
            values_list=df.loc[index,dag_conf].tolist()
            #dag_conf形如['reso', 'fps', 'encoder', 'face_detection_ip', 'face_detection_cpu_util_limit', 'face_detection_mem_util_limit','gender_classification_ip', 'gender_classification_cpu_util_limit', 'gender_classification_mem_util_limit',]
            #values_list形如['990p', 24, 'JPEG', '114.212.81.11', 1.0, 1.0, '114.212.81.11', 1.0, 1.0]
            
            #(3.2.2)#判断当前配置在不在分区中
            if_in_sec=True 
            for conf_name in self.conf_names:
                if values_list[dag_conf.index(conf_name)] not in conf_and_serv_info[conf_name]:
                    if_in_sec=False
                    out_sec_num+=1
                    #print('出现错误配置',values_list)
                    #print('范围是', conf_and_serv_info[conf_name])
                    break
            
            #(3.2.3)#配置在分区内的时候，将这个配置加入到字典里。为此，依次求出conf flow_mapping resource_limit
            if if_in_sec: 
                in_sec_num+=1
                conf={}
                for conf_name in self.conf_names:
                    conf[conf_name]=values_list[dag_conf.index(conf_name)]
                    if conf_name == 'fps':
                        conf[conf_name]=int(values_list[dag_conf.index(conf_name)])
                flow_mapping={}
                resource_limit={}
                for serv_name in self.serv_names:

                    flow_mapping[serv_name]={}
                    serv_ip=values_list[dag_conf.index(serv_name+'_ip')]
                    flow_mapping[serv_name]=model_op[serv_ip]

                    resource_limit[serv_name]={}
                    serv_cpu=values_list[dag_conf.index(serv_name+'_cpu_util_limit')]
                    serv_mem=values_list[dag_conf.index(serv_name+'_mem_util_limit')]
                    resource_limit[serv_name]={}
                    resource_limit[serv_name]['cpu_util_limit']=float(serv_cpu)
                    resource_limit[serv_name]['mem_util_limit']=float(serv_mem)
                
                #该配置没有记录在字典中的时候，才计算时延和精度，并录入字典
                conf_str=json.dumps(conf)
                conf_str=json.dumps(conf)+json.dumps(flow_mapping)
                conf_str=json.dumps(conf)+json.dumps(flow_mapping)+json.dumps(resource_limit) 
                if conf_str not in self.explore_conf_dict:
                    condition_all=True  #用于检索字典所需的条件
                    for i in range(0,len(values_list)):
                        condition=df[dag_conf[i]]==values_list[i]    
                        condition_all=condition_all&condition
                
                    conf_df=df[condition_all]
                    if(len(conf_df)>0): #如果满足条件的内容不为空，可以开始用其中的数值来初始化字典
                        avg_delay=0
                        for serv_name in self.serv_names:
                            avg_delay += conf_df[serv_name+'_proc_delay'].mean()
                        task_accuracy = 1.0
                        acc_pre = AccuracyPrediction()
                        obj_size = None
                        obj_speed = None
                        for serv_name in serv_names:
                            if common.service_info_dict[serv_name]["can_seek_accuracy"]:
                                task_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                                    'fps':conf['fps'],
                                    'reso':conf['reso']
                                }, obj_size=obj_size, obj_speed=obj_speed)
                        # 算出时延和精度后，选择性录入字典
                        if avg_delay >= self.explore_conf_dict['delay']['min'] and avg_delay<=self.explore_conf_dict['delay']['max']:
                            self.explore_conf_dict['delay'][conf_str] = avg_delay  #将所有结果保存在字典里
                        if task_accuracy >= self.explore_conf_dict['accuracy']['min'] and task_accuracy <=self.explore_conf_dict['accuracy']['min']:
                            self.explore_conf_dict['accuracy'][conf_str] = task_accuracy  #将所有结果保存在字典里
        
        
        #print('总记录数为',max_index-min_index+1,' 在区间内的有',in_sec_num, '不在区间内的有',out_sec_num)


        #(4)将所有配置都录入字典中之后，字典里存储着该区间下所有采样点对应的时延和精度，从而可求取标准差
        delay_std,accuracy_std=self.exmaine_explore_distribution()
                    
        filename='not exist'
        filepath='not exist'

        #(5)在kb_name的section_info里加入当前区间的信息，然后重新写入文件之中。
        if section_id not in section_info['section_ids']:
            section_info['section_ids'].append(section_id)
        
        if section_id not in section_info.keys():
            section_info[section_id]={}
            section_info[section_id]['filenames']=[]
            section_info[section_id]['bottom_left_plans']=bottom_left_plans
        
        if filename not in section_info[section_id]['filenames']:
            section_info[section_id]['filenames'].append(filename)
        
        with open(self.kb_name+'/'+dag_name+'/'+'section_info'+".json", 'w') as f:  
            json.dump(section_info, f,indent=4) 
        f.close()
        
        #print('已在section_info中更新该分区的filenames信息和bottom_left_plans信息')


        return filename,filepath,bottom_left_plans, delay_std,accuracy_std




    # 基于贝叶斯的区间采样函数
    # sample_for_kb_sections_bayes
    # 用途：为一种区间划分方式，进行一系列文件初始化，并调用贝叶斯优化函数完成采样（不包括建库）
    # 方法：建立各类文件实现初始化
    #      为了防止中断，对每一个分区进行采样的时候，都使用if_continue来判断当前是否需要从data_recovery里保存的数据恢复
    # def sample_for_kb_sections_bayes(self,conf_sections,sec_num,sample_bound,n_trials,bin_nums,delay_range,accuracy_range,if_continue):
    def sample_for_kb_sections_bayes(self,conf_sections,sec_num,sample_bound,n_trials,bin_nums,delay_range,accuracy_range,if_continue):
        '''
        "conf_sections":{
            "reso":{
                "0":["360p","480p","540p"],
                "1":["630p","720p","810p"],
                "2":["900p","990p","1080p"],
            },
            "fps":{
                "0":[1,2,3,4,5],
                "1":[6,7,8,9,10],
                "2":[11,12,13,14,15],
                "3":[16,17,18,19,20],
                "4":[21,22,23,24,25],
                "5":[26,27,28,29,30],
            },
            "encoder":{
                "0":["JPEG"]
            }
        }
        '''
        print('基于嵌套贝叶斯优化来进行区间采样,要采样的总区间数为',sec_num)
        # (1)首先要生成一个初始化文件描述当前分区信息。如果是中断后恢复，那就直接读取旧的section_info信息
        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name

        section_info={}
        if not if_continue: #如果不是恢复而是从头开始：
            section_info['des']=datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
            section_info['serv_names']=self.serv_names
            section_info['conf_names']=self.conf_names
            section_info['conf_sections']=conf_sections
            section_info['section_ids']=list([])
            with open(self.kb_name+'/'+dag_name+'/'+'section_info'+".json", 'w') as f:  
                json.dump(section_info, f,indent=4) 
            f.close()
        else:
            #否则直接读取已有的section_info信息
            with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
                #print('打开知识库section_info:',self.kb_name + '/' + dag_name + '/' + 'section_info.json')
                section_info=json.load(f) 
            f.close()
        
        # （2）进行采样前首先保存好原始的各个配置的取值范围，以供未来恢复    
        original_conf_range={}
        for conf_name in self.conf_names:
            original_conf_range[conf_name]=conf_and_serv_info[conf_name]
        

        # （3）根据if_continue判断当前是应该恢复上一次中断，还是进行重新采样
        #      如果为False，就表示需要重新开启，此时才需要对左上角和右下角进行采样。
        # 求最小分区id和最大分区id
        section_id_min=''
        section_id_max=''
        for i in range(0,len(self.conf_names)):
            conf_name=self.conf_names[i]   
            max_idx=str(len(conf_sections[conf_name])-1) 
            if i>0:
                section_id_min+='-'
                section_id_max+='-'
            section_id_min=section_id_min + conf_name+'='+'0'
            section_id_max=section_id_max + conf_name+'='+max_idx
        
        sec_sum=0 #已选择的区间数
        # 如果不是中断恢复，则求左下角和右上角区间
        if not if_continue:
            filename,filepath,bottom_left_plans,delay_std,accuracy_std=self.sample_and_record_bayes_for_section_test(
                                                                                conf_sections=conf_sections,
                                                                                section_id=section_id_min,
                                                                                sample_bound=sample_bound,
                                                                                n_trials=n_trials,
                                                                                bin_nums=bin_nums,
                                                                                delay_range=delay_range,
                                                                                accuracy_range=accuracy_range)
            sec_sum+=1
            if section_id_max!=section_id_min:
                filename,filepath,bottom_left_plans,delay_std,accuracy_std=self.sample_and_record_bayes_for_section_test(
                                                                                    conf_sections=conf_sections,
                                                                                    section_id=section_id_max,
                                                                                    sample_bound=sample_bound,
                                                                                    n_trials=n_trials,
                                                                                    bin_nums=bin_nums,
                                                                                    delay_range=delay_range,
                                                                                    accuracy_range=accuracy_range)
                sec_sum+=1
            else:
                print('右上角等于左下角，不必再采样')
                

            

        #(5)开始从分区信息中提取分区，从而进行嵌套贝叶斯优化。使用ask_and_tell方法。
        conf_sec_select={}
        for conf_name in self.conf_names:
             conf_sec_select[conf_name]=list(conf_sections[conf_name].keys())
        print(conf_sec_select)
        
        '''
        conf_sec_select形如:
        {   
            'reso': ['0', '1', '2'], 
            'fps': ['0', '1', '2', '3', '4', '5'], 
            'encoder': ['0']
        }
        '''
        # 根据if_continue来决定study的建立形式，是新建，还是加载已有的 
        
        study = None
        study_num=0

        if if_continue:
            with open(self.kb_name +  '/' + dag_name +'/data_recovery/sec_sum.json', 'r') as f:  
                data = json.load(f)  
            f.close()
            sec_sum = data["sec_sum"]
            study_num = data["study_num"]
            print('恢复中断数据，上一次执行到',sec_sum,"上一次study_num是",study_num)
            study = optuna.load_study(study_name="study"+str(study_num),storage="sqlite:///" + self.kb_name + '/' + dag_name +"/data_recovery/kb_study.db")
        else:
            # 重新开启一个study，初始为1。
            study_num+=1
            print("创建第",study_num,"个study")
            study = optuna.create_study(directions=['minimize' for _ in range(2)], sampler=optuna.samplers.NSGAIISampler(),
                                        study_name="study"+str(study_num),storage="sqlite:///" + self.kb_name + '/' + dag_name +"/data_recovery/kb_study.db") 
 

        repeat_num=0 #如果连续多次采样了已经采样的点，说明陷入了死循环。
        while True:
            if sec_sum==sec_num or section_id_min == section_id_max: #只有在总数没有达标，且左下角区间和右上角区间不相等的时候，才进行依次采样
                print('当前sec_sum是',sec_sum)
                break
            else:
                print('当前sec_sum是',sec_sum)
                trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.
                section_id=''
                for i in range(0,len(self.conf_names)):
                    conf_name=self.conf_names[i]                    
                    sec_choice=trial.suggest_categorical(conf_name+'_sec',conf_sec_select[conf_name])
                    # 根据分区选择结果重置取值范围 
                    conf_and_serv_info[conf_name]=conf_sections[conf_name][sec_choice]
                    # 计算所得section_id
                    if i>0:
                        section_id+='-'
                    section_id=section_id+conf_name+'='+sec_choice

                print('选择新分区id是',section_id)

                # 读取最新的section_info，获取已经采样的区间
                section_info={}
                with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
                    section_info=json.load(f) 
                f.close()

                if section_id_max==section_id or section_id_min==section_id:
                    print('与左下角或右上角重合，不必考虑')
                elif section_id in section_info['section_ids']:
                    print("该分区已经采样过，忽略")
                    repeat_num+=1
                    # 如果重复次数实在过于多，就说明已经陷入了局部最优解，此时需要重置study了。
                    if repeat_num > max(10, len(section_info['section_ids'])):
                        study_num+=1 #开始建立一个新的study_num
                        print("创建第",study_num,"个study")
                        with open(self.kb_name + '/' + dag_name +'/data_recovery/sec_sum.json', 'w') as f:  
                            data={"sec_sum":sec_sum,"study_num":study_num}
                            json.dump(data, f)
                        f.close()
                        study = optuna.create_study(directions=['minimize' for _ in range(2)], sampler=optuna.samplers.NSGAIISampler(),
                                        study_name="study"+str(study_num),storage="sqlite:///" + self.kb_name + '/' + dag_name +"/data_recovery/kb_study.db") 
                        repeat_num = 0 
                        
                #跳过以上条件后，才对section_id进行采样
                else:
                    filename,filepath,bottom_left_plans,delay_std,accuracy_std=self.sample_and_record_bayes_for_section_test(
                                                                                    conf_sections=conf_sections,
                                                                                    section_id=section_id,
                                                                                    sample_bound=sample_bound,
                                                                                    n_trials=n_trials,
                                                                                    bin_nums=bin_nums,
                                                                                    delay_range=delay_range,
                                                                                    accuracy_range=accuracy_range)

                    # 用返回值指导下一次采样
                    ans=[delay_std,accuracy_std]
                    study.tell(trial,ans)  # tell the pair of trial and objective value

                    # 完成tell之后才记录当前的最新sec_sum
                    sec_sum+=1
                    with open(self.kb_name + '/' + dag_name +'/data_recovery/sec_sum.json', 'w') as f:  
                        data={"sec_sum":sec_sum,"study_num":study_num}
                        json.dump(data, f)
                    f.close()

        #最后恢复原始的各个配置的取值
        for conf_name in self.conf_names:
            conf_and_serv_info[conf_name]=original_conf_range[conf_name]

        #每次区间采样时的相关内容已经在调用sample_and_record_bayes_for_section_test的时候写入section_info了
    


    # get_clusters
    # 用途:将一系列分区编号聚类为指定的数量
    # 方法：使用k_means
    # 返回值：clusters_info,形如：
    '''
    {   
        0: {
            'center': array([1280.,   30.,    0.]), 
            'section_ids': ['reso=1280-fps=30-encoder=0']
            }, 

        1: {
            'center': array([400.,  45.,   0.]), 
            'section_ids': ['reso=400-fps=45-encoder=0']
            }, 
        
        2: {
            'center': array([7.20e+02, 5.25e+01, 5.00e-01]), 
            'section_ids': ['reso=640-fps=60-encoder=1', 'reso=800-fps=45-encoder=0']
            }
    }
    '''
    def get_clusters(self,section_ids,n_clusters):
        
        # 从已采样的分区中提取特征向量
        features=[]
        for section_id in section_ids:
            matches = re.findall(r'(\w+)=(\d+)', section_id)  
            features.append(np.array([int(value) for _, value in matches]))
        features=np.array(features)
        

        # 使用KMeans将其聚类为n_clusters类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features) 

        # 然后输出每一个类的聚类中心，以及各自内部含有的区间

        clusters_info={}
        # 标注每一个聚类的名称和聚类中心
        for i, center in enumerate(kmeans.cluster_centers_):  
            clusters_info[i]={}
            clusters_info[i]['center']=center #center形如[1280.   30.    0.]
            clusters_info[i]['section_ids']=[]
        
        # 加入每一个聚类中所含有的区间
        labels = kmeans.predict(features)
        # 遍历数据和标签，将样本添加到对应的聚类中  
        for index, label in enumerate(labels):  
            clusters_info[label]['section_ids'].append(section_ids[index])  
        
        return clusters_info

    # get_integer_points_on_line
    # 用途:输入起点start和终点end，也就是两个聚类的中心，将其连线线段上的各个整数坐标点输出。两个输入量全部都是特征向量
    # 方法：十分复杂的数据处理
    # 返回值：integer_points 里面每一个元素都是一个形如[3, 7, 0]的特征向量，可以对应到self.conf_names里的各个坐标点

    def get_features_on_line(self,start, end):  
        # 转换输入为NumPy数组，并四舍五入到最近的整数  
        start = np.round(start).astype(int)  
        end = np.round(end).astype(int)  
    
        # 计算两个点之间的差值  
        diff = end - start  
    
        # 创建一个从0到1（包含1）的步长数组，步长数量取决于线段上可能的整数点数量  
        # 注意：这里我们简单地假设最大步数在每个维度上是diff的最大值加1，  
        # 但这可能会产生超出预期的点，特别是当线段非常接近但不完全沿坐标轴时  
        steps = np.linspace(0, 1, np.max(np.abs(diff)) + 2, endpoint=True)  
    
        # 初始化一个列表来存储整数点坐标  
        integer_points = []  
    
        # 对于每个步骤，计算线段上的点并四舍五入到整数  
        for t in steps:  
            point = start + t * diff  
            integer_point = np.round(point).astype(int)  
            integer_points.append(integer_point.tolist())  
    
        # 去除重复的点和起始/终止点（如果它们被重复计算了）  
        integer_points = list(set(map(tuple, integer_points)))  
        integer_points.sort(key=lambda x: np.linalg.norm(np.array(x) - start, ord=1))  
        integer_points = [list(p) for p in integer_points if p != start.tolist() or p == end.tolist()]  
    
        return integer_points

    # get_distance
    # 用途:获取两个向量之间的整数距离
    def get_distance(self, vec1, vec2):  

        """  
        计算两个整数向量之间的欧几里得距离  
        :param vec1: 第一个整数向量，如 [1, 2, 3]  
        :param vec2: 第二个整数向量，如 [4, 5, 6]  
        :return: 两个向量之间的欧几里得距离  
        """ 
    
        if len(vec1) != len(vec2):  
            raise ValueError("两个向量的长度必须相等")  
    
        distance_squared = sum([(a - b) ** 2 for a, b in zip(vec1, vec2)])  
        return math.sqrt(distance_squared)  
    
    # get_unsampled_section_ids
    # 用途:计算出当前知识库中所有尚未采样的区间的编号
    def get_unsampled_section_ids(self,conf_sec_select,sampled_section_ids):
    
         # 先用conf_sec_select计算所有的区间编号
        '''
        {   
            'reso': ['0', '1', '2'], 
            'fps': ['0', '1', '2', '3', '4', '5'], 
            'encoder': ['0']
        }
        '''
        conf_idx_range=[]
        for conf_name in self.conf_names:
            conf_idx_range.append(conf_sec_select[conf_name])
        conf_idx_combines = itertools.product(*conf_idx_range)
        all_section_ids = []
        for conf_idx_combine in conf_idx_combines:
            # 一个conf_idx_combine代表一个可能的坐标点形如('0', '0', '0')
            section_id=''
            for i in range(0,len(self.conf_names)):
                conf_name=self.conf_names[i]                   
                # 计算所得section_id
                if i>0:
                    section_id+='-'
                section_id=section_id+conf_name+'='+conf_idx_combine[i]
            all_section_ids.append(section_id)
        
        #all_section_ids里有所有的区间编号,sampled_section_ids里是已经采样的，两个都是字符串构成的列表，要从all_section_ids里删除sampled_section_ids里已经重复的内容
        sampled_set = set(sampled_section_ids)  
        unsampled_section_ids =  [section_id for section_id in all_section_ids if section_id not in sampled_set]
        return unsampled_section_ids


    # score_cluster
    # 用途:对当前知识库上各个聚类进行打分，打分结果录入到score_cluster当中并返回
    #      score_n_trials是指对于每一个聚类要评分多久
    '''
    {   
        0: {
            'center': array([1280.,   30.,    0.]), 
            'section_ids': ['reso=1280-fps=30-encoder=0']
            }, 

        1: {
            'center': array([400.,  45.,   0.]), 
            'section_ids': ['reso=400-fps=45-encoder=0']
            }, 
        
        2: {
            'center': array([7.20e+02, 5.25e+01, 5.00e-01]), 
            'section_ids': ['reso=640-fps=60-encoder=1', 'reso=800-fps=45-encoder=0']
            }
    }
    # 写入之前需要将center部分其转化为列表的形式
    '''
    def score_cluster(self, clusters_info, sec_sum, score_n_trials, optimze_goal):

        # 首先建立好评估器
        from kb_sec_score_clus_test import KnowledgeBaseScorer
        work_condition_range={
            "obj_n": [1],
            "obj_size": [50000],
            "obj_speed":[0]
        }
        user_constraint_range={  
            "delay": [0.4,0.7,1.0],
            "accuracy": [0.3,0.6,0.9]
        }
        rsc_constraint_range=[0.4,0.7,1.0]
        bandwidth_range=[20000,40000,60000,80000,100000]
        kb_scorer = KnowledgeBaseScorer(conf_names=self.conf_names,
                                    serv_names=self.serv_names,
                                    service_info_list=self.service_info_list,
                                    work_condition_range=work_condition_range,
                                    user_constraint_range=user_constraint_range,
                                    rsc_constraint_range=rsc_constraint_range,
                                    bandwidth_range=bandwidth_range,
                                    )



        runtime_filename='runtime2-135cons'
        #'''
        for key in clusters_info.keys():
            cluster_info = clusters_info[key]
            clus_id = key
            cluster_section_ids = cluster_info['section_ids']
            clus_score=kb_scorer.score_kb_by_runtimes_for_cluster(kb_name=self.kb_name,refer_kb_name=self.refer_kb_name,
                                                                runtime_filename=runtime_filename,n_trials=score_n_trials,
                                                                optimze_goal=optimze_goal,search_num=1,
                                                                section_ids_sub=cluster_section_ids,sec_sum=sec_sum,clus_id=clus_id,
                                                                )
            
            clusters_info[key]['score']=clus_score
            center_array = clusters_info[key]['center']
            clusters_info[key]['center'] = center_array.tolist()
            print('聚类',key,'中心是',clusters_info[key]['center'], '大小是',len(clusters_info[key]["section_ids"]),'得分是',clusters_info[key]['score'], )
        #'''
        # 以下是为了方便测试
        '''
        {   
            0: {
                'center': array([1280.,   30.,    0.]), 
                'section_ids': ['reso=1280-fps=30-encoder=0']
                }, 

            1: {
                'center': array([400.,  45.,   0.]), 
                'section_ids': ['reso=400-fps=45-encoder=0']
                }, 
            
            2: {
                'center': array([7.20e+02, 5.25e+01, 5.00e-01]), 
                'section_ids': ['reso=640-fps=60-encoder=1', 'reso=800-fps=45-encoder=0']
                }
        }
        '''
        '''
        for key in clusters_info.keys():        
            time.sleep(1)    
            clusters_info[key]['score']=random.random()
            center_array = clusters_info[key]['center']
            clusters_info[key]['center'] = center_array.tolist()
            print('聚类',key,'中心是',clusters_info[key]['center'], '大小是',len(clusters_info[key]["section_ids"]),'得分是',clusters_info[key]['score'], )
        '''
        
        # 最后得到一个全新的clusters_info,里面标注了每一个聚类的分数
        return clusters_info


    # get_next_section_id
    # 用途：根据聚类和打分的结果选出下一个section_id。
    # 方法: 第四步：假设建立了1、2、3、4个聚类，每一个聚类的性能依次降低，那么开始建立1-2，1-3，1-4；2-3，2-4；3-4这些连线，
    #       建立列表[1-2，1-3，1-4，2-3，2-4，3-4]。
	#       遍历列表中的每一条连线：如果连接线的中心为空，选择中心；否则，向分数最高的一端靠拢，寻找第一个没有采样的区间；
    #       再否则，向靠近分数较低的那一段移动，寻找第一个没有采样的区间。持续遍历直到找到一个合适的点。
    #       如果整个列表中没有一条连线上有区间可以采，那就遍历尚未采样的区间，寻找和区间1最近的那一个区间采样。
    '''
    conf_sec_select形如:
    {   
        'reso': ['0', '1', '2'], 
        'fps': ['0', '1', '2', '3', '4', '5'], 
        'encoder': ['0']
    }
    '''
    '''
    clusters_info形如:
    {   
        0: {
            'center': array([1280.,   30.,    0.]), 
            'section_ids': ['reso=1280-fps=30-encoder=0'],
            'score':0.7
            }, 

        1: {
            'center': array([400.,  45.,   0.]), 
            'section_ids': ['reso=400-fps=45-encoder=0'],
            'score':0.65
            }, 
        
        2: {
            'center': array([7.20e+02, 5.25e+01, 5.00e-01]), 
            'section_ids': ['reso=640-fps=60-encoder=1', 'reso=800-fps=45-encoder=0'],
            'score':0.85
            }
    }
    '''

    def get_next_section_id(self, clusters_info ,sampled_section_ids, conf_sec_select):
        print('开始基于当前聚类结果选择下一个区间')

        #(1)首先要建立[1-2，1-3，1-4，2-3，2-4，3-4]
        # 对于clusters_info，首先按照score排序，得到新的key列表，然后就可以得到以上的若干对start end，之后就可以用来依次求解了。
        # 然后再利用combine从conf_sec_select里面强制构建90个所有的区间，删除已经在section_ids里面的，从剩下的直接计算距离并求出最小的。
        # 如下，首先按照score进行从大到小的排序，
        sorted_clus_ids = sorted(clusters_info.keys(), key=lambda k: clusters_info[k]['score'], reverse=True)  
        print('对已完成的聚类按照得分排序',sorted_clus_ids)
        # 然后对sorted_clus_ids形如[1,2,3,4]，得到[1-2，1-3，1-4，2-3，2-4，3-4]，记录在start_end_pairs内，这个列表里每一个元素都形如[1,4]表示一条线段的起始点和终点
        start_end_pairs=[]
        for i in range(0,len(sorted_clus_ids)-1):
            start=sorted_clus_ids[i]
            for j in range(i+1,len(sorted_clus_ids)):
                end=sorted_clus_ids[j]
                start_end_pairs.append([start,end])
        print('根据排序结果,依次确定各个聚类中心对',start_end_pairs)
        # start_end_pairs形如[[1,2]，[1,3]，[1,4]，[2,3]，[2,4]，[3,4]],每一个元素都代表一个线段
        
        if_find_next_section_id = False
        next_section_id=''

        #(2)对于每一个start_end_pair对，连接其线段，并得到线段上的各个整数坐标点;然后检查这个线段上的点是否可以作为下一个坐标
        for start_end_pair in start_end_pairs:
            print('当前处理的聚类中心对是',start_end_pair)

            start = clusters_info[start_end_pair[0]]['center']
            end   = clusters_info[start_end_pair[1]]['center']

            integer_points=self.get_features_on_line(start=start,end=end)
            candidate_section_ids = []
            # 所得integer_points是一系列的坐标点，每一个坐标点都是一个多维向量, 将其依次转化为section_id的形式
            for point in integer_points:
                # 要将point转化为section_id
                section_id=''
                for i in range(0,len(self.conf_names)):
                    conf_name=self.conf_names[i]                   
                    # 计算所得section_id
                    if i>0:
                        section_id+='-'
                    section_id=section_id+conf_name+'='+str(point[i])
                candidate_section_ids.append(section_id)
            
            length = len(candidate_section_ids)
            # candidate_section_ids记录了start 到 end 线段上所有的区间的坐标
            # 检查这条线段上有没有可以作为下一个采样点的区间
            # 1、看中点部分是否已经被采样
            print('尝试查看中点区间的可用性')
            middle_section_id = candidate_section_ids[length // 2]
            if middle_section_id not in sampled_section_ids:
                print('两个聚类中心的中点未采样，可以作为新采样目标')
                if_find_next_section_id = True
                next_section_id = middle_section_id
                break # 如果中点符合，本线段上有采样点，结束

            # 2、从中点往左，寻找第一个没有被采样的点，尽可能靠近中点
            print('尝试从线段中心位置向得分更高的聚类中心移动')
            for i in range(max(0, length//2 - 1), -1, -1):
                section_id = candidate_section_ids[i]
                if section_id not in sampled_section_ids:
                    print('找到第一个可用区间')
                    if_find_next_section_id = True
                    next_section_id = section_id
                    break
            
            if if_find_next_section_id:
                break # 如果本线段上有采样点，结束

            # 2、从中点往右，寻找第一个没有被采样的点，尽可能靠近中点
            print('尝试从线段中心位置向得分更低的聚类中心移动')
            for i in range(min(length-1, length//2 + 1), length):
                section_id = candidate_section_ids[i]
                if section_id not in sampled_section_ids:
                    print('找到第一个可用区间')
                    if_find_next_section_id = True
                    if_find_next_section_id = True
                    next_section_id = section_id
                    break
            
            if if_find_next_section_id:
                break # 如果本线段上有采样点，结束
        

        #(3)如果所有线段坐标上都不存在合适的采样区间，那就遍历全部的section_id可能取值，寻找和最高分聚类中心最近的
        if not if_find_next_section_id:
            print('此前查找全部失败,准备从剩余可用区间中选择最靠近最佳聚类的区间')

            # 1、首先获取目前最优的聚类中心的坐标，形如[0.7, 1.2, 0 ]
            best_clus_center_featues = clusters_info[sorted_clus_ids[0]]['center'] 

            # 2、然后获取当前所有没有采样的区间编号
            unsampled_section_ids = self.get_unsampled_section_ids(conf_sec_select=conf_sec_select,sampled_section_ids=sampled_section_ids)
            print('当前未采样的区间编号是',unsampled_section_ids)

            # 3、从unsampled_section_ids找到和best_clus_center_featues最近的那个id
            min_distance = float('inf')
            for section_id in unsampled_section_ids:
                matches = re.findall(r'(\w+)=(\d+)', section_id)  
                vec = np.array([int(value) for _, value in matches])
                distance = self.get_distance(vec1=vec,vec2=best_clus_center_featues)
                if distance < min_distance: #只要发现更小的距离，就更新next_section_id
                    next_section_id = section_id
        # 经过以上操作就能得到新的section_id    
        
        return next_section_id



    # 基于聚类的区间采样函数
    # sample_for_kb_sections_clus
    # 用途：为一种区间划分方式，进行一系列文件初始化，然后用基于聚类的方式完成采样
    # 方法：建立各类文件实现初始化
    #      为了防止中断，对每一个分区进行采样的时候，都使用if_continue来判断当前是否需要从data_recovery里保存的数据恢复
    # def sample_for_kb_sections_clus(self,conf_sections,sec_num,sample_bound,n_trials,bin_nums,delay_range,accuracy_range,if_continue):
    def sample_for_kb_sections_clus(self,conf_sections,sec_num,sample_bound,n_trials,bin_nums,delay_range,accuracy_range,n_clusters,score_n_trials,optimze_goal,if_continue):
        print('开始进行基于聚类的区间采样,以建立知识库.本次建库采用模拟采样,基于已建立完成的90i90知识库模拟采样结果')
        print('知识库名称是',self.kb_name,'每次聚类聚类出',n_clusters,'个区间')
        '''
        "conf_sections":{
            "reso":{
                "0":["360p","480p","540p"],
                "1":["630p","720p","810p"],
                "2":["900p","990p","1080p"],
            },
            "fps":{
                "0":[1,2,3,4,5],
                "1":[6,7,8,9,10],
                "2":[11,12,13,14,15],
                "3":[16,17,18,19,20],
                "4":[21,22,23,24,25],
                "5":[26,27,28,29,30],
            },
            "encoder":{
                "0":["JPEG"]
            }
        }
        '''
        # (1)首先要生成一个初始化文件描述当前分区信息。如果是中断后恢复，那就直接读取旧的section_info信息
        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name

        section_info={}
        if not if_continue: #如果不是恢复而是从头开始,写入初始化的section_info和clus_info
            section_info['des']=datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
            section_info['serv_names']=self.serv_names
            section_info['conf_names']=self.conf_names
            section_info['conf_sections']=conf_sections
            section_info['section_ids']=list([])
            with open(self.kb_name+'/'+dag_name+'/'+'section_info'+".json", 'w') as f:  
                json.dump(section_info, f,indent=4) 
            f.close()
            clus_info={}
            with open(self.kb_name + '/' + dag_name +'/clus_info.json', 'w') as f:  
                json.dump(clus_info, f)
            f.close()
        else:
            #否则直接读取已有的section_info信息
            with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
                #print('打开知识库section_info:',self.kb_name + '/' + dag_name + '/' + 'section_info.json')
                section_info=json.load(f) 
            f.close()
        
        # （2）进行采样前首先保存好原始的各个配置的取值范围，以供未来恢复    
        original_conf_range={}
        for conf_name in self.conf_names:
            original_conf_range[conf_name]=conf_and_serv_info[conf_name]
        

        # （3）初始阶段，获取一系列初始区间坐标，每一个坐标的取值要么是最小要么是最大
        
        sec_sum=0 #初始已经采样的区间数

        section_ids_begin=[]
        idx_min_max=[]
        # 根据if_continue判断当前是应该恢复上一次中断，还是进行重新采样
        #      如果为False，就表示需要重新开启，此时才需要对左上角和右下角进行采样。
        # 求最小分区id和最大分区id
        for i in range(0,len(self.conf_names)):
            conf_name=self.conf_names[i] 
            min_idx=str(0)  
            max_idx=str(len(conf_sections[conf_name])-1)
            if max_idx == min_idx:
                idx_min_max.append([min_idx])
            else:
                idx_min_max.append([min_idx,max_idx])
        
        # idx_min_max形如[['0','8'],['0','9'],['0']] 
        idx_combines = itertools.product(*idx_min_max)
        for idx_combine in idx_combines:
            #idx_combine形如('0', '0', '0')
            section_id_temp=''
            for i in range(0,len(self.conf_names)):
                conf_name = self.conf_names[i]
                if i>0:
                    section_id_temp+='-'
                section_id_temp=section_id_temp + conf_name+'='+idx_combine[i]
            #section_idx_temp形如reso=0-fps=0-encoder=0
            section_ids_begin.append(section_id_temp)
        
        print('开始第一轮采样,选取在配置空间中处于边缘极点的一系列区间如下')
        print(section_ids_begin)

        
        # 如果不是中断恢复，则对初始采样点进行依次采样
        if not if_continue:
            print('当前并非是从中断中恢复,故可以依次采样初始选择的一批区间')
            for section_id in section_ids_begin:
                filename,filepath,bottom_left_plans,delay_std,accuracy_std=self.sample_and_record_bayes_for_section_test(
                                                                                    conf_sections=conf_sections,
                                                                                    section_id=section_id,
                                                                                    sample_bound=sample_bound,
                                                                                    n_trials=n_trials,
                                                                                    bin_nums=bin_nums,
                                                                                    delay_range=delay_range,
                                                                                    accuracy_range=accuracy_range)
                #最后恢复原始的各个配置的取值
                for conf_name in self.conf_names:
                    conf_and_serv_info[conf_name]=original_conf_range[conf_name]
                sec_sum+=1
        
        # 完成以上初始采样点后，开始进入循环:聚类，然后根据聚类结果选择下一个区间，直到满足要求
          
        #(5)完成第一阶段的采样后，开始进入循环过程

        conf_sec_select={}
        for conf_name in self.conf_names:
             conf_sec_select[conf_name]=list(conf_sections[conf_name].keys())
        #print(conf_sec_select)
        '''
        conf_sec_select形如:
        {   
            'reso': ['0', '1', '2'], 
            'fps': ['0', '1', '2', '3', '4', '5'], 
            'encoder': ['0']
        }
        '''
        # 根据if_continue来决定study的建立形式，是新建，还是加载已有的 
        if if_continue:
            with open(self.kb_name +  '/' + dag_name +'/data_recovery/sec_sum.json', 'r') as f:  
                data = json.load(f)  
            f.close()
            sec_sum = data["sec_sum"]
            print('恢复中断数据，上一次执行到',sec_sum)
        else:
            print('完成对若干初始区间的采样后,开始通过基于聚类的方式选择下一个区间')


        while True:
            if sec_sum==sec_num or len(section_ids_begin)==1: #只有在总数没有达标,且整个知识库不是一个区间的时候，才进行依次采样
                print('当前sec_sum是',sec_sum)
                break
            else:
                print('当前sec_sum是',sec_sum)
                # 开始选择新分区。为此，进行以下步骤：
                # 第一步，读取最新的section_info，获取已经采样的区间
                section_info={}
                with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
                    section_info=json.load(f) 
                f.close()

                #对已有区间进行分区划分，并进行聚类，聚类后对每一个聚类打分，然后记录
                section_ids = section_info['section_ids']
                # 可能需要根据实际情况修改n_clusters
                clusters_info = {}
                if n_clusters > len(section_ids):
                    clusters_info = self.get_clusters(section_ids = section_ids,n_clusters = len(section_ids))
                else:
                    clusters_info = self.get_clusters(section_ids = section_ids,n_clusters = n_clusters)
                clusters_info = self.score_cluster(clusters_info = clusters_info, sec_sum = sec_sum, score_n_trials=score_n_trials, optimze_goal=optimze_goal)
                clus_info={}
                with open(self.kb_name + '/' + dag_name +'/clus_info.json', 'r') as f:  
                    clus_info=json.load(f)
                f.close()
                clus_info[str(sec_sum)]=clusters_info
                with open(self.kb_name + '/' + dag_name +'/clus_info.json', 'w') as f:  
                    json.dump(clus_info, f, indent = 4)
                f.close()
                print('已经将最新一批区间聚类和打分结果存储到clus_info文件中，注意查看')
                
                # 根据聚类和打分结果选择新的采样点
                # 为此需要: 打分后的聚类信息，目前已经采样的区间信息，区间可选范围
                section_id = self.get_next_section_id(clusters_info=clusters_info,sampled_section_ids=section_ids,conf_sec_select=conf_sec_select)
                print('选择新分区id是',section_id)
                if section_id in section_info['section_ids']:
                    print("该分区已经采样过，忽略")
                #跳过以上条件后，才对section_id进行采样
                else:
                    filename,filepath,bottom_left_plans,delay_std,accuracy_std=self.sample_and_record_bayes_for_section_test(
                                                                                    conf_sections=conf_sections,
                                                                                    section_id=section_id,
                                                                                    sample_bound=sample_bound,
                                                                                    n_trials=n_trials,
                                                                                    bin_nums=bin_nums,
                                                                                    delay_range=delay_range,
                                                                                    accuracy_range=accuracy_range)
                    #最后恢复原始的各个配置的取值
                    for conf_name in self.conf_names:
                        conf_and_serv_info[conf_name]=original_conf_range[conf_name]

                    # 完成采样之后才更新sec_sum内容
                    sec_sum+=1
                    with open(self.kb_name + '/' + dag_name +'/data_recovery/sec_sum.json', 'w') as f:  
                        data={"sec_sum":sec_sum}
                        json.dump(data, f)
                    f.close()
                    print('完成对新分区的采样和文件写入')

        

        #每次区间采样时的相关内容已经在调用sample_and_record_bayes_for_section_test的时候写入section_info了
    
    








    # update_section_from_file
    # 用途：根据一个csv文件记录的内容，为流水线上所有服务更新一个分区知识库
    # 返回值：无，但是为每一个服务都生成了一个区间的CSV文件

    # update_sections_from_files:
    # 用途：利用section_info里的结果来更新分区知识库
    # 方法：从section_info中提取多个分区，以及相对应的文件，多次调用update_section_from_file生成多个文件

    # swap_node_ip_in_kb：
    # 用途：将知识库中指定的ip替换为另一种ip
    # 方法：读取知识库中所有的配置组合，然后将其中的old_node_ip部分换成new_node_ip。这样做是为了增加知识库的泛用性。
    # 返回值：无

    # draw_scatter：
    # 用途：根据参数给定的x和y序列绘制散点图
    # 方法：不赘述
    # 返回值：无

    # draw_scatter：
    # 用途：根据参数给定的data序列和bins绘制直方图
    # 方法：不赘述
    # 返回值：绘制直方图时返回的a,b,c(array, bins, patches)，
    #        其中，array是每个bin内的数据个数，bins是每个bin的左右端点，patches是生成的每个bin的Patch对象。

    # draw_picture：
    # 用途：根据参数给定的x和y序列绘制曲线图
    # 方法：不赘述
    # 返回值：无

    # draw_delay_and_cons：
    # 用途：在相同的x值上绘制两个y值，如果需要绘制约束的话就用它
    # 方法：不赘述
    # 返回值：无

    
    # draw_picture_from_sample：
    # 用途：根据filepath指定的日志一次性绘制大量图片
    # 方法：不赘述
    # 返回值：无



#以上是KnowledgeBaseBuilder类的全部定义，该类可以让使用者在初始化后完成一次完整的知识库创建，并分析采样的结果
#接下来定义一个新的类，作用的基于知识库进行冷启动
          


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
        "name":'gender_classification',
        "value":'gender_classification_proc_delay',
        "conf":["reso","fps","encoder"]
    },
]



# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","gender_classification"]   


#这个query_body用于测试单位的“人进入会议室”，也就是只有一张脸的情况，工况不变，但是会触发调度器变化，因为ifd很小
#'''
query_body = {
        "node_addr": "192.168.1.7:3001",
        "video_id": 103,     
        "pipeline":  ["face_detection", "gender_classification"],#制定任务类型
        "user_constraint": {
            "delay": 1.0, #用户约束暂时设置为0.3
        }
    }  
#'''


if __name__ == "__main__":

    from RuntimePortrait import RuntimePortrait

    myportrait = RuntimePortrait(pipeline=serv_names)
    #从画像里收集服务在边缘端的资源上限
    # 描述每一种服务所需的中资源阈值，它限制了贝叶斯优化的时候采取怎样的内存取值范围
    '''
    rsc_upper_bound={
        'face_detection':{
            'cpu_limit':0.25,
            'mem_limit':0.015,
        },
        'gender_classification':{
            'cpu_limit':0.6,
            'mem_limit':0.008,
        }
        
    }
    help_cold_start返回值
    {
        'cpu': {  // CPU资源
            'cloud': 0.1,  // 云端的最大资源阈值
            'edge': 0.5  // 边端的最大资源阈值
        },
        'mem': {  // 内存资源
            'cloud': 0.1,
            'edge': 0.5
        }
    }
    '''
    rsc_upper_bound = {}
    for serv_name in serv_names:
        serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
        rsc_upper_bound[serv_name] = {}
        rsc_upper_bound[serv_name]['cpu_limit'] = serv_rsc_cons['cpu']['edge']
        rsc_upper_bound[serv_name]['mem_limit'] = serv_rsc_cons['mem']['edge']
    print("画像提供的资源上限")
    print(rsc_upper_bound)
   


    # 贝叶斯优化时的取值范围，在以下范围内使得采样点尽可能平均
    delay_range={
        'min':0.0,
        "max":1.0
    }
    accuracy_range={
        'min':0.0,
        "max":1.0
    }

    # 在多大范围内取方差
    bin_nums = 100

    # 除了最小和最大区间以外，还需要sec_num个区间
    sec_num = 0
    # 每一种配置下进行多少次采样
    sample_bound = 5
    # 为每一个区间进行贝叶斯优化时采样多少次
    n_trials = 200 #20
    #(将上述三个量相乘，就足以得到要采样的总次数，这个次数与建立知识库所需的时延一般成正比)
    
    # 是否进行稀疏采样(贝叶斯优化)
    need_sparse_kb = 1
    # 是否进行严格采样（遍历所有配置）


    task_name = "gender_classify"

    refer_kb_name = 'kb_data_90i90_no_clst-1'  #作为参考，从中提取采样结果，模拟真实采样

    # kb_data_20-1_90i90_no_clst-1表示从kb_data_90i90_no_clst-1中提取了20个区间，20-1中的-1表示这是第一次尝试
    # kb_data_20-1_90i90_2clst-1表示基于90i90的模拟采样，使用基于聚类的方法采样20次，每次聚类为2个
    kb_name = 'kb_data_20-1_90i90_no_clst-5'
    
    

    #未来还是需要类似record_name的存在。
    record_name=kb_name+'/'+'0_'+datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')+task_name+"_"+"bayes"+str(need_sparse_kb)+\
              "bin_nums"+str(bin_nums)+"sample_bound"+str(sample_bound)+"n_trials"+str(n_trials)
              

    kb_builder=KnowledgeBaseBuilder(expr_name="tight_build_gender_classify_cold_start04",
                                    node_ip='192.168.1.7',
                                    node_addr="192.168.1.7:3001",
                                    query_addr="114.212.81.11:3000",
                                    service_addr="114.212.81.11:3500",
                                    query_body=query_body,
                                    conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    rsc_upper_bound=rsc_upper_bound,
                                    kb_name=kb_name,
                                    refer_kb_name=refer_kb_name)

    #建立基于贝叶斯优化的稀疏知识库
    if need_sparse_kb==1:
        '''
        conf_sections={
            "reso":{
                "0":["360p","480p","540p","630p","720p","810p","900p","990p","1080p"],
            },
            "fps":{
                "0":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
            },
            "encoder":{
                "0":["JPEG"]
            }
        }
        conf_sections={
            "reso":{
                "0":["360p","480p","540p"],
                "1":["630p","720p","810p"],
                "2":["900p","990p","1080p"],
            },
            "fps":{
                "0":[1,2,3,4,5],
                "1":[6,7,8,9,10],
                "2":[11,12,13,14,15],
                "3":[16,17,18,19,20],
                "4":[21,22,23,24,25],
                "5":[26,27,28,29,30],
            },
            "encoder":{
                "0":["JPEG"]
            }
        }
        '''
        conf_sections={
            "reso":{
                "0":["360p"],
                "1":["480p"],
                "2":["540p"],
                "3":["630p"],
                "4":["720p"],
                "5":["810p"],
                "6":["900p"],
                "7":["990p"],
                "8":["1080p"],
            },
            "fps":{
                "0":[1,2,3],
                "1":[4,5,6],
                "2":[7,8,9],
                "3":[10,11,12],
                "4":[13,14,15],
                "5":[16,17,18],
                "6":[19,20,21],
                "7":[22,23,24],
                "8":[25,26,27],
                "9":[28,29,30],
            },
            "encoder":{
                "0":["JPEG"]
            }
        }
        '''
        # 1000个点从18个区间里选择6个（含左下右上）
        n_trials=33
        sec_num=4

        # 1000个点从18个区间里选择5个（含左下右上）
        n_trials=40
        sec_num=3

        # 1000个点从18个区间里选择4个（含左下右上）
        n_trials=50
        sec_num=2

        # 1000个点从18个区间里选择3个（含左下右上）
        n_trials=66
        sec_num=1
        '''

        n_trials=10  #每个区间采样多少个点,如果为0说明只采取了左下角点
        sec_num=20 #一共要选取多少个区间。其中必然包含左下角和右上角两个极端情况
        if_continue = False #为False表示从头开始，否则从上一次中断位置开始

        # 以下是基于聚类指导采样时，专用的额外参数
        n_clusters = 5 #如果基于聚类指导采样，n_cluster指出了采样过程中要聚类为多少。初始尝试聚类为3。
        score_n_trials = 100 #如果基于聚类指导采样， score_n_trials指出了每次聚类后分析知识库可用性时需要尝试的次数
        optimze_goal = common.MIN_DELAY #聚类分析的时候暂时用这个来作为优化目标
        
        # 确定当前资源限制之后，就可以开始采样了。
        '''
        kb_builder.sample_for_kb_sections_clus(conf_sections=conf_sections,
                                               sec_num=sec_num,
                                               sample_bound=sample_bound,
                                               n_trials=n_trials,
                                               bin_nums=bin_nums,
                                               delay_range=delay_range,
                                               accuracy_range=accuracy_range,
                                               n_clusters=n_clusters,
                                               score_n_trials=score_n_trials,
                                               optimze_goal=optimze_goal,
                                               if_continue=if_continue
                                               )
        '''
        kb_builder.sample_for_kb_sections_bayes(conf_sections=conf_sections,
                                                sec_num=sec_num,
                                                sample_bound=None,
                                                n_trials=None,
                                                bin_nums=bin_nums,
                                                delay_range=delay_range,
                                                accuracy_range=accuracy_range,
                                                if_continue=if_continue)
        #'''
        # self,conf_sections,sec_num,sample_bound,n_trials,bin_nums,delay_range,accuracy_range,if_continue
            
    


    exit()

    
