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

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
matplotlib.use('TkAgg')

#试图以类的方式将整个独立建立知识库的过程模块化



from common import KB_DATA_PATH,model_op,conf_and_serv_info

# 每一个特定任务对应一个KnowledgeBaseBuilder类
class KnowledgeBaseBuilder():   
    
    #知识库建立者，每次运行的时候要指定本次实验名称，用于作为记录实验的文件
    def __init__(self,expr_name,node_ip,node_addr,query_addr,service_addr,query_body,conf_names,serv_names,service_info_list,rsc_upper_bound,kb_name):

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
    def send_query_and_start(self):  
        # 发出提交任务的请求
        self.query_body['node_addr']=self.node_addr
        r = self.sess.post(url="http://{}/query/submit_query".format(self.query_addr),
                    json=query_body)
        resp = r.json()
        self.query_id = resp["query_id"]
        print('成功发起query',self.query_id)

        #获取一个可用初始配置
        conf = dict({"reso": "360p", "fps": 1, "encoder": "JPEG"})
        flow_mapping = dict()
        resource_limit = dict()

        for serv_name in self.serv_names:
            flow_mapping[serv_name] = model_op[common.cloud_ip]
            resource_limit[serv_name] = {"cpu_util_limit": 1.0, "mem_util_limit": 1.0}
        
        # 持续发送这个初始配置直到成功
        r = self.sess.post(url="http://{}/query/update_prev_plan".format(self.query_addr),
                        json={"job_uid": self.query_id, "video_conf": conf, "flow_mapping": flow_mapping,"resource_limit":resource_limit})
        while not r.json():
            r = self.sess.post(url="http://{}/query/update_prev_plan".format(self.query_addr),
                        json={"job_uid": self.query_id, "video_conf": conf, "flow_mapping": flow_mapping,"resource_limit":resource_limit})
        print("成功为query启动初始配置")

        # 尝试获取运行时情境，如果成功获取运行时情境说明任务已经顺林执行
        r = self.sess.get(url="http://{}/query/get_portrait_info/{}".format(self.query_addr, self.query_id))  
        
        # 持续获取运行时情境直到得到可用资源
        while True:
            if not r.json():
                r  = self.sess.get(url="http://{}/query/get_portrait_info/{}".format(self.query_addr, self.query_id)) 
            elif len(r.json().keys())==0:
                r  = self.sess.get(url="http://{}/query/get_portrait_info/{}".format(self.query_addr, self.query_id)) 
            else:
                break

        
        print("成功获取query的运行时情境画像,query已经开始执行")
        portrait_info=r.json()
        print(portrait_info.keys())
        self.available_resource=portrait_info['available_resource']
        # 根据运行时情境初始化当前可用资源情况

        return self.query_id
    

    
    # init_record_file：
    # 用途：初始化一个csv文件，fieldnames包括n_loop','frame_id','all_delay','edge_mem_ratio'，所有配置
    #      以及每一个服务（含中间传输阶段）的ip、时间、cpu和mem相关限制等。未来可能还需要再修改。
    # 方法：根据初始conf_names和serv_names获知所有配置参数和各个服务名
    # 返回值：该csv文件名，同时在当前目录下生成一个csv文件
    def init_record_file(self):
        time.sleep(1) #这是为了防止采样时间太短导致文件名无法区分，因为strftime的精度只到秒级
        filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + os.path.basename(__file__).split('.')[0] + \
            '_' + str(self.query_body['user_constraint']['delay']) + \
            '_' + self.expr_name + \
            '.csv'
        
        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name
        
        # 首先要确保相关的文件路径已经创建好了
        filepath = self.kb_name+'/'+dag_name+'/'+'record_data'+'/'+filename
    
        self.fp = open(filepath, 'w', newline='')

        fieldnames = ['n_loop',
                     'frame_id',
                     'all_delay',
                     'obj_n',
                     'bandwidth'
                     ]
        # 得到配置名
        for i in range(0, len(self.conf_names)):
            fieldnames.append(self.conf_names[i])  

        #serv_names形如['face_detection', 'face_alignment']
        for i in range(0, len(self.serv_names)):
            serv_name = self.serv_names[i]
            
            field_name = serv_name+'_role'
            fieldnames.append(field_name)
            field_name = serv_name+'_ip'
            fieldnames.append(field_name)
            field_name = serv_name+'_proc_delay'
            fieldnames.append(field_name)

            field_name = serv_name+'_trans_ip'
            fieldnames.append(field_name)
            field_name = serv_name+'_trans_delay'
            fieldnames.append(field_name)


            field_name = serv_name + '_cpu_util_limit'
            fieldnames.append(field_name)
            field_name = serv_name + '_cpu_util_use'
            fieldnames.append(field_name)


            field_name = serv_name + '_mem_util_limit'
            fieldnames.append(field_name)
            field_name = serv_name + '_mem_util_use'
            fieldnames.append(field_name)


        self.writer = csv.DictWriter(self.fp, fieldnames=fieldnames)
        self.writer.writeheader()
        self.written_n_loop.clear() #用于存储各个轮的序号，防止重复记录

        return filename,filepath

    # write_in_file：
    # 用途：在csv文件中写入一条执行记录
    # 方法：参数r2r3r4分别表示资源信息、执行回应和runtime_info，从中提取可以填充csv文件的信息，利用字典把所有不重复的感知结果都保存在updatetd_result之中
    # 返回值：updatetd_result，保存了当前的运行时情境和执行结果
    def write_in_file(self,r2,r3,r4):   #pipeline指定了任务类型   
        system_status = r2.json()
        result = r3.json()
        portrait_info=r4.json()
        # 及时更新当前可用资源
        self.available_resource=portrait_info['available_resource']
        #print(portrait_info.keys())
        #在建立知识库的阶段，画像信息中只有如下内容重要：available_resource，cur_latency，'cur_process_latency，exe_plan，process_delay这几个而已。别的都不重要。
        '''
        print('available_resource',portrait_info['available_resource'])
        print('cur_latency',portrait_info['cur_latency'])
        print('cur_process_latency',portrait_info['cur_process_latency'])
        print('exe_plan',portrait_info['exe_plan'])
        print('process_delay',portrait_info['process_delay'])

        print("system_status")
        print(system_status)
        print('result')
        print(result)
        print('portrait_info')
        print(portrait_info)
        '''
        
        # system_status的典型结构
        '''
       {
            "cloud": {
                "114.212.81.11": {  //以ip为key，标记一个节点
                    "device_state": {  //节点的整体资源利用情况
                        "cpu_ratio": [0.2, 0.0, 0.2, 0.1, 0.3, 0.2, 0.0, 0.7, 0.9],  //节点各个cpu的占用百分比列表
                        "mem_ratio": 5.4,  //节点的内存占用百分比
                        "net_ratio(MBps)": 0.31806,  //节点的带宽
                        "swap_ratio": 0.0, //节点交换内存使用情况
                        "gpu_mem":  {  //节点各个GPU的显存占用百分比字典
                            "0": 0.012761433919270834, // 第0张显卡
                            "1": 0.012761433919270834, // 第1张显卡
                        },
                        "gpu_utilization": {  //节点各个GPU的计算能力利用率百分比字典
                            "0": 7, // 第0张显卡；nano或tx2没有显卡，因此只有"0"这一个键；服务器有多张显卡
                            "1": 8, // 第1张显卡
                        },
                    },
                    "service_state": {  //节点上为各个服务分配的资源情况
                        "face_alignment": {
                            "cpu_util_limit": 0.5,
                            "mem_util_limit": 0.5
                        },
                        "face_detection": {
                            "cpu_util_limit": 0.5,
                            "mem_util_limit": 0.5
                        }
                    }
            }
            },
            "host": {
                "172.27.142.109": {
                    "device_state": {  //节点的整体资源利用情况
                        "cpu_ratio": [0.2, 0.0, 0.2, 0.1, 0.3, 0.2, 0.0, 0.7, 0.9],  //节点各个cpu的占用百分比列表
                        "mem_ratio": 5.4,  //节点的内存占用百分比
                        "net_ratio(MBps)": 0.31806,  //节点的带宽
                        "swap_ratio": 0.0, //节点交换内存使用情况
                        "gpu_mem":  {  //节点各个GPU的显存占用百分比字典
                            "0": 0.012761433919270834, // 第0张显卡
                            "1": 0.012761433919270834, // 第1张显卡
                        },
                        "gpu_utilization": {  //节点各个GPU的计算能力利用率百分比字典
                            "0": 7, // 第0张显卡；nano或tx2没有显卡，因此只有"0"这一个键；服务器有多张显卡
                            "1": 8, // 第1张显卡
                        },
                    },
                    "service_state": {  //节点上为各个服务分配的资源情况
                        "face_alignment": {
                            "cpu_util_limit": 0.5,
                            "mem_util_limit": 0.5
                        },
                        "face_detection": {
                            "cpu_util_limit": 0.5,
                            "mem_util_limit": 0.5
                        }
                    }
                }
            }
        }
                
        '''
        # result的典型结构：
        '''
        {
            // 该部分是列表，代表最近10帧的处理结果。经过改造，其包含每一个任务的执行和传输时延。
            "appended_result": [
                {
                        'count_result': {'total': 24, 'up': 20}, 
                        'delay': 0.16154261735769418, 
                        'execute_flag': True, 
                        'ext_plan': {
                                    'flow_mapping': 
                                        {   
                                            'face_alignment': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
                                            'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'} 
                                        }, 
                                    'video_conf':   {'encoder': 'JPEG', 'fps': 1, 'reso': '360p'}
                                    }, 

                        'ext_runtime': {
                                            'delay': 0.16154261735769418, 
                                            'obj_n': 24.0, 
                                            'obj_size': 219.36678242330404, 
                                            'obj_stable': 1, 
                                            'plan_result': 
                                                {
                                                    'delay': {'face_alignment': 0.09300840817964993, 'face_detection': 0.06853420917804424}, 
                                                    'process_delay': {'face_alignment': 0.08888898446009709, 'face_detection': 0.060828484021700345}
                                                }
                                        }, 
                        'frame_id': 25.0, 
                        'n_loop': 1, 
                        'proc_resource_info_list': [{'cpu_util_limit': 1.0, 'cpu_util_use': 0.060020833333333336, 'latency': 2.3111135959625244, 'pid': 505503}]
                },
                ...
            ],

            // 留空
            "latest_result": {}
        }
        '''
        # portrait_info的典型结构（性别检测情况下）
        '''
       {
            'cur_latency': 0.5,  // 当前任务的执行时延
            'user_constraint': {  // 当前的用户约束
                "delay": 0.8,
                "accuracy": 0.9
            },
            'if_overtime': True,  // 当前任务是否超时，True--超时；False--未超时
            'available_resource': {  // 当前系统中每个设备上本query可以使用的资源量
                '114.212.81.11': {  // 以ip地址表示各个设备
                    'node_role': 'cloud',  // 当前设备是云还是边
                    'available_cpu': 0.5,  // 当前设备可用的CPU利用率
                    'available_mem': 0.8  // 当前设备可用的内存利用率
                },
                '172.27.132.253': {
                    ...
                }
            },
            'resource_portrait': {  // 当前任务中各个服务的资源画像
                'face_detection': {  // 以服务名为key
                    'node_ip': '172.27.132.253',  // 当前服务的执行节点ip
                    'node_role': 'edge',  // 当前服务的执行节点类型
                    'cpu_util_limit': 0.5,  // 当前服务在当前执行节点上的CPU资源限制
                    'cpu_util_use': 0.4,  // 当前服务在当前执行节点上的实际CPU使用量
                    'mem_util_limit': 0.3,  // 当前服务在当前执行节点上的内存资源限制
                    'mem_util_use': 0.1,  // 当前服务在当前执行节点上的实际内存使用量
                    'cpu_portrait': 0,  // CPU资源画像分类，0--弱；1--中；2--强
                    'cpu_bmi': 0.1, // CPU资源的bmi值, (资源分配量-资源需求量) / 资源需求量
                    'cpu_bmi_lower_bound': 0, 
                    'cpu_bmi_upper_bound': 1.1, // 若'cpu_bmi'在['cpu_bmi_lower_bound', 'cpu_bmi_upper_bound']的范围内，说明cpu资源分配量处于合理范围内；若不在此范围内，则说明不合理
                    'mem_portrait': 0,  // 内存资源画像分类，0--弱；1--中；2--强
                    'mem_bmi': 0.1, // 内存资源的bmi值, (资源分配量-资源需求量) / 资源需求量
                    'mem_bmi_lower_bound': 0, 
                    'mem_bmi_upper_bound': 1.1, // 若'mem_bmi'在['mem_bmi_lower_bound', 'mem_bmi_upper_bound']的范围内，说明内存资源分配量处于合理范围内；若不在此范围内，则说明不合理
                    'resource_demand': {  // 当前服务在当前配置、当前工况下，在系统中各类设备上的中资源阈值。注意：由于目前只有一台服务器，且边缘节点都是tx2，所以没有按照ip进行不同设备的资源预估，而是直接对不同类别的设备进行资源预估
                        'cpu': {  // CPU资源
                            'cloud': {  // 在服务器上的资源阈值
                                'upper_bound': 0.1,  // 中资源阈值的上界
                                'lower_bound': 0.05  // 中资源阈值的下界
                            },
                            'edge': {  // 在边缘设备上的资源阈值
                                'upper_bound': 0.1,
                                'lower_bound': 0.05
                            }
                        },
                        'mem': {  // 内存资源
                            'cloud': {
                                'upper_bound': 0.1,
                                'lower_bound': 0.05
                            },
                            'edge': {
                                'upper_bound': 0.1,
                                'lower_bound': 0.05
                            }
                        }
                    }
                },
                'gender_classification': {
                    ...
                }
            }
        }
        '''
        if 'mem_ratio' in system_status['host'][self.node_ip]['device_state']:
            self.edge_mem_ratio=system_status['host'][self.node_ip]['device_state']['mem_ratio']

        appended_result = result['appended_result'] #可以把得到的结果直接提取出需要的内容，列表什么的。
        latest_result = result['latest_result'] #空的

        # updatetd_result用于存储本次从云端获取的有效的更新结果
        updatetd_result=[]

        for res in appended_result:    
            '''
             row={
                'n_loop':n_loop,
                'frame_id':frame_id,
                'all_delay':all_delay,
                'edge_mem_ratio':edge_mem_ratio,

                'encoder':encoder
                'fps':fps,
                'reso':reso,
                
                'face_detection_role':d_role,
                'face_detection_ip':d_ip,
                'face_detection_proc_delay':d_proc_delay,
                'face_detection_trans_delay':d_trans_delay,

                'face_alignment_ip':a_ip,
                'face_alignment_role':a_role,
                'face_alignment_proc_delay':a_proc_delay,
                'face_alignment_trans_delay':a_trans_delay,
            }
            '''
            row = {}
            # print("查看待处理结果")
            # print(res)
            row['n_loop'] = res['n_loop']
            row['frame_id'] = res['frame_id']
            # row['all_delay']=res['delay']
            row['all_delay'] = res['proc_delay']
            row['obj_n'] = res['obj_n']
            row['bandwidth'] = res['bandwidth']
            # row['edge_mem_ratio']=self.edge_mem_ratio

            for i in range(0, len(self.conf_names)):
                conf_name = self.conf_names[i]   #得到配置名
                row[conf_name] = res['ext_plan']['video_conf'][conf_name]

            # serv_names形如['face_detection', 'face_alignment']
            for i in range(0, len(self.serv_names)):
                serv_name = self.serv_names[i]
                
                serv_role_name = serv_name + '_role'
                serv_ip_name = serv_name + '_ip'
                serv_proc_delay_name = serv_name + '_proc_delay'
                trans_ip_name = serv_name + '_trans_ip'
                trans_delay_name = serv_name + '_trans_delay'

                row[serv_role_name] = res['ext_plan']['flow_mapping'][serv_name]['node_role']
                row[serv_ip_name] = res['ext_plan']['flow_mapping'][serv_name]['node_ip']
                row[serv_proc_delay_name] = res['ext_runtime']['plan_result']['process_delay'][serv_name]
                # row['all_delay'] += row[serv_proc_delay_name]
                row[trans_ip_name] = row[serv_ip_name]
                row[trans_delay_name] = res['ext_runtime']['plan_result']['delay'][serv_name] - row[serv_proc_delay_name]

                # 要从runtime_info里获取资源信息。暂时只提取runtime_portrait列表中的第一个画像
                # 以下用于获取每一个服务对应的cpu资源画像、限制和效果
                # field_name=serv_name+'_cpu_portrait'
                # row[field_name]=portrait_info['resource_portrait'][serv_name]['cpu_portrait']
                field_name = serv_name + '_cpu_util_limit'
                row[field_name] = res['ext_runtime']['proc_resource_info'][serv_name]['cpu_util_limit']
                field_name = serv_name + '_cpu_util_use'
                row[field_name] = res['ext_runtime']['proc_resource_info'][serv_name]['cpu_util_use']

                # 以下用于获取每一个服务对应的内存资源画像、限制和效果
                # field_name=serv_name+'_mem_portrait'
                # row[field_name]=portrait_info['resource_portrait'][serv_name]['mem_portrait']
                field_name = serv_name + '_mem_util_limit'
                row[field_name] = res['ext_runtime']['proc_resource_info'][serv_name]['mem_util_limit']
                field_name = serv_name + '_mem_util_use'
                row[field_name] = res['ext_runtime']['proc_resource_info'][serv_name]['mem_util_use']

                # 以下用于获取每一个服务对应的cpu资源画像、限制和效果
                # field_name=serv_name+'_trans'+'_cpu_portrait'
                # row[field_name]=portrait_info['resource_portrait'][serv_name]['cpu_portrait']
                # field_name = serv_name + '_trans' + '_cpu_util_limit'
                # row[field_name] = res['ext_runtime']['proc_resource_info'][serv_name]['cpu_util_limit']
                # field_name = serv_name + '_trans' + '_cpu_util_use'
                # row[field_name] = res['ext_runtime']['proc_resource_info'][serv_name]['cpu_util_use']

                # 以下用于获取每一个服务对应的内存资源画像、限制和效果
                # field_name=serv_name+'_trans'+'_mem_portrait'
                # row[field_name]=portrait_info['resource_portrait'][serv_name]['mem_portrait']
                # field_name = serv_name + '_trans' + '_mem_util_limit'
                # row[field_name] = res['ext_runtime']['proc_resource_info'][serv_name]['mem_util_limit']
                # field_name = serv_name + '_trans'+'_mem_util_use'
                # row[field_name] = res['ext_runtime']['proc_resource_info'][serv_name]['mem_util_use']
                       
            n_loop = res['n_loop']
            if n_loop not in self.written_n_loop:  #以字典为参数，只有那些没有在字典里出现过的row才会被写入文件，
                print('n_loop', n_loop)
                #print(' all_delay:',portrait_info['cur_latency'],'只考虑处理：',row['all_delay'])
                print('总处理时延：', row['all_delay'])
                #print("face_detection处理时延:",row['face_detection_proc_delay'],' 传输时延:',row['face_detection_trans_delay'])
                print("face_detection处理时延:", row['face_detection_proc_delay'])
                #print("gender_classification处理时延:",row['gender_classification_proc_delay'],' 传输时延:',row['gender_classification_trans_delay'])
                print("gender_classification处理时延:", row['gender_classification_proc_delay'])
                print('reso:', row['reso'], ' fps:', row['fps'])
                face_role = 'node'
                if row['face_detection_ip'] == '114.212.81.11':
                    face_role = 'cloud'
                gender_role = 'node'
                if row['gender_classification_ip'] == '114.212.81.11':
                    gender_role = 'cloud'
                print('face_detection_ip:', face_role, ' gender_classification_ip:', gender_role)
                print('face_detection资源')
                #print('cpu限制',row['face_detection_cpu_util_limit'],'cpu使用',row['face_detection_cpu_util_use'])
                #print('mem限制',row['face_detection_mem_util_limit'],'mem使用',row['face_detection_mem_util_use'])
                #print('face_detection资源需求:',portrait_info['resource_portrait']['face_detection']['resource_demand'])
                print('cpu限制', row['face_detection_cpu_util_limit'])
                print('mem限制', row['face_detection_mem_util_limit'])
                print('gender_classification资源')
                #print('cpu限制',row['gender_classification_cpu_util_limit'],'cpu使用',row['gender_classification_cpu_util_use'])
                #print('mem限制',row['gender_classification_mem_util_limit'],'mem使用',row['gender_classification_mem_util_use'])
                #print('gender_classification资源需求:',portrait_info['resource_portrait']['gender_classification']['resource_demand'])
                print('cpu限制', row['gender_classification_cpu_util_limit'])
                print('mem限制', row['gender_classification_mem_util_limit'])
                print()
                self.writer.writerow(row)
                print("写入成功")
                self.written_n_loop[n_loop] = 1
                #完成文件写入之后，将对应的row和配置返回以供分析。由于存在延迟，这些新数据对应的conf和flow_mapping可能和前文指定的不同
                updatetd_result.append({"row":row,"conf":res['ext_plan']['video_conf'],"flow_mapping":res['ext_plan']['flow_mapping'],"resource_limit":res['ext_plan']['resource_limit']})

        #updatetd_result会返回本轮真正检测到的全新数据。在最糟糕的情况下，updatetd_result会是一个空列表。
        return updatetd_result


    # post_get_write：
    # 用途：更新调度计划，感知运行时情境，并调用write_in_file记录在csv文件中
    # 方法：使用update_plan接口将conf,flow_mapping,resource_limit应用在query_id指定的任务中
    #      依次获取运行时情境
    #      使用write_in_file方法将感知到的结果写入文件之中
    # 返回值：包含updated_result的键值对
    def post_get_write(self,conf,flow_mapping,resource_limit):
        # print("开始发出消息并配置")

        #（1）更新配置
        #print('发出新配置')
        r1 = self.sess.post(url="http://{}/query/update_prev_plan".format(self.query_addr),
                        json={"job_uid": self.query_id, "video_conf": conf, "flow_mapping": flow_mapping,"resource_limit":resource_limit})
        if not r1.json():
            return {"status":0,"des":"fail to update plan"}
        

        #（2）获取资源情境,获取node_ip指定的边缘节点的内存使用率
        r2 = self.sess.get(url="http://{}/get_system_status".format(self.service_addr))
        if not r2.json():
            return {"status":1,"des":"fail to get resource info"}
        '''
        else:
            print("收到资源情境为:")
            print(r2.json())
        '''
          
        #（3）查询执行结果并处理
        r3 = self.sess.get(url="http://{}/query/get_result/{}".format(self.query_addr, self.query_id))  
        if not r3.json():
            return {"status":2,"des":"fail to post one query request"}
        
        # (4) 查看当前运行时情境
        r4 = self.sess.get(url="http://{}/query/get_portrait_info/{}".format(self.query_addr, self.query_id))  
        #print(r4)
        if not r4.json():
            return {"status":2,"des":"fail to post one query request"}
        '''
        else:
            print("收到运行时情境为:")
            print(r4.json())
        '''

        # 如果r1 r2 r3都正常
        updatetd_result=self.write_in_file(r2=r2,r3=r3,r4=r4)

        return {"status":3,"des:":"succeed to record a row","updatetd_result":updatetd_result}
    
    # get_write：
    # 用途：感知运行时情境，并调用write_in_file记录在csv文件中。相比post_get_write，不会修改调度计划。
    # 方法：依次获取运行时情境
    #      使用write_in_file方法将感知到的结果写入文件之中
    # 返回值：包含updated_result的键值对
    def get_write(self):
         
        #（1）获取资源情境,获取node_ip指定的边缘节点的内存使用率
        r2 = self.sess.get(url="http://{}/get_system_status".format(self.service_addr))
        if not r2.json():
            return {"status":1,"des":"fail to get resource info"}
        '''
        else:
            print("收到资源情境为:")
            print(r2.json())
        '''
          
        #（2）查询执行结果并处理
        r3 = self.sess.get(url="http://{}/query/get_result/{}".format(self.query_addr, self.query_id))  
        if not r3.json():
            return {"status":2,"des":"fail to post one query request"}
        
        # (4) 查看当前运行时情境
        r4 = self.sess.get(url="http://{}/query/get_portrait_info/{}".format(self.query_addr, self.query_id))  
        #print("r4",r4)
        if not r4.json():
            return {"status":2,"des":"fail to post one query request"}
        '''
        else:
            print("收到运行时情境为:")
            print(r4.json())
        '''
        # 如果r1 r2 r3都正常
        updatetd_result=self.write_in_file(r2=r2,r3=r3,r4=r4)

        return {"status":3,"des:":"succeed to record a row","updatetd_result":updatetd_result}

    # collect_for_sample：
    # 用途：获取特定配置下的一系列采样结果，并将平均结果记录在explore_conf_dict贝叶斯字典中
    # 方法：在指定参数的配置下，反复执行post_get_write获取sample_bound个不重复的结果，并计算平均用时avg_delay
    #      之后将当前配置作为键、avg_delay作为值，以键值对的形式将该配置的采样结果保存在字典中，以供贝叶斯优化时的计算   
    # 返回值：当前conf,flow_mapping,resource_limit所对应的sample_bound个采样结果的平均时延
    def collect_for_sample(self,conf,flow_mapping,resource_limit):
        
        # return 1  #这是为了方便测试功能
        sample_num=0
        sample_result=[]
        all_delay=0
        print('当前采样配置')
        print(conf)
        print(flow_mapping)
        print(resource_limit)
        while(sample_num<self.sample_bound):# 只要已经收集的符合要求的采样结果不达标，就不断发出请求，直到在本配置下成功获取sample_bound个样本
            get_resopnse=self.post_get_write(conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit)
            if(get_resopnse['status']==3): #如果正常返回了结果，就可以从中获取updatetd_result了
                updatetd_result=get_resopnse['updatetd_result']
                # print("展示updated_result")
                # print(updatetd_result)
                # updatetd_result包含一系列形如{"row":row,"conf":res['ext_plan']['video_conf'],"flow_mapping":res['ext_plan']['flow_mapping']}
                # 对于获取的结果，首先检查其conf和flow_mapping是否符合需要，仅在符合的情况下才增加采样点
                for i in range(0,len(updatetd_result)):
                    #print(updatetd_result[i])
                    row0=updatetd_result[i]['row']
                    
                    if updatetd_result[i]['conf']==conf and updatetd_result[i]['flow_mapping']==flow_mapping and updatetd_result[i]['resource_limit']==resource_limit :
                        all_delay+=updatetd_result[i]["row"]["all_delay"]
                        print("该配置符合要求，可作为采样点之一")
                        sample_num+=1

        avg_delay=all_delay/self.sample_bound
        # (1)得到该配置下的平均时延
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
        # (2)得到该配置下的精度
        
        
        print("完成对当前配置的",self.sample_bound,"次采样，平均时延是", avg_delay,"精度是", task_accuracy)
        #融合所有配置，得到一个表述当前调度策略的唯一方案
        conf_str=json.dumps(conf)+json.dumps(flow_mapping)+json.dumps(resource_limit)  
        print(conf_str,avg_delay,task_accuracy)
        # 仅在贝叶斯优化目标是建立稀疏知识库的情况下才会将内容保存在字典中，目的是为了求标准差
        # 初始，explore_conf_dict会被设置一个区间，即min和max对应的内容。如果要建立稀疏知识库，那应该令这一区间内的分布尽可能均匀
        if avg_delay >= self.explore_conf_dict['delay']['min'] and avg_delay<=self.explore_conf_dict['delay']['max']:
            self.explore_conf_dict['delay'][conf_str] = avg_delay  #将所有结果保存在字典里
        if task_accuracy >= self.explore_conf_dict['accuracy']['min'] and task_accuracy <=self.explore_conf_dict['accuracy']['min']:
            self.explore_conf_dict['accuracy'][conf_str] = task_accuracy  #将所有结果保存在字典里

               

        return avg_delay
    

    



    # just_record：
    # 用途：不进行任何配置指定，单纯从当前正在执行的系统中进行record_num次采样。其作用和sample_and_rescord（采样并记录）相对。
    # 方法：初始化一个csv文件，进行record_num次调用get_write并记录当前运行时情境和执行结果，期间完全不控制系统的调度策略
    # 返回值：csv的文件名
    def just_record(self,record_num):
        filename,filepath = self.init_record_file()
        record_sum = 0
        while(record_sum < record_num):
            get_resopnse = self.get_write()
            if(get_resopnse['status'] == 3):
                updatetd_result = get_resopnse['updatetd_result']
                for i in range(0, len(updatetd_result)):
                    #print(updatetd_result[i])
                    record_sum += 1

        self.fp.close()
        print("记录结束，查看文件")
        return filename,filepath 
    

    # sample_and_record：
    # 用途：遍历conf_list、ip、cpu和mem所有配置，对每一种配置进行采样并记录结果。和just_record相对。
    # 方法：初始化一个csv文件，然后生成配置遍历的全排列，对每一种配置都调用collect_for_sample进行采样和记录
    # 返回值：csv的文件名
    def sample_and_record(self,sample_bound):
        self.sample_bound=sample_bound
        filename,filepath = self.init_record_file()
        #执行一次get_and_write就可以往文件里成果写入数据


        # 现在有self.conf_names方便选取conf和flow_mapping了，可以得到任意组合。
        # 当前实现中conf_names里指定的配置是每一个服务都共享的配置，只有model需要特殊考虑。
        # 所以应该记录下服务的数目然后对应处理
        
        conf_list=[]
        for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
            conf_list.append(conf_and_serv_info[conf_name])
        # conf_list会包含各类配置参数取值范围，例如分辨率、帧率等

        serv_ip_list=[]
        for serv_name in self.serv_names:
            serv_ip=serv_name+"_ip"
            serv_ip_list.append(conf_and_serv_info[serv_ip])
        # serv_ip_list包含各个模型的ip的取值范围

        serv_cpu_list=[]
        for serv_name in self.serv_names:
            serv_cpu=serv_name+"_cpu_util_limit"
            serv_cpu_list.append(conf_and_serv_info[serv_cpu])
        # serv_cpu_list包含各个模型的cpu使用率的取值范围

        conf_combine=itertools.product(*conf_list)
        for conf_plan in conf_combine:
            serv_ip_combine=itertools.product(*serv_ip_list)
            for serv_ip_plan in serv_ip_combine:# 遍历所有配置和卸载策略组合
                serv_cpu_combind=itertools.product(*serv_cpu_list)
                for serv_cpu_plan in serv_cpu_combind:
                    #不把内存作为一种配置旋钮 
                    conf={}
                    flow_mapping={}
                    resource_limit={}
                    for i in range(0,len(self.conf_names)):
                        conf[self.conf_names[i]]=conf_plan[i]
                    for i in range(0,len(self.serv_names)):
                        flow_mapping[self.serv_names[i]]=model_op[serv_ip_plan[i]]
                        resource_limit[self.serv_names[i]]={}
                        resource_limit[self.serv_names[i]]["cpu_util_limit"]=serv_cpu_plan[i]
                        resource_limit[self.serv_names[i]]["mem_util_limit"]=1.0
                    '''
                    resource_limit={
                        "face_detection": {
                            "cpu_util_limit": 1,
                            "mem_util_limit": 1,
                        },
                        "face_alignment": {
                            "cpu_util_limit": 1,
                            "mem_util_limit": 1,
                        }
                    }
                    '''
                    self.collect_for_sample(conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit)
                    print("完成一种配置下的数据记录")
                

        # 最后一定要关闭文件
        self.fp.close()
        print("记录结束，查看文件")
        return filename,filepath 


    # objective：
    # 用途：作为贝叶斯优化采样时需要优化的目标函数。
    # 返回值：csv的文件名
    def objective(self,trial):
        conf={}
        flow_mapping={}
        resource_limit={}

        #self.available_resource描述了当前可用资源
        for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
            # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
            conf[conf_name]=trial.suggest_categorical(conf_name,conf_and_serv_info[conf_name])
        
        # 确定配置后，接下来选择云边协同切分点，以及各个服务的资源使用量。此时要格外注意。
        # 流水线中索引小于edge_cloud_cut_point的将被留在边缘执行，因此edge_cloud_cut_point的最大取值是边缘所能负载的最多的服务个数
        # 在指定的配置之下，可以计算出其最低资源需求（cpu按照0.05，mem按照其最低资源需求）
        # 根据self.available_resource中的可用资源量来计算当前配置下最多有多少个服务可以留在边缘
        edge_serv_num=0
        edge_cpu_use=0.0
        edge_mem_use=0.0
        # 注意，需要保证cpu以0.05为最小单位时不会出现偏差小数点
        # 如果让0.6-0.05，那么会得到0.5499999999999999这种结果，此时只能通过round保留为小数点后两位解决
        # 因此以0.05为单位计算时要时刻使用round
        '''
        available_resource 
            {
            '114.212.81.11': 
                {'available_cpu': 1.0, 'available_mem': 1.0, 'node_role': 'cloud'}, 
            '192.168.1.7':
                {'available_cpu': 1.0, 'available_mem': 1.0, 'node_role': 'edge'}
            }
        '''
        myportrait=RuntimePortrait(pipeline=self.serv_names)
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

            if edge_cpu_use <= self.available_resource[common.edge_ip]['available_cpu'] and \
                edge_mem_use <= self.available_resource[common.edge_ip]['available_mem']:
                edge_serv_num+=1 #说明可转移到边缘端的服务又增加了一个
            else:
                break
        # 最终edge_serv_num就是切分点的上限
        conf_and_serv_info["edge_cloud_cut_point"]=[i for i in range(edge_serv_num + 1)]
        choice_idx = trial.suggest_int('edge_cloud_cut_choice', 0, len(conf_and_serv_info["edge_cloud_cut_point"])-1)
        edge_cloud_cut_choice=conf_and_serv_info["edge_cloud_cut_point"][choice_idx]
       
        # 为了防止边缘端上给各个服务分配的cpu使用率被用光，每一个服务能够被分配的CPU使用率存在上限
        # 下限为0.05，上限为目前可用资源量减去剩下的服务数量乘以0.05

        edge_cpu_left=self.available_resource[common.edge_ip]['available_cpu']
        for i in range(1,len(self.serv_names)+1):
            serv_name = self.serv_names[i-1]
            
            if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                flow_mapping[serv_name] = model_op[common.edge_ip]
            else:  # 服务索引大于等于云边协同切分点，在云执行
                flow_mapping[serv_name] = model_op[common.cloud_ip]
            
            # 根据云边切分点选择flow_mapping的选择
            # 接下来进行cpu使用率的选择（但是不进行mem的选择，假设mem不变，在判断切分点的时候就已经确保了内存捕获超过限制）
            # 所以内存限制永远是1.0
            serv_cpu_limit=serv_name+"_cpu_util_limit"
            serv_mem_limit=serv_name+"_mem_util_limit"

            resource_limit[serv_name]={}
            resource_limit[serv_name]["mem_util_limit"]=1.0

            if flow_mapping[serv_name]["node_role"] =="cloud":  #对于云端没必要研究资源约束下的情况
                resource_limit[serv_name]["cpu_util_limit"]=1.0
                
            else: #边端服务会消耗cpu资源
                '''
                # 描述每一种服务所需的中资源阈值
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
                '''
                
                # cpu_upper_bound
                # edge_cloud_cut_choice实际上是要留在边缘的服务的数量，i表示当前是第几个留在边缘的服务，i>=1
                # edge_cloud_cut_choice-i表示还剩下多少服务要分配到边缘上。至少给它们每一个留下0.05的cpu资源
                cpu_upper_bound=edge_cpu_left-round((edge_cloud_cut_choice-i)*0.05,2)
                cpu_upper_bound=round(cpu_upper_bound,2)
                cpu_upper_bound=max(0.05,cpu_upper_bound) #分配量不得超过此值
                cpu_upper_bound=min(self.rsc_upper_bound[serv_name]['cpu_limit'],cpu_upper_bound) #同时也不应该超出其中资源上限
                # 最后给出当前服务可选的cpu
                cpu_select=[]
                cpu_select=[x for x in conf_and_serv_info[serv_cpu_limit] if x <= cpu_upper_bound]

                choice_idx=trial.suggest_int(serv_cpu_limit,0,len(cpu_select)-1)
                resource_limit[serv_name]["cpu_util_limit"]=cpu_select[choice_idx]
                
                edge_cpu_left=round(edge_cpu_left-resource_limit[serv_name]["cpu_util_limit"],2)

                
        
        '''
        resource_limit={
            "face_detection": {
                "cpu_util_limit": 1,
                "mem_util_limit": 1,
            },
            "face_alignment": {
                "cpu_util_limit": 1,
                "mem_util_limit": 1,
            }
        }
        '''
        
        avg_delay= self.collect_for_sample(conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit)
        # 使用此函数，目标是最优化采样得到的时延
        return self.exmaine_explore_distribution()


    # 以贝叶斯采样的方式获取离线知识库
    # sample_and_record_bayes
    # 用途：在特定的各类.conf_names范围下（区间范围下）进行采样，一共采样,n_trials个点，每一种配置采样sample_bound次
    # 方法：调用一层贝叶斯优化
    # 返回值：在该区间上采样得到的文件名，以及表示本次采样最终性能的study.best_value
    def sample_and_record_bayes(self,sample_bound,n_trials,bin_nums,delay_range,accuracy_range):
        self.sample_bound=sample_bound
        self.bin_nums=bin_nums

        filename,filepath = self.init_record_file()

        # 设置self.explore_conf_dict[conf_str]，初始化一个下限和一个上限（上限和下限都可能无法达到），
        # 这个上限和下限可以通过别的方法来获取，目前先假设通过超参数来设置。比如设置为如下，寻找0和0.7之间均匀分布的采样点
        self.explore_conf_dict["delay"]["min"]=delay_range["min"]
        self.explore_conf_dict["delay"]["max"]=delay_range["max"]
        self.explore_conf_dict["accuracy"]["min"]=accuracy_range["min"]
        self.explore_conf_dict["accuracy"]["max"]=accuracy_range["max"]

        #执行一次get_and_write就可以往文件里成果写入数据
        study = optuna.create_study(directions=['minimize' for _ in range(2)], sampler=optuna.samplers.NSGAIISampler())  
        study.optimize(self.objective,n_trials=n_trials)

        self.fp.close()
        print("记录结束，返回filename, filepath, delay_std,accuracy_std")

        delay_std,accuracy_std=self.exmaine_explore_distribution()

        return filename,filepath,delay_std,accuracy_std
    

    # 以贝叶斯采样的方式获取离线知识库
    # 相比前一种方法，这种方法首先要对分区的左下角进行若干次采样，然后才进行贝叶斯优化采样
    def sample_and_record_bayes_for_section(self,sample_bound,n_trials,bin_nums,delay_range,accuracy_range):
        self.sample_bound=sample_bound
        self.bin_nums=bin_nums

        filename,filepath = self.init_record_file()

        # 设置self.explore_conf_dict[conf_str]，初始化一个下限和一个上限（上限和下限都可能无法达到），
        # 这个上限和下限可以通过别的方法来获取，目前先假设通过超参数来设置。比如设置为如下，寻找0和0.7之间均匀分布的采样点
        self.explore_conf_dict["delay"]["min"]=delay_range["min"]
        self.explore_conf_dict["delay"]["max"]=delay_range["max"]
        self.explore_conf_dict["accuracy"]["min"]=accuracy_range["min"]
        self.explore_conf_dict["accuracy"]["max"]=accuracy_range["max"]

        # 进行贝叶斯优化采样之前，首先对左下角进行若干次采样
        bottom_left_conf={}
        
        for conf_name in self.conf_names:
            bottom_left_conf[conf_name]=conf_and_serv_info[conf_name][0]  #取最小值
        bottom_left_plans=self.get_bottom_left_plans(conf=bottom_left_conf)
        print('开始对该分区的左下角进行采样，左下角配置为:',bottom_left_conf)
        for bottom_left_plan in bottom_left_plans:
            conf=bottom_left_plan['conf']
            flow_mapping=bottom_left_plan['flow_mapping']
            resource_limit=bottom_left_plan['resource_limit']
            avg_delay= self.collect_for_sample(conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit)


        print('完成对该分区左下角的采样，开始对分区进行贝叶斯优化采样')
        study = optuna.create_study(directions=['minimize' for _ in range(2)], sampler=optuna.samplers.NSGAIISampler()) 
        study.optimize(self.objective,n_trials=n_trials)


        self.fp.close()
        print("完成对该分区的全部采样返回filename, filepath,bottom_left_plans,delay_std,accuracy_std")
        delay_std,accuracy_std=self.exmaine_explore_distribution()
        return filename,filepath,bottom_left_plans, delay_std,accuracy_std




    # 基于区间的贝叶斯采样函数
    # sample_for_kb_sections_bayes
    # 用途：为一种区间划分方式，进行一系列文件初始化，并调用贝叶斯优化函数完成采样（不包括建库）
    # 方法：建立各类文件实现初始化
    def sample_for_kb_sections_bayes(self,conf_sections,sec_num,sample_bound,n_trials,bin_nums,delay_range,accuracy_range):
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
        # (1)首先要生成一个初始化文件描述当前分区信息

        section_info={}
        section_info['des']=datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
        section_info['serv_names']=self.serv_names
        section_info['conf_names']=self.conf_names
        section_info['conf_sections']=conf_sections
        section_info['section_ids']=list([])
        
        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name
        
        # （2）进行采样前首先保存好原始的各个配置的取值范围，以供未来恢复    
        original_conf_range={}
        for conf_name in self.conf_names:
            original_conf_range[conf_name]=conf_and_serv_info[conf_name]


        # (3)计算整个空间中的左下角分区和右上角分区
        bottom_left_plans=[]
        # (3.1)计算左下角的最小分区
        section_id_min=''
        for i in range(0,len(self.conf_names)):
            conf_name=self.conf_names[i]            
            conf_and_serv_info[conf_name]=conf_sections[conf_name]['0']
            # 计算所得section_id
            if i>0:
                section_id_min+='-'
            section_id_min=section_id_min+conf_name+'='+'0'
        print('准备对分区采样,当前左下角分区id为:',section_id_min)
        # 首先清空已有的采样字典
        self.clear_explore_distribution(delay_range=delay_range,accuracy_range=accuracy_range)
        filename,filepath,bottom_left_plans,delay_std,accuracy_std=self.sample_and_record_bayes_for_section(sample_bound=sample_bound,
                                                                             n_trials=n_trials,bin_nums=bin_nums,
                                                                             delay_range=delay_range,accuracy_range=accuracy_range)
        if section_id_min not in section_info['section_ids']:
            section_info['section_ids'].append(section_id_min)
            section_info[section_id_min]={}
            section_info[section_id_min]['filenames']=[]
            section_info[section_id_min]['bottom_left_plans']=bottom_left_plans

        if filename not in section_info[section_id_min]['filenames']:
            section_info[section_id_min]['filenames'].append(filename)
        
        with open(self.kb_name+'/'+dag_name+'/'+'section_info'+".json", 'w') as f:  
            json.dump(section_info, f,indent=4) 
        f.close()
        
        print('已在section_info中更新左下角分区的filenames信息和bottom_left_plans信息')

         # (3.2)计算右上角的最大分区
        section_id_max=''
        for i in range(0,len(self.conf_names)):
            conf_name=self.conf_names[i]    
            max_idx=str(len(conf_sections[conf_name])-1)        
            conf_and_serv_info[conf_name]=conf_sections[conf_name][max_idx]
            # 计算所得section_id
            if i>0:
                section_id_max+='-'
            section_id_max=section_id_max+conf_name+'='+max_idx
        print('准备对分区采样,当前右上角分区id为:',section_id_max)
        if section_id_max==section_id_min:
            print('右上角等于左下角，不必再采样')
        else:
            # 首先清空已有的采样字典
            self.clear_explore_distribution(delay_range=delay_range,accuracy_range=accuracy_range)
            filename,filepath,bottom_left_plans,delay_std,accuracy_std=self.sample_and_record_bayes_for_section(sample_bound=sample_bound,
                                                                                n_trials=n_trials,bin_nums=bin_nums,
                                                                                delay_range=delay_range,accuracy_range=accuracy_range)
        if section_id_max not in section_info['section_ids']:
            section_info['section_ids'].append(section_id_max)
            section_info[section_id_max]={}
            section_info[section_id_max]['filenames']=[]
            section_info[section_id_max]['bottom_left_plans']=bottom_left_plans

        if filename not in section_info[section_id_max]['filenames']:
            section_info[section_id_max]['filenames'].append(filename)
        
        with open(self.kb_name+'/'+dag_name+'/'+'section_info'+".json", 'w') as f:  
            json.dump(section_info, f,indent=4) 
        f.close()
        
        print('已在section_info中更新该右上角分区的filenames信息和bottom_left_plans信息')


        #(3)开始从分区信息中提取分区，从而进行嵌套贝叶斯优化。使用ask_and_tell方法。
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
        # 
        study = optuna.create_study(directions=['minimize' for _ in range(2)], sampler=optuna.samplers.NSGAIISampler())  
        sec_sum=0
        while True:
            if sec_sum==sec_num or section_id_max==section_id_min: #如果只有一种区间，那永远不可能满足循环终止条件
                print('当前sec_sum是',sec_sum)
                break
            else:
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

                print('分区id是',section_id)
                if section_id_max==section_id or section_id_min==section_id:
                    print('与左下角或右上角重合，不必考虑')
                else:
                    print('准备对分区采样,当前分区id为:',section_id)
                    sec_sum+=1

                    # 首先清空已有的采样字典
                    self.clear_explore_distribution(delay_range=delay_range,accuracy_range=accuracy_range)
                    filename,filepath,bottom_left_plans,delay_std,accuracy_std=self.sample_and_record_bayes_for_section(sample_bound=sample_bound,
                                                                                        n_trials=n_trials,bin_nums=bin_nums,
                                                                                         delay_range=delay_range,accuracy_range=accuracy_range)
                    if section_id not in section_info['section_ids']:
                        section_info['section_ids'].append(section_id)
                    if section_id not in section_info.keys():
                        section_info[section_id]={}
                        section_info[section_id]['filenames']=[]
                        
                    section_info[section_id]['filenames'].append(filename)
                    section_info[section_id]['bottom_left_plans']=bottom_left_plans

                    with open(self.kb_name+'/'+dag_name+'/'+'section_info'+".json", 'w') as f:  
                        json.dump(section_info, f,indent=4) 
                    f.close()
                    
                    print('已在section_info中更新该分区的filenames信息和bottom_left_plans信息')

                    ans=[delay_std,accuracy_std]
                    study.tell(trial,ans)  # tell the pair of trial and objective value
    


        #最后恢复原始的各个配置的取值，并写入section_info
        for conf_name in self.conf_names:
            conf_and_serv_info[conf_name]=original_conf_range[conf_name]
        
        with open(self.kb_name+'/'+dag_name+'/'+'section_info'+".json", 'w') as f:  
            json.dump(section_info, f,indent=4) 
        f.close()

    '''
     with open(self.kb_name+'/'+eval_name+".json", 'r') as f:  
            evaluator = json.load(f)  
        f.close()
        return evaluator  #返回加载得到的字典

    with open(self.kb_name+'/'+eval_name+".json", 'w') as f:  
            json.dump(evaluator, f,indent=4) 
        f.close()

    '''
    # update_section_from_file
    # 用途：根据一个csv文件记录的内容，为流水线上所有服务更新一个分区知识库
    # 返回值：无，但是为每一个服务都生成了一个区间的CSV文件
    def update_section_from_file(self,filename,section_id):
        # section_id形如"reso=0-fps=0-encoder=0",filename是以csv结尾的文件名
        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name
        
        #(1)读取csv文件中内容
        filepath = self.kb_name + '/' + dag_name + '/' + 'record_data' + '/' +filename
        df = pd.read_csv(filepath) #读取数据内容

        #(2)获取section_info，因为后续需要更新每一个服务的conf_info，也就是在特定分区下各个配置的取值范围
        section_info={}
        with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
            section_info=json.load(f) 
        f.close()
        # 为这个分区创建conf_info
        if 'conf_info' not in section_info[section_id].keys():
            section_info[section_id]['conf_info']={}
            for serv_name in self.serv_names:
                section_info[section_id]['conf_info'][serv_name]={}
        
        # 获取当前分区下各个配置的取值范围
        sec_conf_range={} 
        for part in section_id.split('-'):
            # 使用 split('=') 分割键值对  
            conf_name, value = part.split('=')  
            # 将键值对添加到字典中  
            sec_conf_range[conf_name] = section_info['conf_sections'][conf_name][value]
            

        #(3)依次为每一个服务更新该分区下的知识库
        # 首先，完成对每一个服务的性能评估器的初始化,并提取其字典，加入到性能评估器列表中
        for service_info in self.service_info_list:
            
            #(3.1)计算知识库中该服务需要的所有配置旋钮,构建service_conf
            service_conf=list(service_info['conf']) # 形如":["reso","fps"，"encoder""]
            # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
            service_conf.append(service_info['name']+'_ip')
            # 然后再添加资源约束，目前只考虑cpu使用率限制 cpu_util_limit
            service_conf.append(service_info['name']+'_cpu_util_limit')
            # 然后再添加资源约束，也就是mem限制 mem_util_limit
            service_conf.append(service_info['name']+'_mem_util_limit')
            #service_conf形如['reso', 'fps', 'encoder', 'face_detection_ip', 'face_detection_cpu_util_limit', 'face_detection_mem_util_limit']
           
            #(3.2)为了更新该服务在该分区下的json文件，首先判断是否存在这个json，不存在就创建一个
            eval_path=self.kb_name+'/'+ dag_name + '/' + service_info['name'] + '/' +section_id + '.json'
            evaluator=dict()
            if not ( os.path.exists(eval_path) and os.path.isfile(eval_path) ): #如果文件不存在，创立文件并写入空字典
                with open(eval_path, 'w') as f:  
                    json.dump(evaluator, f,indent=4) 
                f.close()
            # 然后读取文件中的原始evaluator，试图更新它
            with open(eval_path, 'r') as f:  
                evaluator = json.load(f)  
            f.close()

            #(3.2)更新该服务的各个配置取值范围，记录在section_info之中，此处进行相关初始化
            conf_info=dict() #记录该分区中已经出现的配置
            print('为',section_id,'分区更新',service_info['name'],'的conf_info')
            for conf_name in service_conf: #conf_and_serv_info中包含了service_conf中所有配置的可能取值 
                conf_info[conf_name]=set()
                if conf_name not in section_info[section_id]['conf_info'][service_info['name']].keys():
                    section_info[section_id]['conf_info'][service_info['name']][conf_name]=[]
                

            #(3.3)计算每一种文件中出现过的配置取值并记录到evaluator里,同时更新section_info中的conf_info
            print('为',section_id,'分区更新',service_info['name'],'的evaluator')
            min_index=df.index.min()
            max_index=df.index.max()
            conf_kind=0 #记录总配置数
            used_set=set() #防止配置重复


            for index in range(min_index,max_index+1):
                #遍历每一个配置
                values_list=df.loc[index,service_conf].tolist()
                #values_list形如['990p', 24, 'JPEG', '114.212.81.11', 1.0, 1.0]
                #service_conf形如['reso', 'fps', 'encoder', 'face_detection_ip', 'face_detection_cpu_util_limit', 'face_detection_mem_util_limit']
                if_in_sec=True #判断当前配置在不在分区中
                for conf_name in self.conf_names:
                    if values_list[service_conf.index(conf_name)] not in sec_conf_range[conf_name]:
                        if_in_sec=False
                        print('出现错误配置',values_list)
                        print('范围是', sec_conf_range[conf_name])
                        break
                
                if if_in_sec: #如果当前配置在区间范围内，才有资格更新知识库
                    #更新该服务当前的conf_info，也就是各个配置的取值范围
                    for i in range(0,len(service_conf)):
                        conf_info[service_conf[i]].add(values_list[i])

                    dict_key=''
                    for i in range(0,len(service_conf)):
                        dict_key+=service_conf[i]+'='+str(values_list[i])+' '
                    
                    if dict_key not in used_set:
                        used_set.add(dict_key) #如果该配置尚未被记录，需要求avg_delay
                        condition_all=True  #用于检索字典所需的条件
                        for i in range(0,len(values_list)):
                            condition=df[service_conf[i]]==values_list[i]    #相应配置名称对应的配置参数等于当前配置组合里的配置
                            condition_all=condition_all&condition
                        
                        # 联立所有条件从df中获取对应内容,conf_df里包含满足所有条件的列构成的df
                        conf_df=df[condition_all]
                        if(len(conf_df)>0): #如果满足条件的内容不为空，可以开始用其中的数值来初始化字典
                            conf_kind+=1
                            avg_value=conf_df[service_info['value']].mean()  #获取均值
                            evaluator[dict_key]=avg_value

            #利用该服务的conf_info来更新section_info
            print('展示当前conf_info')
            print(conf_info)
            print(section_info[section_id]['conf_info'][service_info['name']])
            for conf_name in service_conf:
                temp_list=section_info[section_id]['conf_info'][service_info['name']][conf_name]+list(conf_info[conf_name])
                print(type(temp_list),type(temp_list[0]))
                if isinstance(temp_list[0],np.int64):
                    temp_list=[int(x) for x in temp_list]

                section_info[section_id]['conf_info'][service_info['name']][conf_name]=list(set(temp_list))

            print(section_info[section_id]['conf_info'][service_info['name']])
                                                                

            #完成对该服务的evaluator的处理，写入字典中
            with open(eval_path, 'w') as f:  
                json.dump(evaluator, f,indent=4) 
            f.close()
            #把更新后的section_info也写入字典中
            with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'w') as f:  
                json.dump(section_info, f,indent=4) 
            f.close()

            #self.evaluator_dump(evaluator=evaluator,eval_name=service_info['name'])
            print("该服务",service_info['name'],"涉及配置组合总数",conf_kind)



    # update_sections_from_files:
    # 用途：利用section_info里的结果来更新分区知识库
    # 方法：从section_info中提取多个分区，以及相对应的文件，多次调用update_section_from_file生成多个文件
    def update_sections_from_files(self):

        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name
        section_info={}
        with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
            section_info=json.load(f) 
        f.close()
        section_ids=section_info['section_ids']
        for section_id in section_ids:
            for filename in section_info[section_id]['filenames']:
                self.update_section_from_file(filename=filename,section_id=section_id)
        print('完成对全部分区的知识库更新')

    
    # swap_node_ip_in_kb：
    # 用途：将知识库中指定的ip替换为另一种ip
    # 方法：读取知识库中所有的配置组合，然后将其中的old_node_ip部分换成new_node_ip。这样做是为了增加知识库的泛用性。
    # 返回值：无
    def swap_node_ip_in_kb(self,old_node_ip,new_node_ip):

        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name
        
        #(1)读取section_info并更新其中的old_node_ip
        section_info={}
        with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
            section_info=json.load(f) 
        f.close()


        for section_id in section_info['section_ids']:
            #（1.1）更新该分区的左下角计划
            bottom_left_plans = section_info[section_id]["bottom_left_plans"]
            for i in range(0,len(bottom_left_plans)):
                plan=bottom_left_plans[i]
                for serv_name in plan['flow_mapping'].keys():
                    if plan['flow_mapping'][serv_name]['node_ip']==old_node_ip:
                        section_info[section_id]["bottom_left_plans"][i]['flow_mapping'][serv_name]['node_ip']=new_node_ip
            # (1.2)更新该分区的conf_info
            conf_info = section_info[section_id]["conf_info"]
            for serv_name in conf_info.keys():
                ip_list=conf_info[serv_name][serv_name+'_ip']
                if new_node_ip not in ip_list:
                    ip_list.append(new_node_ip)
                # 增加新ip的同时删除老ip
                new_ip_list = [ip for ip in ip_list if ip != old_node_ip]  
                section_info[section_id]['conf_info'][serv_name][serv_name+'_ip']=new_ip_list
            
        #最后重新写入section_info
        with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'w') as f:  
            json.dump(section_info, f,indent=4) 
        f.close()


        #（2）将每一个服务下的每一个分区的json文件中的旧ip换成新ip
        for serv_name in self.serv_names:
            for section_id in section_info['section_ids']:
                evaluator={}
                with open(self.kb_name + '/' + dag_name + '/' + serv_name+ '/' + section_id+'.json', 'r') as f:  
                    evaluator=json.load(f) 
                f.close()

                new_evaluator=dict()
                keys_to_remove=[]

                for dict_key in evaluator.keys():
                    conf_key=str(dict_key)
                    conf_ip=serv_name+'_ip='
                    pattern = rf"{re.escape(conf_ip)}([^ ]*)" 
                    match = re.search(pattern, conf_key)
                    if match:
                        old_ip=match.group(1)
                        if old_ip==old_node_ip:
                            keys_to_remove.append(conf_key)
                            new_conf_key = re.sub(pattern, f"{conf_ip}{new_node_ip}", conf_key) 
                            if new_conf_key not in evaluator:
                                new_evaluator[new_conf_key]=evaluator[conf_key]
         
                for key in keys_to_remove:  
                    del evaluator[key]  
                            
                merged_dict = {**evaluator, **new_evaluator}  
                #完成对该服务的evaluator的处理
                with open(self.kb_name + '/' + dag_name + '/' + serv_name+ '/' + section_id+'.json', 'w') as f:  
                    json.dump(merged_dict, f,indent=4) 
                f.close()

        print('完成对知识库中特定ip配置的替换')
        


    # draw_scatter：
    # 用途：根据参数给定的x和y序列绘制散点图
    # 方法：不赘述
    # 返回值：无
    def draw_scatter(self,x_value,y_value,title_name):
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'
        plt.yticks(fontproperties='Times New Roman', )
        plt.xticks(fontproperties='Times New Roman', )
        plt.scatter(x_value,y_value,s=0.5)
        plt.title(title_name)
        plt.show()
    
    # draw_scatter：
    # 用途：根据参数给定的data序列和bins绘制直方图
    # 方法：不赘述
    # 返回值：绘制直方图时返回的a,b,c(array, bins, patches)，
    #        其中，array是每个bin内的数据个数，bins是每个bin的左右端点，patches是生成的每个bin的Patch对象。
    def draw_hist(self,data,title_name,bins):
        print('准备绘制中')
        # print(data)
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'
        plt.yticks(fontproperties='Times New Roman', )
        plt.xticks(fontproperties='Times New Roman', )
        a,b,c=plt.hist(x=data,bins=bins)
        plt.title(title_name)
        plt.show()
        return a,b,c

    # draw_picture：
    # 用途：根据参数给定的x和y序列绘制曲线图
    # 方法：不赘述
    # 返回值：无
    def draw_picture(self, x_value, y_value, title_name, figure_broaden=False, xlabel=None, ylabel=None):
        # if figure_broaden:
        #     plt.figure(figsize=[8, 5])  
        # else:
        #     plt.figure(figsize=[5.5, 4.5])  
        plt.figure(figsize=[8, 5])
        if xlabel:
            plt.xlabel(xlabel, fontdict={'fontsize': 13, 'family': 'SimSun'})
        if ylabel:
            plt.ylabel(ylabel, fontdict={'fontsize': 13, 'family': 'SimSun'})
        plt.yticks(fontproperties='Times New Roman')
        plt.xticks(fontproperties='Times New Roman')
        plt.plot(x_value, y_value)
        plt.title(title_name, fontdict={'fontsize': 15, 'family': 'SimSun'})
        plt.grid(ls="--", alpha=0.4)  # 绘制虚线网格
        plt.show()

    # draw_delay_and_cons：
    # 用途：在相同的x值上绘制两个y值，如果需要绘制约束的话就用它
    # 方法：不赘述
    # 返回值：无
    def draw_delay_and_cons(self, x_value1, y_value1, y_value2, title_name):
        plt.figure(figsize=[8, 5])  
        plt.xlabel("帧数", fontdict={'fontsize': 13, 'family': 'SimSun'})
        plt.ylabel("时延/s", fontdict={'fontsize': 13, 'family': 'SimSun'})
        plt.yticks(fontproperties='Times New Roman')
        plt.xticks(fontproperties='Times New Roman')
        plt.plot(x_value1, y_value1, label="执行时延")
        plt.plot(x_value1, y_value2, label="时延约束")
        plt.title(title_name, fontdict={'fontsize': 15, 'family': 'SimSun'})
        plt.grid(ls="--", alpha=0.4)  # 绘制虚线网格
        plt.legend(prop={'family': 'SimSun', 'size': 9})
        plt.show()

    
    # draw_picture_from_sample：
    # 用途：根据filepath指定的日志一次性绘制大量图片
    # 方法：不赘述
    # 返回值：无
    def draw_picture_from_sample(self,filepath): #根据文件采样结果绘制曲线
        df = pd.read_csv(filepath)
        df = df.drop(index=[0])
        df = df[df.all_delay<3]

        # 要求绘制
        # 总时延随时间的变化
        # 每一个服务各自的时延随时间的变化
        # 总时延和所有服务的时延随时间的变化
        # 各个配置各自的变化
        # self.conf_names
        # self.serv_names

        x_list = []
        for i in df['n_loop']:
            x_list.append(i)

        cons_delay = []
        for x in df['n_loop']:
            cons_delay.append(self.query_body['user_constraint']['delay'])

        # 绘制总时延和约束时延
        self.draw_delay_and_cons(x_value1=x_list, y_value1=df['all_delay'], y_value2=cons_delay, title_name="执行时延随时间变化图")

        bandwidth_list = []
        for bandwidth in df['bandwidth']:
            bandwidth_list.append(bandwidth)
        self.draw_picture(x_list, bandwidth_list, title_name='带宽/时间', figure_broaden=True, xlabel='帧数', ylabel='带宽/ (kB/s)')
        
        
        face_detection_cpu_util_limit = None
        face_detection_serv_node = None
        gender_classification_cpu_util_limit = None
        gender_classification_serv_node = None
        
        for serv_name in self.serv_names:
            serv_role_name = serv_name+'_role'
            serv_ip_name = serv_name+'_ip'
            serv_ip_list = df[serv_ip_name].tolist()
            serv_node_list = []
            for ip in serv_ip_list: 
                if ip == '114.212.81.11':
                    serv_node_list.append('cloud')
                else:
                    serv_node_list.append('edge')
            serv_proc_delay_name = serv_name+'_proc_delay'
            trans_ip_name = serv_name+'_trans_ip'
            trans_delay_name = serv_name+'_trans_delay'
            # 绘制各个服务的处理时延以及ip变化
                        # 以下用于获取每一个服务对应的内存资源画像、限制和效果
            mem_portrait = serv_name+'_mem_portrait'
            mem_util_limit = serv_name+'_mem_util_limit'
            mem_util_use = serv_name+'_mem_util_use'

            cpu_portrait = serv_name+'_cpu_portrait'
            cpu_util_limit = serv_name+'_cpu_util_limit'
            cpu_util_use = serv_name+'_cpu_util_use'

            # self.draw_picture(x_value=x_list, y_value=serv_node_list, title_name=serv_name+"执行节点随时间变化图", figure_broaden=True, xlabel='帧数', ylabel='执行节点')
            if serv_name == 'face_detection':
                face_detection_serv_node = serv_node_list
            else:
                gender_classifity_serv_node = serv_node_list
                
            # self.draw_picture(x_value=x_list,y_value=df[serv_ip_name],title_name=serv_ip_name+"/时间",figure_broaden=True)
            # self.draw_picture(x_value=x_list,y_value=df[serv_proc_delay_name],title_name=serv_proc_delay_name+"/时间")
            # self.draw_picture(x_value=x_list,y_value=df[trans_delay_name],title_name=trans_delay_name+"/时间")

            # self.draw_picture(x_value=x_list, y_value=df[mem_portrait], title_name=mem_portrait+"/时间")
            # self.draw_picture(x_value=x_list, y_value=df[mem_util_limit], title_name=mem_util_limit+"/时间")

            # print(df[mem_util_use])
            # self.draw_picture(x_value=x_list, y_value=df[mem_util_use], title_name=mem_util_use+"/时间")

            # self.draw_picture(x_value=x_list, y_value=df[cpu_portrait], title_name=cpu_portrait+"/时间")
            if serv_name == 'face_detection':
                face_detection_cpu_util_limit = df[cpu_util_limit]
            else:
                gender_classifity_cpu_util_limit = df[cpu_util_limit]
            # self.draw_picture(x_value=x_list, y_value=df[cpu_util_limit], title_name=serv_name+" CPU分配量随时间变化图", xlabel='帧数', ylabel='CPU分配量')
            # self.draw_picture(x_value=x_list, y_value=df[cpu_util_use], title_name=serv_name+" CPU使用量随时间变化图", xlabel='帧数', ylabel='CPU使用量')
        
        plt.figure(figsize=[8, 5])  
        plt.xlabel("帧数", fontdict={'fontsize': 13, 'family': 'SimSun'})
        plt.ylabel("执行节点", fontdict={'fontsize': 13, 'family': 'SimSun'})
        plt.yticks(fontproperties='Times New Roman')
        plt.xticks(fontproperties='Times New Roman')
        plt.plot(x_list, face_detection_serv_node, label="face_detection")
        plt.plot(x_list, gender_classifity_serv_node, label="gender_classification")
        plt.title('执行节点随时间变化图', fontdict={'fontsize': 15, 'family': 'SimSun'})
        plt.grid(ls="--", alpha=0.4)  # 绘制虚线网格
        plt.legend(prop={'family': 'SimSun', 'size': 9})
        plt.show()
        
        plt.figure(figsize=[8, 5])  
        plt.xlabel("帧数", fontdict={'fontsize': 13, 'family': 'SimSun'})
        plt.ylabel("CPU分配量", fontdict={'fontsize': 13, 'family': 'SimSun'})
        plt.yticks(fontproperties='Times New Roman')
        plt.xticks(fontproperties='Times New Roman')
        plt.plot(x_list, face_detection_cpu_util_limit, label="face_detection")
        plt.plot(x_list, gender_classifity_cpu_util_limit, label="gender_classification")
        plt.title('CPU分配量随时间变化图', fontdict={'fontsize': 15, 'family': 'SimSun'})
        plt.grid(ls="--", alpha=0.4)  # 绘制虚线网格
        plt.legend(prop={'family': 'SimSun', 'size': 9})
        plt.show()
        
        conf_draw_dict = {
            'reso': {
                'title_name': '分辨率',
                'ylabel': '分辨率'
            },
            'fps': {
                'title_name': '帧率',
                'ylabel': '帧率'
            }
        }
        for conf_name in self.conf_names:
            if conf_name in conf_draw_dict:
                self.draw_picture(x_value=df['n_loop'], y_value=df[conf_name], title_name=conf_draw_dict[conf_name]['title_name']+"随时间变化图", xlabel='帧数', ylabel=conf_draw_dict[conf_name]['ylabel'])
        
        # plt.show()

    


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
    need_sparse_kb = 0
    # 是否进行严格采样（遍历所有配置）
    need_tight_kb = 0
    # 是否根据某个csv文件绘制画像 
    need_to_draw = 0
    # 是否需要基于初始采样结果建立一系列字典，也就是时延有关的知识库
    need_to_build = 1
    # 是否需要将某个文件的内容更新到知识库之中
    need_to_add = 0
    # 判断是否需要在知识库中存放新的边缘ip，利用已有的更新
    need_new_ip = 0

    #是否需要发起一次简单的查询并测试调度器的功能
    need_to_test = 0

    #是否需要验证一下函数的正确性
    need_to_verify = 0


    task_name = "gender_classify"

    kb_name = 'kb_data_6sec-1'

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
                                    kb_name=kb_name)


    filepath=''

    if need_to_verify==1:
        conf_sections={
        }
        '''
        kb_builder.sample_for_kb_sections_bayes(conf_sections=conf_sections,sec_num=sec_num,sample_bound=sample_bound,
                                                n_trials=n_trials,bin_nums=bin_nums,
                                                delay_range=delay_range,accuracy_range=accuracy_range)
        '''
        #kb_builder.update_sections_from_files()
        '''
        csv_paths=['kb_data/face_detection-gender_classification/record_data/20240509_23_06_41_kb_sec_builder_1.0_tight_build_gender_classify_cold_start04.csv',\
                   'kb_data/face_detection-gender_classification/record_data/20240509_22_49_27_kb_sec_builder_1.0_tight_build_gender_classify_cold_start04.csv']
        for csv_path in csv_paths:
            df=pd.read_csv(csv_path)
            column_name = 'encoder'  
            fixed_value = 'JPEG'  
            # 替换列的值  
            df[column_name] = fixed_value  
            # 将修改后的DataFrame写回到CSV文件  
            df.to_csv(csv_path, index=False)  # index=False是为了避免将索引写入CSV文件
        '''
        #kb_builder.swap_node_ip_in_kb(old_node_ip='192.168.1.8',new_node_ip='192.168.1.7')

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
        '''
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
        # 33 4 一共6个区间

        n_trials=50
        sec_num=2
        
        kb_builder.send_query_and_start() 
        # 确定当前资源限制之后，就可以开始采样了。
        kb_builder.sample_for_kb_sections_bayes(conf_sections=conf_sections,sec_num=sec_num,sample_bound=sample_bound,
                                                n_trials=n_trials,bin_nums=bin_nums,
                                                 delay_range=delay_range,accuracy_range=accuracy_range)
        
    
    # 关于是否需要建立知识库：可以根据txt文件中的内容来根据采样结果建立知识库
    if need_to_build==1:
        filepath=''
        kb_builder.update_sections_from_files()
        print("完成时延知识库的建立")
    
    if need_new_ip==1:
        # 此处可以直接修改知识库中的特定配置，比如删除所有特定ip下的配置，或者基于各个边缘端都相同的思想添加新的边缘端ip配置，或者将知识库中的某个ip换成新的ip
        kb_builder.swap_node_ip_in_kb(old_node_ip='172.27.143.164',new_node_ip='192.168.1.7')
        print('完成更新')
    
        # 建立基于遍历各类配置的知识库
    if need_tight_kb==1:
        kb_builder.send_query_and_start() 
        kb_builder.sample_and_record(sample_bound=10) #表示对于所有配置组合每种组合采样sample_bound次。

        #是否需要发起一次简单的查询并测试调度器的功能
    if need_to_test==1:
        kb_builder.send_query() 
        filepath=kb_builder.just_record(record_num=350)

        kb_builder.anylze_explore_result(filepath=filepath)
        kb_builder.draw_picture_from_sample(filepath=filepath)

    # 关于是否需要绘制图像
    if need_to_draw==1:
        print('准备画画')
        filepath='kb_data/20240407_20_52_49_kb_builder_0.3_tight_build_gender_classify_cold_start04.csv'
        kb_builder.anylze_explore_result(filepath=filepath)
        kb_builder.draw_picture_from_sample(filepath=filepath)


    exit()

    
