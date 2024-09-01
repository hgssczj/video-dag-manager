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
from scheduler_2_evolve.offline_simulator import OfflineSimulator
from itertools import product

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
matplotlib.use('TkAgg')

#试图以类的方式将整个独立建立知识库的过程模块化

#本kb_sec_builder_div_test力图把云边协同切分点也作为区间知识库的一个配置


from common import model_op,conf_and_serv_info


scheduler_folder_name = 'scheduler_2_evolve'



# 每一个特定任务对应一个KnowledgeBaseBuilder类
class KnowledgeBaseBuilder():   
    
    #知识库建立者，每次运行的时候要指定本次实验名称，用于作为记录实验的文件
    def __init__(self,expr_name,node_ip,node_addr,query_addr,service_addr,query_body,task_conf_names,rsc_conf_names,serv_names,service_info_list,rsc_upper_bound,kb_name):

        #(1)基础初始化：实验名称、ip、各类节点地址
        self.expr_name = expr_name #用于构建record_fname的实验名称
        self.node_ip = node_ip #指定一个ip,用于从resurece_info获取相应资源信息
        self.node_addr = node_addr #指定node地址，向该地址更新调度策略
        self.query_addr = query_addr #指定query地址，向该地址提出任务请求并获取结果
        self.service_addr = service_addr #指定service地址，从该地址获得全局资源信息


        #(2)任务初始化：待执行任务的query内容、任务相关的配置、涉及的服务名称、服务相关的信息、任务的id
        self.query_body = query_body
        self.query_id = None #查询返回的id
        self.serv_names = serv_names
        self.service_info_list=service_info_list

        self.task_conf_names = task_conf_names  #任务相关配置。对于视频流，一般就是["reso","fps","encoder"]
        self.rsc_conf_names = rsc_conf_names #资源相关配置。一般包括["edge_cloud_cut_point","face_detection_ip","gender_classification_ip",
                                #"face_detection_mem_util_limit","face_detection_cpu_util_limit",
                                # "gender_classification_mem_util_limit","gender_classification_cpu_util_limit"]
        self.sec_conf_names = []  # 用于进行区间划分的配置，比如可以用reso、fps、edge_cloud_cut_point来进行区间划分

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

    # examine_explore_distribution:
    # 用途：获取贝叶斯优化采样过程中，采样结果的均匀程度，即方差（可考虑改成CoV）
    # 方法：将字典explore_conf_dict中的键值对的值提取出来构建数组，提取方差
    # 返回值：方差
    def examine_explore_distribution(self):
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



    # send_query_only：
    # 用途：发出查询，启动当前系统中的一个任务，不指定初始配置
    # 方法：向query_addr发出query_body，启动其中指定的任务
    # 返回值：query_id，会被保存在当前知识库建立者的成员中
    def send_query_only(self):  
        # 发出提交任务的请求
        self.query_body['node_addr']=self.node_addr
        r = self.sess.post(url="http://{}/query/submit_query".format(self.query_addr),
                    json=query_body)
        resp = r.json()
        self.query_id = resp["query_id"]
        print('成功发起query',self.query_id)

        return self.query_id
    
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
        filepath = scheduler_folder_name + '/' + self.kb_name+'/'+dag_name+'/'+'record_data'+'/'+filename
    
        self.fp = open(filepath, 'w', newline='')

        fieldnames = ['n_loop',
                     'frame_id',
                     'all_delay',
                     'obj_n',
                     'bandwidth'
                     ]
        # 得到配置名
        for i in range(0, len(self.task_conf_names)):
            fieldnames.append(self.task_conf_names[i])  

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

            for i in range(0, len(self.task_conf_names)):
                conf_name = self.task_conf_names[i]   #得到配置名
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




    # collect_for_sample_offline_simulator
    # 用途：和collect_for_sample一样用于获得某个配置对应的处理时延，并进行标准差的求解、对文件的写入；但是，它使用offline_simulator里的内容，直接获取结果
    #       这样一来，就不需要真的运行分布式程序，也能获得采样结果了
    # 方法：调用offline_simulator中已有的接口，直接获取ffline_simulator里的“完美知识库”
  
    def collect_for_sample_offline_simulator(self,conf,flow_mapping,resource_limit):
        
        #print('当前采样配置')
        #print(conf)
        #print(flow_mapping)
        #print(resource_limit)
        '''
        work_condition={
            "obj_n": 20,
            "obj_stable": True,
            "obj_size": 300,
            "delay": 0.2  #这玩意包含了传输时延，我不想看
            }

        conf=dict({"reso": "360p", "fps": 30, "encoder": "JPEG"})
        flow_mapping=dict({
            "face_detection": {"model_id": 0, "node_ip": "114.212.81.11", "node_role": "cloud"}, 
            "gender_classification": {"model_id": 0, "node_ip": "192.168.1.7", "node_role": "host"}
            })
        resource_limit=dict({
            "face_detection": {"cpu_util_limit": 1.0, "mem_util_limit": 1.0}, 
            "gender_classification": {"cpu_util_limit": 0.95, "mem_util_limit": 1.0}
            })
        '''

        # 调用离线模拟器来获取服务的处理时延
        json_folder_path = 'offline_simulation'
        offline_simulator = OfflineSimulator(serv_names = serv_names,
                                         service_info_list = service_info_list,
                                         json_folder_path = json_folder_path,
                                         )
        
        # 用于模拟真实场景，离线建库一般用的是只有一张人脸的稳定视频画面
        work_condition={
            "obj_n": 1,
            "obj_stable": True,
            "obj_size": 300,
            "delay": 0.2  
            }
        status, proc_delay_csv = offline_simulator.get_proc_delay_from_csv(conf=conf,
                                                                            flow_mapping=flow_mapping,
                                                                            resource_limit=resource_limit,
                                                                            work_condition = work_condition
                                                                            )
        # proc_delay_csv是一个字典，形如：
        '''
        {
            'face_detection':[1.2,1.3,1.23],
            'gender_classification':[0.5,0.7,0.6]
        }
        '''
        
        # 现在根据模拟结果来设置row。由于这是一个用于模拟的过程，所以处理时延以外的感知结果全部用-1表示。

        row = {}

        row['n_loop'] = -1
        row['frame_id'] = -1
        row['all_delay'] = -1
        row['obj_n'] = -1
        row['bandwidth'] = -1

        for i in range(0, len(self.task_conf_names)):
            conf_name = self.task_conf_names[i]   #得到任务配置名
            row[conf_name] = conf [conf_name]

        # serv_names形如['face_detection', 'face_alignment']
        for i in range(0, len(self.serv_names)):
            serv_name = self.serv_names[i]
            
            serv_role_name = serv_name + '_role'
            serv_ip_name = serv_name + '_ip'
            serv_proc_delay_name = serv_name + '_proc_delay'
            trans_ip_name = serv_name + '_trans_ip'
            trans_delay_name = serv_name + '_trans_delay'
      
            row[serv_role_name] = flow_mapping[serv_name]["node_role"]
            row[serv_ip_name] = flow_mapping[serv_name]['node_ip']
            # 接下来是关键：获取该服务的传输时延
            # print(proc_delay_csv[serv_name])
            row[serv_proc_delay_name] = sum(proc_delay_csv[serv_name])/len(proc_delay_csv[serv_name])

            row[trans_ip_name] = row[serv_ip_name]
            row[trans_delay_name] = -1

            field_name = serv_name + '_cpu_util_limit'
            row[field_name] = resource_limit[serv_name]['cpu_util_limit']
            field_name = serv_name + '_cpu_util_use'
            row[field_name] = -1

            field_name = serv_name + '_mem_util_limit'
            row[field_name] = resource_limit[serv_name]['mem_util_limit']
            field_name = serv_name + '_mem_util_use'
            row[field_name] = -1


        n_loop = -1
        '''
        print('n_loop', n_loop)
        print('处理时延之和：', row['face_detection_proc_delay'] + row['gender_classification_proc_delay'])
        print("face_detection处理时延:", row['face_detection_proc_delay'])
        print("gender_classification处理时延:", row['gender_classification_proc_delay'])
        print('reso:', row['reso'], ' fps:', row['fps'])
        '''
        face_role = 'node'
        if row['face_detection_ip'] == '114.212.81.11':
            face_role = 'cloud'
        gender_role = 'node'
        if row['gender_classification_ip'] == '114.212.81.11':
            gender_role = 'cloud'
        '''
        print('face_detection_ip:', face_role, ' gender_classification_ip:', gender_role)
        print('face_detection资源')

        print('cpu限制', row['face_detection_cpu_util_limit'])
        print('mem限制', row['face_detection_mem_util_limit'])
        print('gender_classification资源')

        print('cpu限制', row['gender_classification_cpu_util_limit'])
        print('mem限制', row['gender_classification_mem_util_limit'])
        print()
        '''
        self.writer.writerow(row)
        print("写入成功")
       

        # (1)得到该配置下的平均时延
        # 前面已经算出了avg_delay
        avg_delay = 0.0
        for serv_name in self.serv_names:
            avg_delay += row[serv_proc_delay_name]


        # (2)得到该配置下的精度
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
       
        
        print("完成对当前配置的一次离线模拟采样，平均时延是", avg_delay,"精度是", task_accuracy)
       
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






    # objective：
    # 用途：作为贝叶斯优化采样时需要优化的目标函数。
    # 返回值：csv的文件名
    def objective(self,trial):
        conf={}
        flow_mapping={}
        resource_limit={}

        #self.available_resource描述了当前可用资源
        for task_conf_name in self.task_conf_names:   #选择任务配置
            # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
            conf[task_conf_name]=trial.suggest_categorical(task_conf_name,conf_and_serv_info[task_conf_name])
        
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
                # 注意，不得不用suggest来解决optuna中同名变量取值范围不能变的问题
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
        return self.examine_explore_distribution()


    # objective_offline_simulator：
    # 用途：作为贝叶斯优化采样时需要优化的目标函数,但是调用了collect_for_sample_offline_simulator，进行离线模拟
    # 返回值：csv的文件名
    def objective_offline_simulator(self,trial):
        conf={}
        flow_mapping={}
        resource_limit={}

        #self.available_resource描述了当前可用资源
        for task_conf_name in self.task_conf_names:   #选择任务配置
            # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
            conf[task_conf_name]=trial.suggest_categorical(task_conf_name,conf_and_serv_info[task_conf_name])
        
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
                # 注意，不得不用suggest来解决optuna中同名变量取值范围不能变的问题
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
        print('选出conf',conf)
        avg_delay= self.collect_for_sample_offline_simulator(conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit)
        # 使用此函数，目标是最优化采样得到的时延
        return self.examine_explore_distribution()


    # 以贝叶斯采样的方式获取离线知识库
    # 相比前一种方法，这种方法首先要对分区的左下角进行若干次采样，然后才进行贝叶斯优化采样
    # 使用该函数之前，必须确保conf_and_serv_info里的内容已经完成调整
    # 注意：如果if_offline为True，使用离线模拟进行采样
    def sample_and_record_bayes_for_section(self,sample_bound,n_trials,bin_nums,delay_range,accuracy_range,if_offline):
        self.sample_bound=sample_bound
        self.bin_nums=bin_nums

        filename,filepath = self.init_record_file()

        # 设置self.explore_conf_dict[conf_str]，初始化一个下限和一个上限（上限和下限都可能无法达到），
        # 这个上限和下限可以通过别的方法来获取，目前先假设通过超参数来设置。比如设置为如下，寻找0和0.7之间均匀分布的采样点
        self.explore_conf_dict["delay"]["min"]=delay_range["min"]
        self.explore_conf_dict["delay"]["max"]=delay_range["max"]
        self.explore_conf_dict["accuracy"]["min"]=accuracy_range["min"]
        self.explore_conf_dict["accuracy"]["max"]=accuracy_range["max"]

        print('开始对分区进行贝叶斯优化采样')
        study = optuna.create_study(directions=['minimize' for _ in range(2)], sampler=optuna.samplers.NSGAIISampler()) 
        if if_offline:
            study.optimize(self.objective_offline_simulator,n_trials=n_trials)
        else:
            study.optimize(self.objective,n_trials=n_trials)

        self.fp.close()
        print("完成对该分区的全部采样,返回filename, filepath,bottom_left_plans,delay_std,accuracy_std")
        delay_std,accuracy_std=self.examine_explore_distribution()
        return filename,filepath,delay_std,accuracy_std

    # 基于区间的遍历采样函数
    # sample_for_kb_sections_traverse_bayes
    # 用途：为一种区间划分方式，进行一系列文件初始化，并在指定的区间里依次进行贝叶斯优化采样（不包括建库）
    # 方法：建立各类文件实现初始化
    #      为了防止中断,遍历各个区间进行采样之前，会忽略已采样的内容

    def sample_for_kb_sections_traverse_bayes(self,conf_sections,section_ids_to_sample,sample_bound,n_trials,bin_nums,delay_range,accuracy_range,if_continue,if_offline):
        '''
         div_conf_sections={
            "reso":{
                "0":["360p","480p","540p"],
                "1":["630p","720p","810p"],
                "2":["900p","990p","1080p"],
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
            },
            "edge_cloud_cut_range":{
                "0":[0],
                "1":[1],
                "2":[2],
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
        # 获取当前用来划分区间的各个配置名，比如['reso','fps','edge_cloud_cut_range']
        self.sec_conf_names = list(conf_sections.keys())
        if not if_continue: #如果不是恢复而是从头开始：
            section_info['des']=datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
            section_info['serv_names']=self.serv_names 
            section_info['sec_conf_names']=self.sec_conf_names #用于划分区间的配置
            section_info['conf_sections']=conf_sections #具体的区间划分方式
            section_info['section_ids']=list([]) #目前已经完成采样的区间编号
        else:
            #否则直接读取已有的section_info信息。从指定的self.kb_name里获取。
            with open(self.kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
                #print('打开知识库section_info:',self.kb_name + '/' + dag_name + '/' + 'section_info.json')
                section_info=json.load(f) 
            f.close()
        
        # （2）进行采样前首先保存好原始的各个配置的取值范围，以供未来恢复    
        original_conf_range={}
        # 遍历用于划分区间的配置
        for sec_conf_name in self.sec_conf_names:
            original_conf_range[ sec_conf_name]=conf_and_serv_info[ sec_conf_name]
     

        # (3)开始对各个分区进行遍历采样
        for section_id in section_ids_to_sample: #遍历有待采样的区间，一个区间编号形如： "reso=0-fps=0-encoder=0"
            
            print('准备处理的分区id是',section_id)

            #(3.1)判断当前分区是否已经被采样
            if section_id in section_info['section_ids']:
                print("该分区已经采样过，忽略")               
            else:
                print('准备对新分区采样,当前分区id为:',section_id)

                #(3.2)根据分区编号重置conf_and_serc_info的值，限定在分区取值范围内进行
                for part in section_id.split('-'):
                    # 使用 split('=') 分割键值对  
                    sec_conf_name, sec_choice = part.split('=') #value形如'0'
                    # 开始重置conf_and_serv_info，以备采样
                    conf_and_serv_info[sec_conf_name]=conf_sections[sec_conf_name][sec_choice]
                    print(sec_conf_name,conf_and_serv_info[sec_conf_name] )
                
                #(3.3)清空已有的采样字典，然后以调用sample_and_record_bayes_for_section在区间内采样
                self.clear_explore_distribution(delay_range=delay_range,accuracy_range=accuracy_range)
                filename,filepath,delay_std,accuracy_std=self.sample_and_record_bayes_for_section(sample_bound=sample_bound,
                                                                                                n_trials=n_trials,bin_nums=bin_nums,
                                                                                                delay_range=delay_range,accuracy_range=accuracy_range,
                                                                                                if_offline = if_offline)
                #(3.4)准备更新section_info信息到目录中
                if section_id not in section_info['section_ids']:
                    section_info['section_ids'].append(section_id)
                if section_id not in section_info.keys():
                    section_info[section_id]={}
                    section_info[section_id]['filenames']=[]
                    
                section_info[section_id]['filenames'].append(filename)

                with open(self.kb_name+'/'+dag_name+'/'+'section_info'+".json", 'w') as f:  
                    json.dump(section_info, f,indent=4) 
                f.close()
                
                print('已在section_info中更新该分区的相关信息')

            
        #最后恢复原始的各个配置的取值，并写入section_info
        for sec_conf_name in self.sec_conf_names:
            conf_and_serv_info[sec_conf_name]=original_conf_range[sec_conf_name]
        
        with open(self.kb_name+'/'+dag_name+'/'+'section_info'+".json", 'w') as f:  
            json.dump(section_info, f,indent=4) 
        f.close()



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
        filepath = scheduler_folder_name + '/' + self.kb_name + '/' + dag_name + '/' + 'record_data' + '/' +filename
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
                for conf_name in self.sec_conf_names:
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


    # get_all_section_ids：
    # 用途：根据conf_sections里的划分，获取这种划分下所有可能的section_id
    def get_all_section_ids(self,conf_sections):

        self.sec_conf_names = list(conf_sections.keys())
        
        conf_sec_select={}
        for conf_name in self.sec_conf_names:
             conf_sec_select[conf_name]=list(conf_sections[conf_name].keys())
        print(conf_sec_select)
        
        '''
        conf_sec_select形如:
        {   
            'reso': ['0', '1', '2'], 
            'fps': ['0', '1', '2', '3', '4', '5'], 
            'edge_cloud_cut_point': ['0', '1', '2']
        }
        '''
        # 现在我要基于conf_sec_select得到所有可能的section_id。

        combinations = list(product(*conf_sec_select.values()))  

        all_section_ids=[]

        for combo in combinations: # combo形如('0', '0', '1')
            section_id=''
            for i in range(0,len(self.sec_conf_names)):
                conf_name = self.sec_conf_names[i]
                if i>0:
                    section_id+='-'
                section_id = section_id + conf_name + '=' + combo[i]
            all_section_ids.append(section_id)
        
        print(all_section_ids)
        return all_section_ids



    #'''
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
        for conf_name in self.task_conf_names:
            if conf_name in conf_draw_dict:
                self.draw_picture(x_value=df['n_loop'], y_value=df[conf_name], title_name=conf_draw_dict[conf_name]['title_name']+"随时间变化图", xlabel='帧数', ylabel=conf_draw_dict[conf_name]['ylabel'])
        
        plt.show()
    #'''

#以上是KnowledgeBaseBuilder类的全部定义，该类可以让使用者在初始化后完成一次完整的知识库创建，并分析采样的结果
#接下来定义一个新的类，作用的基于知识库进行冷启动

# 使用KnowledgeBaseBuilder需要提供以下参数：
# service_info_list描述了构成流水线的所有阶段的服务。下图表示face_detection和face_alignment构成的流水线
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



# 表示流水线上的任务配置
task_conf_names=["reso","fps","encoder"]
# 表示流水线上的资源配置，包括云边协同切分点、设备ip，cpu使用率和内存限制(由于无法细致调节内存限制，此处不处理)
rsc_conf_names=["edge_cloud_cut_point","face_detection_ip","gender_classification_ip",\
                "face_detection_mem_util_limit","face_detection_cpu_util_limit",\
                "gender_classification_mem_util_limit","gender_classification_cpu_util_limit"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","gender_classification"]   


#这个query_body用于测试单位的“人进入会议室”，也就是只有一张脸的情况，工况不变，但是会触发调度器变化，因为ifd很小

#'''
query_body = {
        "node_addr": "192.168.1.7:3001",
        "video_id": 103,     
        "pipeline":  ["face_detection", "gender_classification"],#制定任务类型
        "user_constraint": {
            "delay": 0.3,  # 用户时延约束暂时设置为0.3
            "accuracy": 0.6,  # 用户精度约束暂时设置为0.6
            'rsc_constraint': {  # 注意，没有用到的设备的ip这里必须删除，因为贝叶斯求解多目标优化问题时优化目标的数量是由这里的设备ip决定的
                "114.212.81.11": {"cpu": 1.0, "mem": 1000}, 
                "192.168.1.7": {"cpu": 1.0, "mem": 1000} 
            }
        }
    }  
#'''


if __name__ == "__main__":

    from scheduler_1_wzl.RuntimePortrait import RuntimePortrait

    myportrait = RuntimePortrait(pipeline=serv_names,user_constraint=query_body['user_constraint'])
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
    sec_num = 20
    # 每一种配置下进行多少次采样
    sample_bound = 5
    # 为每一个区间进行贝叶斯优化时采样多少次
    n_trials = 5 #20
    #(将上述三个量相乘，就足以得到要采样的总次数，这个次数与建立知识库所需的时延一般成正比)
    
    # 是否进行稀疏采样(贝叶斯优化)
    need_sparse_kb = 0
    # 是否进行严格采样（遍历所有配置）
    need_tight_kb = 0
    # 是否根据某个csv文件绘制画像 
    need_to_draw = 0
    # 是否需要基于初始采样结果建立一系列字典，也就是时延有关的知识库
    need_to_build = 0
    # 是否需要将某个文件的内容更新到知识库之中
    need_to_add = 0
    # 判断是否需要在知识库中存放新的边缘ip，利用已有的更新
    need_new_ip = 0
    # 开始遍历conf_and_serv_info中的配置
    need_sample_and_record = 0

    #是否需要发起一次简单的查询并测试调度器的功能
    need_to_test = 1

    #是否需要验证一下函数的正确性
    need_to_verify = 0


    task_name = "gender_classify"

    kb_name = 'kb_data_90i90_offline_simu-1'

            
    kb_builder=KnowledgeBaseBuilder(expr_name="gender_classify_offline_simu",
                                    node_ip='192.168.1.7',
                                    node_addr="192.168.1.7:3001",
                                    query_addr="114.212.81.11:3000",
                                    service_addr="114.212.81.11:3500",
                                    query_body=query_body,
                                    task_conf_names=task_conf_names,
                                    rsc_conf_names=rsc_conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    rsc_upper_bound=rsc_upper_bound,
                                    kb_name=kb_name)


    filepath=''



    # 是否需要进行一次普通测试
    if need_to_test==1:
        kb_builder.send_query_only()
        filepath=kb_builder.just_record(record_num=350)

        kb_builder.draw_picture_from_sample(filepath=filepath)


    #建立基于贝叶斯优化的稀疏知识库
    if need_sparse_kb==1:
    
        conf_sections={
            "reso":{
                "0":["360p","480p","540p"],
                "1":["630p","720p","810p"],
                "2":["900p","990p","1080p"],
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
            "edge_cloud_cut_point":{
                "0":[0],
                "1":[1],
                "2":[2],
            }
        }
        # 获得全部区间的编号
        section_ids_to_sample = kb_builder.get_all_section_ids(conf_sections = conf_sections)
        
        if_continue = False
        if_offline = True

        if not if_offline: # 如果通过在线采样建库而非离线采样
            kb_builder.send_query_and_start() 
    
        kb_builder.sample_for_kb_sections_traverse_bayes(conf_sections = conf_sections, # 区间划分方式
                                                   section_ids_to_sample = section_ids_to_sample, # 带采样区间编号
                                                   sample_bound = sample_bound, # 每种配置采样几次（求平均）
                                                   n_trials = n_trials,bin_nums = bin_nums, # 贝叶斯优化总采样次数和标准差间隔
                                                   delay_range = delay_range, accuracy_range = accuracy_range, # 求标准差的时延和精度范畴
                                                   if_continue = if_continue, #是否是从中断中恢复的
                                                   if_offline = if_offline  # 是否需要通过离线模拟来建库
                                                  )
        
    # 关于是否需要建立知识库：可以根据txt文件中的内容来根据采样结果建立知识库
    if need_to_build==1:
        filepath=''
        kb_builder.update_sections_from_files()
        print("完成时延知识库的建立")
        

    if need_sample_and_record == 1:

        '''
        # 研究不同cpu对face_detection服务的影响
        sample_range={  #各种配置参数的可选值
    
            "reso":['360p','720p','1080p'],
            "fps":[30],
            "encoder":['JPEG'],
            
            "face_detection_ip":["192.168.1.7"],
            "gender_classification_ip":["114.212.81.11"],

            "face_detection_cpu_util_limit":[0.05,0.15,0.25,0.35,0.45],
            "gender_classification_cpu_util_limit":[1.0],

        }
        '''
        # 研究不同cpu对gender_classification服务的影响。第二阶段放在边端，第一阶段放在云端。
        sample_range={  #各种配置参数的可选值
    
            "reso":['360p','720p','1080p'],
            "fps":[30],
            "encoder":['JPEG'],
            
            "face_detection_ip":["114.212.81.11"],
            "gender_classification_ip":["192.168.1.7"],

            "face_detection_cpu_util_limit":[1.0],
            "gender_classification_cpu_util_limit":[0.05,0.15,0.25,0.35,0.45],

        }


        kb_builder.send_query_and_start()

        kb_builder.sample_and_record(sample_bound=sample_bound,sample_range=sample_range,delay_range=delay_range,accuracy_range=accuracy_range)

    if need_to_verify==1:
        '''
        conf_sections={
        }
        
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

            #是否需要发起一次简单的查询并测试调度器的功能
    
    
    

    '''
    # 关于是否需要建立知识库：可以根据txt文件中的内容来根据采样结果建立知识库
    if need_to_build==1:
        filepath=''
        kb_builder.update_sections_from_files()
        print("完成时延知识库的建立")
    
    if need_new_ip==1:
        # 此处可以直接修改知识库中的特定配置，比如删除所有特定ip下的配置，或者基于各个边缘端都相同的思想添加新的边缘端ip配置，或者将知识库中的某个ip换成新的ip
        kb_builder.swap_node_ip_in_kb(old_node_ip='172.27.143.164',new_node_ip='192.168.1.7')
        print('完成更新')
    




    # 关于是否需要绘制图像
    if need_to_draw==1:
        print('准备画画')
        filepath='kb_data/20240407_20_52_49_kb_builder_0.3_tight_build_gender_classify_cold_start04.csv'
        kb_builder.anylze_explore_result(filepath=filepath)
        kb_builder.draw_picture_from_sample(filepath=filepath)
    '''


    exit()

    
