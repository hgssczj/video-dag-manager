
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

from common import KB_DATA_PATH,MAX_NUMBER, model_op, conf_and_serv_info, service_info_dict





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


    
    # sample_and_record：
    # 用途：遍历sample_range中的所有配置，对每一种配置进行采样并记录结果。和just_record相对。
    # 方法：初始化一个csv文件，然后生成配置遍历的全排列，对每一种配置都调用collect_for_sample进行采样和记录
    # 返回值：csv的文件名
    def sample_and_record(self,sample_bound,sample_range,delay_range,accuracy_range):


        self.clear_explore_distribution(delay_range=delay_range,accuracy_range=accuracy_range)

        self.sample_bound=sample_bound
        filename,filepath = self.init_record_file()
        #执行一次get_and_write就可以往文件里成果写入数据


        # 现在有self.conf_names方便选取conf和flow_mapping了，可以得到任意组合。
        # 当前实现中conf_names里指定的配置是每一个服务都共享的配置，只有model需要特殊考虑。
        # 所以应该记录下服务的数目然后对应处理
        
        conf_list=[]
        for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
            conf_list.append(sample_range[conf_name])
        # conf_list会包含各类配置参数取值范围，例如分辨率、帧率等

        serv_ip_list=[]
        for serv_name in self.serv_names:
            serv_ip=serv_name+"_ip"
            serv_ip_list.append(sample_range[serv_ip])
        # serv_ip_list包含各个模型的ip的取值范围

        serv_cpu_list=[]
        for serv_name in self.serv_names:
            serv_cpu=serv_name+"_cpu_util_limit"
            serv_cpu_list.append(sample_range[serv_cpu])
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
            self.draw_picture(x_value=x_list,y_value=df[serv_proc_delay_name],title_name=serv_proc_delay_name+"/时间")
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


# 离线模拟采样器，用于根据已有的采样csv文件生成一系列json文件。
# 可以调用其中的接口，模拟任意配置下采样得到的一系列性能。其中，帧率对性能的影响会进行专门处理。
# 
class OfflineSimulator():   
    
    #知识库建立者，每次运行的时候要指定本次实验名称，用于作为记录实验的文件
    def __init__(self,serv_names,service_info_list,json_folder_path):

        self.json_folder_path = json_folder_path
        self.serv_names = serv_names 
        self.service_info_list=service_info_list

    # get_json_from_csv_for_service
    # 用途：对csv_folder_path路径下的若干csv文件，为每一个服务分析其每一种配置下采样得到的若干性能，记录在json_folder_path下的json文件中
    #      比如，这些csv文件里记录了某个服务在某种配置下的5种性能，那么就要将其都记录为列表，然后将列表存储在json文件之中
    # 返回值：无，但是为每一个服务都生成了一个区间的CSV文件
    def get_json_from_csv_for_service(self,csv_folder_path):

        #(1) 遍历csv_folder_path目录下所有的csv文件,读取为df
        for filename in os.listdir(csv_folder_path):  

            csv_filepath = os.path.join(csv_folder_path, filename)  
        
            df = pd.read_csv(csv_filepath)
            # 用于避免配置的重复
            used_set=set()

            # (2) 对于当前的这个csv文件，准备更新每一个服务的json字典
            for service_info in self.service_info_list:
                # 对于每一个服务，首先确定服务有哪些配置，构建conf_list
                service_conf=list(service_info['conf']) # 形如":["reso","fps"]
                # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
                service_conf.append(service_info['name']+'_ip')
                # 然后再添加资源约束，目前只考虑cpu使用率限制 cpu_util_limit
                service_conf.append(service_info['name']+'_cpu_util_limit')
                # 然后再添加资源约束，也就是mem限制 mem_util_limit
                service_conf.append(service_info['name']+'_mem_util_limit')
                #service_conf形如['reso', 'fps', 'encoder', 'face_detection_ip', 'face_detection_cpu_util_limit', 'face_detection_mem_util_limit']
            
                # 这个service_conf会作为json文件键值对中的键

                #(3)在更新这个服务对应的json文件之前，首先确认这个文件是否存在。如果已经存在，那就直接加载旧的内容；否则，创建一个新文件并读取空内容。

                eval_path = self.json_folder_path+'/' + service_info['name'] + '.json'
                evaluator=dict()
                if not ( os.path.exists(eval_path) and os.path.isfile(eval_path) ): #如果文件不存在，创立文件并写入空字典
                    with open(eval_path, 'w') as f:  
                        json.dump(evaluator, f,indent=4) 
                    f.close()
                # 然后读取文件中的原始evaluator，试图更新它
                with open(eval_path, 'r') as f:  
                    evaluator = json.load(f)  
                f.close()

                # (4)从csv文件中读取内容准备写入这个json文件字典

                min_index=df.index.min()
                max_index=df.index.max()
                conf_kind=0 #记录总配置数
                for index in range(min_index,max_index+1):
                    #遍历每一个配置
                    values_list=df.loc[index,service_conf].tolist()
                    #values_list形如['990p', 24, 'JPEG', '114.212.81.11', 1.0, 1.0]，获取了service_conf对应的配置取值
                    dict_key=''
                    for i in range(0,len(service_conf)):
                        dict_key+=service_conf[i]+'='+str(values_list[i])+' '
                    #得到当前行对应的一个配置
                    
                    if dict_key not in used_set:
                        used_set.add(dict_key)
                        #如果该配置尚未被记录，需要求avg_delay
                        condition_all=True  #用于检索字典所需的条件
                        for i in range(0,len(values_list)):
                            condition=df[service_conf[i]]==values_list[i]    #相应配置名称对应的配置参数等于当前配置组合里的配置
                            condition_all=condition_all&condition
                        
                        # 联立所有条件从df中获取对应内容,conf_df里包含满足所有条件的列构成的df
                        conf_df=df[condition_all]
                        if(len(conf_df)>0): #如果满足条件的内容不为空，可以开始用其中的数值来初始化字典
                            conf_kind+=1
                            evaluator[dict_key]=list(conf_df[service_info['value']])

                #完成对该服务的evaluator的处理
                with open(eval_path, 'w') as f:  
                    json.dump(evaluator, f,indent=4) 
                f.close()
                
                print(service_info['name']+"在该csv文件中更新的配置组合数为",conf_kind)


    # get_proc_delay_from_csv
    # 用途：基于get_json_from_csv_for_service的结果，从json文件中评估任意配置下流水线上各个服务的时延
    # 方法：json文件中是以列表的形式存储这些时延的，因此我也直接返回列表好了。注意，对fps进行了专门的按比例处理。
    # 返回值：各个服务的列表形式的处理时延（经过fps修正）
    def get_proc_delay_from_csv(self,conf,flow_mapping,resource_limit,work_condition):

        # csv文件中只存储了30fps的情况，要根据输入配置的帧率做出相应的调整。
        # 此处保留真实fps，并修改参数中的fps为30。
        true_fps = conf["fps"]
        conf["fps"]=30

        # 存储配置对应的各阶段时延，以及总时延
        proc_delay_csv={}
        
        status = 0  # 为0表示字典中没有满足配置的存在
        # 对于service_info_list里的service_info依次评估性能
        for service_info in self.service_info_list:
            # （1）加载服务对应的性能评估器
            f = open(self.json_folder_path+'/'+service_info['name']+".json")  
            evaluator = json.load(f)
            f.close()
            # （2）获取service_info里对应的服务配置参数，从参数conf中获取该服务配置旋钮的需要的各个值，加上ip选择
            # 得到形如["360p"，"1","JPEG","114.212.81.11"]的conf_for_dict，用于从字典中获取性能指标
            service_conf = list(service_info['conf']) # 形如":["reso","fps","encoder"]
                # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
            conf_for_dict = []
                #形如：["360p"，"1","JPEG"]
            
           
            for service_conf_name in service_conf:
                conf_for_dict.append(str(conf[service_conf_name]))   
            

            # 完成以上操作后，conf_for_dict内还差ip地址,首先要判断当前评估器是不是针对传输阶段进行的：
            ip_for_dict_index = service_info['name'].find("_trans") 
            if ip_for_dict_index > 0:
                # 当前是trans，则去除服务名末尾的_trans，形如“face_detection”
                ip_for_dict_name = service_info['name'][0:ip_for_dict_index] 
            else: # 当前不是trans
                ip_for_dict_name = service_info['name']
            ip_for_dict = flow_mapping[ip_for_dict_name]['node_ip']
                
            cpu_for_dict = resource_limit[ip_for_dict_name]["cpu_util_limit"]
            mem_for_dict = resource_limit[ip_for_dict_name]["mem_util_limit"]
            
            conf_for_dict.append(str(ip_for_dict))  
            conf_for_dict.append(str(cpu_for_dict)) 
            conf_for_dict.append(str(mem_for_dict)) 

            service_conf.append(service_info['name']+'_ip')
            service_conf.append(service_info['name']+'_cpu_util_limit')
            service_conf.append(service_info['name']+'_mem_util_limit')
            # 形如["360p"，"1","JPEG","114.212.81.11","0.1"."0.1"]

            # （3）根据conf_for_dict，从性能评估器中提取该服务的评估时延
     
            dict_key=''
            for i in range(0, len(service_conf)):
                dict_key += service_conf[i] + '=' + conf_for_dict[i] + ' '

            
            if dict_key not in evaluator:
                #print('配置不存在',dict_key)
                return status, proc_delay_csv
            
            pred_delay = evaluator[dict_key]
            
            #  (4) 如果pred_delay为0，意味着这一部分对应的性能估计结果不存在，该配置在知识库中没有找到合适的解。此时直接返回结果。
            if pred_delay == 0:
                return status, proc_delay_csv
            
            #  (5)对预测出时延根据工况进行修改
            #  注意，此处获得的pred_delay_list和pred_delay_total里的时延都是单一工况下的，因此需要结合工况进行调整
            obj_n = 1
            if 'obj_n' in work_condition:
                obj_n = work_condition['obj_n']
            if service_info_dict[service_info['name']]["vary_with_obj_n"]:
                pred_delay = [x * obj_n for x in pred_delay ]



            # （6）根据true_fps的取值进行专门的修正
            pred_delay = [x * float(true_fps/30.0) for x in pred_delay ]

            # （7）将预测的时延添加到列表中
            proc_delay_csv[service_info['name']] = pred_delay

        conf['fps']=true_fps #如果不在此处还原，会导致参数本身被直接修改


        status = 1
        return status, proc_delay_csv  # 返回各个部分的时延








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


    from RuntimePortrait import RuntimePortrait

    myportrait = RuntimePortrait(pipeline=serv_names,user_constraint=query_body['user_constraint'])
    #从画像里收集服务在边缘端的资源上限
    # 描述每一种服务所需的中资源阈值，它限制了贝叶斯优化的时候采取怎样的内存取值范围

    rsc_upper_bound = {}
    for serv_name in serv_names:
        serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
        rsc_upper_bound[serv_name] = {}
        rsc_upper_bound[serv_name]['cpu_limit'] = serv_rsc_cons['cpu']['edge']
        rsc_upper_bound[serv_name]['mem_limit'] = serv_rsc_cons['mem']['edge']
    #print("画像提供的资源上限")
    #print(rsc_upper_bound)


    # 贝叶斯优化时的取值范围，在以下范围内使得采样点尽可能平均
    delay_range={
        'min':0.0,
        "max":1.0
    }
    accuracy_range={
        'min':0.0,
        "max":1.0
    }


    # 每一种配置下进行多少次采样
    sample_bound = 5

    task_name = "gender_classify"

    kb_name = 'offline_simulation'


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
    
    csv_folder_path = 'offline_simulation/face_detection-gender_classification/record_data'
    json_folder_path = 'offline_simulation'


    offline_simulator = OfflineSimulator(serv_names = serv_names,
                                         service_info_list = service_info_list,
                                         json_folder_path = json_folder_path,
                                         )
    
    need_sample_and_record = 0

    if need_sample_and_record == 1:

        # 处理人脸检测的10种情况
        # 情况3-人脸检测在540p下随cpu使用率的变化
        # '''
        sample_range={  #各种配置参数的可选值
    
            "reso":['1080p'],
            "fps":[30],
            "encoder":['JPEG'],
            
            "face_detection_ip":["114.212.81.11"],
            "gender_classification_ip":["192.168.1.7"],

            "face_detection_cpu_util_limit":[1.00],
            "gender_classification_cpu_util_limit":[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
                                                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],

        }
        # '''
        '''
        sample_range={  #各种配置参数的可选值
    
            "reso":["360p", "480p", "540p", "630p", "720p", "810p", "900p", "990p", "1080p"],
            "fps":[30],
            "encoder":['JPEG'],
            
            "face_detection_ip":["114.212.81.11"],
            "gender_classification_ip":["114.212.81.11"],

            "face_detection_cpu_util_limit":[1.0],
            "gender_classification_cpu_util_limit":[1.0],

        }
        '''
       

        kb_builder.send_query_and_start()

        filename,filepath = kb_builder.sample_and_record(sample_bound=sample_bound,sample_range=sample_range,delay_range=delay_range,accuracy_range=accuracy_range)

        kb_builder.draw_picture_from_sample(filepath=filepath)


    need_get_json_from_csv = 0

    if need_get_json_from_csv == 1:
    
        offline_simulator.get_json_from_csv_for_service(csv_folder_path = csv_folder_path)

    need_get_proc_delay_from_csv = 1

    if need_get_proc_delay_from_csv == 1:

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
        
        status, proc_delay_csv = offline_simulator.get_proc_delay_from_csv(conf=conf,
                                                                            flow_mapping=flow_mapping,
                                                                            resource_limit=resource_limit,
                                                                            work_condition = work_condition
                                                                            )
        print(status)

        for key in proc_delay_csv:
            print(key, proc_delay_csv[key])