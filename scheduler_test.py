# 该代码用于验证调度器的准确性。它发出一个请求，并记录后续的变化情况。

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
from scheduler_1_wzl.RuntimePortrait import RuntimePortrait
from itertools import product

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
matplotlib.use('TkAgg')

#试图以类的方式将整个独立建立知识库的过程模块化

#本kb_sec_builder_div_test力图把云边协同切分点也作为区间知识库的一个配置


from common import model_op,conf_and_serv_info

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
        filepath =  self.kb_name+'/'+dag_name+'/'+'record_data'+'/'+filename
    
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
                if 'conf_portrait' in portrait_info.keys():
                    print('精度画像是(0-弱  3-强):',portrait_info['conf_portrait'] )
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
    
    # draw_hist：
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
        "video_id": 4,     
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

    
    
    #是否需要发起一次简单的查询并测试调度器的功能
    need_to_test = 1

    task_name = "gender_classify"

    kb_name = 'kb_record_test'
            
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
        filename, filepath=kb_builder.just_record(record_num=100)
        print(filename,filepath)

        kb_builder.draw_picture_from_sample(filepath=filepath)




    exit()

    