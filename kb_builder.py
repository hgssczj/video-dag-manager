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

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
matplotlib.use('TkAgg')

#试图以类的方式将整个独立建立知识库的过程模块化



from common import KB_DATA_PATH,NO_BAYES_GOAL,BEST_ALL_DELAY,BEST_STD_DELAY,model_op,conf_and_serv_info

# 每一个特定任务对应一个KnowledgeBaseBuilder类
class KnowledgeBaseBuilder():   
    
    #知识库建立者，每次运行的时候要指定本次实验名称，用于作为记录实验的文件
    def __init__(self,expr_name,node_ip,node_addr,query_addr,service_addr,query_body,conf_names,serv_names,service_info_list,rsc_upper_bound):

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
        self.bin_nums= None
        self.bayes_goal= NO_BAYES_GOAL

        #(5)通信初始化：用于建立连接
        self.sess = requests.Session()  #客户端通信

        #(6)中资源阈值初始化：描述每一个服务所需的中资源阈值上限
        self.rsc_upper_bound = rsc_upper_bound
 
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

    def set_bayes_goal(self,bayes_goal):    #设置贝叶斯函数的优化目标
        self.bayes_goal=bayes_goal


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
        assert self.bayes_goal
        value_list=[]
        for conf_str in self.explore_conf_dict:
            value=self.explore_conf_dict[conf_str]
            value_list.append(value)
        # print(value_list)
        value_list=np.array(value_list)
        value_list=sorted(value_list)
        min_value=min(value_list) # 当前已经取得的最大采样值
        max_value=max(value_list) # 当前已经取得的最小采样值
        # print(min_value,max_value)
        bins=bins = np.linspace(min_value, max_value, self.bin_nums+1) 
        # print(bins)
        count_bins=np.bincount(np.digitize(value_list,bins=bins))

        return count_bins.std()  #返回方差
    
    # anylze_explore_result:
    # 用途：分析采样结果的均匀分布程度
    # 方法：从指定文件路径中，提取‘all_delay’键对应的内容，得到数组，并绘制直方图，分析其在不同区间的取值分布情况
    # 返回值：无
    def anylze_explore_result(self, filepath):  #分析记录下来的文件结果，也就是采样结果
        df = pd.read_csv(filepath)
        # df = df[df.all_delay<1]

        x_list = [i for i in range(0, len(df))]
        soretd_value = sorted(df['all_delay'])
        satisfy_constraint_num = sum(i <= 0.3 for i in soretd_value)
        satisfy_constraint_rate = satisfy_constraint_num / len(soretd_value)
        print("时延达标率为:{}".format(satisfy_constraint_rate))
        # a, b, c = self.draw_hist(data=soretd_value, title_name='分布', bins=100)
        
        # print(a)
        # print(a.std())
        # print(a.sum())

    # evaluator_init：
    # 用途：为一个服务建立一个由键值对构成的空白字典（空白知识库），以json文件形式保存
    # 方法：从conf_list中按顺序提取涉及的各个配置名称，并构建嵌套字典，初始值全部是0，
    # 返回值：无，但会在当前目录保存一个json文件
    def evaluator_init(self,eval_name):  
        evaluator=dict()
        with open(KB_DATA_PATH+'/'+eval_name+".json", 'w') as f:  
            json.dump(evaluator, f,indent=4) 
        f.close()

        print("完成对评估器",eval_name,"的空白json文件初始化")
    
    # evaluator_load：
    # 用途：读取某个json文件里保存的字典
    # 方法：根据eval_name从当前目录下打开相应json文件并返回
    # 返回值：从文件里读取的字典
    def evaluator_load(self,eval_name):
        with open(KB_DATA_PATH+'/'+eval_name+".json", 'r') as f:  
            evaluator = json.load(f)  
        f.close()
        return evaluator  #返回加载得到的字典
 
    # evaluator_dump：
    # 用途：将字典内容重新写入文件
    # 方法：将参数里的字典写入参数里指定的eval_name文件中
    # 返回值：无
    def evaluator_dump(self,evaluator,eval_name):  #将字典重新写入文件
        with open(KB_DATA_PATH+'/'+eval_name+".json", 'w') as f:  
            json.dump(evaluator, f,indent=4) 
        f.close()

    # send_query：
    # 用途：发出查询，启动当前系统中的一个任务
    # 方法：向query_addr发出query_body，启动其中指定的任务
    # 返回值：query_id，会被保存在当前知识库建立者的成员中
    def send_query(self):  
        # 发出提交任务的请求
        self.query_body['node_addr']=self.node_addr
        r = self.sess.post(url="http://{}/query/submit_query".format(self.query_addr),
                    json=query_body)
        print(r)
        resp = r.json()
        print(resp)
        self.query_id = resp["query_id"]
        return self.query_id
    
    # init_record_file：
    # 用途：初始化一个csv文件，fieldnames包括n_loop','frame_id','all_delay','edge_mem_ratio'，所有配置
    #      以及每一个服务（含中间传输阶段）的ip、时间、cpu和mem相关限制等。未来可能还需要再修改。
    # 方法：根据初始conf_names和serv_names获知所有配置参数和各个服务名
    # 返回值：该csv文件名，同时在当前目录下生成一个csv文件
    def init_record_file(self):
        
        filename = KB_DATA_PATH+'/'+datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + os.path.basename(__file__).split('.')[0] + \
            '_' + str(self.query_body['user_constraint']['delay']) + \
            '_' + self.expr_name + \
            '.csv'

        self.fp = open(filename, 'w', newline='')

        fieldnames = ['n_loop',
                     'frame_id',
                     'all_delay',
                     # 'edge_mem_ratio',
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
            field_name = serv_name+'_trans_data_size'
            fieldnames.append(field_name)

            # 以下用于获取每一个服务对应的cpu资源画像、限制和效果
            # field_name = serv_name + '_cpu_portrait'
            # fieldnames.append(field_name)
            field_name = serv_name + '_cpu_util_limit'
            fieldnames.append(field_name)
            field_name = serv_name + '_cpu_util_use'
            fieldnames.append(field_name)

            
            # 以下用于获取每一个服务对应的cpu资源画像、限制和效果
            # field_name=serv_name+'_trans'+'_cpu_portrait'
            # fieldnames.append(field_name)
            # field_name=serv_name+'_trans'+'_cpu_util_limit'
            # fieldnames.append(field_name)
            # field_name=serv_name+'_trans'+'_cpu_util_use'
            # fieldnames.append(field_name)
            

            # 以下用于获取每一个服务对应的内存资源画像、限制和效果
            # field_name=serv_name+'_mem_portrait'
            # fieldnames.append(field_name)
            field_name = serv_name + '_mem_util_limit'
            fieldnames.append(field_name)
            field_name = serv_name + '_mem_util_use'
            fieldnames.append(field_name)

            # 以下用于获取每一个服务对应的内存资源画像、限制和效果
            # field_name=serv_name+'_trans'+'_mem_portrait'
            # fieldnames.append(field_name)
            # field_name=serv_name+'_trans'+'_mem_util_limit'
            # fieldnames.append(field_name)
            # field_name=serv_name+'_trans'+'_mem_util_use'
            # fieldnames.append(field_name)


        self.writer = csv.DictWriter(self.fp, fieldnames=fieldnames)
        self.writer.writeheader()
        self.written_n_loop.clear() #用于存储各个轮的序号，防止重复记录

        return filename

    # write_in_file：
    # 用途：在csv文件中写入一条执行记录
    # 方法：参数r2r3r4分别表示资源信息、执行回应和runtime_info，从中提取可以填充csv文件的信息，利用字典把所有不重复的感知结果都保存在updatetd_result之中
    # 返回值：updatetd_result，保存了当前的运行时情境和执行结果
    def write_in_file(self,r2,r3,r4):   #pipeline指定了任务类型   
        system_status = r2.json()
        result = r3.json()
        # portrait_info=r4.json()
        
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
                trans_data_size_name = serv_name + '_trans_data_size'

                row[serv_role_name] = res['ext_plan']['flow_mapping'][serv_name]['node_role']
                row[serv_ip_name] = res['ext_plan']['flow_mapping'][serv_name]['node_ip']
                row[serv_proc_delay_name] = res['ext_runtime']['plan_result']['process_delay'][serv_name]
                # row['all_delay'] += row[serv_proc_delay_name]
                row[trans_ip_name] = row[serv_ip_name]
                row[trans_delay_name] = res['ext_runtime']['plan_result']['delay'][serv_name] - row[serv_proc_delay_name]
                row[trans_data_size_name] = res['data_trans_size'][serv_name]
                
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
        # r4 = self.sess.get(url="http://{}/query/get_portrait_info/{}".format(self.query_addr, self.query_id))  
        # #print(r4)
        # if not r4.json():
        #     return {"status":2,"des":"fail to post one query request"}
        '''
        else:
            print("收到运行时情境为:")
            print(r4.json())
        '''
        
        # 如果r1 r2 r3都正常
        updatetd_result=self.write_in_file(r2=r2,r3=r3,r4=None)

        return {"status":3,"des:":"succeed to record a row","updatetd_result":updatetd_result}
    
    # get_write：
    # 用途：感知运行时情境，并调用write_in_file记录在csv文件中。相比post_get_write，不会修改调度计划。
    # 方法：依次获取运行时情境
    #      使用write_in_file方法将感知到的结果写入文件之中
    # 返回值：包含updated_result的键值对
    def get_write(self):
        # print("In get_write")
        #（1）获取资源情境,获取node_ip指定的边缘节点的内存使用率
        r2 = self.sess.get(url="http://{}/get_system_status".format(self.service_addr))
        if not r2.json():
            return {"status":1,"des":"fail to request /get_system_status"}
        '''
        else:
            print("收到资源情境为:")
            print(r2.json())
        '''
          
        #（2）查询执行结果并处理
        r3 = self.sess.get(url="http://{}/query/get_result/{}".format(self.query_addr, self.query_id))  
        if not r3.json():
            return {"status":2,"des":"fail to request /query/get_result"}
        
        # (4) 查看当前运行时情境
        r4 = self.sess.get(url="http://{}/query/get_portrait_info/{}".format(self.query_addr, self.query_id))  
        #print("r4",r4)
        if not r4.json():
            return {"status":2,"des":"fail to request /query/get_portrait_info"}
        '''
        else:
            print("收到运行时情境为:")
            print(r4.json())
        '''
        # 如果r1 r2 r3都正常
        updatetd_result=self.write_in_file(r2=r2,r3=r3,r4=r4)

        return {"status":3,"des":"succeed to record a row","updatetd_result":updatetd_result}

    # collect_for_sample：
    # 用途：获取特定配置下的一系列采样结果，并将平均结果记录在explore_conf_dict贝叶斯字典中
    # 方法：在指定参数的配置下，反复执行post_get_write获取sample_bound个不重复的结果，并计算平均用时avg_delay
    #      之后将当前配置作为键、avg_delay作为值，以键值对的形式将该配置的采样结果保存在字典中，以供贝叶斯优化时的计算   
    # 返回值：当前conf,flow_mapping,resource_limit所对应的sample_bound个采样结果的平均时延
    def collect_for_sample(self, conf, flow_mapping, resource_limit):
        sample_num=0
        sample_result=[]
        all_delay=0
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
        print("完成对当前配置的",self.sample_bound,"次采样，平均时延是", avg_delay)
        #融合所有配置，得到一个表述当前调度策略的唯一方案
        conf_str=json.dumps(conf)+json.dumps(flow_mapping)+json.dumps(resource_limit)  
        print(conf_str,avg_delay)
        # 仅在贝叶斯优化目标是建立稀疏知识库的情况下才会将内容保存在字典中，目的是为了求标准差
        if self.bayes_goal!=NO_BAYES_GOAL:
            if avg_delay>=self.explore_conf_dict['min'] and avg_delay<=self.explore_conf_dict['max']:
                # 初始，explore_conf_dict会被设置一个区间，即min和max对应的内容。如果要建立稀疏知识库，那应该令这一区间内的分布尽可能均匀
                self.explore_conf_dict[conf_str]=avg_delay  #将所有结果保存在字典里

        return avg_delay
    
    # just_record：
    # 用途：不进行任何配置指定，单纯从当前正在执行的系统中进行record_num次采样。其作用和sample_and_rescord（采样并记录）相对。
    # 方法：初始化一个csv文件，进行record_num次调用get_write并记录当前运行时情境和执行结果，期间完全不控制系统的调度策略
    # 返回值：csv的文件名
    def just_record(self,record_num):
        filename = self.init_record_file()
        record_sum = 0
        while(record_sum < record_num):
            get_resopnse = self.get_write()
            if get_resopnse['status'] == 3:
                updatetd_result = get_resopnse['updatetd_result']
                for i in range(0, len(updatetd_result)):
                    # print(updatetd_result[i])
                    record_sum += 1
            else:
                pass
                # print(get_resopnse['des'])

        self.fp.close()
        print("记录结束，查看文件")
        return filename 
    
    # sample_and_record：
    # 用途：遍历conf_list、ip、cpu和mem所有配置，对每一种配置进行采样并记录结果。和just_record相对。
    # 方法：初始化一个csv文件，然后生成配置遍历的全排列，对每一种配置都调用collect_for_sample进行采样和记录
    # 返回值：csv的文件名
    def sample_and_record(self,sample_bound):
        self.sample_bound=sample_bound
        filename=self.init_record_file()
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

        serv_mem_list=[]
        for serv_name in self.serv_names:
            serv_mem=serv_name+"_mem_util_limit"
            serv_mem_list.append(conf_and_serv_info[serv_mem])
        # serv_cpu_list包含各个模型的cpu使用率的取值范围

        # 为了简单起见，暂时只关注0.1，0.2,0.3的取值范围，而且应该从大到小，这样方便计算;同时简化对ip的选择
        '''
        serv_cpu_list=[[0.3],[0.3]]
        serv_cpu_list=[[0.3],[0.2]]
        serv_cpu_list=[[0.3],[0.1]]
        serv_cpu_list=[[0.2],[0.3]]
        serv_cpu_list=[[0.2],[0.2]]
        serv_cpu_list=[[0.2],[0.1]]
        serv_cpu_list=[[0.1],[0.3]]
        serv_cpu_list=[[0.1],[0.2]]
        serv_cpu_list=[[0.1],[0.1]]
        '''

        conf_combine=itertools.product(*conf_list)
        for conf_plan in conf_combine:
            serv_ip_combine=itertools.product(*serv_ip_list)
            for serv_ip_plan in serv_ip_combine:# 遍历所有配置和卸载策略组合
                serv_cpu_combind=itertools.product(*serv_cpu_list)
                for serv_cpu_plan in serv_cpu_combind: 
                    serv_mem_combind=itertools.product(*serv_mem_list)
                    for serv_mem_plan in serv_mem_combind: 
                        conf={}
                        flow_mapping={}
                        resource_limit={}
                        for i in range(0,len(self.conf_names)):
                            conf[self.conf_names[i]]=conf_plan[i]
                        for i in range(0,len(self.serv_names)):
                            flow_mapping[self.serv_names[i]]=model_op[serv_ip_plan[i]]
                            resource_limit[self.serv_names[i]]={}
                            resource_limit[self.serv_names[i]]["cpu_util_limit"]=serv_cpu_plan[i]
                            resource_limit[self.serv_names[i]]["mem_util_limit"]=serv_mem_plan[i]
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
        return filename 
        
    # objective：
    # 用途：作为贝叶斯优化采样时需要优化的目标函数。
    # 方法：根据bayes_goal的取值，选择优化目标，可能是使当前采样结果最小，也可能是为了求平均值。
    # 返回值：csv的文件名
    def objective(self,trial):
        conf={}
        flow_mapping={}
        resource_limit={}

        for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
            # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
            conf[conf_name]=trial.suggest_categorical(conf_name,conf_and_serv_info[conf_name])
 
        for serv_name in self.serv_names:
            serv_ip=serv_name+"_ip"
            
            flow_mapping[serv_name]=model_op[trial.suggest_categorical(serv_ip,conf_and_serv_info[serv_ip])]
            
            serv_cpu_limit=serv_name+"_cpu_util_limit"
            serv_mem_limit=serv_name+"_mem_util_limit"
            resource_limit[serv_name]={}
            if flow_mapping[serv_name]["node_role"] =="cloud":  #对于云端没必要研究资源约束下的情况
                resource_limit[serv_name]["cpu_util_limit"]=1.0
                resource_limit[serv_name]["mem_util_limit"]=1.0
            else:
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
                cpu_select=[]
                cpu_select=[x for x in conf_and_serv_info[serv_cpu_limit] if x <= self.rsc_upper_bound[serv_name]['cpu_limit']]

                mem_select=[]
                mem_select=[x for x in conf_and_serv_info[serv_mem_limit] if x <= self.rsc_upper_bound[serv_name]['mem_limit']]

                
                resource_limit[serv_name]["cpu_util_limit"]=trial.suggest_categorical(serv_cpu_limit,cpu_select)
                resource_limit[serv_name]["mem_util_limit"]=trial.suggest_categorical(serv_mem_limit,mem_select)
        
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

        if self.bayes_goal==BEST_ALL_DELAY:
            return avg_delay
        elif self.bayes_goal==BEST_STD_DELAY:
            return self.exmaine_explore_distribution()


    # 以贝叶斯采样的方式获取离线知识库
    #  kb_builder.sample_and_record_bayes(sample_bound=10,n_trials=80,bin_nums=100,bayes_goal=BEST_ALL_DELAY)
    def sample_and_record_bayes(self,sample_bound,n_trials,bin_nums,bayes_goal,min_val,max_val):

        print("开始进行基于贝叶斯优化的稀疏知识库建立")

        self.sample_bound=sample_bound
        self.bin_nums=bin_nums
        self.bayes_goal=bayes_goal

        filename=self.init_record_file()

        # 设置self.explore_conf_dict[conf_str]，初始化一个下限和一个上限（上限和下限都可能无法达到），
        # 这个上限和下限可以通过别的方法来获取，目前先假设通过超参数来设置。比如设置为如下，寻找0和0.7之间均匀分布的采样点
        self.explore_conf_dict["min"]=min_val
        self.explore_conf_dict["max"]=max_val

        #执行一次get_and_write就可以往文件里成果写入数据
        study = optuna.create_study()
        study.optimize(self.objective,n_trials=n_trials)

        print(study.best_params)

        self.fp.close()
        print("记录结束，查看文件")
        return filename
    
    
    # 从采样结果（得到的配置文件）中收集数据用于床在evaluator评估器.service_info_list包含了要被建立评估器的流水线上的每一个服务的相关信息
    '''
    service_info_list内的元素描述了服务的名称、要评估的性能名称在配置文件里的名字、影响到该服务性能指标的配置,但是不包含卸载策略

    conf_and_serv_info: #各种配置参数的可选值,根据service_info_list里的配置名可以在conf_and_serv_info找到配置的可选范围

    '''
    # create_evaluator_from_samples：
    # 用途：根据记录的日志文件建立字典，每一个配置对应一个平均值
    # 方法：首先为每一个服务用evaluator_init建立一个字典，然后读取filepath参数指定的文件填充这些字典
    # 返回值：无，但是会得到各个服务阶段的字典

    def create_evaluator_from_samples(self,filepath):
        df = pd.read_csv(filepath)
        used_set=set()
        # 首先，完成对每一个服务的性能评估器的初始化,并提取其字典，加入到性能评估器列表中
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
           

            # 基于得到的conf_list来构建该服务的性能评估器，记为evaluator
            self.evaluator_init(eval_name=service_info["name"])
            evaluator=self.evaluator_load(eval_name=service_info["name"])

            min_index=df.index.min()
            max_index=df.index.max()
            conf_kind=0 #记录总配置数
            for index in range(min_index,max_index+1):
                #遍历每一个配置
                values_list=df.loc[index,service_conf].tolist()
                #values_list形如['990p', 24, 'JPEG', '114.212.81.11', 1.0, 1.0]
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
                        avg_value=conf_df[service_info['value']].mean()  #获取均值
                        evaluator[dict_key]=avg_value

            #完成对该服务的evaluator的处理
            self.evaluator_dump(evaluator=evaluator,eval_name=service_info['name'])
            print("该服务",service_info['name'],"涉及配置组合总数",conf_kind)

    # update_evaluator_from_samples：
    # 用途：根据记录的日志文件更新字典，每一个配置对应一个平均值
    # 方法：然后读取filepath参数指定的文件，更新已经存在的字典，不会重新建立
    # 返回值：无，但是会得到被更新的字典

    
    def update_evaluator_from_samples(self,filepath):
        df = pd.read_csv(filepath)
        used_set=set()
        # 首先，完成对每一个服务的性能评估器的初始化,并提取其字典，加入到性能评估器列表中
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
           

            # 不初始化，直接加载
            evaluator=self.evaluator_load(eval_name=service_info["name"])

            min_index=df.index.min()
            max_index=df.index.max()
            conf_kind=0 #记录总配置数
            for index in range(min_index,max_index+1):
                #遍历每一个配置
                values_list=df.loc[index,service_conf].tolist()
                #values_list形如['990p', 24, 'JPEG', '114.212.81.11', 1.0, 1.0]
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
                        avg_value=conf_df[service_info['value']].mean()  #获取均值
                        evaluator[dict_key]=avg_value

            #完成对该服务的evaluator的处理
            self.evaluator_dump(evaluator=evaluator,eval_name=service_info['name'])
            print("该服务",service_info['name'],"涉及配置组合总数",conf_kind)


    # create_conf_info_from_samples：
    # 用途：根据记录的日志文件，建立专属于每一个service服务的conf_and_serv_info
    # 方法：首先为每一个服务建立一个字典，然后读取filepath参数指定的文件填充这些字典，最后保存为json形式
    # 返回值：无，但是会得到各个服务阶段中各个配置的具体参数取值范围
    def create_conf_info_from_samples(self,filepath):
        df = pd.read_csv(filepath)
        # 首先，完成对每一个服务的性能评估器的初始化,并提取其字典，加入到性能评估器列表中
        for service_info in self.service_info_list:
            # 对于每一个服务，首先确定服务有哪些配置，构建conf_list
            service_conf=list(service_info['conf']) # 形如":["reso","fps"]
            # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
            service_conf.append(service_info['name']+'_ip')
            # 然后再添加资源约束，目前只考虑cpu使用率限制 cpu_util_limit
            service_conf.append(service_info['name']+'_cpu_util_limit')
            # 然后再添加资源约束，也就是mem限制 mem_util_limit
            service_conf.append(service_info['name']+'_mem_util_limit')

            conf_info=dict()

            for conf_name in service_conf: #conf_and_serv_info中包含了service_conf中所有配置的可能取值
                conf_info[conf_name]=list(set(df[conf_name].to_list()))
            
            #现在字典里存储着有关该服务每一个配置在当前文件夹中的所有取值，将其存入
            with open(KB_DATA_PATH+'/'+ service_info['name']+'_conf_info'+'.json', 'w') as f:  
                json.dump(conf_info, f)
    
    # update_conf_info_from_samples：
    # 用途：根据记录的日志文件，更新专属于每一个service服务的conf_and_serv_info
    # 方法：首先读取该服务的conf字典，然后读取filepath参数指定的文件更新这些字典，最后保存为json形式
    # 返回值：无，但是会得到各个服务阶段中各个配置的具体参数取值范围
    def update_conf_info_from_samples(self,filepath):
        df = pd.read_csv(filepath)
        # 首先，完成对每一个服务的性能评估器的初始化,并提取其字典，加入到性能评估器列表中
        for service_info in self.service_info_list:
            # 对于每一个服务，首先确定服务有哪些配置，构建conf_list
            service_conf=list(service_info['conf']) # 形如":["reso","fps"]
            # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
            service_conf.append(service_info['name']+'_ip')
            # 然后再添加资源约束，目前只考虑cpu使用率限制 cpu_util_limit
            service_conf.append(service_info['name']+'_cpu_util_limit')
            # 然后再添加资源约束，也就是mem限制 mem_util_limit
            service_conf.append(service_info['name']+'_mem_util_limit')

            conf_info=dict()

            with open(KB_DATA_PATH+'/'+ service_info['name']+'_conf_info'+'.json', 'r') as f:  
                old_conf_info = json.load(f)  
                #print(old_conf_info)

                for conf_name in service_conf: #conf_and_serv_info中包含了service_conf中所有配置的可能取值
                    temp_list=old_conf_info[conf_name]+(df[conf_name].to_list())
                    #print(temp_list)
                    conf_info[conf_name]=list(set(temp_list))
            
            #现在字典里存储着有关该服务每一个配置在当前文件夹中的所有取值，将其存入
            with open(KB_DATA_PATH+'/'+ service_info['name']+'_conf_info'+'.json', 'w') as f:  
                json.dump(conf_info, f) 
    
    # add_node_ip_in_kb：
    # 用途：默认知识库中所有的边缘端性能一样，用已有的边缘端ip对应配置作为新边缘端ip的对应配置，进一步丰富知识库
    # 方法：读取知识库中所有的配置组合，然后将其中设计到边缘端的那些修改为新ip，然后加入知识库
    # 返回值：无，但是会得到全新的、含有新边缘端ip的知识库
    def add_node_ip_in_kb(self,new_node_ip):

        for service_info in self.service_info_list:
            evaluator=self.evaluator_load(eval_name=service_info["name"])
            new_evaluator=dict()
            for dict_key in evaluator.keys():
                conf_key=str(dict_key)
                conf_ip=str(service_info["name"])+'_ip='
                pattern = rf"{re.escape(conf_ip)}([^ ]*)" 
                match = re.search(pattern, conf_key)
                if match:
                    old_ip=match.group(1)
                    # 首先判断字典中已有的配置是不是边缘节点，如果是才考虑下一步
                    if model_op[old_ip]['node_role']=='host':
                        new_conf_key = re.sub(pattern, f"{conf_ip}{new_node_ip}", conf_key) 
                        if new_conf_key not in evaluator:
                            new_evaluator[new_conf_key]=evaluator[conf_key]

            merged_dict = {**evaluator, **new_evaluator}  
            #完成对该服务的evaluator的处理
            self.evaluator_dump(evaluator= merged_dict,eval_name=service_info['name'])


            # 根据情况调整conf_info.json
            with open(KB_DATA_PATH+'/'+ service_info['name']+'_conf_info'+'.json', 'r') as f:  
                old_conf_info = json.load(f)  
                #print(old_conf_info)
                ip_list=list(old_conf_info[service_info["name"]+"_ip"])
                if new_node_ip not in ip_list:
                    ip_list.append(new_node_ip)
                old_conf_info[service_info["name"]+"_ip"]=ip_list

            
            #现在字典里存储着有关该服务每一个配置在当前文件夹中的所有取值，将其存入
            with open(KB_DATA_PATH+'/'+ service_info['name']+'_conf_info'+'.json', 'w') as f:  
                json.dump(old_conf_info, f) 
        
        print('完成对知识库中特定ip配置的增添')

    # swap_node_ip_in_kb：
    # 用途：将知识库中指定的ip替换为另一种ip
    # 方法：读取知识库中所有的配置组合，然后将其中的old_node_ip部分换成new_node_ip
    # 返回值：无
    def swap_node_ip_in_kb(self,old_node_ip,new_node_ip):

        for service_info in self.service_info_list:
            evaluator=self.evaluator_load(eval_name=service_info["name"])
            new_evaluator=dict()
            keys_to_remove=[]
            for dict_key in evaluator.keys():
                conf_key=str(dict_key)
                conf_ip=str(service_info["name"])+'_ip='
                pattern = rf"{re.escape(conf_ip)}([^ ]*)" 
                match = re.search(pattern, conf_key)
                if match:
                    old_ip=match.group(1)
                    # 首先判断字典中已有的配置是不是边缘节点，如果是才考虑下一步
                    if old_ip==old_node_ip:
                        keys_to_remove.append(conf_key)
                        new_conf_key = re.sub(pattern, f"{conf_ip}{new_node_ip}", conf_key) 
                        if new_conf_key not in evaluator:
                            new_evaluator[new_conf_key]=evaluator[conf_key]
            
            for key in keys_to_remove:  
                del evaluator[key]  
                        
            merged_dict = {**evaluator, **new_evaluator}  
            #完成对该服务的evaluator的处理
            self.evaluator_dump(evaluator= merged_dict,eval_name=service_info['name'])


            # 根据情况调整conf_info.json
            with open(KB_DATA_PATH+'/'+ service_info['name']+'_conf_info'+'.json', 'r') as f:  
                old_conf_info = json.load(f)  
                #print(old_conf_info)
                ip_list=list(old_conf_info[service_info["name"]+"_ip"])
                if new_node_ip not in ip_list:
                    ip_list.append(new_node_ip)
                # 增加新ip的同时删除老ip
                new_ip_list = [ip for ip in ip_list if ip != old_node_ip]  
                old_conf_info[service_info["name"]+"_ip"]=new_ip_list

            #现在字典里存储着有关该服务每一个配置在当前文件夹中的所有取值，将其存入
            with open(KB_DATA_PATH+'/'+ service_info['name']+'_conf_info'+'.json', 'w') as f:  
                json.dump(old_conf_info, f) 
        print('完成对知识库中特定ip配置的替换')
        
    # delete_node_ip_in_kb：
    # 用途：删除知识库中的某些ip
    # 方法：将含有参数字符串在内的键值对全部消除,同时重写conf_info文件
    # 返回值：无
    def delete_node_ip_in_kb(self,old_node_ip):
        for service_info in self.service_info_list:
            # 不初始化，直接加载
            evaluator=self.evaluator_load(eval_name=service_info["name"])
            keys_to_remove = [k for k in evaluator if old_node_ip in k]  
            for key in keys_to_remove:  
                del evaluator[key]  
            #完成对该服务的evaluator的处理
            self.evaluator_dump(evaluator= evaluator,eval_name=service_info['name'])

            #之后要把这个ip也从conf_info里删除
            # 根据情况调整conf_info.json
            with open(KB_DATA_PATH+'/'+ service_info['name']+'_conf_info'+'.json', 'r') as f:  
                old_conf_info = json.load(f)  
                #print(old_conf_info)
                ip_list=list(old_conf_info[service_info["name"]+"_ip"])
                new_ip_list = [ip for ip in ip_list if ip != old_node_ip]  
                old_conf_info[service_info["name"]+"_ip"]=new_ip_list

            #现在字典里存储着有关该服务每一个配置在当前文件夹中的所有取值，将其存入
            with open(KB_DATA_PATH+'/'+ service_info['name']+'_conf_info'+'.json', 'w') as f:  
                json.dump(old_conf_info, f) 
        print('完成对知识库中特定ip配置的删除')



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

   
    # find_non_zero_element_rand：
    # 用途：从numbers列表中随机找到一个不为0的值，以及其索引
    # 方法：先提取列表中所有大于0的元素，将其索引和取值构建为新列表；然后从新列表中随机抽取元素
    # 返回值：如果不存在不为0的值，就返回-1，-1；否则返回随机正数在原列表中的索引和取值
    def find_non_zero_element_rand(self, numbers): 
  
        posi_index_num=[]
        for i in range(0,len(numbers)):
            if numbers[i]>0:
                posi_index_num.append([i,numbers[i]])
        
        if len(posi_index_num)==0:
            return -1,-1
        else:
            res=random.choice(posi_index_num) 
            return res[0],res[1]
    
    # find_non_zero_element_max：
    # 用途：从numbers列表中找到一个最大的不为0的值，以及其索引，
    # 方法：先提取列表中最大元素的索引和值，判断是不是0
    # 返回值：如果不存在不为0的值，就返回-1，-1；否则返回随机正数在原列表中的索引和取值
    def find_non_zero_element_max(self, numbers): 
  
       max_element=max(numbers)
       max_idx=numbers.index(max_element)

       if max_element==0:
           return -1,-1
       else:
           return max_idx,max_element
    

    # get_comb_list_dec_max
    # 用途：得到一系列描述各个服务内存限制的元组，但保证每一个元素都降序排列。
    #      例如，value_range形如[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]，假设有2个服务，那么一共有100种描述资源限制的配置组合
    #      由于资源限制只有在由大到小取的时候才有效，因此采样时所使用的元组中每一个元素都必须不断减小
    #      比如（0.3，0.3）到（0.3，0.2）到（0.2，0.2）到（0.2，0.1）到（0.1，0.1）。而（0.3，0.1）到（0.2，0.9）就不行，因为限制升高了
    # 方法：我设计了一个非常巧妙的方法来解决问题。dec_max表示每次下降时选择列表里最大的元素。
    def get_comb_list_dec_max(self):

        # (1)得到离散取值序列从小到大的排列结果
        value_lists=[]
        comb_item=[]
        for serv_name in self.serv_names:
            # 从而获取了每一个服务中，可选区内存限制值构成的列表（含其中资源上限，且保证递增）
            temp_value_list=sorted([value for value in conf_and_serv_info[serv_name+'_mem_util_limit'] if value<=self.rsc_upper_bound[serv_name]['mem_limit']])
            print(temp_value_list)
            value_lists.append(temp_value_list)
            # 初始资源限制组合都是各个服务的中资源阈值
            comb_item.append(len(temp_value_list)-1)
                
        #初始化即将存储递减组合的列表
        comb_list_dec=[]
        comb_list_dec.append(copy.deepcopy(comb_item))


        # 从目前的comb_item里选取一个非0且最大的元素，以及其idx，减去1；如果不为0就继续选。
        idx,element=self.find_non_zero_element_max(comb_item)
        while idx!=-1:# 只要当前的comb_item还有继续下降的空间，就应该进行一次下降，然后把下降后的结果保存到comb_list_dec里面
            comb_item[idx]-=1
            comb_list_dec.append(copy.deepcopy(comb_item))
            idx,element=self.find_non_zero_element_max(comb_item)
        # 循环结束的时候一定是因为全部是0了
        for i in range(0,len(comb_list_dec)): #现在开始将索引转化为内容
            for j in range(0,len(comb_list_dec[i])):
                temp_idx=int(comb_list_dec[i][j])   #形如（2，2）中的2都是索引，需要通过value_list转化为真正的值
                comb_list_dec[i][j]=value_lists[j][temp_idx]
        
        return comb_list_dec
    

    # get_comb_list_dec_rand
    # 用途：同上，只不过下降时是随机的
    # 方法：我设计了一个非常巧妙的方法来解决问题。以下方式通过“随机下降”来解决问题。dec_rand表示每次下降时选择列表里随机一个元素
    def get_comb_list_dec_rand(self):
        # (1)得到离散取值序列从小到大的排列结果
        value_lists=[]
        comb_item=[]
        for serv_name in self.serv_names:
            # 从而获取了每一个服务中，可选区内存限制值构成的列表（含其中资源上限，且保证递增）
            temp_value_list=sorted([value for value in conf_and_serv_info[serv_name+'_mem_util_limit'] if value<=self.rsc_upper_bound[serv_name]['mem_limit']])
            value_lists.append(temp_value_list)
            # 初始资源限制组合都是各个服务的中资源阈值
            comb_item.append(len(temp_value_list)-1)
                
        #初始化即将存储递减组合的列表
        comb_list_dec=[]
        comb_list_dec.append(copy.deepcopy(comb_item))

        # 从目前的comb_item里选取一个非0且最大的元素，以及其idx，减去1；如果不为0就继续选。
        idx,element=self.find_non_zero_element_rand(comb_item)
        while idx!=-1:# 只要当前的comb_item还有继续下降的空间，就应该进行一次下降，然后把下降后的结果保存到comb_list_dec里面
            comb_item[idx]-=1
            comb_list_dec.append(copy.deepcopy(comb_item))
            idx,element=self.find_non_zero_element_rand(comb_item)
        # 循环结束的时候一定是因为全部是0了
        for i in range(0,len(comb_list_dec)): #现在开始将索引转化为内容
            for j in range(0,len(comb_list_dec[i])):
                temp_idx=int(comb_list_dec[i][j])   #形如（2，2）中的2都是索引，需要通过value_list转化为真正的值
                comb_list_dec[i][j]=value_lists[j][temp_idx]
        
        return comb_list_dec

    # get_combs_rand
    # 用途：从以上函数中得到的omb_list_dec中，以随机的方式选取comb_num个组合构建新的列表
    def get_combs_rand(self,comb_list_dec,comb_num):
        # 如果comb_num太大就直接返回原来的comb_list_dec
        if len(comb_list_dec)<comb_num:
            return comb_list_dec  
        # 否则从中随机取样
        random_idx_list = random.sample(list(range(0,len(comb_list_dec))), comb_num)
        random_idx_list.sort()
        random_list=[]
        for idx in random_idx_list:
            random_list.append(comb_list_dec[idx])
        return random_list


    # get_combs_cert
    # 用途：从以上函数中得到的omb_list_dec中，以等距的方式选取comb_num个组合构建新的列表
    def get_combs_cert(self,comb_list_dec,comb_num):
        # 如果comb_num太大就直接返回原来的comb_list_dec
        if len(comb_list_dec)<comb_num:
            return comb_list_dec  
        # 否则
        idx=0
        num=0
        certain_list=[]
        while idx < len(comb_list_dec) and num<comb_num:
            certain_list.append(comb_list_dec[idx])
            idx+=int(len(comb_list_dec)/comb_num)
            num+=1
        return certain_list


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
    # {
    #     "name":'face_detection_trans',
    #     "value":'face_detection_trans_delay',
    #     "conf":["reso","fps","encoder"]
    # },
    # {
    #     "name":'gender_classification_trans',
    #     "value":'gender_classification_trans_delay',
    #     "conf":["reso","fps","encoder"]
    # },
]



# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","gender_classification"]   


#这个query_body用于测试单位的“人进入会议室”，也就是只有一张脸的情况，工况不变，但是会触发调度器变化，因为ifd很小
#'''
query_body = {
        "node_addr": "192.168.1.9:4001",
        "video_id": 4,     
        "pipeline":  ["face_detection", "gender_classification"],#制定任务类型
        "user_constraint": {
            "delay": 0.3,  # 用户时延约束暂时设置为0.3
            "accuracy": 0.6,  # 用户精度约束暂时设置为0.6
            'rsc_constraint': {  # 注意，没有用到的设备的ip这里必须删除，因为贝叶斯求解多目标优化问题时优化目标的数量是由这里的设备ip决定的
                "114.212.81.11": {"cpu": 1.0, "mem": 1000}, 
                "192.168.1.9": {"cpu": 1.0, "mem": 1000}
            }
        }
    }  
#'''


if __name__ == "__main__":

    from RuntimePortrait import RuntimePortrait

    myportrait = RuntimePortrait(pipeline=serv_names, user_constraint=query_body['user_constraint'])
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
    min_val = 0.0
    max_val = 1.0

    # 在多大范围内取方差
    bin_nums = 100

    # 取多少个典型内存限制进行贝叶斯优化
    comb_num = 10 #10
    # 每一种配置下进行多少次采样
    sample_bound = 5
    # 每一次贝叶斯优化时尝试多少次
    n_trials = 20 #20
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

    #是否需要发起一次简单的查询并测试调度器的功能
    need_to_test = 1

    #获取内存资源限制列表的时候，需要两步，第一步是下降，第二部是采取，两种方法都可以随机，也都可以不随机
    dec_rand = 0
    sel_rand = 0 

    task_name = "gender_classify"

    record_name=KB_DATA_PATH+'/'+'0_'+datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')+task_name+"_"+"bayes"+str(need_sparse_kb)+\
              "dec_rand"+str(dec_rand)+"sel_rand"+str(sel_rand)+"mem_num"+str(comb_num)+\
              "min_val"+str(min_val)+"max_val"+str(max_val)+\
              "bin_nums"+str(bin_nums)+"sample_bound"+str(sample_bound)+"n_trials"+str(n_trials)
              

    kb_builder=KnowledgeBaseBuilder(expr_name="tight_build_gender_classify_cold_start04",
                                    node_ip='192.168.1.9',
                                    node_addr="192.168.1.9:4001",
                                    query_addr="114.212.81.11:4000",
                                    service_addr="114.212.81.11:4500",
                                    query_body=query_body,
                                    conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    rsc_upper_bound=rsc_upper_bound)
    filepath=''
    #是否需要发起一次简单的查询并测试调度器的功能
    if need_to_test==1:
        kb_builder.send_query() 
        filepath=kb_builder.just_record(record_num=200)

        kb_builder.anylze_explore_result(filepath=filepath)
        kb_builder.draw_picture_from_sample(filepath=filepath)

    # 关于是否需要绘制图像
    if need_to_draw==1:
        print('准备画画')
        filepath='kb_data/20240407_20_52_49_kb_builder_0.3_tight_build_gender_classify_cold_start04.csv'
        kb_builder.anylze_explore_result(filepath=filepath)
        kb_builder.draw_picture_from_sample(filepath=filepath)
    
    

    #建立基于贝叶斯优化的稀疏知识库
    if need_sparse_kb==1:

        kb_builder.send_query() 
        if dec_rand==1:
            mem_comb_list_dec=kb_builder.get_comb_list_dec_rand()  #获取随机下降结果
        else:
            mem_comb_list_dec=kb_builder.get_comb_list_dec_max()  #每次下降取最大的
        print(mem_comb_list_dec)

        if sel_rand==1:
            mem_combs=kb_builder.get_combs_rand(comb_list_dec=mem_comb_list_dec,comb_num=comb_num)  #随机选取
        else:
            mem_combs=kb_builder.get_combs_cert(comb_list_dec=mem_comb_list_dec,comb_num=comb_num) #等距选取
        print(mem_combs)

        filename_list=[]  #存放对每一种内存限制进行采样的文件名称
        for comb in mem_combs:
            print(comb)
            for i in range(0,len(serv_names)):
                #通过conf_and_info来强行限制内存的具体取值
                conf_and_serv_info[serv_names[i]+'_mem_util_limit']=[comb[i]]
                print(serv_names[i]+'_mem_util_limit',conf_and_serv_info[serv_names[i]+'_mem_util_limit'])
            
            # 确定当前资源限制之后，就可以开始采样了。
            filename=kb_builder.sample_and_record_bayes(sample_bound=sample_bound,n_trials=n_trials,bin_nums=bin_nums,bayes_goal=BEST_STD_DELAY,min_val=min_val,max_val=max_val)
            filename_list.append(filename)
        
        #把所有文件名记录下来
        fp = open(record_name+".txt", "w")
        # 写入字符串
        for filename in filename_list:
            fp.write(filename+'\n')
        # 关闭文件
        fp.close()
        print("初始知识库相关文件参见"+record_name+".txt")


    # 建立基于遍历各类配置的知识库
    if need_tight_kb==1:
        kb_builder.send_query() 
        kb_builder.sample_and_record(sample_bound=5) #表示对于所有配置组合每种组合采样sample_bound次。

    # 关于是否需要建立知识库：可以根据txt文件中的内容来根据采样结果建立知识库
    if need_to_build==1:
        record_name=KB_DATA_PATH+'/0_gender_classify_bayes1dec_rand0sel_rand0mem_num10min_val0.0max_val1.0bin_nums100sample_bound5n_trials20.txt'

        with open(record_name, 'r') as file:
            # 逐行读取文件内容，将每行内容（即每个单词）打印出来
            # 简单来说，每一个内存组合都对应一个记录下来的采样文件，所以对于每一个文件都要建立知识库。
            # 先create一个知识库，之后在此基础上不断更新。
            i=0
            for line in file:
                print(line.strip())
                filepath=line.strip()
                if i==0:
                    kb_builder.create_evaluator_from_samples(filepath=filepath)
                else:
                    kb_builder.update_evaluator_from_samples(filepath=filepath)
                i+=1
        print("完成时延知识库的建立")
        
        with open(record_name, 'r') as file:
            # 逐行读取文件内容，将每行内容（即每个单词）打印出来
            # 简单来说，每一个内存组合都对应一个记录下来的采样文件，所以对于每一个文件都要建立知识库。
            # 先create一个知识库，之后在此基础上不断更新。
            i=0
            for line in file:
                print(line.strip())
                filepath=line.strip()
                time.sleep(5)
                if i==0:
                    kb_builder.create_conf_info_from_samples(filepath=filepath)
                else:
                    kb_builder.update_conf_info_from_samples(filepath=filepath)
                i+=1
        print("完成所选配置范围的确定")

        # 是否需要绘画展示文件中的数据结果
    
    # 如果想要用新记录的csv文件内容来更新知识库
    if need_to_add==1:
        
        new_list=['kb_data/20240511_16_57_05_kb_builder_0.3_tight_build_gender_classify_cold_start04.csv',
        'kb_data/20240511_17_36_55_kb_builder_0.3_tight_build_gender_classify_cold_start04.csv',
        'kb_data/20240511_19_19_00_kb_builder_0.3_tight_build_gender_classify_cold_start04.csv',
        'kb_data/20240511_19_30_24_kb_builder_0.3_tight_build_gender_classify_cold_start04.csv']
        
        for filepath in new_list:
            kb_builder.update_evaluator_from_samples(filepath=filepath)
            kb_builder.update_conf_info_from_samples(filepath=filepath)
    
    
    if need_new_ip==1:
        # 此处可以直接修改知识库中的特定配置，比如删除所有特定ip下的配置，或者基于各个边缘端都相同的思想添加新的边缘端ip配置，或者将知识库中的某个ip换成新的ip
        kb_builder.add_node_ip_in_kb(new_node_ip='192.168.1.9')
        # kb_builder.delete_node_ip_in_kb(old_node_ip='172.27.143.164')
        # kb_builder.swap_node_ip_in_kb(old_node_ip='172.27.143.164',new_node_ip='192.168.1.7')
        print('完成更新')


    exit()

    
