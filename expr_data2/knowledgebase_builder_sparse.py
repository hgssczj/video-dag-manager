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
import matplotlib.pyplot as plt
import sklearn.metrics
import time
import optuna
import itertools
import random
plt.rcParams['font.sans-serif'] = ['SimHei'] # 运行配置参数中的字体（font）为黑体（SimHei）

#试图以类的方式将整个独立建立知识库的过程模块化

NO_BAYES_GOAL=0
BEST_ALL_DELAY=1
BEST_STD_DELAY=2
MAX_NUMBER=999999

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

# conf_and_serv_info表示每一种配置的取值范围。
'''
conf_and_serv_info={  #各种配置参数的可选值
    "reso":["360p", "480p", "720p", "1080p"],
    "fps":[1, 5, 10, 20, 30],
    "encoder":["JPEG"],
    
    "face_alignment_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],
    "face_alignment_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_alignment_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],

    "face_alignment_trans_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],
    "face_alignment_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_alignment_trans_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_trans_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],

}
'''
'''
# 以下是缩小范围版，节省知识库大小
conf_and_serv_info={  #各种配置参数的可选值
    "reso":["360p","480p","720p","1080p"],
    "fps":[1, 5, 10, 20, 30],
    "encoder":["JPEG"],
    
    "face_alignment_ip":["114.212.81.11","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["114.212.81.11","172.27.151.145"],
    "car_detection_ip":["114.212.81.11","172.27.151.145"],
    "face_alignment_mem_util_limit":[1.0],
    "face_alignment_cpu_util_limit":[1.0],
    "face_detection_mem_util_limit":[1.0],
    "face_detection_cpu_util_limit":[1.0],
    "car_detection_mem_util_limit":[1.0],
    "car_detection_cpu_util_limit":[1.0],

    "face_alignment_trans_ip":["114.212.81.11","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["114.212.81.11","172.27.151.145"],
    "car_detection_trans_ip":["114.212.81.11","172.27.151.145"],
    "face_alignment_trans_mem_util_limit":[1.0],
    "face_alignment_trans_cpu_util_limit":[1.0],
    "face_detection_trans_mem_util_limit":[1.0],
    "face_detection_trans_cpu_util_limit":[1.0],
    "car_detection_trans_mem_util_limit":[1.0],
    "car_detection_trans_cpu_util_limit":[1.0],

}
'''
#'''
#"172.27.151.145"
conf_and_serv_info={  #各种配置参数的可选值
    "reso":["360p", "480p", "720p", "1080p"],
    "fps":[1, 5, 10, 20, 30],
    "encoder":["JPEG"],
    
    "face_alignment_ip":["114.212.81.11","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["114.212.81.11","172.27.151.145"],
    "car_detection_ip":["114.212.81.11","172.27.151.145"],
    "face_alignment_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_alignment_cpu_util_limit":[1.0],
    "face_detection_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_cpu_util_limit":[1.0],
    "car_detection_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_cpu_util_limit":[1.0],

    "face_alignment_trans_ip":["114.212.81.11","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["114.212.81.11","172.27.151.145"],
    "car_detection_trans_ip":["114.212.81.11","172.27.151.145"],
    "face_alignment_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_alignment_trans_cpu_util_limit":[1.0],
    "face_detection_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_trans_cpu_util_limit":[1.0],
    "car_detection_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_trans_cpu_util_limit":[1.0],

}

#'''

# 调度策略三要素：conf，flow_mapping，resource_limit，如下：
'''
conf:
{
    'reso': '480p', 
    'fps': 1, 
    'encoder': 'JPEG'
}
flow_mapping:
{
    'face_detection': 
    {
        'model_id': 0, 
        'node_ip': '172.27.151.145', 
        'node_role': 'host'
    }, 
    'face_alignment': 
    {
        'model_id': 0, 
        'node_ip': '172.27.151.145',
        'node_role': 'host'
    }
}

resource_limit:
{
    'face_detection': 
    {
        'cpu_util_limit': 1.0, 
        'mem_util_limit': 1.0
    }, 
    'face_alignment': 
    {
        'cpu_util_limit': 1.0, 
        'mem_util_limit': 1.0
    }
}
'''



# 每一个特定任务对应一个KnowledgeBaseBuilder类
class KnowledgeBaseBuilder():   
    
    #知识库建立者，每次运行的时候要指定本次实验名称，用于作为记录实验的文件
    def __init__(self,expr_name,node_ip,node_addr,query_addr,service_addr,query_body,conf_names,serv_names,service_info_list):

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
    def anylze_explore_result(self,filepath):  #分析记录下来的文件结果，也就是采样结果
        
        df = pd.read_csv(filepath)
        df = df[df.all_delay<1]

        x_list=[i for i in range(0,len(df))]
        soretd_value=sorted(df['all_delay'])
        a,b,c=self.draw_hist(data=soretd_value,title_name='分布',bins=100)
        
        a=list(a)
        
        a=np.array(a)
        print(a)
        print(a.std())
        print(sum(a))

    # evaluator_init：
    # 用途：为一个服务建立一个由键值对构成的空白字典（空白知识库），以json文件形式保存
    # 方法：从conf_list中按顺序提取涉及的各个配置名称，并构建嵌套字典，初始值全部是0，
    # 返回值：无，但会在当前目录保存一个json文件
    def evaluator_init(self,conf_list,eval_name):  
        #conf_num是建立评估器需要的配置旋钮的总数，conf_list里包含了一系列list，其中每一个都是某一种配置的可选组合
        models_dicts=[]
        conf_num=len(conf_list)
        for i in range(0,conf_num):
            temp=dict()
            models_dicts.append(temp)
        for key in conf_list[conf_num-1]:
             models_dicts[conf_num-1][key]=0
        for i in range(1,conf_num):
            for key in conf_list[conf_num-i-1]:
                models_dicts[conf_num-i-1][key]=models_dicts[conf_num-i]
        
        evaluator=models_dicts[0]  #获取最终的评估器
        # print(evaluator)
        # 建立初始化性能评估器并写入文件
        f=open(eval_name+".json","w")
        json.dump(evaluator,f,indent=1)
        f.close()
        print("完成对评估器",eval_name,"的空白json文件初始化")
    
    
    # evaluator_load：
    # 用途：读取某个json文件里保存的字典
    # 方法：根据eval_name从当前目录下打开相应json文件并返回
    # 返回值：从文件里读取的字典
    def evaluator_load(self,eval_name):
        f=open(eval_name+".json")
        evaluator=json.load(f)
        f.close()
        return evaluator  #返回加载得到的字典
    
    # evaluator_dump：
    # 用途：将字典内容重新写入文件
    # 方法：将参数里的字典写入参数里指定的eval_name文件中
    # 返回值：无
    def evaluator_dump(self,evaluator,eval_name):  #将字典重新写入文件
        f=open(eval_name+".json","w")
        json.dump(evaluator,f,indent=1)
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
        
        filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + os.path.basename(__file__).split('.')[0] + \
            '_' + str(self.query_body['user_constraint']['delay']) + \
            '_' + str(self.query_body['user_constraint']['accuracy']) + \
            '_' + self.expr_name + \
            '.csv'

        self.fp = open(filename, 'w', newline='')

        fieldnames=[]
        fieldnames=[ 'n_loop',
                    'frame_id',
                    'all_delay',
                    'edge_mem_ratio']
         #得到配置名
        for i in range(0,len(self.conf_names)):
            fieldnames.append(self.conf_names[i])  

        #serv_names形如['face_detection', 'face_alignment']
        for i in range(0,len(self.serv_names)):
            serv_name=self.serv_names[i]
            
            field_name=serv_name+'_role'
            fieldnames.append(field_name)
            field_name=serv_name+'_ip'
            fieldnames.append(field_name)
            field_name=serv_name+'_proc_delay'
            fieldnames.append(field_name)

            field_name=serv_name+'_trans_ip'
            fieldnames.append(field_name)
            field_name=serv_name+'_trans_delay'
            fieldnames.append(field_name)

            # 以下用于获取每一个服务对应的cpu资源画像、限制和效果
            field_name=serv_name+'_cpu_portrait'
            fieldnames.append(field_name)
            field_name=serv_name+'_cpu_util_limit'
            fieldnames.append(field_name)
            field_name=serv_name+'_cpu_util_use'
            fieldnames.append(field_name)

            
            # 以下用于获取每一个服务对应的cpu资源画像、限制和效果
            field_name=serv_name+'_trans'+'_cpu_portrait'
            fieldnames.append(field_name)
            field_name=serv_name+'_trans'+'_cpu_util_limit'
            fieldnames.append(field_name)
            field_name=serv_name+'_trans'+'_cpu_util_use'
            fieldnames.append(field_name)
            

            # 以下用于获取每一个服务对应的内存资源画像、限制和效果
            field_name=serv_name+'_mem_portrait'
            fieldnames.append(field_name)
            field_name=serv_name+'_mem_util_limit'
            fieldnames.append(field_name)
            field_name=serv_name+'_mem_util_use'
            fieldnames.append(field_name)

            # 以下用于获取每一个服务对应的内存资源画像、限制和效果
            field_name=serv_name+'_trans'+'_mem_portrait'
            fieldnames.append(field_name)
            field_name=serv_name+'_trans'+'_mem_util_limit'
            fieldnames.append(field_name)
            field_name=serv_name+'_trans'+'_mem_util_use'
            fieldnames.append(field_name)


        self.writer = csv.DictWriter(self.fp, fieldnames=fieldnames)
        self.writer.writeheader()
        self.written_n_loop.clear() #用于存储各个轮的序号，防止重复记录

        return filename

    # write_in_file：
    # 用途：在csv文件中写入一条执行记录
    # 方法：参数r2r3r4分别表示资源信息、执行回应和runtime_info，从中提取可以填充csv文件的信息，利用字典把所有不重复的感知结果都保存在updatetd_result之中
    # 返回值：updatetd_result，保存了当前的运行时情境和执行结果
    def write_in_file(self,r2,r3,r4):   #pipeline指定了任务类型   
        resource_info = r2.json()
        resp = r3.json()
        runtime_info=r4.json()
        # resource_info的典型结构
        '''
        {
            'cloud': {
                '114.212.81.11': {
                    'cpu_ratio': 1.3, 
                    'gpu_compute_utilization': {'0': 0, '1': 0, '2': 0, '3': 0}, 
                    'gpu_mem_total': {'0': 24.0, '1': 24.0, '2': 24.0, '3': 24.0}, 
                    'gpu_mem_utilization': {'0': 5.941518147786459, '1': 1.2865702311197915, '2': 1.2865702311197915, '3': 1.2865702311197915}, 
                    'mem_ratio': 4.6, 
                    'mem_total': 251.56013107299805, 
                    'n_cpu': 48, 
                    'net_ratio(MBps)': 0.76916, 
                    'swap_ratio': 0.0
                }
            }, 
            'host': {
                '172.27.151.145': {
                    'cpu_ratio': 13.0, 
                    'gpu_compute_utilization': {'0': 0.0}, 
                    'gpu_mem_total': {'0': 3.9}, 
                    'gpu_mem_utilization': {'0': 30.865302452674282}, 
                    'mem_ratio': 76.3, 
                    'mem_total': 7.675579071044922, 
                    'n_cpu': 4, 
                    'net_ratio(MBps)': 0.0125, 
                    'swap_ratio': 36.6
                }
            }
        }
        
        '''
        # runtime_info的典型结构（抬头检测情况下）
        '''
        {
            'delay': 0.34250664710998535, 
            'obj_n': 7.0, 
            'obj_size': 4432.916892585744, 
            'obj_stable': True, 
            'runtime_portrait': {
                'face_alignment': [
                    {
                        'resource_runtime': {
                            'all_latency': 0.27319955825805664, 
                            'compute_latency': 0.11972928047180176, 
                            'cpu_portrait': 0, 
                            'cpu_util_limit': 1.0, 
                            'cpu_util_use': 0.2729375, 
                            'device_ip': '114.212.81.11', 
                            'mem_portrait': 0, 
                            'mem_util_limit': 1.0, 
                            'mem_util_use': 0.010848892680669442, 
                            'pid': 2210028
                        }, 
                        'task_conf': {'encoder': 'JPEG', 'fps': 30, 'reso': '1080p'}, 
                        'work_runtime': {'obj_n': 7}
                    }
                ], 
                'face_detection': [
                    {
                        'resource_runtime': {
                            'all_latency': 0.41181373596191406, 
                            'compute_latency': 0.016497373580932617, 
                            'cpu_portrait': 0, 
                            'cpu_util_limit': 0.3, 
                            'cpu_util_use': 0.10039583333333334, 
                            'device_ip': '114.212.81.11', 
                            'mem_portrait': 0, 
                            'mem_util_limit': 1.0, 
                            'mem_util_use': 0.008297492975166465, 
                            'pid': 2210015
                        }, 
                        'task_conf': {'encoder': 'JPEG', 'fps': 30, 'reso': '1080p'}, 
                        'work_runtime': {'obj_n': 7}
                    }
                ]
            }
        }
        
        '''

        edge_mem_ratio=resource_info['host'][self.node_ip]['mem_ratio']


        appended_result = resp['appended_result'] #可以把得到的结果直接提取出需要的内容，列表什么的。
        latest_result = resp['latest_result'] #空的

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
            row={}
            # print("查看待处理结果")
            # print(res)
            row['n_loop']=res['n_loop']
            row['frame_id']=res['frame_id']
            row['all_delay']=res[ 'delay']
            row['edge_mem_ratio']=edge_mem_ratio

            for i in range(0,len(self.conf_names)):
                conf_name=self.conf_names[i]   #得到配置名
                row[conf_name]=res['ext_plan']['video_conf'][conf_name]

            #serv_names形如['face_detection', 'face_alignment']
            for i in range(0,len(self.serv_names)):
                serv_name=self.serv_names[i]
                
                serv_role_name=serv_name+'_role'

                serv_ip_name=serv_name+'_ip'
                serv_proc_delay_name=serv_name+'_proc_delay'

                trans_ip_name=serv_name+'_trans_ip'
                trans_delay_name=serv_name+'_trans_delay'

                row[serv_role_name]=res['ext_plan']['flow_mapping'][serv_name]['node_role']
                row[serv_ip_name]=res['ext_plan']['flow_mapping'][serv_name]['node_ip']
                row[serv_proc_delay_name]=res['ext_runtime']['plan_result']['process_delay'][serv_name]
                row[trans_ip_name]=row[serv_ip_name]
                row[trans_delay_name]=res['ext_runtime']['plan_result']['delay'][serv_name]-row[serv_proc_delay_name]

                # 要从runtime_info里获取资源信息。暂时只提取runtime_portrait列表中的第一个画像
                # 以下用于获取每一个服务对应的cpu资源画像、限制和效果
                field_name=serv_name+'_cpu_portrait'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['cpu_portrait']
                field_name=serv_name+'_cpu_util_limit'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['cpu_util_limit']
                field_name=serv_name+'_cpu_util_use'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['cpu_util_use']

                # 以下用于获取每一个服务对应的内存资源画像、限制和效果
                field_name=serv_name+'_mem_portrait'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['mem_portrait']
                field_name=serv_name+'_mem_util_limit'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['mem_util_limit']
                field_name=serv_name+'_mem_util_use'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['mem_util_use']

                # 以下用于获取每一个服务对应的cpu资源画像、限制和效果
                field_name=serv_name+'_trans'+'_cpu_portrait'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['cpu_portrait']
                field_name=serv_name+'_trans'+'_cpu_util_limit'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['cpu_util_limit']
                field_name=serv_name+'_trans'+'_cpu_util_use'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['cpu_util_use']

                # 以下用于获取每一个服务对应的内存资源画像、限制和效果
                field_name=serv_name+'_trans'+'_mem_portrait'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['mem_portrait']
                field_name=serv_name+'_trans'+'_mem_util_limit'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['mem_util_limit']
                field_name=serv_name+'_trans'+'_mem_util_use'
                row[field_name]=runtime_info['runtime_portrait'][serv_name][0]['resource_runtime']['mem_util_use']
                       
            n_loop=res['n_loop']
            if n_loop not in self.written_n_loop:  #以字典为参数，只有那些没有在字典里出现过的row才会被写入文件，
                self.writer.writerow(row)
                print("写入成功")
                self.written_n_loop[n_loop] = 1
                #完成文件写入之后，将对应的row和配置返回以供分析。由于存在延迟，这些新数据对应的conf和flow_mapping可能和前文指定的不同
                updatetd_result.append({"row":row,"conf":res['ext_plan']['video_conf'],"flow_mapping":res['ext_plan']['flow_mapping'],"resource_limit":res['ext_plan']['resource_limit']})

        #updatetd_result会返回本轮真正检测到的全新数据。在最糟糕的情况下，updatetd_result会是一个空列表。
        return updatetd_result


    # post_get_write：
    # 用途：更新调度计划，感知运行时情境，并调用post_get_write记录在csv文件中
    # 方法：使用update_plan接口将conf,flow_mapping,resource_limit应用在query_id指定的任务中
    #      通过get_resource_info和get_result和get_runtime依次获取运行时情境
    #      使用post_get_write方法将感知到的结果写入文件之中
    # 返回值：包含updated_result的键值对
    def post_get_write(self,conf,flow_mapping,resource_limit):
        # print("开始发出消息并配置")
        #（1）更新配置
        r1 = self.sess.post(url="http://{}/job/update_plan".format(self.node_addr),
                        json={"job_uid": self.query_id, "video_conf": conf, "flow_mapping": flow_mapping,"resource_limit":resource_limit})
        if not r1.json():
            return {"status":0,"des":"fail to update plan"}
         
        #（2）获取资源情境,获取node_ip指定的边缘节点的内存使用率
        r2 = self.sess.get(url="http://{}/get_resource_info".format(self.service_addr))
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
        r4 = self.sess.get(url="http://{}/query/get_runtime/{}".format(self.query_addr, self.query_id))  
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
    # 用途：感知运行时情境，并调用post_get_write记录在csv文件中。相比post_get_result，不会修改调度计划。
    # 方法：通过get_resource_info和get_result和get_runtime依次获取运行时情境
    #      使用post_get_write方法将感知到的结果写入文件之中
    # 返回值：包含updated_result的键值对
    def get_write(self):
         
        #（1）获取资源情境,获取node_ip指定的边缘节点的内存使用率
        r2 = self.sess.get(url="http://{}/get_resource_info".format(self.service_addr))
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
        r4 = self.sess.get(url="http://{}/query/get_runtime/{}".format(self.query_addr, self.query_id))  
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
                    print(updatetd_result[i])
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
        filename=self.init_record_file()
        record_sum=0
        while(record_sum<record_num):
            get_resopnse=self.get_write()
            if(get_resopnse['status']==3):
                updatetd_result=get_resopnse['updatetd_result']
                for i in range(0,len(updatetd_result)):
                    print(updatetd_result[i])
                    record_sum+=1

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
            resource_limit[serv_name]["cpu_util_limit"]=trial.suggest_categorical(serv_cpu_limit,conf_and_serv_info[serv_cpu_limit])
            resource_limit[serv_name]["mem_util_limit"]=trial.suggest_categorical(serv_mem_limit,conf_and_serv_info[serv_mem_limit])
        
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
            conf_list=[]
            for conf_name in service_conf: #conf_and_serv_info中包含了service_conf中所有配置的可能取值
                conf_list.append(conf_and_serv_info[conf_name]) 
                   
            # 基于得到的conf_list来构建该服务的性能评估器，记为evaluator
            self.evaluator_init(conf_list=conf_list,eval_name=service_info["name"])
            evaluator=self.evaluator_load(eval_name=service_info["name"])

            conf_kind=0
            conf_combine=itertools.product(*conf_list)  #基于conf_list，所得conf_combine包含所有配置参数的组合
            for conf_for_file in conf_combine: #遍历每一种配置组合
                conf_for_dict=list(str(item) for item in conf_for_file)
                # print(conf_for_dict)
                # conf_for_dict形如['480p', '30', 'JPEG', '172.27.143.164', '0.5']。
                # conf_for_file形如('480p', 30, 'JPEG', '172.27.143.164', 0.5)，是元组且保留整数类型，用于从df中提取数据
                condition_all=True  #用于检索字典所需的条件
                for i in range(0,len(conf_for_dict)):
                    condition=df[service_conf[i]]==conf_for_file[i]    #相应配置名称对应的配置参数等于当前配置组合里的配置
                    '''
                    if conf_for_file==('360p', 1, 'JPEG', '114.212.81.11', 1.0):
                        print("存在此配置")
                        print(conf_for_file[i])
                        print(df[service_conf[i]])
                        print(condition)
                    '''
                    condition_all=condition_all&condition
                # 联立所有条件从df中获取对应内容,conf_df里包含满足所有条件的列构成的df
                conf_df=df[condition_all]
                if(len(conf_df)>0): #如果满足条件的内容不为空，可以开始用其中的数值来初始化字典
                    # print("存在满足条件的字典")
                    conf_kind+=1
                    avg_value=conf_df[service_info['value']].mean()  #获取均值
                    # print(conf_df[['d_ip','a_ip','fps','reso',service_info['value']]])
                    sub_evaluator=evaluator
                    for i in range(0,len(conf_for_dict)-1):
                        sub_evaluator=sub_evaluator[conf_for_dict[i]]
                    sub_evaluator[conf_for_dict[len(conf_for_dict)-1]]=avg_value
            #完成对evaluator的处理
            self.evaluator_dump(evaluator=evaluator,eval_name=service_info['name'])
            '''
            print("问题：")
            print(df['encoder'])
            print(df['encoder']=='JPEG')
            print(df['encoder']==conf_for_file[2])
            '''
            print("该服务",service_info['name'],"涉及配置组合总数",conf_kind)
    

    # update_evaluator_from_samples：
    # 用途：根据记录的日志文件更新字典，每一个配置对应一个平均值
    # 方法：然后读取filepath参数指定的文件，更新已经存在的字典，不会重新建立
    # 返回值：无，但是会得到被更新的字典
    def update_evaluator_from_samples(self,filepath):
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
            conf_list=[]
            for conf_name in service_conf: #conf_and_serv_info中包含了service_conf中所有配置的可能取值
                conf_list.append(conf_and_serv_info[conf_name]) 
                   
            # 直接读取已经初始化完毕的字典
            evaluator=self.evaluator_load(eval_name=service_info["name"])

            conf_kind=0
            conf_combine=itertools.product(*conf_list)  #基于conf_list，所得conf_combine包含所有配置参数的组合
            for conf_for_file in conf_combine: #遍历每一种配置组合
                conf_for_dict=list(str(item) for item in conf_for_file)
                # print(conf_for_dict)
                # conf_for_dict形如['480p', '30', 'JPEG', '172.27.143.164', '0.5']。
                # conf_for_file形如('480p', 30, 'JPEG', '172.27.143.164', 0.5)，是元组且保留整数类型，用于从df中提取数据
                condition_all=True  #用于检索字典所需的条件
                for i in range(0,len(conf_for_dict)):
                    condition=df[service_conf[i]]==conf_for_file[i]    #相应配置名称对应的配置参数等于当前配置组合里的配置
                    '''
                    if conf_for_file==('360p', 1, 'JPEG', '114.212.81.11', 1.0):
                        print("存在此配置")
                        print(conf_for_file[i])
                        print(df[service_conf[i]])
                        print(condition)
                    '''
                    condition_all=condition_all&condition
                # 联立所有条件从df中获取对应内容,conf_df里包含满足所有条件的列构成的df
                conf_df=df[condition_all]
                if(len(conf_df)>0): #如果满足条件的内容不为空，可以开始用其中的数值来初始化字典
                    # print("存在满足条件的字典")
                    conf_kind+=1
                    avg_value=conf_df[service_info['value']].mean()  #获取均值
                    # print(conf_df[['d_ip','a_ip','fps','reso',service_info['value']]])
                    sub_evaluator=evaluator
                    for i in range(0,len(conf_for_dict)-1):
                        sub_evaluator=sub_evaluator[conf_for_dict[i]]
                    sub_evaluator[conf_for_dict[len(conf_for_dict)-1]]=avg_value
            #完成对evaluator的处理
            self.evaluator_dump(evaluator=evaluator,eval_name=service_info['name'])
            '''
            print("问题：")
            print(df['encoder'])
            print(df['encoder']=='JPEG')
            print(df['encoder']==conf_for_file[2])
            '''
            print("该服务",service_info['name'],"涉及配置组合总数",conf_kind)
    
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
    def draw_picture(self,x_value,y_value,title_name):
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'
        plt.yticks(fontproperties='Times New Roman', )
        plt.xticks(fontproperties='Times New Roman', )
        plt.plot(x_value,y_value)
        plt.title(title_name)
        plt.show()

    # draw_delay_and_cons：
    # 用途：在相同的x值上绘制两个y值，如果需要绘制约束的话就用它
    # 方法：不赘述
    # 返回值：无
    def draw_delay_and_cons(self,x_value1,y_value1,y_value2,title_name):
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'
        plt.ylim(0,1)
        plt.yticks(fontproperties='Times New Roman', )
        plt.xticks(fontproperties='Times New Roman', )
        plt.plot(x_value1,y_value1,label="实际时延")
        plt.plot(x_value1,y_value2,label="时延约束")
        plt.title(title_name)
        plt.legend()
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
        self.conf_names
        self.serv_names

        cons_delay=[]
        for x in df['n_loop']:
            cons_delay.append(self.query_body['user_constraint']['delay'])

        # 绘制总时延和约束时延
        self.draw_delay_and_cons(x_value1=df['n_loop'],y_value1=df['all_delay'],y_value2=cons_delay,title_name="all_delay&constraint_delay/时间")

        # '''
        for serv_name in self.serv_names:
            # if serv_name=="face_alignment":  #专门研究人脸检测情况
            # if serv_name=="face_detection":  #专门研究姿态估计情况
                # continue
            serv_role_name=serv_name+'_role'
            serv_ip_name=serv_name+'_ip'
            serv_proc_delay_name=serv_name+'_proc_delay'
            trans_ip_name=serv_name+'_trans_ip'
            trans_delay_name=serv_name+'_trans_delay'
            # 绘制各个服务的处理时延以及ip变化
                        # 以下用于获取每一个服务对应的内存资源画像、限制和效果
            mem_portrait=serv_name+'_mem_portrait'
            mem_util_limit=serv_name+'_mem_util_limit'
            mem_util_use=serv_name+'_mem_util_use'

            cpu_portrait=serv_name+'_cpu_portrait'
            cpu_util_limit=serv_name+'_cpu_util_limit'
            cpu_util_use=serv_name+'_cpu_util_use'

            self.draw_picture(x_value=df['n_loop'],y_value=df[serv_ip_name],title_name=serv_ip_name+"/时间")
            self.draw_picture(x_value=df['n_loop'],y_value=df[serv_proc_delay_name],title_name=serv_proc_delay_name+"/时间")
            self.draw_picture(x_value=df['n_loop'],y_value=df[trans_delay_name],title_name=trans_delay_name+"/时间")

            self.draw_picture(x_value=df['n_loop'],y_value=df[mem_portrait],title_name=mem_portrait+"/时间")
            self.draw_picture(x_value=df['n_loop'],y_value=df[mem_util_limit],title_name=mem_util_limit+"/时间")

             # print(df[mem_util_use])
            self.draw_picture(x_value=df['n_loop'],y_value=df[mem_util_use],title_name=mem_util_use+"/时间")

            self.draw_picture(x_value=df['n_loop'],y_value=df[cpu_portrait],title_name=cpu_portrait+"/时间")
            self.draw_picture(x_value=df['n_loop'],y_value=df[cpu_util_limit],title_name=cpu_util_limit+"/时间")
            self.draw_picture(x_value=df['n_loop'],y_value=df[cpu_util_use],title_name=cpu_util_use+"/时间")
            
        
        for conf_name in self.conf_names:
          self.draw_picture(x_value=df['n_loop'],y_value=df[conf_name],title_name=conf_name+"/时间")
        # '''


#以上是KnowledgeBaseBuilder类的全部定义，该类可以让使用者在初始化后完成一次完整的知识库创建，并分析采样的结果
#接下来定义一个新的类，作用的基于知识库进行冷启动
          
# ColdStartPlanner可以基于建立好的知识库，提供冷启动所需的一系列策略
class ColdStartPlanner():   
    
    #冷启动计划者，初始化时需要conf_names,serv_names,service_info_list,user_constraint一共四个量
    #在初始化过程中，会根据这些参数，制造conf_list，serv_ip_list，serv_cpu_list以及serv_meme_list
    def __init__(self,conf_names,serv_names,service_info_list,user_constraint):
        self.conf_names=conf_names
        self.serv_names=serv_names
        self.service_info_list=service_info_list
        self.user_constraint=user_constraint

        
        self.conf_list=[]
        for conf_name in conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip和资源限制
            self.conf_list.append(conf_and_serv_info[conf_name])
        # conf_list会包含各类配置参数取值范围，例如分辨率、帧率等

        self.serv_ip_list=[]
        for serv_name in serv_names:
            serv_ip=serv_name+"_ip"
            self.serv_ip_list.append(conf_and_serv_info[serv_ip])
        # serv_ip_list包含各个模型的ip的取值范围

        self.serv_cpu_list=[]
        for serv_name in serv_names:
            serv_cpu=serv_name+"_cpu_util_limit"
            self.serv_cpu_list.append(conf_and_serv_info[serv_cpu])
        # serv_cpu_list包含各个模型的cpu使用率的取值范围

        self.serv_mem_list=[]
        for serv_name in serv_names:
            serv_mem=serv_name+"_mem_util_limit"
            self.serv_mem_list.append(conf_and_serv_info[serv_mem])
        # serv_mem_list包含各个模型的mem使用率的取值范围
        
        #例如,可能的一组取值如下：
        # conf_names=["reso","fps","encoder"]
        # serv_names=["face_detection", "face_alignment"]
        # conf_list=[ ["360p", "480p", "720p", "1080p"],[1, 5, 10, 20, 30],["JPEG"]]
        # serv_ip_list=[["114.212.81.11","172.27.143.164","172.27.151.145"],["114.212.81.11","172.27.143.164","172.27.151.145"]]
        # serv_cpu_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]
        # serv_mem_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]

    # get_pred_delay：
    # 用途：根据参数里指定的配置，根据知识库来预测对应性能，如果在知识库中找不到，则返回status为0；否则为1
    # 方法：依次遍历评估性能所需的各个字典并根据参数设置从中获取性能评估结果
    # 返回值：status,pred_delay_list,pred_delay_total，描述性能评估成果与否、预测的各阶段时延、预测的总时延
    '''
    'video_conf':   
    {    'encoder': 'JPEG', 'fps': 1, 'reso': '360p'    }
    'flow_mapping': 
    {   
        'face_alignment': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
        'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'} 
    }
    resource_limit={
        "face_detection": {
            "cpu_util_limit": 0.1,
            "mem_util_limit": 0.2,
        },
        "face_alignment": {
            "cpu_util_limit": 0.1,
            "mem_util_limit": 0.2,
        }
    }

    '''
    def get_pred_delay(self,conf, flow_mapping, resource_limit):
        # 知识库所在目录名
        # 存储配置对应的各阶段时延，以及总时延
        pred_delay_list=[]
        pred_delay_total=0
        status=0  #为0表示字典中没有满足配置的存在
        # 对于service_info_list里的service_info依次评估性能
        for service_info in self.service_info_list:
            # （1）加载服务对应的性能评估器
            f=open(service_info['name']+".json")  
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
            
            #完成以上操作后，conf_for_dict内还差ip地址,首先要判断当前评估器是不是针对传输阶段进行的：
            ip_for_dict_index=service_info['name'].find("_trans") 
            if ip_for_dict_index>0:
                # 当前是trans，则去除服务名末尾的_trans，形如“face_detection”
                ip_for_dict_name=service_info['name'][0:ip_for_dict_index] 
            else: #当前不是trans
                ip_for_dict_name=service_info['name']
            ip_for_dict=flow_mapping[ip_for_dict_name]['node_ip']
                
            cpu_for_dict=resource_limit[ip_for_dict_name]["cpu_util_limit"]
            mem_for_dict=resource_limit[ip_for_dict_name]["mem_util_limit"]
            conf_for_dict.append(str(ip_for_dict))  
            conf_for_dict.append(str(cpu_for_dict)) 
            conf_for_dict.append(str(mem_for_dict)) 
            # 形如["360p"，"1","JPEG","114.212.81.11","0.1"."0.1"]

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
    

    # get_coldstart_plan_rotate：
    # 用途：遍历conf_list、ip、cpu和mem所有配置，选择一个能够满足约束的冷启动计划。
    # 方法：生成配置遍历的全排列，对每一种配置都调用get_pred_delay预测时延，选择较好的
    # 返回值：最好的conf, flow_mapping, resource_limit
    def get_coldstart_plan_rotate(self):
        
        #例如,可能的一组取值如下：
        # conf_names=["reso","fps","encoder"]
        # serv_names=["face_detection", "face_alignment"]
        # conf_list=[ ["360p", "480p", "720p", "1080p"],[1, 5, 10, 20, 30],["JPEG"]]
        # serv_ip_list=[["114.212.81.11","172.27.143.164","172.27.151.145"],["114.212.81.11","172.27.143.164","172.27.151.145"]]
        # serv_cpu_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]
        # serv_mem_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]
        
        # 获取时延约束
        delay_constraint = self.user_constraint["delay"]


        best_conf={}
        best_flow_mapping={}
        best_resource_limit={}
        best_pred_delay_list=[]
        best_pred_delay_total=-1

        min_conf={}
        min_flow_mapping={}
        min_resource_limit={}
        min_pred_delay_list=[]
        min_pred_delay_total=-1

        conf_combine=itertools.product(*self.conf_list)
        for conf_plan in conf_combine:
            serv_ip_combine=itertools.product(*self.serv_ip_list)
            for serv_ip_plan in serv_ip_combine:# 遍历所有配置和卸载策略组合
                serv_cpu_combind=itertools.product(*self.serv_cpu_list)
                for serv_cpu_plan in serv_cpu_combind: 
                    serv_mem_combind=itertools.product(*self.serv_mem_list)
                    for serv_mem_plan in serv_mem_combind: 
                        conf={}
                        flow_mapping={}
                        resource_limit={}
                        for i in range(0,len(conf_names)):
                            conf[conf_names[i]]=conf_plan[i]
                        for i in range(0,len(serv_names)):
                            flow_mapping[serv_names[i]]=model_op[serv_ip_plan[i]]
                            resource_limit[serv_names[i]]={}
                            resource_limit[serv_names[i]]["cpu_util_limit"]=serv_cpu_plan[i]
                            resource_limit[serv_names[i]]["mem_util_limit"]=serv_mem_plan[i]
                        # 右上，得到了conf，flow_mapping，以及resource_limit
                        status,pred_delay_list,pred_delay_total = self.get_pred_delay(conf=conf,
                                                                                flow_mapping=flow_mapping,
                                                                                resource_limit=resource_limit,
                                                                                )
                        if status == 0: #如果为0，意味着配置在字典中找不到对应的性能评估结果，知识库没有存储这种配置对应的估计结果
                            continue
                        if best_pred_delay_total<0:   #初始化最优配置和最小配置
                            best_conf=conf
                            best_flow_mapping=flow_mapping
                            best_resource_limit=resource_limit
                            best_pred_delay_list=pred_delay_list
                            best_pred_delay_total=pred_delay_total

                            min_conf=conf
                            min_flow_mapping=flow_mapping
                            min_resource_limit=resource_limit
                            min_pred_delay_list=pred_delay_list
                            min_pred_delay_total=pred_delay_total

                        elif pred_delay_total < delay_constraint*0.7 and pred_delay_total>best_pred_delay_total: #选出一个接近约束且比较大的
                            best_conf=conf
                            best_flow_mapping=flow_mapping
                            best_resource_limit=resource_limit
                            best_pred_delay_list=pred_delay_list
                            best_pred_delay_total=pred_delay_total
                        
                        elif pred_delay_total < best_pred_delay_total: #选出一个最小的
                            min_conf=conf
                            min_flow_mapping=flow_mapping
                            min_resource_limit=resource_limit
                            min_pred_delay_list=pred_delay_list
                            min_pred_delay_total=pred_delay_total
        
        # 完成遍历后，应该可以找到一个比较优秀的冷启动结果
        '''
        print("最优配置是：")
        print(best_conf)
        print(best_flow_mapping)
        print(best_resource_limit)
        print(best_pred_delay_list)
        print(best_pred_delay_total)
        print("最小配置是：")
        print(min_conf)
        print(min_flow_mapping)
        print(min_resource_limit)
        print(min_pred_delay_list)
        print(min_pred_delay_total)
        print("时延约束是",delay_constraint)
        '''       

        if min_pred_delay_total > delay_constraint:
            # print("约束过于严格，选择最小配置")
            return min_conf, min_flow_mapping, min_resource_limit
        else:
            # print("约束不算特别严格，选择最优策略")
            return best_conf, best_flow_mapping, best_resource_limit


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
            resource_limit[serv_name]["cpu_util_limit"]=trial.suggest_categorical(serv_cpu_limit,conf_and_serv_info[serv_cpu_limit])
            resource_limit[serv_name]["mem_util_limit"]=trial.suggest_categorical(serv_mem_limit,conf_and_serv_info[serv_mem_limit])
        

        status,pred_delay_list,pred_delay_total = self.get_pred_delay(conf=conf,flow_mapping=flow_mapping,resource_limit=resource_limit)
        if status==0:  #返回0说明相应配置压根不存在，此时返回MAX_NUMBER。贝叶斯优化的目标是让返回值尽可能小，这种MAX_NUMBER的情况自然会被尽量避免
            return MAX_NUMBER
        else:  #如果成功找到了一个可行的策略，按照如下方式计算返回值，目的是得到尽可能靠近约束时延0.7倍且小于约束时延的配置
            delay_constraint = self.user_constraint["delay"]
            if pred_delay_total <= 0.7*delay_constraint:
                return delay_constraint - pred_delay_total
            elif pred_delay_total > 0.3*delay_constraint:
                return 0.3*delay_constraint + pred_delay_total
            

    # get_coldstart_plan_bayes：
    # 用途：基于贝叶斯优化，选择一个能够满足约束的冷启动计划。
    # 方法：通过贝叶斯优化，在有限轮数内选择一个最优的结果
    # 返回值：最好的conf, flow_mapping, resource_limit
    def get_coldstart_plan_bayes(self,n_trials):

        # 现在有self.conf_names方便选取conf和flow_mapping了，可以得到任意组合。
        # 当前实现中conf_names里指定的配置是每一个服务都共享的配置，只有model需要特殊考虑。
        # 所以应该记录下服务的数目然后对应处理

        study = optuna.create_study()
        study.optimize(self.objective,n_trials=n_trials)

        #所得到的study.best_params是一个字典,形如：
        '''
        {
        'reso': '1080p', 
        'fps': 30, 
        'encoder': 'JPEG', 
        'face_detection_ip': '114.212.81.11', 
        'face_detection_cpu_util_limit': 0.26, 
        'face_detection_mem_util_limit': 0.08, 
        'face_alignment_ip': '114.212.81.11', 
        'face_alignment_cpu_util_limit': 0.08, 
        'face_alignment_mem_util_limit': 0.08
        }
        '''
        # print("展示贝叶斯优化寻找的最优配置")
        # print(study.best_params)
        conf={}
        flow_mapping={}
        resource_limit={}
        for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
            # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
            conf[conf_name]=study.best_params[conf_name]

        for serv_name in self.serv_names:
            serv_ip=serv_name+"_ip"
            flow_mapping[serv_name]=model_op[study.best_params[serv_ip]]
            serv_cpu_limit=serv_name+"_cpu_util_limit"
            serv_mem_limit=serv_name+"_mem_util_limit"
            resource_limit[serv_name]={}
            resource_limit[serv_name]["cpu_util_limit"]=study.best_params[serv_cpu_limit]
            resource_limit[serv_name]["mem_util_limit"]=study.best_params[serv_mem_limit]

        return conf, flow_mapping, resource_limit
    
    
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
    # 方法：我设计了一个非常巧妙的方法来解决问题。以下方式通过“随机下降”来解决问题。dec_max表示每次下降时选择列表里最大的元素。
    #       参数里，value_range表示离散的取值范围，serv_num表示涉及的服务数量
    def get_comb_list_dec_max(self,value_range,serv_num):
        # value_range形如[0.3,0.2,0.1]
        # (1)得到离散取值序列从小到大的排列结果
        value_list=sorted(value_range)  #value_list形如[0.1,0.2,0.3]
        # (2)初始化即将存储递减组合的列表
        comb_list_dec=[]
        # (3)初始化要被放入列表的第一个配置组合,包含value_list中的最大索引，长度等于流水线中的服务数，comb_item形如[2,2]
        comb_item=[(len(value_list)-1) for _ in range(serv_num)]
        comb_list_dec.append(copy.deepcopy(comb_item))

        # 接下来循环选取形如[2,2]的comb_item中的一个元素，减去1；如果不为0就继续选。
        idx,element=self.find_non_zero_element_max(comb_item)
        while idx!=-1:# 只要当前的comb_item还有继续下降的空间，就应该进行一次下降，然后把下降后的结果保存到comb_list_dec里面
            comb_item[idx]-=1
            comb_list_dec.append(copy.deepcopy(comb_item))
            idx,element=self.find_non_zero_element_max(comb_item)
        # 循环结束的时候一定是因为全部是0了
        for i in range(0,len(comb_list_dec)): #现在开始将索引转化为内容
            for j in range(0,len(comb_list_dec[i])):
                temp_idx=int(comb_list_dec[i][j])   #形如（2，2）中的2都是索引，需要通过value_list转化为真正的值
                comb_list_dec[i][j]=value_list[temp_idx]
        
        return comb_list_dec
    

    # get_comb_list_dec_rand
    # 用途：同上，只不过下降时是随机的
    # 方法：我设计了一个非常巧妙的方法来解决问题。以下方式通过“随机下降”来解决问题。dec_rand表示每次下降时选择列表里随机一个元素
    #       参数里，value_range表示离散的取值范围，serv_num表示涉及的服务数量
    def get_comb_list_dec_rand(self,value_range,serv_num):
        # value_range形如[0.3,0.2,0.1]
        # (1)得到离散取值序列从小到大的排列结果
        value_list=sorted(value_range)  #value_list形如[0.1,0.2,0.3]
        # (2)初始化即将存储递减组合的列表
        comb_list_dec=[]
        # (3)初始化要被放入列表的第一个配置组合,包含value_list中的最大索引，长度等于流水线中的服务数，comb_item形如[2,2]
        comb_item=[(len(value_list)-1) for _ in range(serv_num)]
        comb_list_dec.append(copy.deepcopy(comb_item))

        # 接下来循环选取形如[2,2]的comb_item中的一个元素，减去1；如果不为0就继续选。
        idx,element=self.find_non_zero_element_rand(comb_item)
        while idx!=-1:# 只要当前的comb_item还有继续下降的空间，就应该进行一次下降，然后把下降后的结果保存到comb_list_dec里面
            comb_item[idx]-=1
            comb_list_dec.append(copy.deepcopy(comb_item))
            idx,element=self.find_non_zero_element_rand(comb_item)
        # 循环结束的时候一定是因为全部是0了
        for i in range(0,len(comb_list_dec)): #现在开始将索引转化为内容
            for j in range(0,len(comb_list_dec[i])):
                temp_idx=int(comb_list_dec[i][j])   #形如（2，2）中的2都是索引，需要通过value_list转化为真正的值
                comb_list_dec[i][j]=value_list[temp_idx]
        
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


# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","face_alignment"]   

#以下是发出query请求时的内容。注意video_id。当前文件需要配合query_manager_v2.py运行，后者使用的调度器会根据video_id的取值判断是否会运行。
#建议将video_id设置为99，它对应的具体视频内容可以在camera_simulation里找到，可以自己定制。query_manager_v2.py的调度器发现query_id为99的时候，
#不会进行调度动作。因此，知识库建立者可以自由使用update_plan接口操控任务的调度方案，不会受到云端调度器的影响了。
query_body = {
        "node_addr": "172.27.151.145:5001",
        "video_id": 99,   
        "pipeline": ["face_detection", "face_alignment"],#制定任务类型
        "user_constraint": {
            "delay": 0.6,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }  
#'''
'''
service_info_list=[
    {
        "name":'car_detection',
        "value":'car_detection_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    {
        "name":'car_detection_trans',
        "value":'car_detection_trans_delay',
        "conf":["reso","fps","encoder"]
    },
]

# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["car_detection"]   
# 进行车辆检测
query_body = {
        "node_addr": "172.27.151.145:5001",
        "video_id": 101,   
        "pipeline": ["car_detection"],#制定任务类型
        "user_constraint": {
            "delay": 0.1,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }  
'''


if __name__ == "__main__":

    # 内存限制的取值范围
    mem_range=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]
    
    # 取多少个典型内存限制进行贝叶斯优化
    comb_num=10
    
    # 贝叶斯优化时的取值范围，在以下范围内使得采样点尽可能平均
    min_val=0.0
    max_val=0.7

    # 在多大范围内取方差
    bin_nums=100

    # 每一种配置下进行多少次采样
    sample_bound=5

    # 每一次贝叶斯优化时尝试多少次
    n_trials=20
    
    # 是否进行稀疏采样
    need_sparse_kb=0
    # 是否进行严格采样
    need_tight_kb=0

    #获取内存资源限制列表的时候，需要两步，第一步是下降，第二部是采取，两种方法都可以随机，也都可以不随机
    dec_rand=0
    sel_rand=0 

    task_name="headup_detect"

    record_name=datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')+task_name+"_"+"bayes"+str(need_sparse_kb)+\
              "dec_rand"+str(dec_rand)+"sel_rand"+str(sel_rand)+"mem_num"+str(comb_num)+\
              "min_val"+str(min_val)+"max_val"+str(max_val)+\
              "bin_nums"+str(bin_nums)+"sample_bound"+str(sample_bound)+"n_trials"+str(n_trials)
              



    kb_builder=KnowledgeBaseBuilder(expr_name="sparse_build_headup_detect_control_mem",
                                    node_ip='172.27.151.145',
                                    node_addr="172.27.151.145:5001",
                                    query_addr="114.212.81.11:5000",
                                    service_addr="114.212.81.11:5500",
                                    query_body=query_body,
                                    conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list)
    cold_starter=ColdStartPlanner(conf_names=conf_names,
                                  serv_names=serv_names,
                                  service_info_list=service_info_list,
                                  user_constraint=query_body['user_constraint'])
    

    if need_sparse_kb==1:


        kb_builder.send_query() 

        print("开算")
        if dec_rand==1:
            mem_comb_list_dec=cold_starter.get_comb_list_dec_rand( mem_range,len(serv_names))  #获取随机下降结果
        else:
            mem_comb_list_dec=cold_starter.get_comb_list_dec_max( mem_range,len(serv_names))
        print(mem_comb_list_dec)

        if sel_rand==1:
            mem_combs=cold_starter.get_combs_rand(comb_list_dec=mem_comb_list_dec,comb_num=comb_num)
        else:
            mem_combs=cold_starter.get_combs_cert(comb_list_dec=mem_comb_list_dec,comb_num=comb_num)
        
        print(mem_combs)

        filename_list=[]  #存放对每一种内存限制进行采样的文件名称
        for comb in mem_combs:
            print(comb)
            for i in range(0,len(serv_names)):
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



        '''
        idx,val=cold_starter.find_non_zero_element_rand(numbers=numbers)
        print(idx,val)
        idx,val=cold_starter.find_non_zero_element_max(numbers=numbers)
        print(idx,val)
        '''
         
        # kb_builder.send_query() 
        # filename=kb_builder.sample_and_record_bayes(sample_bound=10,n_trials=100,bin_nums=100,bayes_goal=BEST_STD_DELAY)
            
        
        # kb_builder.send_query() 
        # 以最优化时延为目标
        # kb_builder.sample_and_record_bayes(sample_bound=10,n_trials=80,bin_nums=100,bayes_goal=BEST_ALL_DELAY)
        # 以在区间上平均为目标
        '''
        #这里包含流水线里涉及的各个服务的名称
        serv_names=["face_detection","face_alignment"]   
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
        "face_alignment_mem_util_limit":[1.0],
        "face_detection_mem_util_limit":[1.0],
        "face_alignment_trans_mem_util_limit":[1.0],
        "face_detection_trans_mem_util_limit":[1.0],
        '''
        # kb_builder.sample_and_record_bayes(sample_bound=10,n_trials=80,bin_nums=100,bayes_goal=BEST_STD_DELAY)


    
    if need_tight_kb==1:
        kb_builder.send_query() 
        kb_builder.sample_and_record(sample_bound=100) #表示对于所有配置组合每种组合采样sample_bound次。
    

    filepath='20240116_16_29_09_knowledgebase_builder_sparse_0.1_0.7_sparse_build_headup_detect.csv' #优化时延的贝叶斯稀疏采样
    filepath='20240116_17_11_50_knowledgebase_builder_sparse_0.1_0.7_sparse_build_headup_detect.csv' #优化方差的贝叶斯稀疏采样
    # 姿态估计分析
    filepath='20240116_19_07_18_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv' #10次采样分析内存限制起效时间
    filepath='20240116_19_42_18_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv' #20次采样分析内存限制起效时间
    filepath='20240116_19_55_00_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样分析内存限制起效时间
    filepath='20240116_20_35_07_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样分析，但是内存限制从大到小
    filepath='20240116_20_56_39_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样，内存从小到大，没有cpu限制变化
    filepath='20240116_21_08_50_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样，内存从大到小，没有cpu限制变化
    filepath='20240117_10_36_46_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，内存限制从小到大，其他不变
    filepath='20240117_11_00_44_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，内存限制从大到小，其他不变
    filepath='20240117_16_47_07_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #10次采样，内存限制从大到小，其他不变
    filepath='20240117_11_19_17_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样，ip变化，cpu限制从小到大
    filepath='20240117_11_30_35_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #10次采样，ip变化，cpu限制从小到大
    filepath='20240117_11_36_05_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #10次采样，ip变化，cpu限制从大到小


    #人脸检测分析
    filepath='20240117_15_15_00_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样，内存限制从小到大
    filepath='20240117_15_22_25_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，内存限制从小到大
    filepath='20240117_15_29_07_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样，内存限制从大到小
    filepath='20240117_15_34_31_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #10次采样，内训限制从大到小
    filepath='20240117_15_41_48_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #10次采样，cpu限制从小到大
    filepath='20240117_16_10_17_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样，cpu限制从小到大
    filepath='20240117_16_18_37_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样，cpu限制从大到小
    filepath='20240117_16_31_29_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，cpu限制从大到小
    # 如果想要基于need_to_test或need_to_build的结果进行可视化分析，可调用以下if条件对应的代码进行绘图。
    
    

    #姿态估计
    filepath='20240117_11_00_44_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，内存限制0.3,0.2,0.1，360p，其他不变
    filepath='20240117_19_19_36_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，内存限制0.3,0.2,0.1，1080p，其他不变
    filepath='20240117_19_30_28_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，内存限制0.25,0.15,0.05，1080p，其他不变
    filepath='20240117_19_50_17_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #20次采样，内存限制0.1,0.08,0.06,0.04,0.02，1080p，其他不变
    filepath='20240117_21_01_26_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #50次采样，内存限制0.1,0.08,0.06,0.04,0.02，1080p，其他不变
    filepath='20240117_21_12_00_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，内存限制从0.3到0.02不断递减，1080p，边端，其他不变
    filepath='20240118_10_41_26_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，cpu限制从0.3到0.02不断递减，1080p，边端，其他不变
    filepath='20240118_11_36_58_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，cpu限制从0.3到0.02不断递减，1080p，云端，其他不变

    # 人脸检测
    filepath='20240118_14_36_01_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，内存限制从0.3到0.02不断递减，1080p，边端，其他不变
    filepath='20240118_15_42_16_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，cpu限制从0.3到0.02不断递减，1080p，边端，其他不变
    filepath='20240118_16_16_31_knowledgebase_builder_sparse_0.1_0.7_tight_build_headup_detect.csv'  #100次采样，cpu限制从0.3到0.02不断递减，1080p，云端，其他不变

    filepath='20240116_17_11_50_knowledgebase_builder_sparse_0.1_0.7_sparse_build_headup_detect.csv' #优化方差的贝叶斯稀疏采样
        

    
    filepath='20240126_19_49_43_knowledgebase_builder_sparse_0.1_0.7_sparse_build_headup_detect_control_mem.csv'
    filepath='20240126_19_50_33_knowledgebase_builder_sparse_0.1_0.7_sparse_build_headup_detect_control_mem.csv'
    filepath='20240126_19_48_34_knowledgebase_builder_sparse_0.1_0.7_sparse_build_headup_detect_control_mem.csv'
    need_to_draw=0
    if need_to_draw==1:
        kb_builder.anylze_explore_result(filepath=filepath)
        kb_builder.draw_picture_from_sample(filepath=filepath)
    


    need_to_build=0
    if need_to_build==1:
        record_name="headup_detect_bayes1dec_rand1sel_rand1mem_num3min_val0.0max_val0.7bin_nums100sample_bound5n_trials10.txt"
        record_name='20240126_22_05_45headup_detect_bayes1dec_rand0sel_rand0mem_num10min_val0.0max_val0.7bin_nums100sample_bound5n_trials20.txt'
        with open(record_name, 'r') as file:
            # 逐行读取文件内容，将每行内容（即每个单词）打印出来
            i=0
            for line in file:
                print(line.strip())
                filepath=line.strip()
                if i==0:
                    kb_builder.create_evaluator_from_samples(filepath=filepath)
                else:
                    kb_builder.update_evaluator_from_samples(filepath=filepath)
                i+=1



    need_cold_start=0
    if need_cold_start==1:
        # 首先基于文件进行初始化
        cons_delay_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        n_trials_list=[100,200,300,400,500,600,700,800,900,1000]
        cons_delay_list=[0.4]
        n_trials_list=[100,100,100,100,100,100,100,100,100,100,
                       112,112,112,112,112,112,112,112,112,
                       125,125,125,125,125,125,125,125,
                       143,143,143,143,143,143,143,
                       167,167,167,167,167,167,
                       200,200,200,200,200,
                       250,250,250,250,
                       334,334,334,
                       500,500,
                       1000]
        
        record_cons_delay=[]
        
        record_n_trials=[]
        record_status=[]
        record_pred_delay=[]

        record_best_status=[]
        record_best_delay=[]

        for cons_delay in cons_delay_list:
            cold_starter.user_constraint["delay"]=cons_delay  #设置一个cons_delay时延
            # 首先，记录一下通过遍历能拿到的最优解
            conf, flow_mapping, resource_limit = cold_starter.get_coldstart_plan_rotate()
            status0,pred_delay_list0,pred_delay_total0=cold_starter.get_pred_delay(conf=conf,
                                                                flow_mapping=flow_mapping,
                                                                resource_limit=resource_limit)

            for n_trials in n_trials_list:
                conf, flow_mapping, resource_limit = cold_starter.get_coldstart_plan_bayes(n_trials=n_trials)
                status,pred_delay_list,pred_delay_total=cold_starter.get_pred_delay(conf=conf,
                                                                    flow_mapping=flow_mapping,
                                                                    resource_limit=resource_limit)

                record_cons_delay.append(cons_delay)
                record_n_trials.append(n_trials)
                record_status.append(status)
                record_pred_delay.append(pred_delay_total)
                record_best_status.append(status0)
                record_best_delay.append(pred_delay_total0)
        
        #现在要把cold_plan_list整个写入文件：
        with open(datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')+'cold_plan_list.txt', 'w') as file:
            file.writelines(str(['cons','n_trials','rotate_status','bayes_status','rotate_pred_delay','bayes_pred_delay'])+'\n')
            file.writelines(str(record_cons_delay)+'\n')
            file.writelines(str(record_n_trials)+'\n')
            file.writelines(str(record_best_status)+'\n')
            file.writelines(str(record_status)+'\n')
            file.writelines(str(record_best_delay)+'\n')
            file.writelines(str(record_pred_delay)+'\n')
            
           
        
        file.close()

    need_anylze_plan=1
    if need_anylze_plan==1:
        plan_file_name='20240127_19_51_37cold_plan_list.txt'
        plan_file_name='20240127_16_53_18cold_plan_list.txt'
        file = open(plan_file_name, "r")
        lines = file.readlines()
        file.close()
        print(lines)
        read_res_list=[]
        num=0


        for line in lines:
            
            # plan_item=list(line)
            new_line=line[:-1]
            #print(type(new_line))
            #print(new_line)
            new_line=new_line.replace("'",'')
            new_line=new_line.replace(',','')
            new_line=new_line.replace('[','')
            new_line=new_line.replace(']','')
            #print(type(new_line))
            #print(new_line)

            new_line=new_line.split(' ')
            #print(type(new_line))
            #print(new_line)

            if num>0: #第二行开始才是数字
                number_list=[float(item) for item in new_line]
                #print(type(number_list))
                #print(number_list)
                read_res_list.append(number_list)

            num+=1

        for read_res in read_res_list:
            print(read_res)
        
        cons_delay_list=read_res_list[0]
        n_trials_list=read_res_list[1]
        rotate_status_list=read_res_list[2]
        bayes_status_list=read_res_list[3]
        rotate_pred_delay=read_res_list[4]
        bayes_pred_delay=read_res_list[5]

        # 现在要将它们绘制到一幅图里。
        x_value=[]
        for i in range(0,len(n_trials_list)):
            x_value.append(i)
       

        # 现在的问题是如何在横轴上显示各个n_trials_list
        print(n_trials_list)

        
        # 现在需要绘制cons_delay_list
        fig=plt.figure()
        ax1=fig.add_subplot()
        ax1.plot(x_value,cons_delay_list,label="任务时延约束")
        
        ax1.plot(x_value,rotate_pred_delay,label="遍历查找冷所得启动时延")
        ax1.plot(x_value,bayes_pred_delay,label="贝叶斯查找所得冷启动时延")
        
        ax1.set_ylim(0,1.2)
        ax1.set_ylabel('时延$\mathrm{(s)}$',fontsize=15)
        ax1.set_xlabel('实验次数',fontsize=15)
        plt.yticks(fontsize=14)
        #plt.xticks([0,10,19,27,34,40,45,49,52,54],fontsize=14)
        plt.xticks(fontsize=14)

        #for x_line in [0,10,19,27,34,40,45,49,52,54]:
        #    plt.axvline(x=x_line, color='red')

        ax1.legend(loc='upper left',frameon=True,fontsize=14) #可以显示图例，标注label
        #plt.savefig('z_mypicture/infer_1.png', dpi=600)    # dpi     分辨率

        ax2=ax1.twinx()
        ax2.plot(x_value,n_trials_list,linestyle='--',label='贝叶斯优化采样次数')
        ax2.set_ylabel('贝叶斯优化采样次数n_trials',fontsize=15)
        ax2.set_ylim(0,10000)
        ax2.legend(loc='upper right',frameon=True,fontsize=14) #可以显示图例，标注label
        plt.yticks([100,500,1000,1500],fontsize=14)
        # plt.xticks(fontsize=14)
         # plt.yticks(fontsize=14)
        plt.show()








            

    
                




        '''
        cold_starter.user_constraint["delay"]
        n_trials=1000
        conf, flow_mapping, resource_limit = cold_starter.get_coldstart_plan_bayes(n_trials=n_trials)
        print("时延约束为",query_body['user_constraint']['delay'])
        print("基于贝叶斯查找,n_trials数目为",n_trials)
        print(conf)
        print(flow_mapping)
        print(resource_limit)
        print("展示预测时延")
        print(cold_starter.get_pred_delay(conf=conf,
                                          flow_mapping=flow_mapping,
                                          resource_limit=resource_limit))
        '''
        
        '''
        conf, flow_mapping, resource_limit = cold_starter.get_coldstart_plan_rotate()
        print("基于遍历查找")
        print(conf)
        print(flow_mapping)
        print(resource_limit)
        print("展示预测时延")
        print(cold_starter.get_pred_delay(conf=conf,
                                          flow_mapping=flow_mapping,
                                          resource_limit=resource_limit))
        '''
    
        
        

    exit()

    





    


'''
result：
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
}

resource_info：
{'cloud': {
    '114.212.81.11': {
        'cpu_ratio': 33.5, 
        'gpu_compute_utilization': {'0': 0, '1': 0, '2': 0, '3': 0}, 
        'gpu_mem_total': {'0': 24.0, '1': 24.0, '2': 24.0, '3': 24.0}, 
        'gpu_mem_utilization': {'0': 12.830352783203125, '1': 1.2865702311197915, '2': 1.2865702311197915, '3': 1.2865702311197915}, 
        'mem_ratio': 10.5, 
        'mem_total': 251.56013107299805, 
        'n_cpu': 48, 
        'net_ratio(MBps)': 0.42016, 
        'swap_ratio': 0.0}
        }, 
'host': {
    '172.27.133.85': {
        'cpu_ratio': 0.0, 
        'gpu_compute_utilization': {'0': 0.0}, 
        'gpu_mem_total': {'0': 3.9}, 
        'gpu_mem_utilization': {'0': 29.561654115334534}, 
        'mem_ratio': 79.1, 
        'mem_total': 7.675579071044922, 
        'n_cpu': 4, 
        'net_ratio(MBps)': 14.03965, 
        'swap_ratio': 0.0}
        }
}


'''
