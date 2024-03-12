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
import time
import optuna
import itertools
import random
plt.rcParams['font.sans-serif'] = ['SimHei'] # 运行配置参数中的字体（font）为黑体（SimHei）

#试图以类的方式将整个独立建立知识库的过程模块化

NO_BAYES_GOAL=0 #按照遍历配置组合的方式来建立知识库
BEST_ALL_DELAY=1 #以最小化总时延为目标，基于贝叶斯优化建立知识库（密集，而不集中）
BEST_STD_DELAY=2 #以最小化不同配置间时延的差别为目标，基于贝叶斯优化建立知识库（稀疏，而均匀）

kb_data_path="kb_data"

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
#'''
# 以下是缩小范围版，节省知识库大小
conf_and_serv_info={  #各种配置参数的可选值
    "reso":["360p","480p","720p","1080p"],
    "fps":[1, 5, 10, 20, 30],
    "encoder":["JPEG"],
    
    "face_alignment_ip":["114.212.81.11","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["114.212.81.11","172.27.151.145"],
    "car_detection_ip":["114.212.81.11","172.27.151.145"],
    "face_alignment_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_alignment_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],

    "face_alignment_trans_ip":["114.212.81.11","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["114.212.81.11","172.27.151.145"],
    "car_detection_trans_ip":["114.212.81.11","172.27.151.145"],
    "face_alignment_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_alignment_trans_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_trans_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_trans_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],

}
#'''
#'''
#"172.27.151.145"
'''
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
# 对于给定的conf_and_serv_info,在这个范围内进行知识库建立（遍历配置组合或者贝叶斯优化）
'''
'''
conf_and_serv_info={  #各种配置参数的可选值
    "reso":["360p"],
    "fps":[10],
    "encoder":["JPEG"],
    
    "face_alignment_ip":["172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["172.27.151.145"],
    "car_detection_ip":["172.27.151.145"],
    "face_alignment_mem_util_limit":[0.3],
    "face_alignment_cpu_util_limit":[0.35],
    "face_detection_mem_util_limit":[0.5],
    "face_detection_cpu_util_limit":[0.1],
    "car_detection_mem_util_limit":[1.00],
    "car_detection_cpu_util_limit":[1.0],

    "face_alignment_trans_ip":["172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["172.27.151.145"],
    "car_detection_trans_ip":["172.27.151.145"],
    "face_alignment_trans_mem_util_limit":[0.3],
    "face_alignment_trans_cpu_util_limit":[0.35],
    "face_detection_trans_mem_util_limit":[0.5],
    "face_detection_trans_cpu_util_limit":[0.1],
    "car_detection_trans_mem_util_limit":[1.00],
    "car_detection_trans_cpu_util_limit":[1.0],

}
'''
'''
conf_and_serv_info={  #各种配置参数的可选值
    "reso":["360p"],
    "fps":[5],
    "encoder":["JPEG"],
    
    "face_alignment_ip":["172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["172.27.151.145"],
    "car_detection_ip":["172.27.151.145"],
    "face_alignment_mem_util_limit":[0.3],
    "face_alignment_cpu_util_limit":[0.3],
    "face_detection_mem_util_limit":[0.1],
    "face_detection_cpu_util_limit":[0.1],
    "car_detection_mem_util_limit":[1.00],
    "car_detection_cpu_util_limit":[1.0],

    "face_alignment_trans_ip":["172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["172.27.151.145"],
    "car_detection_trans_ip":["172.27.151.145"],
    "face_alignment_trans_mem_util_limit":[0.3],
    "face_alignment_trans_cpu_util_limit":[0.3],
    "face_detection_trans_mem_util_limit":[0.1],
    "face_detection_trans_cpu_util_limit":[0.1],
    "car_detection_trans_mem_util_limit":[1.00],
    "car_detection_trans_cpu_util_limit":[1.0],

}
'''



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
        
        f=open(kb_data_path+'/'+eval_name+".json","w")
        json.dump(evaluator,f,indent=1)
        f.close()
        print("完成对评估器",eval_name,"的空白json文件初始化")
    
    
    # evaluator_load：
    # 用途：读取某个json文件里保存的字典
    # 方法：根据eval_name从当前目录下打开相应json文件并返回
    # 返回值：从文件里读取的字典
    def evaluator_load(self,eval_name):
        f=open(kb_data_path+'/'+eval_name+".json")
        evaluator=json.load(f)
        f.close()
        return evaluator  #返回加载得到的字典
    
    # evaluator_dump：
    # 用途：将字典内容重新写入文件
    # 方法：将参数里的字典写入参数里指定的eval_name文件中
    # 返回值：无
    def evaluator_dump(self,evaluator,eval_name):  #将字典重新写入文件
        f=open(kb_data_path+'/'+eval_name+".json","w")
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
        
        filename = kb_data_path+'/'+datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
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
                for cpu_limit in conf_and_serv_info[serv_cpu_limit]:
                    if cpu_limit <= self.rsc_upper_bound[serv_name]['cpu_limit']:
                        cpu_select.append(cpu_limit)
                mem_select=[]
                for mem_limit in conf_and_serv_info[serv_mem_limit]:
                    if mem_limit <= self.rsc_upper_bound[serv_name]['mem_limit']:
                        mem_select.append(mem_limit)
                
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
            with open(kb_data_path+'/'+ service_info['name']+'_conf_info'+'.json', 'w') as f:  
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

            with open(kb_data_path+'/'+ service_info['name']+'_conf_info'+'.json', 'r') as f:  
                old_conf_info = json.load(f)  
                #print(old_conf_info)

                for conf_name in service_conf: #conf_and_serv_info中包含了service_conf中所有配置的可能取值
                    temp_list=old_conf_info[conf_name]+(df[conf_name].to_list())
                    #print(temp_list)
                    conf_info[conf_name]=list(set(temp_list))
            
            #现在字典里存储着有关该服务每一个配置在当前文件夹中的所有取值，将其存入
            with open(kb_data_path+'/'+ service_info['name']+'_conf_info'+'.json', 'w') as f:  
                json.dump(conf_info, f) 



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

        x_list=[]
        for i in df['n_loop']:
            x_list.append(i)

        cons_delay=[]
        for x in df['n_loop']:
            cons_delay.append(self.query_body['user_constraint']['delay'])

        # 绘制总时延和约束时延
        self.draw_delay_and_cons(x_value1=x_list,y_value1=df['all_delay'],y_value2=cons_delay,title_name="all_delay&constraint_delay/时间")

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

            self.draw_picture(x_value=x_list,y_value=df[serv_ip_name],title_name=serv_ip_name+"/时间")
            self.draw_picture(x_value=x_list,y_value=df[serv_proc_delay_name],title_name=serv_proc_delay_name+"/时间")
            self.draw_picture(x_value=x_list,y_value=df[trans_delay_name],title_name=trans_delay_name+"/时间")

            self.draw_picture(x_value=x_list,y_value=df[mem_portrait],title_name=mem_portrait+"/时间")
            self.draw_picture(x_value=x_list,y_value=df[mem_util_limit],title_name=mem_util_limit+"/时间")

             # print(df[mem_util_use])
            self.draw_picture(x_value=x_list,y_value=df[mem_util_use],title_name=mem_util_use+"/时间")

            self.draw_picture(x_value=x_list,y_value=df[cpu_portrait],title_name=cpu_portrait+"/时间")
            self.draw_picture(x_value=x_list,y_value=df[cpu_util_limit],title_name=cpu_util_limit+"/时间")
            self.draw_picture(x_value=x_list,y_value=df[cpu_util_use],title_name=cpu_util_use+"/时间")
            
        
        for conf_name in self.conf_names:
          self.draw_picture(x_value=df['n_loop'],y_value=df[conf_name],title_name=conf_name+"/时间")
        # '''


   
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

# 描述每一种服务所需的中资源阈值，它限制了贝叶斯优化的时候采取怎样的内存取值范围
rsc_upper_bound={
    'face_detection':{
        'cpu_limit':0.4,
        'mem_limit':0.5,
    },
    'face_alignment':{
        'cpu_limit':0.4,
        'mem_limit':0.5,
    }
    
}


# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","face_alignment"]   

#以下是发出query请求时的内容。注意video_id。当前文件需要配合query_manager_v2.py运行，后者使用的调度器会根据video_id的取值判断是否会运行。
#建议将video_id设置为99，它对应的具体视频内容可以在camera_simulation里找到，可以自己定制。query_manager_v2.py的调度器发现query_id为99的时候，
#不会进行调度动作。因此，知识库建立者可以自由使用update_plan接口操控任务的调度方案，不会受到云端调度器的影响了。
'''
query_body = {
        "node_addr": "172.27.151.145:5001",
        "video_id": 99,   
        "pipeline": ["face_detection", "face_alignment"],#制定任务类型
        "user_constraint": {
            "delay": 0.6,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }  
'''
'''
#这个query_body用于测试单位的“人进入会议室”，也就是只有一张脸的情况，工况不变，不触发调度器变化
query_body = {
        "node_addr": "172.27.151.145:5001",
        "video_id": 99,     
        "pipeline": ["face_detection", "face_alignment"],#制定任务类型
        "user_constraint": {
            "delay": 0.7,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }  
'''
#这个query_body用于测试单位的“人进入会议室”，也就是只有一张脸的情况，工况不变，但是会触发调度器变化，因为ifd很小
query_body = {
        "node_addr": "172.27.151.145:5001",
        "video_id": 4,     
        "pipeline": ["face_detection", "face_alignment"],#制定任务类型
        "user_constraint": {
            "delay": 0.7,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }  
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

    # 内存限制的取值范围 稍微限制一下有助于缩小取值范围
    mem_range=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]
    
    # 贝叶斯优化时的取值范围，在以下范围内使得采样点尽可能平均
    min_val=0.0
    max_val=0.7

    # 在多大范围内取方差
    bin_nums=100

    # 取多少个典型内存限制进行贝叶斯优化
    comb_num=10 #10
    # 每一种配置下进行多少次采样
    sample_bound=5
    # 每一次贝叶斯优化时尝试多少次
    n_trials=20 #20
    #(将上述三个量相乘，就足以得到要采样的总次数，这个次数与建立知识库所需的时延一般成正比)
    
    # 是否进行稀疏采样(贝叶斯优化)
    need_sparse_kb=0
    # 是否进行严格采样（遍历所有配置）
    need_tight_kb=0
    # 是否根据某个csv文件绘制画像 
    need_to_draw=0
    # 是否需要基于初始采样结果建立一系列字典，也就是时延有关的知识库
    need_to_build=0

    #是否需要发起一次简单的查询并测试调度器的功能
    need_to_test=1


    #获取内存资源限制列表的时候，需要两步，第一步是下降，第二部是采取，两种方法都可以随机，也都可以不随机
    dec_rand=0
    sel_rand=0 

    task_name="headup_detect"

    record_name=kb_data_path+'/'+'0_'+datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')+task_name+"_"+"bayes"+str(need_sparse_kb)+\
              "dec_rand"+str(dec_rand)+"sel_rand"+str(sel_rand)+"mem_num"+str(comb_num)+\
              "min_val"+str(min_val)+"max_val"+str(max_val)+\
              "bin_nums"+str(bin_nums)+"sample_bound"+str(sample_bound)+"n_trials"+str(n_trials)
              

    kb_builder=KnowledgeBaseBuilder(expr_name="tight_build_headup_detect_people_in_mmeeting",
                                    node_ip='172.27.151.145',
                                    node_addr="172.27.151.145:5001",
                                    query_addr="114.212.81.11:5000",
                                    service_addr="114.212.81.11:5500",
                                    query_body=query_body,
                                    conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    rsc_upper_bound=rsc_upper_bound)
    if need_to_test==1:
        #kb_builder.send_query() 
        #filepath=kb_builder.just_record(record_num=200)
        filepath='kb_data/20240312_17_36_56_kb_builder_0.7_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
        kb_builder.anylze_explore_result(filepath=filepath)
        kb_builder.draw_picture_from_sample(filepath=filepath)


    if need_sparse_kb==1:

        kb_builder.send_query() 

        print("开算")
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
        kb_builder.sample_and_record(sample_bound=50) #表示对于所有配置组合每种组合采样sample_bound次。
    
    
    # 给单个采样得到的csv文件画像，filepath自由指定
    filepath='20240220_20_45_45_knowledgebase_builder_sparse_0.6_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
    filepath='kb_data/20240305_15_42_51_kb_builder_0.6_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
    filepath='kb_data/20240305_15_45_02_kb_builder_0.6_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
    filepath='kb_data/20240305_16_04_08_kb_builder_0.6_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
    filepath='kb_data/20240305_21_59_32_kb_builder_0.6_0.7_tight_build_headup_detect_people_in_mmeeting.csv'

    # 基于配置4，0.2约束下
    filepath='kb_data/20240305_22_16_38_kb_builder_0.2_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
    # 基于配置1，0.2约束下
    filepath='kb_data/20240305_22_22_48_kb_builder_0.2_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
    # 基于配置1，0.15与0.25区分配置下
    filepath='kb_data/20240305_22_31_05_kb_builder_0.2_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
     # 基于配置1，0.3，0.1约束下
    filepath='kb_data/20240305_22_41_59_kb_builder_0.2_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
    # 基于360p，5fps约束下
    filepath='kb_data/20240305_23_13_53_kb_builder_0.2_0.7_tight_build_headup_detect_people_in_mmeeting.csv'
    
    if need_to_draw==1:
        kb_builder.anylze_explore_result(filepath=filepath)
        kb_builder.draw_picture_from_sample(filepath=filepath)
    

    # 关于是否需要建立知识库：可以根据txt文件中的内容来根据采样结果建立知识库
    if need_to_build==1:
        record_name='kb_data/0_20240305_15_42_51headup_detect_bayes1dec_rand0sel_rand0mem_num10min_val0.0max_val0.7bin_nums100sample_bound5n_trials20.txt'
        record_name='kb_data/0_20240310_20_48_27headup_detect_bayes1dec_rand0sel_rand0mem_num2min_val0.0max_val0.7bin_nums100sample_bound5n_trials3.txt'
        record_name='kb_data/0_20240310_21_01_31headup_detect_bayes1dec_rand0sel_rand0mem_num10min_val0.0max_val0.7bin_nums100sample_bound5n_trials20.txt'
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
