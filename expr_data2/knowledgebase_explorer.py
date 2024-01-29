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
plt.rcParams['font.sans-serif'] = ['SimHei'] # 运行配置参数中的字体（font）为黑体（SimHei）



#'''
# 从x0到x9，一共10个配置参数，每一种有10个取值，结合起来是10的10次方，1 00 0000 0000种配置组合
conf_and_serv_info={  
    "x0":[0,1,2,3,4,5,6,7,8,9], 
    "x1":[0,1,2,3,4,5,6,7,8,9],
    "x2":[0,1,2,3,4,5,6,7,8,9],
    "x3":[0,1,2,3,4,5,6,7,8,9],
    "x4":[0,1,2,3,4,5,6,7,8,9],
    "x5":[0,1,2,3,4,5,6,7,8,9],
    "x6":[0,1,2,3,4,5,6,7,8,9],
    "x7":[0,1,2,3,4,5,6,7,8,9],
    "x8":[0,1,2,3,4,5,6,7,8,9],
    "x9":[0,1,2,3,4,5,6,7,8,9],
}
#'''


conf_names=["x0","x1","x2","x3","x4","x5","x6","x7"]


# 每一个特定任务对应一个KnowledgeBaseBuilder类
class KnowledgeBaseBuilder():   
    
    #知识库建立者，每次运行的时候要指定本次实验名称，用于作为记录实验的文件
    def __init__(self,expr_name,node_ip,node_addr,query_addr,service_addr,query_body,conf_names,serv_names,service_info_list):
        self.expr_name = expr_name #用于构建record_fname的实验名称
        self.node_ip = node_ip #指定一个ip,用于从resurece_info获取相应资源信息
        self.node_addr = node_addr #指定node地址，向该地址更新调度策略
        self.query_addr = query_addr #指定query地址，向该地址提出任务请求并获取结果
        self.service_addr = service_addr #指定service地址，从该地址获得全局资源信息
        
        self.query_body = query_body
        self.conf_names = conf_names
        self.serv_names = serv_names
        self.service_info_list=service_info_list


        self.query_id = None #查询返回的id
        self.fp=None
        self.written_n_loop = dict()
        self.writer = None
        self.sample_bound = None

        # 用于记录字典查询过程中设计到的配置，不希望其重复
        self.explore_conf_dict=dict()

        self.sess = requests.Session()  #客户端通信
    
    def set_bin_nums(self,bin_nums):
        self.bin_nums=bin_nums

    def set_bayes_goal(self,bayes_goal):    #设置贝叶斯函数的优化目标
        print("设置完成")
        self.bayes_goal=bayes_goal

    # 绘制散点图
    def draw_scatter(self,x_value,y_value,title_name):
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'
        plt.yticks(fontproperties='Times New Roman', )
        plt.xticks(fontproperties='Times New Roman', )
        plt.scatter(x_value,y_value,s=0.5)
        plt.title(title_name)
        plt.show()
    
    # 绘制柱状图，需要指定bins
    def draw_hist(self,data,title_name,bins):
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'
        plt.yticks(fontproperties='Times New Roman', )
        plt.xticks(fontproperties='Times New Roman', )
        plt.ylim(0,175)
        a,b,c=plt.hist(x=data,bins=bins)
        plt.title(title_name)
        plt.show()
        return a,b,c

    # 将字典explore_conf_dict中的键值对的值提取出来构建数组，返回方差
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
    

    def anylze_explore_result(self,filepath):  #分析记录下来的文件结果，也就是采样结果
        
        df = pd.read_csv(filepath)

        x_list=[i for i in range(0,len(df))]
        soretd_value=sorted(df['value'])
        a,b,c=self.draw_hist(data=soretd_value,title_name='分布',bins=100)
        
        a=list(a)
        
        a=np.array(a)
        print(a)
        print(a.std())
        print(sum(a))

    
    # 用于将explore_conf_dict字典里保存的无重复的采样结果存储到配置文件之中，其涉及的配置总数往往小于实际的采样数目，因为贝叶斯采样可能会重复对相同的配置采样
    def trans_dict_into_file(self): 
        
        #首先建立文件
        filename = 'explore_test_' + \
            '_' + datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + os.path.basename(__file__).split('.')[0] + \
            '_' + self.expr_name + \
            '.csv'

        self.fp = open(filename, 'w', newline='')
        fieldnames=[]
         #得到配置名
        for i in range(0,len(self.conf_names)):
            fieldnames.append(self.conf_names[i])  
        fieldnames.append("value")

        self.writer = csv.DictWriter(self.fp, fieldnames=fieldnames)
        self.writer.writeheader()

        #然后开始写入
        for conf_str in self.explore_conf_dict:
            #conf_str形如"['x0':1, 'x1':2]"
            conf=json.loads(conf_str)
            row=conf
            row['value']=self.explore_conf_dict[conf_str]
            self.writer.writerow(row)
        
        # self.explore_conf_dict.clear()

    def anylze_explore_result(self,filepath):  #分析记录下来的文件结果，也就是采样结果
        
        df = pd.read_csv(filepath)

        x_list=[i for i in range(0,len(df))]
        soretd_value=sorted(df['value'])
        a,b,c=self.draw_hist(data=soretd_value,title_name='分布',bins=100)
        
        a=list(a)
        
        a=np.array(a)
        print(a)
        print(a.std())
        print(sum(a))


    # 用于根据conf配置返回一个采样值，其中会将配置本身和采样结果都记录在字典explore_conf_dict之内
    def sample_test(self,conf):
        value=0
        for conf_name in conf:
            base_num=10**int(conf_name[-1])  #对于“x0”,所得base_num是1；对于“x1”，所得base_num是10
            value+=base_num*conf[conf_name]
        
        conf_str=json.dumps(conf)  #将配置转化为一个字符串
        print(conf_str,value)
        self.explore_conf_dict[conf_str]=value  #将所有结果保存在字典里
        
        return value

    # 用于作为贝叶斯优化中的待优化执行函数，优化目标应该是让结果尽可能均匀
    def objective_explore(self,trial):
        conf={}
        for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
            # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
            conf[conf_name]=trial.suggest_categorical(conf_name,conf_and_serv_info[conf_name])

        
        
        value=self.sample_test(conf)  #获取采样结果
        #print("当前已采样配置总数为",len(self.explore_conf_dict))
        #print(self.explore_conf_dict)
        std_value=self.exmaine_explore_distribution()

        print("当前已采样配置总数为",len(self.explore_conf_dict),"  采样方差为",std_value)
        
        if self.bayes_goal==1:
            return value  #最小估计值作为贝叶斯的依据
        elif self.bayes_goal==2:
            return std_value  #标准差作为贝叶斯的依据

    # 以贝叶斯采样的方式获取离线知识库
    def sample_and_explore_bayes_test(self,n_trials,bin_nums,bayes_goal):
        self.explore_conf_dict.clear()

        # 最开始必须设置一下最大和最小
        max_conf={}
        min_conf={}
        for conf_name in self.conf_names:
            max_conf[conf_name]=9
            min_conf[conf_name]=0
        
        self.sample_test(min_conf)
        self.sample_test(max_conf)

        self.set_bin_nums(bin_nums=bin_nums)  #先切分为50份吧
        self.set_bayes_goal(bayes_goal=bayes_goal)


        assert self.bin_nums
        assert self.bayes_goal


        study = optuna.create_study()
        study.optimize(self.objective_explore,n_trials=n_trials)
        self.trans_dict_into_file()
        print("记录结束，查看文件")
        
        # 对于这个文件，需要统计其中经过的不同配置并根据具体对应的value大小将其罗列出来
        print(type(study.best_params))
        print(study.best_params)

        return study.best_params

    


# 尝试进行探索
service_info_list=[
]

serv_names=[]   
query_body = {
    }  

if __name__ == "__main__":

    bayes_goal=2

    kb_builder=KnowledgeBaseBuilder(expr_name="bayes_goal_"+str(bayes_goal),
                                    node_ip='172.27.151.145',
                                    node_addr="172.27.151.145:5001",
                                    query_addr="114.212.81.11:5000",
                                    service_addr="114.212.81.11:5500",
                                    query_body=query_body,
                                    conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list)

    
    need_to_explore=0
    if need_to_explore==1:
        filename=kb_builder.sample_and_explore_bayes_test(n_trials=50,bin_nums=200,bayes_goal=bayes_goal)
    
    need_to_anylze=1
    if need_to_anylze==1:
        filepath='explore_test__20231219_17_41_10_knowledgebase_explorer_bayes_goal_1.csv'
        filepath='explore_test__20231219_17_44_24_knowledgebase_explorer_bayes_goal_2.csv'
        # filepath='explore_test__20231220_12_28_17_knowledgebase_explorer_bayes_goal_2.csv'
        # filepath='explore_test__20231220_12_32_20_knowledgebase_explorer_bayes_goal_2.csv'
        # filepath='explore_test__20231220_12_34_17_knowledgebase_explorer_bayes_goal_2.csv'
        kb_builder.anylze_explore_result(filepath=filepath)



    exit()

    





    
