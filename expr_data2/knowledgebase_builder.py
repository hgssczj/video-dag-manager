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

#试图以类的方式将整个独立建立知识库的过程模块化

# 以下是可能影响任务性能的可配置参数，用于指导模块化知识库的建立
model_op={  
            "model1":{
                "model_id": 0,
                "node_ip": "114.212.81.11",
                "node_role": "cloud"
            },
            "model2": {
                "model_id": 0,
                "node_ip": "172.27.133.85",
                "node_role": "host"  
            }
        }
reso_op=["1080p","720p","480p","360p"]
fps_op=[30,20,10,5,1]
encoder_op=["JPEG"]



class KnowledgeBaseBuilder():   
    
    #知识库建立者，每次运行的时候要指定本次实验名称，用于作为记录实验的文件
    def __init__(self,expr_name,node_ip,node_addr,query_addr,service_addr):
        self.expr_name = expr_name #用于构建record_fname的实验名称
        self.node_ip = node_ip #指定一个ip,用于从resurece_info获取相应资源信息
        self.node_addr = node_addr #指定node地址，向该地址更新调度策略
        self.query_addr = query_addr #指定query地址，向该地址提出任务请求并获取结果
        self.service_addr = service_addr #指定service地址，从该地址获得全局资源信息
        self.query_id = None #查询返回的id
        self.written_n_loop = dict()
        self.writer = None
        self.sample_bound = None

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

    
    #初始化模块化的性能评估器，需要根据一系列配置来建立评估器的空白字典
    def evaluator_init(self,conf_list,eval_name):  
        #conf_num是建立评估器需要的配置旋钮的总数，conf_list里包含了一系列list，其中每一个都是某一种配置的可选组合
        models_dicts=[]
        conf_num=len(conf_list)
        for i in range(0,conf_num):
            temp=dict()
            models_dicts.append(temp)
        for key in conf_list[0]:
             models_dicts[0][key]=0
        for i in range(1,conf_num):
            for key in conf_list[i]:
                models_dicts[i][key]=models_dicts[i-1]
        
        evaluator=models_dicts[conf_num-1]  #获取最终的评估器
        # 建立初始化性能评估器并写入文件
        f=open(eval_name+".json","w")
        json.dump(evaluator,f,indent=1)
        f.close()
        print("完成对评估器",eval_name,"的空白json文件初始化")
    
    # 加载存储了评估器的json文件得到字典
    def evaluator_load(self,eval_name):
        f=open(eval_name+".json")
        evaluator=json.load(f)
        f.close()
        return evaluator  #返回加载得到的字典

    #以query为参数发动查询，之后会得到query_id用于执行post_get_write等操作的基本参数
    def send_query(self,query_body):  
        # 发出提交任务的请求
        query_body['node_addr']=self.node_addr
        r = self.sess.post(url="http://{}/query/submit_query".format(self.query_addr),
                    json=query_body)
        resp = r.json()
        print(resp)
        self.query_id = resp["query_id"]
        return self.query_id

    #发出一次请求并获取结果，期间更新配置，获取情境，查询结果，然后记录。
    def post_get_write(self,conf,flow_mapping):
        #（1）更新配置
        r = self.sess.post(url="http://{}/job/update_plan".format(self.node_addr),
                        json={"job_uid": self.query_id, "video_conf": conf, "flow_mapping": flow_mapping})
        if not r.json():
            return {"status":0,"des":"fail to update plan"}
         
        #（2）获取资源情境,获取node_ip指定的边缘节点的内存使用率
        r = self.sess.get(url="http://{}/get_resource_info".format(self.service_addr))
        if not r.json():
            return {"status":1,"des":"fail to get resource info"}
        resource_info = r.json()
        edge_mem_ratio=resource_info['host'][self.node_ip]['mem_ratio']
         
        #（3）查询执行结果并处理
        r = self.sess.get(url="http://{}/query/get_result/{}".format(self.query_addr, self.query_id))  
        if not r.json():
            return {"status":2,"des":"fail to post one query request"}
         
        resp = r.json()
        appended_result = resp['appended_result'] #可以把得到的结果直接提取出需要的内容，列表什么的。
        latest_result = resp['latest_result'] #空的
        # updatetd_result用于存储本次从云端获取的有效的更新结果
        updatetd_result=[]

        for res in appended_result:
            #print("开始提取")
            n_loop=res['n_loop']
            frame_id=res['frame_id']
            total=res['count_result']['total']
            up=res['count_result']['up']

            d_role=res['ext_plan']['flow_mapping']['face_detection']['node_role']
            d_ip=res['ext_plan']['flow_mapping']['face_detection']['node_ip']
            a_role=res['ext_plan']['flow_mapping']['face_alignment']['node_role']
            a_ip=res['ext_plan']['flow_mapping']['face_alignment']['node_ip']

            encoder=res['ext_plan']['video_conf']['encoder']
            fps=res['ext_plan']['video_conf']['fps']
            reso=res['ext_plan']['video_conf']['resolution']

            obj_n=res['ext_runtime']['obj_n']
            obj_size=res['ext_runtime']['obj_size']
            obj_stable=res['ext_runtime']['obj_stable']
            all_delay=res['ext_runtime'][ 'delay']
            
            d_proc_delay=res['ext_runtime']['plan_result']['process_delay']['face_detection']
            d_trans_delay=res['ext_runtime']['plan_result']['delay']['face_detection']-d_proc_delay
            a_proc_delay=res['ext_runtime']['plan_result']['process_delay']['face_alignment']
            a_trans_delay=res['ext_runtime']['plan_result']['delay']['face_alignment']-a_proc_delay
            
            row={
                'n_loop':n_loop,
                'frame_id':frame_id,
                'total':total,
                'up':up,
                'd_role':d_role,
                'd_ip':d_ip,
                'a_role':a_role,
                'a_ip':a_ip,
                'encoder':encoder,
                'fps':fps,
                'reso':reso,
                'obj_n':obj_n,
                'obj_size':obj_size,
                'obj_stable':obj_stable,
                'all_delay':all_delay,
                'd_proc_delay':d_proc_delay,
                'd_trans_delay':d_trans_delay,
                'a_proc_delay':a_proc_delay,
                'a_trans_delay':a_trans_delay,
                'edge_mem_ratio':edge_mem_ratio
            }
            if n_loop not in self.written_n_loop:  #以字典为参数，只有那些没有在字典里出现过的row才会被写入文件，
                #print("获取新检测结果")
                #print(row)
                #print("展示当前环境")
                #print(resource_info)
                self.writer.writerow(row)
                self.written_n_loop[n_loop] = 1
                #完成文件写入之后，将对应的row和配置返回以供分析。由于存在延迟，这些新数据对应的conf和flow_mapping可能和前文指定的不同
                updatetd_result.append({"row":row,"conf":res['ext_plan']['video_conf'],"flow_mapping":res['ext_plan']['flow_mapping']})

        #updatetd_result会返回本轮真正检测到的全新数据。在最糟糕的情况下，updatetd_result会是一个空列表。
        return {"status":3,"des:":"succeed to record a row","updatetd_result":updatetd_result}
    
    # 以下函数通过反复调用post_get_write并获取updatetd_result，直到其中与配置要求匹配的结果达到了sample_bound个
    # sample_bound表示在该配置下进行采样的总次数。
    def collect_for_sample(self,conf,flow_mapping):
        sample_num=0
        sample_result=[]
        all_delay=0
        while(sample_num<self.sample_bound):# 只要已经收集的符合要求的采样结果不达标，就不断发出请求，直到在本配置下成功获取sample_bound个样本
            get_resopnse=self.post_get_write(conf=conf,flow_mapping=flow_mapping)
            if(get_resopnse['status']==3): #如果正常返回了结果，就可以从中获取updatetd_result了
                updatetd_result=get_resopnse['updatetd_result']
                # updatetd_result包含一系列形如{"row":row,"conf":res['ext_plan']['video_conf'],"flow_mapping":res['ext_plan']['flow_mapping']}
                # 对于获取的结果，首先检查其conf和flow_mapping是否符合需要，仅在符合的情况下才增加采样点
                for i in range(0,len(updatetd_result)):
                    print(updatetd_result[i])
                    if updatetd_result[i]['conf']==conf and updatetd_result[i]['flow_mapping']==flow_mapping:
                        all_delay+=updatetd_result[i]["row"]["all_delay"]
                        print("该配置符合要求，可作为采样点之一")
                        sample_num+=1

        avg_delay=all_delay/self.sample_bound
        print("完成对当前配置的",self.sample_bound,"次采样，平均时延是", avg_delay)
        return avg_delay

    # 以下函数根据配置多次调用collect_for_sample，对指定配置进行多次采样，每次采样都会成功录入相关结果到配置文件之中
    def sample_and_record(self,sample_bound):
        self.sample_bound=sample_bound
        #首先建立一个文件
        filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + os.path.basename(__file__).split('.')[0] + \
            '_' + str(query_body['user_constraint']['delay']) + \
            '_' + str(query_body['user_constraint']['accuracy']) + \
            '_' + self.expr_name + \
            '.csv'

        fp = open(filename, 'w', newline='')
        fieldnames = ['n_loop',
                    'frame_id',
                    'total',
                    'up',
                    'd_role',
                    'd_ip',
                    'a_role',
                    'a_ip',
                    'encoder',
                    'fps',
                    'reso',
                    'obj_n',
                    'obj_size',
                    'obj_stable',
                    'all_delay',
                    'd_proc_delay',
                    'd_trans_delay',
                    'a_proc_delay',
                    'a_trans_delay',
                    'edge_mem_ratio']
        self.writer = csv.DictWriter(fp, fieldnames=fieldnames)
        self.writer.writeheader()
        self.written_n_loop.clear() #用于存储各个轮的序号，防止重复记录

        #执行一次get_and_write就可以往文件里成果写入数据
        for reso in reso_op:
            for fps in fps_op:
                for model in model_op.keys():
                    conf={
                            "resolution": reso,
                            "fps": fps,
                            "encoder": "JEPG"
                        }
                    flow_mapping={
                            "face_detection": model_op[model],
                            "face_alignment": model_op[model]
                        }
           
                    # 简单起见，每次先采样5次
                    self.collect_for_sample(conf=conf,flow_mapping=flow_mapping)
                    print("完成一种配置下的数据记录")

        fp.close()
        print("记录结束，查看文件")
        

    def objective(self,trial):
        reso = trial.suggest_categorical('reso',reso_op)
        fps  = trial.suggest_categorical('fps',fps_op)
        model = trial.suggest_categorical('model',model_op.keys())
        conf={
            "resolution": reso,
            "fps": fps,
            "encoder": "JEPG"
        }
        flow_mapping={
            "face_detection": model_op[model],
            "face_alignment": model_op[model]
        }
        return self.collect_for_sample(conf=conf,flow_mapping=flow_mapping)

    # 以贝叶斯采样的方式获取离线知识库
    def sample_and_record_bayes(self,sample_bound,n_trials):
        self.sample_bound=sample_bound
        #首先建立一个文件
        filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + os.path.basename(__file__).split('.')[0] + \
            '_' + str(query_body['user_constraint']['delay']) + \
            '_' + str(query_body['user_constraint']['accuracy']) + \
            '_' + self.expr_name + \
            '.csv'

        fp = open(filename, 'w', newline='')
        fieldnames = ['n_loop',
                    'frame_id',
                    'total',
                    'up',
                    'd_role',
                    'd_ip',
                    'a_role',
                    'a_ip',
                    'encoder',
                    'fps',
                    'reso',
                    'obj_n',
                    'obj_size',
                    'obj_stable',
                    'all_delay',
                    'd_proc_delay',
                    'd_trans_delay',
                    'a_proc_delay',
                    'a_trans_delay',
                    'edge_mem_ratio']
        self.writer = csv.DictWriter(fp, fieldnames=fieldnames)
        self.writer.writeheader()
        self.written_n_loop.clear() #用于存储各个轮的序号，防止重复记录

        #执行一次get_and_write就可以往文件里成果写入数据
        study = optuna.create_study()
        study.optimize(self.objective,n_trials=n_trials)

        print(study.best_params['model'],study.best_params['reso'],study.best_params['fps'])

        fp.close()
        print("记录结束，查看文件")
        
        
  





#可能需要建一个类来专门负责朝云端发送请求
#以下是发出查询的基本内容，指定了Node地址和视频id，以及流水线要求、用户约束
query_body = {
        "node_addr": "172.27.133.85:7001",
        "video_id": 99,   #99号是专门用于离线启动知识库的测试视频
        "pipeline": ["face_detection", "face_alignment"],#制定任务类型
        "user_constraint": {
            "delay": 0.3,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }

if __name__ == "__main__":

    kb_builder=KnowledgeBaseBuilder(expr_name="headup-detect_video99_8confs_bayes",
                                    node_ip='172.27.133.85',
                                    node_addr="172.27.133.85:7001",
                                    query_addr="114.212.81.11:7000",
                                    service_addr="114.212.81.11:7500")
    #发出查询请求
    kb_builder.send_query(query_body=query_body) 
    #kb_builder.sample_and_record(sample_bound=7) #调用函数以5为每种配置的采样大小进行循环采样
    kb_builder.sample_and_record_bayes(sample_bound=5,n_trials=40)

    '''
        node_ip='172.27.133.85'
        node_addr = "172.27.133.85:7001" #指定边缘地址
        query_addr = "114.212.81.11:7000" #指定云端地址
        service_addr="114.212.81.11:7500" #云端服务地址
        '''
    '''
    conf_list=[encoder_op,fps_op,reso_op,model_op.keys()]  #使用四组初始配置自动生成字典
    kb_builder.evaluator_init(conf_list=conf_list,eval_name="new_face_detect")
    print( kb_builder.evaluator_load('new_face_detect'))
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
            'video_conf':   {'encoder': 'JEPG', 'fps': 1, 'resolution': '360p'}
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
