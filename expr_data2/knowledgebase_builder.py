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

#试图以类的方式将整个独立建立知识库的过程模块化

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

conf_and_serv_info={  #各种配置参数的可选值
    "reso":["360p", "480p", "720p", "1080p"],
    "fps":[1, 5, 10, 20, 30],
    "encoder":["JPEG"],
    "face_alignment_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],
    "face_alignment_trans_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["114.212.81.11","172.27.143.164","172.27.151.145"],
}

# 我人为以下的servcie_info_list实际上应该根据流水线的需要随时组建

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



conf_names=["reso","fps","encoder"]   #这里存储流水线任务涉及的所有配置参数组合，但不报考任务卸载flow_mapping对应的内容
serv_names=["face_detection","face_alignment"]   #这里包含流水线里涉及的各个服务的名称

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
        for key in conf_list[conf_num-1]:
             models_dicts[conf_num-1][key]=0
        for i in range(1,conf_num):
            for key in conf_list[conf_num-i-1]:
                models_dicts[conf_num-i-1][key]=models_dicts[conf_num-i]
        
        evaluator=models_dicts[0]  #获取最终的评估器
        print(evaluator)
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
    
    def evaluator_dump(self,evaluator,eval_name):  #将字典重新写入文件
        f=open(eval_name+".json","w")
        json.dump(evaluator,f,indent=1)
        f.close()

    #以query为参数发动查询，之后会得到query_id用于执行post_get_write等操作的基本参数
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
    

    # 将post_get_write获取的结果全部写入到文件之中。从文件中提取列的方式应该根据任务类型决定
    # 未来工况情境和资源情境都可以直接通过app_server的接口获取，所以从配置文件中只需要读取

    # 以下函数用于创建一个记录采样结果的文件
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
        for i in range(0,len(self.conf_names)):
            fieldnames.append(self.conf_names[i])   #得到配置名

        #serv_names形如['face_detection', 'face_alignment']
        for i in range(0,len(self.serv_names)):
            serv_name=self.serv_names[i]
            
            serv_role_name=serv_name+'_role'
            serv_ip_name=serv_name+'_ip'
            serv_proc_delay_name=serv_name+'_proc_delay'

            trans_ip_name=serv_name+'_trans_ip'
            trans_delay_name=serv_name+'_trans_delay'

            fieldnames.append(serv_role_name)
            fieldnames.append(serv_ip_name)
            fieldnames.append(serv_proc_delay_name)
            fieldnames.append(trans_ip_name)
            fieldnames.append(trans_delay_name)

        self.writer = csv.DictWriter(self.fp, fieldnames=fieldnames)
        self.writer.writeheader()
        self.written_n_loop.clear() #用于存储各个轮的序号，防止重复记录

        return filename



    def write_in_file(self,r2,r3):   #pipeline指定了任务类型   
        resource_info = r2.json()
        resp = r3.json()

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

            '''
            encoder=res['ext_plan']['video_conf']['encoder']
            fps=res['ext_plan']['video_conf']['fps']
            reso=res['ext_plan']['video_conf']['resolution']
   
            d_proc_delay=res['ext_runtime']['plan_result']['process_delay']['face_detection']
            d_trans_delay=res['ext_runtime']['plan_result']['delay']['face_detection']-d_proc_delay
            a_proc_delay=res['ext_runtime']['plan_result']['process_delay']['face_alignment']
            a_trans_delay=res['ext_runtime']['plan_result']['delay']['face_alignment']-a_proc_delay

            all_delay=res['ext_runtime'][ 'delay']
            '''
                       
            n_loop=res['n_loop']
            if n_loop not in self.written_n_loop:  #以字典为参数，只有那些没有在字典里出现过的row才会被写入文件，
                self.writer.writerow(row)
                print("写入成功")
                self.written_n_loop[n_loop] = 1
                #完成文件写入之后，将对应的row和配置返回以供分析。由于存在延迟，这些新数据对应的conf和flow_mapping可能和前文指定的不同
                updatetd_result.append({"row":row,"conf":res['ext_plan']['video_conf'],"flow_mapping":res['ext_plan']['flow_mapping']})

        #updatetd_result会返回本轮真正检测到的全新数据。在最糟糕的情况下，updatetd_result会是一个空列表。
        return updatetd_result


    #发出一次请求并获取结果，期间更新配置，获取情境，查询结果，然后记录。
    def post_get_write(self,conf,flow_mapping):
        # print("开始发出消息并配置")
        #（1）更新配置
        r1 = self.sess.post(url="http://{}/job/update_plan".format(self.node_addr),
                        json={"job_uid": self.query_id, "video_conf": conf, "flow_mapping": flow_mapping})
        if not r1.json():
            return {"status":0,"des":"fail to update plan"}
         
        #（2）获取资源情境,获取node_ip指定的边缘节点的内存使用率
        r2 = self.sess.get(url="http://{}/get_resource_info".format(self.service_addr))
        if not r2.json():
            return {"status":1,"des":"fail to get resource info"}
        else:
            print("收到资源情境为:")
            print(r2.json())
          
        #（3）查询执行结果并处理
        r3 = self.sess.get(url="http://{}/query/get_result/{}".format(self.query_addr, self.query_id))  
        if not r3.json():
            return {"status":2,"des":"fail to post one query request"}
        
                # (4) 查看当前运行时情境
        r4 = self.sess.get(url="http://{}/query/get_runtime/{}".format(self.query_addr, self.query_id))  
        if not r4.json():
            return {"status":2,"des":"fail to post one query request"}
        else:
            print("收到运行时情境为:")
            print(r4.json())
        
        # 如果r1 r2 r3都正常
        updatetd_result=self.write_in_file(r2=r2,r3=r3)

        return {"status":3,"des:":"succeed to record a row","updatetd_result":updatetd_result}
    
    # 不更新配置，仅仅是获取结果
    def get_write(self):
         
        #（1）获取资源情境,获取node_ip指定的边缘节点的内存使用率
        r2 = self.sess.get(url="http://{}/get_resource_info".format(self.service_addr))
        if not r2.json():
            return {"status":1,"des":"fail to get resource info"}
        else:
            print("收到资源情境为:")
            print(r2.json())
          
        #（2）查询执行结果并处理
        r3 = self.sess.get(url="http://{}/query/get_result/{}".format(self.query_addr, self.query_id))  
        if not r3.json():
            return {"status":2,"des":"fail to post one query request"}
        
        # (4) 查看当前运行时情境
        r4 = self.sess.get(url="http://{}/query/get_runtime/{}".format(self.query_addr, self.query_id))  
        if not r4.json():
            return {"status":2,"des":"fail to post one query request"}
        else:
            print("收到运行时情境为:")
            print(r4.json())
        # 如果r1 r2 r3都正常
        updatetd_result=self.write_in_file(r2=r2,r3=r3)

        return {"status":3,"des:":"succeed to record a row","updatetd_result":updatetd_result}



    # 以下函数通过反复调用post_get_write并获取updatetd_result，sample_bound表示在该配置下进行采样的总次数。
    def collect_for_sample(self,conf,flow_mapping):
        sample_num=0
        sample_result=[]
        all_delay=0
        while(sample_num<self.sample_bound):# 只要已经收集的符合要求的采样结果不达标，就不断发出请求，直到在本配置下成功获取sample_bound个样本
            get_resopnse=self.post_get_write(conf=conf,flow_mapping=flow_mapping)
            if(get_resopnse['status']==3): #如果正常返回了结果，就可以从中获取updatetd_result了
                updatetd_result=get_resopnse['updatetd_result']
                # print("展示updated_result")
                # print(updatetd_result)
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
    
    def just_record_with_static_plan(self,record_num,conf,flow_mapping):
        filename=self.init_record_file()
        record_sum=0
        while(record_sum<record_num):
            get_resopnse=self.post_get_write(conf=conf,flow_mapping=flow_mapping)
            if(get_resopnse['status']==3):
                updatetd_result=get_resopnse['updatetd_result']
                for i in range(0,len(updatetd_result)):
                    print(updatetd_result[i])
                    record_sum+=1

        self.fp.close()
        print("记录结束，查看文件")
        return filename 



    # 以下函数根据配置多次调用collect_for_sample，对指定配置进行多次采样，每次采样都会成功录入相关结果到配置文件之中
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

        conf_combine=itertools.product(*conf_list)
        for conf_plan in conf_combine:
            serv_ip_combine=itertools.product(*serv_ip_list)
            for serv_ip_plan in serv_ip_combine:# 遍历所有配置和卸载策略组合
                conf={}
                flow_mapping={}
                for i in range(0,len(self.conf_names)):
                    conf[self.conf_names[i]]=conf_plan[i]
                for i in range(0,len(self.serv_names)):
                    flow_mapping[self.serv_names[i]]=model_op[serv_ip_plan[i]]
                self.collect_for_sample(conf=conf,flow_mapping=flow_mapping)
                print("完成一种配置下的数据记录")
                
        '''
        for reso in conf_and_serv_info['reso']:
            for fps in conf_and_serv_info["fps"]:
                for model in model_op.keys():
                    conf={
                            "reso": reso,
                            "fps": fps,
                            "encoder": "JEPG"
                        }
                    flow_mapping={
                            "face_detection": model_op[model],
                            "face_alignment": model_op[model]
                        }
                    
                    self.collect_for_sample(conf=conf,flow_mapping=flow_mapping)
                    print("完成一种配置下的数据记录")
        '''
        # 最后一定要关闭文件
        self.fp.close()
        print("记录结束，查看文件")
        return filename 
        

    def objective(self,trial):
        conf={}
        flow_mapping={}

        for conf_name in self.conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
            # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
            conf[conf_name]=trial.suggest_categorical(conf_name,conf_and_serv_info[conf_name])
 
        for serv_name in self.serv_names:
            serv_ip=serv_name+"_ip"
            flow_mapping[serv_name]=model_op[trial.suggest_categorical(serv_ip,conf_and_serv_info[serv_ip])]
        
        '''
        reso = trial.suggest_categorical('reso',conf_and_serv_info['reso'])
        fps  = trial.suggest_categorical('fps',conf_and_serv_info['fps'])
        model = trial.suggest_categorical('model',model_op.keys())
        conf={
            "reso": reso,
            "fps": fps,
            "encoder": "JEPG"
        }
        flow_mapping={
            "face_detection": model_op[model],
            "face_alignment": model_op[model]
        }
        '''

        return self.collect_for_sample(conf=conf,flow_mapping=flow_mapping)

    # 以贝叶斯采样的方式获取离线知识库
    def sample_and_record_bayes(self,sample_bound,n_trials):
        self.sample_bound=sample_bound

        filename=self.init_record_file()

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
    def create_evaluator_from_samples(self,filepath):
        df = pd.read_csv(filepath)
        # 首先，完成对每一个服务的性能评估器的初始化,并提取其字典，加入到性能评估器列表中
        for service_info in self.service_info_list:
            # 对于每一个服务，首先确定服务有哪些配置，构建conf_list
            service_conf=list(service_info['conf']) # 形如":["reso","fps"]
            # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
            service_conf.append(service_info['name']+'_ip')
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
                # conf_for_dict形如['114.212.81.11', '360p', '1']。
                # conf_for_file形如（'114.212.81.11', '360p', 1），是元组且保留整数类型，用于从df中提取数据
                condition_all=True  #用于检索字典所需的条件
                for i in range(0,len(conf_for_dict)):
                    condition=df[service_conf[i]]==conf_for_file[i]    #相应配置名称对应的配置参数等于当前配置组合里的配置
                    condition_all=condition_all&condition
                # 联立所有条件从df中获取对应内容,conf_df里包含满足所有条件的列构成的df
                conf_df=df[condition_all]
                if(len(conf_df)>0): #如果满足条件的内容不为空，可以开始用其中的数值来初始化字典
                    conf_kind+=1
                    avg_value=conf_df[service_info['value']].mean()  #获取均值
                    # print(conf_df[['d_ip','a_ip','fps','reso',service_info['value']]])
                    sub_evaluator=evaluator
                    for i in range(0,len(conf_for_dict)-1):
                        sub_evaluator=sub_evaluator[conf_for_dict[i]]
                    sub_evaluator[conf_for_dict[len(conf_for_dict)-1]]=avg_value
            #完成对evaluator的处理
            self.evaluator_dump(evaluator=evaluator,eval_name=service_info['name'])
            print("该服务",service_info['name'],"涉及配置组合总数",conf_kind)
    

    def draw_picture(self,x_value,y_value,title_name):
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'
        plt.yticks(fontproperties='Times New Roman', )
        plt.xticks(fontproperties='Times New Roman', )
        plt.plot(x_value,y_value)
        plt.title(title_name)
        plt.show()
    
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

    
    '''
    n_loop,frame_id,all_delay,edge_mem_ratio,reso,fps,encoder, 
    1,3.0,1.2593842148780823,46.0,1080p,10,JPEG,

    face_detection_role,face_detection_ip,face_detection_proc_delay,face_detection_trans_ip,face_detection_trans_delay,
    cloud,114.212.81.11,0.404582679271698,114.212.81.11,0.11322259902954102,

    face_alignment_role,face_alignment_ip,face_alignment_proc_delay,face_alignment_trans_ip,face_alignment_trans_delay
    host,172.27.151.145,0.7347646951675415,172.27.151.145,0.006814241409301758

    '''
    '''
    

    query_body = {
        "node_addr": "172.27.143.164",
        "video_id": 100,   #100号是测试调度器的
        "pipeline": ["face_detection", "face_alignment"],#制定任务类型
        "user_constraint": {
            "delay": 0.3,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }  
    '''
    def draw_picture_from_sample(self,filepath): #根据文件采样结果绘制曲线
        df = pd.read_csv(filepath)
        df = df.drop(index=[0])

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

        '''
        for serv_name in self.serv_names:
            serv_role_name=serv_name+'_role'
            serv_ip_name=serv_name+'_ip'
            serv_proc_delay_name=serv_name+'_proc_delay'
            trans_ip_name=serv_name+'_trans_ip'
            trans_delay_name=serv_name+'_trans_delay'
            # 绘制各个服务的处理时延以及ip变化

            self.draw_picture(x_value=df['n_loop'],y_value=df[serv_ip_name],title_name=serv_ip_name+"/时间")
            self.draw_picture(x_value=df['n_loop'],y_value=df[serv_proc_delay_name],title_name=serv_proc_delay_name+"/时间")
            self.draw_picture(x_value=df['n_loop'],y_value=df[trans_delay_name],title_name=trans_delay_name+"/时间")

        
        for conf_name in self.conf_names:
            self.draw_picture(x_value=df['n_loop'],y_value=df[conf_name],title_name=conf_name+"/时间")
        '''




query_body = {
        "node_addr": "172.27.151.145:5001",
        "video_id": 99,   #如果需要建立新的知识库，此处video_id应该改为99
        "pipeline": ["face_detection", "face_alignment"],#制定任务类型
        "user_constraint": {
            "delay": 0.1,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }  

if __name__ == "__main__":

    kb_builder=KnowledgeBaseBuilder(expr_name="headup-detect_video100_verify",
                                    node_ip='172.27.151.145',
                                    node_addr="172.27.151.145:5001",
                                    query_addr="114.212.81.11:5000",
                                    service_addr="114.212.81.11:5500",
                                    query_body=query_body,
                                    conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list)

    # 在一个静态配置之下持续采样获取数据 使用此方法时采用99作为视频编号
    need_to_verify=1  
    if need_to_verify==1:
        kb_builder.send_query() 
        conf={
            "reso": "360p",
            "fps": 1,
            "encoder": "JEPG"
        }
        flow_mapping={
            "face_detection": {
                "model_id": 0,
                "node_ip": "172.27.151.145",
                "node_role": "host"  
            },
            "face_alignment": {
                "model_id": 0,
                "node_ip": "114.212.81.11",
                "node_role": "cloud"
            }
        }
        kb_builder.just_record_with_static_plan(record_num=50,conf=conf,flow_mapping=flow_mapping)

    
   # 如果只想发出一个查询并查看任务的执行情况，使用以下if条件对应的代码。此时，查询发出的query_body内的video_id最好不要是99，
   # 因为这种情况下云端会认为当前处于知识库建立阶段而不执行调度。
   # 云端调度器仅在video_id不为99的时候才会发挥调度作用。因此video_id为99的视频是专门在知识库建立过程中用于采样的。
    need_to_test=0
    if need_to_test==1: 
        kb_builder.send_query() 
        kb_builder.just_record(record_num=50)
    
    # 如果想要建立知识库，使用以下if条件对应的代码。sample_and_record是循环每一种配置的采样的方法。sample_and_record_bayes基于贝叶斯进行采样。
    # sample_bound表示对于每一种配置进行采样的次数，会取采样次数的平均值来作为这种配置下对应的性能指标。
    # sample_and_record_bayes方法的n_trials参数用于指定进行多少次贝叶斯优化的采样。
    # 执行完以下代码后，会得到一个本目录下的文件，记录所有的采样结果。根据这个文件可以提取得到json形式的知识库。
    need_to_build=0
    if need_to_build==1:
        kb_builder.send_query() 
        kb_builder.sample_and_record(sample_bound=10) #调用函数以5为每种配置的采样大小进行循环采样
        # filename=kb_builder.sample_and_record_bayes(sample_bound=10,n_trials=80)
    
    filepath='20231130_21_16_40_knowledgebase_builder_0.3_0.7_headup-detect_video99_newbuilder_bayes.csv'
    filepath='20231130_19_44_42_knowledgebase_builder_0.3_0.7_headup-detect_video99_newbuilder_rotate.csv'
    filepath='20231203_17_17_09_knowledgebase_builder_0.3_0.7_headup-detect_video99_newbuilder_bayes.csv'
    filepath='20231203_17_37_27_knowledgebase_builder_0.3_0.7_headup-detect_video99_newbuilder_rotate.csv'
    
    filepath='20231203_17_17_09_knowledgebase_builder_0.3_0.7_headup-detect_video99_newbuilder_bayes.csv'

    filepath='20231205_15_44_34_knowledgebase_builder_0.4_0.7_headup-detect_video100_test.csv'
    filepath='20231205_15_46_10_knowledgebase_builder_0.3_0.7_headup-detect_video100_test.csv'
    filepath='20231205_15_48_05_knowledgebase_builder_0.2_0.7_headup-detect_video100_test.csv'
    filepath='20231205_15_50_36_knowledgebase_builder_0.1_0.7_headup-detect_video100_test.csv'
    filepath='20231205_15_42_28_knowledgebase_builder_0.5_0.7_headup-detect_video100_test.csv' 
    
    
    
    
    
    # 如果想要基于need_to_test或need_to_build的结果进行可视化分析，可调用以下if条件对应的代码进行绘图。
    need_to_draw=0
    if need_to_draw==1:
        kb_builder.draw_picture_from_sample(filepath=filepath)

    # 如果想要基于need_to_test或need_to_build的结果建立知识库，可调用以下if条件对应的代码在本目录下生成各个模块的知识库，以json文件形式存储。
    need_to_create=0
    if need_to_create==1:
        kb_builder.create_evaluator_from_samples(filepath=filepath)

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
            'video_conf':   {'encoder': 'JEPG', 'fps': 1, 'reso': '360p'}
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
