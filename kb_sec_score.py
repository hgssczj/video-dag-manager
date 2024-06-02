# 该文件用于给知识库的可用性进行打分

import json
import datetime
import matplotlib.pyplot as plt
import optuna
import itertools
import logging
import field_codec_utils
import cv2
from logging_utils import root_logger
optuna.logging.set_verbosity(logging.WARNING)  

plt.rcParams['font.sans-serif'] = ['SimHei'] # 运行配置参数中的字体（font）为黑体（SimHei）

import common
from common import KB_DATA_PATH,MAX_NUMBER, model_op, conf_and_serv_info, service_info_dict
from common import MIN_DELAY,MAX_ACCURACY,MIN_RESOURCE
from AccuracyPrediction import AccuracyPrediction
from kb_sec_user import KnowledgeBaseUser



class KnowledgeBaseScorer():   
    
    #知识库建立者，每次运行的时候要指定本次实验名称，用于作为记录实验的文件
    def __init__(self,conf_names,serv_names,service_info_list,work_condition_range,user_constraint_range,rsc_constraint_range,bandwidth_range):
        self.conf_names = conf_names
        self.serv_names = serv_names
        self.service_info_list = service_info_list
        self.work_condition_range = work_condition_range
        self.user_constraint_range = user_constraint_range
        self.rsc_constraint_range = rsc_constraint_range
        self.bandwidth_range = bandwidth_range

    # runtime_simulator
    # 用途：生成工况、画像、带宽的不同组合，模拟不同的工况情境，每一种情境都有一个唯一的编号
    #      如下，生成不同的  work_condition，portrait_info，bandwidth_dict

    '''

    情境参数 目标数量取3种 大小3种 速度5种 带宽5种 一共225种
    约束参数 时延约束3种 精度约束3种 资源约束3种 一共27种
    这样一共是2025种 一个文件就可以放下。
    这6075种情境 每一种都有两个优化目标(时延优先 精度优先)
    确定情境和优化目标之后,可以从如下角度评估知识库：
    尝试不同的n_trials大小,评估每一种情况下能找到的解的质量(如果在约束内，以精度/时延为质量;不在约束内则为0)
    按理来说,对于每一种n_trials贝叶斯优化查到的解都是不一样的,因此应该多多尝试才对,把每一次尝试得到的分数相加，或者求平均值才对
    '''
    # 用于列举各种情境参数和约束参数，并将其记录到文件中，形如obj_n=10-obj_size=150000-obj_speed=800-delay=0.4-accuracy=0.3

    def generate_runtime(self,runtime_filename):
        # 在文件中写入一系列情境
        with open('kb_score/'+runtime_filename+'.txt', 'w', encoding='utf-8') as f:  
            # 数量，大小，速度，带宽
            for obj_n in self.work_condition_range["obj_n"]:
                for obj_size in self.work_condition_range["obj_size"]:
                    for obj_speed in self.work_condition_range["obj_speed"]:
                        for bandwidth in self.bandwidth_range:
                            # 时延，精度，资源约束
                            for delay in self.user_constraint_range['delay']:
                                for accuracy in self.user_constraint_range["accuracy"]:
                                    for edge_cpu in self.rsc_constraint_range:
                                        # 建造runtime
                                        runtime='obj_n='+str(obj_n)+'-'+\
                                                'obj_size='+str(obj_size)+'-'+\
                                                'obj_speed='+str(obj_speed)+'-'+\
                                                'bandwidth='+str(bandwidth)+'-'+\
                                                'delay='+str(delay)+'-'+\
                                                'accuracy='+str(accuracy)+'-'+\
                                                'edge_cpu='+str(edge_cpu)
                                        f.write(f"{runtime}\n") 
        print('完成写入')
    
    # 从形如obj_n=10-obj_size=150000-obj_speed=800-delay=0.4-accuracy=0.3的字符串中提取work_condition和user_constraint
    def get_runtime_from_str(self,runtime_str):

        params = runtime_str.split('-')  
        # 创建一个字典来存储参数名和值  
        params_dict = {}  
        # 遍历每个子字符串，并再次按'='分割以获取参数名和值  
        for param in params:  
            key, value = param.split('=')  # 假设每个参数都有值且格式正确  
            params_dict[key] = value  
        
        # 现在你可以通过键来访问参数的值了  
        obj_n = int(params_dict['obj_n'])
        obj_size = float(params_dict['obj_size'])
        obj_speed = float(params_dict['obj_speed'])
        delay = float(params_dict['delay'])
        accuracy = float(params_dict['accuracy'])
        edge_cpu = float(params_dict['edge_cpu'])
        bandwidth = float(params_dict['bandwidth'])

        work_condition={
            "obj_n": obj_n,
            "obj_size": obj_size,
            "obj_speed":obj_speed
        }
        user_constraint={  
            "delay": delay,
            "accuracy": accuracy
        }
        

        runtime={}
        runtime['work_condition']=work_condition
        runtime['user_constraint']=user_constraint
        runtime['edge_cpu']=edge_cpu
        runtime['bandwidth']=bandwidth

        return runtime
    
    # 确定贝叶斯优化目标
    # 从指定runtime_filename文件中，依次读取runtime;
    # 在指定n_trials下，搜索num次，每次对所得最优解打分，求平均值
    def score_kb_by_runtimes(self,kb_name,runtime_filename,n_trials,optimze_goal,search_num):
        # 首先建立字典,选择正确字典路径        
        score_dict=dict()
        tested_num=0
        print(tested_num)
        # 首先确保文件是存在的
        with open('kb_score'+'/'+kb_name+'/'+runtime_filename+"-"+'optimze_goal='+str(optimze_goal)+"-"+'n_trials='+str(n_trials)+"-"+'search_num='+str(search_num)+".json", 'w') as f:  
            json.dump(score_dict, f,indent=4) 
        with open('kb_score/'+runtime_filename+'.txt', 'r', encoding='utf-8') as f:   
            for line in f: 
                tested_num+=1
                print('当前处于第',tested_num,'个情境的知识库评估')
                runtime_str = line.strip()  # strip() 用于去除每行末尾的换行符（如果有的话）  
                # 在这里处理每个字符串，例如打印它  
                print("开始处理情境",runtime_str)
                runtime=self.get_runtime_from_str(runtime_str=runtime_str)
                #print(runtime)
                '''
                runtime['work_condition']=work_condition
                runtime['user_constraint']=user_constraint
                runtime['edge_cpu']=edge_cpu
                runtime['bandwidth']=bandwidth
                '''
                work_condition=runtime['work_condition']
                user_constraint=runtime['user_constraint']
                rsc_constraint={
                    "114.212.81.11": {"cpu": 1.0, "mem": 1.0}, 
                    "192.168.1.7": {"cpu": runtime['edge_cpu'], "mem": 1.0}
                }
                bandwidth_dict={}
                bandwidth_dict['kB/s']=runtime['bandwidth']

                portrait_info={}
                frame = cv2.imread('video_frames/cold_start_4/frame_1080p.jpg')  
                frame=field_codec_utils.encode_image(frame)
                portrait_info['frame']=frame
                portrait_info['data_trans_size']={}
                portrait_info['data_trans_size']['face_detection']=200000
                portrait_info['data_trans_size']['gender_classification']=15000
                
                from RuntimePortrait import RuntimePortrait
                myportrait=RuntimePortrait(pipeline=serv_names)
                rsc_upper_bound={}
                for serv_name in serv_names:
                    serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
                    rsc_upper_bound[serv_name]={}
                    rsc_upper_bound[serv_name]['cpu_limit']=serv_rsc_cons['cpu']['edge']
                    rsc_upper_bound[serv_name]['mem_limit']=serv_rsc_cons['mem']['edge']
                rsc_down_bound={
                    "face_detection": {"cpu_limit": 0.0, "mem_limit": 0.0}, 
                    "gender_classification": {"cpu_limit": 0.0, "mem_limit": 0.0}
                }
        
                kb_user=KnowledgeBaseUser(conf_names=conf_names,
                                            serv_names=serv_names,
                                            service_info_list=service_info_list,
                                            user_constraint=user_constraint,
                                            rsc_constraint=rsc_constraint,
                                            rsc_upper_bound=rsc_upper_bound,
                                            rsc_down_bound=rsc_down_bound,
                                            work_condition=work_condition,
                                            portrait_info=portrait_info,
                                            bandwidth_dict=bandwidth_dict,
                                            kb_name=kb_name
                                            )
                score_in_runtime=0
                for i in range(0,search_num):
                    params_in_cons,params_out_cons,next_plan=kb_user.get_plan_in_cons(n_trials=n_trials,optimze_goal=optimze_goal)
                    # 计算当前查询下的分数
                    if len(params_in_cons)==0:
                        score_in_runtime+=0 #无约束内解，得分为0
                    else: #根据优化目标打分
                        '''
                            ans_dict['pred_delay_list'] = pred_delay_list
                            ans_dict['pred_delay_total'] = pred_delay_total
                            ans_dict['deg_violate'] = deg_violate
                            ans_dict['task_accuracy'] = task_accuracy

                        '''
                        if optimze_goal==MIN_DELAY:
                           # 将时延从小到大排序
                           sorted_params = sorted(params_in_cons, key=lambda d: d['pred_delay_total'])
                           max_score=1.0 - float(sorted_params[0]['pred_delay_total']) / float( user_constraint['delay'])
                           score_in_runtime += max_score
                        
                        elif optimze_goal==MAX_ACCURACY:
                           # 将精度从大到小排序
                           sorted_params = sorted(params_in_cons, key=lambda d: d['task_accuracy'],reverse=True)
                           max_score=1.0 - float( user_constraint['accuracy']) / float(sorted_params[0]['task_accuracy'])
                           score_in_runtime += max_score
                
                # 把search_num次查询结果的总分求平均，得到当前runtime下n_trials次查询的平均表现
                score_in_runtime=score_in_runtime/float(search_num)
                print('该情境下得分为',score_in_runtime)
                
                # 然后记录为字典
                score_dict[runtime_str]=score_in_runtime
            
            f.close()
        
        # 最后把评估该知识库的字典写入文件
        with open('kb_score'+'/'+kb_name+'/'+runtime_filename+"-"+'optimze_goal='+str(optimze_goal)+"-"+'n_trials='+str(n_trials)+"-"+'search_num='+str(search_num)+".json", 'w') as f:  
            json.dump(score_dict, f,indent=4) 
        f.close()


    # anylze_score_result
    # 用途：展示不同知识库或同一知识库在不同情境上的打分效果
    # 方法：从runtime_filename里依次读取各个情境，作为横轴；然后查找对应分数，得到纵轴。由此绘制曲线。
    '''
    filedict={
       'name':'filepath'
    }
    filedict里存储着每一条曲线的名字,用于作为label
    '''
    def anylze_score_result(self,title_name,runtime_filename,file_dict):
        # 
        x_value=[]
        y_value_dict={}
        for key in file_dict.keys():
            y_value_dict[key]=[]
        i=0
        with open('kb_score/'+runtime_filename+'.txt', 'r', encoding='utf-8') as f:   
            for line in f: 
                runtime_str = line.strip()  #获取一个运行时情境
                i=i+1
                x_value.append(i) 
                # 为每一个文件构建y_value
                for key in file_dict.keys(): # file_dict[key]是一个filepath，对应一个评分结果
                    filepath = file_dict[key]
                    with open(filepath, 'r') as tf:  
                        score_result = json.load(tf)  
                    tf.close()
                    y_value_dict[key].append(score_result[runtime_str])
        f.close()

        #现在得到x_value，以及y_value_dict，可以开始绘图
                
        

        plt.figure(figsize=[8, 5])  
        plt.xlabel("情境类型", fontdict={'fontsize': 13, 'family': 'SimSun'})
        plt.ylabel("得分/s", fontdict={'fontsize': 13, 'family': 'SimSun'})
        plt.yticks(fontproperties='Times New Roman')
        plt.xticks(fontproperties='Times New Roman')
        
        for key in file_dict.keys():
            plt.plot(x_value, y_value_dict[key], label=key)
        
        plt.title(title_name, fontdict={'fontsize': 15, 'family': 'SimSun'})
        plt.grid(ls="--", alpha=0.4)  # 绘制虚线网格
        plt.legend(prop={'family': 'SimSun', 'size': 9})
        plt.show()


        return 1
                    

    # 分析区间知识库中各个区间的分布
    # 
    def anylze_sections_distribution(self,kb_name):
        
        # (1)获取section_info
        dag_name=''
        for serv_name in self.serv_names:
            if len(dag_name)==0:
                dag_name+=serv_name
            else:
                dag_name+='-'+serv_name
        section_info={}
        with open(kb_name + '/' + dag_name + '/' + 'section_info.json', 'r') as f:  
            #print('打开知识库section_info:',self.kb_name + '/' + dag_name + '/' + 'section_info.json')
            section_info=json.load(f) 
        f.close()

        # （2）获取section_ids并将其转化为一个个坐标
        section_ids = section_info["section_ids"]
        fps_sec_choice=[]
        reso_sec_choice=[]
        for section_id in section_ids:
            for part in section_id.split('-'):
                # 使用 split('=') 分割键值对  
                conf_name, value = part.split('=')  
                if conf_name == 'fps':
                    fps_sec_choice.append(int(value))
                elif conf_name == 'reso':
                    reso_sec_choice.append(int(value))
        print(fps_sec_choice)
        print(reso_sec_choice)

        
        #（3）绘制网格图，首先确保横纵坐标分别是帧率和分辨率
        conf_sections = section_info["conf_sections"]
        fps_keys = list(conf_sections["fps"].keys())
        reso_keys = list(conf_sections["reso"].keys())
        
        x_labels = fps_keys
        y_labels = reso_keys
        print(x_labels)
        print(y_labels)

        for i in range(1,len(section_ids)):
            # 设置x轴的刻度标签为字符串  
            print('展示',i+1,'个区间')
            x_data = fps_sec_choice[:i+1]
            y_data = reso_sec_choice[:i+1]
            plt.xticks(range(len(x_labels)), x_labels) 
            plt.yticks(range(len(y_labels)), y_labels)  
            plt.gca().set_aspect('equal', adjustable='box')  #确保横纵坐标间隔一定相同
            # 画出网格（虽然对于分类数据可能不太常见，但可以根据需要添加）  
            plt.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')  

            print(x_data)
            print(y_data)

            # 创建一个新的图形  
            #plt.figure() 
            sizes=[960 for t in range(0,len(x_data))]
     
            facecolor=['green' for t in range(0,len(x_data))]
            
            # 画出数据点（注意：对于y轴，我们通常不使用字符串标签）  
            plt.scatter(x_data, y_data,  marker='s', s=sizes, edgecolor='black', facecolor=facecolor)  
            
            
            plt.xlabel("fps区间", fontdict={'fontsize': 13, 'family': 'SimSun'})
            plt.ylabel("reso区间", fontdict={'fontsize': 13, 'family': 'SimSun'})
            
            plt.title('采样区间分布图', fontdict={'fontsize': 15, 'family': 'SimSun'})

            plt.legend(prop={'family': 'SimSun', 'size': 9})
            plt.show()

        
        





# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","gender_classification"] 

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
'''
work_condition_range={
    "obj_n": [1,5,10],
    "obj_size": [50000,100000,150000],
    "obj_speed":[0,260,520,780,800]
}
'''
work_condition_range={
    "obj_n": [1],
    "obj_size": [50000],
    "obj_speed":[0]
}

user_constraint_range={  
    "delay": [0.4,0.7,1.0],
    "accuracy": [0.3,0.6,0.9]
}
rsc_constraint_range=[0.4,0.7,1.0]

bandwidth_range=[20000,40000,60000,80000,100000]




if __name__ == "__main__":

    if_need_generate=0
    if_need_score=1
    
    if_need_draw=0
    if_need_anylze=0

    kb_scorer = KnowledgeBaseScorer(conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    work_condition_range=work_condition_range,
                                    user_constraint_range=user_constraint_range,
                                    rsc_constraint_range=rsc_constraint_range,
                                    bandwidth_range=bandwidth_range)
    
    
    if if_need_generate==1:
        kb_scorer.generate_runtime(runtime_filename='runtime2-125cons')
    
    if if_need_score==1:
        kb_name='kb_data_6sec-1'
        runtime_filename='runtime2-135cons'
        for optimze_goal in [MAX_ACCURACY]:
            for n_trials in [100,200,300]:
                kb_scorer.score_kb_by_runtimes(kb_name=kb_name,runtime_filename=runtime_filename,n_trials=n_trials,optimze_goal=optimze_goal,search_num=1)
    
    if if_need_draw==1:
        titlename="最小化时延"
        runtime_filename="runtime2-135cons"
        #'''
        file_dict={
            "sec=2 num=200":"kb_score/kb_data_3sec-1/runtime2-135cons-optimze_goal=0-n_trials=200-search_num=1.json",
            "sec=3 num=200":"kb_score/kb_data_4sec-1/runtime2-135cons-optimze_goal=0-n_trials=200-search_num=1.json",
            "sec=4 num=200":"kb_score/kb_data_5sec-1/runtime2-135cons-optimze_goal=0-n_trials=200-search_num=1.json",
            "sec=5 num=200":"kb_score/kb_data_6sec-1/runtime2-135cons-optimze_goal=0-n_trials=200-search_num=1.json",
        }
        #'''
        '''
        file_dict={
            "num=100":"kb_score/kb_data_6sec-1/runtime2-135cons-optimze_goal=1-n_trials=100-search_num=1.json",
            "num=200":"kb_score/kb_data_6sec-1/runtime2-135cons-optimze_goal=1-n_trials=200-search_num=1.json",
            "num=300":"kb_score/kb_data_6sec-1/runtime2-135cons-optimze_goal=1-n_trials=300-search_num=1.json",
        }
        '''

        kb_scorer.anylze_score_result(title_name=titlename,runtime_filename=runtime_filename,file_dict=file_dict)

    if if_need_anylze==1:

        kb_scorer.anylze_sections_distribution(kb_name='kb_data_21i90_no_clst-1')