import requests
import time
import csv
import json
import copy
import os
import datetime

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
reso_op=["360p","480p","720p","1080p"]
fps_op=[1,5,10,20,30]
encoder_op=["JPEG"]



#可能需要建一个类来专门负责朝云端发送请求


if __name__ == "__main__":

    knowledge_base_existed=0  #为1表示已经建立好了相关文件
    if knowledge_base_existed==0: #如果还没有建立好，就根据配置建立存储模块化知识库的文件
        # 一个任务模型的参数包含model_id、node_ip和node_role三大部分
        # 刻画任何一个模型都需要三个参数.op表示可选项

        models_dicts=[]
        for i in range(0,4):
            temp=dict()
            models_dicts.append(temp)

        #开始建立
        for key in encoder_op:
            models_dicts[0][key]=0
        for key in fps_op:
            models_dicts[1][key]=models_dicts[0]
        for key in reso_op:
            models_dicts[2][key]=models_dicts[1]
        for key in model_op.keys():
            models_dicts[3][key]=models_dicts[2]
        
        #评估人脸检测和姿态估计的知识库，保存为字典
        eval_face_detect=copy.deepcopy(models_dicts[3])
        eval_face_align=copy.deepcopy(models_dicts[3])

        # 建立初始化性能评估器并写入文件
        f1=open("eval_face_detect.json","w")
        json.dump(eval_face_detect,f1,indent=1)
        f1.close()

        f1=open("eval_face_align.json","w")
        json.dump(eval_face_align,f1,indent=1)
        f1.close()
    else:
        print("人脸检测和姿态估计的模块化知识库已建立,可直接读取或更新")

    sess = requests.Session()
    expr_name = 'tx2-cloud-raw-headup_detect-' #执行人脸检测任务
    node_addr = "172.27.133.85:7001" #指定边缘地址
    node_ip='172.27.133.85'
    query_addr = "114.212.81.11:7000" #指定云端地址
    service_addr="114.212.81.11:7500" #云端服务地址
    # 提交测试任务：使用99号视频进行测试的时候，用户约束是无意义的，因为目的是进行离线采样。此时云端不会调度。
    query_body = {
        "node_addr": node_addr,
        "video_id": 99,   #99号是专门用于离线启动知识库的测试视频
        "pipeline": ["face_detection", "face_alignment"],#制定任务类型
        "user_constraint": {
            "delay": 0.3,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }
    # 发出提交任务的请求
    r = sess.post(url="http://{}/query/submit_query".format(query_addr),
                  json=query_body)
    resp = r.json()
    print(resp)
    query_id = resp["query_id"]
    #创建文件用于录入结果
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
        '_' + os.path.basename(__file__).split('.')[0] + \
        '_' + str(query_body['user_constraint']['delay']) + \
        '_' + str(query_body['user_constraint']['accuracy']) + \
        '_' + expr_name + \
        '.csv'

    with open(filename, 'w', newline='') as fp:
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
        wtr = csv.DictWriter(fp, fieldnames=fieldnames)
        wtr.writeheader()

        written_n_loop = dict() #用于存储各个轮的序号，防止重复记录
        n_loop_begin=0
        n_loop=0  #初始为0，用于在数量足够大的时候终止查询


        conf_version={
            1:{
                "conf":{
                    "resolution": "360p",
                    "fps": 1,
                    "encoder": "JEPG"
                },
                "flow_mapping":{
                    "face_detection": model_op['model1'],
                    "face_alignment": model_op['model1']
                }
            },
            2:{
                "conf":{
                    "resolution": "360p",
                    "fps": 1,
                    "encoder": "JEPG"
                },
                "flow_mapping":{
                    "face_detection": model_op['model2'],
                    "face_alignment": model_op['model2']
                }
            },
            3:{
                "conf":{
                    "resolution": "360p",
                    "fps": 30,
                    "encoder": "JEPG"
                },
                "flow_mapping":{
                    "face_detection": model_op['model1'],
                    "face_alignment": model_op['model1']
                }
            },
            4:{
                "conf":{
                    "resolution": "360p",
                    "fps": 30,
                    "encoder": "JEPG"
                },
                "flow_mapping":{
                    "face_detection": model_op['model2'],
                    "face_alignment": model_op['model2']
                }
            },
            5:{
                "conf":{
                    "resolution": "1080p",
                    "fps": 1,
                    "encoder": "JEPG"
                },
                "flow_mapping":{
                    "face_detection": model_op['model1'],
                    "face_alignment": model_op['model1']
                }
            },
            6:{
                "conf":{
                    "resolution": "1080p",
                    "fps": 1,
                    "encoder": "JEPG"
                },
                "flow_mapping":{
                    "face_detection": model_op['model2'],
                    "face_alignment": model_op['model2']
                }
            },
            7:{
                "conf":{
                    "resolution": "1080p",
                    "fps": 30,
                    "encoder": "JEPG"
                },
                "flow_mapping":{
                    "face_detection": model_op['model1'],
                    "face_alignment": model_op['model1']
                }
            },
            8:{
                "conf":{
                    "resolution": "1080p",
                    "fps": 30,
                    "encoder": "JEPG"
                },
                "flow_mapping":{
                    "face_detection": model_op['model2'],
                    "face_alignment": model_op['model2']
                }
            },
        }

        conf={
                "resolution": "360p",
                "fps": 1,
                "encoder": "JEPG"
            }
        flow_mapping={
                "face_detection": model_op['model1'],
                "face_alignment": model_op['model1']
            }

        # 轮询结果+落盘
        while True:
            r = None
            try:
                time.sleep(1)

                #由于使用了99号视频源，云端不会进行调度，调度计划由本文件指定。

                

                r = sess.post(url="http://{}/job/update_plan".format(node_addr),
                        json={"job_uid": query_id, "video_conf": conf, "flow_mapping": flow_mapping})
         
                
                #获取资源情境
                r = sess.get(url="http://{}/get_resource_info".format(service_addr))
                #print(r)
                resource_info = r.json()
                edge_mem_ratio=resource_info['host'][node_ip]['mem_ratio']
                #print(resource_info)
                #'''

                print("post one query request")  #查询结果
                r = sess.get(
                    url="http://{}/query/get_result/{}".format(query_addr, query_id))  
                
                
                #print(f'response:{r.json()}')
                if not r.json():
                    continue
                resp = r.json()
                appended_result = resp['appended_result'] #可以把得到的结果直接提取出需要的内容，列表什么的。
                latest_result = resp['latest_result'] #空的

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
                    if n_loop not in written_n_loop:  #表示获取了一个新的数据
                        print("获取新检测结果")
                        print(row)
                        print("展示当前环境")
                        print(resource_info)
                        wtr.writerow(row)
                        written_n_loop[n_loop] = 1
                '''
                if n_loop>50:
                    print("数据收集已满足要求，停止查询")
                    break
                '''
                
                
            except Exception as e:
                if r:
                    print("got serv result: {}".format(r.text))
                # print("caught exception: {}".format(e), exc_info=True)
                print("caught exception: {}".format(e))
                break
    

    print("完成查询，开始解读文件",filename)

    


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
