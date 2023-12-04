import requests
import time
import csv
import json
import copy
import os
import datetime

if __name__ == "__main__":
    sess = requests.Session()
    # 实验名称
    expr_name = 'tx2-cloud-raw'
    # 提交请求 指定要分析哪一个边缘端上的视频流
    node_addr = "172.27.133.85:7001" #指定边缘地址
    query_addr = "114.212.81.11:7000" #指定云端地址

    # 提交测试任务：使用99号视频进行测试的时候，用户约束是无意义的，因为目的是进行离线采样。
    query_body = {
        "node_addr": node_addr,
        "video_id": 99,   #99号是专门用于离线启动知识库的测试视频
        "pipeline": ["face_detection", "face_alignment"],
        "user_constraint": {
            "delay": 0.3,  #用户约束暂时设置为0.3
            "accuracy": 0.7
        }
    }

    # 一个任务模型的参数包含model_id、node_ip和node_role三大部分
    # 刻画任何一个模型都需要三个参数.op表示可选项
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

    models_dicts=[]
    for i in range(0,4):
        temp=dict()
        models_dicts.append(temp)

    
    for key in encoder_op:
        models_dicts[0][key]=0
    for key in fps_op:
        models_dicts[1][key]=models_dicts[0]
    for key in reso_op:
        models_dicts[2][key]=models_dicts[1]
    for key in model_op.keys():
        models_dicts[3][key]=models_dicts[2]
    
    print(models_dicts[3])

    eval_face_detect=copy.deepcopy(models_dicts[3])
    eval_face_align=copy.deepcopy(models_dicts[3])
    #使用字典初始化2个评估模型。

    #码率、帧率、分辨率、模型类型（卸载方式），一共4个配置。检查每一个模型在40种配置下的情况。
    #但是，还需要考虑模型在边上互相影响时的情况，也就是内存资源的情况。这个等之后再慢慢研究吧！

    '''建立初始化性能评估器并写入文件
    f1=open("eval_face_detect.json","w")
    json.dump(eval_face_detect,f1,indent=1)
    f1.close()

    f1=open("eval_face_align.json","w")
    json.dump(eval_face_align,f1,indent=1)
    f1.close()
    '''

 
    # 发出提交任务的请求
    r = sess.post(url="http://{}/query/submit_query".format(query_addr),
                  json=query_body)
    # 得到回复
    resp = r.json()
    print(resp)
    query_id = resp["query_id"] #得到云端反馈的任务编号


    #设置用于保存每一条运行结果的文件
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
                        'a_trans_delay']
        wtr = csv.DictWriter(fp, fieldnames=fieldnames)
        wtr.writeheader()

        written_n_loop = dict() #用于存储各个轮的序号，防止重复记录

        # 轮询结果+落盘
        while True:
            r = None
            try:
                time.sleep(1)

                #选择调度策略的范围和制定模型的范围是两回事
                #第一步是根据采样需要指定调度策略
                conf={
                        "reso": "360p",
                        "fps": 1,
                        "encoder": "JEPG"
                    }
                flow_mapping={
                        "face_detection": model_op['model1'],
                        "face_alignment": model_op['model1']
                    }
                r = sess.post(url="http://{}/job/update_plan".format(node_addr),
                        json={"job_uid": query_id, "video_conf": conf, "flow_mapping": flow_mapping})

                print("post one query request")
                r = sess.get(
                    url="http://{}/query/get_result/{}".format(query_addr, query_id))
                
                
                
                #print(f'response:{r.json()}')
                if not r.json():
                    continue
                resp = r.json()

                appended_result = resp['appended_result'] #可以把得到的结果直接提取出需要的内容，列表什么的。
                latest_result = resp['latest_result'] #空的
                 

                print("appended_result的开始与结束")
                print("开始：")
                print(appended_result[0])
                print("结束：")
                length=len(appended_result)
                print(appended_result[length-1])
                # print('latest_result')
                # print(latest_result)

                for res in appended_result:
                    print("开始提取")
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
                    reso=res['ext_plan']['video_conf']['reso']

                    obj_n=res['ext_runtime']['obj_n']
                    obj_size=res['ext_runtime']['obj_size']
                    obj_stable=res['ext_runtime']['obj_stable']
                    all_delay=res['ext_runtime'][ 'delay']
                    
                    d_proc_delay=res['ext_runtime']['plan_result']['process_delay']['face_detection']
                    d_trans_delay=res['ext_runtime']['plan_result']['delay']['face_detection']-d_proc_delay
                    a_proc_delay=res['ext_runtime']['plan_result']['process_delay']['face_alignment']
                    a_trans_delay=res['ext_runtime']['plan_result']['delay']['face_alignment']-a_proc_delay
                    
                    "开始指定row"
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
                        'a_trans_delay':a_trans_delay
                    }
                    if n_loop not in written_n_loop:
                        print("开始写入")
                        wtr.writerow(row)
                        written_n_loop[n_loop] = 1
                print("written one query response, len written_n_loop={}".format(
                    len(written_n_loop.keys())))
                
            except Exception as e:
                if r:
                    print("got serv result: {}".format(r.text))
                # print("caught exception: {}".format(e), exc_info=True)
                print("caught exception: {}".format(e))
                break
'''
开始：
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
'''
