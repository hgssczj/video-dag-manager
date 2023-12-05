from logging_utils import root_logger
import pandas as pd
import os
import itertools
import json
import time
import copy

prev_conf = dict()
prev_flow_mapping = dict()
prev_runtime_info = dict()

init_conf = dict()
init_flow_mapping = dict()

cloud_ip="114.212.81.11"

knowledgebase_path="knowledgebase_rotate"  

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
service_info_dict={
    'face_detection':{
        "name":'face_detection',
        "value":'face_detection_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    'face_alignment':{
        "name":'face_alignment',
        "value":'face_alignment_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    'face_detection_trans':{
        "name":'face_detection_trans',
        "value":'face_detection_trans_delay',
        "conf":["reso","fps","encoder"]
    },
    'face_alignment_trans':{
        "name":'face_alignment_trans',
        "value":'face_alignment_trans_delay',
        "conf":["reso","fps","encoder"]
    }
}

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

''' # 根据流水线pipeline 可以从service_info_dict里提取信息构建service_info_list
service_info_list=[
    {
        "name":'face_detection',
        "value":'face_detection_proc_delay',
        "conf":["reso","fps"]
    },
    {
        "name":'face_alignment',
        "value":'face_alignment_proc_delay',
        "conf":["reso","fps"]
    },
    {
        "name":'face_detection_trans',
        "value":'face_detection_trans_delay',
        "conf":["reso","fps"]
    },
    {
        "name":'face_alignment_trans',
        "value":'face_alignment_trans_delay',
        "conf":["reso","fps"]
    },
]
'''
# 给定flow_map，根据kb获取处理时延
def get_process_delay(resolution=None, flow_map=None):
    sum_delay = 0.0
    for taskname in flow_map:
        pf_filename = 'profile/{}.pf'.format(taskname)
        pf_table = None
        if os.path.exists(pf_filename):
            pf_table = pd.read_table(pf_filename, sep='\t', header=None,
                                    names=['resolution', 'node_role', 'delay'])
        else:
            root_logger.warning("using profile/face_detection.pf for taskname={}".format(taskname))
            pf_table = pd.read_table('profile/face_detection.pf', sep='\t', header=None,
                                    names=['resolution', 'node_role', 'delay'])
        # root_logger.info(pf_table)
        node_role = 'cloud' if flow_map[taskname]['node_role'] == 'cloud' else 'edge'
        pf_table['node_role'] = pf_table['node_role'].astype(str)
        matched_row = pf_table.loc[
            (pf_table['node_role'] == node_role) & \
            (pf_table['resolution'] == resolution)
        ]
        delay = matched_row['delay'].values[0]
        root_logger.info('get profiler delay={} for taskname={} node_role={}'.format(
            delay, taskname, flow_map[taskname]['node_role']
        ))

        sum_delay += delay
    
    root_logger.info('get sum_delay={} by knowledge base'.format(sum_delay))

    return sum_delay
# TODO：给定flow_map，获取传输时延
def get_transfer_delay(resolution=None, flow_map=None, resource_info=None):
    return 0.0
# 获取总预估的时延

'''
"flow_mapping": {
    "face_detection": {
        "model_id": 0,
        "node_ip": "114.212.81.11",
        "node_role": "cloud"
    },
    "face_alignment": {
        "model_id": 0,
        "node_ip": "114.212.81.11",
        "node_role": "cloud"
    }
},
"conf": {
    "reso": "360p"
    "fps": 1,
    "encoder": "JEPG",
}
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
'''
def get_pred_delay(conf=None, flow_mapping=None,service_info_list=None):
    # 根据conf和flow_mapping，从知识库中计算时延
    global knowledgebase_path

    pred_delay_list=[]
    pred_delay_total=0
    for service_info in service_info_list:
        f=open(knowledgebase_path+"/"+service_info['name']+".json")  #加载服务对应的知识库
        evaluator=json.load(f)
        f.close()
        service_conf=list(service_info['conf']) # 形如":["reso","fps","encoder"]
        # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
        conf_for_dict=[]
        for service_conf_name in service_conf:
            conf_for_dict.append(str(conf[service_conf_name]))   #形如：["360p"，"1","JPEG"]
        # 现在还差ip地址：
        ip_for_dict_index=service_info['name'].find("_trans") 
        if ip_for_dict_index>0:
            ip_for_dict_name=service_info['name'][0:ip_for_dict_index] #去除服务名末尾的_trans，形如“face_detection”
        else: #当前不是trans
            ip_for_dict_name=service_info['name']
        ip_for_dict=flow_mapping[ip_for_dict_name]['node_ip']
        conf_for_dict.append(str(ip_for_dict))  # 形如["360p"，"1","JPEG","114.212.81.11"]
        # 现在可以用conf_for_dict从模块化知识库里提取相应的时延
        sub_evaluator=evaluator
        for i in range(0,len(conf_for_dict)-1):
            sub_evaluator=sub_evaluator[conf_for_dict[i]]
        pred_delay=sub_evaluator[conf_for_dict[len(conf_for_dict)-1]]
        pred_delay_list.append(pred_delay)
    
    for pred_delay in pred_delay_list:
        pred_delay_total+=pred_delay
    
    return pred_delay_list,pred_delay_total  # 返回各个部分的时延
        


# TODO：给定fps和resolution，结合运行时情境，获取预测时延
def get_pred_acc(conf_fps=None, cam_fps=None, resolution=None, runtime_info=None):
    if runtime_info and 'obj_stable' in runtime_info:
        if runtime_info['obj_stable']>=0.3 and conf_fps < 20:
            return 0.6
    return 0.9
# ---- 冷启动 ----
def get_flow_map(dag=None, resource_info=None, offload_ptr=None):
    cold_flow_mapping = dict()
    flow = dag["flow"]

    for idx in range(len(flow)):
        taskname = flow[idx]
        if idx <= offload_ptr:
            cold_flow_mapping[taskname] = {
                "model_id": 0,
                "node_role": "host",
                "node_ip": list(resource_info["host"].keys())[0]
            }
        else:
            cold_flow_mapping[taskname] = {
                "model_id": 0,
                "node_role": "cloud",
                "node_ip": list(resource_info["cloud"].keys())[0]
            }
    
    return cold_flow_mapping

def get_cold_start_plan(
    job_uid=None,
    dag=None,
    resource_info=None,
    runtime_info=None,
    user_constraint=None,
):
    assert job_uid, "should provide job_uid"

    global prev_conf, prev_flow_mapping
    global init_conf,init_flow_mapping
    global conf_and_serv_info,service_info_dict

     # 调度器首先要根据dag来得到关于服务的各个信息,
    serv_names=dag['flow']
    service_info_list=[]  #建立service_info_list，需要处理阶段，也需要传输阶段
    for serv_name in serv_names:
        service_info_list.append(service_info_dict[serv_name])
        service_info_list.append(service_info_dict[serv_name+"_trans"])

    conf=dict()
    for service_info in service_info_list:
        service_conf_names = service_info['conf']
        for service_conf_name in service_conf_names:
            if service_conf_name not in conf:
                conf[service_conf_name]=1
    conf_names=list(conf.keys())  #conf_names里保存所有配置参数的名字
    conf_list=[]
    for conf_name in conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
        conf_list.append(conf_and_serv_info[conf_name])
    serv_ip_list=[]
    for serv_name in serv_names:
        serv_ip=serv_name+"_ip"
        serv_ip_list.append(conf_and_serv_info[serv_ip])
    # conf_names=["reso","fps","encoder"]
    # serv_names=["face_detection", "face_alignment"]
    # conf_list=[ ["360p", "480p", "720p", "1080p"],[1, 5, 10, 20, 30],["JPEG"]]
    # serv_ip_list=[["114.212.81.11","172.27.143.164","172.27.151.145"],["114.212.81.11","172.27.143.164","172.27.151.145"]]

    print("初始冷启动,获取任务信息：")
    print("conf_names:",conf_names)
    print("serv_names",serv_names)
    print("conf_list",conf_list)
    print("serv_ip_list",serv_ip_list)

    # 时延约束
    delay_constraint = user_constraint["delay"]

    best_conf={}
    best_flow_mapping={}
    best_pred_delay_list=[]
    best_pred_delay_total=-1

    min_conf={}
    min_flow_mapping={}
    min_pred_delay_list=[]
    min_pred_delay_total=-1
    
    conf_combine=itertools.product(*conf_list)

    for conf_plan in conf_combine:
        serv_ip_combine=itertools.product(*serv_ip_list)
        for serv_ip_plan in serv_ip_combine:# 遍历所有配置和卸载策略组合
            conf={}
            flow_mapping={}
            for i in range(0,len(conf_names)):
                conf[conf_names[i]]=conf_plan[i]
            for i in range(0,len(serv_names)):
                flow_mapping[serv_names[i]]=model_op[serv_ip_plan[i]]
            # 经过遍历，得到一组conf额一组flow_mapping。现在需要根据conf和flow_mapping来从知识库中估计时延
            pred_delay_list,pred_delay_total = get_pred_delay(conf=conf,flow_mapping=flow_mapping,service_info_list=service_info_list)

            if best_pred_delay_total<0:   #初始化最优配置和最小配置
                best_conf=conf
                best_flow_mapping=flow_mapping
                best_pred_delay_list=pred_delay_list
                best_pred_delay_total=pred_delay_total

                min_conf=conf
                min_flow_mapping=flow_mapping
                min_pred_delay_list=pred_delay_list
                min_pred_delay_total=pred_delay_total

            elif pred_delay_total < delay_constraint*0.7 and pred_delay_total>best_pred_delay_total: #选出一个接近约束且比较大的
                best_conf=conf
                best_flow_mapping=flow_mapping
                best_pred_delay_list=pred_delay_list
                best_pred_delay_total=pred_delay_total
            
            elif pred_delay_total < best_pred_delay_total: #选出一个最小的
                min_conf=conf
                min_flow_mapping=flow_mapping
                min_pred_delay_list=pred_delay_list
                min_pred_delay_total=pred_delay_total
    
    # 完成遍历后，应该可以找到一个比较优秀的冷启动结果
    print("最优配置是：")
    print(best_conf)
    print(best_flow_mapping)
    print(best_pred_delay_list)
    print(best_pred_delay_total)
    print("最小配置是：")
    print(min_conf)
    print(min_flow_mapping)
    print(min_pred_delay_list)
    print(min_pred_delay_total)
    print("时延约束是",delay_constraint)

    if min_pred_delay_total > delay_constraint:
        print("约束过于严格，选择最小配置")
        prev_conf[job_uid] = min_conf
        prev_flow_mapping[job_uid] = min_flow_mapping
        init_conf[job_uid] = copy.deepcopy(min_conf)
        init_flow_mapping[job_uid] =copy.deepcopy(min_flow_mapping)
        return prev_conf[job_uid], prev_flow_mapping[job_uid]
    else:
        print("约束不算特别严格，选择最优策略")
        prev_conf[job_uid] = best_conf
        prev_flow_mapping[job_uid] = best_flow_mapping
        init_conf[job_uid] = copy.deepcopy(best_conf)
        init_flow_mapping[job_uid] =copy.deepcopy(best_flow_mapping)
        return prev_conf[job_uid], prev_flow_mapping[job_uid]




# ---- TODO：基于try_expand_resource改进得到，在时延不满足要求时，根据需要决定是卸载到云端还是卸载到边缘----
def try_adjust_resource(next_flow_mapping=None, err_level=None, resource_info=None,runtime_info=None,user_constraint=None,job_uid=None):
    # 当前时延不足，可能是传输时延导致的。所以首先要进行判断，到底应该卸载到云还是卸载到边。
    # 所以，首先应该看传输时延的问题。runtime_info里含有总时延和各阶段处理时延，二者相减可以得到传输时延；不如这样吧，如果发现传输时延占据了总时延的一定比例，就选择从云端卸载到边缘端。
    # 至于卸载到哪一个边缘端，不如就设置为init_flow_mapping[job_uid][taskname]["node_ip"]。从云卸载到边比从边卸载到云更加容易。
    #进行粗略版本的多边协同修改：node_role为host时，暂且将node_ip挪到其他边缘。详细判断等之后再慢慢考虑。
    global  init_flow_mapping
    tune_msg = None
    # 首先要获取传输时延
    print("当前各阶段时延是：")
    print(runtime_info['plan_result'])
    process_delay=0
    total_delay=float(runtime_info['delay']) #总时延
    for k,v in (runtime_info['plan_result']['process_delay'].items()):
        process_delay+=v  #累加存储的各阶段处理时延
    transmit_delay=float(max(0.0,total_delay-process_delay))
    print("求出传输时延是",transmit_delay)

    #如果传输时延占据总时延的六成以上，就考虑从云端撤回，如何
    if transmit_delay>=0.6*total_delay:
        print("传输时延过大，尝试从云卸载到边")
        print("查看next_flow_mapping")
        print(next_flow_mapping)
        for taskname, task_mapping in list(next_flow_mapping.items()):#流水线正向寻找任务，把遇到的第一个云端给卸载回边缘
            if task_mapping["node_role"] == "cloud": #考虑是卸载到云端还是卸载到边缘端。如果需要移动到边就在第一个边呆着。
                print(" -------- send to edge --------")
                next_flow_mapping[taskname]["node_role"] = "host"
                next_flow_mapping[taskname]["node_ip"] = list(resource_info["host"].keys())[0]  
                tune_msg = "task-{} send to host".format(taskname)
                break
    
    else:# 否则，还是考虑将任务卸载到云端。以下是考虑卸载到云端的操作
        print("传输时延较小，尝试从边卸载到云")
        print("查看next_flow_mapping")
        print(next_flow_mapping)
        for taskname, task_mapping in reversed(list(next_flow_mapping.items())):#流水线反向寻找任务
            if task_mapping["node_role"] == "host": #考虑是卸载到云端还是卸载到边缘端
                edge_num=len(list(resource_info["host"].keys()))
                pan=0
                for i in range(0,edge_num-1): #从所有边缘端中，取当前边缘端的下一个边缘端，如果到末尾了就从头开始。如果最后选出来的ip和初始冷启动的ip不一样，就设置Pan=1
                #表示找到了合适卸载的边缘端。无论是否成果，最后都会break，因为没有必要继续找其他可能有效的边缘端了。
                    if task_mapping["node_ip"]==list(resource_info["host"].keys())[i]:
                        print(" -------- try to send to another edge --------")
                        next_flow_mapping[taskname]["node_role"] = "host"
                        next_flow_mapping[taskname]["node_ip"] = list(resource_info["host"].keys())[(i+1)%edge_num]
                        print("当前找到的边缘端ip和初始冷启动ip分别是(相等就换到云):")
                        print(next_flow_mapping[taskname]["node_ip"],init_flow_mapping[job_uid][taskname]["node_ip"])
                        tune_msg = "task-{} send to another edge".format(taskname)
                        # 说明一下：由于初始化问题，如果一个任务冷启动时被分配到边，那一定是在第一个边上。所以，如果不断移动边发现回到第一个边，就考虑卸载到云
                        # 反之，如果新找到的边缘端ip不是第一个边缘端的ip，就设置pan=1，不需要移动到云
                        if next_flow_mapping[taskname]["node_ip"]!=list(resource_info["host"].keys())[0]:
                            pan=1 
                        break  
                if pan== 0: #如果一番寻找后回到了最开始那个边缘，就意味着必须卸载到云端了，因为所有的边缘都已经试过了。
                    print(" -------- send to cloud --------")
                    next_flow_mapping[taskname]["node_role"] = "cloud"
                    next_flow_mapping[taskname]["node_ip"] = list(
                        resource_info["cloud"].keys())[0]
                    tune_msg = "task-{} send to cloud".format(taskname)
                break
    
    return tune_msg, next_flow_mapping
# ---- TODO：根据应用情境，尝试减少计算量 ----
def try_reduce_calculation(
    next_video_conf=None,
    err_level=None,
    runtime_info=None,
    init_prior=1,
    best_effort=False
):
    global conf_and_serv_info

    resolution_index = conf_and_serv_info['reso'].index(next_video_conf["reso"])
    fps_index = conf_and_serv_info['fps'].index(next_video_conf["fps"])

    tune_msg = None

    # TODO：根据运行时情境初始化优先级，实现最佳匹配
    total_prior = 2
    curr_prior = init_prior

    # 无法最佳匹配时，根据收益大小优先级调度
    while True:
        if curr_prior == 1:
            if fps_index > 0:
                print(" -------- fps lower -------- (init_prior={})".format(init_prior))
                next_video_conf["fps"] = conf_and_serv_info['fps'][fps_index - 1]
                tune_msg = "fps {} -> {}".format(conf_and_serv_info['fps'][fps_index],
                                                conf_and_serv_info['fps'][fps_index - 1])

        if curr_prior == 0:
            if resolution_index > 0:
                print(" -------- resolution lower -------- (init_prior={})".format(init_prior))
                next_video_conf["reso"] = conf_and_serv_info['reso'][resolution_index - 1]
                tune_msg = "resolution {} -> {}".format(conf_and_serv_info['reso'][resolution_index],
                                                        conf_and_serv_info['reso'][resolution_index - 1])
        
        # 按优先级依次选择可调的配置
        if best_effort and not tune_msg:
            curr_prior = (curr_prior + 1) % total_prior
            if curr_prior == init_prior:
                break
        if best_effort and tune_msg:
            break
        if not best_effort:
            break

    
    return tune_msg, next_video_conf
# ---- 负反馈 ----
def adjust_parameters(output=0, job_uid=None,
                      dag=None,
                      user_constraint=None,
                      resource_info=None,
                      runtime_info=None):
    assert job_uid, "should provide job_uid"

    global prev_conf, prev_flow_mapping, prev_runtime_info
    global conf_and_serv_info

    next_video_conf = prev_conf[job_uid]
    next_flow_mapping = prev_flow_mapping[job_uid]

    # 仅支持pipeline
    flow = dag["flow"]
    assert isinstance(flow, list), "flow not list"

    resolution_index = conf_and_serv_info['reso'].index(next_video_conf["reso"])
    fps_index = conf_and_serv_info['fps'].index(next_video_conf["fps"])

    err_level = round(output)
    if err_level < -3:
        err_level = -3
    elif err_level > 3:
        err_level = 3

    tune_msg = None

    # TODO：参照对应的边端sniffer解析运行时情境
    print('---- runtime_info in the past time slot ----')
    print('runtime_info = {}'.format(runtime_info))
    # obj_n = runtime_info['obj_n']

    if err_level > 0:
        # level > 0，时延满足要求
        # TODO：结合运行时情境（应用），可以进一步优化其他目标（精度、云端开销等）：
        #              优化目标优先级：时延 > 精度 > 云端开销
        #              若优化目标为最大化精度，在达不到要求时，可以提高fps和resolution；
        #              若优化目标为最小化云端开销，可以拉回到边端计算；
        tune_level = err_level
        pred_acc = get_pred_acc(conf_fps=next_video_conf['fps'], cam_fps=30.0,
                                resolution=next_video_conf["reso"],
                                runtime_info=runtime_info)
        
        # 若此时预测精度达不到要求，可以提高fps和resolution
        if pred_acc < user_constraint["accuracy"]:
            # 根据不同程度的 delay-acc trade-off，在不同的delay级别调整不同的参数
            while not tune_msg and tune_level > 0:
                if tune_level == 2:
                    if fps_index + 1 < len(conf_and_serv_info['fps']):
                        print(" -------- fps higher -------- (err_level={}, tune_msg={})".format(err_level, tune_msg))
                        next_video_conf["fps"] = conf_and_serv_info['fps'][fps_index + 1]
                        tune_msg = "fps {} -> {}".format(conf_and_serv_info['fps'][fps_index],
                                                        conf_and_serv_info['fps'][fps_index + 1])

                elif tune_level == 1:
                    if resolution_index + 1 < len(conf_and_serv_info['reso']):
                        print(" -------- resolution higher -------- (err_level={}, tune_msg={})".format(err_level, tune_msg))
                        next_video_conf["reso"] = conf_and_serv_info['reso'][resolution_index + 1]
                        tune_msg = "resolution {} -> {}".format(conf_and_serv_info['reso'][resolution_index],
                                                                conf_and_serv_info['reso'][resolution_index + 1])
                
                # 按优先级依次选择可调的配置
                if not tune_msg:
                    tune_level -= 1
        else:
            if 'obj_stable' in runtime_info and (runtime_info['obj_stable']<0.3):
                # 场景稳定，优先降低帧率
                init_prior = 1
                best_effort = False
                tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                                err_level=err_level, 
                                                                runtime_info=runtime_info,
                                                                init_prior=init_prior, best_effort=best_effort)
  
    elif err_level < 0:

        if 'obj_stable' in runtime_info and (runtime_info['obj_stable']<0.3):
            # 场景稳定，优先降低帧率
            init_prior = 1
            best_effort = False
            tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                               err_level=err_level, 
                                                               runtime_info=runtime_info,
                                                               init_prior=init_prior, best_effort=best_effort)

        elif 'obj_size' in runtime_info and runtime_info['obj_size'] > 500:
            # 场景不稳定，但物体够大，优先降低分辨率
            init_prior = 0
            best_effort = False
            tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                               err_level=err_level, 
                                                               runtime_info=runtime_info,
                                                               init_prior=init_prior, best_effort=best_effort)



        if not tune_msg:
              tune_msg, next_flow_mapping = try_adjust_resource(next_flow_mapping=next_flow_mapping, err_level=err_level, resource_info=resource_info,runtime_info=runtime_info,user_constraint=user_constraint,job_uid=job_uid)

        if not tune_msg:
            init_prior = 1
            best_effort = True
            tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                               err_level=err_level, 
                                                               runtime_info=runtime_info,
                                                               init_prior=init_prior, best_effort=best_effort)

    prev_conf[job_uid] = next_video_conf
    prev_flow_mapping[job_uid] = next_flow_mapping
    prev_runtime_info[job_uid] = runtime_info

    print(prev_flow_mapping[job_uid])
    print(prev_conf[job_uid])
    print(prev_runtime_info[job_uid])
    root_logger.info("tune_msg: {}".format(tune_msg))
    
    return prev_conf[job_uid], prev_flow_mapping[job_uid]


# -----------------
# ---- 调度入口 ----
def scheduler(
    job_uid=None,
    dag=None,
    resource_info=None,
    runtime_info=None,
    user_constraint=None,
):
    # 初始情况下选择使用冷启动策略
    if not bool(runtime_info) or not bool(user_constraint):
        root_logger.info("to get COLD start executation plan")
        return get_cold_start_plan(
            job_uid=job_uid,
            dag=dag,
            resource_info=resource_info,
            runtime_info=runtime_info,
            user_constraint=user_constraint
        )

    # ---- 若有负反馈结果，则进行负反馈调节 ----
    global prev_conf, prev_flow_mapping

    return prev_conf[job_uid], prev_flow_mapping[job_uid]  #沿用之前的配置
