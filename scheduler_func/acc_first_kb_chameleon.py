import field_codec_utils
import cv2
import common
from logging_utils import root_logger
import requests
import time
import json
import copy
import math

sess = requests.Session()


def generate_rsc_combination(rsc_list, depth, length):
    res = []
    if depth == length:
        for temp in rsc_list:
            res.append([temp])
    else:
        temp_res = generate_rsc_combination(rsc_list, depth+1, length)
        for temp_rsc in rsc_list:
            for temp in temp_res:
                res.append([temp_rsc] + temp)
    return res


def get_rsc_combination(rsc_list, serv_num, rsc_constraint):
    # 此函数用于获取在边端部署的各个服务的资源分配量的组合方式（满足用户资源约束的部分）
    '''
    rsc_list: 资源的取值列表
    serv_num: 在边端部署的服务数量
    rsc_constraint: 用户的资源约束
    '''
    if serv_num == 0:
        return []
    else:
        res = generate_rsc_combination(rsc_list, 1, serv_num)  # 生成所有的资源组合
        
        res_in_cons = []
        for temp_res in res:
            if sum(temp_res) <= rsc_constraint:  # 仅保留符合用户资源约束的部分
                res_in_cons.append(temp_res)
        
        return res_in_cons
    
    
def get_limit_resource_url(device_type):
    if device_type == 'cloud':
        url = "http://{}:{}/limit_task_resource".format('114.212.81.11', 4520)
    else:
        url = "http://{}:{}/limit_task_resource".format('114.212.81.156', 25717)
    return url


def limit_service_resource(limit_url, task_limit):
    r = None
    try:
        r = sess.post(url=limit_url, json=task_limit)
        return r.json()

    except Exception as e:
        if r:
            root_logger.error("In acc_first_kb_chameleon limit_resource: {}".format(r.text))
        root_logger.error("In acc_first_kb_chameleon limit_resource: caught exception: {}".format(e), exc_info=True)
        return None


def get_chosen_service_url(device_type, taskname):
    if device_type == 'cloud':
        url = "http://{}:{}/execute_task/{}".format('114.212.81.11', 4520, taskname)
    else:
        # 向边端发起post请求的url。注意，由于边缘设备放置在路由器下，云端向边端发起post请求时需要在路由器内部进行端口映射，目前路由器中配置的端口映射方式见chameleon.md
        url = "http://{}:{}/execute_task/{}".format('114.212.81.156', 25717, taskname)
    return url


def invoke_service(serv_url, input_ctx):
    r = None
    try:
        r = sess.post(url=serv_url, json=input_ctx)
        return r.json()

    except Exception as e:
        if r:
            root_logger.error("In acc_first_kb_chameleon got serv result: {}".format(r.text))
        root_logger.error("In acc_first_kb_chameleon caught exception: {}".format(e), exc_info=True)
        return None


def cal_iou(predict_bbox, predict_reso, gt_bbox):
    predict_bbox[0] = predict_bbox[0] * (1920 / common.resolution_wh[predict_reso]['w'])
    predict_bbox[2] = predict_bbox[2] * (1920 / common.resolution_wh[predict_reso]['w'])
    predict_bbox[1] = predict_bbox[1] * (1080 / common.resolution_wh[predict_reso]['h'])
    predict_bbox[3] = predict_bbox[3] * (1080 / common.resolution_wh[predict_reso]['h'])
    
    xmin1, ymin1, xmax1, ymax1 = predict_bbox
    xmin2, ymin2, xmax2, ymax2 = gt_bbox
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou


def cal_accuracy(predict_data, predict_reso, label_data):
    '''
    此函数用于计算预测数据predict_data和真实数据label_data相比的精度, 其中predict_data是在predict_reso分辨率下检测的结果
    '''
    tp = fp = fn = 0
    
    predict_obj_num = len(predict_data)
    gt_obj_num = len(label_data)
    
    if predict_obj_num == 0 and gt_obj_num == 0:
        return tp, fp, fn
    elif predict_obj_num == 0 and gt_obj_num != 0:
        fn = gt_obj_num
        return tp, fp, fn
    elif predict_obj_num != 0 and gt_obj_num == 0:
        fp = predict_obj_num
        return tp, fp, fn
        
    for predict_bbox in predict_data:
        for gt_bbox in label_data:
            temp_iou = cal_iou(predict_bbox, predict_reso, gt_bbox)
            if temp_iou >= 0.7:
                tp += 1
    
    fp = predict_obj_num - tp
    fn = gt_obj_num - tp
    
    return tp, fp, fn
    

def get_lat_and_acc(pipeline, frame_act_result_list, frame_test_result_list, reso):
    assert len(frame_act_result_list) == len(frame_test_result_list)
    
    latency = 0
    acc = 0
    
    if pipeline[0] == 'face_detection':
        #### 1. 求出时延
        for frame_test_result in frame_test_result_list:
            latency += frame_test_result['delay_computed']
        latency = latency / len(frame_test_result_list)
        
        #### 2. 求出精度
        for i in range(len(frame_test_result_list)):
            assert 'bbox' in frame_act_result_list[i]['face_detection'] and 'bbox' in frame_test_result_list[i]['face_detection']
            
            temp_tp, temp_fp, temp_fn = cal_accuracy(frame_test_result_list[i]['face_detection']['bbox'], reso, frame_act_result_list[i]['face_detection']['bbox'])
            if temp_tp == 0 and temp_fp == 0 and temp_fn == 0:
                acc += 1
            elif temp_tp + temp_fp == 0 or temp_tp + temp_fn == 0:
                acc += 0
            else:
                temp_precision = temp_tp / (temp_tp + temp_fp)
                temp_recall = temp_tp / (temp_tp + temp_fn)
                
                if temp_precision + temp_recall < 1e-9:
                    acc += 0
                else:
                    temp_f1_score = 2 * temp_precision * temp_recall / (temp_precision + temp_recall)  # 以f1-score作为精度衡量指标
                    acc += temp_f1_score
        acc = acc / len(frame_test_result_list)
    
    return latency, acc
        
            
def get_top_k_plans(pipeline, frames, origin_fps, user_constraint, bandwidth_dict, k=5):
    '''
    frames: 视频块, Chameleon算法在此视频块上运行各种配置
    origin_fps: 视频流的初始帧率
    user_constraint: 用户约束
    bandwidth_dict: 云边之间的带宽信息
    '''
    # 此函数用于实现Chameleon算法中获取top-k配置的功能
    if pipeline[0] == 'face_detection':
        reso_range = ['360p', '1080p']  # ['360p', '480p', '720p', '1080p']
        fps_range = [1, 20]  # [1, 5, 10, 20, 30]
        encoder_range = ['JPEG']
        cpu_range = [0.40]
        '''
        cpu_range = [0.05,0.10,0.15,0.20,
                     0.25,0.30,0.35,0.40,
                     0.45,0.50,0.55,0.60]
        '''
        mem_range = [1.0]
        '''
        缩小配置空间之后, 找出top-K大约耗时104秒, 从top-K中找最优大约耗时64s
        '''
        
        cloud_ip = "114.212.81.11"  # 云端服务器的ip
        edge_ip = "192.168.1.9"  # 运行调度系统的边缘设备的ip
        
        plans_in_acc_cons = []  # 满足用户精度约束的调度策略
        plans_out_acc_cons = []  # 不满足用户精度约束的调度策略
        
        ###### 1. 以黄金配置运行视频块，获取工况的真实值，用于后续其他配置计算精度使用
        frame_list = []
        for frame in frames:
            frame_arr = field_codec_utils.decode_image(frame)
            frame_resized = cv2.resize(frame_arr, (
                                common.resolution_wh['1080p']['w'],
                                common.resolution_wh['1080p']['h']
                            ))
            frame_resized_str = field_codec_utils.encode_image(frame_resized)
            frame_list.append(frame_resized_str)
        
        frame_act_result_list = []  # 以黄金配置运行视频块得到的各帧的执行结果
        for frame in frame_list:
            output_ctx = {
                'image': frame
            }
            frame_result_dict = dict()  # 当前帧的执行结果
            for taskname in pipeline:
                ## 进行服务调用
                input_ctx = output_ctx
                service_url = get_chosen_service_url('cloud', taskname)
                
                output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                # 重试
                while not output_ctx:
                    time.sleep(1)
                    output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                
                frame_result_dict[taskname] = output_ctx

            frame_act_result_list.append(frame_result_dict)
        
        
        ###### 2. 遍历所有可能的各种配置，获取其执行时延、精度等指标
        for cloud_edge_cut_index in range(len(pipeline) + 1):  # 遍历所有可能的云边协同切分点：第一个被卸载到云端执行的服务的索引
            for reso in reso_range:  # 遍历所有可能的分辨率
                for fps in fps_range:  # 遍历所有可能的帧率
                    for encoder in encoder_range:  # 遍历所有可能的编码方式
                        ### 1. 根据帧率和分辨率准备视频帧
                        frame_list = []
                        for frame in frames:
                            frame_arr = field_codec_utils.decode_image(frame)
                            frame_resized = cv2.resize(frame_arr, (
                                                common.resolution_wh[reso]['w'],
                                                common.resolution_wh[reso]['h']
                                            ))
                            frame_resized_str = field_codec_utils.encode_image(frame_resized)
                            frame_list.append(frame_resized_str)
                        
                        frame_index_list = []
                        for index in range(0, len(frames), math.ceil(origin_fps/fps)):
                            frame_index_list.append(index)
                        root_logger.info("In acc_first_kb_chameleon, 调度信息:{}, {}".format(len(frames), math.ceil(origin_fps/fps)))
                        assert len(frame_index_list) > 0
                        
                        
                        ### 2. 根据云边协同切分点和资源设置调用服务
                        rsc_combination_in_cons = get_rsc_combination(cpu_range, cloud_edge_cut_index, user_constraint['rsc_constraint']["192.168.1.9"]["cpu"])
                        
                        if cloud_edge_cut_index == 0:  # 全在云端执行
                            frame_result_dict_computed = dict()  # 存储帧的执行结果，key为帧索引，value为执行结果。注意，由于存在跳帧，此变量保存的是实际执行的帧的结果
                            for frame_index in frame_index_list:
                                temp_frame = frame_list[frame_index]
                                output_ctx = {
                                    'image': temp_frame
                                }
                                limit_rsc_url = get_limit_resource_url('cloud')
                                
                                frame_result_dict = dict()
                                frame_result_dict['delay'] = dict()
                                frame_result_dict['process_delay'] = dict()
                                data_to_cloud = 0
                                
                                for taskname in pipeline:
                                    ## 1. 进行资源限制
                                    task_limit = {}
                                    task_limit['task_name'] = taskname
                                    task_limit['cpu_util_limit'] = 1.0
                                    task_limit['mem_util_limit'] = 1.0

                                    resp = limit_service_resource(limit_url=limit_rsc_url, task_limit=task_limit)
                                    # 重试,完成资源限制
                                    while not resp:
                                        time.sleep(1)
                                        resp = limit_service_resource(limit_url=limit_rsc_url, task_limit=task_limit)
                                    
                                    ## 2. 进行服务调用
                                    input_ctx = output_ctx
                                    service_url = get_chosen_service_url('cloud', taskname)
                                    data_to_cloud += len(json.dumps(input_ctx).encode('utf-8'))
                                    
                                    st_time = time.time()
                                    output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                                    # 重试
                                    while not output_ctx:
                                        time.sleep(1)
                                        output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                                    ed_time = time.time()
                                    
                                    data_to_cloud += len(json.dumps(output_ctx).encode('utf-8'))
                                    frame_result_dict[taskname] = output_ctx
                                    frame_result_dict['delay'][taskname] = ed_time - st_time
                                    frame_result_dict['process_delay'][taskname] = output_ctx['proc_resource_info']['compute_latency']
                                    
                                total_frame_delay = 0
                                total_frame_process_delay = 0
                                for taskname in frame_result_dict['delay']:
                                    frame_result_dict['delay'][taskname] = \
                                        frame_result_dict['delay'][taskname] * fps / origin_fps 
                                    total_frame_delay += frame_result_dict['delay'][taskname]
                                
                                for taskname in frame_result_dict['process_delay']:
                                    frame_result_dict['process_delay'][taskname] = \
                                        frame_result_dict['process_delay'][taskname] * fps / origin_fps 
                                    total_frame_process_delay += frame_result_dict['process_delay'][taskname]
                                
                                frame_result_dict['delay_total'] = total_frame_delay
                                frame_result_dict['process_delay_total'] = total_frame_process_delay
                                trans_delay = data_to_cloud / (bandwidth_dict['kB/s'] * 1000)
                                trans_delay = trans_delay * fps / origin_fps
                                frame_result_dict['trans_delay_total'] = trans_delay
                                frame_result_dict['delay_computed'] = frame_result_dict['process_delay_total'] + frame_result_dict['trans_delay_total']
                                
                                frame_result_dict_computed[frame_index] = frame_result_dict
                            
                            frame_result_list_total = []  # 存储所有帧的执行结果，由frame_result_dict_computed得到
                            for i in range(len(frame_list)):
                                if i in frame_result_dict_computed:
                                    frame_result_list_total.append(frame_result_dict_computed[i])
                                else:  # 否则以最近执行的帧的结果作为跳过的帧的执行结果
                                    frame_result_list_total.append(frame_result_list_total[-1])
                            
                            assert len(frame_result_list_total) == len(frame_list)
                            
                            # 将当前配置的执行结果与黄金配置的执行结果进行对比，获取当前配置的精度、时延等指标
                            temp_latency, temp_acc = get_lat_and_acc(pipeline, frame_act_result_list, frame_result_list_total, reso)
                            temp_plan_info = {
                                'conf': {
                                    'reso': reso,
                                    'fps': fps,
                                    'encoder': encoder
                                },
                                'flow_mapping': {
                                },
                                'resource_limit': {
                                },
                                'latency': temp_latency,
                                'accuracy': temp_acc
                            }
                            for taskname in pipeline:
                                temp_plan_info['flow_mapping'][taskname] = common.model_op[cloud_ip]
                                temp_plan_info['resource_limit'][taskname] = {"cpu_util_limit": 1.0, "mem_util_limit": 1.0}
                            if temp_acc >= user_constraint['accuracy']:
                                plans_in_acc_cons.append(temp_plan_info)
                            else:
                                plans_out_acc_cons.append(temp_plan_info)

                        else:  # 有服务在边端执行
                            for rsc_combination in rsc_combination_in_cons:  # 遍历所有的资源限制组合
                                assert len(rsc_combination) == cloud_edge_cut_index
                                
                                frame_result_dict_computed = dict()
                                for frame_index in frame_index_list:
                                    temp_frame = frame_list[frame_index]
                                    output_ctx = {
                                        'image': temp_frame
                                    }
                                    
                                    frame_result_dict = dict()
                                    frame_result_dict['delay'] = dict()
                                    frame_result_dict['process_delay'] = dict()
                                    data_to_cloud = 0
                                
                                    for index in range(cloud_edge_cut_index):
                                        ## 1. 进行资源限制
                                        limit_rsc_url = get_limit_resource_url('edge')
                                        
                                        taskname = pipeline[index]
                                        task_limit = {}
                                        task_limit['task_name'] = taskname
                                        task_limit['cpu_util_limit'] = rsc_combination[index]
                                        task_limit['mem_util_limit'] = 1.0

                                        resp = limit_service_resource(limit_url=limit_rsc_url, task_limit=task_limit)
                                        # 重试,完成资源限制
                                        while not resp:
                                            time.sleep(1)
                                            resp = limit_service_resource(limit_url=limit_rsc_url, task_limit=task_limit)
                                        
                                        ## 2. 进行服务调用
                                        input_ctx = output_ctx
                                        service_url = get_chosen_service_url('edge', taskname)
                                        
                                        st_time = time.time()
                                        output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                                        # 重试
                                        while not output_ctx:
                                            time.sleep(1)
                                            output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                                        ed_time = time.time()
                                        
                                        frame_result_dict[taskname] = output_ctx
                                        frame_result_dict['delay'][taskname] = ed_time - st_time
                                        frame_result_dict['process_delay'][taskname] = output_ctx['proc_resource_info']['compute_latency']
                                        
                                    for index in range(cloud_edge_cut_index, len(pipeline)):
                                        ## 1. 进行资源限制
                                        limit_rsc_url = get_limit_resource_url('cloud')
                                        
                                        taskname = pipeline[index]
                                        task_limit = {}
                                        task_limit['task_name'] = taskname
                                        task_limit['cpu_util_limit'] = 1.0
                                        task_limit['mem_util_limit'] = 1.0

                                        resp = limit_service_resource(limit_url=limit_rsc_url, task_limit=task_limit)
                                        # 重试,完成资源限制
                                        while not resp:
                                            time.sleep(1)
                                            resp = limit_service_resource(limit_url=limit_rsc_url, task_limit=task_limit)
                                        
                                        ## 2. 进行服务调用
                                        input_ctx = output_ctx
                                        service_url = get_chosen_service_url('cloud', taskname)
                                        data_to_cloud += len(json.dumps(input_ctx).encode('utf-8'))
                                        
                                        st_time = time.time()
                                        output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                                        # 重试
                                        while not output_ctx:
                                            time.sleep(1)
                                            output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                                        ed_time = time.time()
                                        data_to_cloud += len(json.dumps(output_ctx).encode('utf-8'))
                                        
                                        frame_result_dict[taskname] = output_ctx
                                        frame_result_dict['delay'][taskname] = ed_time - st_time
                                        frame_result_dict['process_delay'][taskname] = output_ctx['proc_resource_info']['compute_latency']
                                    
                                    total_frame_delay = 0
                                    total_frame_process_delay = 0
                                    for taskname in frame_result_dict['delay']:
                                        frame_result_dict['delay'][taskname] = \
                                            frame_result_dict['delay'][taskname] * fps / origin_fps 
                                        total_frame_delay += frame_result_dict['delay'][taskname]
                                    
                                    for taskname in frame_result_dict['process_delay']:
                                        frame_result_dict['process_delay'][taskname] = \
                                            frame_result_dict['process_delay'][taskname] * fps / origin_fps 
                                        total_frame_process_delay += frame_result_dict['process_delay'][taskname]
                                    
                                    frame_result_dict['delay_total'] = total_frame_delay
                                    frame_result_dict['process_delay_total'] = total_frame_process_delay
                                    trans_delay = data_to_cloud / (bandwidth_dict['kB/s'] * 1000)
                                    trans_delay = trans_delay * fps / origin_fps
                                    frame_result_dict['trans_delay_total'] = trans_delay
                                    frame_result_dict['delay_computed'] = frame_result_dict['process_delay_total'] + frame_result_dict['trans_delay_total']
                                    
                                    frame_result_dict_computed[frame_index] = frame_result_dict
                                
                                frame_result_list_total = []
                                for i in range(len(frame_list)):
                                    if i in frame_result_dict_computed:
                                        frame_result_list_total.append(frame_result_dict_computed[i])
                                    else:
                                        frame_result_list_total.append(frame_result_list_total[-1])
                                
                                assert len(frame_result_list_total) == len(frame_list)
                                
                                temp_latency, temp_acc = get_lat_and_acc(pipeline, frame_act_result_list, frame_result_list_total, reso)
                                temp_plan_info = {
                                    'conf': {
                                        'reso': reso,
                                        'fps': fps,
                                        'encoder': encoder
                                    },
                                    'flow_mapping': {
                                    },
                                    'resource_limit': {
                                    },
                                    'latency': temp_latency,
                                    'accuracy': temp_acc
                                }
                                
                                for index in range(cloud_edge_cut_index):
                                    taskname = pipeline[index]
                                    temp_plan_info['flow_mapping'][taskname] = common.model_op[edge_ip]
                                    temp_plan_info['resource_limit'][taskname] = {"cpu_util_limit": rsc_combination[index], "mem_util_limit": 1.0}
                                
                                for index in range(cloud_edge_cut_index, len(pipeline)):
                                    taskname = pipeline[index]
                                    temp_plan_info['flow_mapping'][taskname] = common.model_op[cloud_ip]
                                    temp_plan_info['resource_limit'][taskname] = {"cpu_util_limit": 1.0, "mem_util_limit": 1.0}
                                  
                                if temp_acc >= user_constraint['accuracy']:
                                    plans_in_acc_cons.append(temp_plan_info)
                                else:
                                    plans_out_acc_cons.append(temp_plan_info)
    
        plans_in_acc_cons.sort(key=lambda x:x['latency'])
        plans_out_acc_cons.sort(key=lambda x:(-x['accuracy'], x['latency']))
        
        top_k_plans = []
        count = 0
        for plan in plans_in_acc_cons:
            if count < k:
                top_k_plans.append(plan)
                count += 1
        
        for plan in plans_out_acc_cons:
            if count < k:
                top_k_plans.append(plan)
                count += 1
        
        return top_k_plans
        

def get_best_plan(pipeline, frames, origin_fps, user_constraint, bandwidth_dict, top_k_plans):
    # 此函数用于实现Chameleon算法中在top-k配置里选择最优配置的功能
    if pipeline[0] == 'face_detection':
        ###### 1. 以黄金配置运行视频块，获取工况的真实值，用于后续其他配置计算精度使用
        frame_list = []
        for frame in frames:
            frame_arr = field_codec_utils.decode_image(frame)
            frame_resized = cv2.resize(frame_arr, (
                                common.resolution_wh['1080p']['w'],
                                common.resolution_wh['1080p']['h']
                            ))
            frame_resized_str = field_codec_utils.encode_image(frame_resized)
            frame_list.append(frame_resized_str)
        
        frame_act_result_list = []
        for frame in frame_list:
            output_ctx = {
                'image': frame
            }
            frame_result_dict = dict()
            for taskname in pipeline:
                ## 进行服务调用
                input_ctx = output_ctx
                service_url = get_chosen_service_url('cloud', taskname)
                
                output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                # 重试
                while not output_ctx:
                    time.sleep(1)
                    output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                
                frame_result_dict[taskname] = output_ctx

            frame_act_result_list.append(frame_result_dict)
        
        ###### 2. 重新运行top_k配置并且重新计算其时延、精度等指标
        plans_in_acc_cons = []
        plans_out_acc_cons = []
        
        for plan in top_k_plans:
            ### 1. 根据帧率和分辨率准备视频帧
            reso = plan['conf']['reso']
            fps = plan['conf']['fps']
            
            frame_list = []
            for frame in frames:
                frame_arr = field_codec_utils.decode_image(frame)
                frame_resized = cv2.resize(frame_arr, (
                                    common.resolution_wh[reso]['w'],
                                    common.resolution_wh[reso]['h']
                                ))
                frame_resized_str = field_codec_utils.encode_image(frame_resized)
                frame_list.append(frame_resized_str)
            
            frame_index_list = []
            for index in range(0, len(frames), math.ceil(origin_fps/fps)):
                frame_index_list.append(index)
            assert len(frame_index_list) > 0
            
            ### 2. 按照调度计划执行
            frame_result_dict_computed = dict()
            for frame_index in frame_index_list:
                temp_frame = frame_list[frame_index]
                output_ctx = {
                    'image': temp_frame
                }
                
                frame_result_dict = dict()
                frame_result_dict['delay'] = dict()
                frame_result_dict['process_delay'] = dict()
                data_to_cloud = 0
                
                for taskname in pipeline:
                    ## 1. 进行资源限制
                    task_limit={}
                    task_limit['task_name'] = taskname
                    task_limit['cpu_util_limit'] = plan['resource_limit'][taskname]['cpu_util_limit']
                    task_limit['mem_util_limit'] = plan['resource_limit'][taskname]['mem_util_limit']
                    
                    device_type = plan['flow_mapping'][taskname]['node_role']
                    limit_rsc_url = get_limit_resource_url(device_type)

                    resp = limit_service_resource(limit_url=limit_rsc_url, task_limit=task_limit)
                    # 重试,完成资源限制
                    while not resp:
                        time.sleep(1)
                        resp = limit_service_resource(limit_url=limit_rsc_url, task_limit=task_limit)
                    
                    ## 2. 进行服务调用
                    input_ctx = output_ctx
                    service_url = get_chosen_service_url(device_type, taskname)
                    if device_type == 'cloud':
                        data_to_cloud += len(json.dumps(input_ctx).encode('utf-8'))
                    
                    st_time = time.time()
                    output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                    # 重试
                    while not output_ctx:
                        time.sleep(1)
                        output_ctx = invoke_service(serv_url=service_url, input_ctx=input_ctx)
                    ed_time = time.time()
                    
                    if device_type == 'cloud':
                        data_to_cloud += len(json.dumps(output_ctx).encode('utf-8'))
                    frame_result_dict[taskname] = output_ctx
                    frame_result_dict['delay'][taskname] = ed_time - st_time
                    frame_result_dict['process_delay'][taskname] = output_ctx['proc_resource_info']['compute_latency']
                
                total_frame_delay = 0
                total_frame_process_delay = 0
                for taskname in frame_result_dict['delay']:
                    frame_result_dict['delay'][taskname] = \
                        frame_result_dict['delay'][taskname] * fps / origin_fps 
                    total_frame_delay += frame_result_dict['delay'][taskname]
                
                for taskname in frame_result_dict['process_delay']:
                    frame_result_dict['process_delay'][taskname] = \
                        frame_result_dict['process_delay'][taskname] * fps / origin_fps 
                    total_frame_process_delay += frame_result_dict['process_delay'][taskname]
                
                frame_result_dict['delay_total'] = total_frame_delay
                frame_result_dict['process_delay_total'] = total_frame_process_delay
                trans_delay = data_to_cloud / (bandwidth_dict['kB/s'] * 1000)
                trans_delay = trans_delay * fps / origin_fps
                frame_result_dict['trans_delay_total'] = trans_delay
                frame_result_dict['delay_computed'] = frame_result_dict['process_delay_total'] + frame_result_dict['trans_delay_total']
                
                frame_result_dict_computed[frame_index] = frame_result_dict

            frame_result_list_total = []
            for i in range(len(frame_list)):
                if i in frame_result_dict_computed:
                    frame_result_list_total.append(frame_result_dict_computed[i])
                else:
                    frame_result_list_total.append(frame_result_list_total[-1])
            
            assert len(frame_result_list_total) == len(frame_list)

            temp_latency, temp_acc = get_lat_and_acc(pipeline, frame_act_result_list, frame_result_list_total, reso)
            
            updated_plan_info = copy.deepcopy(plan)
            updated_plan_info['latency'] = temp_latency
            updated_plan_info['accuracy'] =  temp_acc
            
            if temp_acc >= user_constraint['accuracy']:
                plans_in_acc_cons.append(updated_plan_info)
            else:
                plans_out_acc_cons.append(updated_plan_info)
        
        plans_in_acc_cons.sort(key=lambda x:x['latency'])
        plans_out_acc_cons.sort(key=lambda x:(-x['accuracy'], x['latency']))
        
        best_plan = plans_in_acc_cons[0] if len(plans_in_acc_cons) != 0 else plans_out_acc_cons[0]
        
        return best_plan
    
    return None


# 返回极端情况下的调度策略：保障一定精度下的最低配置、所有服务上云
def get_default_case(
    pipeline,
):
    conf = dict({"reso": "360p", "fps": 1, "encoder": "JPEG"})
    flow_mapping = dict()
    resource_limit = dict()
    cloud_ip = ''
    for device_ip in common.model_op.keys():
        if common.model_op[device_ip]['node_role'] == 'cloud':
            cloud_ip = device_ip
            break
    for serv_name in pipeline:
        flow_mapping[serv_name] = common.model_op[cloud_ip]
        resource_limit[serv_name] = {"cpu_util_limit": 1.0, "mem_util_limit": 1.0}
    
    return conf, flow_mapping, resource_limit
