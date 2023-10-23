from logging_utils import root_logger
import pandas as pd
import os

import time
import copy

prev_video_conf = dict()

prev_flow_mapping = dict()

init_video_conf = dict()
init_flow_mapping = dict()

prev_runtime_info = dict()

available_fps = [1, 5, 10, 20, 30]
available_resolution = ["360p", "480p", "720p", "1080p"]
# available_npxpf = [480*360, 858*480, 1280*720, 1920*1080]

lastTime = time.time()


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.dt = dt
        self.previous_error = 0
        self.integral = 0

    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        print(output)
        return output

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
def get_pred_delay(conf_fps=None, cam_fps=None, resolution=None, flow_map=None, resource_info=None):
    # 给定flow_map，
    # resolution vs process_delay：基于kb
    # resolution vs transfer_delay：基于带宽计算
    # fps vs delay：比例关系

    process_sum_delay = get_process_delay(resolution=resolution, flow_map=flow_map)
    transfer_sum_delay = get_transfer_delay(resolution=resolution, flow_map=flow_map, resource_info=resource_info)

    total_delay = (process_sum_delay + transfer_sum_delay) * conf_fps / cam_fps

    return total_delay

'''
# TODO：给定fps和resolution，结合运行时情境，获取预测时延
def get_pred_acc(conf_fps=None, cam_fps=None, resolution=None, runtime_info=None):
    if runtime_info and 'obj_stable' in runtime_info:
        if not runtime_info['obj_stable'] and conf_fps < 20:
            return 0.6
    return 0.9
'''

#'''
# TODO：给定fps和resolution，结合运行时情境，获取预测时延
def get_pred_acc(conf_fps=None, cam_fps=None, resolution=None, runtime_info=None):
    if runtime_info and 'obj_stable' in runtime_info:
        if runtime_info['obj_stable']>=0.3 and conf_fps < 20:
            return 0.6
    return 0.9
#'''









# ---------------
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
    user_constraint=None,
):
    assert job_uid, "should provide job_uid"

    global prev_video_conf, prev_flow_mapping
    global init_video_conf,init_flow_mapping
    global available_fps, available_resolution

    # 时延优先策略：算量最小，算力最大
    cold_video_conf = {
        "resolution": "360p",
        "fps": 30,
        # "ntracking": 5,
        "encoder": "JPEG",
    }
    cold_flow_mapping = dict()
    for taskname in dag["flow"]:
        cold_flow_mapping[taskname] = {
            "model_id": 0,
            "node_role": "host",
            "node_ip": list(resource_info["host"].keys())[0]
        }

    delay_ub = user_constraint["delay"]
    delay_lb = delay_ub
    acc_ub = user_constraint["accuracy"]
    acc_lb = acc_ub

    min_delay_delta = None
    min_acc_delta = None

    # 调度维度：nproc，切分点，fps，resolution
    for fps in available_fps:
        for resol in available_resolution:
            for offload_ptr in range(0, len(dag["flow"])):
                # 枚举所有策略，根据knowledge base预测时延和精度，找出符合用户约束的。
                # 若无法同时满足，优先满足时延要求。尽量满足精度要求（不要求是最优解，所以可以提前退出）
                flow_map = get_flow_map(dag=dag,
                                        resource_info=resource_info, 
                                        offload_ptr=offload_ptr)
                cam_fps = 30.0
                delay = get_pred_delay(conf_fps=fps, cam_fps=cam_fps,
                                       resolution=resol,
                                       flow_map=flow_map,
                                       resource_info=resource_info)
                acc = get_pred_acc(conf_fps=fps, cam_fps=cam_fps,
                                   resolution=resol)
                
                if delay < delay_ub:
                    # 若时延符合要求，找最符合精度要求的
                    # 防止符合要求的配置被替换
                    min_delay_delta = 0.0
                    if not min_acc_delta or min_acc_delta > abs(acc_lb - acc):
                        cold_video_conf["resolution"] = resol
                        cold_video_conf["fps"] = fps
                        cold_flow_mapping = flow_map
                        min_acc_delta = abs(acc_lb - acc)
                else:
                    # 若时延不符合要求，找出尽量符合的
                    if not min_delay_delta or min_delay_delta > abs(delay_ub - delay):
                        cold_video_conf["resolution"] = resol
                        cold_video_conf["fps"] = fps
                        cold_flow_mapping = flow_map
                        min_delay_delta = abs(delay_ub - delay)

    prev_video_conf[job_uid] = cold_video_conf
    prev_flow_mapping[job_uid] = cold_flow_mapping
    init_video_conf[job_uid] = copy.deepcopy(cold_video_conf)
    init_flow_mapping[job_uid] =copy.deepcopy(cold_flow_mapping)
    print("当前边缘",list(resource_info["host"].keys()),"冷启动策略",init_flow_mapping[job_uid])

    return prev_video_conf[job_uid], prev_flow_mapping[job_uid]









# -------------------------------------------
# ---- TODO：根据资源情境，尝试分配更多资源----
def try_expand_resource(next_flow_mapping=None, err_level=None, resource_info=None,job_uid=None):
    #进行粗略版本的多边协同修改：node_role为host时，暂且将node_ip挪到其他边缘。挪动方式是遍历从0开始的u偶有边缘，如果遍历一圈回来，就挪到云端。
    global  init_flow_mapping
    tune_msg = None
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
                    # if next_flow_mapping[taskname]["node_ip"]!=init_flow_mapping[job_uid][taskname]["node_ip"]: 这里需要注意，一开始任务冷启动的时候可能卸载到云端。所以不能直接和冷启动时的ip比较。
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


# -------------------------------------------
# ---- TODO：基于try_expand_resource改进得到，在时延不满足要求时，根据需要决定是卸载到云端还是卸载到边缘----
#   try_adjust_resource(next_flow_mapping=next_flow_mapping, err_level=err_level, resource_info=resource_info,runtime_info=runtime_info,user_constraint=user_constraint,job_uid=job_uid)
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

# -----------------------------------------
# ---- TODO：根据应用情境，尝试减少计算量 ----
def try_reduce_calculation(
    next_video_conf=None,
    err_level=None,
    runtime_info=None,
    init_prior=1,
    best_effort=False
):
    global available_fps, available_resolution

    resolution_index = available_resolution.index(
        next_video_conf["resolution"])
    fps_index = available_fps.index(next_video_conf["fps"])

    tune_msg = None

    # TODO：根据运行时情境初始化优先级，实现最佳匹配
    total_prior = 2
    curr_prior = init_prior

    # 无法最佳匹配时，根据收益大小优先级调度
    while True:
        if curr_prior == 1:
            if fps_index > 0:
                print(" -------- fps lower -------- (init_prior={})".format(init_prior))
                next_video_conf["fps"] = available_fps[fps_index - 1]
                tune_msg = "fps {} -> {}".format(available_fps[fps_index],
                                                available_fps[fps_index - 1])

        if curr_prior == 0:
            if resolution_index > 0:
                print(" -------- resolution lower -------- (init_prior={})".format(init_prior))
                next_video_conf["resolution"] = available_resolution[resolution_index - 1]
                tune_msg = "resolution {} -> {}".format(available_resolution[resolution_index],
                                                        available_resolution[resolution_index - 1])
        
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

# ----------------
# ---- 负反馈 ----
def adjust_parameters(output=0, job_uid=None,
                      dag=None,
                      user_constraint=None,
                      resource_info=None,
                      runtime_info=None):
    assert job_uid, "should provide job_uid"

    global prev_video_conf, prev_flow_mapping, prev_runtime_info
    global available_fps, available_resolution

    next_video_conf = prev_video_conf[job_uid]
    next_flow_mapping = prev_flow_mapping[job_uid]

    # 仅支持pipeline
    flow = dag["flow"]
    assert isinstance(flow, list), "flow not list"

    resolution_index = available_resolution.index(
        next_video_conf["resolution"])
    fps_index = available_fps.index(next_video_conf["fps"])

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
                                resolution=next_video_conf["resolution"],
                                runtime_info=runtime_info)
        
        # 若此时预测精度达不到要求，可以提高fps和resolution
        if pred_acc < user_constraint["accuracy"]:
            # 根据不同程度的 delay-acc trade-off，在不同的delay级别调整不同的参数
            while not tune_msg and tune_level > 0:
                if tune_level == 2:
                    if fps_index + 1 < len(available_fps):
                        print(" -------- fps higher -------- (err_level={}, tune_msg={})".format(err_level, tune_msg))
                        next_video_conf["fps"] = available_fps[fps_index + 1]
                        tune_msg = "fps {} -> {}".format(available_fps[fps_index],
                                                        available_fps[fps_index + 1])

                elif tune_level == 1:
                    if resolution_index + 1 < len(available_resolution):
                        print(" -------- resolution higher -------- (err_level={}, tune_msg={})".format(err_level, tune_msg))
                        next_video_conf["resolution"] = available_resolution[resolution_index + 1]
                        tune_msg = "resolution {} -> {}".format(available_resolution[resolution_index],
                                                                available_resolution[resolution_index + 1])
                
                # 按优先级依次选择可调的配置
                if not tune_msg:
                    tune_level -= 1
        else:
            '''
            if 'obj_stable' in runtime_info and runtime_info['obj_stable']:
                # 场景稳定，优先降低帧率
                init_prior = 1
                best_effort = False
                tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                                err_level=err_level, 
                                                                runtime_info=runtime_info,
                                                                init_prior=init_prior, best_effort=best_effort)
            '''

            #'''
            if 'obj_stable' in runtime_info and (runtime_info['obj_stable']<0.3):
                # 场景稳定，优先降低帧率
                init_prior = 1
                best_effort = False
                tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                                err_level=err_level, 
                                                                runtime_info=runtime_info,
                                                                init_prior=init_prior, best_effort=best_effort)
            #'''
    elif err_level < 0:
        # level < 0，时延不满足要求
        # TODO：结合运行时情境（资源），应该调整策略，以降低时延：
        #              （1）若场景稳定性，降低帧率；若场景目标较大，降低分辨率
        #              （2）分配更多资源；
        #              （3）任务卸载到空闲节点（云/边）；
        #              （4）最后考虑降低fps和resolution；
        #       结合运行时情境（应用），调整fps和resolution，比如：
        #              场景稳定则优先降低fps（对精度影响较小）
        #              物体较大则降低resolution（对精度影响较小）
        '''
        if 'obj_stable' in runtime_info and runtime_info['obj_stable']:
            # 场景稳定，优先降低帧率
            init_prior = 1
            best_effort = False
            tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                               err_level=err_level, 
                                                               runtime_info=runtime_info,
                                                               init_prior=init_prior, best_effort=best_effort)
        '''
        #'''
        if 'obj_stable' in runtime_info and (runtime_info['obj_stable']<0.3):
            # 场景稳定，优先降低帧率
            init_prior = 1
            best_effort = False
            tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                               err_level=err_level, 
                                                               runtime_info=runtime_info,
                                                               init_prior=init_prior, best_effort=best_effort)
        #'''

        elif 'obj_size' in runtime_info and runtime_info['obj_size'] > 500:
            # 场景不稳定，但物体够大，优先降低分辨率
            init_prior = 0
            best_effort = False
            tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                               err_level=err_level, 
                                                               runtime_info=runtime_info,
                                                               init_prior=init_prior, best_effort=best_effort)



        if not tune_msg:
            # tune_msg, next_flow_mapping = try_expand_resource(next_flow_mapping=next_flow_mapping, err_level=err_level, resource_info=resource_info,job_uid=job_uid)
            tune_msg, next_flow_mapping = try_adjust_resource(next_flow_mapping=next_flow_mapping, err_level=err_level, resource_info=resource_info,runtime_info=runtime_info,user_constraint=user_constraint,job_uid=job_uid)

        if not tune_msg:
            # 资源分配完毕，且无法根据情境降低计算量，则按收益大小降低计算量
            init_prior = 1
            best_effort = True
            tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf, 
                                                               err_level=err_level, 
                                                               runtime_info=runtime_info,
                                                               init_prior=init_prior, best_effort=best_effort)

    prev_video_conf[job_uid] = next_video_conf
    prev_flow_mapping[job_uid] = next_flow_mapping
    prev_runtime_info[job_uid] = runtime_info

    print(prev_flow_mapping[job_uid])
    print(prev_video_conf[job_uid])
    print(prev_runtime_info[job_uid])
    root_logger.info("tune_msg: {}".format(tune_msg))
    
    return prev_video_conf[job_uid], prev_flow_mapping[job_uid]









# -----------------
# ---- 调度入口 ----
def scheduler(
    job_uid=None,
    dag=None,
    resource_info=None,
    runtime_info=None,
    user_constraint=None,
):
    print("当前调度器获得的runtime_info(含各阶段时延):")
    print(runtime_info)

    assert job_uid, "should provide job_uid for scheduler to get prev_plan of job"

    root_logger.info(
        "scheduling for job_uid-{}, runtime_info=\n{}".format(job_uid, runtime_info))

    global lastTime

    if not bool(runtime_info) or not bool(user_constraint):
        root_logger.info("to get COLD start executation plan")
        return get_cold_start_plan(
            job_uid=job_uid,
            dag=dag,
            resource_info=resource_info,
            user_constraint=user_constraint
        )

    # ---- 若有负反馈结果，则进行负反馈调节 ----
    global prev_video_conf, prev_flow_mapping

    assert job_uid in prev_video_conf, \
        "job_uid not in prev_video_conf(keys={})".format(
            prev_video_conf.keys())
    assert job_uid in prev_flow_mapping, \
        "job_uid not in prev_video_conf(keys={})".format(
            prev_flow_mapping.keys())

    video_conf = None
    flow_mapping = None

    delay_ub = user_constraint["delay"]
    delay_lb = delay_ub

    # set pidController param
    Kp, Ki, Kd = 1, 0.1, 0.01
    setpoint = delay_ub
    dt = time.time() - lastTime
    pidControl = PIDController(Kp, Ki, Kd, setpoint, dt)

    # TODO：参照对应的边端sniffer解析运行时情境
    print('---- runtime_info in the past time slot ----')
    print('runtime_info = {}'.format(runtime_info))

    avg_delay = runtime_info['delay']
    output = pidControl.update(avg_delay)

    # adjust parameters

    return adjust_parameters(output, job_uid=job_uid,
                             dag=dag,
                             user_constraint=user_constraint,
                             resource_info=resource_info,
                             runtime_info=runtime_info)
