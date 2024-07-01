import sys  
import optuna
import logging
optuna.logging.set_verbosity(logging.WARNING)
import itertools
import json
import os
from logging_utils import root_logger
from common import model_op,service_info_dict,conf_and_serv_info,ip_range,reso_range,fps_range, edge_cloud_cut_range, cpu_range, mem_range
import common
from RuntimePortrait import RuntimePortrait
from AccuracyPrediction import AccuracyPrediction
from kb_user_wzl import KnowledgeBaseUser
from macro_plan_helper import MacroPlanHelper
from DelayPredictModel import DelayPredictor


# 保存历史调度策略
prev_conf = dict()
prev_flow_mapping = dict()
prev_resource_limit = dict()
prev_runtime_info = dict()


def generate_combination(length):
    if length == 1:
        return [[0, 0], [-1, 0]]
    else:
        smaller_arrays = generate_combination(length - 1)
        arrays = []
        for array in smaller_arrays:
            arrays.append(array + [0, 0])
            arrays.append(array + [-1, 0])
        return arrays

def get_combination(list_dict, depth, length):
    if depth == length:
        return list_dict[depth]
    else:
        deeper_lists = get_combination(list_dict, depth+1, length)
        temp_lists = list_dict[depth]
        res_lists = []
        for temp_list in temp_lists:
            for deeper_list in deeper_lists:
                res_lists.append(temp_list + deeper_list)
        
        return res_lists
    

# 宏观调度计划制定函数，返回所有可能的宏观调度计划
def macro_judge(
    job_uid=None,
    exec_work_condition=None,
    act_work_condition=None,
    portrait_info=None,
    user_constraint=None,
    rsc_constraint=None,
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    all_proc_delay=None,
    bandwidth_dict=None
):

    # 工况为空或没有执行时延（执行结果），说明视频流分析还未开始，此时进行冷启动
    if not bool(exec_work_condition) or not bool(user_constraint) or all_proc_delay == 0:
        return {
            'cold_start_flag': True,
            'macro_plans': [],
        }
    
    # 从画像信息中提取调度策略而不是从云端保存的历史调度策略中获取，这是为了避免云边之间并发导致的策略不一致
    old_conf = portrait_info['exe_plan'][common.PLAN_KEY_VIDEO_CONF]
    old_flow_mapping = portrait_info['exe_plan'][common.PLAN_KEY_FLOW_MAPPING]
    old_resource_limit = portrait_info['exe_plan'][common.PLAN_KEY_RESOURCE_LIMIT]
    macro_plans = []
    
    ########################## 1. 确定各类配置的调整方向和范围 ##########################
    assert act_work_condition is not None
    conf_adjust_plans = []  # 配置调整的方向列表，每个元素为一个列表:[帧率调整方向, 分辨率调整方向]
    conf_adjust_direction = 0  # 配置应该提高还是降低
    conf_upper_bound = dict()  # 配置提高的上界
    conf_lower_bound = dict()  # 配置降低的下界
    
    acc_predict_model = AccuracyPrediction()
    middle_conf_list, middle_plus_conf_list = acc_predict_model.get_middle_conf(serv_names[0], user_constraint['accuracy'], act_work_condition['obj_size'], act_work_condition['obj_speed'])
    old_reso = old_conf['reso']
    old_fps = old_conf['fps']
    old_accuracy = acc_predict_model.predict(
        serv_names[0],
        {
            'reso': old_reso,
            'fps': old_fps
        },
        act_work_condition['obj_size'], 
        act_work_condition['obj_speed']
    )
    acc_constraint = user_constraint['accuracy']
    assert 'conf_portrait' in portrait_info
    
    # 说明：配置的强中弱类别划分应该使用真实的工况（在黄金配置下得到的工况）计算当前配置的精度，而不是使用执行结果提取的工况计算当前配置的精度。
    # 但由于目前画像的实现为每一帧结果都产生画像，若每一帧都计算真实工况则代价太大，因此修改调度器为每个调度周期在重新制定调度策略时对少量帧进行分析获取真实工况
    # 为了快速实现这一想法，将配置的画像类别在调度器中重新计算
    if old_accuracy < acc_constraint:
        portrait_info['conf_portrait'] = 0  # 低于精度约束，则配置画像类别为弱
    else:
        portrait_info['conf_portrait'] = 3  # 高于精度约束，则配置画像类别为强或中，具体类别放到下面进行判定
        
    
    if (old_fps, old_reso) in middle_conf_list:  # 若当前配置已经为中配置，则所有配置无需再调整
        portrait_info['conf_portrait'] = 1  # 配置画像类别为中
        conf_adjust_direction = 0
        conf_adjust_plans = [[0, 0]]
    else:
        for i in range(len(middle_conf_list)):  # 遍历所有的中配置，每一个中配置都为当前配置指定调整的方向
            middle_conf = middle_conf_list[i]
            temp_fps = middle_conf[0]
            temp_reso = middle_conf[1]
            temp_conf_adjust_plan = []
            
            if old_fps == temp_fps:  # 帧率的调整方向
                temp_conf_adjust_plan.append(0)
            elif old_fps > temp_fps:
                temp_conf_adjust_plan.append(-1)
            else:
                temp_conf_adjust_plan.append(1)
            
            if int(old_reso[:-1]) == int(temp_reso[:-1]):  # 分辨率的调整方向
                temp_conf_adjust_plan.append(0)
            elif int(old_reso[:-1]) > int(temp_reso[:-1]):
                temp_conf_adjust_plan.append(-1)
            else:
                temp_conf_adjust_plan.append(1)
            
            conf_adjust_plans.append(temp_conf_adjust_plan)
            temp_conf_adjust_plan_str = str(temp_conf_adjust_plan[0]) + '_' + str(temp_conf_adjust_plan[1])
            
            if portrait_info['conf_portrait'] == 0:  # 提高配置，给出配置提升的上界：当前中配置高一档
                conf_adjust_direction = 1
                conf_upper_bound[temp_conf_adjust_plan_str] = middle_plus_conf_list[i]
            else:
                conf_adjust_direction = -1  # 降低配置，给出配置降低的下界：当前中配置
                conf_lower_bound[temp_conf_adjust_plan_str] = middle_conf
    
    
    ########################## 2. 确定云边协同切分点的调整方向 ##########################
    edge_cloud_cut_plans = []  # 云边协同切分点调整的方向列表
    
    cur_cut_point_index = len(serv_names)  # 当前调度策略中云边协同切分点的位置：第一个被卸载到云端执行的服务的索引
    for i in range(len(serv_names)):
        if old_flow_mapping[serv_names[i]]['node_role'] == 'cloud':
            cur_cut_point_index = i
            break
    
    delay_predictor = DelayPredictor(serv_names)
    if cur_cut_point_index != 0:  # 尝试卸载更多的服务到云
        ###### 判断多卸载一个服务到云带来的执行时延收益与传输时延开销的大小 ######
        try_move_index = cur_cut_point_index - 1
        
        ### 执行时延收益
        cloud_exe_delay = -1  # 该服务在云端的理想执行时延
        if serv_names[try_move_index] == 'face_detection':
            cloud_exe_delay = delay_predictor.predict({
                    'delay_type': 'proc_delay',
                    'predict_info': {
                        'service_name': serv_names[try_move_index],  
                        'fps': old_fps,  
                        'reso': old_reso,
                        'node_role': 'server'
                    }
                })
        elif serv_names[try_move_index] == 'gender_classification':
            temp_obj_n = act_work_condition['obj_n']
            cloud_exe_delay = delay_predictor.predict({
                'delay_type': 'proc_delay',
                'predict_info': {
                    'service_name': serv_names[try_move_index],  
                    'fps': temp_fps,  
                    'obj_n': temp_obj_n,
                    'node_role': 'server'
                }
            })
        assert cloud_exe_delay != -1
        
        cur_edge_exe_delay = portrait_info['process_delay'][serv_names[try_move_index]]  # 该服务当前在边端的理想执行时延
        
        exe_delay_benefit = cur_edge_exe_delay - cloud_exe_delay  # 执行时延收益
        
        ### 传输时延开销
        temp_data_trans_size = portrait_info['data_trans_size'][serv_names[try_move_index]]
        temp_trans_delay = temp_data_trans_size / (bandwidth_dict['kB/s'] * 1000)  # 该服务挪到云端带来的传输时延开销
        
        if exe_delay_benefit >= 5*temp_trans_delay:  # 若执行时延收益远大于传输时延开销，则提高云边协同切分点设置
            edge_cloud_cut_plans.append(1)
        
    if cur_cut_point_index != len(serv_names):  # 尝试将更多服务拉回到边端
        ###### 判断多拉回一个服务到边带来的执行时延损失与传输时延收益的大小 ######
        ### 执行时延损失
        edge_exe_delay = -1  # 该服务在边端的理想执行时延，假设边端资源充足
        if serv_names[cur_cut_point_index] == 'face_detection':
            edge_exe_delay = delay_predictor.predict({
                    'delay_type': 'proc_delay',
                    'predict_info': {
                        'service_name': serv_names[cur_cut_point_index],  
                        'fps': old_fps,  
                        'reso': old_reso,
                        'node_role': 'edge'
                    }
                })
        elif serv_names[cur_cut_point_index] == 'gender_classification':
            temp_obj_n = act_work_condition['obj_n']
            edge_exe_delay = delay_predictor.predict({
                'delay_type': 'proc_delay',
                'predict_info': {
                    'service_name': serv_names[cur_cut_point_index],  
                    'fps': temp_fps,  
                    'obj_n': temp_obj_n,
                    'node_role': 'edge'
                }
            })
        assert edge_exe_delay != -1
        
        cur_cloud_exe_delay = portrait_info['process_delay'][serv_names[cur_cut_point_index]]  # 该服务当前在云端的执行时延
        
        exe_delay_loss = edge_exe_delay - cur_cloud_exe_delay
        
        ### 传输时延收益
        temp_data_trans_size = portrait_info['data_trans_size'][serv_names[cur_cut_point_index]]
        temp_trans_delay = temp_data_trans_size / (bandwidth_dict['kB/s'] * 1000)
        
        # 这里预估的边端的执行时延是在边端资源充足情况下的执行时延，实际的边端执行时延大于等于这里求出的时延，即实际执行时延的损失要高于这里求出来的执行时延损失。
        # 逻辑上，只有传输时延收益能够抵消掉较小的执行时延损失，我们才会给出降低云边协同切分点的建议；如果传输时延收益连较小的执行时延损失都无法抵消，那就更不可能抵消实际的、大的执行时延损失
        if temp_trans_delay >= 3*exe_delay_loss:  # 若传输时延收益明显高于执行时延损失，则降低云边协同切分点设置
            edge_cloud_cut_plans.append(-1)
        
    edge_cloud_cut_plans.append(0)  # 添加一个云边协同切分点不变的建议
    
    ########################## 3. 确定各个服务资源的调整方向 ##########################
    cut_rsc_plans = []  # 云边协同切分点与各个服务资源的调整方向列表，每个元素为一个列表:[云边协同切分点调整方向, 服务1CPU调整方向, 服务1内存调整方向, 服务2CPU调整方向, 服务2内存调整方向...]
    
    for edge_cloud_cut_plan in edge_cloud_cut_plans:
        temp_cut_rsc_plan = []
        temp_cut_rsc_plan.append(edge_cloud_cut_plan)
        
        if edge_cloud_cut_plan == 0:
            ### 1.确定计算资源的调整方向
            # 求出当前在边端执行的所有服务的CPU使用率之和
            cpu_total_alloc = 0
            device_ip = None
            for serv_name in serv_names:
                if old_flow_mapping[serv_name]['node_role'] != 'cloud':
                    device_ip = old_flow_mapping[serv_name]['node_ip']
                    cpu_total_alloc += old_resource_limit[serv_name]['cpu_util_limit']
            
            if device_ip is None or cpu_total_alloc == 0:  # 此时所有服务均在云端执行，则所有服务的资源分配方式不变
                for i in range(len(serv_names)):
                    temp_cut_rsc_plan.append(0)
                    temp_cut_rsc_plan.append(0)
                
                assert len(temp_cut_rsc_plan) == (1 + 2 * len(serv_names))
                cut_rsc_plans.append(temp_cut_rsc_plan)
            else:
                if cpu_total_alloc > rsc_constraint[device_ip]['cpu']:  # 在边端执行的服务超出了计算资源约束
                    edge_rsc_adjust_comb = generate_combination(cur_cut_point_index)  # 在边端执行的服务的资源调整方向组合，每个服务的计算资源可能不变也可能降低
                    cloud_rsc_adjust_plan = []  # 在云端执行的服务的资源调整方向为不变
                    for i in range(cur_cut_point_index, len(serv_names)):
                        cloud_rsc_adjust_plan += [0, 0]
                    for edge_rsc_adjust_plan in edge_rsc_adjust_comb:
                        temp_cut_rsc_plan_1 = temp_cut_rsc_plan + edge_rsc_adjust_plan + cloud_rsc_adjust_plan
                        
                        assert len(temp_cut_rsc_plan_1) == (1 + 2 * len(serv_names))
                        cut_rsc_plans.append(temp_cut_rsc_plan_1)
                
                else:  # 在边端执行的服务满足计算资源约束
                    edge_rsc_adjust_dict = dict()
                    assert 'resource_portrait' in portrait_info
                    for i in range(cur_cut_point_index):  # 判断各个在边端执行的服务的强中弱，并给出调整建议
                        if portrait_info['resource_portrait'] == 3:  # 满足资源约束且资源画像为强或中，则调整建议为不变
                            edge_rsc_adjust_dict[i] = [[0, 0]]
                        else:  # 资源画像为弱
                            ### 1.判断CPU资源是否为弱
                            temp_cpu_limit = old_resource_limit[serv_names[i]]["cpu_util_limit"]
                        
                            temp_cpu_demand = portrait_info['resource_info'][serv_names[i]]['resource_demand']['cpu']['edge']
                            temp_cpu_demand_upper_bound = temp_cpu_demand['upper_bound']
                            temp_cpu_demand_lower_bound = temp_cpu_demand['lower_bound']
                            
                            if temp_cpu_limit < temp_cpu_demand_lower_bound:  # CPU处于弱资源，调整建议为不变或提高资源
                                edge_rsc_adjust_dict[i] = [[0, 0], [1, 0]]
                            else:  # CPU资源处于中或强，但资源画像为弱，说明其他资源为弱，需要进一步判断其他资源。但由于目前只考虑CPU，所以代码先这样实现
                                edge_rsc_adjust_dict[i] = [[0, 0]]
                            
                            ### TODO：判断其他类型资源
                        '''
                        temp_cpu_limit = old_resource_limit[serv_names[i]]["cpu_util_limit"]
                        
                        temp_cpu_demand = portrait_info['resource_info'][serv_names[i]]['resource_demand']['cpu']['edge']
                        temp_cpu_demand_upper_bound = temp_cpu_demand['upper_bound']
                        temp_cpu_demand_lower_bound = temp_cpu_demand['lower_bound']
                        
                        if temp_cpu_limit < temp_cpu_demand_lower_bound:  # 处于弱资源的服务，调整建议为不变或提高资源
                            edge_rsc_adjust_dict[i] = [[0, 0], [1, 0]]
                        elif temp_cpu_limit >= temp_cpu_demand_lower_bound and temp_cpu_limit <= temp_cpu_demand_upper_bound:  # 处于中资源的服务，调整建议为不变
                            edge_rsc_adjust_dict[i] = [[0, 0]]
                        else:  # 处于强资源的服务，调整建议为不变或降低
                            edge_rsc_adjust_dict[i] = [[0, 0], [-1, 0]]
                        '''
                    
                    edge_rsc_adjust_plans = get_combination(edge_rsc_adjust_dict, 0, cur_cut_point_index-1)
                    
                    cloud_rsc_adjust_plan = []  # 在云端执行的服务的资源调整方向为不变
                    for i in range(cur_cut_point_index, len(serv_names)):
                        cloud_rsc_adjust_plan += [0, 0]
                    
                    for edge_rsc_adjust_plan in edge_rsc_adjust_plans:
                        temp_cut_rsc_plan_1 = edge_rsc_adjust_plan + cloud_rsc_adjust_plan
                        
                        assert len(temp_cut_rsc_plan_1) == (1 + 2 * len(serv_names))
                        cut_rsc_plans.append(temp_cut_rsc_plan_1)
                
        else:  # 若云边协同切分点发生了变化，则各个服务的资源应该自适应的调整，宏观指导难以给出各个服务资源明确的调整方向
            for i in range(len(serv_names)):
                temp_cut_rsc_plan.append(2)  # 服务的计算资源自适应的调整
                temp_cut_rsc_plan.append(0)  # 由于目前不考虑内存，服务的存储资源可始终设置为1.0并保持不变

            assert len(temp_cut_rsc_plan) == (1 + 2 * len(serv_names))
            cut_rsc_plans.append(temp_cut_rsc_plan)
    
    for conf_adjust_plan in conf_adjust_plans:
        for cut_rsc_plan in cut_rsc_plans:
            temp_macro_plan = conf_adjust_plan + cut_rsc_plan
            assert len(temp_macro_plan) == (3 + 2 * len(serv_names))
            macro_plans.append(temp_macro_plan)
    
    root_logger.info('New macro plans:{}'.format(macro_plans))
    return {
            'cold_start_flag': False,
            'macro_plans': macro_plans,
            'conf_adjust_direction': conf_adjust_direction,  # 配置调整的方向
            # 配置调整的边界
            'conf_upper_bound': conf_upper_bound,
            'conf_lower_bound': conf_lower_bound
        }


# 冷启动专用的贝叶斯优化查找函数
# 该函数的作用是使用KnowledgeBaseUser基于多次贝叶斯优化，试图找到一个可行解。其得出的结果必然是可用的。
# 使用该函数的时候，会在给定约束范围内寻找一个能够使用的最优解，但这个最优解本身未必能够使性能最优，因此只可用于冷启动。
def get_coldstart_plan_bayes(
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    rsc_upper_bound=None,
    rsc_down_bound=None,
    work_condition=None,
    portrait_info=None,
    bandwidth_dict=None
):
    # (1)建立冷启动器
    cold_starter = KnowledgeBaseUser(conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    user_constraint=user_constraint,
                                    rsc_constraint=rsc_constraint,
                                    rsc_upper_bound=rsc_upper_bound,
                                    rsc_down_bound=rsc_down_bound,
                                    work_condition=work_condition,
                                    portrait_info=portrait_info,
                                    bandwidth_dict=bandwidth_dict
                                    )

    # (2)依次尝试不同的n_trail，
    # 分别存储查到的满足约束和不满足约束的解。查找在找到一个绝佳解的时候停止。
    n_trials_range = [100,200,300,400,500]  # 尝试的贝叶斯优化查找次数
    params_in_acc_in_rsc_cons_total = []  # 既满足精度约束也满足资源约束的解
    params_in_acc_out_rsc_cons_total = []  # 满足精度约束但不满足资源约束的解
    params_out_acc_cons_total = []  # 不满足精度约束的解
    
    for n_trials in n_trials_range:
        params_in_acc_in_rsc_cons, params_in_acc_out_rsc_cons, params_out_acc_cons, _ = cold_starter.get_plan_in_cons_2(n_trials=n_trials)
        params_in_acc_in_rsc_cons_total.extend(params_in_acc_in_rsc_cons)
        params_in_acc_out_rsc_cons_total.extend(params_in_acc_out_rsc_cons)
        params_out_acc_cons_total.extend(params_out_acc_cons)
        
        # 看看完全满足约束的解中，是否能出现一个绝佳解。绝佳解定义为时延低于0.1，所以按照时延从小到大排序
        sorted_params_temp = sorted(params_in_acc_in_rsc_cons, key=lambda item:(item['pred_delay_total']))
        if len(sorted_params_temp) > 0 and sorted_params_temp[0]['pred_delay_total'] < 0.2:
            root_logger.info("找到一个绝佳解，停止继续搜索")
            break
    
    # （3）按照以下方式对得到的解进行处理：注意，对查到的解进行处理之后的调度策略是知识库中不存在的，这相当于一种探索，执行之后将其对应的结果写入到知识库中，实现对知识库的不断丰富
    '''
        (1).对于能满足精度和约束的解
        希望时延尽可能低

        (2).对于能满足精度约束但不能满足资源约束的解
        选择资源约束违反程度最小的，然后尝试在该设备上对资源进行均分，无法均分则挪到云端

        (3).对于不能满足精度约束的解
        选择精度违反程度最小的，然后将该解的配置降到最低。
    '''
    sorted_params = []
    post_process_index = -1  # 记录后处理进行的方式
    if len(params_in_acc_in_rsc_cons_total) > 0:
        post_process_index = 0
        sorted_params = sorted(params_in_acc_in_rsc_cons_total, key=lambda item:(item['pred_delay_total']))

    elif len(params_in_acc_out_rsc_cons_total) > 0:
        post_process_index = 1
        sorted_params = sorted(params_in_acc_out_rsc_cons_total, key=lambda item:(item['deg_violate']))

    elif len(params_out_acc_cons_total) > 0:
        post_process_index = 2
        sorted_params = sorted(params_out_acc_cons_total, key=lambda item:(-item['task_accuracy']))
    
    else:
        #如果找不到任何一类可用解，说明知识库严重不可用
        ans_found = 0
        conf = {}
        flow_mapping = {}
        resource_limit = {}
        root_logger.warning("In get_coldstart_plan_bayes(), 知识库未能获取任何可用解")
        return ans_found, conf, flow_mapping, resource_limit

    # 找到可调整的解，对其进行修正  
    best_params = sorted_params[0]
    ans_found = 1
    conf = best_params['conf']
    flow_mapping = best_params['flow_mapping']
    resource_limit = best_params['resource_limit']
    
    ''' 最优解的格式形如：
    {
    'conf': {'reso': '1080p', 'fps': 29, 'encoder': 'JPEG'}, 
    'flow_mapping': {'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
                     'gender_classification': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}}, 
    'resource_limit': {'face_detection': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}, 
                       'gender_classification': {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0}}, 
    'pred_delay_list': [0.48846824169158937, 0.00422081947326656], 
    'pred_delay_total': 0.49268906116485595, 
    'num_cloud': 2, 
    'deg_violate': 0, 
    'task_accuracy': 0.9589738585430643
    }
    '''

    if post_process_index == 0:  # 对于能满足精度和资源约束的最优解，无需处理直接返回
        return ans_found, conf, flow_mapping, resource_limit
    
    elif post_process_index == 1:  # 对于能满足精度但不能满足资源约束的解，首先尝试将资源按照约束值均分给各个服务，若无法均分则挪到云端
        serv_offload_cloud_idx = len(serv_names)  # 如果要将一些任务挪到云端，这个索引作为起始索引。初始为服务的总长，如果减小了，说明有必要挪到云端
        cloud_ip = ''
        for device_ip in rsc_constraint.keys():
            # 只针对非云设备计算违反资源约束程度
            if model_op[device_ip]['node_role'] == 'cloud':
                cloud_ip = device_ip
            else:  # 开始计算当前最优解是否在该设备上违反了资源约束; 如果违反了约束且不可均分，将该设备上的服务挪到云端
                cpu_util = 0
                mem_util = 0
                move_to_cloud = 0  # 如果不得不挪到云端，move_to_cloud会等于1，这发生在现有资源连0.05粒度的均匀分配都做不到的时候
                device_serv_names = []  # 当前ip上所使用的所有服务
                
                # 计算该设备上的资源消耗量
                for serv_name in resource_limit.keys():
                    if flow_mapping[serv_name]['node_ip'] == device_ip:
                        device_serv_names.append(serv_name)
                        cpu_util = round(cpu_util + resource_limit[serv_name]['cpu_util_limit'], 2)
                        mem_util = round(mem_util + resource_limit[serv_name]['mem_util_limit'], 2)
                
                # 如果违反了资源约束，进行平分（要求均分后的结果是0.05的倍数）
                if cpu_util > rsc_constraint[device_ip]['cpu']:
                    root_logger.info('In get_coldstart_plan_bayes(), cpu不满足约束,要重新分配')
                    theoretical_share = round(rsc_constraint[device_ip]['cpu'] / len(device_serv_names), 2)
                    cpu_share = round(int(theoretical_share / 0.05) * 0.05, 2)
                    if cpu_share == 0:
                        move_to_cloud = 1
                    else:  # 否则是可分的,将该设备上所有服务的cpu限制都改为此cpu_share
                        for serv_name in device_serv_names:
                            resource_limit[serv_name]['cpu_util_limit'] = cpu_share
                        
                # 如果违反了资源约束，进行平分（要求评分后的结果是0.05的倍数）
                if mem_util > rsc_constraint[device_ip]['mem']:
                    # 此时需要进行重新分配。
                    root_logger.info('In get_coldstart_plan_bayes(), mem不满足约束,要重新分配')
                    theoretical_share = round(rsc_constraint[device_ip]['mem'] / len(device_serv_names), 2)
                    mem_share = round(int(theoretical_share / 0.001) * 0.001, 3)
                    if mem_share == 0:
                        move_to_cloud = 1
                    else: #否则，是可分的,将该设备上所有服务的mem限制都改为此mem_share
                        for serv_name in device_serv_names:
                            resource_limit[serv_name]['mem_util_limit'] = mem_share
                
                # 如果move_to_cloud是1，就说明这个ip上的服务都要挪到云端去
                if move_to_cloud == 1:
                    for serv_name in device_serv_names:
                        # 开始重新定位起始被挪到云端的索引
                        serv_offload_cloud_idx = min(serv_offload_cloud_idx, serv_names.index(serv_name))
        
        for idx in range(serv_offload_cloud_idx, len(serv_names)):
            # 从这个索引开始的所有服务都要挪到云端
            serv_name = serv_names[idx]
            if flow_mapping[serv_name]['node_role'] != 'cloud':
                flow_mapping[serv_name] = model_op[cloud_ip]
                resource_limit[serv_name]['cpu_util_limit'] = 1.0
                resource_limit[serv_name]['mem_util_limit'] = 1.0

    elif post_process_index == 2:  # 对于无法满足精度约束解，不处理直接返回
        return ans_found, conf, flow_mapping, resource_limit
    
    # 最终会得到一个真正的最优解
    root_logger.info("In get_coldstart_plan_bayes(), 冷启动正式最优解:")
    root_logger.info('conf:{}'.format(conf))
    root_logger.info('flow_mapping:{}'.format(flow_mapping))
    root_logger.info('resource_limit:{}'.format(resource_limit))

    return ans_found, conf, flow_mapping, resource_limit

# 正常调度阶段使用的贝叶斯优化查找函数
# 该函数的作用是使用KnowledgeBaseUser基于多次贝叶斯优化，试图找到一个可行解。
def get_scheduler_plan_bayes(
    conf_names=None,
    serv_names=None,
    service_info_list=None,
    rsc_constraint=None,
    user_constraint=None,
    rsc_upper_bound=None,
    rsc_down_bound=None,
    work_condition=None,
    portrait_info=None,
    bandwidth_dict=None,
    macro_plan=None,
    macro_plan_dict=None
):
    #################### 1. 获取历史调度策略 ####################
    # 从画像信息中提取调度策略而不是从云端保存的历史调度策略中获取，这是为了避免云边之间并发导致的策略不一致
    old_conf = portrait_info['exe_plan'][common.PLAN_KEY_VIDEO_CONF]
    old_flow_mapping = portrait_info['exe_plan'][common.PLAN_KEY_FLOW_MAPPING]
    old_resource_limit = portrait_info['exe_plan'][common.PLAN_KEY_RESOURCE_LIMIT]
    old_edge_cloud_cut_choice = len(serv_names)
    for serv in serv_names:
        if old_flow_mapping[serv]['node_role'] == 'cloud':
            old_edge_cloud_cut_choice = serv_names.index(serv)
            break
    
    #################### 2. 根据宏观调度指导进行范围限制 ####################
    ########### 2.1 对帧率、分辨率范围进行限制 ###########
    old_fps_index = fps_range.index(old_conf['fps'])
    old_reso_index = reso_range.index(old_conf['reso'])
    if macro_plan_dict['conf_adjust_direction'] == 0:  # 若配置无需改变，则限定配置查找范围为当前配置
        new_fps_range = [old_conf['fps']]
        conf_and_serv_info['fps'] = new_fps_range
        
        new_reso_range = [old_conf['reso']]
        conf_and_serv_info['reso'] = new_reso_range
    
    else:  # 若配置需要改变，则依据宏观建议限定的上下界来限定范围
        adjust_str = str(macro_plan[0]) + '_' + str(macro_plan[1])
        if macro_plan_dict['conf_adjust_direction'] == 1:
            assert adjust_str in macro_plan_dict['conf_upper_bound']
            new_fps_bound, new_reso_bound = macro_plan_dict['conf_upper_bound'][adjust_str]
        else:
            assert adjust_str in macro_plan_dict['conf_lower_bound']
            new_fps_bound, new_reso_bound = macro_plan_dict['conf_lower_bound'][adjust_str]
        
        assert new_fps_bound in fps_range
        new_fps_index = fps_range.index(new_fps_bound)
        fps_min_index = min(old_fps_index, new_fps_index)
        fps_max_index = max(old_fps_index, new_fps_index)
        new_fps_range = fps_range[fps_min_index: fps_max_index+1]
        conf_and_serv_info['fps'] = new_fps_range
        
        assert new_reso_bound in reso_range
        new_reso_index = reso_range.index(new_reso_bound)
        reso_min_index = min(old_reso_index, new_reso_index)
        reso_max_index = max(old_reso_index, new_reso_index)
        new_reso_range = reso_range[reso_min_index: reso_max_index+1]
        conf_and_serv_info['reso'] = new_reso_range
        
    
    ########### 2.2 对云边协同方式进行限制 ###########
    new_edge_cloud_cut_range = []
    if macro_plan[2] == 0:
        new_edge_cloud_cut_range = [old_edge_cloud_cut_choice]
    elif macro_plan[2] == 1:
        if old_edge_cloud_cut_choice == 0:
            new_edge_cloud_cut_range = [0]
        else:
            new_edge_cloud_cut_range = [i for i in range(old_edge_cloud_cut_choice)]
    else:
        if old_edge_cloud_cut_choice == len(serv_names):
            new_edge_cloud_cut_range = [len(serv_names)]
        else:
            new_edge_cloud_cut_range = [i for i in range(old_edge_cloud_cut_choice + 1, len(serv_names) + 1)]
    conf_and_serv_info['edge_cloud_cut_point'] = new_edge_cloud_cut_range
    
    
    ########### 2.3 对资源调整范围进行限制 ###########
    for i in range(len(serv_names)):
        temp_str = serv_names[i] + '_cpu_util_limit'
        if macro_plan[3 + 2*i] == 2:
            conf_and_serv_info[temp_str] = cpu_range
        else:
            temp_cpu_limit = old_resource_limit[serv_names[i]]["cpu_util_limit"]
            if temp_cpu_limit == 1.0:  # 暂时认为只有在云端才会设置cpu利用率为1.0，因此在云端执行的服务CPU调整范围为[1.0]
                conf_and_serv_info[temp_str] = [1.0]
            else:
                temp_cpu_limit_index = cpu_range.index(temp_cpu_limit)
                if macro_plan[3 + 2*i] == 0:
                    conf_and_serv_info[temp_str] = [temp_cpu_limit]
                elif macro_plan[3 + 2*i] == 1:
                    if temp_cpu_limit_index == len(cpu_range) - 1:
                        conf_and_serv_info[temp_str] = [temp_cpu_limit]
                    else:
                        conf_and_serv_info[temp_str] = cpu_range[temp_cpu_limit_index+1: len(cpu_range)]
                else:
                    if temp_cpu_limit_index == 0:
                        conf_and_serv_info[temp_str] = [temp_cpu_limit]
                    else:
                        conf_and_serv_info[temp_str] = cpu_range[0: temp_cpu_limit_index]

        temp_str = serv_names[i] + '_mem_util_limit'
        conf_and_serv_info[temp_str] = mem_range
    
    ########### 2.4 对特殊情况进行判断 ###########
    ###### 2.4.1 宏观调度计划认为调度策略无需修改
    not_need_modify = True
    for plan in macro_plan:
        if plan != 0:
            not_need_modify = False
            break
    if not_need_modify:
        return 1, old_conf, old_flow_mapping, old_resource_limit, None
    
    ########### 2.5 查表 ###########
    ans_found = 0
    conf = {}
    flow_mapping = {}
    resource_limit = {}
        
    # (1).建立查表器
    plan_builder = KnowledgeBaseUser(conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    user_constraint=user_constraint,
                                    rsc_constraint=rsc_constraint,
                                    rsc_upper_bound=rsc_upper_bound,
                                    rsc_down_bound=rsc_down_bound,
                                    work_condition=work_condition,
                                    portrait_info=portrait_info,
                                    bandwidth_dict=bandwidth_dict
                                    )

    # (2)依次尝试不同的n_trail，
    # 分别存储查到的满足约束和不满足约束的解。查找在找到一个绝佳解的时候停止。
    n_trials_range = [100,200,300,400,500]  # 尝试的贝叶斯优化查找次数
    params_in_acc_in_rsc_cons_total = []  # 既满足时延约束也满足资源约束的解
    
    for n_trials in n_trials_range:
        params_in_acc_in_rsc_cons, _, _, next_plan = plan_builder.get_plan_in_cons_2(n_trials=n_trials)
        params_in_acc_in_rsc_cons_total.extend(params_in_acc_in_rsc_cons)
        
        # 看看完全满足约束的解中，是否能出现一个绝佳解。绝佳解定义为时延低于0.2，所以按照时延从小到大排序
        sorted_params_temp = sorted(params_in_acc_in_rsc_cons, key=lambda item:(item['pred_delay_total']))
        if len(sorted_params_temp) > 0 and sorted_params_temp[0]['pred_delay_total'] < 0.2:
            root_logger.info("找到一个绝佳解，停止继续搜索")
            break
    
    # （3）与冷启动不同，正式调度要求必须找到在当前宏观建议下既满足时延约束又满足资源约束的解，同时task_accuracy尽可能大
    sorted_params = []
    if len(params_in_acc_in_rsc_cons_total) == 0:
        return ans_found, conf, flow_mapping, resource_limit, next_plan
    else:
        sorted_params = sorted(params_in_acc_in_rsc_cons_total, key=lambda item:(item['pred_delay_total']))

    best_params = sorted_params[0]
    ans_found = 1
    conf = best_params['conf']
    flow_mapping = best_params['flow_mapping']
    resource_limit = best_params['resource_limit']
    
    # 恢复修改之前的值
    conf_and_serv_info['fps'] = fps_range
    conf_and_serv_info['reso'] = reso_range
    conf_and_serv_info['edge_cloud_cut_point'] = edge_cloud_cut_range
    for i in range(len(serv_names)):
        temp_str = serv_names[i] + '_cpu_util_limit'
        conf_and_serv_info[temp_str] = cpu_range
        
        temp_str = serv_names[i] + '_mem_util_limit'
        conf_and_serv_info[temp_str] = mem_range
    
    return ans_found, conf, flow_mapping, resource_limit, next_plan

# 返回极端情况下的调度策略：保障一定精度下的最低配置、所有服务上云
def get_extreme_case(
    serv_names=None,
):
    conf = dict({"reso": "480p", "fps": 5, "encoder": "JPEG"})
    flow_mapping = dict()
    resource_limit = dict()
    cloud_ip = ''
    for device_ip in model_op.keys():
        if model_op[device_ip]['node_role'] == 'cloud':
            cloud_ip = device_ip
            break
    for serv_name in serv_names:
        flow_mapping[serv_name] = model_op[cloud_ip]
        resource_limit[serv_name] = {"cpu_util_limit": 1.0, "mem_util_limit": 1.0}
    
    return conf, flow_mapping, resource_limit

# 将上一个调度周期内的调度策略及执行时延更新到对应的知识库中
def update_kb(conf_names=None,
            serv_names=None,
            service_info_list=None,
            rsc_constraint=None,
            user_constraint=None,
            rsc_upper_bound=None,
            rsc_down_bound=None,
            work_condition=None,
            portrait_info=None,
            bandwidth_dict=None):
    
    # (1).建立查表器
    plan_builder = KnowledgeBaseUser(conf_names=conf_names,
                                    serv_names=serv_names,
                                    service_info_list=service_info_list,
                                    user_constraint=user_constraint,
                                    rsc_constraint=rsc_constraint,
                                    rsc_upper_bound=rsc_upper_bound,
                                    rsc_down_bound=rsc_down_bound,
                                    work_condition=work_condition,
                                    portrait_info=portrait_info,
                                    bandwidth_dict=bandwidth_dict
                                    )
    
    
    # (2).更新知识库
    assert portrait_info
    plan_builder.update_kb(portrait_info['exe_plan'], portrait_info['process_delay'])
    

def scheduler(
    job_uid=None,
    dag=None,
    system_status=None,
    portrait_info=None,
    user_constraint=None,
    bandwidth_dict=None,
    act_work_condition=None
):
    root_logger.info('当前调度器感知到边缘到云端的带宽是:{}'.format(bandwidth_dict))

    assert job_uid, "should provide job_uid"
    
    # 调度器首先要根据dag来得到关于服务的各个信息
    #（1）构建serv_names:根据dag获取serv_names，涉及的各个服务名
    #（2）构建service_info_list：根据serv_names，加上“_trans”构建传输阶段，从service_info_dict提取信息构建service_info_list
    # (3) 构建conf_names：根据service_info_list里每一个服务的配置参数，得到conf_names，保存所有影响任务的配置参数（帧率、分辨率等，不含ip）
    #（4）获取每一个设备的资源约束，以及每一个服务的资源上限
    serv_names = dag['flow']
    
    # 获得service_info_list
    service_info_list = []  # 建立service_info_list，需要处理阶段，也需要传输阶段
    for serv_name in serv_names:
        service_info_list.append(service_info_dict[serv_name])
        # 不考虑传输时延
        # service_info_list.append(service_info_dict[serv_name+"_trans"])
    
    # 获得conf_names
    conf_names = []  # conf_names里保存所有配置参数的名字
    for service_info in service_info_list:
        service_conf_names = service_info['conf']
        for service_conf_name in service_conf_names:
            if service_conf_name not in conf_names:
                conf_names.append(service_conf_name)  
    
    # 获得各阶段处理时延all_proc_delay
    all_proc_delay = 0
    if portrait_info:
        all_proc_delay = portrait_info['cur_process_latency']
        root_logger.info('感知到总处理时延是:{}'.format(all_proc_delay))
    
    # 获取当前工况
    exec_work_condition = dict()
    if portrait_info:
        assert 'work_condition' in portrait_info
        exec_work_condition = portrait_info['work_condition']
    
    # 获得设备的资源约束rsc_constraint
    rsc_constraint = user_constraint['rsc_constraint']
    
    
    ############### 1. 获取宏观调度建议列表 ###############
    macro_plan_dict = macro_judge(
                                job_uid=job_uid,
                                exec_work_condition=exec_work_condition,
                                act_work_condition=act_work_condition,
                                portrait_info=portrait_info,
                                user_constraint=user_constraint,
                                rsc_constraint=rsc_constraint,
                                conf_names=conf_names,
                                serv_names=serv_names,
                                service_info_list=service_info_list,
                                all_proc_delay=all_proc_delay,
                                bandwidth_dict=bandwidth_dict
                                )
    
    ############### 2. 根据宏观调度建议列表进行微观搜索 ###############
    myportrait = RuntimePortrait(pipeline=serv_names, user_constraint=user_constraint)
    rsc_upper_bound = {}
    rsc_down_bound = {}
    for serv_name in serv_names:
        serv_rsc_cons = myportrait.help_cold_start(service=serv_name)
        rsc_upper_bound[serv_name] = {}
        rsc_upper_bound[serv_name]['cpu_limit'] = serv_rsc_cons['cpu']['edge']
        rsc_upper_bound[serv_name]['mem_limit'] = serv_rsc_cons['mem']['edge']
        rsc_down_bound[serv_name] = {}
        rsc_down_bound[serv_name]['cpu_limit'] = 0.0
        rsc_down_bound[serv_name]['mem_limit'] = 0.0
        
    if macro_plan_dict['cold_start_flag']:
        root_logger.info("进行冷启动")
            
        ans_found, conf, flow_mapping, resource_limit = get_coldstart_plan_bayes(conf_names=conf_names,
                                                                                serv_names=serv_names,
                                                                                service_info_list=service_info_list,
                                                                                rsc_constraint=rsc_constraint,
                                                                                user_constraint=user_constraint,
                                                                                rsc_upper_bound=rsc_upper_bound,
                                                                                rsc_down_bound=rsc_down_bound,
                                                                                work_condition=exec_work_condition,
                                                                                portrait_info=portrait_info,
                                                                                bandwidth_dict=bandwidth_dict)
        
        if ans_found == 1:  # 如果确实找到了，需要更新字典里之前保存的三大配置
            prev_conf[job_uid] = conf
            prev_flow_mapping[job_uid] = flow_mapping
            prev_resource_limit[job_uid] = resource_limit
        else:
            root_logger.warning("查表已经失败,使用极端配置")
            conf, flow_mapping, resource_limit = get_extreme_case(serv_names=serv_names)
            prev_conf[job_uid] = conf
            prev_flow_mapping[job_uid] = flow_mapping
            prev_resource_limit[job_uid] = resource_limit
            root_logger.warning('最终采用:极端情况')
        
        # 为了测试方便，以下人为设置初始冷启动值，以便查看更多稳定可靠的效果。但是下面这一部分代码实际上不能作为真正的冷启动。
        '''
        conf = dict({
        "reso": "360p", "fps": 30, "encoder": "JPEG", 
        })
        flow_mapping = dict({
            "face_detection": {"model_id": 0, "node_ip": "192.168.1.7", "node_role": "host"}, 
            "gender_classification": {"model_id": 0, "node_ip": "192.168.1.7", "node_role": "host"}
        })
        resource_limit = dict({
            "face_detection": {"cpu_util_limit": 0.25, "mem_util_limit": 0.004}, 
            "gender_classification": {"cpu_util_limit": 0.75, "mem_util_limit": 0.008}
        })
        prev_conf[job_uid] = conf
        prev_flow_mapping[job_uid] = flow_mapping
        prev_resource_limit[job_uid] = resource_limit
        root_logger.info('目前使用了默认配置代替冷启动过程')
        '''
        
    else:
        root_logger.info("正常调度周期重新制定调度策略")
        ############### 2.1 判断前一阶段的使用的调度策略是否在知识库中，若不在则更新知识库 ###############
        update_kb(conf_names=conf_names,
                serv_names=serv_names,
                service_info_list=service_info_list,
                rsc_constraint=rsc_constraint,
                user_constraint=user_constraint,
                rsc_upper_bound=rsc_upper_bound,
                rsc_down_bound=rsc_down_bound,
                work_condition=exec_work_condition,
                portrait_info=portrait_info,
                bandwidth_dict=bandwidth_dict
            )
        
        ############### 2.2 按照两重贝叶斯优化的方式确定最优解 ###############
        ############### 2.2.1 第一重贝叶斯优化：选择最优的调度方向 ###############
        macro_plan_selector = MacroPlanHelper(conf_names=conf_names,
                                            serv_names=serv_names,
                                            service_info_list=service_info_list,
                                            rsc_constraint=rsc_constraint,
                                            user_constraint=user_constraint,
                                            rsc_upper_bound=rsc_upper_bound,
                                            rsc_down_bound=rsc_down_bound,
                                            work_condition=act_work_condition,
                                            portrait_info=portrait_info,
                                            bandwidth_dict=bandwidth_dict,
                                            macro_plan_dict=macro_plan_dict)
        
        k = 3  # 最优的宏观调度计划数量
        top_k_macro_plan_index_list = macro_plan_selector.get_top_k_macro_plans(k)
        
        ############### 2.2.2 第二重贝叶斯优化：按照优先级在top-k个调度方向上寻找最优解 ###############
        next_try_plan = None  # 在最优的宏观建议下，查表的贝叶斯优化模型下一个建议的调度计划，当所有宏观调度计划均在知识库中查找失败时使用此调度计划
        if_find_best_solution = False
        for macro_plan_index in top_k_macro_plan_index_list:
            ans_found, conf, flow_mapping, resource_limit, next_plan = get_scheduler_plan_bayes(conf_names=conf_names,
                                                                                    serv_names=serv_names,
                                                                                    service_info_list=service_info_list,
                                                                                    rsc_constraint=rsc_constraint,
                                                                                    user_constraint=user_constraint,
                                                                                    rsc_upper_bound=rsc_upper_bound,
                                                                                    rsc_down_bound=rsc_down_bound,
                                                                                    work_condition=act_work_condition,
                                                                                    portrait_info=portrait_info,
                                                                                    bandwidth_dict=bandwidth_dict,
                                                                                    macro_plan=macro_plan_dict['macro_plans'][macro_plan_index],
                                                                                    macro_plan_dict=macro_plan_dict
                                                                                    )
            if next_try_plan is None:
                next_try_plan = next_plan
            
            if ans_found == 1:
                prev_conf[job_uid] = conf
                prev_flow_mapping[job_uid] = flow_mapping
                prev_resource_limit[job_uid] = resource_limit
                if_find_best_solution = True
                break
            
        if not if_find_best_solution:  # 若根据宏观建议列表未能在知识库中找到一个满足约束的解，则使用next_try_plan进行探索
            prev_conf[job_uid] = next_try_plan["conf"]
            prev_flow_mapping[job_uid] = next_try_plan["flow_mapping"]
            prev_resource_limit[job_uid] = next_try_plan["resource_limit"]
            
          
    return prev_conf[job_uid], prev_flow_mapping[job_uid], prev_resource_limit[job_uid]  # 沿用之前的配置
