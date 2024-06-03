import json
import datetime
import matplotlib.pyplot as plt
import optuna
import itertools
import logging
import field_codec_utils
import cv2
import sys
from logging_utils import root_logger
optuna.logging.set_verbosity(logging.WARNING)  

plt.rcParams['font.sans-serif'] = ['SimHei'] # 运行配置参数中的字体（font）为黑体（SimHei）

import common
from common import KB_DATA_PATH,MAX_NUMBER, model_op, conf_and_serv_info, service_info_dict
from AccuracyPrediction import AccuracyPrediction

#试图以类的方式将整个独立建立知识库的过程模块化

          
# KnowledgeBaseUser可以基于建立好的知识库，提供冷启动所需的一系列策略
# 实际上它可以充分使用知识库
class  KnowledgeBaseUser():   
    
    #冷启动计划者，初始化时需要conf_names,serv_names,service_info_list,user_constraint一共四个量
    #在初始化过程中，会根据这些参数，制造conf_list，serv_ip_list，serv_cpu_list以及serv_meme_list
    def __init__(self, conf_names, serv_names, service_info_list, user_constraint, rsc_constraint, rsc_upper_bound, rsc_down_bound, work_condition, portrait_info, bandwidth_dict, conf_adjust_direction=None, conf_upper_bound=None, conf_lower_bound=None):
        self.conf_names = conf_names
        self.serv_names = serv_names
        self.service_info_list = service_info_list
        self.user_constraint = user_constraint
        self.rsc_constraint = rsc_constraint
        self.rsc_upper_bound = rsc_upper_bound
        self.rsc_down_bound = rsc_down_bound
        self.work_condition = work_condition
        self.portrait_info = portrait_info
        self.bandwidth_dict = bandwidth_dict
        self.conf_adjust_direction = conf_adjust_direction
        self.conf_upper_bound = conf_upper_bound
        self.conf_lower_bound = conf_lower_bound

        # 工况和画像均为空，则此时知识库用于冷启动；否则用于正常调度
        self.cold_start_flag = True if not self.work_condition and not self.portrait_info else False
        
        self.conf_list = []
        for conf_name in conf_names:   #conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip和资源限制
            self.conf_list.append(conf_and_serv_info[conf_name])
        # conf_list会包含各类配置参数取值范围，例如分辨率、帧率等

        self.serv_ip_list = []
        for serv_name in serv_names:
            serv_ip = serv_name + "_ip"
            self.serv_ip_list.append(conf_and_serv_info[serv_ip])
        # serv_ip_list包含各个模型的ip的取值范围

        self.serv_cpu_list = []
        for serv_name in serv_names:
            serv_cpu = serv_name + "_cpu_util_limit"
            self.serv_cpu_list.append(conf_and_serv_info[serv_cpu])
        # serv_cpu_list包含各个模型的cpu使用率的取值范围

        self.serv_mem_list = []
        for serv_name in serv_names:
            serv_mem = serv_name + "_mem_util_limit"
            self.serv_mem_list.append(conf_and_serv_info[serv_mem])
        # serv_mem_list包含各个模型的mem使用率的取值范围
        
        self.n_trial_unit = 20  # 贝叶斯优化查找的粒度
        
        #例如,可能的一组取值如下：
        # conf_names=["reso","fps","encoder"]
        # serv_names=["face_detection", "face_alignment"]
        # conf_list=[ ["360p", "480p", "720p", "1080p"],[1, 5, 10, 20, 30],["JPEG"]]
        # serv_ip_list=[["114.212.81.11","172.27.143.164","172.27.151.145"],["114.212.81.11","172.27.143.164","172.27.151.145"]]
        # serv_cpu_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]
        # serv_mem_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]

    # 知识库更新接口，调度器使用此接口将前一调度周期内使用的调度策略与性能指标之间的关系更新入知识库
    def update_kb(self, exe_plan, proc_delay_dict):
        # 更新所有服务的处理时延知识库
        for serv_name in self.serv_names:
            # 1. 加载服务已有的知识库
            kb_file_name = KB_DATA_PATH + '/' + serv_name + ".json"
            with open(kb_file_name, 'r', encoding='utf-8') as file:  
                evaluator = json.load(file)
            
            # 2. 将服务当前的调度策略转为知识库中key
            temp_conf = exe_plan["video_conf"]
            temp_flow_mapping = exe_plan["flow_mapping"]
            temp_resource_limit = exe_plan["resource_limit"]
            
            temp_key_str = ''
            for conf in self.conf_names:
                temp_key_str += (conf + '=' + str(temp_conf[conf]) + ' ')
            
            temp_key_str += (serv_name + '_ip' + '=' + temp_flow_mapping[serv_name]['node_ip'] + ' ')
            temp_key_str += (serv_name + '_cpu_util_limit' + '=' + str(temp_resource_limit[serv_name]['cpu_util_limit']) + ' ')
            temp_key_str += (serv_name + '_mem_util_limit' + '=' + str(temp_resource_limit[serv_name]['mem_util_limit']) + ' ')
            
            temp_proc_delay = proc_delay_dict[serv_name]
            
            # 3. 判断当前调度策略是否在知识库中，若不在，则写入知识库
            if temp_key_str not in evaluator:
                evaluator[temp_key_str] = temp_proc_delay
                with open(kb_file_name, 'w', encoding='utf-8') as file:
                    json.dump(evaluator, file)

                root_logger.info("更新服务{}的知识库, key为:{}".format(serv_name, temp_key_str))
    
    
    # get_pred_delay：
    # 用途：根据参数里指定的配置，根据知识库来预测对应性能，如果在知识库中找不到，则返回status为0；否则为1
    # 方法：依次遍历评估性能所需的各个字典并根据参数设置从中获取性能评估结果
    # 返回值：status,pred_delay_list,pred_delay_total，描述性能评估成果与否、预测的各阶段时延、预测的总时延

    def get_pred_delay(self, conf, flow_mapping, resource_limit, edge_cloud_cut_choice):
        # 存储配置对应的各阶段时延，以及总时延
        pred_delay_list = []
        pred_delay_total = 0
        status = 0  # 为0表示字典中没有满足配置的存在
        # 对于service_info_list里的service_info依次评估性能
        for service_info in self.service_info_list:
            # （1）加载服务对应的性能评估器
            f = open(KB_DATA_PATH+'/'+service_info['name']+".json")  
            evaluator = json.load(f)
            f.close()
            # （2）获取service_info里对应的服务配置参数，从参数conf中获取该服务配置旋钮的需要的各个值，加上ip选择
            # 得到形如["360p"，"1","JPEG","114.212.81.11"]的conf_for_dict，用于从字典中获取性能指标
            service_conf = list(service_info['conf']) # 形如":["reso","fps","encoder"]
                # 对于得到的service_conf，还应该加上“ip”也就是卸载方式
            conf_for_dict = []
                #形如：["360p"，"1","JPEG"]
            for service_conf_name in service_conf:
                conf_for_dict.append(str(conf[service_conf_name]))   
            
            # 完成以上操作后，conf_for_dict内还差ip地址,首先要判断当前评估器是不是针对传输阶段进行的：
            ip_for_dict_index = service_info['name'].find("_trans") 
            if ip_for_dict_index > 0:
                # 当前是trans，则去除服务名末尾的_trans，形如“face_detection”
                ip_for_dict_name = service_info['name'][0:ip_for_dict_index] 
            else: # 当前不是trans
                ip_for_dict_name = service_info['name']
            ip_for_dict = flow_mapping[ip_for_dict_name]['node_ip']
                
            cpu_for_dict = resource_limit[ip_for_dict_name]["cpu_util_limit"]
            mem_for_dict = resource_limit[ip_for_dict_name]["mem_util_limit"]
            
            conf_for_dict.append(str(ip_for_dict))  
            conf_for_dict.append(str(cpu_for_dict)) 
            conf_for_dict.append(str(mem_for_dict)) 

            service_conf.append(service_info['name']+'_ip')
            service_conf.append(service_info['name']+'_cpu_util_limit')
            service_conf.append(service_info['name']+'_mem_util_limit')
            # 形如["360p"，"1","JPEG","114.212.81.11","0.1"."0.1"]

            # （3）根据conf_for_dict，从性能评估器中提取该服务的评估时延
     
            dict_key=''
            for i in range(0, len(service_conf)):
                dict_key += service_conf[i] + '=' + conf_for_dict[i] + ' '

            
            if dict_key not in evaluator:
                #print('配置不存在',dict_key)
                return status, pred_delay_list, pred_delay_total
            
            pred_delay = evaluator[dict_key]
            
            #  (4) 如果pred_delay为0，意味着这一部分对应的性能估计结果不存在，该配置在知识库中没有找到合适的解。此时直接返回结果。
            if pred_delay == 0:
                return status, pred_delay_list, pred_delay_total
            
            #  (5)对预测出时延根据工况进行修改
            #  注意，此处获得的pred_delay_list和pred_delay_total里的时延都是单一工况下的，因此需要结合工况进行调整
            obj_n = 1
            if 'obj_n' in self.work_condition:
                obj_n = self.work_condition['obj_n']
            if service_info_dict[service_info['name']]["vary_with_obj_n"]:
                pred_delay = pred_delay * obj_n

            # （6）将预测的时延添加到列表中
            pred_delay_list.append(pred_delay)

        # 计算总处理时延
        for pred_delay in pred_delay_list:
            pred_delay_total += pred_delay
        
        
        ############ 在当前带宽状态下预估传输时延 ############
        if not self.cold_start_flag:  # 非冷启动时才可预测传输时延，因为冷启动时没有工况、画像、视频帧信息，无法预估传输的数据量，也无法预估传输时延
            index = edge_cloud_cut_choice
            edge_to_cloud_data = 0
            while index < len(self.serv_names):
                edge_to_cloud_data += self.portrait_info['data_trans_size'][self.serv_names[index]]  # 以上次执行时各服务之间数据的传输作为当前传输的数据量
                index += 1
            
            if edge_cloud_cut_choice == 0:  # 此时需要将图像传到云端，图像的数据量很大，需要较精确的计算
                pre_frame_str_size = len(self.portrait_info['frame'])
                pre_frame = field_codec_utils.decode_image(self.portrait_info['frame'])
                temp_width = common.resolution_wh[conf['reso']]['w']
                temp_height = common.resolution_wh[conf['reso']]['h']
                temp_frame = cv2.resize(pre_frame, (temp_width, temp_height))
                temp_frame_str_size = len(field_codec_utils.encode_image(temp_frame))
                
                edge_to_cloud_data -= pre_frame_str_size
                edge_to_cloud_data += temp_frame_str_size
            
            trans_delay = edge_to_cloud_data / (self.bandwidth_dict['kB/s'] * 1000) * conf['fps'] / 30  # 与处理时延、总体时延相同，传输时延也需要考虑帧率的影响
            pred_delay_total += trans_delay
        
        
        status = 1
        return status, pred_delay_list, pred_delay_total  # 返回各个部分的时延
    

    # get_coldstart_plan_rotate：
    # 用途：遍历conf_list、ip、cpu和mem所有配置，选择一个能够满足约束的冷启动计划。
    # 方法：生成配置遍历的全排列，对每一种配置都调用get_pred_delay预测时延，选择较好的
    # 返回值：最好的conf, flow_mapping, resource_limit
    def get_coldstart_plan_rotate(self):
        
        #例如,可能的一组取值如下：
        # conf_names=["reso","fps","encoder"]
        # serv_names=["face_detection", "face_alignment"]
        # conf_list=[ ["360p", "480p", "720p", "1080p"],[1, 5, 10, 20, 30],["JPEG"]]
        # serv_ip_list=[["114.212.81.11","172.27.143.164","172.27.151.145"],["114.212.81.11","172.27.143.164","172.27.151.145"]]
        # serv_cpu_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]
        # serv_mem_list=[[0.3,0.2,0.1],[0.3,0.2,0.1]]
        
        # 获取时延约束
        delay_constraint = self.user_constraint["delay"]


        best_conf={}
        best_flow_mapping={}
        best_resource_limit={}
        best_pred_delay_list=[]
        best_pred_delay_total=-1

        min_conf={}
        min_flow_mapping={}
        min_resource_limit={}
        min_pred_delay_list=[]
        min_pred_delay_total=-1

        conf_combine=itertools.product(*self.conf_list)
        for conf_plan in conf_combine:
            serv_ip_combine=itertools.product(*self.serv_ip_list)
            for serv_ip_plan in serv_ip_combine:# 遍历所有配置和卸载策略组合
                serv_cpu_combind=itertools.product(*self.serv_cpu_list)
                for serv_cpu_plan in serv_cpu_combind: 
                    serv_mem_combind=itertools.product(*self.serv_mem_list)
                    for serv_mem_plan in serv_mem_combind: 
                        conf={}
                        flow_mapping={}
                        resource_limit={}
                        for i in range(0,len(conf_names)):
                            conf[conf_names[i]]=conf_plan[i]
                        for i in range(0,len(serv_names)):
                            flow_mapping[serv_names[i]]=model_op[serv_ip_plan[i]]
                            resource_limit[serv_names[i]]={}
                            resource_limit[serv_names[i]]["cpu_util_limit"]=serv_cpu_plan[i]
                            resource_limit[serv_names[i]]["mem_util_limit"]=serv_mem_plan[i]
                        # 右上，得到了conf，flow_mapping，以及resource_limit
                        status,pred_delay_list,pred_delay_total = self.get_pred_delay(conf=conf,
                                                                                flow_mapping=flow_mapping,
                                                                                resource_limit=resource_limit,
                                                                                )
                        if status == 0: #如果为0，意味着配置在字典中找不到对应的性能评估结果，知识库没有存储这种配置对应的估计结果
                            continue
                        if best_pred_delay_total<0:   #初始化最优配置和最小配置
                            best_conf=conf
                            best_flow_mapping=flow_mapping
                            best_resource_limit=resource_limit
                            best_pred_delay_list=pred_delay_list
                            best_pred_delay_total=pred_delay_total

                            min_conf=conf
                            min_flow_mapping=flow_mapping
                            min_resource_limit=resource_limit
                            min_pred_delay_list=pred_delay_list
                            min_pred_delay_total=pred_delay_total

                        elif pred_delay_total < delay_constraint*0.7 and pred_delay_total>best_pred_delay_total: #选出一个接近约束且比较大的
                            best_conf=conf
                            best_flow_mapping=flow_mapping
                            best_resource_limit=resource_limit
                            best_pred_delay_list=pred_delay_list
                            best_pred_delay_total=pred_delay_total
                        
                        elif pred_delay_total < best_pred_delay_total: #选出一个最小的
                            min_conf=conf
                            min_flow_mapping=flow_mapping
                            min_resource_limit=resource_limit
                            min_pred_delay_list=pred_delay_list
                            min_pred_delay_total=pred_delay_total
        
        # 完成遍历后，应该可以找到一个比较优秀的冷启动结果
        '''
        print("最优配置是：")
        print(best_conf)
        print(best_flow_mapping)
        print(best_resource_limit)
        print(best_pred_delay_list)
        print(best_pred_delay_total)
        print("最小配置是：")
        print(min_conf)
        print(min_flow_mapping)
        print(min_resource_limit)
        print(min_pred_delay_list)
        print(min_pred_delay_total)
        print("时延约束是",delay_constraint)
        '''       

        if min_pred_delay_total > delay_constraint:
            # print("约束过于严格，选择最小配置")
            return min_conf, min_flow_mapping, min_resource_limit
        else:
            # print("约束不算特别严格，选择最优策略")
            return best_conf, best_flow_mapping, best_resource_limit

    # 由trial给出下一组要尝试的参数并返回
    def get_next_params(self, trial):
        conf = {}
        flow_mapping = {}
        resource_limit = {}

        for conf_name in self.conf_names:   # conf_names含有流水线上所有任务需要的配置参数的总和(分辨率、帧率等)，但是不包括其所在的ip
            conf[conf_name] = trial.suggest_categorical(conf_name, conf_and_serv_info[conf_name])

        edge_cloud_cut_choice = trial.suggest_categorical('edge_cloud_cut_choice', conf_and_serv_info["edge_cloud_cut_point"])
        for serv_name in self.serv_names:
            with open(KB_DATA_PATH+'/'+ serv_name+'_conf_info'+'.json', 'r') as f:  
                conf_info = json.load(f)  
                if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                    flow_mapping[serv_name] = model_op[common.edge_ip]
                else:  # 服务索引大于等于云边协同切分点，在云执行
                    flow_mapping[serv_name] = model_op[common.cloud_ip]
                
                serv_cpu_limit = serv_name + "_cpu_util_limit"
                serv_mem_limit = serv_name + "_mem_util_limit"
                resource_limit[serv_name] = {}
                if flow_mapping[serv_name]["node_role"] == "cloud":  #对于云端没必要研究资源约束下的情况
                    resource_limit[serv_name]["cpu_util_limit"] = 1.0
                    resource_limit[serv_name]["mem_util_limit"] = 1.0
                    # 根据flow_mapping中的node_ip可以从rsc_constraint中获得该设备上的资源约束，通过serv_name可以获得rsc_upper_bound上的约束，
                    # 二者取最小的，用于限制从conf_info中获取的可取值范围，从而限制查找值。
                else:
                    device_ip = flow_mapping[serv_name]["node_ip"]
                    cpu_upper_limit = min(self.rsc_constraint[device_ip]['cpu'], self.rsc_upper_bound[serv_name]['cpu_limit'])
                    cpu_down_limit = max(0.0, self.rsc_down_bound[serv_name]['cpu_limit'])
                    mem_upper_limit = min(self.rsc_constraint[device_ip]['mem'], self.rsc_upper_bound[serv_name]['mem_limit'])
                    mem_down_limit = max(0.0, self.rsc_down_bound[serv_name]['mem_limit'])


                    cpu_choice_range = [item for item in conf_info[serv_cpu_limit] if item <= cpu_upper_limit and item >= cpu_down_limit]
                    mem_choice_range = [item for item in conf_info[serv_mem_limit] if item <= mem_upper_limit and item >= mem_down_limit]
                    # 要防止资源约束导致取值范围为空的情况
                    if len(cpu_choice_range) == 0:
                        cpu_choice_range = [item for item in conf_info[serv_cpu_limit]]
                    if len(mem_choice_range) == 0:
                        mem_choice_range = [item for item in conf_info[serv_mem_limit]]
                    resource_limit[serv_name]["cpu_util_limit"] = trial.suggest_categorical(serv_cpu_limit, cpu_choice_range)
                    resource_limit[serv_name]["mem_util_limit"] = trial.suggest_categorical(serv_mem_limit, mem_choice_range)

        return conf, flow_mapping, resource_limit
    
    # 由trial给出下一组要尝试的参数并返回
    def get_next_params_1(self, trial):
        conf = {}
        flow_mapping = {}
        resource_limit = {}

        for conf_name in self.conf_names:   # conf_names含有流水线上所有任务需要的配置参数的总和(分辨率、帧率等)，但是不包括其所在的ip
            conf[conf_name] = trial.suggest_categorical(conf_name, conf_and_serv_info[conf_name])

        edge_cloud_cut_choice = trial.suggest_categorical('edge_cloud_cut_choice', conf_and_serv_info["edge_cloud_cut_point"])
        for serv_name in self.serv_names:
            with open(KB_DATA_PATH+'/'+ serv_name+'_conf_info'+'.json', 'r') as f:  
                conf_info = json.load(f)  
                if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                    flow_mapping[serv_name] = model_op[common.edge_ip]
                else:  # 服务索引大于等于云边协同切分点，在云执行
                    flow_mapping[serv_name] = model_op[common.cloud_ip]
                
                serv_cpu_limit = serv_name + "_cpu_util_limit"
                serv_mem_limit = serv_name + "_mem_util_limit"
                resource_limit[serv_name] = {}
                if flow_mapping[serv_name]["node_role"] == "cloud":  #对于云端没必要研究资源约束下的情况
                    resource_limit[serv_name]["cpu_util_limit"] = 1.0
                    resource_limit[serv_name]["mem_util_limit"] = 1.0
                    # 根据flow_mapping中的node_ip可以从rsc_constraint中获得该设备上的资源约束，通过serv_name可以获得rsc_upper_bound上的约束，
                    # 二者取最小的，用于限制从conf_info中获取的可取值范围，从而限制查找值。
                else:
                    device_ip = flow_mapping[serv_name]["node_ip"]
                    cpu_upper_limit = min(self.rsc_constraint[device_ip]['cpu'], self.rsc_upper_bound[serv_name]['cpu_limit'])
                    cpu_down_limit = max(0.0, self.rsc_down_bound[serv_name]['cpu_limit'])
                    mem_upper_limit = min(self.rsc_constraint[device_ip]['mem'], self.rsc_upper_bound[serv_name]['mem_limit'])
                    mem_down_limit = max(0.0, self.rsc_down_bound[serv_name]['mem_limit'])


                    cpu_choice_range = []
                    for item in conf_info[serv_cpu_limit]:  # CPU资源取值在xxx_conf_info.json里
                        if item in conf_and_serv_info[serv_cpu_limit]:  # CPU资源取值在宏观调控限制的范围里
                            if item <= cpu_upper_limit and item >= cpu_down_limit:  # CPU资源取值在资源预估器预估的范围里
                                cpu_choice_range.append(item)  # 需要同时满足上述三个条件才可以作为贝叶斯优化查找的选项
                    
                    mem_choice_range = [1.0]
                    
                    # 要防止资源约束导致取值范围为空的情况s
                    if len(cpu_choice_range) == 0:
                        for item in conf_info[serv_cpu_limit]:  # CPU资源取值在xxx_conf_info.json里
                            if item in conf_and_serv_info[serv_cpu_limit]:  # CPU资源取值在宏观调控限制的范围里
                                cpu_choice_range.append(item)  # 需要同时满足上述三个条件才可以作为贝叶斯优化查找的选项
                    
                    if len(mem_choice_range) == 0:
                        mem_choice_range = [1.0]
                    assert len(cpu_choice_range) != 0 and len(mem_choice_range) != 0
                    resource_limit[serv_name]["cpu_util_limit"] = trial.suggest_categorical(serv_cpu_limit, cpu_choice_range)
                    resource_limit[serv_name]["mem_util_limit"] = trial.suggest_categorical(serv_mem_limit, mem_choice_range)

        return conf, flow_mapping, resource_limit
    
    
    # 该objective是一个动态多目标优化过程，能够求出帕累托最优解（时延和资源）
    def objective(self, trial):
        conf = {}
        flow_mapping = {}
        resource_limit = {}

        for conf_name in self.conf_names:   # conf_names含有流水线上所有任务需要的配置参数的总和(分辨率、帧率等)，但是不包括其所在的ip
            conf[conf_name] = trial.suggest_categorical(conf_name, conf_and_serv_info[conf_name])

        edge_cloud_cut_choice = trial.suggest_categorical('edge_cloud_cut_choice', conf_and_serv_info["edge_cloud_cut_point"])
        for serv_name in self.serv_names:
            with open(KB_DATA_PATH+'/'+ serv_name+'_conf_info'+'.json', 'r') as f:  
                conf_info = json.load(f)  
                if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                    flow_mapping[serv_name] = model_op[common.edge_ip]
                else:  # 服务索引大于等于云边协同切分点，在云执行
                    flow_mapping[serv_name] = model_op[common.cloud_ip]
                
                serv_cpu_limit = serv_name + "_cpu_util_limit"
                serv_mem_limit = serv_name + "_mem_util_limit"
                resource_limit[serv_name] = {}
                if flow_mapping[serv_name]["node_role"] == "cloud":  #对于云端没必要研究资源约束下的情况
                    resource_limit[serv_name]["cpu_util_limit"] = 1.0
                    resource_limit[serv_name]["mem_util_limit"] = 1.0
                    # 根据flow_mapping中的node_ip可以从rsc_constraint中获得该设备上的资源约束，通过serv_name可以获得rsc_upper_bound上的约束，
                    # 二者取最小的，用于限制从conf_info中获取的可取值范围，从而限制查找值。
                else:
                    device_ip = flow_mapping[serv_name]["node_ip"]
                    cpu_upper_limit = min(self.rsc_constraint[device_ip]['cpu'], self.rsc_upper_bound[serv_name]['cpu_limit'])
                    cpu_down_limit = max(0.0, self.rsc_down_bound[serv_name]['cpu_limit'])
                    mem_upper_limit = min(self.rsc_constraint[device_ip]['mem'], self.rsc_upper_bound[serv_name]['mem_limit'])
                    mem_down_limit = max(0.0, self.rsc_down_bound[serv_name]['mem_limit'])


                    cpu_choice_range = [item for item in conf_info[serv_cpu_limit] if item <= cpu_upper_limit and item >= cpu_down_limit]
                    mem_choice_range = [item for item in conf_info[serv_mem_limit] if item <= mem_upper_limit and item >= mem_down_limit]
                    # 要防止资源约束导致取值范围为空的情况
                    if len(cpu_choice_range) == 0:
                        cpu_choice_range = [item for item in conf_info[serv_cpu_limit]]
                    if len(mem_choice_range) == 0:
                        mem_choice_range = [item for item in conf_info[serv_mem_limit]]
                    resource_limit[serv_name]["cpu_util_limit"] = trial.suggest_categorical(serv_cpu_limit, cpu_choice_range)
                    resource_limit[serv_name]["mem_util_limit"] = trial.suggest_categorical(serv_mem_limit, mem_choice_range)
       
        
        mul_objects = []
        # 这是一个多目标优化问题，所以既要满足时延约束，又要满足资源约束。以下是获取时延约束的情况，要让最后的结果尽可能小。首先加入时延优化目标：
        # 以下返回的结果已经经过了工况处理，按照线性化乘以了obj_n。在不考虑传输时延的时候，结果是比较准确不会波动的。
        status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf, flow_mapping=flow_mapping, resource_limit=resource_limit, edge_cloud_cut_choice=edge_cloud_cut_choice)

        # 对于conf, flow_mapping=flow_mapping, resource_limit=resource_limit，还要求精度
        # 首先判断当前服务是都有精度
        task_accuracy = 1.0
        acc_pre = AccuracyPrediction()
        obj_size = None if 'obj_size' not in self.work_condition else self.work_condition['obj_size']
        obj_speed = None if 'obj_speed' not in self.work_condition else self.work_condition['obj_speed']
        for serv_name in serv_names:
            if service_info_dict[serv_name]["can_seek_accuracy"]:
                task_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                    'fps':conf['fps'],
                    'reso':conf['reso']
                }, obj_size=obj_size, obj_speed=obj_speed)

        # mul_objects.append(0.1*(1.0-task_accuracy))
        
        if status == 0:  # 返回0说明相应配置压根不存在，此时返回MAX_NUMBER。贝叶斯优化的目标是让返回值尽可能小，这种MAX_NUMBER的情况自然会被尽量避免
            mul_objects.append(MAX_NUMBER)
        else:  #如果成功找到了一个可行的策略，按照如下方式计算返回值，目的是得到尽可能靠近约束时延0.7倍且小于约束时延的配置
            delay_constraint = self.user_constraint["delay"]
            if pred_delay_total <= 0.85*delay_constraint:
                mul_objects.append(delay_constraint - pred_delay_total)
            elif pred_delay_total > 0.85*delay_constraint:
                mul_objects.append(0.15*delay_constraint + pred_delay_total)
        
        # 然后加入各个设备上的优化目标：我需要获取每一个设备上的cpu约束，以及mem约束。
        
        for device_ip in self.rsc_constraint.keys():
            cpu_util = 0
            mem_util = 0
            # 检查该device_ip上消耗了多少资源
            for serv_name in resource_limit.keys():
                if flow_mapping[serv_name]['node_ip'] == device_ip:
                    cpu_util += resource_limit[serv_name]['cpu_util_limit']
                    mem_util += resource_limit[serv_name]['mem_util_limit']
            
            # 如果cpu_util和mem_util取值都为0，说明用户根本没有在这个ip上消耗资源，此时显然是满足优化目标的，直接返回两个0即可
            if cpu_util == 0 and mem_util == 0:
                mul_objects.append(0)
                mul_objects.append(0)
            
            # 否则，说明在这个ip上消耗资源了。
            # 此时，如果是在云端，资源消耗根本不算什么,但是应该尽可能不选择云端才对。所以此处应该返回相对较大的结果。
            # 那么应该返回多少比较合适呢？先设置为1看看。
            elif model_op[device_ip]['node_role'] == 'cloud':
                mul_objects.append(1.0)
                mul_objects.append(1.0)

            # 只有在ip为边端且消耗了资源的情况下，才需要尽可能让资源最小化。这里的0.5是为了减小资源约束相对于时延的占比权重
            else: 
                mul_objects.append(0.5*cpu_util)
                mul_objects.append(0.5*mem_util)
                
                

        # 返回的总优化目标数量应该有：1+2*ips个
        return mul_objects


    # 该objective是一个动态多目标优化过程，能够求出帕累托最优解（精度和资源）
    def objective_1(self, trial):
        conf = {}
        flow_mapping = {}
        resource_limit = {}

        for conf_name in self.conf_names:   # conf_names含有流水线上所有任务需要的配置参数的总和(分辨率、帧率等)，但是不包括其所在的ip
            conf[conf_name] = trial.suggest_categorical(conf_name, conf_and_serv_info[conf_name])

        edge_cloud_cut_choice = trial.suggest_categorical('edge_cloud_cut_choice', conf_and_serv_info["edge_cloud_cut_point"])
        for serv_name in self.serv_names:
            with open(KB_DATA_PATH+'/'+ serv_name+'_conf_info'+'.json', 'r') as f:  
                conf_info = json.load(f)  
                if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                    flow_mapping[serv_name] = model_op[common.edge_ip]
                else:  # 服务索引大于等于云边协同切分点，在云执行
                    flow_mapping[serv_name] = model_op[common.cloud_ip]
                
                serv_cpu_limit = serv_name + "_cpu_util_limit"
                serv_mem_limit = serv_name + "_mem_util_limit"
                resource_limit[serv_name] = {}
                if flow_mapping[serv_name]["node_role"] == "cloud":  #对于云端没必要研究资源约束下的情况
                    resource_limit[serv_name]["cpu_util_limit"] = 1.0
                    resource_limit[serv_name]["mem_util_limit"] = 1.0
                    # 根据flow_mapping中的node_ip可以从rsc_constraint中获得该设备上的资源约束，通过serv_name可以获得rsc_upper_bound上的约束，
                    # 二者取最小的，用于限制从conf_info中获取的可取值范围，从而限制查找值。
                else:
                    device_ip = flow_mapping[serv_name]["node_ip"]
                    cpu_upper_limit = min(self.rsc_constraint[device_ip]['cpu'], self.rsc_upper_bound[serv_name]['cpu_limit'])
                    cpu_down_limit = max(0.0, self.rsc_down_bound[serv_name]['cpu_limit'])
                    mem_upper_limit = min(self.rsc_constraint[device_ip]['mem'], self.rsc_upper_bound[serv_name]['mem_limit'])
                    mem_down_limit = max(0.0, self.rsc_down_bound[serv_name]['mem_limit'])

                    cpu_choice_range = []
                    for item in conf_info[serv_cpu_limit]:  # CPU资源取值在xxx_conf_info.json里
                        if item in conf_and_serv_info[serv_cpu_limit]:  # CPU资源取值在宏观调控限制的范围里
                            if item <= cpu_upper_limit and item >= cpu_down_limit:  # CPU资源取值在资源预估器预估的范围里
                                cpu_choice_range.append(item)  # 需要同时满足上述三个条件才可以作为贝叶斯优化查找的选项
                    
                    mem_choice_range = [1.0]  # 不再更改内存分配量，统一设置为1.0
                    
                    # 要防止资源约束导致取值范围为空的情况
                    if len(cpu_choice_range) == 0:
                        for item in conf_info[serv_cpu_limit]:  # CPU资源取值在xxx_conf_info.json里
                            if item in conf_and_serv_info[serv_cpu_limit]:  # CPU资源取值在宏观调控限制的范围里
                                cpu_choice_range.append(item)  # 需要同时满足上述三个条件才可以作为贝叶斯优化查找的选项
                    
                    if len(mem_choice_range) == 0:
                        mem_choice_range = [1.0]
                    assert len(cpu_choice_range) != 0 and len(mem_choice_range) != 0
                    resource_limit[serv_name]["cpu_util_limit"] = trial.suggest_categorical(serv_cpu_limit, cpu_choice_range)
                    resource_limit[serv_name]["mem_util_limit"] = trial.suggest_categorical(serv_mem_limit, mem_choice_range)
       
        
        mul_objects = []
        # 这是一个多目标优化问题，所以既要满足时延约束，又要满足资源约束。以下是获取时延约束的情况，要让最后的结果尽可能小。首先加入时延优化目标：
        # 以下返回的结果已经经过了工况处理，按照线性化乘以了obj_n。在不考虑传输时延的时候，结果是比较准确不会波动的。
        status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf, flow_mapping=flow_mapping, resource_limit=resource_limit, edge_cloud_cut_choice=edge_cloud_cut_choice)

        # 对于conf, flow_mapping=flow_mapping, resource_limit=resource_limit，还要求精度
        # 首先判断当前服务是都有精度
        task_accuracy = 1.0
        acc_pre = AccuracyPrediction()
        obj_size = None if 'obj_size' not in self.work_condition else self.work_condition['obj_size']
        obj_speed = None if 'obj_speed' not in self.work_condition else self.work_condition['obj_speed']
        for serv_name in serv_names:
            if service_info_dict[serv_name]["can_seek_accuracy"]:
                task_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                    'fps':conf['fps'],
                    'reso':conf['reso']
                }, obj_size=obj_size, obj_speed=obj_speed)

        acc_constraint = self.user_constraint["accuracy"]
        if task_accuracy >= acc_constraint:
            mul_objects.append(task_accuracy - acc_constraint)
        else:
            mul_objects.append(10*(acc_constraint - task_accuracy))
        
        if status == 0:  # 返回0说明相应配置压根不存在，此时返回MAX_NUMBER。贝叶斯优化的目标是让返回值尽可能小，这种MAX_NUMBER的情况自然会被尽量避免
            mul_objects.append(MAX_NUMBER)
        else:  
            mul_objects.append(pred_delay_total)
        
        # 然后加入各个设备上的优化目标：我需要获取每一个设备上的cpu约束，以及mem约束。
        
        for device_ip in self.rsc_constraint.keys():
            cpu_util = 0
            mem_util = 0
            # 检查该device_ip上消耗了多少资源
            for serv_name in resource_limit.keys():
                if flow_mapping[serv_name]['node_ip'] == device_ip:
                    cpu_util += resource_limit[serv_name]['cpu_util_limit']
                    mem_util += resource_limit[serv_name]['mem_util_limit']
            
            # 如果cpu_util和mem_util取值都为0，说明用户根本没有在这个ip上消耗资源，此时显然是满足优化目标的，直接返回两个0即可
            if cpu_util == 0 and mem_util == 0:
                mul_objects.append(0)
                mul_objects.append(0)
            
            # 否则，说明在这个ip上消耗资源了。
            # 此时，如果是在云端，资源消耗根本不算什么,但是应该尽可能不选择云端才对。所以此处应该返回相对较大的结果。
            # 那么应该返回多少比较合适呢？先设置为1看看。
            elif model_op[device_ip]['node_role'] == 'cloud':
                mul_objects.append(1.0)
                mul_objects.append(1.0)

            # 只有在ip为边端且消耗了资源的情况下，才需要尽可能让资源最小化。这里的0.5是为了减小资源约束相对于时延的占比权重
            else: 
                mul_objects.append(0.5*cpu_util)
                mul_objects.append(0.5*mem_util)
                
        # 返回的总优化目标数量应该有：2+2*ips个
        return mul_objects

        
    # get_plan_in_cons：
    # 用途：基于贝叶斯优化，调用多目标优化的贝叶斯模型，得到一系列帕累托最优解
    # 方法：通过贝叶斯优化，在有限轮数内选择一个最优的结果
    # 返回值：满足时延和资源约束的解；满足时延约束不满足资源约束的解；不满足时延约束的解；贝叶斯优化模型建议的下一组参数值
    def get_plan_in_cons(self, n_trials):
        # 开始多目标优化，由于时延和资源为优化问题的约束条件，因此将其作为多目标优化的目标
        study = optuna.create_study(directions=['minimize' for _ in range(1+2*len(self.rsc_constraint.keys()))])  
        study.optimize(self.objective, n_trials=n_trials)

        ans_params = []
        trials = sorted(study.best_trials, key=lambda t: t.values) # 对字典best_trials按values从小到达排序  
        
        # 此处的帕累托最优解可能重复，因此首先要提取内容，将其转化为字符串记录在ans_params中，然后从ans_params里删除重复项
        for trial in trials:
            conf = {}
            flow_mapping = {}
            resource_limit = {}
            for conf_name in self.conf_names:   # conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
                # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
                conf[conf_name] = trial.params[conf_name]
            edge_cloud_cut_choice = trial.params['edge_cloud_cut_choice']
            
            for serv_name in self.serv_names:
                if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                    flow_mapping[serv_name] = model_op[common.edge_ip]
                else:  # 服务索引大于等于云边协同切分点，在云执行
                    flow_mapping[serv_name] = model_op[common.cloud_ip]

                serv_cpu_limit = serv_name + "_cpu_util_limit"
                serv_mem_limit = serv_name + "_mem_util_limit"
                resource_limit[serv_name] = {}

                if flow_mapping[serv_name]["node_role"] == "cloud":
                    resource_limit[serv_name]["cpu_util_limit"] = 1.0
                    resource_limit[serv_name]["mem_util_limit"] = 1.0
                else:
                    resource_limit[serv_name]["cpu_util_limit"] = trial.params[serv_cpu_limit]
                    resource_limit[serv_name]["mem_util_limit"] = trial.params[serv_mem_limit]
            
            ans_item = {}
            ans_item['conf'] = conf
            ans_item['flow_mapping'] = flow_mapping
            ans_item['resource_limit'] = resource_limit
            ans_item['edge_cloud_cut_choice'] = edge_cloud_cut_choice
            
            ans_params.append(json.dumps(ans_item))
            '''
            {   'reso': '720p', 'fps': 10, 'encoder': 'JPEG'}
            {   'face_detection': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}, 
                'face_alignment': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}}
            {   'face_detection': {'cpu_util_limit': 0.2, 'mem_util_limit': 0.45}, 
                'face_alignment': {'cpu_util_limit': 0.1, 'mem_util_limit': 0.45}}
            '''

        # 从ans_params里删除重复项，并选择真正status不为0的有效帕累托最优解返回。同时存储该配置对应的预估时延
        ans_params_set = list(set(ans_params))
        # 保存满足时延约束的解
        params_in_delay_in_rsc_cons = []
        params_in_delay_out_rsc_cons = []
        params_out_delay_cons = []
        for param in ans_params_set:
            ans_dict = json.loads(param)
            conf = ans_dict['conf']
            flow_mapping = ans_dict['flow_mapping']
            resource_limit = ans_dict['resource_limit']
            edge_cloud_cut_choice = ans_dict['edge_cloud_cut_choice']
            status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf,
                                                                            flow_mapping=flow_mapping,
                                                                            resource_limit=resource_limit,
                                                                            edge_cloud_cut_choice=edge_cloud_cut_choice)
            
            # 求出精度
            task_accuracy = 1.0
            acc_pre = AccuracyPrediction()
            obj_size = None if 'obj_size' not in self.work_condition else self.work_condition['obj_size']
            obj_speed = None if 'obj_speed' not in self.work_condition else self.work_condition['obj_speed']
            for serv_name in serv_names:
                if service_info_dict[serv_name]["can_seek_accuracy"]:
                    task_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                        'fps':conf['fps'],
                        'reso':conf['reso']
                    }, obj_size=obj_size, obj_speed=obj_speed)
                    
            if status != 0:  # 如果status不为0，才说明这个配置是有效的，否则是无效的
                deg_violate = 0 #违反资源约束的程度
                for device_ip in self.rsc_constraint.keys():
                    # 只针对非云设备计算违反资源约束程度
                    if model_op[device_ip]['node_role'] != 'cloud':
                        cpu_util = 0
                        mem_util = 0
                        for serv_name in resource_limit.keys():
                            if flow_mapping[serv_name]['node_ip'] == device_ip:
                                # 必须用round保留小数点，因为python对待浮点数不精确，0.35加上0.05会得到0.39999……
                                cpu_util = round(cpu_util + resource_limit[serv_name]['cpu_util_limit'], 2)
                                mem_util = round(mem_util + resource_limit[serv_name]['mem_util_limit'], 2)
                        cpu_util_ratio = float(cpu_util) / float(self.rsc_constraint[device_ip]['cpu'])
                        mem_util_ratio = float(mem_util) / float(self.rsc_constraint[device_ip]['mem'])
                        
                        if cpu_util_ratio > 1:
                            deg_violate += cpu_util_ratio
                        if mem_util_ratio > 1:
                            deg_violate += mem_util_ratio
                    
                ans_dict['pred_delay_list'] = pred_delay_list
                ans_dict['pred_delay_total'] = pred_delay_total
                ans_dict['deg_violate'] = deg_violate
                ans_dict['task_accuracy'] = task_accuracy

                # 根据配置是否满足时延约束，将其分为两类
                if pred_delay_total <= 0.95*self.user_constraint["delay"] and ans_dict['deg_violate'] == 0:
                    params_in_delay_in_rsc_cons.append(ans_dict)
                elif pred_delay_total <= 0.95*self.user_constraint["delay"] and ans_dict['deg_violate'] > 0:
                    params_in_delay_out_rsc_cons.append(ans_dict)
                else:
                    params_out_delay_cons.append(ans_dict)
        
        # 构建贝叶斯优化模型建议的下一组参数值
        next_trial = study.ask()
        next_conf, next_flow_mapping, next_resource_limit = self.get_next_params(next_trial)
        next_plan = {
            'conf': next_conf,
            'flow_mapping': next_flow_mapping,
            'resource_limit': next_resource_limit
        }
              
        return params_in_delay_in_rsc_cons, params_in_delay_out_rsc_cons, params_out_delay_cons, next_plan
    
    
    # get_plan_in_cons_1：
    # 用途：基于贝叶斯优化，调用多目标优化的贝叶斯模型，得到一系列帕累托最优解
    # 方法：通过贝叶斯优化，在有限轮数内选择一个最优的结果
    # 返回值：满足精度和资源约束的解；满足精度约束不满足资源约束的解；不满足精度约束的解；贝叶斯优化模型建议的下一组参数值
    def get_plan_in_cons_1(self, n_trials):
        # 开始多目标优化，由于时延和资源为优化问题的约束条件，因此将其作为多目标优化的目标
        study = optuna.create_study(directions=['minimize' for _ in range(2+2*len(self.rsc_constraint.keys()))])  
        study.optimize(self.objective_1, n_trials=n_trials)

        ans_params = []
        trials = sorted(study.best_trials, key=lambda t: t.values) # 对字典best_trials按values从小到达排序  
        
        # 此处的帕累托最优解可能重复，因此首先要提取内容，将其转化为字符串记录在ans_params中，然后从ans_params里删除重复项
        for trial in trials:
            conf = {}
            flow_mapping = {}
            resource_limit = {}
            for conf_name in self.conf_names:   # conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
                # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
                conf[conf_name] = trial.params[conf_name]
            edge_cloud_cut_choice = trial.params['edge_cloud_cut_choice']
            
            for serv_name in self.serv_names:
                if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                    flow_mapping[serv_name] = model_op[common.edge_ip]
                else:  # 服务索引大于等于云边协同切分点，在云执行
                    flow_mapping[serv_name] = model_op[common.cloud_ip]

                serv_cpu_limit = serv_name + "_cpu_util_limit"
                serv_mem_limit = serv_name + "_mem_util_limit"
                resource_limit[serv_name] = {}

                if flow_mapping[serv_name]["node_role"] == "cloud":
                    resource_limit[serv_name]["cpu_util_limit"] = 1.0
                    resource_limit[serv_name]["mem_util_limit"] = 1.0
                else:
                    resource_limit[serv_name]["cpu_util_limit"] = trial.params[serv_cpu_limit]
                    resource_limit[serv_name]["mem_util_limit"] = trial.params[serv_mem_limit]
            
            ans_item = {}
            ans_item['conf'] = conf
            ans_item['flow_mapping'] = flow_mapping
            ans_item['resource_limit'] = resource_limit
            ans_item['edge_cloud_cut_choice'] = edge_cloud_cut_choice
            
            ans_params.append(json.dumps(ans_item))
            '''
            {   'reso': '720p', 'fps': 10, 'encoder': 'JPEG'}
            {   'face_detection': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}, 
                'face_alignment': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}}
            {   'face_detection': {'cpu_util_limit': 0.2, 'mem_util_limit': 0.45}, 
                'face_alignment': {'cpu_util_limit': 0.1, 'mem_util_limit': 0.45}}
            '''

        # 从ans_params里删除重复项，并选择真正status不为0的有效帕累托最优解返回。同时存储该配置对应的预估时延
        ans_params_set = list(set(ans_params))
        # 保存满足时延约束的解
        params_in_acc_in_rsc_cons = []
        params_in_acc_out_rsc_cons = []
        params_out_acc_cons = []
        for param in ans_params_set:
            ans_dict = json.loads(param)
            conf = ans_dict['conf']
            flow_mapping = ans_dict['flow_mapping']
            resource_limit = ans_dict['resource_limit']
            edge_cloud_cut_choice = ans_dict['edge_cloud_cut_choice']
            status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf,
                                                                            flow_mapping=flow_mapping,
                                                                            resource_limit=resource_limit,
                                                                            edge_cloud_cut_choice=edge_cloud_cut_choice)
            
            # 求出精度
            task_accuracy = 1.0
            acc_pre = AccuracyPrediction()
            obj_size = None if 'obj_size' not in self.work_condition else self.work_condition['obj_size']
            obj_speed = None if 'obj_speed' not in self.work_condition else self.work_condition['obj_speed']
            for serv_name in serv_names:
                if service_info_dict[serv_name]["can_seek_accuracy"]:
                    task_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                        'fps':conf['fps'],
                        'reso':conf['reso']
                    }, obj_size=obj_size, obj_speed=obj_speed)
                    
            if status != 0:  # 如果status不为0，才说明这个配置是有效的，否则是无效的
                deg_violate = 0 #违反资源约束的程度
                for device_ip in self.rsc_constraint.keys():
                    # 只针对非云设备计算违反资源约束程度
                    if model_op[device_ip]['node_role'] != 'cloud':
                        cpu_util = 0
                        mem_util = 0
                        for serv_name in resource_limit.keys():
                            if flow_mapping[serv_name]['node_ip'] == device_ip:
                                # 必须用round保留小数点，因为python对待浮点数不精确，0.35加上0.05会得到0.39999……
                                cpu_util = round(cpu_util + resource_limit[serv_name]['cpu_util_limit'], 2)
                                mem_util = round(mem_util + resource_limit[serv_name]['mem_util_limit'], 2)
                        cpu_util_ratio = float(cpu_util) / float(self.rsc_constraint[device_ip]['cpu'])
                        mem_util_ratio = float(mem_util) / float(self.rsc_constraint[device_ip]['mem'])
                        
                        if cpu_util_ratio > 1:
                            deg_violate += cpu_util_ratio
                        if mem_util_ratio > 1:
                            deg_violate += mem_util_ratio
                    
                ans_dict['pred_delay_list'] = pred_delay_list
                ans_dict['pred_delay_total'] = pred_delay_total
                ans_dict['deg_violate'] = deg_violate
                ans_dict['task_accuracy'] = task_accuracy

                # 根据配置是否满足精度约束，将其分为三类
                if task_accuracy >= self.user_constraint["accuracy"] and ans_dict['deg_violate'] == 0:
                    params_in_acc_in_rsc_cons.append(ans_dict)
                elif task_accuracy >= self.user_constraint["accuracy"] and ans_dict['deg_violate'] > 0:
                    params_in_acc_out_rsc_cons.append(ans_dict)
                else:
                    params_out_acc_cons.append(ans_dict)
        
        # 构建贝叶斯优化模型建议的下一组参数值
        next_trial = study.ask()
        next_conf, next_flow_mapping, next_resource_limit = self.get_next_params_1(next_trial)
        next_plan = {
            'conf': next_conf,
            'flow_mapping': next_flow_mapping,
            'resource_limit': next_resource_limit
        }
              
        return params_in_acc_in_rsc_cons, params_in_acc_out_rsc_cons, params_out_acc_cons, next_plan
    
    
    # get_plan_in_cons_2：
    # 其用途、方法、返回值与get_plan_in_cons_1均一致，但是贝叶斯优化查找的过程以较小的粒度为单位进行（例如20次），而不是直接进行大量查找，用于减少不必要的查找，提高查找速度
    def get_plan_in_cons_2(self, n_trials):
        # 开始多目标优化，由于精度和资源为优化问题的约束条件，因此将其作为多目标优化的目标
        study = optuna.create_study(directions=['minimize' for _ in range(2+2*len(self.rsc_constraint.keys()))])  
        count = 0
        
        # 保存满足约束的解
        ans_params = []
        params_in_acc_in_rsc_cons = []
        params_in_acc_out_rsc_cons = []
        params_out_acc_cons = []
        
        prev_in_acc_cons_minimun_delay = -1  # 前一轮查找过程中满足精度约束的最低时延
    
        
        while count < n_trials:  # 以self.n_trial_unit为单位进行贝叶斯优化查找，最多进行n_trials次查找
            study.optimize(self.objective_1, n_trials=self.n_trial_unit)
            
            ########## 1. 整理本轮贝叶斯优化查找的结果 ##########
            cur_in_acc_cons_minimun_delay = sys.maxsize  # 本轮查找过程中满足精度约束的最低时延
            temp_ans_params = []
            trials = sorted(study.best_trials, key=lambda t: t.values) # 对字典best_trials按values从小到达排序  
            
            # 此处的帕累托最优解可能重复，因此首先要提取内容，将其转化为字符串记录在ans_params中，然后从ans_params里删除重复项
            for trial in trials:
                conf = {}
                flow_mapping = {}
                resource_limit = {}
                for conf_name in self.conf_names:   # conf_names含有流水线上所有任务需要的配置参数的总和，但是不包括其所在的ip
                    # conf_list会包含各类配置参数取值范围,例如分辨率、帧率等
                    conf[conf_name] = trial.params[conf_name]
                edge_cloud_cut_choice = trial.params['edge_cloud_cut_choice']
                
                for serv_name in self.serv_names:
                    if self.serv_names.index(serv_name) < edge_cloud_cut_choice:  # 服务索引小于云边协同切分点，在边执行
                        flow_mapping[serv_name] = model_op[common.edge_ip]
                    else:  # 服务索引大于等于云边协同切分点，在云执行
                        flow_mapping[serv_name] = model_op[common.cloud_ip]

                    serv_cpu_limit = serv_name + "_cpu_util_limit"
                    serv_mem_limit = serv_name + "_mem_util_limit"
                    resource_limit[serv_name] = {}

                    if flow_mapping[serv_name]["node_role"] == "cloud":
                        resource_limit[serv_name]["cpu_util_limit"] = 1.0
                        resource_limit[serv_name]["mem_util_limit"] = 1.0
                    else:
                        resource_limit[serv_name]["cpu_util_limit"] = trial.params[serv_cpu_limit]
                        resource_limit[serv_name]["mem_util_limit"] = trial.params[serv_mem_limit]
                
                ans_item = {}
                ans_item['conf'] = conf
                ans_item['flow_mapping'] = flow_mapping
                ans_item['resource_limit'] = resource_limit
                ans_item['edge_cloud_cut_choice'] = edge_cloud_cut_choice
                
                temp_ans_params.append(json.dumps(ans_item))
                '''
                {   'reso': '720p', 'fps': 10, 'encoder': 'JPEG'}
                {   'face_detection': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}, 
                    'face_alignment': {'model_id': 0, 'node_ip': '172.27.151.145', 'node_role': 'host'}}
                {   'face_detection': {'cpu_util_limit': 0.2, 'mem_util_limit': 0.45}, 
                    'face_alignment': {'cpu_util_limit': 0.1, 'mem_util_limit': 0.45}}
                '''

            # 从temp_ans_params里删除重复项，相当于对本轮的查找去重
            temp_ans_params_set = list(set(temp_ans_params))
            for param in temp_ans_params_set:
                if param not in ans_params:  #  全局去重，本轮未重复不代表全局未重复
                    ans_params.append(param)
                    
                    ans_dict = json.loads(param)
                    conf = ans_dict['conf']
                    flow_mapping = ans_dict['flow_mapping']
                    resource_limit = ans_dict['resource_limit']
                    edge_cloud_cut_choice = ans_dict['edge_cloud_cut_choice']
                    status, pred_delay_list, pred_delay_total = self.get_pred_delay(conf=conf,
                                                                                    flow_mapping=flow_mapping,
                                                                                    resource_limit=resource_limit,
                                                                                    edge_cloud_cut_choice=edge_cloud_cut_choice)
                    
                    # 求出精度
                    task_accuracy = 1.0
                    acc_pre = AccuracyPrediction()
                    obj_size = None if 'obj_size' not in self.work_condition else self.work_condition['obj_size']
                    obj_speed = None if 'obj_speed' not in self.work_condition else self.work_condition['obj_speed']
                    for serv_name in serv_names:
                        if service_info_dict[serv_name]["can_seek_accuracy"]:
                            task_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                                'fps':conf['fps'],
                                'reso':conf['reso']
                            }, obj_size=obj_size, obj_speed=obj_speed)
                            
                    if status != 0:  # 如果status不为0，才说明这个配置是有效的，否则是无效的
                        deg_violate = 0 #违反资源约束的程度
                        for device_ip in self.rsc_constraint.keys():
                            # 只针对非云设备计算违反资源约束程度
                            if model_op[device_ip]['node_role'] != 'cloud':
                                cpu_util = 0
                                mem_util = 0
                                for serv_name in resource_limit.keys():
                                    if flow_mapping[serv_name]['node_ip'] == device_ip:
                                        # 必须用round保留小数点，因为python对待浮点数不精确，0.35加上0.05会得到0.39999……
                                        cpu_util = round(cpu_util + resource_limit[serv_name]['cpu_util_limit'], 2)
                                        mem_util = round(mem_util + resource_limit[serv_name]['mem_util_limit'], 2)
                                cpu_util_ratio = float(cpu_util) / float(self.rsc_constraint[device_ip]['cpu'])
                                mem_util_ratio = float(mem_util) / float(self.rsc_constraint[device_ip]['mem'])
                                
                                if cpu_util_ratio > 1:
                                    deg_violate += cpu_util_ratio
                                if mem_util_ratio > 1:
                                    deg_violate += mem_util_ratio
                            
                        ans_dict['pred_delay_list'] = pred_delay_list
                        ans_dict['pred_delay_total'] = pred_delay_total
                        ans_dict['deg_violate'] = deg_violate
                        ans_dict['task_accuracy'] = task_accuracy

                        # 根据配置是否满足精度约束，将其分为三类
                        if task_accuracy >= self.user_constraint["accuracy"] and ans_dict['deg_violate'] == 0:
                            params_in_acc_in_rsc_cons.append(ans_dict)
                            cur_in_acc_cons_minimun_delay = min(cur_in_acc_cons_minimun_delay, pred_delay_total)
                        elif task_accuracy >= self.user_constraint["accuracy"] and ans_dict['deg_violate'] > 0:
                            params_in_acc_out_rsc_cons.append(ans_dict)
                        else:
                            params_out_acc_cons.append(ans_dict)
                
            ########## 2. 判断本轮查找与上一轮查找相比是否出现了饱和 ##########
            if prev_in_acc_cons_minimun_delay == -1:  # 目前还未获得合理的结果
                if cur_in_acc_cons_minimun_delay != sys.maxsize:  # 本轮获得了合理的结果，进行赋值
                    prev_in_acc_cons_minimun_delay = cur_in_acc_cons_minimun_delay
            elif cur_in_acc_cons_minimun_delay != sys.maxsize:
                if cur_in_acc_cons_minimun_delay >= 0.95*prev_in_acc_cons_minimun_delay:
                    break
                else:
                    prev_in_acc_cons_minimun_delay = cur_in_acc_cons_minimun_delay
            else:
                break
            
            count += self.n_trial_unit
        
        # 构建贝叶斯优化模型建议的下一组参数值
        next_trial = study.ask()
        next_conf, next_flow_mapping, next_resource_limit = self.get_next_params_1(next_trial)
        next_plan = {
            'conf': next_conf,
            'flow_mapping': next_flow_mapping,
            'resource_limit': next_resource_limit
        }
        
        return params_in_acc_in_rsc_cons, params_in_acc_out_rsc_cons, params_out_acc_cons, next_plan
        

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


# 下图的conf_names表示流水线上所有服务的conf的总和。
conf_names=["reso","fps","encoder"]

#这里包含流水线里涉及的各个服务的名称
serv_names=["face_detection","gender_classification"]   


user_constraint={
    "delay": 1.0,  #用户约束暂时设置为0.3
    "accuracy": 0.7
}

work_condition={
    "obj_n": 1,
    "obj_stable": True,
    "obj_size": 300,
    "delay": 0.2  #这玩意包含了传输时延，我不想看
}



if __name__ == "__main__":

    
    from RuntimePortrait import RuntimePortrait
    myportrait=RuntimePortrait(pipeline=serv_names)
    rsc_upper_bound={}
    for serv_name in serv_names:
        serv_rsc_cons=myportrait.help_cold_start(service=serv_name)
        rsc_upper_bound[serv_name]={}
        rsc_upper_bound[serv_name]['cpu_limit']=serv_rsc_cons['cpu']['edge']
        rsc_upper_bound[serv_name]['mem_limit']=serv_rsc_cons['mem']['edge']
    print("画像提供的资源上限")
    print(rsc_upper_bound)

    with open('static_data.json', 'r') as f:  
        static_data = json.load(f)
    
    rsc_constraint=static_data['rsc_constraint']

    #设置资源阈值的上限
    '''
    rsc_upper_bound={
        "face_detection": {"cpu_limit": 0.25, "mem_limit": 0.012}, 
        "gender_classification": {"cpu_limit": 0.1, "mem_limit": 0.008}
    }
    '''
    #设置资源阈值的下限
    rsc_down_bound={
        "face_detection": {"cpu_limit": 0.1, "mem_limit": 0.0002}, 
        "gender_classification": {"cpu_limit": 0.1, "mem_limit": 0.001}
    }


    conf_and_serv_info['reso']=['360p']
    conf_and_serv_info['fps']=[5]
    for serv_name in serv_names:
        conf_and_serv_info[serv_name+'_ip']=['192.168.1.9']


              
    cold_starter=KnowledgeBaseUser(conf_names=conf_names,
                                  serv_names=serv_names,
                                  service_info_list=service_info_list,
                                  user_constraint=user_constraint,
                                  rsc_constraint=rsc_constraint,
                                  rsc_upper_bound=rsc_upper_bound,
                                  rsc_down_bound=rsc_down_bound,
                                  work_condition=work_condition,
                                  portrait_info=None,
                                  bandwidth_dict=None
                                  )
    need_pred_delay=1
    if need_pred_delay==1:
        conf=dict({"reso": "360p", "fps": 10, "encoder": "JPEG"})
        flow_mapping=dict({
            "face_detection": {"model_id": 0, "node_ip": "172.27.132.253", "node_role": "host"}, 
            "gender_classification": {"model_id": 0, "node_ip": "172.27.132.253", "node_role": "host"}
            })
        resource_limit=dict({
            "face_detection": {"cpu_util_limit": 0.2, "mem_util_limit": 0.004}, 
            "gender_classification": {"cpu_util_limit": 0.2, "mem_util_limit": 0.008}
            })
        import time
        time.sleep(1)
        print('开始查找')
        status,pred_delay_list,pred_delay_total=cold_starter.get_pred_delay(conf=conf,
                                                                flow_mapping=flow_mapping,
                                                                resource_limit=resource_limit,
                                                                edge_cloud_cut_choice=0)
        print(status,pred_delay_list,pred_delay_total)

    need_bayes_search=0
    if need_bayes_search==1:
        cons_delay_list=[1.0]
        n_trials_list=[300]

        for cons_delay in cons_delay_list:
            cold_starter.user_constraint['delay']=cons_delay
            
            for n_trials in n_trials_list:
                params_in_delay_in_rsc_cons,params_in_delay_out_rsc_cons,params_out_delay_cons = cold_starter.get_plan_in_cons(n_trials=n_trials)
                print('满足全部约束解',len(params_in_delay_in_rsc_cons))
                for item in params_in_delay_in_rsc_cons:
                    print(item)

                print('满足时延，不满足资源约束解',len(params_in_delay_out_rsc_cons))
                for item in params_in_delay_out_rsc_cons:
                    print(item)
                
                print('不满足时延约束解',len(params_out_delay_cons))
                for item in params_out_delay_cons:
                    print(item)
            
           


    exit()