import cv2
import numpy as np
import math
import flask
import flask.logging
import flask_cors
import random
import requests
import threading
import multiprocessing as mp
import queue
import time
import functools
import argparse
from werkzeug.serving import WSGIRequestHandler
import sys

import field_codec_utils
from logging_utils import root_logger
import logging_utils

import common
import json
from PortraitModel import PortraitModel
import torch

class Query():
    CONTENT_ELE_MAXN = 50

    def __init__(self, query_id, node_addr, video_id, pipeline, user_constraint):
        self.query_id = query_id
        # 查询指令信息
        self.node_addr = node_addr
        self.video_id = video_id
        self.pipeline = pipeline
        self.user_constraint = user_constraint
        self.flow_mapping = None
        self.video_conf = None
        self.resource_limit = None
        # NOTES: 目前仅支持流水线
        assert isinstance(self.pipeline, list)
        # 查询指令结果
        self.result = None
        # 以下两个内容用于存储边端同步来的运行时情境
        self.runtime_pkg_list = dict()  # 此变量用于存储工况情境参数，key为工况类型'obj_n'、'obj_size'等；value为列表类型
        self.current_work_condition = dict()  # 此变量用于存储当前工况，key为工况类型'obj_n'、'obj_size'等
        self.work_condition_list = []  # 此变量用于存储历史工况，每一个元素为一个dict，字典的key为工况类型'obj_n'、'obj_size'等
        self.runtime_info_list = []  # 此变量用于存储完整的历史运行时情境信息，便于建立画像使用
        # 历史记录
        self.plan_list = []
        
        # 加载与当前pipeline对应的运行时情境画像模型
        self.portrait_model_dict = dict()  # 此变量用于保存pipeline中各个服务的画像预估模型
        
        assert(len(self.pipeline) >= 1)
        for service in self.pipeline:
            self.portrait_model_dict[service] = dict()
            
            # CPU利用率预估模型
            self.portrait_model_dict[service]['cpu'] = dict()
            cpu_edge_model_file = "models/" + service + "_cpu_edge.pth"
            cpu_server_model_file = "models/" + service + "_cpu_server.pth"
            cpu_edge_model = PortraitModel()
            cpu_edge_model.load_state_dict(torch.load(cpu_edge_model_file))
            self.portrait_model_dict[service]['cpu']['edge'] = cpu_edge_model
            cpu_server_model = PortraitModel()
            cpu_server_model.load_state_dict(torch.load(cpu_server_model_file))
            self.portrait_model_dict[service]['cpu']['server'] = cpu_server_model
            
            # 内存使用量预估模型
            self.portrait_model_dict[service]['mem'] = dict()
            mem_edge_model_file = "models/" + service + "_mem_edge.pth"
            mem_server_model_file = "models/" + service + "_mem_server.pth"
            mem_edge_model = PortraitModel()
            mem_edge_model.load_state_dict(torch.load(mem_edge_model_file))
            self.portrait_model_dict[service]['mem']['edge'] = mem_edge_model
            mem_server_model = PortraitModel()
            mem_server_model.load_state_dict(torch.load(mem_server_model_file))
            self.portrait_model_dict[service]['mem']['server'] = mem_server_model

        # 获取当前系统资源，为画像提供计算依据
        self.server_total_mem = None
        self.edge_total_mem = None
        
        r = query_manager.sess.get(url="http://{}/get_cluster_info".format(query_manager.service_cloud_addr))
        resource_info = r.json()
        
        for ip_addr in resource_info:
            if resource_info[ip_addr]['node_role'] == 'cloud':
                self.server_total_mem = resource_info[ip_addr]['device_state']['mem_total'] * 1024 * 1024 * 1024
            else:
                self.edge_total_mem = resource_info[ip_addr]['device_state']['mem_total'] * 1024 * 1024 * 1024
        
        assert(self.server_total_mem is not None)
        assert(self.edge_total_mem is not None)
    # ---------------------------------------
    # ---- 属性 ----
    def set_plan(self, video_conf, flow_mapping, resource_limit):
        while len(self.plan_list) >= QueryManager.LIST_BUFFER_SIZE_PER_QUERY:
            print("len(self.plan_list)={}".format(len(self.plan_list)))
            del self.plan_list[0]
        self.plan_list.append(self.get_plan())

        self.flow_mapping = flow_mapping
        self.video_conf = video_conf
        self.resource_limit = resource_limit
        assert isinstance(self.flow_mapping, dict)
        assert isinstance(self.video_conf, dict)
        assert isinstance(self.resource_limit,dict)

    def get_plan(self):
        return {
            common.PLAN_KEY_VIDEO_CONF: self.video_conf,
            common.PLAN_KEY_FLOW_MAPPING: self.flow_mapping,
            common.PLAN_KEY_RESOURCE_LIMIT: self.resource_limit
        }
    
    def update_runtime(self, runtime_info):
        # 更新云端的运行时情境信息
        
        # 1.更新工况信息，便于前端展示(绘制折线图等)
        self.update_work_condition(runtime_info)
        
        # 2.保存完整的运行时情境参数，为调度器查表提供参考
        self.runtime_info_list.append(runtime_info)
        # 避免保存过多的内容导致爆内存
        if len(self.runtime_info_list) > Query.CONTENT_ELE_MAXN:
            del self.runtime_info_list[0]

    def update_work_condition(self, runtime_info):
        # 更新工况信息，便于前端展示(绘制折线图等)
        for taskname in runtime_info:
            if taskname == 'end_pipe':
                if 'delay' not in self.runtime_pkg_list:
                    self.runtime_pkg_list['delay'] = list()

                if len(self.runtime_pkg_list['delay']) > Query.CONTENT_ELE_MAXN:
                    del self.runtime_pkg_list['delay'][0]
                self.runtime_pkg_list['delay'].append(runtime_info[taskname]['delay'])

            # 对face_detection的结果，提取运行时情境
            # TODO：目标数量、目标大小、目标速度
            if taskname == 'face_detection':
                # 定义运行时情境字段
                if 'obj_n' not in self.runtime_pkg_list:
                    self.runtime_pkg_list['obj_n'] = list()
                if 'obj_size' not in self.runtime_pkg_list:
                    self.runtime_pkg_list['obj_size'] = list()

                # 更新各字段序列（防止爆内存）
                if len(self.runtime_pkg_list['obj_n']) > Query.CONTENT_ELE_MAXN:
                    del self.runtime_pkg_list['obj_n'][0]
                self.runtime_pkg_list['obj_n'].append(len(runtime_info[taskname]['faces']))

                obj_size = 0
                for x_min, y_min, x_max, y_max in runtime_info[taskname]['bbox']:
                    # TODO：需要依据分辨率转化
                    obj_size += (x_max - x_min) * (y_max - y_min)
                obj_size /= len(runtime_info[taskname]['bbox'])

                if len(self.runtime_pkg_list['obj_size']) > Query.CONTENT_ELE_MAXN:
                    del self.runtime_pkg_list['obj_size'][0]
                self.runtime_pkg_list['obj_size'].append(obj_size)

    def aggregate_work_condition(self):
        # TODO：聚合情境感知参数的时间序列，给出预估值/统计值
        runtime_desc = dict()
        for k, v in self.runtime_pkg_list.items():
            runtime_desc[k] = sum(v) * 1.0 / len(v)

        # 获取场景稳定性
        if 'obj_n' in self.runtime_pkg_list.keys():
            runtime_desc['obj_stable'] = True if np.std(self.runtime_pkg_list['obj_n']) < 0.3 else False

        # 每次调用agg后清空
        self.runtime_pkg_list = dict()
        
        return runtime_desc

    def set_work_condition(self, new_work_condition):
        while len(self.work_condition_list) >= QueryManager.LIST_BUFFER_SIZE_PER_QUERY:
            print("len(self.work_condition_list)={}".format(len(self.work_condition_list)))
            del self.work_condition_list[0]
        self.work_condition_list.append(self.current_work_condition)
        self.current_work_condition = new_work_condition
    
    def get_work_condition(self):
        new_work_condition = self.aggregate_work_condition()
        assert isinstance(new_work_condition, dict)
        if new_work_condition:  # 若new_work_condition非空，则更新current_work_condition；否则保持current_work_condition
            self.set_work_condition(new_work_condition)
        return self.current_work_condition

    def get_portrait_info(self):
        portrait_info = dict()
        
        if len(self.runtime_info_list) == 0:  # 若self.runtime_info_list为空，则说明目前云端没有存储任何运行结果，无法给出画像的信息
            return portrait_info
        
        cur_runtime_info = self.runtime_info_list[-1]  # 以最近的运行时情境为依据获取画像信息
        assert(isinstance(cur_runtime_info, dict))
        
        ###### 1. 判断时延是否满足约束
        assert('end_pipe' in cur_runtime_info)
        cur_latency = cur_runtime_info['end_pipe']['delay']
        assert('user_constraint' in cur_runtime_info)
        cur_user_latency_constraint = cur_runtime_info['user_constraint']['delay']
        if_overtime = True if cur_latency > cur_user_latency_constraint else False
        
        portrait_info['cur_latency'] = cur_latency
        portrait_info['user_constraint'] = cur_runtime_info['user_constraint']
        portrait_info['if_overtime'] = if_overtime
        
        
        ###### 2. 获取当前系统中每个设备上本query可以使用的资源量
        r = query_manager.sess.get(url="http://{}/get_cluster_info".format(query_manager.service_cloud_addr))
        resource_info = r.json()
        portrait_info['available_resource'] = dict()
        
        for ip_addr in resource_info:
            portrait_info['available_resource'][ip_addr] = dict()
            temp_available_cpu = 1.0
            temp_available_mem = 1.0
            temp_node_service_state = resource_info[ip_addr]["service_state"]
            for service in temp_node_service_state:
                if service not in self.pipeline:
                    temp_available_cpu -= temp_node_service_state[service]["cpu_util_limit"]
                    temp_available_mem -= temp_node_service_state[service]["mem_util_limit"]

            portrait_info['available_resource'][ip_addr]['node_role'] = resource_info[ip_addr]['node_role']
            portrait_info['available_resource'][ip_addr]['available_cpu'] = temp_available_cpu
            portrait_info['available_resource'][ip_addr]['available_mem'] = temp_available_mem
            if resource_info[ip_addr]['node_role'] == 'cloud':
                self.server_total_mem = resource_info[ip_addr]['device_state']['mem_total'] * 1024 * 1024 * 1024
            else:
                self.edge_total_mem = resource_info[ip_addr]['device_state']['mem_total'] * 1024 * 1024 * 1024
        
        assert(self.server_total_mem is not None)
        assert(self.edge_total_mem is not None)
        
        ###### 3. 获取当前query中各个服务的资源画像
        portrait_info['resource_portrait'] = dict()
        for service in self.pipeline:
            # 获取当前服务的执行节点
            portrait_info['resource_portrait'][service] = dict()
            portrait_info['resource_portrait'][service]['node_ip'] = cur_runtime_info[service]['proc_resource_info']['node_ip']
            portrait_info['resource_portrait'][service]['node_role'] = cur_runtime_info[service]['proc_resource_info']['node_role']
            
            # 获取当前服务的资源限制和实际资源使用量
            temp_cpu_limit = cur_runtime_info[service]['proc_resource_info']['cpu_util_limit']
            temp_cpu_use = cur_runtime_info[service]['proc_resource_info']['cpu_util_use']
            temp_mem_limit = cur_runtime_info[service]['proc_resource_info']['mem_util_limit']
            temp_mem_use = cur_runtime_info[service]['proc_resource_info']['mem_util_use']
            portrait_info['resource_portrait'][service]['cpu_util_limit'] = temp_cpu_limit
            portrait_info['resource_portrait'][service]['cpu_util_use'] = temp_cpu_use
            portrait_info['resource_portrait'][service]['mem_util_limit'] = temp_mem_limit
            portrait_info['resource_portrait'][service]['mem_util_use'] = temp_mem_use
            
            # 获取当前服务的配置
            temp_task_conf = cur_runtime_info[service]['task_conf']
            temp_fps = temp_task_conf['fps']
            temp_reso = temp_task_conf['reso']
            temp_reso = common.reso_2_index_dict[temp_reso]  # 将分辨率由字符串映射为整数
            
            # 获取当前服务的工况
            if service == 'face_detection':
                temp_obj_num = len(cur_runtime_info[service]['faces'])
            elif service == 'gender_classification':
                temp_obj_num = len(cur_runtime_info[service]['gender_result'])
            
            # 使用模型预测当前的中资源阈值
            temp_task_info = {
                'service_name': service,
                'fps': temp_fps,
                'reso': temp_reso,
                'obj_num': temp_obj_num
            }
            
            temp_resource_demand = self.predict_resource_threshold(temp_task_info)
            
            # 保存服务的资源需求量
            portrait_info['resource_portrait'][service]['resource_demand'] = temp_resource_demand
            
            # 给出 强中弱 分类（0--弱；1--中；2--强），以及bmi指标
            server_cpu_lower_bound = temp_resource_demand['cpu']['cloud']['lower_bound']
            server_cpu_upper_bound = temp_resource_demand['cpu']['cloud']['upper_bound']
            server_mem_lower_bound = temp_resource_demand['mem']['cloud']['lower_bound']
            server_mem_upper_bound = temp_resource_demand['mem']['cloud']['upper_bound']
            edge_cpu_lower_bound = temp_resource_demand['cpu']['edge']['lower_bound']
            edge_cpu_upper_bound = temp_resource_demand['cpu']['edge']['upper_bound']
            edge_mem_lower_bound = temp_resource_demand['mem']['edge']['lower_bound']
            edge_mem_upper_bound = temp_resource_demand['mem']['edge']['upper_bound']
            
            if portrait_info['resource_portrait'][service]['node_role'] == 'cloud':
                if temp_cpu_limit < server_cpu_lower_bound:
                    portrait_info['resource_portrait'][service]['cpu_portrait'] = 0
                elif temp_cpu_limit > server_cpu_upper_bound:
                    portrait_info['resource_portrait'][service]['cpu_portrait'] = 2
                else:
                    portrait_info['resource_portrait'][service]['cpu_portrait'] = 1
                portrait_info['resource_portrait'][service]['cpu_bmi'] = (temp_cpu_limit - server_cpu_lower_bound) / server_cpu_lower_bound
                portrait_info['resource_portrait'][service]['cpu_bmi_lower_bound'] = 0
                portrait_info['resource_portrait'][service]['cpu_bmi_upper_bound'] = (server_cpu_upper_bound - server_cpu_lower_bound) / server_cpu_lower_bound
                
                if temp_mem_limit < server_mem_lower_bound:
                    portrait_info['resource_portrait'][service]['mem_portrait'] = 0
                elif temp_mem_limit > server_mem_upper_bound:
                    portrait_info['resource_portrait'][service]['mem_portrait'] = 2
                else:
                    portrait_info['resource_portrait'][service]['mem_portrait'] = 1
                portrait_info['resource_portrait'][service]['mem_bmi'] = (temp_mem_limit - server_mem_lower_bound) / server_mem_lower_bound
                portrait_info['resource_portrait'][service]['mem_bmi_lower_bound'] = 0
                portrait_info['resource_portrait'][service]['mem_bmi_upper_bound'] = (server_mem_upper_bound - server_mem_lower_bound) / server_mem_lower_bound
                
            else:
                if temp_cpu_limit < edge_cpu_lower_bound:
                    portrait_info['resource_portrait'][service]['cpu_portrait'] = 0
                elif temp_cpu_limit > edge_cpu_upper_bound:
                    portrait_info['resource_portrait'][service]['cpu_portrait'] = 2
                else:
                    portrait_info['resource_portrait'][service]['cpu_portrait'] = 1
                portrait_info['resource_portrait'][service]['cpu_bmi'] = (temp_cpu_limit - edge_cpu_lower_bound) / edge_cpu_lower_bound
                portrait_info['resource_portrait'][service]['cpu_bmi_lower_bound'] = 0
                portrait_info['resource_portrait'][service]['cpu_bmi_upper_bound'] = (edge_cpu_upper_bound - edge_cpu_lower_bound) / edge_cpu_lower_bound
                
                if temp_mem_limit < edge_mem_lower_bound:
                    portrait_info['resource_portrait'][service]['mem_portrait'] = 0
                elif temp_mem_limit > edge_mem_upper_bound:
                    portrait_info['resource_portrait'][service]['mem_portrait'] = 2
                else:
                    portrait_info['resource_portrait'][service]['mem_portrait'] = 1
                portrait_info['resource_portrait'][service]['mem_bmi'] = (temp_mem_limit - edge_mem_lower_bound) / edge_mem_lower_bound
                portrait_info['resource_portrait'][service]['mem_bmi_lower_bound'] = 0
                portrait_info['resource_portrait'][service]['mem_bmi_upper_bound'] = (edge_mem_upper_bound - edge_mem_lower_bound) / edge_mem_lower_bound
                
                
        return portrait_info
    
    def predict_resource_threshold(self, task_info):
        # 预测某个服务在当前配置、当前工况下的资源阈值
        service = task_info['service_name']
        temp_fps = task_info['fps']
        temp_reso = task_info['reso']
        temp_obj_num = task_info['obj_num']
        
        # 使用模型预测当前的中资源阈值
        X_data = np.array([temp_fps, temp_reso, temp_obj_num])
        X_data = X_data.astype(np.float32)
        X_data_tensor = torch.tensor(X_data)
        
        with torch.no_grad():
            edge_cpu_threshold = self.portrait_model_dict[service]['cpu']['edge'](X_data_tensor)
            edge_cpu_threshold = edge_cpu_threshold.numpy()[0]
            if edge_cpu_threshold <= 0:
                edge_cpu_threshold = 1e-4
            if service == 'face_detection':
                edge_cpu_threshold = min(1.0, edge_cpu_threshold * 1.01)
            elif service == 'gender_classification':
                edge_cpu_threshold = min(1.0, edge_cpu_threshold * 1.02)
            edge_cpu_upper_bound = min(1.0, edge_cpu_threshold * 1.02)
            edge_cpu_lower_bound = min(1.0, edge_cpu_threshold)
            
        with torch.no_grad():
            server_cpu_threshold = self.portrait_model_dict[service]['cpu']['server'](X_data_tensor)
            server_cpu_threshold = server_cpu_threshold.numpy()[0]
            if server_cpu_threshold <= 0:
                server_cpu_threshold = 1e-4
            if service == 'face_detection':
                server_cpu_threshold = min(1.0, server_cpu_threshold * 1.05)
            elif service == 'gender_classification':
                server_cpu_threshold = min(1.0, server_cpu_threshold * 1.02)
            server_cpu_upper_bound = min(1.0, server_cpu_threshold * 1.02)
            server_cpu_lower_bound = min(1.0, server_cpu_threshold)
            
        with torch.no_grad():
            edge_mem_threshold = self.portrait_model_dict[service]['mem']['edge'](X_data_tensor)
            edge_mem_threshold = edge_mem_threshold.numpy()[0]
            if service == 'face_detection':
                edge_mem_threshold = edge_mem_threshold * 1.1
            elif service == 'gender_classification':
                edge_mem_threshold = edge_mem_threshold * 1.02
            edge_mem_threshold /= self.edge_total_mem  # 将内存使用总量转为比例
            edge_mem_upper_bound = min(1.0, edge_mem_threshold * 1.02)
            edge_mem_lower_bound = min(1.0, edge_mem_threshold)
            
        with torch.no_grad():
            server_mem_threshold = self.portrait_model_dict[service]['mem']['server'](X_data_tensor)
            server_mem_threshold = server_mem_threshold.numpy()[0]
            if service == 'face_detection':
                server_mem_threshold = server_mem_threshold * 1.1
            elif service == 'gender_classification':
                server_mem_threshold = server_mem_threshold * 1.02
            server_mem_threshold /= self.server_total_mem  # 将内存使用总量转为比例
            server_mem_upper_bound = min(1.0, server_mem_threshold * 1.02)
            server_mem_lower_bound = min(1.0, server_mem_threshold)
            
        return {
                'cpu': {
                    'cloud': {
                        'upper_bound': server_cpu_upper_bound,
                        'lower_bound': server_cpu_lower_bound
                    },
                    'edge': {
                        'upper_bound': edge_cpu_upper_bound,
                        'lower_bound': edge_cpu_lower_bound
                    }
                },
                'mem': {
                    'cloud': {
                        'upper_bound': server_mem_upper_bound,
                        'lower_bound': server_mem_lower_bound
                    },
                    'edge': {
                        'upper_bound': edge_mem_upper_bound,
                        'lower_bound': edge_mem_lower_bound
                    }
                }
            }
        
    def help_cold_start(self, service):
        # 预估一个服务在单位工况、所有配置下的最大中资源阈值，提供给调度器进行冷启动
        resource_threshold =  {
                                'cpu': {  
                                    'cloud': 0,  
                                    'edge': 0  
                                },
                                'mem': {
                                    'cloud': 0,
                                    'edge': 0
                                }
                            }
        # 遍历所有可能的配置
        for fps in common.fps_list:
            for reso in common.reso_2_index_dict:
                temp_service_info = {
                    'service_name': service,
                    'fps': fps,
                    'reso': common.reso_2_index_dict[reso],
                    'obj_num': 1
                }
                
                temp_resource_demand = self.predict_resource_threshold(temp_service_info)
                server_cpu_upper_bound = temp_resource_demand['cpu']['cloud']['upper_bound']
                server_mem_upper_bound = temp_resource_demand['mem']['cloud']['upper_bound']
                edge_cpu_upper_bound = temp_resource_demand['cpu']['edge']['upper_bound']
                edge_mem_upper_bound = temp_resource_demand['mem']['edge']['upper_bound']
                
                resource_threshold['cpu']['cloud'] = max(server_cpu_upper_bound, resource_threshold['cpu']['cloud'])
                resource_threshold['cpu']['edge'] = max(edge_cpu_upper_bound, resource_threshold['cpu']['edge'])
                resource_threshold['mem']['cloud'] = max(server_mem_upper_bound, resource_threshold['mem']['cloud'])
                resource_threshold['mem']['edge'] = max(edge_mem_upper_bound, resource_threshold['mem']['edge'])
        
        return resource_threshold
                

    def set_user_constraint(self, user_constraint):
        self.user_constraint = user_constraint
        assert isinstance(user_constraint, dict)

    def get_user_constraint(self):
        return self.user_constraint
    
    def get_query_id(self):
        return self.query_id
    
    def update_result(self, new_result):
        '''
        更新query的处理结果。
        该函数由query对应的job通过RESTFUL API触发，参见/query/sync_result接口
        由于runtime在云端生成，所以此处根据result更新的时候只会重置plan
        '''
        if not self.result:
            self.result = {
                common.SYNC_RESULT_KEY_APPEND: list(),
                common.SYNC_RESULT_KEY_LATEST: dict()
            }
        assert isinstance(self.result, dict)

        for k, v in new_result.items():
            assert k in self.result.keys()
            if k == common.SYNC_RESULT_KEY_APPEND:
                # 仅保留最近一批结果（防止爆内存）
                if len(self.result[k]) > QueryManager.LIST_BUFFER_SIZE_PER_QUERY:
                    del self.result[k][0]
                self.result[k].append(v)
                # 更新plan
                if isinstance(v, dict):
                    assert(common.SYNC_RESULT_KEY_PLAN in v.keys())
                    assert(common.SYNC_RESULT_KEY_RUNTIME in v.keys())
                    self.set_plan(
                        video_conf=v[common.SYNC_RESULT_KEY_PLAN][common.PLAN_KEY_VIDEO_CONF],
                        flow_mapping=v[common.SYNC_RESULT_KEY_PLAN][common.PLAN_KEY_FLOW_MAPPING],
                        resource_limit=v[common.SYNC_RESULT_KEY_PLAN][common.PLAN_KEY_RESOURCE_LIMIT]
                    )
            elif k == common.SYNC_RESULT_KEY_LATEST:
                # 直接替换结果
                assert isinstance(v, dict)
                self.result[k].update(v)
            else:
                root_logger.error("unsupported sync result key: {}. value is: {}".format(k, v))
#以下的get_last_plan_result和get_appended_result_list在query_manager里被删除了
    
    def get_last_plan_result(self):
        if self.result and 'latest_result' in self.result:
            if 'plan_result' in self.result['latest_result']:
                return self.result['latest_result']['plan_result']
        return None
    
    def get_appended_result_list(self):
        if self.result and 'appended_result' in self.result:
            return self.result['appended_result']
        return None
    
    def get_result(self):
        return self.result

class QueryManager():
    # 保存执行结果的缓冲大小
    LIST_BUFFER_SIZE_PER_QUERY = 10

    def __init__(self):
        self.global_query_count = 0
        self.service_cloud_addr = None
        self.query_dict = dict()  # key: global_job_id；value: Query对象
        self.video_info = dict()

        # keepalive的http客户端
        self.sess = requests.Session()

    def generate_global_job_id(self):
        self.global_query_count += 1
        new_id = "GLOBAL_ID_" + str(self.global_query_count)
        return new_id

    def set_service_cloud_addr(self, addr):
        self.service_cloud_addr = addr

    def add_video(self, node_addr, video_id, video_type):
        if node_addr not in self.video_info:
            self.video_info[node_addr] = dict()
            
        if video_id not in self.video_info[node_addr]:
            self.video_info[node_addr][video_id] = dict()

        self.video_info[node_addr][video_id].update({"type": video_type})

    def submit_query(self, query_id, node_addr, video_id, pipeline, user_constraint):
        # 在本地启动新的job
        assert query_id not in self.query_dict.keys()
        query = Query(query_id=query_id,
                      node_addr=node_addr,
                      video_id=video_id,
                      pipeline=pipeline,
                      user_constraint=user_constraint)
        # job.set_manager(self)
        self.query_dict[query.get_query_id()] = query
        root_logger.info("current query_dict={}".format(self.query_dict.keys()))

    def sync_query_result(self, query_id, new_result):
        assert query_id in self.query_dict

        query = self.query_dict[query_id]
        assert isinstance(query, Query)
        query.update_result(new_result)

    def sync_query_runtime(self, query_id, new_runtime):
        assert query_id in self.query_dict

        query = self.query_dict[query_id]
        assert isinstance(query, Query)
        query.update_runtime(new_runtime)
    
    def get_query_result(self, query_id):
        assert query_id in self.query_dict

        query = self.query_dict[query_id]
        assert isinstance(query, Query)
        return query.get_result()
    
    def get_query_plan(self, query_id):
        assert query_id in self.query_dict

        query = self.query_dict[query_id]
        assert isinstance(query, Query)
        return query.get_plan()

    def get_query_work_condition(self, query_id):
        assert query_id in self.query_dict

        query = self.query_dict[query_id]
        assert isinstance(query, Query)
        return query.get_work_condition()
    
    def get_query_portrait_info(self, query_id):
        assert query_id in self.query_dict

        query = self.query_dict[query_id]
        assert isinstance(query, Query)
        return query.get_portrait_info()









# 单例变量：主线程任务管理器，Manager
# manager = Manager()
query_manager = QueryManager()
# 单例变量：后台web线程
flask.Flask.logger_name = "listlogger"
WSGIRequestHandler.protocol_version = "HTTP/1.1"
query_app = flask.Flask(__name__)
flask_cors.CORS(query_app)

# 模拟云端数据库，维护接入节点及其已经submit的任务的job_uid。
# 用户接口（/user/xxx）争用查询&修改，云端调度器（cloud_scheduler_loop）争用查询
# 单例变量：接入到当前节点的节点信息
node_status = dict()








# 接受用户提交视频流查询
# 递归请求：/job/submit_job
@query_app.route("/query/submit_query", methods=["POST"])
@flask_cors.cross_origin()
def user_submit_query_cbk():
    # 获取用户针对视频流提交的job，转发到对应边端
    para = flask.request.json
    root_logger.info("/query/submit_query got para={}".format(para))
    node_addr = para['node_addr']
    video_id = para['video_id']
    pipeline = para['pipeline']
    user_constraint = para['user_constraint']

    if node_addr not in query_manager.video_info:
        return flask.jsonify({"status": 1, "error": "cannot found {}".format(node_addr)})

    # TODO：在云端注册任务实例，维护job执行结果、调度信息
    job_uid = query_manager.generate_global_job_id()
    new_job_info = {
        'job_uid': job_uid,
        'node_addr': node_addr,
        'video_id': video_id,
        'pipeline': pipeline,
        'user_constraint': user_constraint
    }
    query_manager.submit_query(query_id=new_job_info['job_uid'],
                                node_addr=new_job_info['node_addr'],
                                video_id=new_job_info['video_id'],
                                pipeline=new_job_info['pipeline'],
                                user_constraint=new_job_info['user_constraint'])

    # TODO：在边缘端为每个query创建一个job
    r = query_manager.sess.post("http://{}/job/submit_job".format(node_addr), 
                                json=new_job_info)
    
    # TODO：更新sidechan信息
    # cloud_ip = manager.get_cloud_addr().split(":")[0]
    cloud_ip = "127.0.0.1"
    r_sidechan = query_manager.sess.post(url="http://{}:{}/user/update_node_addr".format(cloud_ip, 5100),
                                   json={"job_uid": job_uid,
                                         "node_addr": node_addr.split(":")[0] + ":5101"})

    return flask.jsonify({"status": 0,
                          "msg": "submitted to (cloud) manager from api: /query/submit_query",
                          "query_id": job_uid,
                          "r_sidechan": r_sidechan.text})

# TODO：同步job的执行结果
@query_app.route("/query/sync_result", methods=["POST"])
@flask_cors.cross_origin()
def query_sync_result_cbk():
    para = flask.request.json

    job_uid = para['job_uid']
    job_result = para['job_result']

    query_manager.sync_query_result(query_id=job_uid, new_result=job_result)

    return flask.jsonify({"status": 500})

@query_app.route("/query/sync_runtime", methods=["POST"])
@flask_cors.cross_origin()
def query_sync_runtime_cbk():
    para = flask.request.json

    job_uid = para['job_uid']
    job_runtime = para['job_runtime']

    query_manager.sync_query_runtime(query_id=job_uid, new_runtime=job_runtime)

    return flask.jsonify({"status": 500})

@query_app.route("/query/get_result/<query_id>", methods=["GET"])
@flask_cors.cross_origin()
def query_get_result_cbk(query_id):
    return flask.jsonify(query_manager.get_query_result(query_id))

@query_app.route("/query/get_plan/<query_id>", methods=["GET"])
@flask_cors.cross_origin()
def query_get_plan_cbk(query_id):
    return flask.jsonify(query_manager.get_query_plan(query_id))

@query_app.route("/query/get_work_condition/<query_id>", methods=["GET"])
@flask_cors.cross_origin()
def query_get_work_condition_cbk(query_id):
    return flask.jsonify(query_manager.get_query_work_condition(query_id))

@query_app.route("/query/get_portrait_info/<query_id>", methods=["GET"])
@flask_cors.cross_origin()
def query_get_portrait_info_cbk(query_id):
    return flask.jsonify(query_manager.get_query_portrait_info(query_id))

@query_app.route("/query/get_agg_info/<query_id>", methods=["GET"])
@flask_cors.cross_origin()
def query_get_agg_info_cbk(query_id):
    resp = dict()
    resp.update(query_manager.get_query_result(query_id))

    # resp["latest_result"] = dict()
    # resp["latest_result"]["plan"] = query_manager.get_query_plan(query_id)
    # resp["latest_result"]["runtime"] = query_manager.get_query_runtime(query_id)
    return flask.jsonify(resp)

@query_app.route("/node/get_video_info", methods=["GET"])
@flask_cors.cross_origin()
def node_video_info():
    return flask.jsonify(query_manager.video_info)

# 接受边缘节点的视频流接入信息
@query_app.route("/node/join", methods=["POST"])
@flask_cors.cross_origin()
def node_join_cbk():
    para = flask.request.json
    root_logger.info("from {}: got {}".format(flask.request.remote_addr, para))
    node_ip = flask.request.remote_addr
    node_port = para['node_port']
    node_addr = node_ip + ":" + str(node_port)
    video_id = para['video_id']
    video_type = para['video_type']

    query_manager.add_video(node_addr=node_addr, video_id=video_id, video_type=video_type)

    return flask.jsonify({"status": 0, "msg": "joined one video to query_manager", "node_addr": node_addr})








def start_query_listener(serv_port=5000):
    query_app.run(host="0.0.0.0", port=serv_port)

# 云端调度器主循环：为manager的所有任务决定调度策略，并主动post策略到对应节点，让节点代理执行
# 不等待执行结果，节点代理执行完毕后post /job/update_plan接口提交结果
'''
def cloud_scheduler_loop(query_manager=None):
    assert query_manager
    assert isinstance(query_manager, QueryManager)

    # import scheduler_func.demo_scheduler
    # import scheduler_func.pid_scheduler
    # import scheduler_func.pid_mogai_scheduler
    # import scheduler_func.pid_content_aware_scheduler
    # import scheduler_func.lat_first_pid
    import scheduler_func.lat_first_pid_muledge


    while True:
        # 每5s调度一次
        time.sleep(3)

        root_logger.info("start new schedule ...")
        try:
            # 获取资源情境
            r = query_manager.sess.get(
                url="http://{}/get_resource_info".format(query_manager.service_cloud_addr))
            resource_info = r.json()
            
            # 访问已注册的所有job实例，获取实例中保存的结果，生成调度策略
            query_dict = query_manager.query_dict.copy()
            for qid, query in query_dict.items():
                assert isinstance(query, Query)

                query_id = query.query_id
                node_addr = query.node_addr
                user_constraint = query.user_constraint
                assert node_addr

                # 获取当前query的运行时情境（query_id == job_uid
                # r = query_manager.sess.get(
                #     url="http://{}/job/get_runtime/{}".format(node_addr, query_id)
                # )
                # runtime_info = r.json()
                # 这里直接使用get_runtime来获取内容，也就是说，不需要向边缘端发出请求来直接获取边缘端情境了。
                runtime_info = query.get_runtime()
                root_logger.info("In cloud_scheduler_loop, runtime_info is {}".format(runtime_info))

                #修改：只有当runtimw_info不存在或者含有delay的时候才运行。
                if not runtime_info or 'delay' in runtime_info :
                    # conf, flow_mapping = scheduler_func.pid_mogai_scheduler.scheduler(
                    # conf, flow_mapping = scheduler_func.pid_content_aware_scheduler.scheduler(
                    # conf, flow_mapping = scheduler_func.lat_first_pid.scheduler(
                    conf, flow_mapping = scheduler_func.lat_first_pid_muledge.scheduler(
                        # flow=job.get_dag_flow(),
                        job_uid=query_id,
                        dag={"generator": "x", "flow": query.pipeline},
                        resource_info=resource_info,
                        runtime_info=runtime_info,
                        # last_plan_res=last_plan_result,
                        user_constraint=user_constraint
                    )
                print("下面展示即将发送到边端的调度计划")
                print(type(query_id),query_id)
                print(type(conf),conf)
                print(type(flow_mapping),flow_mapping)

                resource_limit={
                    "face_detection": {
                        "cpu_util_limit": 1,
                        "mem_util_limit": 1,
                    },
                    "face_alignment": {
                        "cpu_util_limit": 1,
                        "mem_util_limit": 1,
                    }
                }

                # 主动post策略到对应节点（即更新对应视频流query pipeline的执行策略），让节点代理执行，不等待执行结果
                r = query_manager.sess.post(url="http://{}/job/update_plan".format(node_addr),
                            json={"job_uid": query_id, "video_conf": conf, "flow_mapping": flow_mapping,"resource_limit":resource_limit})
        except Exception as e:
            root_logger.error("caught exception, type={}, msg={}".format(repr(e), e), exc_info=True)
'''
# 云端调度器主循环：读取json文件的配置并直接使用，不进行自主调度
def cloud_scheduler_loop_static(query_manager=None):
    assert query_manager
    assert isinstance(query_manager, QueryManager)
    while True:
        # 每5s调度一次
        time.sleep(3)
        root_logger.info("start new schedule ...")
        try:
            # 访问已注册的所有job实例，获取实例中保存的结果，生成调度策略
            query_dict = query_manager.query_dict.copy()
            for qid, query in query_dict.items():
                assert isinstance(query, Query)
                query_id = query.query_id
                if query.video_id!=99:  #如果是99，意味着在进行视频测试，此时云端调度器不工作
                    node_addr = query.node_addr
                    user_constraint = query.user_constraint
                    assert node_addr
                    conf={
                        "reso": "360p",
                        "fps": 1,
                        "encoder": "JPEG"
                    }
                    flow_mapping={
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
                    }
                    resource_limit={
                        "face_detection": {
                            "cpu_util_limit": 1,
                            "mem_util_limit": 1,
                        },
                        "face_alignment": {
                            "cpu_util_limit": 1,
                            "mem_util_limit": 1,
                        }
                    }
                    with open('csy_test_data.json') as f:
                        csy_test_data = json.load(f)
                        conf=csy_test_data['video_conf']
                        flow_mapping=csy_test_data['flow_mapping']
                    print("下面展示即将发送到边端的调度计划，无自主调度，读取文件而已：")
                    print(type(query_id),query_id)
                    print(type(conf),conf)
                    print(type(flow_mapping),flow_mapping)
                    # 主动post策略到对应节点（即更新对应视频流query pipeline的执行策略），让节点代理执行，不等待执行结果
                    r = query_manager.sess.post(url="http://{}/job/update_plan".format(node_addr),
                                json={"job_uid": query_id, "video_conf": conf, "flow_mapping": flow_mapping,'resource_limit':resource_limit})
                
        except Exception as e:
            root_logger.error("caught exception, type={}, msg={}".format(repr(e), e), exc_info=True)

# 云端调度器主循环：基于知识库进行调度器
def cloud_scheduler_loop_kb(query_manager=None):
    assert query_manager
    assert isinstance(query_manager, QueryManager)
    import scheduler_func.lat_first_kb_muledge
    while True:
        # 每5s调度一次
        time.sleep(3)
        print('开始周期性的调度')
        
        root_logger.info("start new schedule ...")
        try:
            # 获取资源情境
            r = query_manager.sess.get(
                url="http://{}/get_resource_info".format(query_manager.service_cloud_addr))
            resource_info = r.json()
            # 为所有query生成调度策略
            query_dict = query_manager.query_dict.copy()
            for qid, query in query_dict.items():
                assert isinstance(query, Query)
                query_id = query.query_id
                
                if query.video_id<99:  #如果是大于等于99，意味着在进行视频测试，此时云端调度器不工作。否则，基于知识库进行调度。
                    print("video_id",query.video_id)
                    node_addr = query.node_addr
                    user_constraint = query.user_constraint
                    assert node_addr

                    runtime_info = query.get_runtime()
                    print("展示当前运行时情境内容")
                    print(runtime_info)
                    #修改：只有当runtimw_info不存在或者含有delay的时候才运行。
                    # best_conf, best_flow_mapping, best_resource_limit
                    if not runtime_info or 'delay' in runtime_info :
                        conf, flow_mapping,resource_limit = scheduler_func.lat_first_kb_muledge.scheduler(
                            job_uid=query_id,
                            dag={"generator": "x", "flow": query.pipeline},
                            resource_info=resource_info,
                            runtime_info=runtime_info,
                            user_constraint=user_constraint
                        )
                    print("下面展示即将发送到边端的调度计划：")
                    print(type(query_id),query_id)
                    print(type(conf),conf)
                    print(type(flow_mapping),flow_mapping)
                    print(type(resource_limit),resource_limit)

                    # 更新边端策略
                    r = query_manager.sess.post(url="http://{}/job/update_plan".format(node_addr),
                                json={"job_uid": query_id, "video_conf": conf, "flow_mapping": flow_mapping,'resource_limit':resource_limit})
                else:
                    print("query_id:",query_id,"不值得调度")
        except Exception as e:
            root_logger.error("caught exception, type={}, msg={}".format(repr(e), e), exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_port', dest='query_port',
                        type=int, default=5000)
    parser.add_argument('--serv_cloud_addr', dest='serv_cloud_addr',
                        type=str, default='127.0.0.1:5500')
    parser.add_argument('--video_cloud_port', dest='video_cloud_port',
                        type=int, default=5100)
    args = parser.parse_args()

    threading.Thread(target=start_query_listener,
                     args=(args.query_port,),
                     name="QueryFlask",
                     daemon=True).start()
    
    time.sleep(1)

    query_manager.set_service_cloud_addr(addr=args.serv_cloud_addr)

    # 启动视频流sidechan（由云端转发请求到边端）
    import cloud_sidechan
    video_serv_inter_port = args.video_cloud_port
    mp.Process(target=cloud_sidechan.init_and_start_video_proc,
               args=(video_serv_inter_port,)).start()
    time.sleep(1)

    cloud_scheduler_loop_kb(query_manager)