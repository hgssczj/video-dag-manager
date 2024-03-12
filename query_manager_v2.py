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
        # 查询指令运行时情境
        self.current_runtime = dict()
        # 以下两个内容用于存储边端同步来的运行时情境
        self.runtime_pkg_list = dict()
        self.runtime_info_list = dict()  # key为任务名，value为列表
        # 历史记录
        self.plan_list = []
        self.runtime_list = []

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

            # 对car_detection的结果，提取目标数量
            # TODO：目标数量、目标大小、目标速度
            if taskname == 'car_detection':
                # 定义运行时情境字段
                if 'obj_n' not in self.runtime_pkg_list:
                    self.runtime_pkg_list['obj_n'] = list()

                # 更新各字段序列（防止爆内存）
                if len(self.runtime_pkg_list['obj_n']) > Query.CONTENT_ELE_MAXN:
                    del self.runtime_pkg_list['obj_n'][0]
                self.runtime_pkg_list['obj_n'].append(
                    sum(list(runtime_info[taskname]['count_result'].values()))
                )

        for service_name in runtime_info:
            if service_name == 'end_pipe':
                continue

            if service_name not in self.runtime_info_list:
                self.runtime_info_list[service_name] = list()

            # 保存用户的任务约束（任务约束针对整个任务而不是某个服务）
            if service_name == 'user_constraint':
                self.runtime_info_list[service_name].append(runtime_info[service_name])
            # 保存每一个服务的资源情境、工况情境、任务可配置参数
            else:
                temp_runtime_dict = dict()
                # 保存服务的资源情境
                temp_runtime_dict['resource_runtime'] = runtime_info[service_name]['proc_resource_info']
                # 保存服务的任务可配置参数
                temp_runtime_dict['task_conf'] = runtime_info[service_name]['task_conf']
                # 保存服务的工况情境
                temp_runtime_dict['work_runtime'] = dict()
                if service_name == 'face_detection':
                    temp_runtime_dict['work_runtime']['obj_n'] = len(runtime_info[service_name]['faces'])
                if service_name == 'face_alignment':
                    temp_runtime_dict['work_runtime']['obj_n'] = runtime_info[service_name]['count_result']['total']
                if service_name == 'car_detection':
                    temp_runtime_dict['work_runtime']['obj_n'] = sum(
                        list(runtime_info[service_name]['count_result'].values()))
                self.runtime_info_list[service_name].append(temp_runtime_dict)
            # 避免保存过多的内容导致爆内存
            if len(self.runtime_info_list[service_name]) > Query.CONTENT_ELE_MAXN:
                del self.runtime_info_list[service_name][0]

    def aggregate_runtime(self):
        # TODO：聚合情境感知参数的时间序列，给出预估值/统计值
        runtime_desc = dict()
        for k, v in self.runtime_pkg_list.items():
            runtime_desc[k] = sum(v) * 1.0 / len(v)

        # 获取场景稳定性
        if 'obj_n' in self.runtime_pkg_list.keys():
            runtime_desc['obj_stable'] = True if np.std(self.runtime_pkg_list['obj_n']) < 0.3 else False

        # 每次调用agg后清空
        self.runtime_pkg_list = dict()

        # 获取运行时情境画像+知识库所需参数
        service_list = list(self.runtime_info_list.keys())
        if len(service_list) == 0:  # service_list长度为0，说明任务还未完整地执行一次，尤其是第一次执行任务时时间很长
            return runtime_desc
        assert 'user_constraint' in service_list
        service_list.remove('user_constraint')
        runtime_desc['runtime_portrait'] = dict()  # key为任务名，value为列表

        for service_name in service_list:
            runtime_desc['runtime_portrait'][service_name] = list()

        for i in range(len(self.runtime_info_list['user_constraint'])):
            delay_constraint = self.runtime_info_list['user_constraint'][i]['delay']  # 用户的时延约束
            delay_exec = 0  # 整个任务实际的执行时延
            for service_name in service_list:
                delay_exec += self.runtime_info_list[service_name][i]['resource_runtime']['all_latency']

            for service_name in service_list:
                cpu_util_use = self.runtime_info_list[service_name][i]['resource_runtime']['cpu_util_use']
                cpu_util_limit = self.runtime_info_list[service_name][i]['resource_runtime']['cpu_util_limit']
                mem_util_use = self.runtime_info_list[service_name][i]['resource_runtime']['mem_util_use']
                mem_util_limit = self.runtime_info_list[service_name][i]['resource_runtime']['mem_util_limit']

                if delay_exec <= delay_constraint:  # 任务执行的时延低于用户约束
                    # 计算资源画像
                    if cpu_util_limit - cpu_util_use >= 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime'][
                            'cpu_portrait'] = 0  # 0表示强，1表示中，2表示弱
                    elif math.fabs(cpu_util_limit - cpu_util_use) < 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 1
                    else:
                        root_logger.warning("Compute resource limit is useless, cpu_util_limit:{}, cpu_util_use{}!"
                                            .format(cpu_util_limit, cpu_util_use))
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 1

                    # 存储资源画像
                    if mem_util_limit - mem_util_use >= 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 0
                    elif math.fabs(mem_util_limit - mem_util_use) < 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 1
                    else:
                        root_logger.warning("Memory resource limit is useless, mem_util_limit:{}, mem_util_use{}!"
                                            .format(mem_util_limit, mem_util_use))
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 1

                else:  # 任务执行的时延高于用户约束
                    if cpu_util_limit - cpu_util_use >= 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 0
                    elif math.fabs(cpu_util_limit - cpu_util_use) < 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 2
                    else:
                        root_logger.warning("Compute resource limit is useless, cpu_util_limit:{}, cpu_util_use{}!"
                                            .format(cpu_util_limit, cpu_util_use))
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 2

                    # 存储资源画像
                    if mem_util_limit - mem_util_use >= 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 0
                    elif math.fabs(mem_util_limit - mem_util_use) < 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 2
                    else:
                        root_logger.warning("Memory resource limit is useless, mem_util_limit:{}, mem_util_use{}!"
                                            .format(mem_util_limit, mem_util_use))
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 2

                runtime_desc['runtime_portrait'][service_name].append(self.runtime_info_list[service_name][i])

        self.runtime_info_list = dict()  # 每次获取运行时情境之后清空

        return runtime_desc
    
    def set_runtime(self, runtime_info):
        while len(self.runtime_list) >= QueryManager.LIST_BUFFER_SIZE_PER_QUERY:
            print("len(self.runtime_list)={}".format(len(self.runtime_list)))
            del self.runtime_list[0]
        self.runtime_list.append(self.current_runtime)
        self.current_runtime = runtime_info
    
    def get_runtime(self):
        new_runtime = self.aggregate_runtime()
        assert isinstance(new_runtime, dict)
        if new_runtime:  # 若new_runtime非空，则更新current_runtime；否则保持current_runtime
            self.set_runtime(new_runtime)
            # self.current_runtime = new_runtime
        return self.current_runtime

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

    def get_query_runtime(self, query_id):
        assert query_id in self.query_dict

        query = self.query_dict[query_id]
        assert isinstance(query, Query)
        return query.get_runtime()









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
@query_app.route("/query/get_runtime/<query_id>", methods=["GET"])
@flask_cors.cross_origin()
def query_get_runtime_cbk(query_id):
    return flask.jsonify(query_manager.get_query_runtime(query_id))

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
                    print("不值得调度")
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