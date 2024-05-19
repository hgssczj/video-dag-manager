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
from RuntimePortrait import RuntimePortrait
import scheduler_func.lat_first_kb_muledge
import scheduler_func.lat_first_kb_muledge_wzl
import scheduler_func.lat_first_kb_muledge_wzl_1


class Query():
    CONTENT_ELE_MAXN = 50

    def __init__(self, query_id, node_addr, video_id, pipeline, user_constraint, job_info):
        self.query_id = query_id
        # 查询指令信息
        self.node_addr = node_addr
        self.video_id = video_id
        self.pipeline = pipeline
        self.user_constraint = user_constraint
        self.job_info=job_info


        self.flow_mapping = None
        self.video_conf = None
        self.resource_limit = None
        self.created_job = False #表示已经生成了job
        # NOTES: 目前仅支持流水线
        assert isinstance(self.pipeline, list)
        # 查询指令结果
        self.result = None
        
        # 历史记录
        self.plan_list = []
        
        # 运行时情境画像模块
        self.runtime_portrait = RuntimePortrait(pipeline, user_constraint)

    # ---------------------------------------
    # ---- 属性 ----
    def set_plan(self, video_conf, flow_mapping, resource_limit):
        while len(self.plan_list) >= QueryManager.LIST_BUFFER_SIZE_PER_QUERY:
            root_logger.info("len(self.plan_list)={}".format(len(self.plan_list)))
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
        self.runtime_portrait.update_runtime(runtime_info)
    
    def get_aggregate_work_condition(self):
        return self.runtime_portrait.get_aggregate_work_condition()

    def get_latest_work_condition(self):
        return self.runtime_portrait.get_latest_work_condition()

    def get_portrait_info(self):
        return self.runtime_portrait.get_portrait_info()
                
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
    
    def set_created_job(self,value):
        self.created_job=value

    def get_created_job(self):
        return self.created_job
    
    def get_node_addr(self):
        return self.node_addr
    
    def get_job_info(self):
        return self.job_info
    
    

class QueryManager():
    # 保存执行结果的缓冲大小
    LIST_BUFFER_SIZE_PER_QUERY = 10

    def __init__(self):
        self.global_query_count = 0
        self.service_cloud_addr = None
        self.query_dict = dict()  # key: global_job_id；value: Query对象
        self.video_info = dict()
        self.bandwidth_dict = dict()

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

    def submit_query(self, query_id, node_addr, video_id, pipeline, user_constraint, job_info):
        # 在本地启动新的job
        assert query_id not in self.query_dict.keys()
        query = Query(query_id=query_id,
                      node_addr=node_addr,
                      video_id=video_id,
                      pipeline=pipeline,
                      user_constraint=user_constraint,
                      job_info=job_info)
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
        #print("要返回的画像")
        x=query.get_portrait_info()
        return x
        #return query.get_portrait_info()









# 单例变量：主线程任务管理器，Manager
# manager = Manager()
query_manager = QueryManager()
# 单例变量：后台web线程
flask.Flask.logger_name = "listlogger"
WSGIRequestHandler.protocol_version = "HTTP/1.1"
query_app = flask.Flask(__name__)
flask_cors.CORS(query_app)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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
        root_logger.info('目前云端video_info为:{}'.format(query_manager.video_info))
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
    root_logger.info('新query的job_info')
    root_logger.info(new_job_info)
    query_manager.submit_query(query_id=new_job_info['job_uid'],
                                node_addr=new_job_info['node_addr'],
                                video_id=new_job_info['video_id'],
                                pipeline=new_job_info['pipeline'],
                                user_constraint=new_job_info['user_constraint'],
                                job_info=new_job_info)
    root_logger.info('完成一次提交，新的字典为')
    root_logger.info(query_manager.query_dict)
    '''
    # 要等任务已经被领取了才算任务提交成功
    while True:
        query=query_manager.query_dict[job_uid]
        if query.get_created_job()==True:
            break
    '''
    return flask.jsonify({"status": 0,
                          "msg": "submitted to (cloud) manager from api: /query/submit_query",
                          "query_id": job_uid,
                         })


# TODO：为无job的query生成job
@query_app.route("/query/get_jobs_info_and_jobs_plan", methods=["POST"])
@flask_cors.cross_origin()
def query_get_job_cbk():
    para = flask.request.json
    jobs_info=dict()
    jobs_plan=dict()
    node_addr = para['node_addr']
    bandwidth = para['bandwidth']
    # 在query_manager里用一个字典保存每一个边缘当前的带宽。
    query_manager.bandwidth_dict[node_addr]=bandwidth
    root_logger.info('当前任务字典query_dict')
    root_logger.info(query_manager.query_dict)
    root_logger.info('当前带宽字典bandwidth_dict')
    root_logger.info(query_manager.bandwidth_dict)

    for query_id, query in query_manager.query_dict.items():
        root_logger.info('{},{},{}'.format(query_id, query.get_node_addr(), node_addr))
        # 获取尚未生成job的query的job_info
        if not query.get_created_job() and query.get_node_addr() == node_addr: 
            query.set_created_job(True) #将其状态表示为已创建job
            jobs_info[query_id]=query.get_job_info()
            root_logger.info('发现新job:{}'.format(jobs_info[query_id]))
        # 获取已经生成job且已经有调度计划的query的调度计划
        # print(query.get_created_job(),query.get_node_addr())
        if query.get_created_job() and query.get_node_addr() == node_addr and \
            query_id in scheduler_func.lat_first_kb_muledge_wzl_1.prev_conf and\
            query_id in scheduler_func.lat_first_kb_muledge_wzl_1.prev_flow_mapping and \
            query_id in scheduler_func.lat_first_kb_muledge_wzl_1.prev_resource_limit:
            root_logger.info('开始更新调度计划')
            jobs_plan[query_id] = {
                'job_uid': query_id,
                'video_conf': scheduler_func.lat_first_kb_muledge_wzl_1.prev_conf[query_id],
                'flow_mapping': scheduler_func.lat_first_kb_muledge_wzl_1.prev_flow_mapping[query_id],
                'resource_limit': scheduler_func.lat_first_kb_muledge_wzl_1.prev_resource_limit[query_id]
            }
    info_and_plan = {
        'jobs_info': jobs_info,
        'jobs_plan': jobs_plan
    }
    # 得到可用来生成job的一系列列表
    return flask.jsonify(info_and_plan)


# TODO：为无job的query生成job
@query_app.route("/query/update_prev_plan", methods=["POST"])
@flask_cors.cross_origin()
def update_prev_plan_cbk():
    para = flask.request.json

    job_uid=para['job_uid']
    scheduler_func.lat_first_kb_muledge_wzl_1.prev_conf[job_uid]=para['video_conf']
    scheduler_func.lat_first_kb_muledge_wzl_1.prev_flow_mapping[job_uid]=para['flow_mapping']
    scheduler_func.lat_first_kb_muledge_wzl_1.prev_resource_limit[job_uid]=para['resource_limit']

    return flask.jsonify({"statys":0,"msg":"prev plan has been updated"})



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
    root_logger.info('发现新边端')
    #root_logger.info("from {}: got {}".format(flask.request.remote_addr, para))
    node_ip = para['node_ip']
    node_port = para['node_port']
    node_addr = node_ip + ":" + str(node_port)
    video_id = para['video_id']
    video_type = para['video_type']
    root_logger.info("node_ip:{},node_port:{},node_addr:{}".format(node_ip,node_port,node_addr))

    query_manager.add_video(node_addr=node_addr, video_id=video_id, video_type=video_type)

    return flask.jsonify({"status": 0, "msg": "joined one video to query_manager", "node_addr": node_addr})


def start_query_listener(serv_port=4000):
    query_app.run(host="0.0.0.0", port=serv_port)


# 云端调度器主循环：基于知识库进行调度器
def cloud_scheduler_loop_kb(query_manager=None):
    assert query_manager
    assert isinstance(query_manager, QueryManager)
    
    while True:
        # 每5s调度一次
        time.sleep(3)
        #print('开始周期性的调度')
        
        root_logger.info("start new schedule ...")
        try:
            # 获取资源情境
            r = query_manager.sess.get(
                url="http://{}/get_system_status".format(query_manager.service_cloud_addr))
            system_status = r.json()
            # 为所有query生成调度策略
            query_dict = query_manager.query_dict.copy()
            for qid, query in query_dict.items():
                assert isinstance(query, Query)
                query_id = query.query_id
                work_condition = query.get_latest_work_condition()
                # work_condition = query.get_aggregate_work_condition()
                portrait_info = query.get_portrait_info()
                # appended_result_list = query.get_appended_result_list()
                if query.video_id < 99:  # 如果是大于等于99，意味着在进行视频测试，此时云端调度器不工作。否则，基于知识库进行调度。
                    root_logger.info("video_id:{}".format(query.video_id))
                    node_addr = query.node_addr  # 形如：192.168.1.9:4001
                    user_constraint = query.user_constraint
                    assert node_addr

                    # print("展示当前工况")
                    # print(work_condition)
                    # 只有当runtime_info不存在(视频流分析还未开始，进行冷启动)或者含有delay的时候(正常的视频流调度)才运行。
                    bandwidth_dict = query_manager.bandwidth_dict.copy()
                    if not work_condition or 'delay' in work_condition :
                        assert node_addr in bandwidth_dict
                        # conf, flow_mapping, resource_limit = scheduler_func.lat_first_kb_muledge.scheduler(
                        #     job_uid=query_id,
                        #     dag={"generator": "x", "flow": query.pipeline},
                        #     system_status=system_status,
                        #     work_condition=work_condition,
                        #     portrait_info=portrait_info,
                        #     user_constraint=user_constraint,
                        #     appended_result_list=appended_result_list,
                        #     bandwidth_dict=bandwidth_dict
                        # )
                        
                        # conf, flow_mapping, resource_limit = scheduler_func.lat_first_kb_muledge_wzl.scheduler(
                        #     job_uid=query_id,
                        #     dag={"generator": "x", "flow": query.pipeline},
                        #     system_status=system_status,
                        #     work_condition=work_condition,
                        #     portrait_info=portrait_info,
                        #     user_constraint=user_constraint,
                        #     bandwidth_dict=bandwidth_dict[node_addr]
                        # )
                        
                        start_time = time.time()
                        conf, flow_mapping, resource_limit = scheduler_func.lat_first_kb_muledge_wzl_1.scheduler(
                            job_uid=query_id,
                            dag={"generator": "x", "flow": query.pipeline},
                            system_status=system_status,
                            portrait_info=portrait_info,
                            user_constraint=user_constraint,
                            bandwidth_dict=bandwidth_dict[node_addr]
                        )
                        end_time = time.time()
                    
                    root_logger.info("调度策略指定的时间:{}".format(end_time - start_time))
                    root_logger.info("下面展示即将更新的调度计划：")
                    root_logger.info("{},{}".format(type(query_id),query_id))
                    root_logger.info("{},{}".format(type(conf),conf))
                    root_logger.info("{},{}".format(type(flow_mapping),flow_mapping))
                    root_logger.info("{},{}".format(type(resource_limit),resource_limit))
                else:
                    root_logger.info("query.video_id:{}, 不值得调度".format(query.video_id))
        except Exception as e:
            root_logger.error("caught exception, type={}, msg={}".format(repr(e), e), exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_port', dest='query_port',
                        type=int, default=4000)
    parser.add_argument('--serv_cloud_addr', dest='serv_cloud_addr',
                        type=str, default='127.0.0.1:4500')
    parser.add_argument('--video_cloud_port', dest='video_cloud_port',
                        type=int, default=4100)
    args = parser.parse_args()

    threading.Thread(target=start_query_listener,
                     args=(args.query_port,),
                     name="QueryFlask",
                     daemon=True).start()
    
    time.sleep(1)

    query_manager.set_service_cloud_addr(addr=args.serv_cloud_addr)


    cloud_scheduler_loop_kb(query_manager)