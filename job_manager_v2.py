import cv2
import numpy
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

import field_codec_utils
from logging_utils import root_logger
import logging_utils

import common
import json
from camera_simulation import video_info_list
from common import resolution_wh
import socket  
from test_iperf_client import get_bandwidth


# 用于获取设备的ip
def get_ip_address():  
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
    try:  
        # doesn't even have to be reachable  
        s.connect(('10.255.255.255', 1))  
        IP = s.getsockname()[0]  
    except Exception:  
        IP = '127.0.0.1'  
    finally:  
        s.close()  
    return IP  


def sfg_get_next_init_task(
    job_uid=None,
    video_cap=None,
    video_conf=None,
    curr_cam_frame_id=None,
    curr_conf_frame_id=None
):
    assert video_cap

    global resolution_wh

    # 从视频流读取一帧，根据fps跳帧
    cam_fps = video_cap.get(cv2.CAP_PROP_FPS)
    conf_fps = min(video_conf['fps'], cam_fps)

    frame = None
    new_cam_frame_id = None
    new_conf_frame_id = None
    while True:
        # 从video_fps中实际读取
        cam_frame_id = video_cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = video_cap.read()
        if not ret:
            root_logger.error('Camera input error，please check.')
            time.sleep(1)
            continue

        assert ret

        conf_frame_id = math.floor((conf_fps * 1.0 / cam_fps) * cam_frame_id)
        if conf_frame_id != curr_conf_frame_id:
            # 提高fps时，conf_frame_id 远大于 curr_conf_frame_id
            # 降低fps时，conf_frame_id 远小于 curr_conf_frame_id
            # 持平fps时，conf_frame_id 最多为 curr_conf_frame_id + 1
            new_cam_frame_id = cam_frame_id
            new_conf_frame_id = conf_frame_id
            break

    print("cam_fps={} conf_fps={}".format(cam_fps, conf_fps))
    print("new_cam_frame_id={} new_conf_frame_id={}".format(new_cam_frame_id, new_conf_frame_id))


    # 根据video_conf['reso']调整大小
    frame = cv2.resize(frame, (
        resolution_wh[video_conf['reso']]['w'],
        resolution_wh[video_conf['reso']]['h']
    ))

    input_ctx = dict()
    # input_ctx['image'] = (video_cap.get(cv2.CAP_PROP_POS_FRAMES), numpy.array(frame).shape)
    st_time = time.time()
    input_ctx['image'] = field_codec_utils.encode_image(frame)
    ed_time = time.time()
    root_logger.info(
        "time consumed in encode-decode: {}".format(ed_time - st_time))
    # input_ctx['image'] = frame.tolist()

    root_logger.warning(
        "only unsupport init task with one image frame as input")

    return new_cam_frame_id, new_conf_frame_id, input_ctx

class JobManager():
    # 保存执行结果的缓冲大小
    LIST_BUFFER_SIZE = 10

    def __init__(self):
        self.cloud_addr = None
        self.local_addr = None

        # 计算服务url
        self.service_cloud_addr = None
        self.service_url = dict()

        # keepalive的http客户端：用于对query manager通信
        self.sess = requests.Session()

        # 本地视频流
        self.video_info_list = video_info_list

        # 模拟数据库：记录下发到本地的job
        self.job_dict = dict()
        # self.job_result_dict = dict()
        
        # 记录本地节点到云端的带宽
        self.bandwidth_2_cloud = None

    def set_service_cloud_addr(self, addr):
        self.service_cloud_addr = addr
    
    # 接入query manager，汇报自身信息
    def join_query_controller(self, query_addr, tracker_port):
        print('准备加入云端')
        self.query_addr = query_addr
        node_ip=get_ip_address()
        self.local_addr=node_ip+":"+str(tracker_port)
        for video_info in self.video_info_list:
            r = self.sess.post(url="http://{}/node/join".format(self.query_addr),
                               json={"node_ip":node_ip,
                                     "node_port": tracker_port,
                                     "video_id": video_info["id"],
                                     "video_type": video_info["type"]})
            #云端不知道边缘端ip，应该用以下方法获得真正的ip
            

    def get_video_info_by_id(self, video_id=id):
        for info in self.video_info_list:
            if info["id"] == video_id:
                return info
        return None
    
    # 获取计算服务url
    def get_chosen_service_url(self, taskname, choice):
        port = self.service_cloud_addr.split(':')[1]
        url = "http://{}:{}/execute_task/{}".format(choice["node_ip"], port, taskname)
        return url
    
    # 计算限制资源的url，与服务的具体选择强相关
    def get_limit_resource_url(self, taskname, choice):
        port = self.service_cloud_addr.split(':')[1]
        url = "http://{}:{}/limit_task_resource".format(choice["node_ip"], port)
        return url

    # 更新调度计划：与通信进程竞争self.job_dict[job.get_job_uid()]，修改job状态
    def update_job_plan(self, job_uid, video_conf, flow_mapping,resource_limit):
        assert job_uid in self.job_dict.keys()

        job = self.job_dict[job_uid]
        assert isinstance(job, Job)
        job.set_plan(video_conf=video_conf, flow_mapping=flow_mapping,resource_limit=resource_limit)

        # root_logger.info("updated job-{} plan".format(job.get_job_uid()))
        
    # 获取运行时情境
    def get_job_runtime(self, job_uid):
        assert job_uid in self.job_dict.keys()

        job = self.job_dict[job_uid]
        assert isinstance(job, Job)
        rt = job.get_runtime()

        # root_logger.info("get runtime of job-{}: {}".format(job_uid, rt))
        return rt
    
    


    # 在本地启动新的job
    def submit_job(self, job_uid, node_addr, video_id, pipeline, user_constraint):
        assert job_uid not in self.job_dict.keys()
        job = Job(job_uid=job_uid,
                  node_addr=node_addr,
                  video_id=video_id,
                  pipeline=pipeline,
                  user_constraint=user_constraint)
        job.set_manager(self)
        self.job_dict[job.get_job_uid()] = job
        # root_logger.info("current job_dict={}".format(self.job_dict.keys()))
    
    # 向云端获取可用于创建job的job_info
    def created_job_and_update_plan_and_get_bandwidth(self):
        if self.local_addr:  # node_ip:tracker_port 4001
            bandwidth = get_bandwidth()
            self.bandwidth_2_cloud = bandwidth
            resp = self.sess.post(url="http://{}/query/get_jobs_info_and_jobs_plan".format(self.query_addr),
                        json={"node_addr": self.local_addr,
                              'bandwidth':bandwidth})
            
            if resp.json():  #如果本地ip已经初始化，且r1.json存在，就说明获取job_info成功了
                print('准备新建job或更新调度计划')
                info_and_plan=resp.json()
                jobs_info=info_and_plan['jobs_info']
                jobs_plan=info_and_plan['jobs_plan']
                # 建立新job
                for job_uid,job_info in jobs_info.items():
                    print('建立新job',job_uid)
                    self.submit_job(
                        job_uid=job_uid,
                        node_addr=job_info['node_addr'],
                        video_id=job_info['video_id'],
                        pipeline=job_info['pipeline'],
                        user_constraint=job_info['user_constraint']
                    )
                # 更新调度计划
                for job_uid,job_plan in jobs_plan.items():
                    print('更新新调度计划',job_uid)
                    print(job_plan)
                    self.update_job_plan(
                        job_uid=job_uid,
                        video_conf=job_plan['video_conf'],
                        flow_mapping=job_plan['flow_mapping'],
                        resource_limit=job_plan['resource_limit']
                    )

                        

    # 工作节点获取未分配工作线程的查询任务
    def start_new_job(self):
        # root_logger.info("job_dict keys: {}".format(self.job_dict.keys()))

        n = 0
        for jid, job in self.job_dict.items():
            assert isinstance(job, Job)
            if job.get_state() == Job.JOB_STATE_READY:
                job.start_worker_loop()
                root_logger.info("start to run job-{} in new thread".format(job.get_job_uid()))
            if job.get_state() == Job.JOB_STATE_RUNNING:
                n += 1

        
        # root_logger.info("{}/{} jobs running".format(n, len(self.job_dict)))

    # TODO：将Job的结果同步到query manager（本地不存放结果）
    def sync_job_result(self, job_uid, job_result, report2qm=True):
        # if job_uid not in self.job_result_dict:
        #     self.job_result_dict[job_uid] = {
        #         "appended_result": list(), "latest_result": dict()}
        # assert isinstance(job_result, dict)
        # assert job_uid in self.job_result_dict
        # for k, v in job_result.items():
        #     assert k in self.job_result_dict[job_uid].keys()
        #     if k == "appended_result":
        #         # 仅保留最近一批结果（防止爆内存）
        #         if len(self.job_result_dict[job_uid][k]) > JobManager.LIST_BUFFER_SIZE:
        #             del self.job_result_dict[job_uid][k][0]
        #         self.job_result_dict[job_uid][k].append(v)
        #     else:
        #         # 直接替换结果
        #         assert isinstance(v, dict)
        #         self.job_result_dict[job_uid][k].update(v)

        if report2qm:
            r = self.sess.post(url="http://{}/query/sync_result".format(self.query_addr),
                               json={"job_uid": job_uid,
                                     "job_result": job_result})

    # 将任务的运行时情境基础信息同步到query manager（本地不存放）
    def sync_job_runtime(self, job_uid, job_runtime, report2qm=True):
        if report2qm:
            r = self.sess.post(url="http://{}/query/sync_runtime".format(self.query_addr),
                               json={"job_uid": job_uid,
                                     "job_runtime": job_runtime})

    def remove_job(self, job):
        # 根据job的id移除job
        del self.job_dict[job.get_job_uid()]






class Job():
    JOB_STATE_UNSCHED = 0
    JOB_STATE_READY = 1
    JOB_STATE_RUNNING = 2

    def __init__(self, job_uid, node_addr, video_id, pipeline, user_constraint):
        # job的全局唯一id
        self.job_uid = job_uid
        self.manager = None
        # 视频分析流信息
        self.node_addr = node_addr
        self.video_id = video_id
        self.pipeline = pipeline
        # 执行状态机（本地不保存结果）
        self.state = Job.JOB_STATE_UNSCHED
        self.worker_thread = None
        # 调度状态机：执行计划与历史计划的执行结果
        self.user_constraint = user_constraint
        self.flow_mapping = None
        self.video_conf = None
        self.resource_limit = None
        # keepalive的http客户端：用于请求计算服务
        self.sess = requests.Session()

        # 拓扑解析dag图
        # NOTES: 目前仅支持流水线
        #        Start -> D -> C -> End
        #          0      1    2     3
        assert isinstance(self.pipeline, list)

    def set_manager(self, manager):
        self.manager = manager
        assert isinstance(self.manager, JobManager)

    def get_job_uid(self):
        return self.job_uid
    
    def get_state(self):
        return self.state

    # ---------------------------------------
    # ---- 执行计划与执行计划结果的相关函数 ----
    def set_plan(self, video_conf, flow_mapping, resource_limit):
        if self.get_state() == Job.JOB_STATE_UNSCHED:
            self.state = Job.JOB_STATE_READY

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
            common.PLAN_KEY_RESOURCE_LIMIT:self.resource_limit
        }

    def set_user_constraint(self, user_constraint):
        self.user_constraint = user_constraint
        assert isinstance(user_constraint, dict)

    def get_user_constraint(self):
        return self.user_constraint
    
    
    # ------------------
    # ---- 执行循环 ----
    def start_worker_loop(self):
        self.worker_thread = threading.Thread(target=self.worker_loop)
        self.worker_thread.start()
        self.state = Job.JOB_STATE_RUNNING

    def worker_loop(self):
        assert isinstance(self.manager, JobManager)

        # 0、初始化数据流来源（TODO：从缓存区读取）
        cap = cv2.VideoCapture(self.manager.get_video_info_by_id(self.video_id)['url'])
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        
        n = 0
        curr_cam_frame_id = 0
        curr_conf_frame_id = 0

        # 逐帧汇报结果，逐帧汇报运行时情境
        while True:
            cur_plan = self.get_plan()  # 保存本次执行任务时的计划，避免出现执行任务过程中云端修改了执行计划，而导致任务执行前后计划不一致
            # 1、根据video_conf，获取本次循环的输入数据（TODO：从缓存区读取）
            cam_frame_id, conf_frame_id, output_ctx = \
                sfg_get_next_init_task(job_uid=self.get_job_uid(),
                                       video_cap=cap,
                                       video_conf=cur_plan[common.PLAN_KEY_VIDEO_CONF],
                                       curr_cam_frame_id=curr_cam_frame_id,
                                       curr_conf_frame_id=curr_conf_frame_id)
            frame_encoded = output_ctx['image']
            root_logger.info("done generator task, get_next_init_task({})".format(output_ctx.keys()))
            
            # 2、执行
            frame_result = dict()  # 只保存视频帧在DAG最后一步的结果，不保留中间任务的结果
            plan_result = dict()
            plan_result['delay'] = dict()  # 保存DAG中每一步执行的时延
            runtime_dict = dict()
            runtime_dict['data_trans_size'] = dict()
            plan_result['process_delay']=dict() #记录当前帧处理过程中每一个task对应的计算时延，process_delay加上网络传输时延才等于delay
            proc_resource_info_dict = dict()
            data_to_cloud = 0  # 记录本次执行边缘端与云端之间的通信数据量
            for taskname in self.pipeline:
                print("开始执行服务", taskname)
                root_logger.info("to forward taskname={}".format(taskname))

                input_ctx = output_ctx
                root_logger.info("get input_ctx({}) of taskname({})".format(
                    input_ctx.keys(),
                    taskname
                ))

                temp_data_trans_size = 0
                # 根据flow_mapping，执行task（本地不保存结果）
                # root_logger.info("flow_mapping ={}".format(self.flow_mapping))
                choice = cur_plan[common.PLAN_KEY_FLOW_MAPPING][taskname]
                # root_logger.info("get choice of '{}' in flow_mapping, choose: {}".format(taskname, choice))
                if choice['node_role'] == 'cloud':  # 如果在云端执行，则计算向云端发送的数据量，单位为字节
                    data_to_cloud += len(json.dumps(input_ctx).encode('utf-8'))
                temp_data_trans_size += len(json.dumps(input_ctx).encode('utf-8'))
                
                # 首先进行资源限制
                url = self.manager.get_limit_resource_url(taskname, choice)
                root_logger.info("获取限制资源的url {}".format(url))

                task_limit={}
                task_limit['task_name']=taskname
                task_limit['cpu_util_limit']=cur_plan[common.PLAN_KEY_RESOURCE_LIMIT][taskname]['cpu_util_limit']
                task_limit['mem_util_limit']=cur_plan[common.PLAN_KEY_RESOURCE_LIMIT][taskname]['mem_util_limit']

                resp=self.limit_service_resource(limit_url=url,taskname=taskname,task_limit=task_limit)
                # 重试,完成资源限制
                while not resp:
                    time.sleep(1)
                    resp=self.limit_service_resource(limit_url=url,taskname=taskname,task_limit=task_limit)
                
                root_logger.info("got limit resp: {}".format(resp.keys()))
                

                url = self.manager.get_chosen_service_url(taskname, choice)
                # root_logger.info("get url {}".format(url))

                st_time = time.time()
  
                output_ctx = self.invoke_service(serv_url=url, taskname=taskname, input_ctx=input_ctx)
                # 重试
                while not output_ctx:
                    time.sleep(1)
                    output_ctx = self.invoke_service(serv_url=url, taskname=taskname, input_ctx=input_ctx)
                ed_time = time.time()
                
                if choice['node_role'] == 'cloud':  # 如果在云端执行，则计算云端返回的数据量，单位为字节
                    data_to_cloud += len(json.dumps(output_ctx).encode('utf-8'))

                temp_data_trans_size += len(json.dumps(output_ctx).encode('utf-8'))
                runtime_dict['data_trans_size'][taskname] = temp_data_trans_size  # 记录各个阶段之间传输的数据量，单位：字节
                
                # 运行时感知：应用无关
                root_logger.info("got service result: {}, (delta_t={})".format(
                                  output_ctx.keys(), ed_time - st_time))
                plan_result['delay'][taskname] = ed_time - st_time
                print("展示云端所发资源信息:") #把各阶段时延都保存到plan_result内以便同步到云端
                print(output_ctx.keys())
                print(output_ctx['proc_resource_info'])
                plan_result['process_delay'][taskname]=output_ctx['proc_resource_info']['compute_latency']
                #下面的注释不用管
                # 运行时感知：应用相关
                # wrapped_ctx = output_ctx.copy()
                # wrapped_ctx['delay'] = (ed_time - st_time) / ((cam_frame_id - curr_cam_frame_id + 1) * 1.0)
                # self.update_runtime(taskname=taskname, output_ctx=wrapped_ctx)
                output_ctx['proc_resource_info']['all_latency'] = ed_time - st_time  # 单位：秒(s)，任务实际执行时延+数据传输时延
                output_ctx['proc_resource_info']['node_ip'] = choice["node_ip"]  # 将当前任务执行的节点也报告给运行时情境
                output_ctx['proc_resource_info']['node_role'] = choice["node_role"]
                output_ctx['proc_resource_info']['cpu_util_limit'] = task_limit['cpu_util_limit']
                output_ctx['proc_resource_info']['mem_util_limit'] = task_limit['mem_util_limit']
                output_ctx['task_conf'] = cur_plan[common.PLAN_KEY_VIDEO_CONF]
                proc_resource_info_dict[taskname] = output_ctx['proc_resource_info']
                runtime_dict[taskname] = output_ctx
                print("完成情境获取")
                # self.update_runtime(taskname=taskname, output_ctx=output_ctx)

            n += 1
            print("流水线初步完成，进行收尾处理")
            total_frame_delay = 0
            total_frame_process_delay = 0
            for taskname in plan_result['delay']:
                plan_result['delay'][taskname] = \
                    plan_result['delay'][taskname] / ((cam_frame_id - curr_cam_frame_id + 1) * 1.0)
                total_frame_delay += plan_result['delay'][taskname]
            
            for taskname in plan_result['process_delay']:
                plan_result['process_delay'][taskname] = \
                    plan_result['process_delay'][taskname] / ((cam_frame_id - curr_cam_frame_id + 1) * 1.0)
                total_frame_process_delay += plan_result['process_delay'][taskname]

            runtime_dict['end_pipe'] = {
                "delay": total_frame_delay,
                "process_delay": total_frame_process_delay,
                "frame_id": cam_frame_id,
                "n_loop": n
            }
            runtime_dict['process_delay'] = plan_result['process_delay']
            runtime_dict['exe_plan'] = cur_plan  # 将当前任务的执行计划也报告给运行时情境，之所以这么做是为了避免并发导致的云、边执行计划不一致
            runtime_dict['frame'] = frame_encoded  # 将视频帧编码后上云，是为了计算目标速度
            runtime_dict['cap_fps'] = cap_fps  # 视频的原始帧率，为了计算目标速度
            # DAG执行结束之后再次更新运行时情境，主要用于运行时情境画像，为知识库建立提供数据
            runtime_dict['user_constraint'] = self.user_constraint
            runtime_dict['bandwidth'] = self.manager.bandwidth_2_cloud  # 本次执行任务时边缘到云的带宽
            runtime_dict['data_to_cloud'] = data_to_cloud  # 本次执行累计向云端发送的数据量

            output_ctx["frame_id"] = cam_frame_id
            output_ctx["n_loop"] = n
            output_ctx["delay"] = total_frame_delay
            output_ctx["proc_delay"] = total_frame_process_delay
            if self.pipeline[-1] == 'gender_classification':
                output_ctx['obj_n'] = len(output_ctx['bbox'])
            frame_result.update(output_ctx)
            frame_result['bandwidth'] = self.manager.bandwidth_2_cloud['kB/s']
            # 将当前帧的运行时情境和调度策略同步推送到云端query manager
            frame_result[common.SYNC_RESULT_KEY_PLAN] = cur_plan
            # frame_result[common.SYNC_RESULT_KEY_RUNTIME] = self.get_runtime()
            frame_result[common.SYNC_RESULT_KEY_RUNTIME] = {}
            # 将plan_result放入frame_result[common.SYNC_RESULT_KEY_RUNTIME]
            frame_result[common.SYNC_RESULT_KEY_RUNTIME]['plan_result']=plan_result
            frame_result[common.SYNC_RESULT_KEY_RUNTIME]['proc_resource_info'] = proc_resource_info_dict
            
            curr_cam_frame_id = cam_frame_id
            curr_conf_frame_id = conf_frame_id

            # 3、通过job manager同步结果到query manager
            #    注意：本地不保存结果
            print("开始情境同步")
            self.manager.sync_job_result(job_uid=self.get_job_uid(),
                                           job_result= {
                                                common.SYNC_RESULT_KEY_APPEND: frame_result,
                                            }
                                        )
            # 4、通过job manager同步运行时情境信息到query manager，本地不保存情境信息
            self.manager.sync_job_runtime(job_uid=self.get_job_uid(), job_runtime=runtime_dict)


    def invoke_service(self, serv_url, taskname, input_ctx):
        root_logger.info("get serv_url={}".format(serv_url))

        r = None

        try:
            r = self.sess.post(url=serv_url, json=input_ctx)
            return r.json()

        except Exception as e:
            if r:
                root_logger.error("got serv result: {}".format(r.text))
            root_logger.error("caught exception: {}".format(e), exc_info=True)
            return None
    
    def limit_service_resource(self, limit_url, taskname, task_limit):
        root_logger.info("get limit_url={}".format(limit_url))

        r = None

        try:
            r = self.sess.post(url=limit_url, json=task_limit)
            return r.json()

        except Exception as e:
            if r:
                root_logger.error("limit_resource: {}".format(r.text))
            root_logger.error("caught exception: {}".format(e), exc_info=True)
            return None


# 单例变量：主线程任务管理器，Manager
job_manager = JobManager()
# 单例变量：后台web线程
flask.Flask.logger_name = "listlogger"
WSGIRequestHandler.protocol_version = "HTTP/1.1"
tracker_app = flask.Flask(__name__)
flask_cors.CORS(tracker_app)


def start_tracker_listener(serv_port=4001):
    tracker_app.run(host="0.0.0.0", port=serv_port)
    # app.run(port=serv_port)
    # app.run(host="*", port=serv_port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_addr', dest='query_addr',
                        type=str, default='114.212.81.11:4000')
    parser.add_argument('--tracker_port', dest='tracker_port',
                        type=int, default=4001)
    parser.add_argument('--serv_cloud_addr', dest='serv_cloud_addr',
                        type=str, default='114.212.81.11:4500')
    parser.add_argument('--video_side_port', dest='video_side_port',
                        type=int, default=4101)
    args = parser.parse_args()

    # 接受下发的query生成job、接收更新的调度策略
    threading.Thread(target=start_tracker_listener,
                    args=(args.tracker_port,),
                    name="TrackerFlask",
                    daemon=True).start()

    time.sleep(1)
    
    # 接入query manger
    job_manager.join_query_controller(query_addr=args.query_addr,
                                      tracker_port=args.tracker_port)
    root_logger.info("joined to query controller")
    
    job_manager.set_service_cloud_addr(addr=args.serv_cloud_addr)


    # 线程轮询启动循环
    # 一个Job对应一个视频流查询、对应一个进程/线程
    while True:
        job_manager.created_job_and_update_plan_and_get_bandwidth()
        job_manager.start_new_job()
        
        # 不需要睡眠，因为created_job_and_update_plan中获取带宽本身会导致休眠
        sleep_sec = 2
        root_logger.warning(f"---- sleeping for {sleep_sec} sec ----")
        time.sleep(sleep_sec)
