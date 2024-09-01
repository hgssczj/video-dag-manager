
from .RuntimePortrait import RuntimePortrait
import common #需要一个common来获取一些重要内容
from logging_utils import root_logger

from .scheduler_func import lat_first_kb_muledge_wzl_1
import time



def get_runtime_str(portrait_info, pipeline, bandwidth_dict, act_work_condition):
    runtime_str = ''
    
    ############ 1. 拼接当前的调度策略 ############
    old_conf = portrait_info['exe_plan'][common.PLAN_KEY_VIDEO_CONF]
    old_flow_mapping = portrait_info['exe_plan'][common.PLAN_KEY_FLOW_MAPPING]
    old_resource_limit = portrait_info['exe_plan'][common.PLAN_KEY_RESOURCE_LIMIT]
    
    conf_list = sorted(list(old_conf.keys()))
    for conf in conf_list:
        runtime_str += conf
        runtime_str += '='
        runtime_str += str(old_conf[conf])
        runtime_str += ' '
    
    for service in pipeline:
        runtime_str += service
        runtime_str += '_ip='
        runtime_str += old_flow_mapping[service]['node_ip']
        runtime_str += ' '
    
    for service in pipeline:
        runtime_str += service
        runtime_str += '_cpu_util_limit='
        runtime_str += str(old_resource_limit[service]['cpu_util_limit'])
        runtime_str += ' '
        
        runtime_str += service
        runtime_str += '_mem_util_limit='
        runtime_str += str(old_resource_limit[service]['mem_util_limit'])
        runtime_str += ' '
    
    
    ############ 2. 拼接当前的资源情境、工况情境 ############
    ###### 2.1 拼接当前的带宽情境 ######
    bandwidth_level = -1
    if bandwidth_dict['MB/s'] >= 100:
        bandwidth_level = 0
    elif bandwidth_dict['MB/s'] >= 50:
        bandwidth_level = 1
    elif bandwidth_dict['MB/s'] >= 10:
        bandwidth_level = 2
    elif bandwidth_dict['MB/s'] >= 5:
        bandwidth_level = 3
    elif bandwidth_dict['MB/s'] >= 4:
        bandwidth_level = 4
    elif bandwidth_dict['MB/s'] >= 3:
        bandwidth_level = 5
    elif bandwidth_dict['MB/s'] >= 2:
        bandwidth_level = 6
    elif bandwidth_dict['MB/s'] >= 1:
        bandwidth_level = 7
    elif bandwidth_dict['kB/s'] >= 900:
        bandwidth_level = 8
    elif bandwidth_dict['kB/s'] >= 800:
        bandwidth_level = 9
    elif bandwidth_dict['kB/s'] >= 700:
        bandwidth_level = 10
    elif bandwidth_dict['kB/s'] >= 600:
        bandwidth_level = 11
    elif bandwidth_dict['kB/s'] >= 500:
        bandwidth_level = 12
    elif bandwidth_dict['kB/s'] >= 400:
        bandwidth_level = 13
    elif bandwidth_dict['kB/s'] >= 300:
        bandwidth_level = 14
    elif bandwidth_dict['kB/s'] >= 200:
        bandwidth_level = 15
    elif bandwidth_dict['kB/s'] >= 100:
        bandwidth_level = 16
    else:
        bandwidth_level = 17
    
    runtime_str += 'bandwidth_level='
    runtime_str += str(bandwidth_level)
    runtime_str += ' '
    
    ###### 2.2 拼接当前的工况情境 ######
    # assert 'work_condition' in portrait_info
    # work_condition = portrait_info['work_condition']
    
    assert 'obj_n' in act_work_condition
    runtime_str += 'obj_n='
    runtime_str += str(int(act_work_condition['obj_n']))
    runtime_str += ' '
    
    assert 'obj_size' in act_work_condition
    obj_size_level = -1
    if int(act_work_condition['obj_size']) == 0:
        obj_size_level = 2
    elif act_work_condition['obj_size'] <= 50000:
        obj_size_level = 1
    elif act_work_condition['obj_size'] <= 100000:
        obj_size_level = 2
    else:
        obj_size_level = 3
    runtime_str += 'obj_size='
    runtime_str += str(obj_size_level)
    runtime_str += ' '
    
    assert 'obj_speed' in act_work_condition
    obj_speed_level = -1
    if int(act_work_condition['obj_speed']) == 0:
        obj_speed_level = 2
    elif act_work_condition['obj_speed'] <= 260:
        obj_speed_level = 1
    elif act_work_condition['obj_speed'] <= 520:
        obj_speed_level = 2
    elif act_work_condition['obj_speed'] <= 780:
        obj_speed_level = 3
    else:
        obj_speed_level = 4
    runtime_str += 'obj_speed='
    runtime_str += str(obj_speed_level)
    runtime_str += ' '
    
    return runtime_str




# SchedulerInterface类
# 用途: 作为当前子目录下所有情境感知和调度功能的综合，对外界提供情境感知、调度计划的接口。外界只需要这一个类就足以实现情境感知、调度的全部功能。
# 方法: 调用当前子目录下其他情境感知等模块。
class AwareScheduler():

    CONTENT_ELE_MAXN = 50

    def __init__(self):

        self.runtime_portrait_dict = dict()
        self.runtime_cache_dict = dict() #缓存所有query最近CONTENT_ELE_MAXN种情境对应的字符串，每一个query_id对应一个列表
        self.scheduling_cache_dict = dict()  # 缓存所有query最近CONTENT_ELE_MAXN种情境采取的对应的调度策略，每一个query_id对应一个列表，与runtime_cache_dict中的一个列表一一对应

        pass


    # 根据query信息，选择初始化或更新内部情境信息
    def update_runtime(self,query,new_runtime):
        
        # 更新的时候首先确保字典中是否有一个这样的情境画像
        query_id = query.query_id

        # 如果原本字典中没有给这个query_id创建画像，现在是要重新增补的
        if query_id not in self.runtime_portrait_dict.keys():
            self.runtime_portrait_dict[query_id] = RuntimePortrait(query.pipeline, query.user_constraint, query.serv_cloud_addr)
        
        # 如果已经存在，就可以开始更新了
        self.runtime_portrait_dict[query_id].update_runtime(new_runtime)


    # 对外展示情境信息
    def get_portrait_info(self,query_id):
        
        runtime_portrait = self.runtime_portrait_dict[query_id]
        assert isinstance(runtime_portrait,RuntimePortrait)
        return runtime_portrait.get_portrait_info()


    # 检测情境是否缓存,如果缓存成功就提供一个缓存的策略
    def get_if_cached(self,query_id, runtime_str):

        if query_id not in self.runtime_cache_dict.keys():
            self.runtime_cache_dict[query_id]=[]
            self.scheduling_cache_dict[query_id]=[]
        
        if runtime_str in self.runtime_cache_dict[query_id]:
            temp_index = self.runtime_cache_dict[query_id].index(runtime_str)

            return {
                'if_cached': True,
                'scheduling_strategy': self.scheduling_cache_dict[query_id][temp_index]

            }
        else:
            return {
                'if_cached': False,
                'scheduling_strategy': None
            }
    
    # 缓存当前情境下的策略
    def cache_scheduling_strategy(self, query_id, runtime_str, scheduling_strategy_dict):
       
        if query_id not in self.runtime_cache_dict.keys():
            self.runtime_cache_dict[query_id]=[]
            self.scheduling_cache_dict[query_id]=[]
       
       
        self.runtime_cache_dict[query_id].append(runtime_str)

        if len(self.runtime_cache_dict[query_id]) > self.CONTENT_ELE_MAXN:
            del self.runtime_cache_dict[query_id][0]
        
        self.scheduling_cache_dict[query_id].append(scheduling_strategy_dict)
        if len(self.scheduling_cache_dict[query_id]) > self.CONTENT_ELE_MAXN:
            del self.scheduling_cache_dict[query_id][0]
        
        # 后续不需要什么set_plan的操作，那是query_manager_v2要做的事情



    # 制定调度计划
    # 需要参数:各个query的基本信息，bandwidth_dict带宽信息, 以及sysytem_status全体系统信息
    # 返回值:一个字典，包含了所有query_id对应的新调度计划
    def get_schedule_plans(self, query_dict, bandwidth_dict, system_status):

        schedule_plans_dict=dict()
        # 冷启动的时候，往往没有在self.runtime_portrait_dict里更新情境，所以要进行初始化，才能用于后续处理
        for query_id, query in query_dict.items():

            if query_id not in  self.runtime_portrait_dict.keys():
                self.runtime_portrait_dict[query_id] = RuntimePortrait(query.pipeline, query.user_constraint, query.serv_cloud_addr)

            runtime_portrait = self.runtime_portrait_dict[query_id]
     
            assert isinstance(runtime_portrait,RuntimePortrait)
            exec_work_condition = runtime_portrait.get_latest_work_condition() # 执行视频流分析任务获得的工况（未必等同于真实的工况）
            portrait_info = runtime_portrait.get_portrait_info() #获取当前任务的情境信息

            
                

            query = query_dict[query_id]
            root_logger.info("video_id:{}".format(query.video_id))
            node_addr = query.node_addr  # 形如：192.168.1.7:3001
            user_constraint = query.user_constraint
            assert node_addr

            # 只有当runtime_info不存在(视频流分析还未开始，进行冷启动)或者含有delay的时候(正常的视频流调度)才运行。
            if not exec_work_condition or 'delay' in exec_work_condition:
                assert node_addr in bandwidth_dict

                #(1)情况1：非冷启动
                if 'delay' in exec_work_condition:
                    act_work_condition = runtime_portrait.get_golden_conf_work_condition() # 执行黄金配置获得的真实工况
                    root_logger.info("Act_work_condition is:{}".format(act_work_condition))

                    runtime_str = get_runtime_str(portrait_info=portrait_info,
                                                  pipeline=query.pipeline,
                                                  bandwidth_dict=bandwidth_dict[node_addr],
                                                  act_work_condition=act_work_condition)
                    cached_info = self.get_if_cached(query_id=query_id,
                                                     runtime_str=runtime_str)
                    
                    # 说明：下面这个缓存命中机制有很大问题，因为过去可用的不代表现在还可用。
                    
                    if_try_cache = 0
                    if if_try_cache == 0:
                        print('当前没有启动缓存机制')
                    #(1.1)情况1.1 缓存命中
                    if cached_info['if_cached'] and if_try_cache:  # 若缓存了对应的调度策略，则直接使用该策略，无需重新查表
                        root_logger.info('当前情境下最优的调度策略保存在缓存中, 无需重新查表, runtime_str:%s', runtime_str)

                        # 更新到调度器内部的宏观变量中
                        lat_first_kb_muledge_wzl_1.prev_conf[query_id] = cached_info['scheduling_strategy']["video_conf"]
                        lat_first_kb_muledge_wzl_1.prev_flow_mapping[query_id] = cached_info['scheduling_strategy']["flow_mapping"]
                        lat_first_kb_muledge_wzl_1.prev_resource_limit[query_id] = cached_info['scheduling_strategy']["resource_limit"]

                        # 保存当前query对应结果
                        scheduling_dict = {
                                "video_conf":cached_info['scheduling_strategy']["video_conf"],
                                "flow_mapping": cached_info['scheduling_strategy']["flow_mapping"],
                                "resource_limit": cached_info['scheduling_strategy']["resource_limit"]
                            }
               
                        schedule_plans_dict[query_id] = scheduling_dict
                    
                    #(1.2)情况1.2 缓存不命中，此时需要搜索
                    else:
                        root_logger.info('当前情境下最优的调度策略不在缓存中, 需要重新查表, 并将新的情境字符串及其对应的调度策略进行缓存. runtime_str:%s', runtime_str)
                        start_time = time.time()
                        # 搜索之中，lat_first_kb_muledge_wzl_1.prev_conf等宏观变量已经得到更新
                        conf, flow_mapping, resource_limit = lat_first_kb_muledge_wzl_1.scheduler(
                                                                        job_uid=query_id,
                                                                        dag={"generator": "x", "flow": query.pipeline},
                                                                        system_status=system_status,
                                                                        portrait_info=portrait_info,
                                                                        user_constraint=user_constraint,
                                                                        bandwidth_dict=bandwidth_dict[node_addr],
                                                                        act_work_condition=act_work_condition)
                        end_time = time.time()
                        root_logger.info("调度策略制定的时间:{}".format(end_time - start_time))

                        scheduling_dict = {
                                    "video_conf": conf,
                                    "flow_mapping": flow_mapping,
                                    "resource_limit": resource_limit
                                }
                        # 缓存调度策略
                        self.cache_scheduling_strategy(query_id=query_id, 
                                                       runtime_str=runtime_str, 
                                                       scheduling_strategy_dict=scheduling_dict)
                        
                        # 保存当前query对应结果
                        schedule_plans_dict[query_id] = scheduling_dict
                
                #(2) 情况2：进行冷启动
                else:
                    root_logger.info('进行冷启动, 需要重新查表!')
                    start_time = time.time()
                    conf, flow_mapping, resource_limit = lat_first_kb_muledge_wzl_1.scheduler(
                                                                    job_uid=query_id,
                                                                    dag={"generator": "x", "flow": query.pipeline},
                                                                    system_status=system_status,
                                                                    portrait_info=portrait_info,
                                                                    user_constraint=user_constraint,
                                                                    bandwidth_dict=bandwidth_dict[node_addr])
                    end_time = time.time()
                    root_logger.info("冷启动调度策略制定的时间:{}".format(end_time - start_time))
                    # 冷启动的时候没有感知到情境，不适合进行缓存

                    scheduling_dict = {
                                "video_conf": conf,
                                "flow_mapping": flow_mapping,
                                "resource_limit": resource_limit
                            }
               
                    # 保存当前query对应结果
                    schedule_plans_dict[query_id] = scheduling_dict
            
            else:
                print('当前任务不处于冷启动也无法进行其他调度')
                # 没有任何情境信息。此时，在schedule_plans_dict中不存在一个相应的query_id对应的调度计划
                pass
        
        # 返回所有query_id对应的调度计划
        return schedule_plans_dict


                        
                        
                    
                        






            



        










        pass

        



        '''

        def update_runtime(self, runtime_info):
            self.runtime_portrait.update_runtime(runtime_info)

        def sync_query_runtime(self, query_id, new_runtime):
            assert query_id in self.query_dict

            query = self.query_dict[query_id]
            assert isinstance(query, Query)
            query.update_runtime(new_runtime)

        '''
