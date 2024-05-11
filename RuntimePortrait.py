from PortraitModel import PortraitModel
import torch
import common
import numpy as np
import requests
import field_codec_utils
import cv2
import math
from AccuracyPrediction import AccuracyPrediction
import DelayPredictModel

class RuntimePortrait():
    CONTENT_ELE_MAXN = 50
    def __init__(self, pipeline, user_constraint):
        self.service_cloud_addr = "114.212.81.11:4500"
        self.sess = requests.Session()
        
        # 存储工况情境的字段
        self.runtime_pkg_list = dict()  # 此变量用于存储工况情境参数，key为工况类型'obj_n'、'obj_size'等；value为列表类型
        self.current_work_condition = dict()  # 此变量用于存储当前工况，key为工况类型'obj_n'、'obj_size'等
        self.work_condition_list = []  # 此变量用于存储历史工况，每一个元素为一个dict，字典的key为工况类型'obj_n'、'obj_size'等
        self.runtime_info_list = []  # 此变量用于存储完整的历史运行时情境信息，便于建立画像使用
        self.slide_win_size = 15  # 构建画像的滑动窗口为：前一调度周期内最多15帧的结果
        self.scheduling_info_list = []  # 存放每一次执行结果对应的调度策略字符串，用于在构建画像时确定滑动窗口
        
        # 加载与当前pipeline对应的运行时情境画像模型
        self.pipeline = pipeline
        self.user_constraint = user_constraint
        self.portrait_model_dict = dict()  # 此变量用于保存pipeline中各个服务的画像预估模型
        self.delay_predictor = DelayPredictModel.DelayPredictor(self.pipeline)  # 时延预估器，预估pipeline中各个服务在指定配置、指定工况、资源充足时的执行时延和传输时延
        self.acc_predictor = AccuracyPrediction()  # 精度预估器
        
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
        self.server_total_mem = 270110633984
        self.edge_total_mem = 8239902720
        
    
    def update_runtime(self, runtime_info):
        # 更新云端的运行时情境信息
        
        # 1.保存完整的运行时情境参数，为调度器查表提供参考
        self.runtime_info_list.append(runtime_info)
        # 避免保存过多的内容导致爆内存
        if len(self.runtime_info_list) > RuntimePortrait.CONTENT_ELE_MAXN:
            del self.runtime_info_list[0]
        
        # 2.更新工况信息，便于前端展示(绘制折线图等)
        self.update_work_condition(runtime_info)
        
        # 3.更新调度信息，便于获取画像时确定滑动窗口
        assert 'exe_plan' in runtime_info
        self.update_scheduling_info(runtime_info['exe_plan'])

    def update_scheduling_info(self, exe_plan):
        # 1.根据当前调度策略生成字符串
        scheduling_info_str = ''
        
        assert 'video_conf' in exe_plan
        for conf_key, conf_value in exe_plan['video_conf'].items():
            scheduling_info_str += conf_key
            scheduling_info_str += '_'
            scheduling_info_str += str(conf_value)
        scheduling_info_str += '_'
        
        assert 'flow_mapping' in exe_plan
        for serv_name, node_info_dict in exe_plan['flow_mapping'].items():
            scheduling_info_str += serv_name
            scheduling_info_str += '_'
            scheduling_info_str += node_info_dict['node_ip']
        scheduling_info_str += '_'
        
        assert 'resource_limit' in exe_plan
        for serv_name, resource_limit_dict in exe_plan['resource_limit'].items():
            scheduling_info_str += serv_name
            for resource_type, resource_value in resource_limit_dict.items():
                scheduling_info_str += '_'
                scheduling_info_str += resource_type
                scheduling_info_str += '_'
                scheduling_info_str += str(resource_value)
    
        
        # 2.保存当前调度策略字符串
        self.scheduling_info_list.append(scheduling_info_str)
        if len(self.scheduling_info_list) > RuntimePortrait.CONTENT_ELE_MAXN:
            del self.scheduling_info_list[0]
        
        # 确保调度策略保存的数量与画像基础信息保存的数量相同，确保在构建画像时二者是对应的
        assert len(self.scheduling_info_list) == len(self.runtime_info_list)
    
    def update_work_condition(self, runtime_info):
        # 更新工况信息，便于前端展示(绘制折线图等)
        for taskname in runtime_info:
            if taskname == 'end_pipe':
                if 'delay' not in self.runtime_pkg_list:
                    self.runtime_pkg_list['delay'] = list()

                self.runtime_pkg_list['delay'].append(runtime_info[taskname]['delay'])
                if len(self.runtime_pkg_list['delay']) > RuntimePortrait.CONTENT_ELE_MAXN:
                    del self.runtime_pkg_list['delay'][0]
                

            # 对face_detection的结果，提取运行时情境
            # TODO：目标数量、目标大小、目标速度
            if taskname == 'face_detection':
                # 定义运行时情境字段
                if 'obj_n' not in self.runtime_pkg_list:
                    self.runtime_pkg_list['obj_n'] = list()
                if 'obj_size' not in self.runtime_pkg_list:
                    self.runtime_pkg_list['obj_size'] = list()
                if 'obj_speed' not in self.runtime_pkg_list:
                    self.runtime_pkg_list['obj_speed'] = list()

                ##### 1.获取当前目标数量
                self.runtime_pkg_list['obj_n'].append(len(runtime_info[taskname]['faces']))
                if len(self.runtime_pkg_list['obj_n']) > RuntimePortrait.CONTENT_ELE_MAXN:
                    del self.runtime_pkg_list['obj_n'][0]
                

                ##### 2.获取当前目标大小
                reso = runtime_info['exe_plan']['video_conf']['reso']
                
                obj_size = 0
                for x_min, y_min, x_max, y_max in runtime_info[taskname]['bbox']:
                    # 将目标大小统一转换为1080p下的大小
                    x_min_1 = x_min / common.resolution_wh[reso]['w'] * 1920
                    y_min_1 = y_min / common.resolution_wh[reso]['h'] * 1080
                    x_max_1 = x_max / common.resolution_wh[reso]['w'] * 1920
                    y_max_1 = y_max / common.resolution_wh[reso]['h'] * 1080
                    obj_size += (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
                if len(runtime_info[taskname]['bbox']) > 0:
                    obj_size /= len(runtime_info[taskname]['bbox'])

                self.runtime_pkg_list['obj_size'].append(obj_size)
                if len(self.runtime_pkg_list['obj_size']) > RuntimePortrait.CONTENT_ELE_MAXN:
                    del self.runtime_pkg_list['obj_size'][0]
                
                
                ##### 3.获取当前目标速度
                cap_fps = runtime_info['cap_fps']
                
                if len(self.runtime_info_list) == 1 or len(runtime_info[taskname]['bbox']) == 0:  # 目前没有之前的结果则无法计算两帧之间的移动，也就无法计算速度；当前帧没有检测到人脸，也无法计算速度
                    obj_speed = 314  # 设置为默认速度
                else:
                    pre_frame = field_codec_utils.decode_image(self.runtime_info_list[-2]['frame'])
                    pre_bbox = self.runtime_info_list[-2][taskname]['bbox']
                    pre_frame_id = self.runtime_info_list[-2]['end_pipe']["frame_id"]
                    
                    if len(pre_bbox) == 0:  # 前一帧也没有目标，无法计算速度
                         obj_speed = 314  # 设置为默认速度
                    else:
                        cur_frame = field_codec_utils.decode_image(runtime_info['frame'])
                        cur_bbox = runtime_info[taskname]['bbox']
                        cur_frame_id = runtime_info['end_pipe']["frame_id"]
                        
                        obj_speed = self.cal_obj_speed(pre_frame, pre_bbox, pre_frame_id, cur_frame, cur_bbox, cur_frame_id, cap_fps)
                    
                self.runtime_pkg_list['obj_speed'].append(obj_speed)
                if len(self.runtime_pkg_list['obj_speed']) > RuntimePortrait.CONTENT_ELE_MAXN:
                    del self.runtime_pkg_list['obj_speed'][0]

                # 确保各类工况保存的值数量相同，进而确保同一索引下的各类工况对应的视频帧相同，以免其他模块获取工况时出现不匹配的情况
                assert len(self.runtime_pkg_list['obj_n']) == len(self.runtime_pkg_list['obj_size']) == len(self.runtime_pkg_list['obj_speed'])
                # 确保工况保存的数量与画像保存的数量相同，确保调度器在获取最新的工况和画像时二者是对应的
                assert len(self.runtime_pkg_list['obj_n']) == len(self.runtime_info_list)
                
    def cal_obj_speed(self, pre_frame, pre_bbox, pre_frame_id, cur_frame, cur_bbox, cur_frame_id, cap_fps):
        # 首先将两帧都转换为1080p的图像，且将两帧的bbox也转化为1080p下的坐标
        pre_frame_1 = cv2.resize(pre_frame, (1920, 1080))
        pre_bbox_1 = []
        for i in range(len(pre_bbox)):
            x_min = pre_bbox[i][0] / pre_frame.shape[1] * 1920
            y_min = pre_bbox[i][1] / pre_frame.shape[0] * 1080
            x_max = pre_bbox[i][2] / pre_frame.shape[1] * 1920
            y_max = pre_bbox[i][3] / pre_frame.shape[0] * 1080
            pre_bbox_1.append([int(x_min), int(y_min), int(x_max), int(y_max)])
        
        cur_frame_1 = cv2.resize(cur_frame, (1920, 1080))
        cur_bbox_1 = []
        for i in range(len(cur_bbox)):
            x_min = cur_bbox[i][0] / cur_frame.shape[1] * 1920
            y_min = cur_bbox[i][1] / cur_frame.shape[0] * 1080
            x_max = cur_bbox[i][2] / cur_frame.shape[1] * 1920
            y_max = cur_bbox[i][3] / cur_frame.shape[0] * 1080
            cur_bbox_1.append([int(x_min), int(y_min), int(x_max), int(y_max)])
        
        
        # 利用tracking+匹配的方式计算目标速度
        speed_list = []
        pre_frame_tracker_list = []
        for temp_box in pre_bbox_1:
            temp_x_min, temp_y_min, temp_x_max, temp_y_max = temp_box
            temp_bbox = (temp_x_min, temp_y_min, temp_x_max - temp_x_min, temp_y_max - temp_y_min)
            
            temp_tracker = cv2.TrackerCSRT_create()  # 对每一个人脸都创建一个tracker
            ok = temp_tracker.init(pre_frame_1, temp_bbox)
         
            pre_frame_tracker_list.append(temp_tracker)
        
        for i in range(len(pre_bbox_1)):
            pre_frame_bbox = pre_bbox_1[i]
            pre_x_min, pre_y_min, pre_x_max, pre_y_max = pre_frame_bbox  # 前一帧中目标的bbox
            
            ok, track_bbox = pre_frame_tracker_list[i].update(cur_frame_1)  # 利用tracker将前一帧中的目标在当前帧中移动
            
            if ok:  # 如果追踪成功，则将追踪之后bbox与当前帧真实的所有目标的bbox进行配对（利用iou），从而实现两帧之间目标的匹配
                track_x_min = int(track_bbox[0])
                track_y_min = int(track_bbox[1])
                track_x_max = int(track_bbox[0] + track_bbox[2])
                track_y_max = int(track_bbox[1] + track_bbox[3])
                
                temp_iou_list = []
                for temp_box in cur_bbox_1:
                    temp_iou = self.cal_iou([track_x_min, track_y_min, track_x_max, track_y_max], temp_box)
                    temp_iou_list.append(temp_iou)
                
                obj_index = np.argmax(np.array(temp_iou_list))  # 与track之后的bbox重叠程度最大的bbox为该目标在当前帧中的bbox
                
                if temp_iou_list[obj_index] >= 0.1:  # track之后的bbox与真实的bbox之间一定要满足一定程度的重叠，否则认为是误判
                    # 此时说明完成了前一帧中的目标和当前帧中的目标之间的匹配，可以计算速度
                    temp_box = cur_bbox_1[obj_index]
                    temp_center = ((temp_box[0] + temp_box[2]) / 2, (temp_box[1] + temp_box[3]) / 2)
                    pre_center = ((pre_x_min + pre_x_max) / 2, (pre_y_min + pre_y_max) / 2)
                    
                    temp_speed_x = math.fabs((temp_center[0] - pre_center[0])) / (cur_frame_id - pre_frame_id) * cap_fps
                    temp_speed_y = math.fabs((temp_center[1] - pre_center[1])) / (cur_frame_id - pre_frame_id) * cap_fps
                    temp_speed = (temp_speed_x ** 2 + temp_speed_y ** 2) ** 0.5
                    
                    speed_list.append(temp_speed)
        
        if len(speed_list) == 0:
            return 0
        return np.max(speed_list)  # 返回速度的最大值，因为最大值决定帧率
            
    def cal_iou(self, predict_bbox, gt_bbox):
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
                
    def aggregate_work_condition(self):
        # TODO：聚合情境感知参数的时间序列，给出预估值/统计值
        runtime_desc = dict()
        for k, v in self.runtime_pkg_list.items():
            if len(v) > 0:
                runtime_desc[k] = sum(v) * 1.0 / len(v)
            else:
                runtime_desc[k] = sum(v)

        # 获取场景稳定性
        if 'obj_n' in self.runtime_pkg_list.keys():
            runtime_desc['obj_stable'] = True if np.std(self.runtime_pkg_list['obj_n']) < 0.3 else False

        # 每次调用agg后清空
        # self.runtime_pkg_list = dict()
        
        return runtime_desc

    def get_latest_work_condition(self):
        latest_work_condition = dict()
        for k, v in self.runtime_pkg_list.items():
            if len(v) > 0:
                latest_work_condition[k] = self.runtime_pkg_list[k][-1]
        
        return latest_work_condition
                
    def set_work_condition(self, new_work_condition):
        while len(self.work_condition_list) >= RuntimePortrait.CONTENT_ELE_MAXN:
            print("len(self.work_condition_list)={}".format(len(self.work_condition_list)))
            del self.work_condition_list[0]
        self.work_condition_list.append(self.current_work_condition)
        self.current_work_condition = new_work_condition
    
    def get_aggregate_work_condition(self):
        new_work_condition = self.aggregate_work_condition()
        assert isinstance(new_work_condition, dict)
        if new_work_condition:  # 若new_work_condition非空，则更新current_work_condition；否则保持current_work_condition
            self.set_work_condition(new_work_condition)
        return self.current_work_condition

    '''
    def get_portrait_info(self):
        portrait_info = dict()
        
        if len(self.runtime_info_list) == 0:  # 若self.runtime_info_list为空，则说明目前云端没有存储任何运行结果，无法给出画像的信息
            return portrait_info
        
        cur_runtime_info = self.runtime_info_list[-1]  # 以最近的运行时情境为依据获取画像信息
        assert(isinstance(cur_runtime_info, dict))
        
        ###### 1. 判断时延是否满足约束
        assert('end_pipe' in cur_runtime_info)
        cur_latency = cur_runtime_info['end_pipe']['delay']
        cur_process_latency = cur_runtime_info['end_pipe']['process_delay']
        assert('user_constraint' in cur_runtime_info)
        cur_user_latency_constraint = cur_runtime_info['user_constraint']['delay']
        if_overtime = True if cur_latency > cur_user_latency_constraint else False
        
        portrait_info['cur_latency'] = cur_latency
        portrait_info['cur_process_latency'] = cur_process_latency
        portrait_info['user_constraint'] = cur_runtime_info['user_constraint']
        portrait_info['if_overtime'] = if_overtime
        
        
        ###### 2. 获取当前系统中每个设备上本query可以使用的资源量
        r = self.sess.get(url="http://{}/get_cluster_info".format(self.service_cloud_addr))
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
            #print("In Query get_portrait_info(), resource_info:{}".format(resource_info))
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
                
        
        ###### 4. 其他信息
        portrait_info['bandwidth'] = cur_runtime_info['bandwidth']  # 云边之间的带宽
        portrait_info['data_to_cloud'] = cur_runtime_info['data_to_cloud']  # 云边之间的数据传输量
        portrait_info['exe_plan'] = cur_runtime_info['exe_plan']
        portrait_info['data_trans_size'] = cur_runtime_info['data_trans_size']  # 各个服务输入和输出的数据量
        portrait_info['frame'] = cur_runtime_info['frame']
        portrait_info['process_delay'] = cur_runtime_info['process_delay']
        
        return portrait_info
    '''
    
    def get_portrait_info(self):
        portrait_info = dict()
        
        if len(self.runtime_info_list) == 0:  # 若self.runtime_info_list为空，则说明目前云端没有存储任何运行结果，无法给出画像的信息
            return portrait_info

        cur_runtime_info = self.runtime_info_list[-1]  # 以最近的运行时情境为依据获取画像信息
        assert(isinstance(cur_runtime_info, dict))
        
        ###################### 1.获取滑动窗口内的工况等情境信息 ######################
        ########### 1.1 确定滑动窗口范围 ###########
        right_index = len(self.scheduling_info_list) - 1  # 滑动窗口的左右索引
        left_index = right_index
        count = 0
        while left_index >= 0 and count <= self.slide_win_size and self.scheduling_info_list[right_index] == self.scheduling_info_list[left_index]:
            left_index -= 1
            count += 1
        
        left_index += 1
        assert left_index >= 0 and right_index >= left_index
        
        ########### 1.2 获取滑动窗口内情境信息 ###########
        sw_runtime_dict = dict()  # 保存滑动窗口内情境信息
        for k, v in self.runtime_pkg_list.items():
            v1 = v[left_index: right_index + 1]
            if len(v1) > 0:
                sw_runtime_dict[k] = sum(v1) * 1.0 / len(v1)
            else:
                sw_runtime_dict[k] = sum(v1)
        portrait_info['work_condition'] = sw_runtime_dict
        
        
        ###################### 2.获取画像类别 ######################
        ########### 2.1 确定配置画像 ###########
        pre_video_conf = cur_runtime_info['exe_plan']['video_conf']  # 前一调度周期内的视频流配置
        
        assert 'obj_size' in sw_runtime_dict and 'obj_speed' in sw_runtime_dict
        sw_obj_size = int(sw_runtime_dict['obj_size'])
        sw_obj_speed = int(sw_runtime_dict['obj_speed'])
        acc_predicted = self.acc_predictor.predict(self.pipeline[0], pre_video_conf, sw_obj_size, sw_obj_speed)  # 在当前工况、当前配置下预测的精度
        
        acc_constraint = self.user_constraint['accuracy']  # 精度约束
        assert acc_constraint <= 0.95
        
        if acc_predicted < acc_constraint:
            portrait_info['conf_portrait'] = 0  # 低于精度约束，则配置画像类别为弱
        # elif acc_predicted - acc_constraint <= 0.05:
        #     portrait_info['conf_portrait'] = 1  # 高于精度约束且只高于精度约束一点，则配置画像类别为中
            # TODO:目前直接使用一个小的固定阈值作为中配置的精度范围；
            # 若老板不认可，则可以按照以下思路实现：假设有一批配置，这些配置都满足精度约束，但这些配置如果降低一档其中某种具体的配置，那么就不再满足精度约束，将这批配置中最大的精度作为中配置的精度上限
        else:
            portrait_info['conf_portrait'] = 3  # 高于精度约束，则配置画像类别为强或中，具体类别放到调度罗盘模块进行判定。这是因为强和中的区分需要通过很多细粒度的判断，画像只是为了给出粗略的判断，要尽量避免在画像的部分进行很多细粒度的操作
        
        
        ########### 2.2 确定资源画像 ###########
        delay_constraint = self.user_constraint['delay']  # 时延约束
        pre_flow_mapping = cur_runtime_info['exe_plan']['flow_mapping']
        
        ##### 2.2.1 确定当前工况、当前配置在理想资源情况下的时延 #####
        ideal_proc_delay = 0
        ideal_trans_delay = 0
        
        for service in self.pipeline:
            temp_fps = pre_video_conf['fps']
            temp_node_role = pre_flow_mapping[service]['node_role']
            temp_node_role = 'server' if temp_node_role == 'cloud' else 'edge'
            
            if service == 'face_detection':
                temp_reso = pre_video_conf['reso']
                temp_proc_delay = self.delay_predictor.predict({
                    'delay_type': 'proc_delay',
                    'predict_info': {
                        'service_name': service,  
                        'fps': temp_fps,  
                        'reso': temp_reso,
                        'node_role': temp_node_role
                    }
                })
                ideal_proc_delay += temp_proc_delay
                
            elif service == 'gender_classification':
                temp_obj_n = self.runtime_pkg_list['obj_n'][-1]
                
                temp_proc_delay = self.delay_predictor.predict({
                    'delay_type': 'proc_delay',
                    'predict_info': {
                        'service_name': service,  
                        'fps': temp_fps,  
                        'obj_n': temp_obj_n,
                        'node_role': temp_node_role
                    }
                })
                ideal_proc_delay += temp_proc_delay
            
            if temp_node_role == 'server':
                temp_trans_data_size = self.runtime_info_list[-1]['data_trans_size'][service]
                assert temp_trans_data_size != 0
                ideal_trans_delay += self.delay_predictor.predict({
                        'delay_type': 'trans_delay',
                        'predict_info': {
                            'fps': pre_video_conf['fps'],  
                            'trans_data_size': temp_trans_data_size,
                        }
                    })
        
        ideal_delay = ideal_proc_delay + ideal_trans_delay
        
        ##### 2.2.2 确定当前实际情况下的时延 #####
        act_proc_delay = 0
        act_trans_delay = 0
        
        for service in self.pipeline:
            act_proc_delay += self.runtime_info_list[-1]['process_delay'][service]
            
            temp_node_role = pre_flow_mapping[service]['node_role']
            if temp_node_role == 'cloud':
                act_trans_delay += (self.runtime_info_list[-1]['delay'][service] - self.runtime_info_list[-1]['process_delay'][service])
        
        act_delay = act_proc_delay + act_trans_delay
        
        ##### 2.2.3 确定资源画像类别 #####
        if act_delay >= 1.02 * ideal_delay:
            portrait_info['resource_portrait'] = 0  # 若实际执行时延高于理想时延，则资源画像为弱
        else:
            portrait_info['resource_portrait'] = 3  # 若实际执行时延等于理想时延，则资源画像为中或强，这一点在画像中无法具体判断，需要在中间模块中进一步判断
        
        
        ###### 3. 判断时延是否满足约束
        assert('end_pipe' in cur_runtime_info)
        cur_latency = cur_runtime_info['end_pipe']['delay']
        cur_process_latency = cur_runtime_info['end_pipe']['process_delay']
        assert('user_constraint' in cur_runtime_info)
        cur_user_latency_constraint = cur_runtime_info['user_constraint']['delay']
        if_overtime = True if cur_latency > cur_user_latency_constraint else False
        
        ###### 4. 获取当前系统中每个设备上本query可以使用的资源量
        r = self.sess.get(url="http://{}/get_cluster_info".format(self.service_cloud_addr))
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
            #print("In Query get_portrait_info(), resource_info:{}".format(resource_info))
            if resource_info[ip_addr]['node_role'] == 'cloud':
                self.server_total_mem = resource_info[ip_addr]['device_state']['mem_total'] * 1024 * 1024 * 1024
            else:
                self.edge_total_mem = resource_info[ip_addr]['device_state']['mem_total'] * 1024 * 1024 * 1024
        
        assert(self.server_total_mem is not None)
        assert(self.edge_total_mem is not None)
        
        
        ###### 5. 获取当前query中各个服务的资源信息
        portrait_info['resource_info'] = dict()
        for service in self.pipeline:
            # 获取当前服务的执行节点
            portrait_info['resource_info'][service] = dict()
            portrait_info['resource_info'][service]['node_ip'] = cur_runtime_info[service]['proc_resource_info']['node_ip']
            portrait_info['resource_info'][service]['node_role'] = cur_runtime_info[service]['proc_resource_info']['node_role']
            
            # 获取当前服务的资源限制和实际资源使用量
            temp_cpu_limit = cur_runtime_info[service]['proc_resource_info']['cpu_util_limit']
            temp_cpu_use = cur_runtime_info[service]['proc_resource_info']['cpu_util_use']
            temp_mem_limit = cur_runtime_info[service]['proc_resource_info']['mem_util_limit']
            temp_mem_use = cur_runtime_info[service]['proc_resource_info']['mem_util_use']
            portrait_info['resource_info'][service]['cpu_util_limit'] = temp_cpu_limit
            portrait_info['resource_info'][service]['cpu_util_use'] = temp_cpu_use
            portrait_info['resource_info'][service]['mem_util_limit'] = temp_mem_limit
            portrait_info['resource_info'][service]['mem_util_use'] = temp_mem_use
            
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
            portrait_info['resource_info'][service]['resource_demand'] = temp_resource_demand
        
        ###### 6. 其他信息
        portrait_info['bandwidth'] = cur_runtime_info['bandwidth']  # 云边之间的带宽
        portrait_info['data_to_cloud'] = cur_runtime_info['data_to_cloud']  # 云边之间的数据传输量
        portrait_info['exe_plan'] = cur_runtime_info['exe_plan']
        portrait_info['data_trans_size'] = cur_runtime_info['data_trans_size']  # 各个服务输入和输出的数据量
        portrait_info['frame'] = cur_runtime_info['frame']
        portrait_info['process_delay'] = cur_runtime_info['process_delay']
        portrait_info['delay'] = cur_runtime_info['delay']
        
      
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
    
