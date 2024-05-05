from PortraitModel import PortraitModel
import torch
import common
import numpy as np
import requests
import field_codec_utils
import cv2
import math

class RuntimePortrait():
    CONTENT_ELE_MAXN = 50
    def __init__(self, pipeline):
        self.service_cloud_addr = "114.212.81.11:4500"
        self.sess = requests.Session()
        
        # 存储工况情境的字段
        self.runtime_pkg_list = dict()  # 此变量用于存储工况情境参数，key为工况类型'obj_n'、'obj_size'等；value为列表类型
        self.current_work_condition = dict()  # 此变量用于存储当前工况，key为工况类型'obj_n'、'obj_size'等
        self.work_condition_list = []  # 此变量用于存储历史工况，每一个元素为一个dict，字典的key为工况类型'obj_n'、'obj_size'等
        self.runtime_info_list = []  # 此变量用于存储完整的历史运行时情境信息，便于建立画像使用
        
        # 加载与当前pipeline对应的运行时情境画像模型
        self.pipeline = pipeline
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
    
