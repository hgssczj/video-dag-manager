SYNC_RESULT_KEY_APPEND = "appended_result"
SYNC_RESULT_KEY_LATEST = "latest_result"
SYNC_RESULT_KEY_PLAN = "ext_plan"
SYNC_RESULT_KEY_RUNTIME = "ext_runtime"

PLAN_KEY_VIDEO_CONF = "video_conf"
PLAN_KEY_FLOW_MAPPING = "flow_mapping"
PLAN_KEY_RESOURCE_LIMIT = "resource_limit"

KB_PLAN_PATH='kb_plan'
KB_DATA_PATH='kb_data'

NO_BAYES_GOAL = 0  # 按照遍历配置组合的方式来建立知识库
BEST_ALL_DELAY = 1  # 以最小化总时延为目标，基于贝叶斯优化建立知识库（密集，而不集中）
BEST_STD_DELAY = 2  # 以最小化不同配置间时延的差别为目标，基于贝叶斯优化建立知识库（稀疏，而均匀）

MAX_NUMBER=999999

COLD_START = -1
EXTREME_CASE = 0
IMPROVE_ACCURACY = 1
REASSIGN_RSC = 2


NO_CHANGE = 10  # 所有配置均不变（视频流配置、资源、云边协同切分点）

# 视频流配置相关
IMPROVE_CONF = 13  # 提升视频流配置，严格大于
DOWNGRADE_CONF = 3  # 降低视频流配置，严格小于
MAINTAIN_CONF = 15  # 维持视频流配置
MAINTAIN_OR_IMPROVE_CONF = 16  # 维持或提升视频流配置，大于等于

# 云边协同方式相关
MOVE_CLOUD = 4  # 挪到云端
BACK_TO_HOST = 11  # 将服务拉回到本地
EDGE_SIDE_COLLABORATION = 12  # 边边协同

# 资源分配相关
CHANGE_RSC_ALLOC_TO_CONS = 14  # 将服务的资源分配量更改为资源约束值，其他所有配置不变





resolution_wh = {
    "360p": {
        "w": 480,
        "h": 360
    },
    "480p": {
        "w": 640,
        "h": 480
    },
    "540p": {
        "w": 960,
        "h": 540
    },
    "630p": {
        "w": 1120,
        "h": 630
    },
    "720p": {
        "w": 1280,
        "h": 720
    },
    "810p": {
        "w": 1440,
        "h": 810
    },
    "900p": {
        "w": 1600,
        "h": 900
    },
    "990p": {
        "w": 1760,
        "h": 990
    },
    "1080p": {
        "w": 1920,
        "h": 1080
    }
}

reso_2_index_dict = {  # 分辨率映射为整数，便于构建训练数据
    "360p": 1,
    "480p": 2,
    "540p": 3,
    "630p": 4,
    "720p": 5,
    "810p": 6,
    "900p": 7,
    "990p": 8,
    "1080p": 9
}

fps_list = [i + 1 for i in range(30)]

# 以下是可能影响任务性能的可配置参数，用于指导模块化知识库的建立
model_op={  
            "114.212.81.11":{
                "model_id": 0,
                "node_ip": "114.212.81.11",
                "node_role": "cloud"
            },
            "192.168.1.7": {
                "model_id": 0,
                "node_ip": "192.168.1.7",
                "node_role": "host"  
            },
            "192.168.1.9": {
                "model_id": 0,
                "node_ip": "192.168.1.9",
                "node_role": "host"  
            },
        }

service_info_dict={
    'face_detection':{
        "name":'face_detection',
        "value":'face_detection_proc_delay',
        "conf":["reso","fps","encoder"],
        "vary_with_obj_n":False,
        "can_seek_accuracy":True
    },
    'gender_classification':{
        "name":'gender_classification',
        "value":'gender_classification_proc_delay',
        "conf":["reso","fps","encoder"],
        "vary_with_obj_n":True,
        "can_seek_accuracy":False
    }
}


# reso_range = list(resolution_wh.keys())  # "360p", "480p", "540p", "630p", "720p", "810p", "900p", "990p", "1080p"
reso_range = ['360p','480p','720p','1080p']  # '360p','480p','720p','1080p'

# fps_range = fps_list
fps_range = [1,5,10,20,30]

encoder_range=['JPEG']

ip_range=["192.168.1.9"]  # ["192.168.1.7","192.168.1.9","114.212.81.11"]
cloud_ip = "114.212.81.11"
edge_ip = "192.168.1.9"


serv_names = ['face_detection', 'gender_classification']

edge_cloud_cut_range = [i for i in range(len(serv_names) + 1)]

'''
mem_range=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,
           0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,
           0.021,0.022,0.023,0.024,0.025,0.026,0.027,0.028,0.029,0.03]
'''
mem_range=[1.0]


cpu_range=[0.05,0.10,0.15,0.20,
           0.25,0.30,0.35,0.40,
           0.45,0.50,0.55,0.60]

# cpu_range=[1.0]

conf_and_serv_info={  #各种配置参数的可选值
    
    "reso":reso_range,
    "fps":fps_range,
    "encoder":encoder_range,
    
    "face_detection_ip":ip_range,
    "gender_classification_ip":ip_range,  
    "edge_cloud_cut_point": edge_cloud_cut_range,

    "face_detection_mem_util_limit":mem_range,
    "face_detection_cpu_util_limit":cpu_range,
    "gender_classification_mem_util_limit":mem_range,
    "gender_classification_cpu_util_limit":cpu_range,

    "face_detection_trans_ip":ip_range,
    "gender_classification_trans_ip":ip_range,   

    "face_detection_trans_mem_util_limit":mem_range,
    "face_detection_trans_cpu_util_limit":cpu_range,
    "gender_classification_trans_mem_util_limit":mem_range,
    "gender_classification_trans_cpu_util_limit":cpu_range,

}
