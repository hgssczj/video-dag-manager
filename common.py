SYNC_RESULT_KEY_APPEND = "appended_result"
SYNC_RESULT_KEY_LATEST = "latest_result"
SYNC_RESULT_KEY_PLAN = "ext_plan"
SYNC_RESULT_KEY_RUNTIME = "ext_runtime"

PLAN_KEY_VIDEO_CONF = "video_conf"
PLAN_KEY_FLOW_MAPPING = "flow_mapping"
PLAN_KEY_RESOURCE_LIMIT = "resource_limit"

KB_PLAN_PATH='kb_plan'
KB_DATA_PATH='kb_data'

NO_BAYES_GOAL=0 #按照遍历配置组合的方式来建立知识库
BEST_ALL_DELAY=1 #以最小化总时延为目标，基于贝叶斯优化建立知识库（密集，而不集中）
BEST_STD_DELAY=2 #以最小化不同配置间时延的差别为目标，基于贝叶斯优化建立知识库（稀疏，而均匀）

MAX_NUMBER=999999

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
            "172.27.143.164": {
                "model_id": 0,
                "node_ip": "172.27.143.164",
                "node_role": "host"  
            },
            "172.27.132.253": {
                "model_id": 0,
                "node_ip": "172.27.132.253",
                "node_role": "host"  
            },
            "172.27.151.145": {
                "model_id": 0,
                "node_ip": "172.27.151.145",
                "node_role": "host"  
            },

        }

service_info_dict={
    'face_detection':{
        "name":'face_detection',
        "value":'face_detection_proc_delay',
        "conf":["reso","fps","encoder"]
    },
    'gender_classification':{
        "name":'gender_classification',
        "value":'gender_classification_proc_delay',
        "conf":["reso","fps","encoder"]
    },
}

conf_and_serv_info={  #各种配置参数的可选值
    
    "reso":["360p", "480p", "720p", "1080p"],
    "fps":[1, 5, 10, 20, 30],
    "encoder":["JPEG"],
    
    #"face_detection_ip":["172.27.143.164"],
    #"gender_classification_ip":["172.27.143.164"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["114.212.81.11"],
    "gender_classification_ip":["114.212.81.11"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
   
    # 注意贝叶斯优化采样的时候，资源约束必须包含1.0，因为我默认放到云端的服务都采用1.0作为约束。
    "face_detection_mem_util_limit":[1.0,0.015,0.014,0.013,0.012,0.011,0.010,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001],
    "face_detection_cpu_util_limit":[1.0,0.05,0.10,0.15,0.20,0.25],
    "gender_classification_mem_util_limit":[1.0,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001],
    "gender_classification_cpu_util_limit":[1.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60],

    
    #"face_detection_trans_ip":["172.27.143.164"],
    #"gender_classification_trans_ip":["172.27.143.164"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["114.212.81.11"],
    "gender_classification_trans_ip":["114.212.81.11"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    
    
    "face_detection_trans_mem_util_limit":[1.0,0.015,0.014,0.013,0.012,0.011,0.010,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001],
    "face_detection_trans_cpu_util_limit":[1.0,0.05,0.10,0.15,0.20,0.25],
    "gender_classification_trans_mem_util_limit":[1.0,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001],
    "gender_classification_trans_cpu_util_limit":[1.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60],

}
#'''