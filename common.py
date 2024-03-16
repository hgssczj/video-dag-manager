SYNC_RESULT_KEY_APPEND = "appended_result"
SYNC_RESULT_KEY_LATEST = "latest_result"
SYNC_RESULT_KEY_PLAN = "ext_plan"
SYNC_RESULT_KEY_RUNTIME = "ext_runtime"

PLAN_KEY_VIDEO_CONF = "video_conf"
PLAN_KEY_FLOW_MAPPING = "flow_mapping"
PLAN_KEY_RESOURCE_LIMIT = "resource_limit"

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