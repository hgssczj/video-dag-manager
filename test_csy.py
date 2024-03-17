#彭师兄同步到边

from logging_utils import root_logger
from query_manager_v2 import Query

if __name__ == "__main__":

    query_id='1'
    node_addr=''
    video_id=1
    pipeline=['face_detection']
    user_constraint={}


    query0 = Query(query_id=query_id,
                node_addr=node_addr,
                video_id=video_id,
                pipeline=pipeline,
                user_constraint=user_constraint)
    
    #不能这样测试，必须得实际运行

    ans1=query0.predict_resource_threshold(
            task_info={
                # 配置相关字段
                'service_name': 'face_detection', # 服务名
                'fps': 15,  # 帧率，取值范围见common.py中变量fps_list
                'reso': 2,  # 分辨率，整数，表示分辨率的字符串到整数的映射见common.py中变量reso_2_index_dict
                # 工况相关字段
                'obj_num': 3  # 目标数量
            }
        )
    print(ans1)

    ans2=query0.help_cold_start(
        service='face_detection'
    )
    print(ans2)



    print("test")