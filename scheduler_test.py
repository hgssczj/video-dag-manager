
from scheduler_func import lat_first_kb_muledge
import json
from kb_builder import conf_and_serv_info

    # 以下是缩小范围版，节省知识库大小
conf_and_serv_info_sampled={  #各种配置参数的可选值
    "reso":["360p","480p","720p","1080p"],
    "fps":[1, 5, 10, 20, 30],
    "encoder":["JPEG"],
    
    "face_alignment_ip":["114.212.81.11","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_ip":["114.212.81.11","172.27.151.145"],
    "car_detection_ip":["114.212.81.11","172.27.151.145"],
    "face_alignment_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_alignment_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],

    "face_alignment_trans_ip":["114.212.81.11","172.27.151.145"],   #这个未来一定要修改成各个模型，比如model1，model2等;或者各个ip
    "face_detection_trans_ip":["114.212.81.11","172.27.151.145"],
    "car_detection_trans_ip":["114.212.81.11","172.27.151.145"],
    "face_alignment_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_alignment_trans_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "face_detection_trans_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_trans_mem_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],
    "car_detection_trans_cpu_util_limit":[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00],

}


if __name__ == "__main__":

    # 要解决贝叶斯优化耗时太长的问题，就要尽可能缩小配置的已有取值范围。也就是说，在用贝叶斯优化搜索知识库进而冷启动的时候，需要自己指定一个conf_serv_info
    # 而且，在利用新的文件更新字典的时候，也需要更新conf_serv_info。
    # 为了方便起见，
    '''
    import pickle

    my_list = [0.05, 2.8, 3.1, 'a', 'b']
    file_path = "data.pkl" # 指定保存文件路径及名称
    with open(file_path, 'wb') as file:
        pickle.dump(my_list, file)
    print("列表已保存到文件")

    file_path = "data.pkl" # 指定保存文件路径及名称
    with open(file_path, 'rb') as file:
        my_list = pickle.load(file)
    print("从文件中读取的列表：", my_list)
    print(type(my_list))
    for x in my_list:
        print(type(x),x)
    '''
    '''
    with open('conf_and_serv_info.json', 'w', encoding='utf-8') as f:
        json.dump(conf_and_serv_info_sampled, f, ensure_ascii=False, indent=4)
    with open('conf_and_serv_info.json', 'r', encoding='utf-8') as f:
        loaded_conf_and_serv_info = json.load(f)

        # 打印读取到的字典以验证内容
        print("从文件中读取的JSON对象：")
        print(type(loaded_conf_and_serv_info))
        print(loaded_conf_and_serv_info)
        for x in loaded_conf_and_serv_info.keys():
            y=loaded_conf_and_serv_info[x]
            print(type(y),type(y[0]))
            print(y)
    '''



    #'''
    job_uid=1
    dag={}
    dag["flow"]= ["face_detection", "face_alignment"]
    user_constraint={}
    user_constraint["delay"]= 0.3


    #conf_and_serv_info["face_alignment_mem_util_limit"]=[0.25,0.3,0.35,0.4]
    #conf_and_serv_info["face_detection_mem_util_limit"]=[0.25]
    #conf_and_serv_info["face_alignment_cpu_util_limit"]=[0.3]
    #conf_and_serv_info["face_detection_cpu_util_limit"]=[0.25]

  


    lat_first_kb_muledge.get_cold_start_plan(
            job_uid=1,
            dag=dag,
            resource_info=None,
            runtime_info=None,
            user_constraint=user_constraint,
        )
    #'''
