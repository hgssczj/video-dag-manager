#彭师兄同步到边
'''
from logging_utils import root_logger
from query_manager_v2 import Query
from kb_builder import KnowledgeBaseBuilder
import json

from scheduler_func import lat_first_kb_muledge
from common import conf_and_serv_info
'''
if __name__ == "__main__":

    #此处函数用于验证调度器的行为
    import socket  
    print('开始测试5')
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
    print('当前内网ip是:')
    print(get_ip_address())
 
    '''
    job_uid=1
    dag={}
    dag["flow"]= ["face_detection", "gender_classification"]
   
    user_constraint={}
    user_constraint["delay"]= 0.3
    appended_result_list=list([])
     # （1）设置当前工况
    work_condition=dict({
        "obj_n": 1
    })
    # （2） 设置当前处理时延，一般只需要模拟超时的情况
    ext_runtime=dict({
                    'plan_result': 
                        {
                        'process_delay': {'face_detection': 1, 'gender_classification':1 }
                        }, 
                    })
    appended_result=dict({'ext_runtime':ext_runtime})
    appended_result_list.append(appended_result)
    # （3）设置上一阶段采取的配置
    lat_first_kb_muledge.prev_conf[job_uid]=dict({
        "reso": "360p", "fps": 30, "encoder": "JPEG", 
    })
    lat_first_kb_muledge.prev_flow_mapping[job_uid]=dict({
        "face_detection": {"model_id": 0, "node_ip": "172.27.143.164", "node_role": "host"}, 
        "gender_classification": {"model_id": 0, "node_ip": "172.27.143.164", "node_role": "host"}
    })
    lat_first_kb_muledge.prev_resource_limit[job_uid]=dict({
        "face_detection": {"cpu_util_limit": 0.2, "mem_util_limit": 0.004}, 
        "gender_classification": {"cpu_util_limit": 0.7, "mem_util_limit": 0.008}
    })
    
    lat_first_kb_muledge.scheduler(
        job_uid=job_uid,
        dag=dag,
        system_status=None,
        work_condition=work_condition,
        portrait_info=None,
        user_constraint=user_constraint,
        appended_result_list=appended_result_list
    )
    '''
