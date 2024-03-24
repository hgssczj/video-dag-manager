#彭师兄同步到边

from logging_utils import root_logger
from query_manager_v2 import Query
from kb_builder import KnowledgeBaseBuilder
import json

from scheduler_func import lat_first_kb_muledge
from common import conf_and_serv_info

if __name__ == "__main__":

    # 要解决贝叶斯优化耗时太长的问题，就要尽可能缩小配置的已有取值范围。也就是说，在用贝叶斯优化搜索知识库进而冷启动的时候，需要自己指定一个conf_serv_info
    # 而且，在利用新的文件更新字典的时候，也需要更新conf_serv_info。
    # 为了方便起见，
    cert_conf={"reso": "360p", "fps": 5, "encoder": "JPEG"}
    conf_and_serv_info['reso']=[cert_conf['reso']]
    conf_and_serv_info['fps']=[cert_conf['fps']]
 
    job_uid=1
    dag={}
    dag["flow"]= ["face_detection", "gender_classification"]
    user_constraint={}
    user_constraint["delay"]= 0.8

    
    # 专门用于测试调度器的冷启动过程，不限制配置参数的取值范围，就看它能不能在合适的情况下尽可能选出在边缘端执行的配置

    '''
    lat_first_kb_muledge.scheduler_only_cold(
        job_uid=1,
        dag=dag,
        system_status=None,
        work_condition={},
        portrait_info=None,
        user_constraint=user_constraint,
        appended_result_list=None
    )
    '''