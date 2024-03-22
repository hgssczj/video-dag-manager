#彭师兄同步到边

from logging_utils import root_logger
from query_manager_v2 import Query
from kb_builder import KnowledgeBaseBuilder
import json

from scheduler_func import lat_first_kb_muledge

if __name__ == "__main__":

    # 要解决贝叶斯优化耗时太长的问题，就要尽可能缩小配置的已有取值范围。也就是说，在用贝叶斯优化搜索知识库进而冷启动的时候，需要自己指定一个conf_serv_info
    # 而且，在利用新的文件更新字典的时候，也需要更新conf_serv_info。
    # 为了方便起见，

 
    job_uid=1
    dag={}
    dag["flow"]= ["face_detection", "face_alignment"]
    user_constraint={}
    user_constraint["delay"]= 0.8



    '''
    lat_first_kb_muledge.scheduler(
            job_uid=1,
            dag=dag,
            resource_info=None,
            runtime_info=None,
            user_constraint=user_constraint,
        )
    '''