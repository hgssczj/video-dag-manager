import numpy as np

class Sniffer():
    CONTENT_ELE_MAXN = 50
    CONTENT_OBJ_MAXN = 5

    def __init__(self, job_uid):
        self.job_uid = job_uid
        self.runtime_pkg_list = dict()
        #定义专门保存目标数的列表和初始值。基本思想是边端不保存大多数runtime只保存obj_n，然后定期根据列表求obj_stable。
        self.obj_n_list=list()
        self.obj_stable_value=1

    # TODO：根据taskname解析output_ctx，得到运行时情境
    def sniff(self, taskname, output_ctx):
        if taskname == 'end_pipe':
            if 'delay' not in self.runtime_pkg_list:
                self.runtime_pkg_list['delay'] = list()
            
            if len(self.runtime_pkg_list['delay']) > Sniffer.CONTENT_ELE_MAXN:
                del self.runtime_pkg_list['delay'][0]
            self.runtime_pkg_list['delay'].append(output_ctx['delay'])

        # 对face_detection的结果，提取运行时情境
        # TODO：目标数量、目标大小、目标速度
        if taskname == 'face_detection' :
            # 定义运行时情境字段
            if 'obj_n' not in self.runtime_pkg_list:
                self.runtime_pkg_list['obj_n'] = list()
            if 'obj_size' not in self.runtime_pkg_list:
                self.runtime_pkg_list['obj_size'] = list()

            # 更新各字段序列（防止爆内存）
            if len(self.runtime_pkg_list['obj_n']) > Sniffer.CONTENT_ELE_MAXN:
                del self.runtime_pkg_list['obj_n'][0]
            self.runtime_pkg_list['obj_n'].append(len(output_ctx['faces']))

            # 更新obj_n_list，存放新的元素。
            if len(self.obj_n_list) == Sniffer.CONTENT_OBJ_MAXN: #如果元素数目已经达到上限，清空列表，更新obj_stable_value
                self.obj_stable_value=float(np.std(self.obj_n_list))
                self.obj_n_list.clear()
                print("更新obj_stable_value并清空列表，更新为",self.obj_stable_value)
            self.obj_n_list.append(len(output_ctx['faces'])) #然后重新放入新获取的内容
            

            obj_size = 0
            for x_min, y_min, x_max, y_max in output_ctx['bbox']:
                # TODO：需要依据分辨率转化
                obj_size += (x_max - x_min) * (y_max - y_min)
            if len(output_ctx['bbox'])>0:
                obj_size /= len(output_ctx['bbox'])


            if len(self.runtime_pkg_list['obj_size']) > Sniffer.CONTENT_ELE_MAXN:
                del self.runtime_pkg_list['obj_size'][0]
            self.runtime_pkg_list['obj_size'].append(obj_size)
        
        # 对car_detection的结果，提取目标数量
        # TODO：目标数量、目标大小、目标速度
        if taskname == 'car_detection':
            # 定义运行时情境字段
            if 'obj_n' not in self.runtime_pkg_list:
                self.runtime_pkg_list['obj_n'] = list()

            # 更新各字段序列（防止爆内存）
            if len(self.runtime_pkg_list['obj_n']) > Sniffer.CONTENT_ELE_MAXN:
                del self.runtime_pkg_list['obj_n'][0]
            self.runtime_pkg_list['obj_n'].append(
                sum(list(output_ctx['count_result'].values()))
            )

                        # 更新obj_n_list，存放新的元素。
            if len(self.obj_n_list) == Sniffer.CONTENT_OBJ_MAXN: #如果元素数目已经达到上限，清空列表，更新obj_stable_value
                self.obj_stable_value=float(np.std(self.obj_n_list))
                self.obj_n_list.clear()
                print("更新obj_stable_value并清空列表，更新为",self.obj_stable_value)
            self.obj_n_list.append(sum(list(output_ctx['count_result'].values()))) #然后重新放入新获取的内容
            

            
    def describe_runtime(self):
        # TODO：聚合情境感知参数的时间序列，给出预估值/统计值
        # 彭师兄的最新实现中，每一次处理完一帧都执行曹书与 
        runtime_desc = dict()
        for k, v in self.runtime_pkg_list.items():
            if len(v)>0:
                runtime_desc[k] = sum(v) * 1.0 / len(v)
            else:
                runtime_desc[k] = sum(v) * 1.0
        
        # 获取场景稳定性
        if 'obj_n' in self.runtime_pkg_list.keys():
            print("来看stable:")
            print("来看stable:")
            print("来看stable:")
            print("来看stable:")
            print("来看satble:")
            print("当前的obj_stable值是:",self.obj_stable_value)
            # runtime_desc['obj_stable'] = True if self.obj_stable_value < 0.3 else False
            runtime_desc['obj_stable'] = self.obj_stable_value #直接赋值

        # 每次调用agg后清空
        self.runtime_pkg_list.clear()
        self.runtime_pkg_list = dict()
        
        return runtime_desc
    