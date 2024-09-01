import numpy as np
import common

# 定义指数函数模型
def exponential_function_1(x, a, b, c):
    return a + b * np.exp(c * x)

def exponential_function_2(x, a, b, c, d):
    return a + b * np.exp(c * (x - d))


class AccuracyPrediction():
    def __init__(self):
        # TODO：为不同的工况建立不同的精度曲线
        pass
    
    def predict(self, service_name, service_conf, obj_size=None, obj_speed=None):
        if service_name == 'face_detection':
            temp_fps = service_conf['fps']
            temp_reso = common.resolution_wh[service_conf['reso']]['h']
            temp_obj_size = obj_size
            temp_obj_speed = obj_speed
            
            if temp_obj_speed is None or temp_obj_speed == -1:  # 当不存在速度字段时，默认速度为第二档
                a1 = 0.97  
                b1 = -1.20
                c1 = -0.78
                acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                acc_2_fps = acc_2_fps[0]
            else:
                if temp_obj_speed <= 260:  # 0.98, -0.75, -0.85
                    a1 = 0.98 
                    b1 = -0.75
                    c1 = -0.85
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                elif temp_obj_speed > 260 and temp_obj_speed <= 520:  # 0.97, -1.20, -0.78
                    a1 = 0.97  
                    b1 = -1.20
                    c1 = -0.78
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                elif temp_obj_speed > 520 and temp_obj_speed <= 780:  # 0.985, -0.9, -0.29
                    a1 = 0.985
                    b1 = -0.9
                    c1 = -0.29
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                else:  # 1.0, -0.84, -0.18
                    a1 = 1.0  
                    b1 = -0.84
                    c1 = -0.18
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                
            if temp_obj_size is None or temp_obj_size == -1:  # 当不存在目标大小字段时，默认大小为第二档
                a2 = 0.99  
                b2 = -0.47
                c2 = -0.008
                d2 = 350
                
                acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                acc_2_reso = acc_2_reso[0]
            elif temp_obj_size == 0:  # 当目标大小为0时，说明场景中此时没有目标，默认目标为最大的大小
                a2 = 0.99  # 计算当前分辨率下的精度
                b2 = -0.2
                c2 = -0.008
                d2 = 350
                
                acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                acc_2_reso = acc_2_reso[0]
            else:
                if temp_obj_size <= 50000:  # 0.98, -0.63, -0.006, 350
                    a2 = 0.98  # 计算当前分辨率下的精度
                    b2 = -0.63
                    c2 = -0.006
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
                elif temp_obj_size > 50000 and temp_obj_size <= 100000:  # 0.99, -0.47, -0.008, 350
                    a2 = 0.99  # 计算当前分辨率下的精度
                    b2 = -0.47
                    c2 = -0.008
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
                else:  # 0.99, -0.2, -0.008, 350
                    a2 = 0.99  # 计算当前分辨率下的精度
                    b2 = -0.2
                    c2 = -0.008
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
            
            acc = acc_2_fps * acc_2_reso  # 将分辨率下的精度与帧率下的精度相乘，得到最终预测的精度
            
            return acc
    
    
    def get_single_conf_acc(self, service_name, conf_info, service_work_condition):
        # 根据精度曲线获取某一种配置在当前工况下的精度
        if service_name == 'face_detection':
            if conf_info['conf_type'] == 'reso':  # 所求配置类型为分辨率
                temp_reso = common.resolution_wh[conf_info['conf_value']]['h']
                
                assert 'obj_size' in service_work_condition
                temp_obj_size = service_work_condition['obj_size']
                
                if temp_obj_size is None or temp_obj_size == -1:  # 当不存在目标大小字段时，默认大小为第二档
                    a2 = 0.99  
                    b2 = -0.47
                    c2 = -0.008
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
                elif temp_obj_size == 0:  # 当目标大小为0时，说明场景中此时没有目标，默认目标为最大的大小
                    a2 = 0.99  # 计算当前分辨率下的精度
                    b2 = -0.2
                    c2 = -0.008
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
                else:
                    if temp_obj_size <= 50000:  # 0.98, -0.63, -0.006, 350
                        a2 = 0.98  # 计算当前分辨率下的精度
                        b2 = -0.63
                        c2 = -0.006
                        d2 = 350
                        
                        acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                        acc_2_reso = acc_2_reso[0]
                    elif temp_obj_size > 50000 and temp_obj_size <= 100000:  # 0.99, -0.47, -0.008, 350
                        a2 = 0.99  # 计算当前分辨率下的精度
                        b2 = -0.47
                        c2 = -0.008
                        d2 = 350
                        
                        acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                        acc_2_reso = acc_2_reso[0]
                    else:  # 0.99, -0.2, -0.008, 350
                        a2 = 0.99  # 计算当前分辨率下的精度
                        b2 = -0.2
                        c2 = -0.008
                        d2 = 350
                        
                        acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                        acc_2_reso = acc_2_reso[0]

                return acc_2_reso
            
            elif conf_info['conf_type'] == 'fps':  # 所求配置类型为帧率
                temp_fps = conf_info['conf_value']
                
                assert 'obj_speed' in service_work_condition
                temp_obj_speed = service_work_condition['obj_speed']
            
                if temp_obj_speed is None or temp_obj_speed == -1:  # 当不存在速度字段时，默认速度为第二档
                    a1 = 0.97  
                    b1 = -1.20
                    c1 = -0.78
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                else:
                    if temp_obj_speed <= 260:  # 0.98, -0.75, -0.85
                        a1 = 0.98 
                        b1 = -0.75
                        c1 = -0.85
                        acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                        acc_2_fps = acc_2_fps[0]
                    elif temp_obj_speed > 260 and temp_obj_speed <= 520:  # 0.97, -1.20, -0.78
                        a1 = 0.97  
                        b1 = -1.20
                        c1 = -0.78
                        acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                        acc_2_fps = acc_2_fps[0]
                    elif temp_obj_speed > 520 and temp_obj_speed <= 780:  # 0.985, -0.9, -0.29
                        a1 = 0.985
                        b1 = -0.9
                        c1 = -0.29
                        acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                        acc_2_fps = acc_2_fps[0]
                    else:  # 1.0, -0.84, -0.18
                        a1 = 1.0  
                        b1 = -0.84
                        c1 = -0.18
                        acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                        acc_2_fps = acc_2_fps[0]
                
                return  acc_2_fps
        
        return -1
        
    
    def if_acc_improved(self, service_name, service_conf, service_work_condition):
        # 判断一个服务在当前工况下提升配置能否带来精度的明显提升
        if service_name == 'face_detection':
            # 当前配置下的精度
            temp_acc = self.predict(service_name, service_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
            temp_fps = service_conf['fps']
            temp_reso = service_conf['reso']
            temp_fps_index = common.fps_range.index(temp_fps)
            temp_reso_index = common.reso_range.index(temp_reso)
            
            improved_fps_index = min(temp_fps_index + 1, len(common.fps_range) - 1)
            improved_reso_index = min(temp_reso_index + 1, len(common.reso_range) - 1)
            improved_conf = {
                'fps': common.fps_range[improved_fps_index],
                'reso': common.reso_range[improved_reso_index],
                'encoder': 'JPEG'
            }
            
            improved_acc = self.predict(service_name, improved_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
            
            if (improved_acc - temp_acc) > 0.05:  # 配置提升一档可提升5%以上
                return True
            
        return False
    
    def if_acc_improved_with_fps(self, service_name, service_conf, service_work_condition):
        if service_name == 'face_detection':
            # 当前帧率、最高分辨率下的精度
            temp_fps = service_conf['fps']
            temp_fps_index = common.fps_range.index(temp_fps)
            temp_conf = {
                'fps': temp_fps,
                'reso': '1080p',
                'encoder': 'JPEG'
            }
            temp_acc = self.predict(service_name, temp_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
            
            # 当前帧率加一档、最高分辨率下的精度
            improved_fps_index = min(temp_fps_index + 1, len(common.fps_range) - 1)
            improved_conf = {
                'fps': common.fps_range[improved_fps_index],
                'reso': '1080p',
                'encoder': 'JPEG'
            }
            improved_acc = self.predict(service_name, improved_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
            
            if (improved_acc - temp_acc) > 0.05:  # 配置提升一档可提升5%以上
                return True
            
        return False
    
    def if_acc_improved_with_reso(self, service_name, service_conf, service_work_condition):
        if service_name == 'face_detection':
            # 当前分辨率、最高帧率下的精度
            temp_reso = service_conf['reso']
            temp_reso_index = common.reso_range.index(temp_reso)
            temp_conf = {
                'fps': 30,
                'reso': temp_reso,
                'encoder': 'JPEG'
            }
            temp_acc = self.predict(service_name, temp_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
            
            # 当前帧率加一档、最高分辨率下的精度
            improved_reso_index = min(temp_reso_index + 1, len(common.reso_range) - 1)
            improved_conf = {
                'fps': 30,
                'reso': common.reso_range[improved_reso_index],
                'encoder': 'JPEG'
            }
            improved_acc = self.predict(service_name, improved_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
            
            if (improved_acc - temp_acc) > 0.05:  # 配置提升一档可提升5%以上
                return True
            
        return False
    
    def get_conf_improve_bound(self, service_name, service_conf, service_work_condition):
        # 给定一个服务提升精度的配置上界
        if service_name == 'face_detection':
            # 当前配置下的精度
            temp_acc = self.predict(service_name, service_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
            temp_fps = service_conf['fps']
            temp_reso = service_conf['reso']
            temp_fps_index = common.fps_range.index(temp_fps)
            temp_reso_index = common.reso_range.index(temp_reso)
            
            # 确定帧率的上界
            new_fps_index = temp_fps_index + 1
            while new_fps_index <= len(common.fps_range) - 1:
                pre_conf = {
                    'fps': common.fps_range[new_fps_index - 1],
                    'reso': '1080p',
                    'encoder': 'JPEG'
                }
                pre_acc = self.predict(service_name, pre_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
                
                cur_conf = {
                    'fps': common.fps_range[new_fps_index],
                    'reso': '1080p',
                    'encoder': 'JPEG'
                }
                cur_acc = self.predict(service_name, cur_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
                
                if cur_acc - pre_acc >= 0.03:
                    new_fps_index += 1
                else:
                    break
            if new_fps_index >= len(common.fps_range):
                new_fps_index = len(common.fps_range) - 1
            
            # 确定分辨率的上界
            new_reso_index = temp_reso_index + 1
            while new_reso_index <= len(common.reso_range) - 1:
                pre_conf = {
                    'fps': 30,
                    'reso': common.reso_range[new_reso_index - 1],
                    'encoder': 'JPEG'
                }
                pre_acc = self.predict(service_name, pre_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
                
                cur_conf = {
                    'fps': 30,
                    'reso': common.reso_range[new_reso_index],
                    'encoder': 'JPEG'
                }
                cur_acc = self.predict(service_name, cur_conf, service_work_condition['obj_size'], service_work_condition['obj_speed'])
                
                if cur_acc - pre_acc >= 0.03:
                    new_reso_index += 1
                else:
                    break
            if new_reso_index >= len(common.reso_range):
                new_reso_index = len(common.reso_range) - 1
        
            return {
                'fps_upper_bound': common.fps_range[new_fps_index],
                'reso_upper_bound': common.reso_range[new_reso_index]
            } 
            
        return {}   
        
    def get_middle_conf(self, service_name, acc_constraint, obj_size=None, obj_speed=None):
        # 此函数用于获取指定服务在指定工况下的中配置列表
        # 中配置的定义：满足精度约束的最低配置，其中任何一种具体的配置降低一档都会导致其不再满足精度约束
        middle_conf_list = []  # 中配置列表
        middle_plus_conf_list = []  # 比中配置高一档的配置列表，用于在提升配置时给出上界
        
        assert service_name == 'face_detection'
        temp_obj_size = obj_size
        temp_obj_speed = obj_speed
        
        if temp_obj_speed is None or temp_obj_speed == -1:  # 当不存在速度字段时，默认速度为第二档
            temp_obj_speed = 300
        
        if temp_obj_size is None or temp_obj_size == -1:  # 当不存在目标大小字段时，默认大小为第二档
            temp_obj_size = 70000
        
        # 采取遍历+剪枝的方式确定中配置，相较于暴力遍历的方法节省巨大的搜索开销，更易于扩展到更多类型的配置
        for i in range(len(common.reso_range)):
            temp_conf_info = {
                'conf_type': 'reso',
                'conf_value': common.reso_range[i]
            }
            temp_work_condition = {
                'obj_size': temp_obj_size,
                'obj_speed': temp_obj_speed
            }
            temp_reso_acc = self.get_single_conf_acc(service_name, temp_conf_info, temp_work_condition)
            assert temp_reso_acc > 0
            
            if temp_reso_acc < acc_constraint:  # 剪枝
                continue
            
            for j in range(len(common.fps_range)):
                
                temp_service_conf = {
                    'reso': common.reso_range[i],
                    'fps': common.fps_range[j]
                }
                temp_acc = self.predict(service_name, temp_service_conf, temp_obj_size, temp_obj_speed)
                if temp_acc >= acc_constraint:
                    if i == 0 and j == 0:  # 若当前配置为最低配置且满足精度约束，则中配置只有最低配置
                        middle_conf_list = [(common.fps_range[j], common.reso_range[i])]
                        
                        middle_plus_reso_index = min(i+1, len(common.reso_range)-1)
                        middle_plus_fps_index = min(j+1, len(common.fps_range)-1)
                        middle_plus_conf_list = [(common.fps_range[middle_plus_fps_index], common.reso_range[middle_plus_reso_index])]
                        
                        return middle_conf_list, middle_plus_conf_list
                    
                    else:
                        # 判断分辨率降一档是否会导致不满足精度约束
                        flag_1 = True
                        if i >= 1:
                            temp_service_conf = {
                                'reso': common.reso_range[i-1],
                                'fps': common.fps_range[j]
                            }
                            temp_acc = self.predict(service_name, temp_service_conf, temp_obj_size, temp_obj_speed)
                            if temp_acc >= acc_constraint:  # 分辨率降一档仍然满足精度约束，则当前配置不是中配置
                                flag_1 = False
                        
                        # 判断帧率降一档是否会导致不满足精度约束
                        flag_2 = True
                        if j >= 1:
                            temp_service_conf = {
                                'reso': common.reso_range[i],
                                'fps': common.fps_range[j-1]
                            }
                            temp_acc = self.predict(service_name, temp_service_conf, temp_obj_size, temp_obj_speed)
                            if temp_acc >= acc_constraint:
                                flag_2 = False  # 帧率降一档仍然满足精度约束，则当前配置不是中配置
                            
                        if flag_1 and flag_2:  # 所有配置都不能再降低，则为中配置
                            middle_conf_list.append((common.fps_range[j], common.reso_range[i]))
                            
                            middle_plus_reso_index = min(i+1, len(common.reso_range)-1)
                            middle_plus_fps_index = min(j+1, len(common.fps_range)-1)
                            middle_plus_conf_list.append((common.fps_range[middle_plus_fps_index], common.reso_range[middle_plus_reso_index]))
                            
                            break  # 剪枝
                            
        assert len(middle_conf_list) != 0  # middle_conf_list长度为0说明最高配置也无法满足精度约束
        return middle_conf_list, middle_plus_conf_list
                

if __name__ == "__main__":
    acc_pred = AccuracyPrediction()
    
    service_name = 'face_detection'
    # service_conf = {
    #     # 配置字段
    #     'fps': 1,
    #     'reso': '900p'
    # }
    
    # print(acc_pred.predict(service_name, service_conf, obj_size=79000, obj_speed=200))
    
    middle_conf_list, middle_plus_conf_list = acc_pred.get_middle_conf(service_name, 0.6, obj_size=79000, obj_speed=200)
    print(middle_conf_list, middle_plus_conf_list)
    
    a = (5, '480p')
    print(a in middle_conf_list)
    
        