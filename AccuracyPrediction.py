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
            
            if temp_obj_speed is None or temp_obj_speed == 0:  # 当不存在速度字段时，默认速度为第二档
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
                
            if temp_obj_size is None or temp_obj_size == 0:  # 当不存在目标大小字段时，默认大小为第二档
                a2 = 0.99  
                b2 = -0.47
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
            
            a2 = 0.99  # 计算当前分辨率下的精度
            b2 = -0.47
            c2 = -0.008
            d2 = 350
            
            acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
            acc_2_reso = acc_2_reso[0]
            
            acc = acc_2_fps * acc_2_reso  # 将分辨率下的精度与帧率下的精度相乘，得到最终预测的精度
            
            return acc
        

if __name__ == "__main__":
    acc_pred = AccuracyPrediction()
    
    service_name = 'face_detection'
    service_conf = {
        # 配置字段
        'fps': 1,
        'reso': '900p'
    }
    
    print(acc_pred.predict(service_name, service_conf, obj_size=79000, obj_speed=200))
        