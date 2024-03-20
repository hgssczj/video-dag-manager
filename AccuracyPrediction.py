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
    
    def predict(self, service_name, service_conf):
        if service_name == 'face_detection':
            temp_fps = service_conf['fps']
            temp_reso = common.resolution_wh[service_conf['reso']]['h']
            temp_obj_size = None
            temp_obj_speed = None
            
            a1 = 0.97  # 计算当前帧率下的精度
            b1 = -1.25
            c1 = -0.8
            acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
            acc_2_fps = acc_2_fps[0]
            
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
        'fps': 30,
        'reso': '1080p'
        
        # TODO：加入工况字段
    }
    
    print(acc_pred.predict(service_name, service_conf))
        