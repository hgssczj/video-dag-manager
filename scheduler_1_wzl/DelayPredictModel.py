import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as scio
from .scheduler_common import MODELS_PATH


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



class FaceDetectionProcDelayModel(nn.Module):
    def __init__(self):
        super(FaceDetectionProcDelayModel, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(5, 5) # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(5, 1)  # 第二个隐藏层到输出层
        self.activation = nn.ReLU()  # 隐藏层使用ReLU激活函数
        
        # 初始化网络参数
        self.initialize_parameters()

    def initialize_parameters(self):
        # 使用Xavier初始化方法初始化参数
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class GenderClassificationProcDelayModel(nn.Module):
    def __init__(self):
        super(GenderClassificationProcDelayModel, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(5, 5) # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(5, 1)  # 第二个隐藏层到输出层
        self.activation = nn.ReLU()  # 隐藏层使用ReLU激活函数
        
        # 初始化网络参数
        self.initialize_parameters()

    def initialize_parameters(self):
        # 使用Xavier初始化方法初始化参数
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    

class TransDelayModel(nn.Module):
    def __init__(self):
        super(TransDelayModel, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(5, 5) # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(5, 1)  # 第二个隐藏层到输出层
        self.activation = nn.ReLU()  # 隐藏层使用ReLU激活函数
        
        # 初始化网络参数
        self.initialize_parameters()

    def initialize_parameters(self):
        # 使用Xavier初始化方法初始化参数
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DelayPredictor():
    def __init__(self, pipeline):
        assert isinstance(pipeline, list) and len(pipeline) >= 1
        self.pipeline = pipeline
        self.delay_model_dict = dict()
        
        for service in self.pipeline:
            # 执行时延预估模型
            self.delay_model_dict[service] = dict()
            
            if service == 'face_detection':
                cloud_proc_delay_model_file = MODELS_PATH + "/face_detection_proc_delay_cloud.pth"
                cloud_proc_delay_model = FaceDetectionProcDelayModel()
                cloud_proc_delay_model.load_state_dict(torch.load(cloud_proc_delay_model_file))
                self.delay_model_dict[service]['server'] = cloud_proc_delay_model
                
                edge_proc_delay_model_file = MODELS_PATH + "/face_detection_proc_delay_edge.pth"
                edge_proc_delay_model = FaceDetectionProcDelayModel()
                edge_proc_delay_model.load_state_dict(torch.load(edge_proc_delay_model_file))
                self.delay_model_dict[service]['edge'] = edge_proc_delay_model
            
            elif service == 'gender_classification':
                cloud_proc_delay_model_file = MODELS_PATH + "/gender_classification_proc_delay_cloud.pth"
                cloud_proc_delay_model = GenderClassificationProcDelayModel()
                cloud_proc_delay_model.load_state_dict(torch.load(cloud_proc_delay_model_file))
                self.delay_model_dict[service]['server'] = cloud_proc_delay_model
                
                edge_proc_delay_model_file = MODELS_PATH + "/gender_classification_proc_delay_edge.pth"
                edge_proc_delay_model = GenderClassificationProcDelayModel()
                edge_proc_delay_model.load_state_dict(torch.load(edge_proc_delay_model_file))
                self.delay_model_dict[service]['edge'] = edge_proc_delay_model
        
        trans_delay_model_file = MODELS_PATH + "/trans_delay.pth"
        trans_delay_model = TransDelayModel()
        trans_delay_model.load_state_dict(torch.load(trans_delay_model_file))
        self.delay_model_dict['trans_delay'] = trans_delay_model  
        
        
    def predict(self, info):
        '''
        info的格式如下:
        {
            'delay_type': 'proc_delay',  # 取值为'proc_delay'、'trans_delay'，分别表示执行时延和传输时延
            'predict_info': {
                'service_name': 'face_detection',  # 服务名称
                'fps': 15  # 'fps'、'reso'、'obj_n'、'trans_data_size'
                'node_role': 'server'  # 取值为'server'、'edge'两类
            }
        }
        '''
        delay_type = info['delay_type']
        predict_info = info['predict_info']
        delay = None
        
        if delay_type == 'proc_delay':  # 预测的是执行时延
            service_name = predict_info['service_name']
            
            if service_name == 'face_detection':
                temp_fps = predict_info['fps']
                temp_reso = reso_2_index_dict[predict_info['reso']]
                
                X_data = np.array([temp_fps, temp_reso])
                X_data = X_data.astype(np.float32)
                X_data_tensor = torch.tensor(X_data)
                
            elif service_name == 'gender_classification':
                temp_fps = predict_info['fps']
                temp_obj_n = predict_info['obj_n']
                
                X_data = np.array([temp_fps, temp_obj_n])
                X_data = X_data.astype(np.float32)
                X_data_tensor = torch.tensor(X_data)
            
            predict_model = self.delay_model_dict[service_name][predict_info['node_role']]
            with torch.no_grad():
                delay_tensor = predict_model(X_data_tensor)
                delay = delay_tensor.numpy()[0]
            
        elif delay_type == 'trans_delay':  # 预测的是传输时延
            temp_fps = predict_info['fps']
            temp_trans_data_size = predict_info['trans_data_size']
            
            X_data = np.array([temp_fps, temp_trans_data_size])
            X_data = X_data.astype(np.float32)
            X_data_tensor = torch.tensor(X_data)
            
            predict_model = self.delay_model_dict['trans_delay']
            with torch.no_grad():
                delay_tensor = predict_model(X_data_tensor)
                delay = delay_tensor.numpy()[0]
        
        return delay


if __name__ == '__main__':
    pipeline = ['face_detection', 'gender_classification']
    delay_predictor = DelayPredictor(pipeline)
    
    info = {
            'delay_type': 'trans_delay', 
            'predict_info': {
                'fps': 15,  
                'trans_data_size': 5000 
            }
        }
    
    print(delay_predictor.predict(info))
    
    
