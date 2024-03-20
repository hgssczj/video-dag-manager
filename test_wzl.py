from PortraitModel import PortraitModel
import torch
import numpy as np

class LoadTest():
    def __init__(self):
        self.pipeline = ["face_detection", "gender_classification"]
        
        self.portrait_model_dict = dict()  # 此变量用于保存pipeline中各个服务的画像预估模型
        
        assert(len(self.pipeline) >= 1)
        for service in self.pipeline:
            self.portrait_model_dict[service] = dict()
            
            # CPU利用率预估模型
            self.portrait_model_dict[service]['cpu'] = dict()
            cpu_edge_model_file = "models/" + service + "_cpu_edge.pth"
            cpu_server_model_file = "models/" + service + "_cpu_server.pth"
            cpu_edge_model = PortraitModel()
            cpu_edge_model.load_state_dict(torch.load(cpu_edge_model_file))
            self.portrait_model_dict[service]['cpu']['edge'] = cpu_edge_model
            cpu_server_model = PortraitModel()
            cpu_server_model.load_state_dict(torch.load(cpu_server_model_file))
            self.portrait_model_dict[service]['cpu']['server'] = cpu_server_model
            
            # 内存使用量预估模型
            self.portrait_model_dict[service]['mem'] = dict()
            mem_edge_model_file = "models/" + service + "_mem_edge.pth"
            mem_server_model_file = "models/" + service + "_mem_server.pth"
            mem_edge_model = PortraitModel()
            mem_edge_model.load_state_dict(torch.load(mem_edge_model_file))
            self.portrait_model_dict[service]['mem']['edge'] = mem_edge_model
            mem_server_model = PortraitModel()
            mem_server_model.load_state_dict(torch.load(mem_server_model_file))
            self.portrait_model_dict[service]['mem']['server'] = mem_server_model
    
    def predict(self):
        X_data = np.array([1.0, 1.0, 1.0])
        X_data = X_data.astype(np.float32)
        X_data_tensor = torch.tensor(X_data)
        
        with torch.no_grad():
            predictions = self.portrait_model_dict["face_detection"]['cpu']['edge'](X_data_tensor)
            pred = predictions.numpy()
            print(pred, pred[0])

if __name__ == '__main__':
    a = LoadTest()
    a.predict()
    resource_info = {
        '114.212.81.11': 
            {'device_state': 
                {
                    'cpu_ratio': 40.9, 
                    'gpu_compute_utilization': 
                        {
                            '0': 0, 
                            '1': 0, 
                            '2': 0, 
                            '3': 0
                        }, 
                        'gpu_mem_total':  
                        {'0': 24.0, '1': 24.0, '2': 24.0, '3': 24.0}, 
                        'gpu_mem_utilization': 
                            {'0': 1.4111836751302085, '1': 1.2761433919270835, '2': 1.2761433919270835, '3': 1.2761433919270835}, 
                        'mem_ratio': 14.9, 
                        'mem_total': 251.56013107299805, 
                        'n_cpu': 48, 
                        'net_ratio(MBps)': 0.11047, 
                        'swap_ratio': 0.0
                }, 
                'node_role': 'cloud', 
                'service_state': {
                    'face_detection': {
                        'cpu_util_limit': 1.0, 'mem_util_limit': 1.0
                    }, 
                    'gender_classification': 
                        {'cpu_util_limit': 1.0, 'mem_util_limit': 1.0
                    }
                }
            }, 
            
        '172.27.132.253': 
            {'device_state': {}, 
            'node_role': 'edge', 
            'service_state': {
                'face_detection': {
                    'cpu_util_limit': 1.0, 'mem_util_limit': 1.0
                }, 
            'gender_classification': {
                'cpu_util_limit': 1.0, 'mem_util_limit': 1.0
                }
            }
        }
    }