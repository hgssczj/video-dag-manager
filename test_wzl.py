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