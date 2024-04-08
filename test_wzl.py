from PortraitModel import PortraitModel
import torch
import numpy as np
import subprocess
import re

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

def ping_win(host):
    # 这里是windows系统下的操作方式
    # 执行ping命令
    process = subprocess.Popen(['ping', '-n', '2', host], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 读取输出
    stdout, stderr = process.communicate()

    # 解析输出，提取往返时延
    if process.returncode == 0:
        # 使用正则表达式提取往返时延，可以先在命令行执行ping观察输出形式，再对应修改正则表达式
        pattern = r"平均 = (\d+)ms"
        match = re.search(pattern, stdout.decode('gb2312'))
        if match:
            return float(match.group(1))  # 返回平均往返时延
    else:
        print("Ping失败:", stderr.decode('gb2312'))


def ping_unix(host):
    # 这里是类unix系统下的操作方式
    # 执行ping命令
    process = subprocess.Popen(['ping', '-c', '4', host], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 读取输出
    stdout, stderr = process.communicate()

    # 解析输出，提取往返时延
    if process.returncode == 0:
        # 使用正则表达式提取往返时延，可以先在命令行执行ping观察输出形式，再对应修改正则表达式
        pattern = r"min/avg/max/mdev = (\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)"
        match = re.search(pattern, stdout.decode('utf-8'))
        if match:
            return float(match.group(2))  # 返回平均往返时延
    else:
        print("Ping失败:", stderr.decode('utf-8'))


if __name__ == '__main__':
    host = '114.212.81.11'
    # 调用ping函数并获取往返时延
    rtt = ping_win(host)
    if rtt is not None:
        print(f"{host} 的平均往返时延为 {rtt} 毫秒")
    else:
        print("无法获取往返时延")