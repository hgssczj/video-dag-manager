import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as scio
from .scheduler_common import MODELS_PATH


# 定义神经网络类
class PortraitModel(nn.Module):
    def __init__(self):
        super(PortraitModel, self).__init__()
        self.fc1 = nn.Linear(3, 10)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(10, 10) # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(10, 10) # 第一个隐藏层到第二个隐藏层
        self.fc4 = nn.Linear(10, 1)  # 第二个隐藏层到输出层
        self.activation = nn.ReLU()  # 隐藏层使用ReLU激活函数
        
        # 初始化网络参数
        self.initialize_parameters()

    def initialize_parameters(self):
        # 使用Xavier初始化方法初始化参数
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


def train_test():
    # 此函数创建简单的数据集并交给网络训练，目的是实现简单但完整的训练流程，为后续正式训练提供模板
    
    # 创建一个数据集
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    y_train = np.array([[2.0, 1.0], [4.0, 8.0], [6.0, 27.0], [8.0, 64.0], [10.0, 125.0]], dtype=np.float32)

    # 转换为PyTorch的Tensor
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)

    # 实例化神经网络，并进行参数初始化
    model = PortraitModel()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    epochs = 1000
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # 模型评估
    with torch.no_grad():
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        print("predictions:{}, loss:{}.".format(predictions, loss))


def face_detection_train():
    raw_train_data_dir = "train_data/"
    raw_train_data_file = "face_detection_2_cpu_edge.mat"
    node_str = (raw_train_data_file.split('.')[0]).split('_')[-1]
    train_model_file = MODELS_PATH + "/face_detection_cpu_" + node_str + ".pth"
    
    raw_train_data = scio.loadmat(raw_train_data_dir + raw_train_data_file)
    raw_train_data = raw_train_data['data']
    # print(type(raw_train_data), raw_train_data.shape)
    
    # print(type(raw_train_data[0][0]), type(raw_train_data[0][1]), type(raw_train_data[0][2]), type(raw_train_data[0][3]), type(raw_train_data[0][4]), type(raw_train_data[0][5]), type(raw_train_data[0][6]), type(raw_train_data[0][7]))
    X_train = raw_train_data[:, :3]
    X_train = X_train.astype(np.float32)
    Y_train = raw_train_data[:, 3]
    Y_train = np.reshape(Y_train, (-1, 1))
    Y_train = Y_train.astype(np.float32)
    # print(X_train.shape, Y_train.shape)
    
    # 转换为PyTorch的Tensor
    X_train_tensor = torch.tensor(X_train)
    Y_train_tensor = torch.tensor(Y_train)

    # 实例化神经网络，并进行参数初始化
    model = PortraitModel()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    epochs = 20000
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.10f}')

    
    # 模型评估
    with torch.no_grad():
        predictions = model(X_train_tensor)
        loss = criterion(predictions, Y_train_tensor)
        print("predictions:{}, loss:{}.".format(predictions, loss))
        
        x = X_train_tensor.numpy()
        y = Y_train_tensor.numpy()
        pred = predictions.numpy()
        
        data = np.hstack((x, y))
        data = np.hstack((data, pred))
        
        np.savetxt("cpu_pred_analyze.txt", data, fmt='%.15f', delimiter=' ')
    
    # 模型保存
    torch.save(model.state_dict(), train_model_file)
    
    
def gender_classification_train():
    raw_train_data_dir = "train_data/"
    raw_train_data_file = "gender_classification_2_cpu_server.mat"
    node_str = (raw_train_data_file.split('.')[0]).split('_')[-1]
    train_model_file = MODELS_PATH + "/gender_classification_cpu_" + node_str + ".pth"
    
    raw_train_data = scio.loadmat(raw_train_data_dir + raw_train_data_file)
    raw_train_data = raw_train_data['data']
    # print(type(raw_train_data), raw_train_data.shape)
    
    # print(type(raw_train_data[0][0]), type(raw_train_data[0][1]), type(raw_train_data[0][2]), type(raw_train_data[0][3]), type(raw_train_data[0][4]), type(raw_train_data[0][5]), type(raw_train_data[0][6]), type(raw_train_data[0][7]))
    X_train = raw_train_data[:, :3]
    X_train = X_train.astype(np.float32)
    Y_train = raw_train_data[:, 3]
    Y_train = np.reshape(Y_train, (-1, 1))
    Y_train = Y_train.astype(np.float32)
    # print(X_train.shape, Y_train.shape)
    
    # 转换为PyTorch的Tensor
    X_train_tensor = torch.tensor(X_train)
    Y_train_tensor = torch.tensor(Y_train)

    # 实例化神经网络，并进行参数初始化
    model = PortraitModel()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    epochs = 20000
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.10f}')

    
    # 模型评估
    with torch.no_grad():
        predictions = model(X_train_tensor)
        loss = criterion(predictions, Y_train_tensor)
        print("predictions:{}, loss:{}.".format(predictions, loss))
        
        x = X_train_tensor.numpy()
        y = Y_train_tensor.numpy()
        pred = predictions.numpy()
        
        data = np.hstack((x, y))
        data = np.hstack((data, pred))
        
        np.savetxt("cpu_pred_analyze.txt", data, fmt='%.15f', delimiter=' ')
    
    # 模型保存
    torch.save(model.state_dict(), train_model_file)


if __name__ == '__main__':
    gender_classification_train()