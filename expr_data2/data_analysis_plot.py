'''
本文件用于对运行service_test.py得到的csv文件进行绘图分析
'''
import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from scipy.optimize import curve_fit
from sympy import symbols, diff

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
matplotlib.use('TkAgg')

plot_color_list = ['blue', 'red', 'green', 'black', 'magenta', 'cyan', 'yellow', 'orange']  # 绘制折线图时所有可用的颜色列表
plot_ls_list = ['solid', 'dotted', 'dashed', 'dashdot', '-', '--', '-.', ':']  # 绘制折线图时所有可用的linestyle列表

def plot_variety_with_conf(conf_2_data_list_dict, conf_name, field_name):
    '''
    本函数用于绘制数据随配置的变化, conf_2_data_list_dict类型为字典, key为配置的值, value为该配置下采集的数据列表
    '''
    mode = 0
    if mode == 0:
        # mode为0时，将某种配置下所有采样点求平均值进行聚合，然后绘制柱状图
        x_data = []
        y_data = []
        for conf, data_list in conf_2_data_list_dict.items():
            x_data.append(conf)
            y_data.append(np.mean(data_list))
        
        
        x_data = ['1', '3', '5', '8', '12', '15']
        y_data = [0.0571, 0.0622, 0.0741, 0.0902, 0.101, 0.114]
        
        # plt.title(field_name + "随" + conf_name + "变化图", fontdict={'fontsize': 15})
        plt.title("CPU利用率工况变化图", fontdict={'fontsize': 15})
        plt.grid(ls="--", alpha=0.5)  # 绘制虚线网格
        # plt.xlabel(conf_name, fontdict={'fontsize': 13})
        plt.xlabel("目标数量", fontdict={'fontsize': 13})
        # plt.ylabel(field_name, fontdict={'fontsize': 13})
        plt.ylabel("CPU利用率", fontdict={'fontsize': 13})
        # plt.ylim(np.min(y_data) * 0.99, np.max(y_data) * 1.01)
        plt.ylim(np.min(y_data) * 0.9, np.max(y_data) * 1.01)
        plt.bar(x_data, y_data)
        plt.show()
    
    else:
        # mode为1时，不对某种配置下的采样点聚合，而是同时绘制所有配置下的折线图
        plt.figure(figsize=(8,5),dpi=80)
        index = 0
        y_max = 0
        y_min = 0
        for conf, data_list in conf_2_data_list_dict.items():
            y_max = max(y_max, np.max(data_list))
            y_min = min(y_min, np.min(data_list))
            plt.plot(data_list, label=conf, color=plot_color_list[index], linestyle=plot_ls_list[index], linewidth=2, alpha=0.8)
            index += 1
        plt.xlabel("采样点", fontdict={'fontsize': 13})
        plt.ylabel(field_name, fontdict={'fontsize': 13})
        plt.ylim(-(y_min*0.1), y_max*1.5)
        plt.title(field_name + "随" + conf_name + "变化图", fontdict={'fontsize': 15})
        plt.grid(ls="--", alpha=0.4)  # 绘制虚线网格
        plt.legend()  # 显示不同图形的文本标签，即显示plt.plot函数的label参数
        plt.show()


def plot_same_conf_samples(data_rows, conf_value, field_name):
    '''
    本函数用于绘制在某种固定配置下所有采样点的变化情况
    '''
    data_list = None
    for row in data_rows:
        temp_conf = row[0]
        temp_field_name = row[1]
        
        if temp_conf == conf_value and temp_field_name == field_name:
            data_list = list(map(float, row[2:]))
    
    assert data_list is not None
    
    plt.figure(figsize=(8,5),dpi=80)
    plt.plot(data_list, color='r', linewidth=3)
    plt.xlabel("采样点", fontdict={'fontsize': 13})
    plt.ylabel(field_name, fontdict={'fontsize': 13})
    plt.title(conf_value + ",  " + field_name + "变化情况", fontdict={'fontsize': 15})
    plt.grid(ls="--", alpha=0.4)
    plt.show()


def plot_multi_files_conf(file_list, conf_value, field_name):
    mode = 1
    if mode == 0:
        # mode为0时，将某种配置下所有采样点求平均值进行聚合，然后绘制柱状图
        x_data = []
        y_data = []
        # 遍历打开所有文件，找到每一个文件中与conf_value、field_name一致的数据行
        for filename in file_list:
            with open(filename, 'r', newline='', encoding="utf-8") as csv_file:
                # 创建 CSV 读取对象
                csv_reader = csv.reader(csv_file)
                # 读取整个文件内容
                all_rows = list(csv_reader)
            
            for row in all_rows:
                temp_conf = row[0]
                temp_field_name = row[1]
                if temp_conf == conf_value and temp_field_name == field_name:
                    temp_data_list = list(map(float, row[2:]))
                    x_data.append(filename.split('_')[8])
                    y_data.append(np.mean(temp_data_list))
        
        plt.title(field_name + "随模型输入大小变化图", fontdict={'fontsize': 15})
        plt.grid(ls="--", alpha=0.5)  # 绘制虚线网格
        plt.xlabel("模型输入大小", fontdict={'fontsize': 13})
        plt.ylabel(field_name, fontdict={'fontsize': 13})
        plt.bar(x_data, y_data)
        plt.show()
        
    else:
        # 初始化画布
        plt.figure(figsize=(8,8),dpi=80)
        index = 0
        y_max = 0
        y_min = 0
        
        # 遍历打开所有文件，找到每一个文件中与conf_value、field_name一致的数据行
        for filename in file_list:
            with open(filename, 'r', newline='', encoding="utf-8") as csv_file:
                # 创建 CSV 读取对象
                csv_reader = csv.reader(csv_file)
                # 读取整个文件内容
                all_rows = list(csv_reader)
            
            for row in all_rows:
                temp_conf = row[0]
                temp_field_name = row[1]
                if temp_conf == conf_value and temp_field_name == field_name:
                    temp_data_list = list(map(float, row[2:]))
                    plt.plot(temp_data_list, label=filename, color=plot_color_list[index], linestyle=plot_ls_list[index], linewidth=2, alpha=0.8)
                    index += 1
                    y_max = max(y_max, np.max(temp_data_list))
                    y_min = min(y_min, np.min(temp_data_list))
                    break
        
        # 显示完整的图像
        plt.xlabel("采样点", fontdict={'fontsize': 13})
        plt.ylabel(field_name, fontdict={'fontsize': 13})
        plt.ylim(y_min*0.95, y_max*1.1)
        plt.title(conf_value + ",  " + field_name + "变化图", fontdict={'fontsize': 15})
        plt.grid(ls="--", alpha=0.4)  # 绘制虚线网格
        plt.legend()  # 显示不同图形的文本标签，即显示plt.plot函数的label参数
        plt.show()
    

def analyze_one_file():
    '''
    本函数用于仅分析一个csv文件内的数据
    '''
    filename = "20240108_17_22_54_face_alignment_test_client_1280_gpu.csv"
    
    #################### 1.读取数据并存储 ####################
    with open(filename, 'r', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 读取对象
        csv_reader = csv.reader(csv_file)
        # 读取整个文件内容
        all_rows = list(csv_reader)
        
    reso_2_cpu_util_use_dict = {}
    reso_2_mem_util_use_dict = {}
    reso_2_mem_use_amount_dict = {}
    reso_2_compute_latency_dict = {}
    reso_2_gpu_proc_time_dict = {}
    reso_2_pre_proc_time_dict = {}
    reso_2_post_proc_time_dict = {}
    
    for row in all_rows:
        temp_conf = row[0]
        temp_field_name = row[1]
        temp_data_list = list(map(float, row[2:]))
        
        if temp_field_name == 'cpu_util_use':
            reso_2_cpu_util_use_dict[temp_conf] = temp_data_list
        if temp_field_name == 'mem_util_use':
            reso_2_mem_util_use_dict[temp_conf] = temp_data_list
        if temp_field_name == 'mem_use_amount':
            reso_2_mem_use_amount_dict[temp_conf] = temp_data_list
        if temp_field_name == 'compute_latency':
            reso_2_compute_latency_dict[temp_conf] = temp_data_list
        if temp_field_name == 'gpu_proc_time':
            reso_2_gpu_proc_time_dict[temp_conf] = temp_data_list
        if temp_field_name == 'pre_proc_time':
            reso_2_pre_proc_time_dict[temp_conf] = temp_data_list
        if temp_field_name == 'post_proc_time':
            reso_2_post_proc_time_dict[temp_conf] = temp_data_list
    
    # cpu_inf_time_dict = {}  # cpu模型推理的时间，当gpu_proc_time字段为0时使用
    # for reso, compute_latency_list in reso_2_compute_latency_dict.items():
    #     cpu_inf_time_dict[reso] = [(compute_latency_list[i] - reso_2_pre_proc_time_dict[reso][i] - reso_2_post_proc_time_dict[reso][i]) for i in range(len(compute_latency_list))]
    
    #################### 2.根据需求绘制 ####################
    plot_variety_with_conf(reso_2_mem_util_use_dict, "分辨率", "内存利用率")
    # plot_same_conf_samples(all_rows, "360p", "cpu_util_use")


def analyze_multi_files():
    '''
    本函数用于同时对比分析多个csv文件中的内容, 每个csv文件中的数据格式(列数、每一列的含义、行数)必须一致
    '''
    file_names = [
        "20240108_17_01_53_face_alignment_test_client_128_gpu.csv",
        "20240108_17_04_03_face_alignment_test_client_160_gpu.csv",
        "20240108_17_07_20_face_alignment_test_client_320_gpu.csv",
        "20240108_17_13_25_face_alignment_test_client_480_gpu.csv",
        "20240108_17_18_23_face_alignment_test_client_640_gpu.csv",
        "20240108_17_22_54_face_alignment_test_client_1280_gpu.csv",
    ]
    
    plot_multi_files_conf(file_names, "1080p", "mem_util_use")


def plot_list_2_bar():
    '''
    本函数用于对列表中数字出现的次数进行统计, 并按照数字大小绘制柱状图
    '''
    data_list = [0.26, 0.26, 0.25000000000000006, 0.26, 0.26, 0.26, 0.26, 0.25000000000000006, 0.27, 0.25000000000000006, 0.25000000000000006, 0.26, 0.26, 0.26, 0.26, 0.27, 0.26, 0.26, 0.26, 0.25000000000000006, 0.31000000000000005, 0.27, 0.26, 0.25000000000000006, 0.26, 0.26, 0.25000000000000006, 0.26, 0.26, 0.26]  
    data_list.sort(key=float)  # 首先对data_list排序
    data_count_dict = {}
    
    for data in data_list:  # 统计所有数字出现的次数
        if data in data_count_dict:
            data_count_dict[data] += 1
        else:
            data_count_dict[data] = 1
    
    x = ["0.001", "0.002", "0.003", "0.004", "0.005", "0.006", "0.007", "0.008", "0.009", "0.01"]
    y = [0.86809, 0.86811, 0.86791, 0.86787, 0.86785, 0.86804, 0.86785, 0.8679, 0.867736842105, 0.868144736842]
    
    # for data, data_count in data_count_dict.items():
    #     x.append(str(data))
    #     y.append(data_count)
    
    # 绘制柱状图
    plt.title("CPU利用率随内存利用率变化图", fontdict={'fontsize': 15})
    plt.grid(ls="--", alpha=0.5)  # 绘制虚线网格
    plt.xlabel("内存利用率", fontdict={'fontsize': 13})
    plt.ylabel("CPU利用率", fontdict={'fontsize': 13})
    plt.tick_params(axis='both',which='major',labelsize=7) 
    plt.bar(x, y)
    plt.show()
    

def plot_cpu_util():
    '''
    本函数用于根据csv文件绘制cpu利用率的均值、方差、标准差随分辨率变化的柱状图
    csv文件中的一行:分辨率、cpu利用率均值、cpu利用率方差、cpu利用率标准差
    '''
    #################### 1.读取数据并存储 ####################
    file_name = ""
    with open(file_name, 'r', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 读取对象
        csv_reader = csv.reader(csv_file)
        # 读取整个文件内容
        all_rows = list(csv_reader)
        
    reso_2_cpu_util_mean_dict = {}  # 分辨率与CPU利用率各种值的映射字典
    reso_2_cpu_util_var_dict = {}
    reso_2_cpu_util_std_dict = {}
    
    for row in all_rows:
        temp_conf = row[0]
        reso_2_cpu_util_mean_dict[temp_conf] = [float(row[1])]  # 为了便于调用已有函数，将一个值也作为list输入
        reso_2_cpu_util_var_dict[temp_conf] = [float(row[2])]
        reso_2_cpu_util_std_dict[temp_conf] = [float(row[3])]
    
    #################### 2.绘制各个图形 ####################
    plot_variety_with_conf(reso_2_cpu_util_mean_dict, "分辨率", "CPU利用率均值")
    plot_variety_with_conf(reso_2_cpu_util_var_dict, "分辨率", "CPU利用率方差")
    plot_variety_with_conf(reso_2_cpu_util_std_dict, "分辨率", "CPU利用率标准差")


def inv_prop_func(x, a, b, c):
    return a / (x ** b) + c

    
def plot_cpu_util_2_latency():
    '''
    本函数用于根据csv文件绘制时延随cpu利用率变化的散点图和拟合曲线图
    csv文件中的一行:cpu利用率、时延
    '''
    #################### 1.读取数据并存储 ####################
    file_name = "20240121_10_56_53_test_process_cpu_util.csv"
    with open(file_name, 'r', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 读取对象
        csv_reader = csv.reader(csv_file)
        # 读取整个文件内容
        all_rows = list(csv_reader)
    
    cpu_util_list = []
    latency_list = []
    for row in all_rows:
        cpu_util_list.append(float(row[0]))
        latency_list.append(float(row[1]))
    
    # 绘制原始数据点
    plt.figure(figsize=(6, 6), dpi=100)
    plt.scatter(cpu_util_list, latency_list, c='blue', s=10)
    plt.xlabel("CPU利用率", fontdict={'fontsize': 13})
    plt.ylabel("latency", fontdict={'fontsize': 13})
    plt.ylim(-0.5, 7)
    # plt.title("时延随CPU利用率变化图", fontdict={'fontsize': 15})
    plt.grid(ls="--", alpha=0.4)  # 绘制虚线网格
    # 手动设置坐标轴刻度间隔以及刻度范围
    # plt.tick_params(axis='both',which='major',labelsize=9)     
    # x_major_locator=MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator=MultipleLocator(10)  # 把y轴的刻度间隔设置为10，并存在变量里
    # ax=plt.gca()  # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数 
    # plt.xlim(-0.5, 11)  #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    # plt.ylim(-5, 110)
    
    # 拟合为反比例函数
    popt_1, pcov_1 = curve_fit(inv_prop_func, cpu_util_list, latency_list)
    plt.plot(cpu_util_list, inv_prop_func(cpu_util_list, *popt_1), 'g-', label='inv_prop_func fit')
    
    # plt.legend()
    plt.show()
    
    # 计算在不限制CPU利用率时的时延
    latency_min = inv_prop_func(1.0, *popt_1)
    # 计算在groundtruth利用率下的时延
    temp_latency = inv_prop_func(0.25, *popt_1)
    # 计算二者的差，用于确定阈值
    latency_dif = temp_latency - latency_min
    
    # 参数拟合的结果
    a_fit, b_fit, c_fit = popt_1
    # 定义符号变量
    x = symbols('x')
    # 定义拟合结果函数
    fit_function = a_fit / (x ** b_fit) + c_fit
    # 计算拟合函数关于 x 的导数
    derivative_fit_function = diff(fit_function, x)
    
    # 定义要计算导数的点
    x_value = 0.25
    # 将 x 替换为指定值
    derivative_at_x = derivative_fit_function.subs(x, x_value)
    # 计算导数的数值
    derivative_value = derivative_at_x.evalf()

    # 打印结果
    print("不限制CPU利用率时的时延为: {}s, 真实CPU利用率下的时延为: {}s, 二者的差异为: {}s".format(latency_min, temp_latency, latency_dif))
    print("拟合参数:", popt_1)
    print("拟合结果函数:", fit_function)
    print("导数:", derivative_fit_function)
    print(f"在 x={x_value} 处的导数值:", derivative_value)

    
    
    
    
if __name__ == "__main__":
    plot_list_2_bar()