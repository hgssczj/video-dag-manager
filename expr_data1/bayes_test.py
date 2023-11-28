#尝试使用贝叶斯分类器分析所得到的模型

import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import sklearn.metrics
import time
import optuna
plt.rcParams['font.sans-serif'] = ['SimHei'] # 运行配置参数中的字体（font）为黑体（SimHei）

#''' #以下内容用于从文件中提取数据并绘制为图像
filename='20231106_headup_detect_delay_test_new3_0.15_0.7_tx2-cloud-raw_host_cloud_1080p.csv'
#现在需要求“均值”
df = pd.read_csv(filename)
df = df[1:]
max_fps = 30
df.loc[df['fps'] >= 30, 'fps'] = max_fps
resolution_map = {"360p": 0, "480p": 1, "720p": 2, "1080p": 3} #分辨率转换为数字
df['reso'] = df['reso'].replace(resolution_map)

print(filename)
print('all_delay',df['all_delay'].mean())
print('d_proc_delay',df['d_proc_delay'].mean())
print('d_trans_delay',df['d_trans_delay'].mean())
print('a_proc_delay',df['a_proc_delay'].mean())
print('a_trans_delay',df['a_trans_delay'].mean())

fig, axs = plt.subplots(3, 2, figsize=(15, 8), dpi=80)


axs[0,0].plot(df['n_loop'], df['total'], label='total')
axs[0,0].plot(df['n_loop'], df['up'], label='up')
axs[0,0].title.set_text('总人数/抬头数 vs 时间')
axs[0,0].legend()

#注意以下绘制的是用户约束为0.3的情况
axs[0,1].hlines(y=0.3, xmin=0, xmax=len(df['n_loop']), linewidth=3, color='r', label='用户约束')
#axs[1,0].plot(df['n_loop'], scipy.signal.savgol_filter(df['all_delay'], 11, 3))
axs[0,1].plot(df['n_loop'], df['all_delay'])
axs[0,1].title.set_text('总处理时延 vs 时间')
axs[0,1].legend()


#axs[1,0].plot(df['n_loop'], scipy.signal.savgol_filter(df['all_delay'], 11, 3))
axs[1,0].plot(df['n_loop'], df['d_proc_delay'],label='d_proc_delay')
axs[1,0].title.set_text('人脸检测时延 vs 时间')
axs[1,0].legend()


axs[1,1].plot(df['n_loop'], df['d_trans_delay'],label='d_trans_delay')
axs[1,1].title.set_text('人脸检测传输时延 vs 时间')
axs[1,1].legend()

axs[2,0].plot(df['n_loop'], df['a_proc_delay'],label='a_proc_delay')
axs[2,0].title.set_text('姿态估计时延 vs 时间')
axs[2,0].set_ylabel('姿态估计时延')
axs[2,0].legend()


axs[2,1].plot(df['n_loop'], df['a_trans_delay'],label='a_trans_delay')
axs[2,1].title.set_text('姿态估计传输时延 vs 时间')
axs[2,1].legend()

plt.show()
#'''

# 以下函数可以将指定文件里的内容中的数据提取出来，用于初始化result字典
def read_anylze(fiename):
    df = pd.read_csv(filename)

    df = df[1:]
    max_fps = 30
    df.loc[df['fps'] >= 30, 'fps'] = max_fps
    resolution_map = {"360p": 0, "480p": 1, "720p": 2, "1080p": 3} #分辨率转换为数字
    df['reso'] = df['reso'].replace(resolution_map)
    key=['all_delay','d_proc_delay','d_trans_delay','a_proc_delay','a_trans_delay']
    dic=dict()
    dic[key[0]]=df[key[0]].mean()
    dic[key[1]]=df[key[1]].mean()
    dic[key[2]]=df[key[2]].mean()
    dic[key[3]]=df[key[3]].mean()
    dic[key[4]]=df[key[4]].mean()
    return dic

result=dict()
result['cc']=dict()
result['hc']=dict()

if(1==1): #以下内容用于从各个文件中提取数据构建result字典，该字典可以用于构建性能评估模块，也就是知识库
    filename='20231106_headup_detect_delay_test_new3_0.15_0.7_tx2-cloud-raw_cloud_cloud_360p.csv'
    result['cc']['360p']=read_anylze(filename)
    filename='20231106_headup_detect_delay_test_new3_0.15_0.7_tx2-cloud-raw_cloud_cloud_480p.csv'
    result['cc']['480p']=read_anylze(filename)
    filename='20231106_headup_detect_delay_test_new3_0.15_0.7_tx2-cloud-raw_cloud_cloud_720p.csv'
    result['cc']['720p']=read_anylze(filename)
    filename='20231106_headup_detect_delay_test_new3_0.15_0.7_tx2-cloud-raw_cloud_cloud_1080p.csv'
    result['cc']['1080p']=read_anylze(filename)

    filename='20231106_headup_detect_delay_test_new3_0.15_0.7_tx2-cloud-raw_host_cloud_360p.csv'
    result['hc']['360p']=read_anylze(filename)
    filename='20231106_headup_detect_delay_test_new3_0.15_0.7_tx2-cloud-raw_host_cloud_480p.csv'
    result['hc']['480p']=read_anylze(filename)
    filename='20231106_headup_detect_delay_test_new3_0.15_0.7_tx2-cloud-raw_host_cloud_720p.csv'
    result['hc']['720p']=read_anylze(filename)
    filename='20231106_headup_detect_delay_test_new3_0.15_0.7_tx2-cloud-raw_host_cloud_1080p.csv'
    result['hc']['1080p']=read_anylze(filename)

    print('cc 360',result['cc']['360p'])
    print('cc 480',result['cc']['480p'])
    print('cc 720',result['cc']['720p'])
    print('cc 1080',result['cc']['1080p'])

    print('hc 360',result['hc']['360p'])
    print('hc 480',result['hc']['480p'])
    print('hc 720',result['hc']['720p'])
    print('hc 1080',result['hc']['1080p'])

#现在要创建模块，每一个字典一个模块，每一个模块有自己的配置组合。暂时不把
serv_names=['d_proc_delay','a_proc_delay']  #为服务建立的性能评估模块
trans_names=['d_trans_delay','a_trans_delay']  #为数据传输建立的性能评估模块

# 以下是初始化模块字典的过程
modules_serv=dict() #为服务建立的性能评估模块
for key1 in serv_names:
    modules_serv[key1]=dict()
    for key2 in ['cc','ch','hc','hh']:
        modules_serv[key1][key2]=dict()
        for key3 in ['360p','480p','720p','1080p']:
            modules_serv[key1][key2][key3]=dict()
            #for key4 in ['1fps','5fps','10fps','20fps','30fps']:
                 #modules_serv[key1][key2][key3][key4]=dict()
modules_trans=dict()  #为数据传输建立的性能评估模块
for key1 in trans_names:  
    modules_trans[key1]=dict()
    for key2 in ['cc','ch','hc','hh']:
        modules_trans[key1][key2]=dict()
        for key3 in ['360p','480p','720p','1080p']:
            modules_trans[key1][key2][key3]=dict()
            #for key4 in ['1fps','5fps','10fps','20fps','30fps']:
                 #modules_trans[key1][key2][key3][key4]=dict()
#服务模块1与2：人脸检测和姿态估计。配置包括：帧率(5种)；分辨率（4种）；云/边（4种类）
for key1 in ['360p','480p','720p','1080p']:
    #cc模式下各个帧率
    for name in serv_names:
        mode='cc'
        mode0='cc'
        modules_serv[name][mode][key1]['1fps']=result[mode0][key1][name]
        modules_serv[name][mode][key1]['5fps']=result[mode0][key1][name]*1.5
        modules_serv[name][mode][key1]['10fps']=result[mode0][key1][name]*2
        modules_serv[name][mode][key1]['20fps']=result[mode0][key1][name]*3
        modules_serv[name][mode][key1]['30fps']=result[mode0][key1][name]*4
        #hc模式下各个帧率
        mode='hc'
        mode0='hc'
        modules_serv[name][mode][key1]['1fps']=result[mode0][key1][name]
        modules_serv[name][mode][key1]['5fps']=result[mode0][key1][name]*1.5
        modules_serv[name][mode][key1]['10fps']=result[mode0][key1][name]*2
        modules_serv[name][mode][key1]['20fps']=result[mode0][key1][name]*3
        modules_serv[name][mode][key1]['30fps']=result[mode0][key1][name]*4
        #ch模式下各个帧率
        mode='ch'  
        mode0='cc'#没有测ch的，所以用cc代替
        modules_serv[name][mode][key1]['1fps']=result[mode0][key1][name]
        modules_serv[name][mode][key1]['5fps']=result[mode0][key1][name]*1.5
        modules_serv[name][mode][key1]['10fps']=result[mode0][key1][name]*2
        modules_serv[name][mode][key1]['20fps']=result[mode0][key1][name]*3
        modules_serv[name][mode][key1]['30fps']=result[mode0][key1][name]*4
        #hh模式下各个帧率
        mode='hh'
        mode0='hc'#没有测hh的，所以用hc代替并修改，假设hh模式下的处理时延是hc模式下的1.5倍
        modules_serv[name][mode][key1]['1fps']=result[mode0][key1][name]*1.5
        modules_serv[name][mode][key1]['5fps']=result[mode0][key1][name]*1.5*1.5
        modules_serv[name][mode][key1]['10fps']=result[mode0][key1][name]*2*1.5
        modules_serv[name][mode][key1]['20fps']=result[mode0][key1][name]*3*1.5
        modules_serv[name][mode][key1]['30fps']=result[mode0][key1][name]*4*1.5
#传输模块1与2：为人脸检测提供数据。配置包括：帧率(5种)；分辨率（4种）；云/边（4种类）
for key1 in ['360p','480p','720p','1080p']:
    #cc模式下各个帧率
    for name in trans_names:
        mode='cc'
        mode0='cc'
        modules_trans[name][mode][key1]['1fps']=result[mode0][key1][name]
        modules_trans[name][mode][key1]['5fps']=result[mode0][key1][name]*1.5
        modules_trans[name][mode][key1]['10fps']=result[mode0][key1][name]*2
        modules_trans[name][mode][key1]['20fps']=result[mode0][key1][name]*3
        modules_trans[name][mode][key1]['30fps']=result[mode0][key1][name]*4
        #hc模式下各个帧率
        mode='hc'
        mode0='hc'
        modules_trans[name][mode][key1]['1fps']=result[mode0][key1][name]
        modules_trans[name][mode][key1]['5fps']=result[mode0][key1][name]*1.5
        modules_trans[name][mode][key1]['10fps']=result[mode0][key1][name]*2
        modules_trans[name][mode][key1]['20fps']=result[mode0][key1][name]*3
        modules_trans[name][mode][key1]['30fps']=result[mode0][key1][name]*4
        #ch模式下各个帧率
        mode='ch'  
        mode0='cc'#没有测ch的，所以用cc代替，假设ch模式下的传输时延是cc模式下单的1.5倍
        modules_trans[name][mode][key1]['1fps']=result[mode0][key1][name]*1.5
        modules_trans[name][mode][key1]['5fps']=result[mode0][key1][name]*1.5*1.5
        modules_trans[name][mode][key1]['10fps']=result[mode0][key1][name]*2*1.5
        modules_trans[name][mode][key1]['20fps']=result[mode0][key1][name]*3*1.5
        modules_trans[name][mode][key1]['30fps']=result[mode0][key1][name]*4*1.5
        #hh模式下各个帧率
        mode='hh'
        mode0='hc'#没有测hh的，所以用hc代替并修改,假设hh模式下的传输时延是hc模式下的一半
        modules_trans[name][mode][key1]['1fps']=result[mode0][key1][name]*0.5
        modules_trans[name][mode][key1]['5fps']=result[mode0][key1][name]*1.5*0.5
        modules_trans[name][mode][key1]['10fps']=result[mode0][key1][name]*2*0.5
        modules_trans[name][mode][key1]['20fps']=result[mode0][key1][name]*3*0.5
        modules_trans[name][mode][key1]['30fps']=result[mode0][key1][name]*4*0.5

#经过前文操作，得到了知识库modules_serv与modules_trans，现在利用知识库来寻找最优配置。
#知识库是一个已经建立好的复杂模型。分别用遍历查表和贝叶斯优化两种方法试图寻找最优解。

idle_runs=0  #这个睡眠时间表示进行一次查表的代价。它越大，遍历查表和贝叶斯优化的区别越明显

def caluclate(mode,reso,fps):  #根据配置计算时延
    
    sum=float(0)
    time.sleep(idle_runs)
    for key1 in serv_names:  #为服务建立的性能评估模块
        sum=sum+modules_serv[key1][mode][reso][fps]
    for key1 in trans_names:
        sum=sum+modules_trans[key1][mode][reso][fps]
    return sum



#以遍历的方式来寻找最优解

def iteration():
    min_value=99
    best_conf=dict()
    best_conf['mode']='hh'
    best_conf['reso']= '1080p'
    best_conf['fps']= '30fps'
    for mode in ['hh','hc','ch','cc']:
        for reso in ['1080p','720p','480p','360p']:
            for fps in ['30fps','20fps','10fps','5fps','1fps']:
                sum=float(0)
                sum=caluclate(mode,reso,fps)
                if sum<min_value:
                    min_value=sum
                    best_conf['mode']=mode
                    best_conf['reso']= reso
                    best_conf['fps']= fps
    return best_conf

#使用贝叶斯优化来解决问题：
def objective(trial):
    mode = trial.suggest_categorical('mode',['hh','hc','ch','cc'])
    reso = trial.suggest_categorical('reso',['1080p','720p','480p','360p'])
    fps  = trial.suggest_categorical('fps',['30fps','20fps','10fps','5fps','1fps'])
    sum=caluclate(mode,reso,fps)

    return sum
# 创建一个study对象并调用该optimize方法超过 100 次试验
study = optuna.create_study()


delay1_list=[]
delay2_list=[]
ans1_list=[]
ans2_list=[]
item1=0.01
item2=40
'''  #item1分析不同代价的影响,item2分析不同搜索次数的影响
# for item1 in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]:
# for item2 in [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]:
    idle_runs=item1
    start_time = time.time()
    best_conf1=iteration()
    end_time = time.time()
    delay1=end_time - start_time
    ans1=caluclate(best_conf1['mode'],best_conf1['reso'],best_conf1['fps'])

    start_time = time.time()
    study.optimize(objective, n_trials=item2)  #60轮
    end_time = time.time()
    delay2=end_time - start_time
    ans2=caluclate(study.best_params['mode'],study.best_params['reso'],study.best_params['fps'])

    delay1_list.append(delay1)
    ans1_list.append(ans1)
    delay2_list.append(delay2)
    ans2_list.append(ans2)
#'''
print("遍历法：")
print(delay1_list)
print(ans1_list)

print("贝叶斯优化法：")
print(delay2_list)
print(ans2_list)

delay1_list=[1.2435669898986816, 2.490180492401123, 2.503232479095459, 3.736240863800049, 4.985394716262817, 4.998213291168213, 6.228398561477661, 7.42325234413147, 7.481369733810425, 8.714864492416382]

delay2_list=[0.6883533000946045, 1.388132095336914, 2.013535499572754, 2.433228015899658, 2.92277193069458, 3.297698736190796, 4.167840242385864, 4.5933897495269775, 5.189121246337891, 5.616417407989502]

idle_runs_list=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
numer_trails_list=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]

#现在要进行绘图
fig, axs = plt.subplots(2,2,figsize=(15, 8), dpi=80)

ans1_list=[0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203]

ans2_list=[0.13082316627869217, 0.10903347358548411, 0.10903347358548411, 0.10903347358548411, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203, 0.10705417367128203]

axs[0,0].plot(idle_runs_list, delay1_list, label='循环遍历')
axs[0,0].plot(idle_runs_list, delay2_list, label='贝叶斯优化')
axs[0,0].set_ylabel('查表时延(s)')
axs[0,0].set_xlabel('单次查表代价(s)')
axs[0,0].legend()


axs[0,1].plot(numer_trails_list, ans1_list, label='循环遍历')
axs[0,1].plot(numer_trails_list, ans2_list, label='贝叶斯优化')
axs[0,1].set_ylabel('最优时延(s)')
axs[0,1].set_xlabel('贝叶斯优化搜索次数')
axs[0,1].legend()



plt.show()
