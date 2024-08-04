# 由于暑期要帮助周文晖完成工作，此处交代一些必要的代码。


import os  
import json  
import csv
import pandas as pd
from AccuracyPrediction import AccuracyPrediction
import numpy as np

reso_range = ["360p", "480p", "540p", "630p", "720p", "810p", "900p", "990p", "1080p"]

#(1)首先要得到cpu使用率如何影响时延的曲线

need_to_get_csv=0
if need_to_get_csv==1:
    # 设定包含JSON文件的目录路径  
    directory_path = 'kb_data_90i90_no_clst-1 重要备份/face_detection-gender_classification/gender_classification'


    reso_cpu_dict={}

   
    for reso_value in reso_range:
        reso_cpu_dict[reso_value]={}


    # 遍历目录下的所有文件  
    for filename in os.listdir(directory_path):  
        if filename.endswith(".json"):  # 确保只处理JSON文件  
            file_path = os.path.join(directory_path, filename)  
            
            # 读取JSON文件  
            with open(file_path, 'r', encoding='utf-8') as file:  
                data = json.load(file)  # 加载JSON数据  
                
                # 遍历并打印每个键值对  
                for conf_str, delay in data.items(): 
                    # conf_str形如"reso=990p fps=6 encoder=JPEG gender_classification_ip=192.168.1.7 gender_classification_cpu_util_limit=0.2 gender_classification_mem_util_limit=1.0 " 
                    params = {k: v for k, v in (item.split('=') for item in conf_str.split())}  
                    # params形如：
                    '''
                    {
                        'reso': '360p', 
                        'fps': '1', 'encoder': 'JPEG', 
                        'gender_classification_ip': '114.212.81.11', 
                        'gender_classification_cpu_util_limit': '1.0', 
                        'gender_classification_mem_util_limit': '1.0'
                    }
                    '''
                    temp_cpu = params['gender_classification_cpu_util_limit']
                    if temp_cpu != '1.0':
                        temp_fps = float(int(params['fps']))
                        temp_value = delay * ( 30 / temp_fps)
                        temp_reso = params['reso']
                        reso_cpu_dict[temp_reso][temp_cpu]=temp_value


    for reso_key in reso_cpu_dict.keys():
        print(reso_key)

        sorted_cpu_delay = sorted(reso_cpu_dict[reso_key].items(), key=lambda x: float(x[0])) 

        csv_filename = 'kb_test/' + reso_key+'.csv'
        with open(csv_filename, mode='w', newline='') as file:  
        # 创建一个csv.DictWriter对象，用于写入数据  
        # fieldnames指定了CSV文件中每列的列名  
            writer = csv.DictWriter(file, fieldnames=['cpu', 'delay'])  
            
            # 写入列名（即字段名）  
            writer.writeheader()  

            # 遍历字典中的每个键值对  
            for cpu, delay in sorted_cpu_delay:  
                # 将键转换为浮点数  
                cpu = float(cpu)  
                # 值已经是浮点数，所以直接使用  
                
                # 创建一个字典，其中包含要写入CSV文件的列名和相应的值  
                row_dict = {'cpu': cpu, 'delay': delay}  
                print(row_dict)
                
                # 将字典写入CSV文件  
                writer.writerow(row_dict)  


need_to_read_csv = 0

if need_to_read_csv==1:
    # 
    draw_dict={}

    # 我只需要540、720、900三种reso下，0.05，0.15，0.25，0.35，0.45，0.55，0.65
    cpu_range = [0.05,0.15,0.25,0.35,0.45,0.55,0.65]
    reso_range = ['540p','720p','900p']

    for reso_value in reso_range:
        draw_dict[reso_value]=[]

        csv_file_path = 'kb_test/'+ reso_value +'.csv'
        
        df = pd.read_csv(csv_file_path)

        cpus = df['cpu'].tolist() 
        delays = df['delay'].tolist()

        for cpu_value in cpu_range:
            draw_dict[reso_value].append(delays[cpus.index(cpu_value)])
    print('最终')
    print(draw_dict)

# 现在就得到如下数据，可以用来画图
'''
cpu_range = [0.05,0.15,0.25,0.35,0.45,0.55,0.65]
draw_dict = {
    '540p': [0.6412476824040998, 0.2013094971577325, 0.1097547320219186, 0.1282310930098065, 0.1071532726287841, 0.1047744132854319, 0.1585455238819121], 
    '720p': [0.6911597207740502, 0.1698428818157741, 0.1123861117022377, 0.1154499530792235, 0.183171731430095, 0.1172786691914433, 0.1685628175735473], 
    '900p': [0.4911928546839747, 0.3265442362538084, 0.1592974624936541, 0.1035806228374612, 0.0838429927825927, 0.1018228407563834, 0.0995221631280306]
}
'''

# 但是上面这个数据还不够好，所以直接对于360，720，1080三个分辨率在[0.05,0.15,0.25,0.35,0.45]的cpu_range下进行重新采样，然后保存到csv文件，准备重新读取
need_to_read_csv_2 = 0
if need_to_read_csv_2 == 1:
    # 'kb_test\face_detection-gender_classification\record_data\20240725_11_33_07_kb_sec_builder_div_test_0.3_tight_build_gender_classify_cold_start04.csv'
    # 'kb_test/face_detection-gender_classification/record_data/20240725_16_20_02_kb_sec_builder_div_test_0.3_tight_build_gender_classify_cold_start04.csv'
    df = pd.read_csv('kb_test/face_detection-gender_classification/record_data/20240725_16_20_02_kb_sec_builder_div_test_0.3_tight_build_gender_classify_cold_start04.csv')  

    reso_range = ['360p','720p','1080p']
    cpu_range = [0.05,0.15,0.25,0.35,0.45]

    csv_dict = {}

    for reso in reso_range:

        csv_dict[reso]={}

        for cpu in cpu_range:

            csv_dict[reso][cpu]={}

            '''
            # 人脸检测相关
            filtered_df = df[(df['reso'] == reso) &   
                 (df['face_detection_role'] == 'host') &   
                 (df['face_detection_cpu_util_limit'] == cpu)]  
            '''
            filtered_df = df[(df['reso'] == reso) &   
                 (df['gender_classification_role'] == 'host') &   
                 (df['gender_classification_cpu_util_limit'] == cpu)]  
            
            delays = filtered_df['gender_classification_proc_delay'].tolist()
            avg_delay = np.mean(delays)  
            std_delay = np.std(delays, ddof=1)  # 使用样本标准差

            csv_dict[reso][cpu]['delays'] = delays
            csv_dict[reso][cpu]['avg_delay']=avg_delay
            csv_dict[reso][cpu]['std_delay']=std_delay
    
    
    draw_dict={}

    for reso in reso_range:
        print(reso)

        draw_dict[reso]={}

        draw_dict[reso]['avg_delay'] = []
        draw_dict[reso]['std_delay'] = []

        for cpu in cpu_range:
            draw_dict[reso]['avg_delay'].append(csv_dict[reso][cpu]['avg_delay'])
            draw_dict[reso]['std_delay'].append(csv_dict[reso][cpu]['std_delay'])

        print(draw_dict[reso]['avg_delay'])
        print(draw_dict[reso]['std_delay'])
        
        '''
        face_detection在不同分辨率下受cpu的影响

        360 cost [0.45977723598480225, 0.14756947755813596, 0.13315094841851124, 0.13337804377079004, 0.1349070866902669]
        360 err  [0.04465865414792748, 0.0074842254076316105, 0.0011817970211527346, 0.0019102512378889958, 0.004817448934968905]
        
        720 cost [2.561431606610616, 0.7787640775953021, 0.7117699554988316, 0.6956813931465149, 0.7162426710128784]
        720 err [0.06760605579562678, 0.010809327293900658, 0.041645315777474004, 0.002524075912894186, 0.035815477477156]
        
        1080 cost [5.800383408864339, 1.764948546886444, 1.5821207761764526, 1.5672398328781127, 1.5676712036132812]
        1080 err [0.11267869119391775, 0.024364995391060403, 0.033173067816923835, 0.0021265514357853388, 0.00392053509037608]
        
        
        gender_classification在不同分辨率下受cpu的影响
        
        360 cost [0.6781484087308248, 0.145938828587532, 0.08548046482933888, 0.06822095976935488, 0.057641363143920855]
        360 err  [0.07735129866192345, 0.014793195764013183, 0.01682239456538382, 0.0077072244802264996, 0.012526349324752719]

        720 cost [0.6466315133231026, 0.1547281401497977, 0.08377132813135779, 0.06801446278889971, 0.07259888308388841]
        720 err [0.13701152152029752, 0.021850445952907433, 0.01524865915462171, 0.025411488178334978, 0.03207688598447997]
        
        1080 cost [0.5668943723042806, 0.15490668160574775, 0.09253084659576412, 0.06353262492588585, 0.057404994964599575]
        1080 err [0.059495582593470854, 0.0227993355014402, 0.027493302571038703, 0.009254091506227143, 0.009171833039690684]
        '''
    
            





#(2)然后要得到fps如何影响精度的曲线

# 首先要获取计算精度的模型



need_to_cal_acc = 0

if need_to_cal_acc==1:

    task_accuracy = 1.0
    acc_pre = AccuracyPrediction()
    obj_size = None 


    obj_speed_list = [200,400,600,800]

    draw_dict={}

    for obj_speed in obj_speed_list:

        draw_dict[obj_speed]=[]

        for fps in [i + 1 for i in range(30)]:

            task_accuracy = acc_pre.predict(service_name='face_detection', service_conf={
                'fps':fps,
                'reso':'1080p'
            }, obj_size=obj_size, obj_speed=obj_speed)
            task_accuracy = round(task_accuracy,2)
            draw_dict[obj_speed].append(task_accuracy)
    
    for speed_key in draw_dict.keys():
        print(speed_key, draw_dict[speed_key])
    
    '''
    对于1-30帧,相关情况如下,可以看出,速度对精度确实有着显著的影响。当然，帧率对精度的影响也很显著。
    200 [0.65, 0.83, 0.91, 0.94, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97]
    400 [0.42, 0.71, 0.84, 0.91, 0.93, 0.95, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96]
    600 [0.31, 0.48, 0.6, 0.69, 0.77, 0.82, 0.86, 0.89, 0.91, 0.92, 0.94, 0.95, 0.95, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97]
    800 [0.29, 0.41, 0.5, 0.58, 0.65, 0.71, 0.75, 0.79, 0.82, 0.85, 0.87, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98] 
    '''


need_to_sort = 1

if need_to_sort == 1:

    reso_list = ['360p','480p','540p','630p','720p','900p','1080p']
    fps_list  = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]

    task_accuracy = 1.0
    acc_pre = AccuracyPrediction()
  
    # 遍历所有组合  
    combinations_with_task_accuracy = []
    for reso in reso_list:  
        for fps in fps_list:  

            task_accuracy = acc_pre.predict(service_name='face_detection', service_conf={
                'fps':fps,
                'reso':reso
            }, obj_size=None, obj_speed=100)
            task_accuracy = round(task_accuracy,2)


  
            combinations_with_task_accuracy.append({'resolution': reso, 'fps': fps, 'task_accuracy':task_accuracy})  

    # 根据精度从大到小排序  
    sorted_combinations = sorted(combinations_with_task_accuracy, key=lambda x: x['task_accuracy'], reverse=True)  
    
    # 移除'precision'键，仅保留'resolution'和'fps'  
    result = [{'resolution': x['resolution'], 'fps': x['fps']} for x in sorted_combinations]  
    
    # 用以上结果模拟240p,360p,480p,540p,720p,900p,1080p下的情况
    def replace_resolution(resolution):  
        if resolution == '630p':  
            return '540p'  
        elif resolution == '540p':  
            return '480p'  
        elif resolution == '480p':  
            return '360p'  
        elif resolution == '360p':  
            return '240p'  
        else:  
            return resolution  # 如果不是这些分辨率之一，就保持不变  
  
    # 遍历列表并替换分辨率  
    updated_result = [{'resolution': replace_resolution(spec['resolution']), 'fps': spec['fps']} for spec in result]  
    
    # 打印更新后的列表  
    print(updated_result)



        
    
   
    







                
  


            