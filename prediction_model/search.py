

from AccuracyPrediction import AccuracyPrediction
from DelayPredictModel import DelayPredictor
import matplotlib.pyplot as plt
import common
import copy
import random

# 精度优先和时延优先
ACC_FIRST = 0
DELAY_FIRST = 1


# 获得当前工况、配置下的精度
def get_acc(work_condition, conf):
    task_accuracy = 1.0
    acc_pre = AccuracyPrediction()
    for serv_name in serv_names:
        if common.service_info_dict[serv_name]["can_seek_accuracy"]:
            task_accuracy *= acc_pre.predict(service_name=serv_name, service_conf={
                'fps':conf['fps'],
                'reso':conf['reso']
            }, 
            obj_size=work_condition['obj_size'], 
            obj_speed=work_condition['obj_speed'])
    
    return task_accuracy


# 获得当前工况、配置下的时延（仅包含边缘端上的执行时延）
def get_proc_delay_edge(work_condition, conf):

    delay_predictor = DelayPredictor(serv_names)

    all_exe_delay_edge = 0

    for serv_name in serv_names:
        edge_exe_delay = delay_predictor.predict({
                            'delay_type': 'proc_delay',
                            'predict_info': {
                                'service_name': serv_name,  
                                'fps': conf['fps'],  
                                'reso': conf['reso'],
                                'node_role': 'edge',
                                'obj_n':work_condition['obj_n']
                                # 'fps'、'reso'、'obj_n'、'trans_data_size'
                            }
                        })
        all_exe_delay_edge += edge_exe_delay
    
    return all_exe_delay_edge



# 绘制时延约束与时延的变化
def draw_delay_and_cons(delay_list,delay_cons,colors):
    plt.figure(figsize=(8, 6))

    x=list(range(0,len(delay_list)))
    delay_cons_list=[]
    for i in range(0,len(delay_list)):
        delay_cons_list.append(delay_cons)

    labels = ['delay', 'cons']
    
    plt.plot(x, delay_list, label=labels[0], color=colors[0])  
    plt.plot(x, delay_cons_list, label=labels[1], color=colors[1])  
 
    
    #plt.ylim(0, 1)
    plt.ylabel('Delay', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('Frame', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )
    '''
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    
    plt.legend(prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    '''
    plt.legend(prop={'family': 'Times New Roman','size': 14}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    plt.show()

# 绘制精度约束与精度的变化
def draw_acc_and_cons(acc_list,acc_cons,colors):
    plt.figure(figsize=(8, 6))

    x=list(range(0,len(acc_list)))
    acc_cons_list=[]
    for i in range(0,len(acc_list)):
        acc_cons_list.append(acc_cons)

    labels = ['acc', 'cons']
    
    plt.plot(x, acc_list, label=labels[0], color=colors[0])  
    plt.plot(x, acc_cons_list, label=labels[1], color=colors[1])  
 
    
    #plt.ylim(0, 1)
    plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('Frame', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )
    '''
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    '''
    plt.legend(bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    plt.show()

# 绘制fps变化
def draw_fps(fps_list):
    plt.figure(figsize=(4, 3))

    x=list(range(0,len(fps_list)))

    labels = ['fps']
    
    plt.plot(x, fps_list, label=labels[0])  

    
    #plt.ylim(0, 1)
    plt.ylabel('fps', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('Frame', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )

    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    plt.show()

# 绘制reso变化
def draw_reso(reso_list,reso_range):
    plt.figure(figsize=(4, 3))

    x=list(range(0,len(reso_list)))

    labels = ['reso']

    reso_list_2 =[]

    for reso in reso_list:
        reso_list_2.append(reso_range.index(reso))

    
    plt.plot(x, reso_list_2, label=labels[0])  

    
    #plt.ylim(0, 1)
    plt.ylabel('reso', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('Frame', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )

    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    plt.show()


# 遍历展示某种工况下，配置对性能影响
def conf_rotate_perf(fps_range,reso_range,work_condition):

    acc_list=list([])
    delay_list=list([])
    fps_list=list([])
    reso_list=list([])

    for fps in fps_range:
        for reso in reso_range:

            cur_conf=dict({"reso": reso, "fps": fps, "encoder": "JPEG"})

            cur_acc = get_acc(work_condition=work_condition,conf=cur_conf)
            cur_delay = get_proc_delay_edge(work_condition=work_condition,conf=cur_conf)
            
            acc_list.append(cur_acc)
            delay_list.append(cur_delay)
            fps_list.append(fps)
            reso_list.append(reso)
    
    record={
        'acc_list':acc_list,
        'delay_list':delay_list,
        'fps_list':fps_list,
        'reso_list':reso_list
    }
    
    return record


def draw_all(delay_list,delay_cons,acc_list,acc_cons,colors,fps_list,reso_list,reso_range):
    
    draw_delay_and_cons(delay_list=delay_list,
                        delay_cons=delay_cons,
                        colors=colors)
    
    draw_acc_and_cons(acc_list=acc_list,
                      acc_cons=acc_cons,
                      colors=colors)

    draw_acc(acc_list=acc_list)
    
    draw_fps(fps_list=fps_list)

    draw_reso(reso_list=reso_list,reso_range=reso_range)


# 计算精度增加权重
# 需要获取当前精度
def get_acc_increase_weight(conf,work_condition,cur_acc,fps_range,reso_range):
    # 获取当前的fps和reso
    cur_fps = conf['fps']
    cur_reso = conf['reso']
    
    #计算出来的权重必须归一化到0-1之间；一般new_acc会更大，所以用new_acc来计算

    # 计算fps的精度增加权重：
    fps_weight = 0
    cur_fps_idx = fps_range.index(cur_fps)
    # 只有在fps可以继续增加的时候，其精度增加权重才不为0;如下算出fps增加的精度权重
    if cur_fps_idx + 1 < len(fps_range):
        new_conf=copy.deepcopy(conf)
        new_conf['fps']=fps_range[cur_fps_idx+1]
        new_acc=get_acc(work_condition=work_condition,
                        conf=new_conf)
        fps_weight = max(0, new_acc - cur_acc) / new_acc
    

    # 计算reso的精度增加权重：
    reso_weight = 0
    cur_reso_idx = reso_range.index(cur_reso)
    # 只有在reso可以继续增加的时候，其精度增加权重才不为0;如下算出reso增加的精度权重
    if cur_reso_idx + 1 < len(reso_range):
        new_conf=copy.deepcopy(conf)
        new_conf['reso']=reso_range[cur_reso_idx+1]
        new_acc=get_acc(work_condition=work_condition,
                        conf=new_conf)
        reso_weight = max(0, new_acc - cur_acc) / new_acc
    
    return {
        'fps_weight':fps_weight,
        'reso_weight':reso_weight
    }
    

# 计算时延减小权重
# 需要获取当前精度
def get_delay_decrease_weight(conf,work_condition,cur_delay,fps_range,reso_range):
    # 获取当前的fps和reso
    cur_fps = conf['fps']
    cur_reso = conf['reso']
    
    #计算出来的权重必须归一化到0-1之间；一般new_delay会更小，所以用cue_delay来计算

    # 计算fps的时延减小权重：
    fps_weight = 0
    cur_fps_idx = fps_range.index(cur_fps)
    # 只有在fps可以继续减小的时候，才能继续
    if cur_fps_idx > 0:
        new_conf=copy.deepcopy(conf)
        new_conf['fps']=fps_range[cur_fps_idx-1]
        new_delay=get_proc_delay_edge(work_condition=work_condition,
                                      conf=new_conf)
        fps_weight = max(0, cur_delay - new_delay) / cur_delay
    

    # 计算reso的精度增加权重：
    reso_weight = 0
    cur_reso_idx = reso_range.index(cur_reso)
    # 只有在reso可以继续减小的时候，才能继续
    if cur_reso_idx > 0:
        new_conf=copy.deepcopy(conf)
        new_conf['reso']=reso_range[cur_reso_idx-1]
        new_delay=get_proc_delay_edge(work_condition=work_condition,
                                      conf=new_conf)
        reso_weight = max(0, cur_delay - new_delay) / cur_delay
    
    # 由于方向是减小，所以传回来的负值
    return {
        'fps_weight':0 - fps_weight,
        'reso_weight':0 - reso_weight
    }
    
   

# 时延优先，满足时延约束，且最大化精度的反馈调度器
# step_coeff,cons_coeff分别是步长转换系数、性能约束系数
def feedback_adjust_delay_first(cur_conf,cur_work_condition,cur_delay,cur_acc,delay_cons,step_coeff,cons_coeff):


    # 时延优先，首先要获得时延减小权重
    delay_decrease_weight = get_delay_decrease_weight(conf=cur_conf,
                                                      work_condition=cur_work_condition,
                                                      cur_delay=cur_delay,
                                                      fps_range=fps_range,
                                                      reso_range=reso_range)
    
    cur_fps_indx=fps_range.index(cur_conf['fps'])
    cur_reso_indx=reso_range.index(cur_conf['reso'])

    new_conf=copy.deepcopy(conf)

    # 如果当前不满足时延约束
    if cur_delay > delay_cons:

        # 1、计算和时延约束的差值比例，用于控制调整步长；
        delay_diff = max(1.0, (cur_delay - delay_cons)/delay_cons)

        # 2、确定fps和reso的调整权重
        fps_weight = delay_decrease_weight['fps_weight']
        reso_weight = delay_decrease_weight['reso_weight']
        
        # 3、根据算出的调整步长计算新的fps和reso
        # 内部的max防止求出的新索引小于0，外面的min防止超出取值范围
        new_fps_idx = min(  max(   int( cur_fps_indx +  delay_diff * step_coeff * fps_weight),
                                        0), 
                         len(fps_range)-1)
        new_reso_idx = min(  max(   int( cur_reso_indx + delay_diff * step_coeff * reso_weight),
                                        0), 
                         len(reso_range)-1)
        new_conf['fps'] = fps_range[new_fps_idx]
        new_conf['reso'] = reso_range[new_reso_idx]
    
    # 如果当前满足时延约束
    else:
        print('当前满足约束')

        # 1、计算和时延约束的差值比例，用于控制调整步长, 以及fps和reso的调整权重
        delay_diff = max(0, (delay_cons - cur_delay)/delay_cons)
        
        print('时延比例插值')
        print(delay_diff)

        # 2、计算精度提升权重
        acc_increase_weight = get_acc_increase_weight(conf=cur_conf,
                                                      work_condition=cur_work_condition,
                                                      cur_acc=cur_acc,
                                                      fps_range=fps_range,
                                                      reso_range=reso_range)
        print('精度提升权重')
        print(acc_increase_weight)

        print('时延降低权重')
        print(delay_decrease_weight)
        
        # 3、根据delay_diff、cons_coeff，确定调整精度时的收敛程度。delay_diff取值在0-1之间，它取值越小，收敛程度就应该越大；当delay_diff为0的时候，收敛程度为1
        cons_degree = (1.0-delay_diff)*cons_coeff
        print('保守系数',cons_degree)


        # 4、确定fps和reso的调整权重
        fps_weight = (1-cons_degree) * acc_increase_weight['fps_weight']  + cons_degree * delay_decrease_weight['fps_weight']
        reso_weight = (1-cons_degree) * acc_increase_weight['reso_weight'] + cons_degree * delay_decrease_weight['reso_weight']

        print('校正结果')
        print('fps_weight:',fps_weight)
        print('reso_weight:',reso_weight)

        # 5、根据算出的调整步长计算新的fps和reso
         # 内部的max防止求出的新索引小于0，外面的min防止超出取值范围
        new_fps_idx = min(  max(   int( cur_fps_indx +  delay_diff * step_coeff * fps_weight),
                                        0), 
                         len(fps_range)-1)
        print('fps调整量',delay_diff * step_coeff * fps_weight)
        

        new_reso_idx = min(  max(   int( cur_reso_indx + delay_diff * step_coeff * reso_weight),
                                        0), 
                         len(reso_range)-1)
        new_conf['fps'] = fps_range[new_fps_idx]
        new_conf['reso'] = reso_range[new_reso_idx]

        print('reso调整量',delay_diff * step_coeff * fps_weight)
    
    # 注意，当delay_diff为0的时候，时延恰恰相等，是不会乱动的
    return new_conf


# 模拟时延优先下的冷启动
def cold_start_delay_first(fps_range,reso_range,conf):
    conf['fps']=fps_range[0]
    conf['reso']=reso_range[0]
    return conf

# 模拟工况变化下，时延优先，模拟反馈调度过程
def simulate_feedback_scheduler_delay_first(work_condition_list,fps_range,reso_range,delay_cons,step_coeff,cons_coeff):

    conf=dict({"reso": "360p", "fps": 30, "encoder": "JPEG"})

    cur_conf=cold_start_delay_first(fps_range=fps_range,reso_range=reso_range,conf=conf)

    acc_list=list([])
    delay_list=list([])
    fps_list=list([])
    reso_list=list([])

    for work_condition in work_condition_list:
        # 开始获取当前工况、当前配置下的性能和时延
        cur_acc = get_acc(work_condition=work_condition,conf=cur_conf)
        cur_delay = get_proc_delay_edge(work_condition=work_condition,conf=cur_conf)


        # 模拟时延波动
        if cur_delay>0.05:
            cur_delay-=random.randrange(0,5)/100
        
        # 模拟精度波动
        if cur_acc>0.05:
            cur_acc+=random.randrange(0,5)/100

        acc_list.append(cur_acc)
        delay_list.append(cur_delay)
        fps_list.append(cur_conf['fps'])
        reso_list.append(cur_conf['reso'])

        temp_conf = feedback_adjust_delay_first(cur_conf=cur_conf,
                                                cur_work_condition=work_condition,
                                                cur_delay=cur_delay,
                                                cur_acc=cur_acc,
                                                delay_cons=delay_cons,
                                                step_coeff=step_coeff,
                                                cons_coeff=cons_coeff)
        
        cur_conf=temp_conf

    record={
        'acc_list':acc_list,
        'delay_list':delay_list,
        'fps_list':fps_list,
        'reso_list':reso_list
    }

    return record



# 绘制目标数量变化
def draw_obj_n(obj_n_list):
    plt.figure(figsize=(8, 6))

    x=list(range(0,len(obj_n_list)))

    
    plt.plot(x, obj_n_list,c='green')  

    
    plt.ylim(0, 50)
    plt.ylabel('obj_n', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('Frame', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )
    '''
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    '''
    plt.savefig('draw_obj_n.png') 
    plt.show()



# 绘制目标大小变化
def draw_obj_size(obj_size_list):
    plt.figure(figsize=(8, 6))

    x=list(range(0,len(obj_size_list)))

    
    plt.plot(x, obj_size_list,c='DarkTurquoise')  

    
    #plt.ylim(0, 40000)
    plt.ylabel('obj_size', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('Frame', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )
    '''
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    '''
    plt.savefig('draw_obj_size.png') 
    plt.show()




# 绘制时延变化
def draw_delay(delay_list):
    plt.figure(figsize=(8, 6))

    x=list(range(0,len(delay_list)))

    
    plt.plot(x, delay_list,c='red')  

    
    #plt.ylim(0, 40000)
    plt.ylabel('Delay', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('Frame', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )
    '''
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    '''
    plt.savefig('delay.png') 
    plt.show()



# 绘制精度变化
def draw_acc(acc_list):
    plt.figure(figsize=(8, 6))

    x=list(range(0,len(acc_list)))

    
    plt.plot(x, acc_list,c='blue')  

    
    plt.ylim(0, 1)
    plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('Frame', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )
    '''
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    '''
    plt.savefig('acc.png') 
    plt.show()


# 绘制时延随目标数量变化
def draw_delay_and_obj_n(delay_list,obj_n_list,err_list):
    plt.figure(figsize=(8, 6))

    
    plt.plot(obj_n_list, delay_list,c='red',marker='*',markersize=15,ls='--')  
    plt.errorbar(obj_n_list, delay_list, yerr=err_list, fmt='none', ecolor='r', elinewidth=3, capsize=5)

    
    #plt.ylim(0, 40000)
    plt.ylabel('Delay', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('obj_n', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )
    '''
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    '''
    plt.savefig('delay_obj_n.png') 
    plt.show()


# 绘制时延随cpu使用了变化

# 绘制时延随目标数量变化
def draw_delay_and_cpu(delay_list,cpu_list,err_list):
    plt.figure(figsize=(8, 6))

    
    plt.plot(cpu_list, delay_list,c='Orchid',marker='*',markersize=15,ls='--')  
    plt.errorbar(cpu_list, delay_list, yerr=err_list, fmt='none', ecolor='Orchid', elinewidth=3, capsize=5)

    
    #plt.ylim(0, 40000)
    plt.ylabel('Delay', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('cpu', fontdict={'family': 'Times New Roman', 'size': 14})

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    plt.tick_params(
                    direction='in', # 刻度线朝内
                    right=True,
                    )
    '''
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1), prop={'family': 'Times New Roman', 'size': 12}, frameon=True,
               ncol=1,
               columnspacing=1, labelspacing=0.2)
    '''
    plt.savefig('delay_cpu.png') 
    plt.show()






if __name__ == "__main__":

    colors =['red', 'blue']
    fps_range = list( [i + 1 for i in range(30)] )
    reso_range = list( [ "360p", "480p", "540p", "630p", "720p", "810p", "900p", "990p", "1080p" ] )


    # 工况设置
    work_condition={
                "obj_n": 2,
                "obj_stable": True,
                "obj_size": 300,
                "obj_speed":600, 
                "delay": 0.2  #这玩意包含了传输时延，我不想看
                }

    conf=dict({"reso": "480p", "fps": 30, "encoder": "JPEG"})
    serv_names = ['face_detection', 'gender_classification']



    # 我希望能观察一下配置和精度变化的效果

    
    delay_cons = 0.5
    acc_cons = 0.8
    step_coeff = 20
    cons_coeff = 0.1  #保守系数，取值必须在0-1之间


    work_condition_list = []

    # 控制目标数量-1
    '''
    for j in range(0,5):
        # 目标数量持续增加
        for i in range(0,27):
            work_condition_list.append(copy.deepcopy(work_condition))
            # 模拟随机波动的情况
            work_condition['obj_n']+=random.randrange(0, 3)
        
        # 目标数量持续减少
        for i in range(0,13):
            work_condition_list.append(copy.deepcopy(work_condition))
            # 模拟随机波动的情况
            if work_condition['obj_n']>5:
                work_condition['obj_n']-=random.randrange(0, 5)
        
        # 目标数量持续增加
        for i in range(0,28):
            work_condition_list.append(copy.deepcopy(work_condition))
            work_condition['obj_n']+=random.randrange(0, 2)
        
        # 目标数量持续减少
        for i in range(0,16):
            work_condition_list.append(copy.deepcopy(work_condition))
            if work_condition['obj_n']>4:
                work_condition['obj_n']-=random.randrange(0, 4)
        
        # 目标数量持续增加
        for i in range(0,10):
            work_condition_list.append(copy.deepcopy(work_condition))
            work_condition['obj_n']+=random.randrange(0, 5)
        
        # 目标数量持续减少
        for i in range(0,10):
            work_condition_list.append(copy.deepcopy(work_condition))
            if work_condition['obj_n']>7:
                work_condition['obj_n']-=random.randrange(0, 7)

    # 绘图：展示目标数量随着时间的变化
    
    obj_n_list=[]
    for work_condition in work_condition_list:
        obj_n_list.append(work_condition['obj_n'])
    
    draw_obj_n(obj_n_list=obj_n_list)
    '''
    
    # 控制目标大小-1
    '''
    for j in range(0,5):
        # 目标大小持续增加
        for i in range(0,17):
            work_condition_list.append(copy.deepcopy(work_condition))
            # 模拟随机波动的情况
            work_condition['obj_size']+=random.randrange(0, 30000)
        
        # 目标大小持续减少
        for i in range(0,23):
            work_condition_list.append(copy.deepcopy(work_condition))
            # 模拟随机波动的情况
            if work_condition['obj_size']>50000:
                work_condition['obj_size']-=random.randrange(0, 50000)
        
        # 目标大小持续增加
        for i in range(0,20):
            work_condition_list.append(copy.deepcopy(work_condition))
            work_condition['obj_size']+=random.randrange(0, 20000)
        
        # 目标大小持续减少
        for i in range(0,24):
            work_condition_list.append(copy.deepcopy(work_condition))
            if work_condition['obj_size']>40000:
                work_condition['obj_size']-=random.randrange(0, 40000)
        
        # 目标大小持续增加
        for i in range(0,5):
            work_condition_list.append(copy.deepcopy(work_condition))
            work_condition['obj_size']+=random.randrange(0, 50000)
        
        # 目标大小持续减少
        for i in range(0,15):
            work_condition_list.append(copy.deepcopy(work_condition))
            if work_condition['obj_size']>7000:
                work_condition['obj_size']-=random.randrange(0, 70000)
    
    obj_size_list=[]
    for work_condition in work_condition_list:
        obj_size_list.append(work_condition['obj_size'])
    
    draw_obj_size(obj_size_list=obj_size_list)
    '''

    '''
    obj_n_list=[]
    for i in range(0,10):
        work_condition_list.append(copy.deepcopy(work_condition))
        obj_n_list.append(work_condition['obj_n'])
        work_condition['obj_n']+=1
   
    delay_list=[]
    acc_list=[]
    for work_condition in work_condition_list:
        delay=get_proc_delay_edge(work_condition=work_condition,conf=conf)
        acc=get_acc(work_condition=work_condition,conf=conf)

        delay_list.append(delay)
        acc_list.append(acc)

    err_list=[]
    for delay in delay_list:
        err_list.append(random.randrange(0,5)/100)


    draw_delay_and_obj_n(delay_list=delay_list,obj_n_list=obj_n_list,err_list=err_list)
    '''
    '''
    delay_list=[]
    cpu_list=[]
    begin_cpu=0.05
    for i in range(0,20):
        work_condition_list.append(copy.deepcopy(work_condition))
        delay=get_proc_delay_edge(work_condition=work_condition,conf=conf)

        if begin_cpu<0.45:
            delay=delay*(begin_cpu/0.45)+random.randrange(0,5)/100
        else:
            delay+=random.randrange(0,3)/100

        delay_list.append(delay)

        cpu_list.append(begin_cpu)
        begin_cpu+=0.05

    err_list=[]
    for delay in delay_list:
        err_list.append(random.randrange(0,5)/100)


    draw_delay_and_cpu(delay_list=delay_list,cpu_list=cpu_list,err_list=err_list)
        

    
    #draw_delay(delay_list=delay_list)
    #draw_acc(acc_list=acc_list)
    '''
    
    # 目标数量持续增加
    for i in range(0,10):
        work_condition_list.append(copy.deepcopy(work_condition))
        # 模拟随机波动的情况
        work_condition['obj_n']+=random.randrange(0, 3)


    # 目标数量保持不变
    for i in range(0,50):
        work_condition_list.append(copy.deepcopy(work_condition))

    
    record = simulate_feedback_scheduler_delay_first(
        work_condition_list=work_condition_list,
        fps_range=fps_range,
        reso_range=reso_range,
        delay_cons=delay_cons,
        step_coeff=step_coeff,
        cons_coeff=cons_coeff
    )
    


    draw_all(
        delay_list=record['delay_list'],
        delay_cons=delay_cons,
        acc_list=record['acc_list'],
        acc_cons=acc_cons,
        colors=colors,
        fps_list=record['fps_list'],
        reso_list=record['reso_list'],
        reso_range=reso_range
    )











