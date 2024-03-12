import pandas as pd
import numpy as np


def if_conf_same(pre_row, temp_row, conf_column_index):
    '''
    判断pre_row使用的配置与temp_row是否相同
    '''
    for index in conf_column_index:
        if pre_row[index] != temp_row[index]:
            return False
    return True


def aggregate_same_conf_array(data_array, agg_column_index, delete_first_row):
    '''
    将data_array(二维数组)按照agg_column_index进行聚合(求均值), 其余的列取任意一行的值即可
    '''
    # print("before aggregate, data_array.shape is :{}".format(data_array.shape))
    if delete_first_row:
        data_array = np.delete(data_array, [0], axis=0)  # 删除每一种配置的第一行数据, 目的是消除服务第一次执行时间过长的影响
    # print("after aggregate, data_array.shape is :{}".format(data_array.shape))
    aggregate_res = []
    for i in range(data_array.shape[1]):
        if i not in agg_column_index:  # 不需要聚合的列数组中每一行的值都是相同的，取第0行该列的值即可
            aggregate_res.append(data_array[0, i])
        else:  # 需要聚合的列则所有行求均值
            aggregate_res.append(data_array[:, i].mean())
    return np.array(aggregate_res)
        
    
def aggregate_knowledgebase_data(file_name, the_first_column_names, conf_names, serv_names, serv_field_names, serv_conf_index, serv_aggregate_index):
    '''
    将knowledgebase_builder.py采样得到的csv文件按配置进行聚合,服务质量为该配置多个采样点的均值
    '''
    ################## 1.获取各个配置项在csv文件中的列索引 ##################
    conf_column_index = []  # 配置列索引，用于判断两条数据是否有相同的配置
    the_first_column_len = len(the_first_column_names)
    conf_len = len(conf_names)
    serv_len = len(serv_names)
    serv_field_len = len(serv_field_names)
    
    for i in range(the_first_column_len, the_first_column_len + conf_len):
        conf_column_index.append(i)
    
    for i in range(serv_len):
        temp_start_index = the_first_column_len + conf_len + i * serv_field_len
        for j in serv_conf_index:
            conf_column_index.append(temp_start_index + j)
    
    ################## 2.获取所有聚合项在csv文件中的列索引 ##################
    agg_column_index = []  # 需要进行聚合的列索引
    agg_column_index.append(2)
    agg_column_index.append(3)
    
    for i in range(serv_len):
        temp_start_index = the_first_column_len + conf_len + i * serv_field_len
        for j in serv_aggregate_index:
            agg_column_index.append(temp_start_index + j)
    
    ################## 3.读取csv文件并进行聚合 ##################
    data = pd.read_csv(file_name, sep=',', header='infer')
    column_name_list = data.columns.tolist()
    data_array = data.values[0::, 0::]
    
    pre_row = data_array[0, :]
    result_array = None
    temp_array_chunk = pre_row
    for i in range(1, data_array.shape[0]):
        temp_row = data_array[i, :]
        conf_same_flag = if_conf_same(pre_row, temp_row, conf_column_index)
        if conf_same_flag: 
            # 若当前行的配置与前一行相同，则将当前行存入temp_array_chunk
            temp_array_chunk = np.vstack((temp_array_chunk, temp_row))
        else:
            # 若当前行与前一行的配置不同，则首先将temp_array_chunk进行聚合
            temp_aggregate_row = aggregate_same_conf_array(temp_array_chunk, agg_column_index, True)
            
            # 将temp_aggregate_row作为最后一行添加到result_array
            if result_array is None:
                result_array = temp_aggregate_row
            else:
                result_array = np.vstack((result_array, temp_aggregate_row))
            
            # 将temp_array_chunk置为temp_row
            temp_array_chunk = temp_row
        
        pre_row = temp_row
    
    final_aggregate_row = aggregate_same_conf_array(temp_array_chunk, agg_column_index, True)  # 将最后一种配置的样本聚合并存入result_array
    result_array = np.vstack((result_array, final_aggregate_row))
    
    ################## 4.将聚合之后的result_array写入新的csv文件 ##################
    new_df_dict = {}
    for i, column_name in enumerate(column_name_list):
        new_df_dict[column_name] = result_array[:, i]
    new_df = pd.DataFrame(new_df_dict)
    new_file_name = file_name.split('.csv')[0] + "_aggregated.csv"
    new_df.to_csv(new_file_name, index=False, sep=',')
    

def get_str_key(data_row, key_index_list):
    '''
    将data_row中所有key_index_list列的元素以字符串形式拼接
    '''
    res = ""
    for index in key_index_list:
        res += (str(data_row[index]) + "_")
    return res
    
            
def aggregate_data_group_by_service(file_name, the_first_column_names, conf_names, serv_names, serv_field_names, serv_conf_index, serv_aggregate_index):
    '''
    将函数aggregate_knowledgebase_data得到的csv文件进一步按照服务进行拆分
    '''
    ################## 1.读取csv文件并进行聚合 ##################
    data = pd.read_csv(file_name, sep=',', header='infer')
    column_name_list = data.columns.tolist()
    data_array = data.values[0::, 0::]
    # print(len(column_name_list), data_array.shape)
    
    column_name_list = column_name_list[len(the_first_column_names):]  # 删除前len(the_first_column_names)列，这些列与知识库无关
    data_array = data_array[:, len(the_first_column_names):]
    # print(len(column_name_list), data_array.shape)
    
    conf_names_len = len(conf_names)
    serv_field_len = len(serv_field_names)
    for i, serv_name in enumerate(serv_names):  # 逐个服务进行拆分
        conf_data = data_array[:, 0:conf_names_len]   # 服务配置部分的数据
        serv_data = data_array[:, (conf_names_len+i*serv_field_len):(conf_names_len+(i+1)*serv_field_len)]  # 服务资源部分的数据
        all_serv_data = np.hstack((conf_data, serv_data))  # 服务完整的数据
        
        serv_column_name_list = column_name_list[0 : conf_names_len] + column_name_list[(conf_names_len+i*serv_field_len) : (conf_names_len+(i+1)*serv_field_len)]  # 该服务对应的列名
        # print(serv_name, all_serv_data.shape, len(serv_column_name_list), serv_column_name_list)
        
        # 获取服务配置字段的列索引
        serv_conf_index_list = [j for j in range(conf_names_len)]
        serv_conf_index_list += [(j + conf_names_len) for j in serv_conf_index]
        # for j in serv_conf_index_list:
        #     print(serv_column_name_list[j])
            
        # 获取服务需要聚合字段的列索引
        serv_agg_index_list = [(j + conf_names_len) for j in serv_aggregate_index]
        # for j in serv_agg_index_list:
        #     print(serv_column_name_list[j])
        
        # 遍历服务数据的每一行，对相同配置的行进行聚合
        serv_data_dict = {}  # key为服务的每一种配置的值，value为该配置下重复的数据行(以数组形式组织)
        for j in range(all_serv_data.shape[0]):
            temp_row = all_serv_data[j, :]
            temp_row_key = get_str_key(temp_row, serv_conf_index_list)
            # print(temp_row_key)
            if temp_row_key in serv_data_dict:
                serv_data_dict[temp_row_key] = np.vstack((serv_data_dict[temp_row_key], temp_row))
            else:
                serv_data_dict[temp_row_key] = temp_row
        
        serv_agg_res = None  # 服务最终聚合之后的数组
        for conf_key, same_conf_data in serv_data_dict.items():
            # print(conf_key, same_conf_data.shape)
            temp_aggregate_row = aggregate_same_conf_array(same_conf_data, serv_agg_index_list, False)
            if serv_agg_res is None:
                serv_agg_res = temp_aggregate_row
            else:
                serv_agg_res = np.vstack((serv_agg_res, temp_aggregate_row))
        
        
        ################## 4.将聚合之后的result_array写入新的csv文件 ##################
        new_df_dict = {}
        for i, column_name in enumerate(serv_column_name_list):
            new_df_dict[column_name] = serv_agg_res[:, i]
        new_df = pd.DataFrame(new_df_dict)
        new_file_name = file_name.split('.csv')[0] + "_" + serv_name + ".csv"
        new_df.to_csv(new_file_name, index=False, sep=',')
        
        

if __name__ == "__main__":
    file_name_1 = "20231229_20_58_49_knowledgebase_builder_0.9_0.7_edge_cap_test_facedetection_edge.csv"
    the_first_column_names = ["n_loop", "frame_id", "all_delay", "edge_mem_ratio"]  # csv文件中与服务配置无关的列
    conf_names = ["reso","fps","encoder"]  # 服务可配置旋钮名
    serv_names = ["face_detection","face_alignment"]  # 服务名列表
    serv_field_names = ["role", "ip", "proc_delay", "trans_ip", "trans_delay", 
                        "cpu_portrait", "cpu_util_limit", "cpu_util_use", "trans_cpu_portrait", "trans_cpu_util_limit", "trans_cpu_util_use",
                        "mem_portrait", "mem_util_limit", "mem_util_use", "trans_mem_portrait", "trans_mem_util_limit", "trans_mem_util_use"]  # 每一个服务采集的数据字段
    serv_conf_index = [1, 6, 12]  # 服务采集的字段中为配置的一部分的字段索引
    serv_aggregate_index = [2, 4, 7, 10, 13, 16]  # 服务采集字段中需要进行聚合的字段索引
    
    aggregate_flag_1 = True
    if aggregate_flag_1:
        aggregate_knowledgebase_data(file_name=file_name_1, the_first_column_names=the_first_column_names, conf_names=conf_names, serv_names=serv_names, 
                                     serv_field_names=serv_field_names, serv_conf_index=serv_conf_index, serv_aggregate_index=serv_aggregate_index)

    file_name_2 = "20231226_10_59_00_knowledgebase_builder_0.9_0.7_headup-detect_video99_resource_limit_resource_rotate_aggregated.csv"
    aggregate_flag_2 = False
    if aggregate_flag_2:
        aggregate_data_group_by_service(file_name=file_name_2, the_first_column_names=the_first_column_names, conf_names=conf_names, serv_names=serv_names, 
                                        serv_field_names=serv_field_names, serv_conf_index=serv_conf_index, serv_aggregate_index=serv_aggregate_index)
    