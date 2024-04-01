# video-dag-manager (no-render-test)


## 1 大致结构

云端运行`query_manger_v2.py`，在5000端口提供服务；也需要运行`app_server_test.py`，在5500端口提供服务。

边端运行`job_manager_v2.py`，在5001端口提供服务；也需要运行`app_client_test,py`，在5500端口提供服务。

此外还需要运行`camera_simulation.py`，在7912端口提供服务。为减轻边缘端负载，将其在云端运行。


总体线程模型如下图所示。用户通过/query/submit_query接口针对视频流提交查询请求（参见`expr/`的测试脚本的POST请求），查询请求被下发到特定的边缘节点后，在边缘节点为任务生成一个Job对象。云端调度器周期性地更新各查询的执行计划（以下称该过程为`调度`）、下发给对应的Job对象，Job对象负责根据调度计划，请求5500的计算服务，汇总结果上报给云：

![queue_manager和job_manager线程模型](./img/queue_manager和job_manager线程模型.png)

Job类：一个视频流查询对应边缘节点的一个Job实例，一个Job实例对应一个线程，用于执行调度计划并定期给云端汇报结果。Job实例类似查询请求（用户在云端提交的，且在云端管理）在边端的执行代理。

JobManager类：负责管理所有视频流查询代理线程的类。同时负责与QueryManager通信。

Job状态主要有三类

- JOB_STATE_UNSCHED：新提交的任务，未有调度计划，不执行
- JOB_STATE_READY：已生成调度计划，可执行，未启动
- JOB_STATE_RUNNING：任务已在线程中启动

## 2 启动方式

版本要求：

- （1）为了开启http的KeepAlive：Werkzeug<=2.0.0，Flask<=2.0.0，Flask-Cors<=2.0.0
- （2）为了运行预加载模型：pytorch==1.13.0，torchvision==0.14.0，对应版本的cuda和cudnn等
- （3）其余依赖可依据报错安装

**注意**：启动`job_manager_V2.py`时，应确保`camera_simulation.py`已经在云端运行。如果一定要在边端运行`camera_simulation.py`，需要修改其中video_info_list所包含的各个链接的ip地址。

云端启动：

```shell
# 在云端将终端拆分为三个，依次运行以下程序。由于运行过程中需要目录下的大量配置文件，应该在video-dag-manager目录下运行`camera_simulation.py`和`query_manger_v2.py`
# 在实验室云服务器上运行代码之前，往往需要执行一次如下命令：
$ unset LD_LIBRARY_PATH
# 运行app_server_test。往往在外层目录运行它。
$ python3.8 SchedulingSystem-main-demo/SchedulingSystem/server/app_server_test.py --server_ip=114.212.81.11 --server_port=5500 --edge_port=5500
# 进入video-dag-manager目录
# 运行视频流提供进程
$ python3.8 camera_simulation.py
# 运行云端查询控制和调度进程
$ python3.8 query_manager_v2.py --query_port=5000 --serv_cloud_addr=114.212.81.11:5500 --video_cloud_port=5100


# 使用如下命令，方便在文件中查询error日志
$ python3 job_manager.py 2>&1 | tee demo.log
```

边缘启动：

```shell
# 在边缘将终端拆分为两个，依次运行以下程序。由于运行过程中需要目录下的大量配置文件，应该在video-dag-manager目录下运行`job_manger_v2.py`
# 在tx2设备上运行代码之前，往往需要执行一次如下命令,防止运行过程中jtop未启动：
$ sudo systemctl restart jtop.service
# 运行app_client_test。往往在外层目录运行它。
$ python3 SchedulingSystem-main-demo/SchedulingSystem/client/app_client_test.py --server_ip=114.212.81.11 --server_port=5500 --edge_ip=0.0.0.0 --edge_port=5500
# 进入video-dag-manager目录
# 运行边缘端任务控制进程
$ python3 job_manager_v2.py --query_addr=114.212.81.11:5000 --tracker_port=5001 --serv_cloud_addr=114.212.81.11:5500 --video_side_port=5101
```

使用知识库：

```shell
# video-dag-manager目录下的expr_data2目录里的knowledgebase_builder提供了建立知识库需要的一系列接口。
# 建议的使用方法是在云和边之外的设备，比如工位主机上运行这个程序。
# 具体来说，先完成云和边上的启动（云端运行`query_manger_v2.py`，在5000端口提供服务；也需要运行`app_server_test.py`，在5500端口提供服务。边端运行`job_manager_v2.py`，在5001端口提供服务；也需要运行`app_client_test,py`，在5500端口提供服务。这里的app_server_test.py和app_client_test,py在运行之后就不需要重新加载服务，因为它已经在主函数里创建了服务。不过目前只能运行人脸检测+姿态估计任务）
# 使用python knowledgebase_builder.py可以朝云端发出query，然后进行采样。详见文件内部说明。
```



## 3 计算服务接口示例

```js
描述：提供D计算服务
接口：POST :5500/service/face_detection
输入
{
    "image": "\x003\x001..."
}
输出
{
    "faces": ["\x003\x001...", ...],  // 检测出来的人脸图像
    "bbox": [[1,2,3,4], [1,2,3,4], ...],
    "prob": []
}

描述：提供C计算服务
接口：POST :5500/service/face_alignment
输入
{
    "faces": ["\x003\x001...", ...],  // 需要做姿态估计的人脸图像
    "bbox": [[1,2,3,4], [1,2,3,4], ...],
    "prob": []
}
输出
{
    "count_result": {  // 可以显示的数值结果
        "up": 6,
        "total": 8
    }
}
```

## 4 QueryManager的RESTFUL接口

```js
描述：边端接入云端，汇报视频流信息
接口：POST :5000/node/join
请求数据：
{

}

描述：获取接入到云端的节点信息
接口：GET :5000/node/get_video_info
返回结果
{
    "192.168.56.102:7000": {
        "0": {
            "type": "traffic flow"
        },
        "1": {
            "type": "people indoor"
        }
    },
    "192.168.56.102:8000": {
        "0": {
            "type": "traffic flow"
        },
        "1": {
            "type": "people indoor"
        }
    }
}

描述：从云端接收用户提交的任务
接口：POST :5000/query/submit_query
请求数据：
{
    "node_addr": "192.168.56.102:7000",
    "video_id": 1,
    "pipeline": ["face_detection", "face_alignment"],
    "user_constraint": {
        "delay": 0.8,
        "accuracy": 0.9
    }
}
返回数据：
{
    "msg": "submitted to (cloud) manager from api: /query/submit_query",
    "query_id": "GLOBAL_ID_1",
    "status": 0
}

描述：返回某个视频查询的当前工况
接口：GET :5000/query/get_work_condition/<query_id>
返回数据：
{
    "obj_n": 5,
    "obj_stable": True,
    "obj_size": 300,
    "delay": 0.2
}


描述：边端同步查询的处理结果
接口：POST :5000/query/sync_result/<query_id>
请求数据：
{

}

描述：从云端获取指定任务的结果
接口：GET :5000/query/get_agg_info/<query_id>
返回结果：
{
    // 该部分是列表，代表最近10帧的处理结果。经过改造，其包含每一个任务的执行和传输时延。
    "appended_result": [
        {
                'count_result': {'total': 24, 'up': 20}, 
                'delay': 0.16154261735769418, 
                'execute_flag': True, 
                'ext_plan': {
                            'flow_mapping': 
                                {   
                                    'face_alignment': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'}, 
                                    'face_detection': {'model_id': 0, 'node_ip': '114.212.81.11', 'node_role': 'cloud'} 
                                }, 
                            'video_conf':   {'encoder': 'JPEG', 'fps': 1, 'reso': '360p'}
                            }, 

                'ext_runtime': {
                                    'delay': 0.16154261735769418, 
                                    'obj_n': 24.0, 
                                    'obj_size': 219.36678242330404, 
                                    'obj_stable': 1, 
                                    'plan_result': 
                                        {
                                            'delay': {'face_alignment': 0.09300840817964993, 'face_detection': 0.06853420917804424}, 
                                            'process_delay': {'face_alignment': 0.08888898446009709, 'face_detection': 0.060828484021700345}
                                        }
                                }, 
                'frame_id': 25.0, 
                'n_loop': 1, 
                'proc_resource_info_list': [{'cpu_util_limit': 1.0, 'cpu_util_use': 0.060020833333333336, 'latency': 2.3111135959625244, 'pid': 505503}]
        },
        ...
    ],

    // 留空
    "latest_result": {}
}
```

## 5 JobManager的RESTFUL接口（一般与用户无关）

```js
描述：指定节点提交任务，该接口在本地为job生成实例，每个job一个线程。主线程轮询任务集，若发现一个新启动的job收到了下发的调度策略，则为该job分配线程并启动。
接口：POST :5001/job/submit_job

{
    "node_addr": "192.168.56.102:7000",
    "video_id": 1,
    "pipeline": ["face_detection", "face_alignment"],
    "user_constraint": {
        "delay": 0.8,
        "accuracy": 0.9
    }
}

描述：云端调度器主动请求，以更新边端的调度计划。
接口：POST: 5001/job/update_plan
请求数据：
{
    "job_uid":
    "video_conf":
    "flow_mapping":
}

描述：云端调度器主动请求，以获取边端对应query的运行时情境。运行时情境为一个调度窗口内任务复杂度（目标数量、资源消耗等）的预估值/统计值
接口：GET：5001/job/get_runtime/<job_uid>
返回结果：
{
    "obj_n": 8.6,
    "obj_size": 645.3215
}
```

## 6 调度器函数（参见`query_manager.py`中cloud_scheduler_loop函数和`scheduler_func/`目录）

云端集中调度，所以需要有通信接口，参见JobManager接口`POST: 5001/job/update_plan`。

调度器应封装为一个函数，决定视频流分析配置、并将DAG Job中的dag.flow的各个任务映射到节点。

### 函数参数

（1）待映射/调度的DAG Job

（2）DAG的输入数据信息（TBD）

（3）资源和服务情况（TBD）

- 各机器CPU、内存、GPU情况
- 各机器服务的请求地址
- 当前节点与其余节点的上行带宽/下行带宽

（4）上一轮调度方案的执行结果（若上一轮调度包含多帧，则取各帧数值结果的平均）

- 一帧图像经过DAG推理的总时延

```js
last_plan_res = {
    "delay": {
        "face_detection": 20,
        "face_alignment": 0.5
    }
}
```

（5）用户约束

- 时延范围
- 精度反馈

```js
user_constraint = {
    "delay": 0.3,  // 用户时延约束，单位秒
    "acc_level": 5,  // 用户给出的精度评级：0~5精确等级递增
}
```

### 函数返回

（1）视频配置

- 分辨率
- 跳帧率/fps
- 编码方式

```js
video_conf = {
    "resolution": "480p",
    "fps": 30,
    "encoder": "JPEG",
}
```

（2）DAG执行策略

- 字典中的key需要与传入的DAG Job中`flow`各字段匹配

```js
flow_mapping = {
    "face_detection": {
        "model_id": 0,  // 大模型、中模型、小模型
        "node_role": "host",  //放到本地执行
        "node_ip": "192.168.56.102",  // 映射的节点
    },
    "face_alignment": {
        "model_id": 0,
        "node_role":  "cloud",  // 放到云端执行
        "node_ip": "114.212.81.11",
    }
}
```

## 7 运行时情境函数（参见`job_manager_v2.py`中worker_loop函数，以及`query_manager_v2.py`中`Query`类的成员runtime_portrait、RuntimePortrait类的成员函数`get_portrait_info`、`predict_resource_threshold`、`help_cold_start`等）

感知流程：

（1）更新运行时情境：边缘端的`Job`实例在执行完一个完整的任务后会将本轮执行过程中的运行时情境信息上传到云（边缘端本地不保存任何情境信息和中间结果，请求云端的`/query/sync_runtime`接口）；云端在拿到一次任务执行的运行时情境信息之后调用runtime_portrait成员的函数对其进行处理，包括整理工况情境以便前端展示，以及后续进行运行时情境画像，为调度器提供参考。

（2）获取运行时情境：若前端想要获取工况情境用于前端展示，则需要通过RESTful接口`/query/get_work_condition/<query_id>`进行访问，具体的返回结果格式见前面所述；若调度器想要获取运行时情境画像参数，则可以通过直接调用`Query`、`RuntimePortrait`类对象的成员函数的方式获取相关信息，具体的函数以及使用方式见下。

##### 1. get_portrait_info():
* 输入参数：无
* 返回值：
```
{
    'cur_latency': 0.5,  // 当前任务的执行时延
    'user_constraint': {  // 当前的用户约束
        "delay": 0.8,
        "accuracy": 0.9
    },
    'if_overtime': True,  // 当前任务是否超时，True--超时；False--未超时
    'available_resource': {  // 当前系统中每个设备上本query可以使用的资源量
        '114.212.81.11': {  // 以ip地址表示各个设备
            'node_role': 'cloud',  // 当前设备是云还是边
            'available_cpu': 0.5,  // 当前设备可用的CPU利用率
            'available_mem': 0.8  // 当前设备可用的内存利用率
        },
        '172.27.132.253': {
            ...
        }
    },
    'resource_portrait': {  // 当前任务中各个服务的资源画像
        'face_detection': {  // 以服务名为key
            'node_ip': '172.27.132.253',  // 当前服务的执行节点ip
            'node_role': 'edge',  // 当前服务的执行节点类型
            'cpu_util_limit': 0.5,  // 当前服务在当前执行节点上的CPU资源限制
            'cpu_util_use': 0.4,  // 当前服务在当前执行节点上的实际CPU使用量
            'mem_util_limit': 0.3,  // 当前服务在当前执行节点上的内存资源限制
            'mem_util_use': 0.1,  // 当前服务在当前执行节点上的实际内存使用量
            'cpu_portrait': 0,  // CPU资源画像分类，0--弱；1--中；2--强
            'cpu_bmi': 0.1, // CPU资源的bmi值, (资源分配量-资源需求量) / 资源需求量
            'cpu_bmi_lower_bound': 0, 
            'cpu_bmi_upper_bound': 1.1, // 若'cpu_bmi'在['cpu_bmi_lower_bound', 'cpu_bmi_upper_bound']的范围内，说明cpu资源分配量处于合理范围内；若不在此范围内，则说明不合理
            'mem_portrait': 0,  // 内存资源画像分类，0--弱；1--中；2--强
            'mem_bmi': 0.1, // 内存资源的bmi值, (资源分配量-资源需求量) / 资源需求量
            'mem_bmi_lower_bound': 0, 
            'mem_bmi_upper_bound': 1.1, // 若'mem_bmi'在['mem_bmi_lower_bound', 'mem_bmi_upper_bound']的范围内，说明内存资源分配量处于合理范围内；若不在此范围内，则说明不合理
            'resource_demand': {  // 当前服务在当前配置、当前工况下，在系统中各类设备上的中资源阈值。注意：由于目前只有一台服务器，且边缘节点都是tx2，所以没有按照ip进行不同设备的资源预估，而是直接对不同类别的设备进行资源预估
                'cpu': {  // CPU资源
                    'cloud': {  // 在服务器上的资源阈值
                        'upper_bound': 0.1,  // 中资源阈值的上界
                        'lower_bound': 0.05  // 中资源阈值的下界
                    },
                    'edge': {  // 在边缘设备上的资源阈值
                        'upper_bound': 0.1,
                        'lower_bound': 0.05
                    }
                },
                'mem': {  // 内存资源
                    'cloud': {
                        'upper_bound': 0.1,
                        'lower_bound': 0.05
                    },
                    'edge': {
                        'upper_bound': 0.1,
                        'lower_bound': 0.05
                    }
                }
            }
        },
        'gender_classification': {
            ...
        }
    }
}
```
##### 2. predict_resource_threshold():
* 输入参数：
    * task_info：字典类型，包含字段如下
```
{
    // 配置相关字段
    'service_name': 'face_detection',  // 服务名
    'fps': 15,  // 帧率，取值范围见common.py中变量fps_list
    'reso': 2,  // 分辨率，整数，表示分辨率的字符串到整数的映射见common.py中变量reso_2_index_dict
    
    // 工况相关字段
    'obj_num': 3  // 目标数量
}
```
* 返回值：
```
{
    'cpu': {  // CPU资源
        'cloud': {  // 在服务器上的资源阈值
            'upper_bound': server_cpu_upper_bound,  // 中资源阈值的上界
            'lower_bound': server_cpu_lower_bound  // 中资源阈值的下界
        },
        'edge': {  // 在边缘设备上的资源阈值
            'upper_bound': edge_cpu_upper_bound,
            'lower_bound': edge_cpu_lower_bound
        }
    },
    'mem': {  // 内存资源
        'cloud': {
            'upper_bound': server_mem_upper_bound,
            'lower_bound': server_mem_lower_bound
        },
        'edge': {
            'upper_bound': edge_mem_upper_bound,
            'lower_bound': edge_mem_lower_bound
        }
    }
}
```

##### 3. help_cold_start():
* 输入参数：
    * service：字符串类型，例如"face_detection"
* 返回值：
```
{
    'cpu': {  // CPU资源
        'cloud': 0.1,  // 云端的最大资源阈值
        'edge': 0.5  // 边端的最大资源阈值
    },
    'mem': {  // 内存资源
        'cloud': 0.1,
        'edge': 0.5
    }
}
```


## 9 知识库建立者knowledgebase_builder

该文件位于expr_data2目录之中。其中实现了一个KnowledgeBaseBuilder类，改类提供建立知识库的方法。
其所建立的知识库以json文件形式存在，往往需要复制粘贴到knowledgebase_bayes或knowledgebase_rotate目录下，被query_manager_v2的调度器使用。