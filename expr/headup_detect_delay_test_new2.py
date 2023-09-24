import requests
import time
import csv
import os
import datetime

if __name__ == "__main__":
    sess = requests.Session()

    # expr_name = 'bigger-tom-cpu'
    # expr_name = 'rack-pure-cloud-cpu'
    # expr_name = 'tx2-pure-edge-gpu'
    # expr_name = 'tx2-gpu-rack-cpu'
    # expr_name = 'rack-pure-cloud-cpu-golden'
    # expr_name = 'tx2-pure-edge-gpu-golden'
    # expr_name = 'rack-pure-cloud-gpu-golden'
    # expr_name = 'tx2-pure-edge-cpu-golden'
    # expr_name = 'desktop-only-kp1-ki02-kd015'
    # expr_name = 'output-scatter'
    expr_name = 'tx2-cloud-raw'

    # 提交请求
    # node_addr = "127.0.0.1:5001"
    # node_addr = "172.27.147.22:5001"
    # node_addr = "172.27.134.58:5001"
    # node_addr = "172.27.146.33:5001"
    node_addr = "172.27.155.56:6001"
    # node_addr = "114.212.81.11:5001"

    query_body = {
        "node_addr": node_addr,
        "video_id": 0,
        "pipeline": ["face_detection", "face_alignment"],
        "user_constraint": {
            "delay": 0.3,
            "accuracy": 0.7
        }
    }


    # query_addr = "192.168.56.102:5000"
    query_addr = "114.212.81.11:6000"
    # query_addr = "172.27.134.58:5000"
    # query_addr = "127.0.0.1:5000"
    # query_addr = "172.27.152.177:5000"
    r = sess.post(url="http://{}/query/submit_query".format(query_addr),
                  json=query_body)

    resp = r.json()
    print(resp)
    query_id = resp["query_id"]

    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
        '_' + os.path.basename(__file__).split('.')[0] + \
        '_' + str(query_body['user_constraint']['delay']) + \
        '_' + str(query_body['user_constraint']['accuracy']) + \
        '_' + expr_name + \
        '.csv'

    with open(filename, 'w', newline='') as fp:
        fieldnames = ['n_loop', 'frame_id', 'total', 'up', 'fps',
                      'resolution', 'delay', 'face_detection','face_detection_ip', 'face_alignment','face_alignment_ip']
        wtr = csv.DictWriter(fp, fieldnames=fieldnames)
        wtr.writeheader()

        written_n_loop = dict()

        # 轮询结果+落盘
        while True:
            r = None
            try:
                time.sleep(1)
                print("post one query request")
                r = sess.get(
                    url="http://{}/query/get_result/{}".format(query_addr, query_id))
                print(f'response:{r.json()}')
                if not r.json():
                    continue
                resp = r.json()

                res_list = resp['appended_result']
                plan = resp['latest_result']['plan']
                # plan_result = resp['latest_result']['plan_result']
                # plan_result = resp['latest_result'].get('plan_result')
                # runtime = resp['latest_result'].get('runtime')

                # if runtime is not None:
                #     delay = runtime.get('delay')
                # else:
                #     delay = 0.0

                fps = plan['video_conf']['fps']
                resolution = plan['video_conf']['resolution']
                # if plan_result is not None:
                #     delay = sum(plan_result['delay'].values())
                # else:
                #     delay = 0.0

                fd_role = plan['flow_mapping']['face_detection']['node_role']
                fd_ip   = plan['flow_mapping']['face_detection']['node_ip']
                fa_role = plan['flow_mapping']['face_alignment']['node_role']
                fa_ip   = plan['flow_mapping']['face_alignment']['node_ip']

                for res in res_list:
                    n_loop, frame_id, total, up, delay = res['n_loop'], res['frame_id'], res[
                        'count_result']['total'], res['count_result']['up'], res['delay']
                    row = {
                        'n_loop': n_loop,
                        'frame_id': frame_id,
                        'total': total,
                        'up': up,
                        'fps': fps,
                        'resolution': resolution,
                        'delay': delay,
                        'face_detection': fd_role,
                        'face_detection_ip': fd_ip,
                        'face_alignment': fa_role,
                        'face_alignment_ip': fa_ip
                    }
                    if n_loop not in written_n_loop:
                        wtr.writerow(row)
                        written_n_loop[n_loop] = 1

                print("written one query response, len written_n_loop={}".format(
                    len(written_n_loop.keys())))

            except Exception as e:
                if r:
                    print("got serv result: {}".format(r.text))
                # print("caught exception: {}".format(e), exc_info=True)
                print("caught exception: {}".format(e))
                break
