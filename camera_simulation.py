import cv2
import subprocess
import flask
# import flask_cors
import multiprocessing
import argparse

from logging_utils import root_logger
import threading

video_source_app = flask.Flask(__name__)

src = None

# TODO 多视频流同时进入时，需要考虑如何建立多视频流输入机制
# 本地视频流
#改造为从云端获取视频流
video_info_list = [
    {"id": 0, "type": "student in classroom", "path": "input/input.mov", "url": "http://114.212.81.11:7912/video0"},
    {"id": 1, "type": "people in meeting-room", "path": "input/input1.mp4",  "url": "http://114.212.81.11:7912/video1"},
    {"id": 3, "type": "traffic flow outdoor", "path": "input/traffic-720p.mp4", "url": "http://114.212.81.11:7912/video3"},
    {"id": 4, "type": "cold_start_face_detect", "path": "input/cold_start_4.mp4", "url": "http://114.212.81.11:7912/video4"},
    {"id": 5, "type": "cut people in meeting-room", "path": "input/test-cut1.mp4", "url": "http://114.212.81.11:7912/video99"},
    {"id": 100, "type": "cut people in meeting-room", "path": "input/test-cut1.mp4", "url": "http://114.212.81.11:7912/video99"},
    {"id": 101, "type": "traffic flow outdoor", "path": "input/traffic-720p.mp4", "url": "http://114.212.81.11:7912/video3"},
    {"id": 102,"type": "people in meeting-room", "path": "input/input1.mp4", "url": "http://114.212.81.11:7912/video1"},
    {"id": 103, "type": "cold_start_face_detect", "path": "input/cold_start_4.mp4", "url": "http://114.212.81.11:7912/video4"}
    # id为100是为了验证知识库的正确性，避免因为99导致调度器不工作
]


def get_video_frame(video_src):
    assert video_src
    video_cap = cv2.VideoCapture(video_src)
    while True:
        ret, frame = video_cap.read()
        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type:image/jpeg\r\n'
                   b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                                                                    b'\r\n' + frame + b'\r\n')
        else:
            video_cap = cv2.VideoCapture(video_src)


@video_source_app.route('/video0')
# @flask_cors.cross_origin()
def read_video0():
    return flask.Response(get_video_frame("input/input.mov"),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

@video_source_app.route('/video1')
# @flask_cors.cross_origin()
def read_video1():
    return flask.Response(get_video_frame("input/input1.mp4"),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

@video_source_app.route('/video3')
# @flask_cors.cross_origin()
def read_video3():
    return flask.Response(get_video_frame("input/traffic-720p.mp4"),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

@video_source_app.route('/video4')
# @flask_cors.cross_origin()
def read_video4():
    return flask.Response(get_video_frame("input/cold_start_4.mp4"),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

@video_source_app.route('/video99')
# @flask_cors.cross_origin()
def read_video99():
    return flask.Response(get_video_frame("input/test-cut1.mp4"),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


def start_video_stream(port):
    video_source_app.run(host="0.0.0.0", port=port)


if __name__ == '__main__':

    start_video_stream(7912)

'''
import cv2
import subprocess
import flask
# import flask_cors
import multiprocessing
import argparse

from logging_utils import root_logger
import threading

video_source_app = flask.Flask(__name__)

src = None

# TODO 多视频流同时进入时，需要考虑如何建立多视频流输入机制
# 本地视频流
video_info_list = [
    {"id": 0, "type": "student in classroom", "path": "input/input.mov", "port":5910, "url": "http://127.0.0.1:5910/video"},
    {"id": 1, "type": "people in meeting-room", "path": "input/input1.mp4", "port":5911, "url": "http://127.0.0.1:5911/video"},
    {"id": 3, "type": "traffic flow outdoor", "path": "input/traffic-720p.mp4", "port":5913, "url": "http://127.0.0.1:5913/video"}
]


def get_video_frame():
    assert src
    video_cap = cv2.VideoCapture(src)
    while True:
        ret, frame = video_cap.read()
        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type:image/jpeg\r\n'
                   b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                                                                    b'\r\n' + frame + b'\r\n')
        else:
            video_cap = cv2.VideoCapture(src)


@video_source_app.route('/video')
# @flask_cors.cross_origin()
def read_video():
    return flask.Response(get_video_frame(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')



def start_video_stream(video_src, port):
    global src
    src = video_src
    video_source_app.run(host="0.0.0.0", port=port)


if __name__ == '__main__':

    for video_info in video_info_list:
        multiprocessing.Process(target=start_video_stream, args=(video_info['path'],video_info['port'])).start()
        

'''

