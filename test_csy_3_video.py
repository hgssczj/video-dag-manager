
# 该函数的作用是把视频中若干帧以frmae形式提取，转化为多种分辨率然后保存。
import cv2  
import os  
  
# 定义分辨率字典  
resolution_wh = {
    "360p": {
        "w": 480,
        "h": 360
    },
    "480p": {
        "w": 640,
        "h": 480
    },
    "540p": {
        "w": 960,
        "h": 540
    },
    "630p": {
        "w": 1120,
        "h": 630
    },
    "720p": {
        "w": 1280,
        "h": 720
    },
    "810p": {
        "w": 1440,
        "h": 810
    },
    "900p": {
        "w": 1600,
        "h": 900
    },
    "990p": {
        "w": 1760,
        "h": 990
    },
    "1080p": {
        "w": 1920,
        "h": 1080
    }
}
  
# 视频文件路径  
video_path = 'input/cold_start_4.mp4'  
# 帧保存目录  
frame_dir = 'video_frames/cold_start_4'  
  
# 确保保存目录存在  
if not os.path.exists(frame_dir):  
    os.makedirs(frame_dir)  
  
# 打开视频文件  
cap = cv2.VideoCapture(video_path)  
  
# 读取一个帧  
ret, frame = cap.read()  
if not ret:  
    print("Error: Could not read frame from video.")  
    exit()  
  
# 遍历分辨率字典并保存帧  
for name, size in resolution_wh.items():  
    # 缩放帧  
    resized_frame = cv2.resize(frame, (size["w"], size["h"]))  
    # 保存帧为文件  
    filename = os.path.join(frame_dir, f'frame_{name}.jpg')  
    cv2.imwrite(filename, resized_frame)  
  
# 关闭视频文件  
cap.release()  
  
# 读取保存的帧为新的帧对象  
for name, size in resolution_wh.items():  
    filename = os.path.join(frame_dir, f'frame_{name}.jpg')  
    # 读取文件为帧  
    read_frame = cv2.imread(filename)  
    # 在这里可以使用read_frame变量进行后续操作  
    # 例如，显示读取的帧：  
    # cv2.imshow(name, read_frame)  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()  
  
# 注意：上面的imshow和waitKey代码在循环中可能会迅速显示并关闭窗口。  
# 如果你希望查看所有帧，请考虑在循环外部或适当的位置使用它们。