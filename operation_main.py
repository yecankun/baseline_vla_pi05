# 检查操作员对导丝的操作，并进行相应的机器人控制
from systemd.slave_coil_driver import SlaveRobot
from utils.camera.camera_index_util import find_intel_realsense_index
from utils.camera.utils_camera import Camera
from utils.camera.utils_cap import CaptureUtil
from utils.robot.utils_piper import PiperRobot
import cv2
import time
from utils.camera.hsv_locate import detect_red, detect_operation_red
from utils.utils_controller import SlaveController, EasyContoller

LOCAL_IP = "192.168.137.1"
LOCAL_PORT = 12201
ROBOT_IP = "192.168.137.3"
ROBOT_PORT = 12301
operation_y_threshold = 5  # 定义操作阈值
def operation_main():
    camera0 = Camera(0)
    camera0.get_image_depth()
    # cap1_index = find_intel_realsense_index()
    # cap1 = CaptureUtil(cap1_index)
    # piper = PiperRobot()
    #slave = SlaveController(LOCAL_IP, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    slave = EasyContoller(LOCAL_IP, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    # ret, frame = cap1.cap.read()
    time.sleep(1)
    print('operation camera initialized')
    # ret, color_image = cap1.cap.read()
    intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
    centers, result_contours, result_img = detect_red(color_image)
    cx, cy = centers[0] if len(centers) > 0 else (0, 0)
    while True:
        pre_time = time.time()
        intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
        image_time = time.time() - pre_time
        pre_time = time.time()
        centers, result_contours, result_img = detect_operation_red(color_image)

        detect_time = time.time() - pre_time
        # 显示检测结果
        if result_img is None:
            print('未获取到操作员图像，继续下一帧')
            continue

        # cv2.imshow("Operation Detection", result_img)
        # cv2.imwrite('/temp/实验1/operation_left2.png', color_image)
        #cv2.imwrite('/home/xwj/桌面/project_2026/temp/operation_up2.png', result_img)
        # cv2.waitKey(1)
        nx, ny = centers[0] if len(centers) > 0 else (0, 0)
        print('操作员当前坐标:', (nx, ny))
        # 判断操作员的动作
        # if (cx, cy) != (0, 0) and (nx, ny) != (0, 0) and ny - cy > operation_y_threshold:
        #     print('操作员向上移动导丝')
        #     piper.step_forward()
        # elif (cx, cy) != (0, 0) and (nx, ny) != (0, 0) and cy - ny > operation_y_threshold:
        #     print('操作员向下移动导丝')
        #     piper.step_backward()
        # elif (cx, cy) != (0, 0) and (nx, ny) != (0, 0) and nx - cx > operation_y_threshold:
        #     print('操作员向左偏转导丝')
        #     #slave.step(-1,1,0)
        # elif (cx, cy) != (0, 0) and (nx, ny) != (0, 0) and cx - nx > operation_y_threshold:
        #     print('操作员向右偏转导丝')
        #     #slave.step(1,-1,0)
        # cx, cy = nx, ny
        slave.step(1, -1, 0)
        save_time("operation_time",image_time,detect_time)
        # 获取图像时间、操作识别时间、转向命令发出时间戳

def slave_coil_test():
    slave_controller = SlaveController(LOCAL_IP, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)

    slave_controller.step(1, 0, 0, 0)

def piper_demo(percent):
    #percent = [0.0,0.0,0.0]
    piper = PiperRobot()
    slave = EasyContoller(LOCAL_IP, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    L = 5
    for _i in range(L):
        time.sleep(0.2)
        piper.step_forward()
        slave.step(percent[0], percent[1], percent[2])

def save_time(file_name, *args):
    root_path = "/home/xwj/桌面/project_2026/temp/data1"
    time_path = f"{root_path}/{file_name}.txt"
    args_string = ' '.join(map(str, args))
    write_in(time_path, args_string)

def write_in(file_name: str, data):
    with open(file_name, encoding="utf-8", mode="a") as file:
        file.write(str(data) + '\n')


import requests
import json
from datetime import datetime
def get_network_time_api():
    """
    通过网络 API 获取统一时间
    """
    try:
        # 使用 WorldTimeAPI
        response = requests.get('http://worldtimeapi.org/api/timezone/UTC')
        data = response.json()

        # 提取时间戳
        datetime_str = data['datetime'][:19]  # 截取到秒
        dt = datetime.fromisoformat(datetime_str.replace('T', ' '))

        # 或者直接获取时间戳
        timestamp = data['unixtime']

        return dt, timestamp
    except Exception as e:
        print(f"Network time API request failed: {e}")
        return datetime.now(), datetime.now().timestamp()


if __name__ == "__main__":
    operation_main()