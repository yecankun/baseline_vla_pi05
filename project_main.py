import threading
import time
import logging
import cv2
import os
from elite import EC

from frunet.collision_detect import analyze_vessel_collision
from frunet.collision_detect2 import CollisionDetector
from utils.camera.camera_index_util import find_hd_pro_c920_index, find_intel_realsense_index
from utils.camera.hsv_locate import detect_red
from utils.camera.utils_camera import Camera, vessel_edge1
from utils.camera.utils_cap import CaptureUtil
from utils.interface.opengl_interface import Interface
from utils.robot.utils_piper import PiperRobot
from utils.utils_controller import EasyContoller
from frunet.infer import infer_image
LOCAL_IP = "192.168.137.201"
LOCAL_PORT = 12208
ROBOT_IP = "192.168.137.3"
ROBOT_PORT = 12301

# 设置是否记录数据
RECORD = False
ROOT_PATH = '/home/xwj/桌面/project_2026/data/branch_right2/pm/'
CAMERA1_PATH = ROOT_PATH + 'camera1.txt'
CAMERA2_PATH = ROOT_PATH + 'camera2.txt'
FORCE_PATH = ROOT_PATH + 'force.txt'
CAMERA1_IMG_PATH = ROOT_PATH + 'camera1_img/'
CAMERA2_IMG_PATH = ROOT_PATH + 'camera2_img/'

# 碰撞状态变量：1 上碰撞，2 下碰撞，3 左碰撞，4 右碰撞， 0 无碰撞
collision = 0

interface = Interface()
stop_event = threading.Event()
def interface_main():
    interface.opengl_main()

def force_feedback():
    global collision
    easy = EasyContoller(LOCAL_IP, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    time.sleep(1)
    while True:
        time.sleep(0.1)
        if collision == 1:
            easy.step(1, 1, 1)
            print('上碰撞，磁力反馈向下')
        elif collision == 2:
            easy.step(-1, -1, -1)
            print('下碰撞，磁力反馈向上')
        elif collision == 3:
            easy.step(0, 1, -1)
            print('左碰撞，磁力反馈向右')
        elif collision == 4:
            easy.step(0, -1, 1)
            print('右碰撞，磁力反馈向左')

def elirobot_main():
    global collision
    camera0 = Camera(0)
    ec = EC(ip="192.168.137.200", auto_connect=True)
    pre = [0,0,0]
    e = 1
    while True:
        time.sleep(0.5)
        pos,result_contours= camera0.predict_pos()
        print(pos)
        if pre[0] != 0 and pos[0] - pre[0] > e:
            current = ec.current_pose
            current[1]+=5
            target = ec.get_inverse_kinematic(pose=current)
            ec.move_joint(target_joint=target, speed=20)
            collision = 1
        elif pre[0] != 0 and pre[0] - pos[0] > e:
            current = ec.current_pose
            current[1] -= 5
            target = ec.get_inverse_kinematic(pose=current)
            ec.move_joint(target_joint=target, speed=20)
            collision = -1
        pre = pos

def camera_main():
    global collision
    log_info=['无', '上碰撞', '下碰撞', '左碰撞', '右碰撞']
    camera0 = Camera(0)
    camera0.get_image_depth()
    # cap_index = find_hd_pro_c920_index()
    # cap = CaptureUtil(cap_index)
    # cap.show_frame_detect()
    print('camera 侧视 initialized')
    time.sleep(1)
    intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
    #vessel=infer_image(color_image,'temp/vessel_edge1.png')
    vessel = vessel_edge1
    print('camera 俯视 initialized')
    collision_detector = CollisionDetector()
    #pos[0] = z, pos[1] = -x, pos[2] = -y
    def transform_head(real_coord):
        sphere_position = interface.sphere_position
        if real_coord != [0, 0, 0]:
            sphere_position[0] = real_coord[2]*10-4
            sphere_position[1] = -real_coord[0]*10+0.73+0.9
            sphere_position[2] = -real_coord[1]*10+0.03
        return sphere_position
    image_index = 1
    pre_time = time.time()
    while True:
        # print('time: ', time.time() - pre_time) # 0.07s
        # pre_time = time.time()
        print('---------------------')
        # time.sleep(0.02)
        # 相机1（侧视）
        pre_time = time.time()
        pos, result_contours = camera0.predict_pos()
        # collision_info2 = cap.show_frame_detect()
        image_time = time.time() - pre_time
        # if pos == [0, 0, 0]:
        #     continue

        print('导丝头真实坐标:',pos)
        #collision_info=analyze_vessel_collision(vessel, result_contours, dist_threshold=2.0)
        pre_time = time.time()
        collision_info = collision_detector.simple_analyze(vessel_mask=vessel, target_contours=result_contours, dist_threshold=6.0)
        collision_info2 = collision_detector.simple_analyze(vessel_mask=vessel, target_contours=result_contours,
                                                           dist_threshold=6.0)
        print('碰撞信息1:', collision_info)
        col1 = collision_info["is_collided"]
        col2 = collision_info2["is_collided"]
        collision_time = time.time() - pre_time
        save_time("project_time",image_time,collision_time)
        # 获取图像与定位、碰撞检测
        # interface.change_pos(transform_head(pos))
        # 相机2 (俯视)
        print('碰撞信息2:', collision_info2)
        if col2:
            collision = collision_info2["side_number"] + 2
        if col1:
            collision = collision_info["side_number"]
        if not col1 and not col2:
            collision = 0
        if collision != 0:
            logging.warning('碰撞状态：%s, 碰撞维度：%s', log_info[collision],collision)
        # 记录数据
        # if RECORD:
        #     write_in(CAMERA1_PATH, f"{camera0.position[0]} {camera0.position[1]}")
        #     write_in(CAMERA2_PATH, f"{cap.position[0]} {cap.position[1]}")
        #     write_in(FORCE_PATH, collision)
        #     cv2.imwrite(f"{CAMERA1_IMG_PATH}/{image_index}.png", camera0.combined_image)
        #     cv2.imwrite(f"{CAMERA2_IMG_PATH}/{image_index}.png", cap.combined_image)
        #     image_index += 1
    print('camera thread ending')

def write_in(file_name: str, data):
    with open(file_name, encoding="utf-8", mode="a") as file:
        file.write(str(data) + '\n')

def save_time(file_name, *args):
    root_path = "/home/xwj/桌面/project_2026/temp/data1"
    time_path = f"{root_path}/{file_name}.txt"
    args_string = ' '.join(map(str, args))
    write_in(time_path, args_string)

def experiment_demo():
    piper = PiperRobot()
    time.sleep(0.5)
    L = 6
    for _i in range(L):
        #time.sleep(0.2)
        piper.step_forward(pause_time=0)
    piper.piper.GripperCtrl(20 * 1000, 1000, 0x01, 0)
    stop_event.set()
    print('实验演示结束，停止相机线程')

def electromagnet_main():
    percent_left = [0, 0, -0.8]
    percent_right = [-1,0.2,-0.8]
    slave = EasyContoller(LOCAL_IP, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    while True:
        time.sleep(0.1)
        slave.step(percent_left[0], percent_left[1], percent_left[2])

if __name__ == "__main__":
    RECORD = False
    if RECORD:
        path_list = [CAMERA1_PATH, CAMERA2_PATH, FORCE_PATH]
        for file in path_list:
            if not os.path.exists(file):
                print(f'文件{file}不存在，创建文件')
                with open(file, 'w') as f:
                    pass
        dir_list = [CAMERA1_IMG_PATH, CAMERA2_IMG_PATH]
        for dir_path in dir_list:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    # t1 = threading.Thread(target=force_feedback)
    t2 = threading.Thread(target=camera_main)
    # t3 = threading.Thread(target=experiment_demo)
    # t4 = threading.Thread(target=electromagnet_main)

    # t1.start()
    t2.start()
    # t3.start()
    # t4.start()

    # t1.join()
    t2.join()
    # t3.join()
    # t4.join()



# 俯视相机在下方的usb，侧视相机在上方的usb

# 侧视：368 414  俯视：172 118 侧面x = -(俯视y - 240)