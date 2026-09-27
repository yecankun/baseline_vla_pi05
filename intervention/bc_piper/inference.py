# inference.py
import threading

import cv2
import torch
import time
from elite import EC
from dataset import trans
from intervention.bc_piper.model import SingleFramePiperPolicyModel
from intervention.utils.data_record_util import record_data
from utils.camera.utils_camera import Camera
from utils.robot.utils_elirobot import get_path_points
from utils.robot.utils_piper import PiperRobot

B = 1
T = 5
DT = 0.1

stop_event = threading.Event()
#right_branch 12 left_branch 14
PIPER_LEN = 14
piper_step = 0


def _read_image_as_numpy_rgb(img, size=(224, 224)):
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

ec = EC(ip="192.168.137.200", auto_connect=True)

camera0 = Camera(1)
camera0.get_image_depth()
print('camera 侧视 initialized')
def control_main():
    global piper_step
    time.sleep(1.5)  # 等待摄像头线程初始化
    piper = PiperRobot()
    intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
    pre_time = time.time()
    while True:
        cur_time = time.time()
        pose = ec.current_pose
        intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
        image = _read_image_as_numpy_rgb(color_image)
        disp = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera View", disp)
        cv2.waitKey(1)

        print(f'waste time: {time.time() - pre_time}')  # 0.15s
        pre_time = time.time()
        small_action = predict(image, pose, piper_step)
        print(f'Inference time: {time.time() - cur_time}s')

        if small_action == 1:
            print(f'Piper running iteration {piper_step + 1}/{PIPER_LEN}')
            piper.step_forward(pause_time=0.8)
            piper_step += 1
        if piper_step >= PIPER_LEN:
            piper.piper.GripperCtrl(20 * 1000, 1000, 0x01, 0)
            stop_event.set()
            break
        print(f'time in control loop: {time.time() - cur_time}s')

    print('Control thread stopped.')
# left: "piper_policy_epoch_20.pth" right: "right_model_state/piper_policy_epoch_20.pth"
ckpt_path="piper_policy_epoch_20.pth"
#ckpt_path = "right_model_state/piper_policy_epoch_20.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SingleFramePiperPolicyModel(
        d_model=128,
        nhead=8,
        num_transformer_layers=1,
        num_tasks=2,
        dropout=0.1
    ).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
def predict(image, pose, count):

    image = trans(image).unsqueeze(0)  # [1, C, H, W]
    pose =torch.tensor(pose, dtype=torch.float).unsqueeze(0)
    count = torch.tensor(count, dtype=torch.float).unsqueeze(0)  # [B, T]
    task_id = torch.tensor([0]).to(device)  # 任务0：左支

    with torch.no_grad():
        small_action = model.get_action(
            image, pose, count, task_id, deterministic=True
        )

    small_action = small_action.item()
    print("Small arm action (0/1):", small_action)
    return small_action

RUN_INDEX = 1
def elirobot_run():

    # 机械臂运行的点位
    RUN_PATH = f"/home/xwj/桌面/project_2026/path/path{RUN_INDEX}_pose.txt"
    elirobot_index = 0
    time.sleep(1.5)  #等待摄像头线程初始化
    # 机械臂运行的点位
    points = get_path_points(run_path=RUN_PATH)
    while elirobot_index < len(points):
        if stop_event.is_set():
            break
        target_pose = points[elirobot_index]
        target = ec.get_inverse_kinematic(pose=target_pose)
        ec.move_joint(target_joint=target, speed=20)
        print(f'Elirobot moved to point {elirobot_index + 1}/{len(points)}')
        elirobot_index += 1
        time.sleep(1)
    #stop_event.set()

RECORD = False
CONTROL_MODE = "bc_piper"
BRANCH = "left"
dt = 0.1
def experiment_record():
    time.sleep(1.5)
    intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
    pre_time = time.time()
    image_index = 1
    while not stop_event.is_set():
        if RECORD:
            record_data(camera0, ec, image_index, CONTROL_MODE, BRANCH)
            image_index += 1
        camera0.predict_pos()
        time.sleep(dt)
        print(f"experiment_record loop time: {time.time() - pre_time}s")
        pre_time = time.time()

if __name__ == "__main__":

    control_thread = threading.Thread(target=control_main)
    elirobot_thread = threading.Thread(target=elirobot_run)
    experiment_thread = threading.Thread(target=experiment_record)

    elirobot_thread.start()
    control_thread.start()
    experiment_thread.start()

    elirobot_thread.join()
    control_thread.join()
    experiment_thread.join()

    print("Inference completed.")

    #效果很差，需要增加数据量（不理想状态）如大机械臂远远领先导丝头，也需要触发小机械臂动作。并且考虑去掉lstm