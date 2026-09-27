# inference.py
import threading

import cv2
import torch
import time
from elite import EC

from intervention.bc_elirobot.dataset import trans
from intervention.bc_elirobot.model import SingleFrameElirobotPolicyModel
from intervention.utils.data_record_util import record_data
from project_main import RECORD
from utils.camera.utils_camera import Camera
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
        if stop_event.is_set():
            break
        cur_time = time.time()
        pose = ec.current_pose
        intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
        image = _read_image_as_numpy_rgb(color_image)
        # disp = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Camera View", disp)
        # cv2.waitKey(1)

        print(f'waste time: {time.time() - pre_time}')  # 0.15s
        pre_time = time.time()
        large_action = predict(image, pose, piper_step, 0)
        cur_pose = ec.current_pose
        new_pose = [
            cur_pose[0] + large_action[0],
            cur_pose[1] + large_action[1],
            cur_pose[2] + large_action[2],
            cur_pose[3],
            cur_pose[4],
            cur_pose[5]

        ]
        print(f'Inference time: {time.time() - cur_time}s')
        print(f'Elirobot moving to new pose: {new_pose}')
        ec.move_joint(target_joint=ec.get_inverse_kinematic(pose=new_pose), speed=100)
        print(f'time in control loop: {time.time() - cur_time}s')

    print('Control thread stopped.')

ckpt_path = "left_model_state/best_elirobot_policy.pth"
#ckpt_path = "right_model_state/2026_3_7/elirobot_policy_epoch_200.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SingleFrameElirobotPolicyModel(
        d_model=128,
        nhead=8,
        num_transformer_layers=1,
        num_tasks=2,
        dropout=0.1
    ).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
def predict(image, pose, count, task_id):

    image = trans(image).unsqueeze(0)  # [1, C, H, W]
    pose =torch.tensor(pose, dtype=torch.float).unsqueeze(0)
    count = torch.tensor(count, dtype=torch.float).unsqueeze(0)  # [B, T]
    task_id = torch.tensor([task_id]).to(device)  # 任务0：左支

    with torch.no_grad():
        large_action = model.get_action(
            image, pose, count, task_id
        )
    large_action = large_action.cpu().numpy()[0]
    print("Large arm action (6D):", large_action)
    return large_action

piper_current_step = 0
def piper_run():
    global piper_current_step
    piper = PiperRobot()
    time.sleep(1)
    for i in range(PIPER_LEN):
        print(f'Piper running iteration {i+1}/{PIPER_LEN}')
        piper.step_forward(pause_time=7)
        piper_current_step += 1
    piper.piper.GripperCtrl(20 * 1000, 1000, 0x01, 0)
    stop_event.set()

RECORD = True
CONTROL_MODE = "bc_elirobot"
BRANCH = "left"
dt = 0.1
def experiment_record():
    intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
    pre_time = time.time()
    image_index = 1
    while not stop_event.is_set():
        if RECORD:
            record_data(camera0, ec, image_index, CONTROL_MODE, BRANCH)
            image_index += 1
        time.sleep(dt)
        print(f"experiment_record loop time: {time.time() - pre_time}s")
        pre_time = time.time()

if __name__ == "__main__":

    control_thread = threading.Thread(target=control_main)
    piper_thread = threading.Thread(target=piper_run)
    experiment_thread = threading.Thread(target=experiment_record)

    piper_thread.start()
    control_thread.start()
    experiment_thread.start()

    piper_thread.join()
    control_thread.join()
    experiment_thread.join()

    print("Inference completed.")

    #效果很差，需要增加数据量（不理想状态）如大机械臂远远领先导丝头，也需要触发小机械臂动作。并且考虑去掉lstm