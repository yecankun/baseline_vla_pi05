# inference.py
import threading

import cv2
import torch
import time
from elite import EC
from dataset import trans
from intervention.dual_bc.model import SingleFrameDualPolicyModel
from intervention.utils.data_record_util import record_data, save_time
from utils.camera.utils_camera import Camera
from utils.interface.opengl_interface import Interface
from utils.robot.utils_elirobot import get_path_points
from utils.robot.utils_piper import PiperRobot

B = 1
T = 5
DT = 0.1

stop_event = threading.Event()
#right_branch 12 left_branch 14
PIPER_LEN = 12
piper_step = 0


def _read_image_as_numpy_rgb(img, size=(224, 224)):
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

ec = EC(ip="192.168.137.200", auto_connect=True)
camera0 = Camera(0)
camera0.get_image_depth()
print('camera 侧视 initialized')

import threading
import time
from queue import Queue


def control_main():
    # global piper_step
    time.sleep(1.5)  # 等待摄像头线程初始化
    piper = PiperRobot()
    intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()

    def elirobot_task(new_pose, result_queue):
        """Elirobot 执行任务的线程函数"""
        start_time = time.time()
        try:
            print(f'Elirobot moving to new pose: {new_pose}')
            ec.move_joint(target_joint=ec.get_inverse_kinematic(pose=new_pose), speed=100)
            elirobot_time = time.time() - start_time
            result_queue.put(('elirobot', elirobot_time))
        except Exception as e:
            result_queue.put(('elirobot', 0))  # 发生错误时返回0
            print(f"Elirobot error: {e}")

    def piper_task(result_queue):
        """Piper 执行任务的线程函数"""
        start_time = time.time()
        global piper_step
        try:
            print(f'Piper running iteration {piper_step + 1}/{PIPER_LEN}')
            piper.step_forward(pause_time=0)
            piper_step += 1
            piper_time = time.time() - start_time
            result_queue.put(('piper', piper_time))
        except Exception as e:
            result_queue.put(('piper', 0))  # 发生错误时返回0
            print(f"Piper error: {e}")

    while True:
        pre_time = time.time()
        pose = ec.current_pose
        intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
        image = _read_image_as_numpy_rgb(color_image)
        # disp = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Camera View", disp)
        # cv2.waitKey(1)
        image_time = time.time() - pre_time
        print(f'waste image time: {image_time}')  # 0.15s

        pre_time = time.time()
        large_action, small_action = predict(image, pose, piper_step)
        model_time = time.time() - pre_time
        print(f'waste model time: {model_time}')  # 0.05s

        pre_time = time.time()
        cur_pose = ec.current_pose
        new_pose = [
            cur_pose[0] + large_action[0],
            cur_pose[1] + large_action[1],
            cur_pose[2] + large_action[2],
            cur_pose[3],
            cur_pose[4],
            cur_pose[5]
        ]

        # 创建队列用于获取线程执行结果
        result_queue = Queue()

        # 启动并行任务
        elirobot_thread = None
        piper_thread = None
        elirobot_start_time = time.time()
        piper_start_time = time.time()

        # 启动 Elirobot 线程
        elirobot_thread = threading.Thread(target=elirobot_task, args=(new_pose, result_queue))
        elirobot_thread.start()

        # 如果需要执行 piper 任务，则启动 piper 线程
        if small_action == 1:
            piper_thread = threading.Thread(target=piper_task, args=(result_queue,))
            piper_thread.start()

        # 等待线程完成并收集结果
        elirobot_time = 0
        piper_time = 0

        if elirobot_thread:
            elirobot_thread.join()

        if piper_thread:
            piper_thread.join()

        # 从队列中获取实际耗时
        results_received = 0
        while results_received < (2 if small_action == 1 else 1):
            try:
                task_type, exec_time = result_queue.get(timeout=1)
                if task_type == 'elirobot':
                    elirobot_time = exec_time
                elif task_type == 'piper':
                    piper_time = exec_time
                results_received += 1
            except:
                # 超时继续等待
                continue

        elirobot_time_total = time.time() - pre_time  # 总耗时（包括线程启动等开销）

        pre_time = time.time()
        real_coord, _ = camera0.predict_pos()
        coord_time = time.time() - pre_time
        pre_time = time.time()
        #interface.update_with_transform(real_coord)
        show_time = time.time() - pre_time

        if RECORD:
            save_time(CONTROL_MODE, BRANCH, image_time, model_time, elirobot_time, piper_time, coord_time, show_time)
            # 感知模块获取图像的时间、推理时间、大机械臂执行时间、小机械臂执行时间、导丝头识别时间、渲染时间

        if piper_step >= PIPER_LEN:
            piper.piper.GripperCtrl(20 * 1000, 1000, 0x01, 0)
            stop_event.set()
            break

    print('Control thread stopped.')



# left: "dual_policy_epoch_100.pth" right: "right_model_state/dual_policy_epoch_100.pth"
#ckpt_path = "right_model_state/dual_policy_epoch_100.pth"
ckpt_path="dual_policy_epoch_100.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SingleFrameDualPolicyModel(
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
        large_action ,small_action = model.get_action(
            image, pose, count, task_id, deterministic=True, prob_threshold = 0.45
        )
    large_action = large_action.cpu().numpy()[0]
    print("Large arm action (6D):", large_action)
    small_action = small_action.item()
    print("Small arm action (0/1):", small_action)

    return large_action, small_action

RECORD = False
CONTROL_MODE = "resnet_bc"
BRANCH = "left"
dt = 0.1
def experiment_record():
    time.sleep(1.5)  # 等待摄像头线程初始化
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

interface = Interface()
def interface_main():
    interface.opengl_main()

if __name__ == "__main__":

    control_thread = threading.Thread(target=control_main)
    # experiment_thread = threading.Thread(target=experiment_record)
    # interface_thread = threading.Thread(target=interface_main)

    control_thread.start()
    # interface_thread.start()

    control_thread.join()
    # interface_thread.join()

    print("Inference completed.")



    #data_right 在第800次结束就可以了