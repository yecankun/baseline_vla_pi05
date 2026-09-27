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

RECORD = True
CONTROL_MODE = "dual_bc"
BRANCH = "left"
dt = 0.1

if __name__=="__main__":
    img_dir = ""
    L = 100
    pose = [-336.1812021877478, 251.6580494092774, 326.988396688898, 3.0661665148309702, 0.03647610536354759, 0.06842115274422467]
    for i in range(100):
        img_path = img_dir + str(i) + ".jpg"
        image = cv2.imread(img_path)
        pre_time = time.time()
        l,s = predict(image, pose,0)
        infer_time = time.time() - pre_time
        save_time(CONTROL_MODE,BRANCH,infer_time)
