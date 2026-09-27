import cv2
import numpy as np

from frunet.collision_detect import analyze_vessel_collision
from frunet.collision_detect2 import CollisionDetector
from utils.camera.hsv_locate import detect_red2

vessel_edge2_path = '/home/xwj/桌面/project_2026/frunet/runs/predicted_vessel_wall2.png'
vessel_edge2 = cv2.imread(vessel_edge2_path)
class CaptureUtil:
    def __init__(self, cap_id=0):
        self.cap = cv2.VideoCapture(cap_id)
        self.collision_detector = CollisionDetector(2.0)
        self.position = [0, 0]
        self.combined_image = None

    def show_frame_detect(self,vessel=vessel_edge2):
        # 创建一个可调整大小的窗口（关键步骤）
        cv2.namedWindow("Red Detection2", cv2.WINDOW_NORMAL)

        # 可选：设置窗口的具体尺寸（宽 x 高）
        cv2.resizeWindow("Red Detection2", width=800, height=600)
        ret, frame = self.cap.read()
        print('俯视---------------------')
        centers, result_contours, result_img = detect_red2(frame)
        if centers:
            (x, y) = centers[0]
            self.position = [x, y]
        # 将 血管模型分割结果 转为 BGR 以便并排显示
        # mask_bgr = cv2.cvtColor(vessel, cv2.COLOR_GRAY2BGR)
        mask_bgr = vessel.copy()
        #collision_info = analyze_vessel_collision(vessel, result_contours, dist_threshold=5.0)
        collision_info = self.collision_detector.simple_analyze(vessel_mask=vessel, target_contours=result_contours, dist_threshold=3.0)
        if collision_info["is_collided"]:
            print('检测到相碰！')
            (cx, cy) = centers[0]
            cv2.putText(mask_bgr, f"Collide", (cx + 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 在掩码上绘制轮廓
        for cnt in result_contours:
            cv2.drawContours(mask_bgr, [cnt], -1, (0, 255, 0), 2)
        if result_img.shape[:2] != mask_bgr.shape[:2]:
            mask_bgr = cv2.resize(mask_bgr, (result_img.shape[1], result_img.shape[0]))

        # 并排显示原图与掩码
        combined = np.hstack((result_img, mask_bgr))
        cv2.imshow("Red Detection2", combined)
        self.combined_image = combined
        #cv2.waitKey(1)
        # 打印坐标信息
        if centers:
            print("检测到的红色区域中心点坐标 (x, y):")
            for i, (x, y) in enumerate(centers, 1):
                print(f"  区域 {i}: ({x}, {y})")
        else:
            print("未检测到红色区域")
            return collision_info
        return collision_info
