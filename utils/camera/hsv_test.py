import cv2
import numpy as np

from frunet.collision_detect2 import CollisionDetector
from utils.camera.utils_camera import vessel_edge1
from utils.camera.utils_cap import vessel_edge2

collision_detector = CollisionDetector(5.0)

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        h, s, v = hsv_img[y, x]
        b, g, r = bgr_img[y, x]
        print(f"点击位置 (x={x}, y={y})")
        print(f"  BGR: ({b}, {g}, {r})")
        print(f"  HSV: (H={h}, S={s}, V={v})")
        print('碰撞信息： ',collision_detector.analyze(vessel_mask=vessel_edge1, target_contours=[np.array([[x,y]])]))

# 读图
#bgr_img = cv2.imread('./test_image4.png')
bgr_img = cv2.imread('/home/xwj/桌面/project_2026/frunet/data2/images/24.png')
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

# 创建窗口并绑定鼠标回调
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', on_mouse_click)

# 显示图像
cv2.imshow('Image', bgr_img)
print("点击图像任意位置查看 HSV 值，按任意键退出...")
cv2.waitKey(0)
cv2.destroyAllWindows()