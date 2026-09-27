import time

import cv2
import numpy as np
from scipy.spatial.distance import cdist


def contours_intersect_or_min_distance(cnt1, cnt2, threshold=100):
    """
    判断两个轮廓是否相碰（相交/接触），若不相碰则返回最小距离。
    参数:
        cnt1, cnt2: OpenCV 轮廓，shape = (N, 1, 2) 或 (N, 2)
        threshold: 距离小于该值视为“相碰”
    返回:
        intersect: bool, 是否相碰
        min_dist: float, 最小距离（若相碰则为0）
    """
    # 确保形状为 (N, 2)
    pts1 = cnt1.reshape(-1, 2).astype(np.float32)
    pts2 = cnt2.reshape(-1, 2).astype(np.float32)
    # 方法1：暴力计算所有点对距离（简单可靠）
    dists = cdist(pts1, pts2, metric='euclidean')
    min_dist = np.min(dists)
    if min_dist < threshold:
        return True, 0.0
    else:
        return False, min_dist

IMG1 = cv2.imread('/home/xwj/桌面/project_2026/frunet/runs/predicted_vessel_wall.png', cv2.IMREAD_GRAYSCALE)
IMG2 = cv2.imread('/home/xwj/桌面/project_2026/frunet/runs/predicted_vessel_wall2.png', cv2.IMREAD_GRAYSCALE)

def is_point_in_vessel(point, img, threshold=128):
    """
    判断给定坐标是否在血管区域中

    参数:
        point: tuple (x, y) - 要检测的点坐标
        image_path: str - 血管模型灰度图路径
        threshold: int - 灰度阈值，默认128

    返回:
        bool: True 表示在血管中，False 表示不在
    """
    x, y = point

    if img is None:
        raise FileNotFoundError(f"无法读取图像文件")

    # 检查坐标是否在图像范围内
    if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
        return False

    # 获取该点的灰度值
    pixel_value = img[y, x]

    # 判断是否超过阈值（认为是血管）
    return pixel_value > threshold

# 侧视图中检测红色区域并返回中心点坐标

def detect_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    output_img = image.copy()
    # 定义红色 HSV 阈值范围
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([170, 50, 50])
    upper2 = np.array([180, 255, 255])

    # 生成两个掩码
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    # 合并掩码
    mask = cv2.bitwise_or(mask1, mask2)
    # 5. 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6. 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_points = []  # 存储中心点 (x, y)

    output_contours = []
    i=0
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 200 > area >= 0 :  # 过滤小噪点（可调）10,200
            # 计算质心（中心点）
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if 150 <= cy <= 400 and is_point_in_vessel((cx, cy), IMG1, 128):  # 只保留y坐标在一定范围内的点
                #if 0 <= cy <= 257:  # 用于识别大机械臂末端计算变换矩阵
                    center_points.append((cx, cy))
                    output_contours.append(cnt)
                    # 绘制轮廓（绿色）

                    cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
                    # 绘制中心点（红色圆）
                    cv2.circle(output_img, (cx, cy), 5, (0, 0, 255), -1)
                    # 可选：在图上标注坐标
                    cv2.putText(output_img, f"({cx},{cy}),{area}", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    return center_points,output_contours,output_img

# 俯视图中检测红色区域并返回中心点坐标
def detect_red2(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    output_img = image.copy()
    # 定义红色 HSV 阈值范围
    lower1 = np.array([0, 30, 50])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([170, 30, 50])
    upper2 = np.array([180, 255, 255])

    # 生成两个掩码
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    # 合并掩码
    mask = cv2.bitwise_or(mask1, mask2)
    # 5. 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6. 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_points = []  # 存储中心点 (x, y)
    output_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 > area >= 30 :  # 过滤小噪点（可调）
            # 计算质心（中心点）
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if 90 <= cy <= 320 and is_point_in_vessel((cx, cy), IMG2, 128):  # 只保留y坐标在一定范围内的点
                    center_points.append((cx, cy))
                    output_contours.append(cnt)
                    # 绘制轮廓（绿色）
                    cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
                    # 绘制中心点（红色圆）
                    cv2.circle(output_img, (cx, cy), 5, (0, 0, 255), -1)
                    # 可选：在图上标注坐标
                    cv2.putText(output_img, f"({cx},{cy}),{area}", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return center_points,output_contours,output_img

def detect_operation_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    output_img = image.copy()
    # 定义红色 HSV 阈值范围
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 100, 100])
    upper2 = np.array([180, 255, 255])

    # 生成两个掩码
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    # 合并掩码
    mask = cv2.bitwise_or(mask1, mask2)
    # 5. 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6. 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_points = []  # 存储中心点 (x, y)
    output_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 800 > area > 200 :  # 过滤小噪点（可调）
            # 计算质心（中心点）
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if 0 <= cy <= 480:  # 只保留y坐标在一定范围内的点
                    if len(center_points) != 0:
                        prev_cx, prev_cy = center_points[0]
                        if cy < prev_cy:
                            continue  # 跳过过于接近的点
                    center_points.append((cx, cy))
                    output_contours.append(cnt)
                    # 绘制轮廓（绿色）
                    cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
                    # 绘制中心点（红色圆）
                    cv2.circle(output_img, (cx, cy), 5, (0, 0, 255), -1)
                    # 可选：在图上标注坐标
                    cv2.putText(output_img, f"({cx},{cy}),{area}", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return center_points,output_contours,output_img

def detect_yellow_and_get_centers(image):
    # 保留原始图像用于绘制
    output_img = image.copy()

    # 2. 转换为 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. 定义黄色 HSV 阈值
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])

    # 4. 生成掩码
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 5. 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6. 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center_points = []  # 存储中心点 (x, y)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if  30 > area > 20:  # 过滤小噪点（可调）
            # 计算质心（中心点）
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center_points.append((cx, cy))

                # 绘制轮廓（绿色）
                cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
                # 绘制中心点（红色圆）
                cv2.circle(output_img, (cx, cy), 5, (0, 0, 255), -1)
                # 可选：在图上标注坐标
                cv2.putText(output_img, f"({cx},{cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return center_points, output_img, mask

# 使用示例
if __name__ == "__main__":

    from utils.camera.utils_camera import Camera
    camera = Camera(0)
    # intr, depth_intrin, image, depth_image, depth_frame = camera.get_image_depth()
    # time.sleep(1)
    # intr, depth_intrin, image, depth_image, depth_frame = camera.get_image_depth()
    # cv2.imwrite('./test_image.png', image)
    try:
        while True:
            # 控制循环频率
            time.sleep(0.2)
            # 从相机获取图像（和深度，如果可用）
            intr, depth_intrin, image, depth_image, depth_frame = camera.get_image_depth()
            if image is None:
                print("Warning: no image returned from camera, retrying...")
                time.sleep(0.1)
                continue
            #centers, result_img, mask = detect_yellow_and_get_centers(image)
            centers, result_img, mask = detect_red(image)
            collision = True
            if collision:
                print('检测到相碰！')
            # 打印坐标信息
            if centers:
                print("检测到的红色区域中心点坐标 (x, y):")
                for i, (x, y) in enumerate(centers, 1):
                    print(f"  区域 {i}: ({x}, {y})")
            else:
                print("未检测到黄色区域")

            # 将 mask 转为 BGR 以便并排显示
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if result_img.shape[:2] != mask_bgr.shape[:2]:
                mask_bgr = cv2.resize(mask_bgr, (result_img.shape[1], result_img.shape[0]))

            # 并排显示原图与掩码
            combined = np.hstack((result_img, mask_bgr))
            cv2.imshow("Yellow Detection - press q to quit", combined)

            # 非阻塞按键检测，按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # 支持 Ctrl-C 退出
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()

    # 结束

# 黄色 HSV 范围：
# H: 15–35（覆盖典型黄色，避开橙色和绿色）
# S ≥ 50（避免灰白区域）
# V ≥ 50（避免过暗区域）
# ⚠️ 实际场景中可能需要根据光照微调，例如在暗光下降低 V 下限，或在强光下提高 S 下限。