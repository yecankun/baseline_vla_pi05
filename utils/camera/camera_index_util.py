import time

import cv2
import os

def get_camera_name(video_index):
    """获取 /dev/video{index} 对应的摄像头名称"""
    try:
        with open(f"/sys/class/video4linux/video{video_index}/name", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading name for video{video_index}: {e}")
        return None

def find_hd_pro_c920_index(camera_name="HD Pro Webcam C920"):
    """查找 HD Pro Webcam C920 的 OpenCV 索引"""
    for i in range(0, 20):  # 假设最多到 video19
        name = get_camera_name(i)
        print(name)
        if name and camera_name in name:
            # 可选：验证是否能打开
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                print('find hd pro c920 index:', i)
                return i
    return None

def find_intel_realsense_index(camera_name="Intel(R) RealSense(TM)"):
    """查找 Intel RealSense 摄像头的 OpenCV 索引"""
    for i in range(0, 20):  # 假设最多到 video19
        name = get_camera_name(i)
        if name and camera_name in name:
            # 可选：验证是否能打开
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                print('find intel realsense index:', i + 2)
                return i + 2
    return None

if __name__ == "__main__":
    index = find_hd_pro_c920_index()
    if index is not None:
        print(f"Found HD Pro Webcam C920 at index: {index}")
        cap = cv2.VideoCapture(index)
    else:
        print("HD Pro Webcam C920 not found.")
    index2 = find_intel_realsense_index()
    if index2 is not None:
        print(f"Found Intel RealSense at index: {index2}")
        cap2 = cv2.VideoCapture(index2)
        while True:
            time.sleep(0.5)
            ret, frame = cap2.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
    else:
        print("Intel RealSense not found.")