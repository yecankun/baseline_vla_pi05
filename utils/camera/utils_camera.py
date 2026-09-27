import os

import pyrealsense2 as rs
import numpy as np
import cv2

vessel_edge1_path = '/home/xwj/桌面/project_2026/frunet/runs/predicted_vessel_wall.png'
vessel_edge1 = cv2.imread(vessel_edge1_path)

class Camera:
    def __init__(self,id):
        self.position = [0, 0]
        # 确定图像的输入分辨率与帧率
        resolution_width = 640  # pixels
        resolution_height = 480  # pixels
        frame_rate = 15  # fps

        # 注册数据流，并对其图像
        self.align = rs.align(rs.stream.color)
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
        ### d435i
        #
        rs_config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, frame_rate)
        # check相机是不是进来了
        connect_device = []
        for d in rs.context().devices:
            print('Found device: ',
                  d.get_info(rs.camera_info.name), ' ',
                  d.get_info(rs.camera_info.serial_number))
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                connect_device.append(d.get_info(rs.camera_info.serial_number))

        if len(connect_device) < 2:
            print('Registrition needs two camera connected.But got one.')
        if(id >= len(connect_device)):
            print('ERROR: id > number of camera')

        # 确认相机并获取相机的内部参数
        self.pipeline1 = rs.pipeline()
        rs_config.enable_device(connect_device[id])
        self.profile = self.pipeline1.start(rs_config)
        # device1 = profile.get_device()
        # device1.hardware_reset()
        self.collision_detector = None
        self.combined_image = None

    def get_image_depth(self):
        # 等待数据进来
        frames1 = self.pipeline1.wait_for_frames()
        # 将进来的RGBD数据对齐
        aligned_frames1 = self.align.process(frames1)
        # 将对其的RGB—D图取出来
        color_frame = aligned_frames1.get_color_frame()

        depth_frame = aligned_frames1.get_depth_frame()
        # depth_data1 = np.asanyarray(depth_frame.get_data(), dtype="float64")

        # cv2.imwrite(os.path.join((save_path), "{}.png".format(data_index)), color_image1)
        # np.save(os.path.join((save_path), "{}".format(data_index)), depth_data1)
        ############### 相机参数的获取 #######################
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                             'ppx': intr.ppx, 'ppy': intr.ppy,
                             'height': intr.height, 'width': intr.width,
                             'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
                             }
        # 保存内参到本地
        # with open('./intrinsics.json', 'w') as fp:
        #     json.dump(camera_parameters, fp)
        #######################################################

        depth_image = np.asanyarray(depth_frame.get_data())  # 深度图（默认16位）
        # depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  #深度图（8位）
        # depth_image_3d = np.dstack((depth_image_8bit,depth_image_8bit,depth_image_8bit))  #3通道深度图
        color_image = np.asanyarray(color_frame.get_data())  # RGB图
        # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
        return intr, depth_intrin, color_image, depth_image, depth_frame
        # return color_image1,depth_data1

    def get_real_pos(self,depth_intrin, depth_frame, coord):
        x = coord[0]
        y = coord[1]
        dis = depth_frame.get_distance(x, y)  # （x, y)点的真实深度值
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y],
                                                            dis)  # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
        print('相机参考系的坐标:{}'.format(camera_coordinate))
        return camera_coordinate

    def predict_pos(self,vessel=vessel_edge1):
        from frunet.collision_detect2 import CollisionDetector
        from utils.camera.hsv_locate import detect_red

        if self.collision_detector is None:
            self.collision_detector = CollisionDetector(6.0)
        if vessel is None:
            raise FileNotFoundError(f"无法读取血管壁图像: {vessel_edge1_path}")
        # 从相机获取图像（和深度，如果可用）
        # 创建一个可调整大小的窗口（关键步骤）
        cv2.namedWindow("Red Detection", cv2.WINDOW_NORMAL)

        # 可选：设置窗口的具体尺寸（宽 x 高）
        cv2.resizeWindow("Red Detection", width=800, height=600)
        print('侧视---------------------')
        intr, depth_intrin, image, depth_image, depth_frame = self.get_image_depth()
        if image is None:
            print("Warning: no image returned from camera, retrying...")
            return
        # centers, result_img, mask = detect_yellow_and_get_centers(image)
        centers, result_contours, result_img = detect_red(image)
        # 将 血管模型分割结果 转为 BGR 以便并排显示
        #mask_bgr = cv2.cvtColor(vessel, cv2.COLOR_GRAY2BGR)
        mask_bgr = vessel.copy()
        # 碰撞检测
        #collision_info = analyze_vessel_collision(vessel, result_contours, dist_threshold=2.0)
        collision_info = self.collision_detector.simple_analyze(vessel_mask=vessel, target_contours=result_contours)
        if collision_info["is_collided"]:
            print('检测到相碰！')
            (cx,cy) = centers[0]
            cv2.putText(mask_bgr, f"Collide", (cx + 10, cy + 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 在掩码上绘制轮廓
        for cnt in result_contours:
            cv2.drawContours(mask_bgr, [cnt], -1, (0, 255, 0), 2)
        if result_img.shape[:2] != mask_bgr.shape[:2]:
            mask_bgr = cv2.resize(mask_bgr, (result_img.shape[1], result_img.shape[0]))

        # 并排显示原图与掩码
        combined = np.hstack((result_img, mask_bgr))
        cv2.imshow("Red Detection", combined)
        cv2.waitKey(1)
        # 打印坐标信息
        if centers:
            print("检测到的红色区域中心点坐标 (x, y):")
            for i, (x, y) in enumerate(centers, 1):
                print(f"  区域 {i}: ({x}, {y})")
        else:
            print("未检测到红色区域")
            self.combined_image = combined
            return [0, 0, 0], result_contours
        (cx,cy) = centers[0]
        self.position = [cx, cy]
        self.combined_image = combined
        real_coord = self.get_real_pos(depth_intrin, depth_frame, [cx,cy])
        return real_coord, result_contours

    def img_to_camera_transform(self, coord):
        intr, depth_intrin, image, depth_image, depth_frame = self.get_image_depth()
        x = coord[0]
        y = coord[1]
        dis = depth_frame.get_distance(x, y)  # （x, y)点的真实深度值
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y],
                                                            dis)  # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
        # print('相机参考系的坐标:{}'.format(camera_coordinate))
        return camera_coordinate

    def predict_pos1(self):
        from utils.camera.get_colors import get_color_regions

        intric, d_intric, img1, depth, f = self.get_image_depth()
        coords = get_color_regions(img1)
        n = len(coords)
        center = [0, 0]
        for j in range(n):
            c = coords[j]
            y = (c[0])  # y
            x = (c[1])  # x
            if y < 420 :
                center=c
                break
            # [0,400] [640,220]
            # y=-0.28125(x-0)+400
        if n == 0 or center[1] == 0:
            print('没有识别到')
            cv2.imshow('Yellow Regions with Centers', img1)
            # cv2.waitKey(0)
            key = cv2.waitKey(1)
            return [0, 0, 0]
        #print('预测的导丝头坐标:{}'.format(center))

        center = np.array(center)
        # 在中心位置绘制一个小圆圈，表示该区域的中心
        cv2.circle(img1, ((int)(center[1]), (int)(center[0])), 5, (0, 255, 0), -1)  # 绿色圆圈
        # 显示最终结果
        cv2.imshow('Yellow Regions with Centers', img1)
        key = cv2.waitKey(1)
        reverse_coord = [center[1], center[0]]
        real_coords = self.get_real_pos(d_intric, f, reverse_coord)
        return real_coords


def test_locate():
    import time
    def write_in(file_name: str, data):
        with open(file_name, encoding="utf-8", mode="a") as file:
            file.write(str(data) + '\n')
    cam = Camera(0)
    intr, depth_intrin, color_image, depth_image, depth_frame = cam.get_image_depth()
    time.sleep(1)
    index = 1
    save_path = "/home/xwj/桌面/project_2026/yolodata"

    while True:
        intr, depth_intrin, color_image, depth_image, depth_frame = cam.get_image_depth()
        #cv2.imwrite(f"{save_path}/image2/{index}.png", color_image)
        cam.predict_pos()
        #write_in(f"{save_path}/pos2.txt", f"{cam.position[0]}, {cam.position[1]}")
        index+=1
        time.sleep(0.1)



if __name__ == '__main__':
    test_locate()
