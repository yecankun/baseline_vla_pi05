import time

import cv2
import numpy as np


def get_color_regions(image):

    # 1. 加载图像
    #image = cv2.imread(img_path)  # 替换为你的图像文件路径
    # 2. 找到红色像素
    # 假设红色的标准是：红色通道高，绿色和蓝色通道低
    #red_mask = (image[:,:,2] > 150) & (image[:,:,1] < 100) & (image[:,:,0] < 100)
    red_mask = (image[:, :, 2] > 140) & (image[:, :, 1] < 150)& (image[:, :, 1] >130) & (image[:, :, 0] < 150) #找红色
    blue_mask=(image[:, :, 2] < 80) & (image[:, :, 1] < 80) & (image[:, :, 0] >75) #找蓝色
    yellow_mask=((image[:, :, 2] >155)&(image[:, :, 2] <175)
                 & (image[:, :, 1] >130)&(image[:, :, 1] <155)
                 & (image[:, :, 0] <125)& (image[:, :, 0] >100)
                 |((image[:, :, 2] >130)&(image[:, :, 2] <145)
                   & (image[:, :, 1] > 115) & (image[:, :, 1] < 135)
                   & (image[:, :, 0] < 105) & (image[:, :, 0] > 90)
                   )) #找黄色
    new_yellow_mask1=((image[:, :, 2] >160)&(image[:, :, 2] <180)
                 & (image[:, :, 1] >130)&(image[:, :, 1] <150)
                 & (image[:, :, 0] <130)& (image[:, :, 0] >115))
    
    new_yellow_mask=((image[:, :, 2]-image[:, :, 1]<=25)&
                     (image[:, :, 2]-image[:, :, 1]>=15)&
                     (image[:, :, 1]-image[:, :, 0]>=10)&
                     (image[:, :, 1]-image[:, :, 0]<=30)&
                     (image[:, :, 0] < 150) & (image[:, :, 0] > 120)&
                     (image[:, :, 2]<170)
                 )
    # 3. 使用连通组件分析找出所有红色联通区域
    # 使用cv2.connectedComponents()找到连通区域
    num_labels, labels = cv2.connectedComponents(yellow_mask.astype(np.uint8))

    # 4. 计算每个连通区域的中心坐标
    centers = []
    for label in range(1, num_labels):  # 忽略背景标签（0）
        # 找到当前连通区域的坐标
        coords = np.column_stack(np.where(labels == label))
        # 计算区域的中心坐标
        center = np.mean(coords, axis=0).astype(int)
        centers.append(center)
        # 在原图上将该区域标记为蓝色
        #image[coords[:, 0], coords[:, 1]] = [255, 0, 0]  # 用蓝色 (255, 0, 0)
    #cv2.destroyAllWindows()
    # 输出所有红色区域的中心坐标
    #print(f"红色区域的中心坐标: {centers}")
    return centers

if __name__=='__main__':
    root_path="D:/datasource/mydata/image/"
    l = 44
    list=[]
    for i in range(l):
        img1 = cv2.imread(root_path + "{}.png".format(i + 1))
        #img1 = finalize_mask(img1, points, region)
        centers=get_color_regions(img1)

        list.append(centers[0])

        # 5. 显示图像
        for center in centers:
            # 在中心位置绘制一个小圆圈，表示该区域的中心
            cv2.circle(img1, (center[1], center[0]), 5, (0, 255, 0), -1)  # 绿色圆圈

        # 显示最终结果
        cv2.imshow('Red Regions with Centers', img1)
        # cv2.waitKey(0)
        key = cv2.waitKey(1)
        time.sleep(0.4)

        # key = cv2.waitKey(1)
        # time.sleep(0.3)

    base=list[21]
    print(base)
    for i in range(l):
        print(list[i][0]-base[0])
        print(list[i][1]-base[1])
