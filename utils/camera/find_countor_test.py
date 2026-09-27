import cv2
import numpy as np

# 读取图像
image_path = './test_image.png'  # 替换为你的图片路径
img = cv2.imread(image_path)
# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊，减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用 Canny 边缘检测
edges = cv2.Canny(blurred, 50, 150)
# 寻找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 筛选较大的轮廓（排除小噪声）
min_area = 20  # 可根据实际情况调整
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
# 定义辅助函数：检查线段AB和CD是否相交

def ccw(A, B, C):

    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# 检查矩形是否与轮廓相交

def is_rect_intersect_contour(rect, contour):
    x, y, w, h = rect
    rectangle_corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for i in range(len(contour)):
        next_i = (i + 1) % len(contour)
        for j in range(4):
            next_j = (j + 1) % 4
            if intersect(rectangle_corners[j], rectangle_corners[next_j],

                         tuple(contour[i][0]), tuple(contour[next_i][0])):

                return True

    return False

rect = (370, 250, 30, 30)  # 定义矩形的位置和大小(x, y, width, height)

# 判断矩形是否与轮廓相碰

for contour in filtered_contours:
    if is_rect_intersect_contour(rect, contour):
        print("矩形与轮廓相碰")
        break
else:
    print("矩形不与任何轮廓相碰")

result = img.copy()
for approx in filtered_contours:
    cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)  # 绿色轮廓
cv2.rectangle(result, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0), 2)  # 蓝色矩形

# 显示结果
cv2.imshow('Detected Contours', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
