import cv2
import numpy as np
import matplotlib.pyplot as plt

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
min_area = 100  # 可根据实际情况调整
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# 绘制轮廓到原图上
result = img.copy()
cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)  # 绿色轮廓

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Detected Contours")
plt.axis('off')

plt.tight_layout()
plt.show()

# 可选：保存结果
#cv2.imwrite('detected_contours.jpg', result)