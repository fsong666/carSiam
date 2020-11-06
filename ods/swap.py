# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import numpy as np
import math
# 读取鱼眼图片
path = '1158.png'
img = cv2.imread(path)
# 设置灰度阈值
T = 140

# 转换为灰度图片
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 提取原图大小
rows, cols = img.shape[:2]
print(rows, cols)


# 从上向下扫描
for i in range(0, rows, 1):
    for j in range(0, cols, 1):
        if img_gray[i, j] >= T:
            if img_gray[i + 1, j] >= T:
                top = i
                break
    else:
        continue
    break
print('top =', top)


# 从下向上扫描
for i in range(rows - 1, -1, -1):
    for j in range(0, cols, 1):
        if img_gray[i, j] >= T:
            if img_gray[i - 1, j] >= T:
                bottom = i
                break
    else:
        continue
    break
print('bottom =', bottom)


# 从左向右扫描
for j in range(0, cols, 1):
    for i in range(top, bottom, 1):
        if img_gray[i, j] >= T:
            if img_gray[i, j + 1] >= T:
                left = j
                break
    else:
        continue
    break
print('left =', left)


# 从右向左扫描
for j in range(cols - 1, -1, -1):
    for i in range(top, bottom, 1):
        if img_gray[i, j] >= T:
            if img_gray[i, j - 1] >= T:
                right = j
                break
    else:
        continue
    break
print('right =', right)


# 计算有效区域半径
R = max((bottom - top) / 2, (right - left) / 2)
R = 300
print('R =', R)


# 提取有效区域
img_valid = img[top:int(top + 2 * R), left:int(left + 2 * R)]
cv2.imwrite('./TestResults/result.jpg', img_valid)


#经度矫正法
m, n, k = img_valid.shape[:3]
print('m,n,k',m,n,k)

result = np.zeros((m,n,k))
Undistortion = []
x = n/2 + 200
y = m/2
for u in range(m):
    for v in range(n):
        i = u
        j = round(math.sqrt(R ** 2 - (y - u) ** 2) * (v - x) / R + x)
        if (R ** 2 - (y - u) ** 2 < 0):
            continue
        result[u,v,0]=img_valid[i,j,0]
        result[u,v,1]=img_valid[i,j,1]
        result[u,v,2]=img_valid[i,j,2]
Undistortion = np.uint8(result)


# 显示图片
cv2.namedWindow("yuantu", 0)
cv2.resizeWindow("yuantu", 600, 1200)
cv2.imshow("yuantu", img)

cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 600, 1200)
cv2.imshow("result", Undistortion)
cv2.waitKey(0)
