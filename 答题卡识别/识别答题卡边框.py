"""识别答题卡的步骤
1. 图像预处理
2. 答题卡处理
3. 筛选出所有的选项
4. 将选项按照题目分组
5. 单独处理每一道题的选项
6. 显示结果

1. 图像预处理
(1) 变成灰度(色彩空间变换)
(2) 高斯滤波
(3) Canny边缘检测
"""
import cv2

img = cv2.imread("image/answer_card.jpg")
cv2.imshow('origial',img)

#转换为灰度图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
#高斯滤波
gaussian = cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow('gaussian',gaussian)

#Canny边缘检测
#Canny(image,min,max)
"""
color<min:被抛弃
min<=color<=max:正常图像
color>max：被认为是真正的边界
"""
edged = cv2.Canny(gaussian,50,200)
cv2.imshow('edged',edged)
#获取轮廓存入cts
cts,h = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
max_area = 0 #最大面积
max_c_index = 0#最大面积的轮廓索引

for i,c in zip(range(len(cts)),cts):
    area = cv2.contourArea(c)  # 计算当前轮廓面积
    p = 0.01 * cv2.arcLength(c, True)
    # 精度就是原始曲线与拟合曲线的最大距离，cv2.approxPolyDP()轮廓拉直简化csts[0]是要简化的那个轮廓，True轮廓是闭合的
    approx = cv2.approxPolyDP(c, p, True)
    print(area, len(approx))
    # if area > max_area:
    #     max_rea = area
    #     max_c_index = i
    if len(approx) == 4:
        max_c_index = i
        break
print(max_c_index)

cv2.drawContours(img,cts,-1,(0,0,255),3)

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()

