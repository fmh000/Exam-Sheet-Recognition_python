"""
逼近

"""
import cv2
from numpy.ma.testutils import approx

img1 = cv2.imread('image/card1.jpg')
cv2.imshow('img1',img1)

img2 = cv2.imread('image/card2.jpg')
cv2.imshow('img2',img2)

def cst_num(img):
    #灰度处理
    gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #阈值变换处理好的图像存入binary，小于阈值127变黑大于变白
    ret,binary = cv2.threshold(gary,127,255,cv2.THRESH_BINARY)
    #获取轮廓存入csts
    csts,h = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print('轮廓顶点个数',len(csts[0]))#csts[0]是第一个轮廓，len是这个轮廓有多少个点
    #用轮廓周长的百分之一作为精度,cv2.arcLength()计算轮廓的周长
    p = 0.01*cv2.arcLength(csts[0],True)
    #精度就是原始曲线与拟合曲线的最大距离，cv2.approxPolyDP()轮廓拉直简化csts[0]是要简化的那个轮廓，True轮廓是闭合的
    approx = cv2.approxPolyDP(csts[0],p,True)
    print("逼近后的顶点个数",len(approx))

print('矩形')
cst_num(img1)
print('梯形')
cst_num(img2)
cv2.waitKey()
cv2.destroyAllWindows()


