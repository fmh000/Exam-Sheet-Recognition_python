#对答题卡进行倾斜校正和裁边处理
"""
进行倾斜校正

通过映射矩阵（M）进行透视变化
dst = cv2.warpPerspective(src,M,dsize)

dst:透视处理后的输出图像
src:要进行透视处理的图像
M：3*3的变换矩阵
dsize:输出图像的尺寸

计算变化矩阵
M  = cv2.getPerspectiveTransform(src,dst)
M：3*3的变换矩阵
src:原始图像的四个顶点
dst:目标图像的四个顶点

步骤：
1.预处理
2.找到所有的轮廓（包括答题卡的轮廓）
3.对轮廓进行逼近，找到顶点数为四的轮廓（找到答题卡的轮廓）
4.针对答题卡轮廓进行倾斜校正和裁边
 （1）对原图答题卡轮廓的四个顶点按x进行排序，找到左侧的两个顶点和右侧的两个顶点
 （2）对左侧两个顶点按y排序，确定左上角顶点和左下角顶点
 （3）对右侧两个顶点按y排序，确定右上角顶点和右下角顶点
 （4）计算校正后的答题卡的新宽度（距离最长的边）
 （5）计算校正后的答题卡的新高度（距离最长的边）
 （6）根据新宽度和新高度构造目标图像（矫正后的答题卡图像）的4个顶点
 （7）使用getPerspectiveTransform函数构造转换矩阵M
 （8）使用warpPerspective函数矫正答题卡，并裁边

"""

import cv2
import  numpy as np

def wrap_perspective(image,pts):
    # step4.1：对原图答题卡轮廓的四个顶点按x进行排序，找到左侧的两个顶点和右侧的两个顶点
    x_sorted = pts[np.argsort(pts[:,0]),:]#pts[:,0]就是把四个点的x坐标取出来
    #获取原图答题卡左侧的两个顶点
    left = x_sorted[:2,:]
    #获取原图答题卡右侧的两个顶点
    right = x_sorted[2:,:]

    #step4.2：对左侧两个顶点按y排序，确定左上角顶点和左下角顶点
    left = left[np.argsort(left[:,1]),:]#把两个 y 值 从小到大排序，返回排序后的索引顺序
    (tl,bl) = left#左上角和左下角
    #step4.3:对右侧两个顶点按y排序，确定右上角顶点和右下角顶点
    right = right[np.argsort(right[:,1]),:]
    (tr,br) = right

    src = np.array([tl,tr,br,bl],dtype = 'float32')#歪

    #step4.4:计算校正后的答题卡的新宽度（距离最长的边）
    """
    (x1,y1) (x2,y2)
    distance = sqrt((x1-x2) **2 + (y1-y2) ** 2)
    """
    #计算顶边长 sqrt开平方根
    width_t = np.sqrt(((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2))
    #计算底边长
    width_b = np.sqrt(((bl[0] - br[0])**2 + (bl[1] - br[1])**2))
    new_width  = max(int(width_b),int(width_t))

    #step4.5:计算校正后的答题卡的新高度（距离最长的边）
    height_l = np.sqrt(((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2) )
    height_r = np.sqrt(((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2) )
    new_height = max(int(height_l),int(height_r))

    # step4.6: 根据新宽度和新高度构造目标图像（矫正后的答题卡图像）的4个顶点
    dst = np.array([
        [0,0],
        [new_width-1,0],
        [new_width-1,new_height-1],
        [0,new_height-1],
    ],dtype = 'float32')#正

    #step4.7:使用getPerspectiveTransform函数构造转换矩阵M
    M = cv2.getPerspectiveTransform(src,dst)#M就是把歪的四个角转成正的四个角的方法

    #step4.8: 使用warpPerspective函数矫正答题卡，并裁边
    result = cv2.warpPerspective(image,M,(new_width,new_height))
    return result

#step1：预处理
img = cv2.imread('image/slant_card.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gaussion = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,50,200)

#step2:找到所有的轮廓（包括答题卡的轮廓）
cts, h = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,cts,-1,(0,0,255),3)
cv2.imshow('draw_contours',img)

for c in cts:
    #step3:对轮廓进行逼近，找到顶点数为四的轮廓（找到答题卡的轮廓）
    p = 0.01 *  cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,p,True)
    if len(approx) == 4:
        print(approx)
        print(approx.reshape(4,2))

        #step4:针对答题卡轮廓进行倾斜校正和裁边
        #approx.reshape(4,2):src
        result = wrap_perspective(img,approx.reshape(4,2))#approx：找到的答题卡 4 个角点用 .reshape(4, 2) 变成 4 行 2 列
        break

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


