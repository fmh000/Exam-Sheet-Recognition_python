"""
step1: 设置选项和标准答案
step2: 读取答题卡图像
step3: 图像预处理（灰度、高斯滤波、阈值变换）
step4: 获取轮廓
step5: 获取包含所有轮廓的矩形（为了方便统计有效像素点）
step6: 将轮廓矩形区域升序排序（根据矩形区域x进行排序）
step7. 通过统计每一个选项轮廓中非零点像素个数得到填涂的选项
       两种方式
step8: 显示结果
"""
import numpy as np#将图像的像素点转换成计算器好处理的多维数组
import cv2#专门用来处理图像的

#step1:设置选项和答案
#定义选项字典
answer_key = {0:'A',1:'B',2:'C',3:'D'}
#定义标准答案
Answer = 'B'

#step2: 读取答题卡图像
img = cv2.imread('image/single_selection.jpg')#定义一个img读取选择的答题卡图像的像素点
cv2.imshow('original',img)#把存在img的原始答题卡图像显示出来，方便核对有没有读对图

#step3: 图像预处理（灰度、高斯滤波、阈值变换）
#转换为灰度图像，调用cv2里的COLOR_BGR2GRAY函数让这个彩色图像BGR(蓝绿红的顺序)转换为GRAY灰度图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#高斯滤波（去噪）
#使用cv2里的高斯滤波函数GaussianBlur，给灰度图像做高斯模糊，让图像更平滑
#gray是刚刚做好的灰度图，（5,5）是高斯核的大小，核越大模糊越强，5是中等模糊，0是标准差sigmaX，控制模糊的扩散程度，一般就写0
gaussian  = cv2.GaussianBlur(gray,(5,5),0)

#阈值变换，将所有选项处理为前景为白色，背景为黑色的图像，反二值处理
#自动算出来的阈值存在ret里，处理好的图像存在thresh里
#cv2.threshold()是专门把图像变成黑白二值图的函数，gary要处理的灰度图，0代表不手动定阈值让电脑自动算，255最大值最亮值纯白，
#inv=反转，让亮的变黑暗的变白，让纸变成黑的，涂的选项变成白的
#cv2.THRESH_OTSU固定写法OTSU大津法自动寻找最佳阈值
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow('thresh',thresh)#弹出一个窗口把处理好的图显示出来

#step4: 获取轮廓
#cnts是所有找到的轮廓列表（ABCD的选项框都会被存在这里）hierarchy是轮廓的层级
#将刚才处理好的图像thresh复制一份防止原图被修改，cv2.RETR_EXTERNAL只找最外层轮廓，cv2.CHAIN_APPROX_SIMPLE压缩轮廓点
cnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#step5: 获取包含所有轮廓的矩形（为了方便统计有效像素点）
#根据每一个选项的矩形区域的横坐标进行排序（升序）
#获取包含轮廓的最小矩形区域：cv2.boundingRect(c) c是轮廓的点坐标集合
#（x,y,width,height）x,y是矩形左上角坐标，w宽度，h高度
boundingBoxes = [cv2.boundingRect(c) for c in cnts]
# print(boundingBoxes)
#step6: 将轮廓矩形区域升序排序（根据矩形区域x进行排序）
def sortFun(b):#定义一个函数辅助排序，告诉程序按照x坐标排序
    return b[0][0]#返回x坐标
(boundingBoxes,cnts) = zip(*sorted(zip(boundingBoxes,cnts),key=lambda b:sortFun(b),reverse=False))#
#打包好的zip boundingbBoxes和cnts就赋值给b了再让b按照sortFun方法排序，升序从小到大，zip(*排序好的列表)拆开
# print(boundingBoxes)

# step7. 通过统计每一个选项轮廓中非零点像素个数得到填涂的选项
max = 0#记录：最多的白色像素数
index = 0#循环计数用
answer_index = 0#填涂选项的索引(0=A,1=B...)
answer = 'A'

for (j,c) in enumerate(cnts):#enumerate是给列表里的每一个东西自动加上一个序号
# j表示第几个选项，c表示当前选项的轮廓
    #构建一个与原始图像尺寸一致的纯黑图像，用来保存每一个选项
    mask = np.zeros(gray.shape,dtype='uint8')#np.zeros创建一个全是0的数组，0代表黑色就是全黑
#gary.shape是指灰度图的高度，宽度，就是新建的全黑图与原图大小一样
#dtype = 'unit8'，unit8=0-255的整数，创建蒙版时必须用它
    #绘制所有的轮廓
    cv2.drawContours(mask,[c],-1,255,-1)
    #画在mask上，画出轮廓填充成白色，[c]是当前选项的轮廓,轮廓索引固定是-1,255白色，-1代表填充内部，把轮廓内填满白色
    mask = cv2.bitwise_and(thresh,mask)#cv2.bitwise_and按位与运算，只有thresh和mask都有白色结果才是白色
# 这步是在轮廓填充内扣出选项，让选项框外面的变成全黑，只有选项是白色的，这里的mask已经是被绘制好轮廓的了
    cv2.imshow('mask'+str(index),mask)#'mask'+str(index)是窗口名字对应着选项
    total = cv2.countNonZero(mask)#统计当前选项框里有多少个白色像素点
    # print(total)
    if total>max:#如果选项的像素大于之前记录的最大像素
        max = total#更新最大值
        answer = answer_key.get(index)#在字典里找索引对应的答案
        answer_index = index#记住当前选项的编号就是答案
    index += 1

# step8: 显示结果
print("学生的选择：",answer)
if answer == Answer:
    color = (0,255,0)#回答正确，用绿色表示
    msg = '回答正确'#msg是massage一段文字
else:
    color = (0,0,255)#回答错误，用红色表示
    msg = '回答错误'
#img答题卡图像，cnts[answer_index]是答案选项的轮廓，-1固定写法代表轮廓，color颜色，2代表线条粗细两个像素宽
cv2.drawContours(img,cnts[answer_index],-1,color,2)
cv2.imshow('result',img)#把画好了答案的最终图片显示出来
print(msg)
cv2.waitKey(0)#让图片停在屏幕上，0指无限等待直到按任意键关闭
cv2.destroyAllWindows()#关闭所有opencv窗口清理干净






