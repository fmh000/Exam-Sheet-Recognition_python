#筛选出答题卡每一道题的选项


import cv2
import numpy as np

thresh = cv2.imread('image/thresh.bmp',0)

#找轮廓
cnts,h = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print('总共找到的轮廓数',cnts)

thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

options = []
for cnt in cnts:
    #获取轮廓的矩形包围框
    x,y,w,h = cv2.boundingRect(cnt)
    #计算长宽比
    ar = w/float(h)
    #根据阈值过滤
    if w >= 40 and h >= 40 and ar >= 0.8 and ar <= 1.2:
        options.append(cnt)
print('选项个数：',len(options))

cv2.imshow('thresh',thresh)


#根据选项轮廓的纵坐标y排序，从而获取每一道题的4个选项轮廓

#获取所有选项轮廓的边界（x,y,w,h）
boundingBoxes = [cv2.boundingRect(c) for c in options]

(options,boundingBoxes) = zip(*sorted(zip(options,boundingBoxes),key = lambda x:x[1][1],reverse = False))

#通过循环将1到6题的每题的四个选项过滤出来
for(index,i) in enumerate(np.arange(0,len(options),4)):
    #获取每一道题的四个选项的轮廓边界（x,y,w,h）
    boundingBoxes = [cv2.boundingRect(c) for c in options[i:i+4]]

    #使用x排序，获取按ABCD的顺序的选项轮廓
    (cnts, boundingBoxes) = zip(*sorted(zip(options[i:i + 4], boundingBoxes), key=lambda x: x[1][0], reverse=False))

    #构造和原来大小一样的全黑图用来显示每道题的四个选项轮廓
    image = np.zeros(thresh.shape,dtype = 'uint8')
    # 单独处理每一个选项
    for (k,cnt) in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.drawContours(image,[cnt],-1,(255,255,255),-1)
        cv2.putText(image,str(k),(x-1,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('result' + str(index),image)

    cv2.waitKey()
    cv2.destroyAllWindows()