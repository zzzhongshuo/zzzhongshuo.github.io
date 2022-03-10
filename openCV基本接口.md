# opencv 基本接口介绍

以下所有操作都基于这三个库：

import cv2

import numpy as np

import matplotlib.pylab as plt

## **图片读取**

img=cv2.imread('image/lenacolor.png', cv2.IMREAD_UNCHANGED)

- **原图展示**
    
    cv2.IMREAD_UNCHANGED
    
- **灰度图展示**
    
    cv2.IMREAD_GRAYSCALE
    
- **彩色图展示**
    
    cv2.IMREAD_COLOR
    

## **图片保存**

cv2.imwrite('image/gray_test.jpg',img)

## **图片展示**

cv2.imshow('original',img)

## **图片暂停展示**

cv2.waitKey(num)

- **按键输入消失**
    
    num<0
    
- **0或不填系数 ，一直不消失**
    
    num==0
    
- **停滞num秒**
    
    num>0
    

## 关闭所有窗口

cv2.destroyAllWindows()

## 图像赋值

- **基本操作**
    
    img[100,100]=255         #灰度图赋值
    
    img[100,100,0]=255        #彩色图单通道赋值
    
    img[100,100]=[255,255,255]       #彩色图多通道赋值
    
- **numpy操作**
    
    img.item(100, 100, 2)          #获得（100,100）点，2通道的值
    
    img.itemset((100, 100, 2), 255)        #设置（100,100）点2通道的值
    

## 获取图像属性

- **获取BGR图 高、宽、深度**
    
    w = img.shape[0]
    
    h = img.shape[1]
    
- **获得图片大小 h*w 或 h*w*d**
    
    img_size=img.size
    
- **获得图片数据类型**
    
    img.dtype
    

## 感兴趣区域ROI （region of interest）

- **获得面部图像**
    
    face= img[220:400, 250:350]
    
- **粘贴脸部图像，可以跨图粘贴**
    
    img[0:180, 0:100]=face
    

## 通道分解合并

- **通道分解方案1**
    
    b=img[:,:,0]
    
    g=img[:,:,1]
    
    r=img[:,:,2]
    
- **通道分解方案2**
    
    b,g,r=cv2.split(img)
    
- **通道合并**
    
    rgb=cv2.merge([r,g,b])
    
- **只显示蓝色通道**
    
    b=cv2.split(a)[0]
    
    g = np.zeros((rows,cols),dtype=a.dtype)
    
    r = np.zeros((rows,cols),dtype=a.dtype)
    
    m=cv2.merge([b,g,r])
    

## 图像加法

- **超过255则为0**
    
    result1= img1 + img2
    
- **超过255则为255**
    
    result2=cv2.add(img1, img2)
    
- **图像带权重融合，第5个参数为偏移量**
    
    result=cv2.addWeighted(img1,0.5,img2,0.5, 0)
    

## 图像类型转换

- **彩色图转灰度图**
    
    img2=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
- **BGR图转RGB图（重点：opencv的通道是 蓝、绿、红跟计算机常用的红、绿、蓝通道相反）**
    
    img2=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
- **灰度图转BGR图,每个通道都是之前的灰度值**
    
    img2=cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    

## 图像缩放 （宽、高）

- **图片缩放->(200,100)**
    
    img2=cv2.resize(img1, (200, 100))
    
- **按比例缩放->(0.5,1.2)**
    
    img2=cv2.resize(img1, (round(cols * 0.5), round(rows * 1.2)))
    
- **按比例缩放，参数版**
    
    img2=cv2.resize(img1, None, fx=1.2, fy=0.5)
    

## 图像翻转

- **上下翻转**
    
    img2=cv2.flip(img1, 0) 
    
- **左右翻转**
    
    img2=cv2.flip(img1, 1)       
    
- **上下、左右翻转**
    
    img2=cv2.flip(img1, -1)       
    

## 图像移动、旋转、缩放

- **图像移动=>(100,200)**
    
    M = np.float32([[1, 0, 100], [0, 1, 200]])
    
    b=cv2.warpAffine(img1, M, (height, width))
    
- **图像中心、旋转45度、缩放0.6**
    
    M=cv2.getRotationMatrix2D((height/2,width/2),45,0.6)
    
    img2=cv2.warpAffine(img1, M, (height, width))
    
- **图像菱形转换**
    
    p1=np.float32([[0,0],[cols-1,0],[0,rows-1]])            #左上角、右上角、左下角
    
    p2=np.float32([[0,rows*0.33],[cols*0.85,rows*0.25],[cols*0.15,rows*0.7]])
    
    M=cv2.getAffineTransform(p1,p2)
    
    dst=cv2.warpAffine(img,M,(cols,rows))
    

## 图像阈值转换 、二值化

- **图像二值化，阈值127,r为返回阈值，b为二值图**
    
    r,b=cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
    
- **图像反二值化**
    
    r,b=cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV) 
    
- **低于threshold则为0**
    
    r,b=cv2.threshold(a,127,255,cv2.THRESH_BINARY) 
    
- **高于threshold则为0**
    
    r,b=cv2.threshold(a,127,255,cv2.THRESH_BINARY_INV) 
    
- **截断=>高于threshold则为threshold**
    
    r,b=cv2.threshold(a,127,255,cv2.THRESH_TRUNC) 
    

## 图像平滑处理

- **均值滤波**
    
    img2=cv2.blur(img1, (5, 5))       #sum(square)/25
    
- **normalize=1 均值滤波,normalize=0 区域内像素求和**
    
    img1=cv2.boxFilter(img, -1, (2, 2), normalize=1)
    
- **高斯滤波,第三个参数是方差,默认0计算公式： sigmaX=sigmaxY=0.3((ksize-1)*0.5-1)+0.8 (注:卷积核只能是奇数)**
    
    img1=cv2.GaussianBlur(img, (3, 3), 0)  #距离像素中心点近的权重较大，以高斯方式往四周分布
    
- **中值滤波,效果非常好?**
    
    img1=cv2.medianBlur(img,3)  #获得中心点附近像素排序后的中值
    

## 形态学操作

- **图像腐蚀，k为全1卷积核**
    
    k=np.ones((5,5),np.uint8)
    
    img1=cv2.erode(img, k, iterations=2)
    
- **图像膨胀**
    
    k=np.ones((5,5),np.uint8)
    
    img1=cv2.dilate(img, k, iterations=2)
    
- **图像开运算 （先腐蚀后膨胀），去掉图形外侧噪点**
    
    k=np.ones((5,5),np.uint8)
    
    img1=cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=2)
    
- **图像闭运算（先膨胀后腐蚀） ，去掉图形内侧噪点**
    
    k=np.ones((5,5),np.uint8)
    
    img1=cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=2)
    
- **图像梯度运算（膨胀-腐蚀）**
    
    k=np.ones((5,5),np.uint8)
    
    img1=cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)
    
- **高帽运算 （原图-开运算），获得图形外噪点**
    
    k=np.ones((5,5),np.uint8)
    
    img1=cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)
    
- **黑帽运算（闭运算-原图），获得图像内噪点**
    
    k=np.ones((10,10),np.uint8)
    
    img1=cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)
    

## 图像梯度，边缘提取

- **sobel梯度边缘提取，卷积核竖向[[-1,-2,-1][0,0,0][1,2,1]]**
    
    sobelx = cv2.Sobel(o,cv2.CV_64F,1,0,ksize=3) #横向边缘提取
    
    sobely = cv2.Sobel(o,cv2.CV_64F,0,1,ksize=3) #竖向边缘提取
    
    sobelx = cv2.convertScaleAbs(sobelx) # 负值取正，图像展示只能有正值
    
    sobely = cv2.convertScaleAbs(sobely)
    
    sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0) #图像融合
    
- **scharr梯度边缘提取，卷积核竖向[[-3,-10,-3][0,0,0][3,10,3]] ，scharr比sobel卷积核过滤出更多细节**
    
    scharrx = cv2.Scharr(o,cv2.CV_64F,1,0)
    
    scharry = cv2.Scharr(o,cv2.CV_64F,0,1)
    
    scharrx = cv2.convertScaleAbs(scharrx) # 负值取正
    
    scharry = cv2.convertScaleAbs(scharry)
    
    scharrxy = cv2.addWeighted(scharrx,0.5,scharry,0.5,0) #图像融合
    
- **拉普拉斯梯度,边缘提取版本1 , 拉普拉斯图像梯度 [[0,1,0][1,-4,1][0,1,0] ]**
    
    img1 = cv2.Laplacian(img, cv2.CV_64F)
    
    img1 = cv2.convertScaleAbs(img1)
    
- **拉普拉斯梯度,边缘提取版本2，结果略有不同**
    
    f=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    
    img1=cv2.filter2D(img, -1, f)
    

## canny边缘检测

- **canny边缘检测理论**
    
    sobel梯度大小：0.5|x|+0.5|y|
    

高斯滤波                             梯度方向：arctan(y/x)                       同方向上保留最大梯度

去噪------------------------->梯度------------------------------------->非极大值抑制---------------------------->

跟高阈值连通的线会保留

滞后阈值--------------------->out

- **canny边缘检测代码**
    
    img1 = cv2.Canny(img,100,200)     #参数：图片、低阈值、高阈值
    

## 图像金字塔

- **图片向下采样，高斯滤波 1/2 删掉偶数列**
    
    img1 = cv2.pyrDown(img)
    
- **图片向上采样 ，面积*2 高斯滤波*4 ，下采样为不可逆运算**
    
    img3=cv2.pyrUp(img2)
    
- **计算拉普拉斯金字塔**
    
    img1 = cv2.pyrDown(img) #下采样
    
    img2=cv2.pyrUp(img1) #上采样
    
    img3=img-img2
    

## 图像轮廓标注

- **1.灰度图转化**
    
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    
- **2.二值图转化**
    
    dep,img_bin=cv2.threshold(gray_img,128,255,cv2.THRESH_BINARY) 
    
- **3.获得图像轮廓**
    
    image_,contours,hierarchy=cv2.findContours(img_bin,mode=cv2.RETR_TREE,
    
    method=cv2.CHAIN_APPROX_SIMPLE) 
    
- **4.原始图像copy，否则会在原图上绘制**
    
    to_write=img.copy() 
    
- **5.红笔绘制图像轮廓**
    
    ret=cv2.drawContours(to_write,contours,-1,(0,0,255),2) 
    

## 直方图

- **matplotlib 绘制直方图**
    
    plt.hist(img.ravel(),256)
    
- **用opencv计算直方图列表**
    
    hist=cv2.calcHist(images= [img],channels=[0],mask=None,histSize=[256],ranges=[0,255])
    
- **掩膜提取局部直方图**
    
    pad=np.zeros(img.shape,np.uint8)
    
    pad[200:400,200:400]=255
    
    hist_MASK=cv2.calcHist(images= [img],channels=[0],mask=pad,histSize=[256],ranges=[0,255])
    
- **opencv 交、并、补、异或操作**
    
    masked_img=cv2.bitwise_and(img,mask)
    
- **直方图均衡化原理**
    
    图像直方图->直方图归一化->累计直方图->*255 x坐标映射->对原来的像素值进行新像素值编码
    
- **直方图均衡化调用**
    
    img1=cv2.equalizeHist(img)
    
- **matplotlib绘制图片前通道转换**
    
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #通道不一致性
    
- **matplotlib多图绘制在一个面板上**
    
    plt.subplot('221'),plt.imshow(img,cmap=plt.cm.gray),plt.axis('off'),plt.title('original')
    
    plt.subplot('222'), plt.imshow(img1, cmap=plt.cm.gray), plt.axis('off')
    
    plt.subplot('223'), plt.hist(img.ravel(),256)
    
    plt.subplot('224'), plt.hist(img1.ravel(), 256)
    

## 图像傅里叶变换（空间域=>频域）

- **图像傅里叶变换 （转化为虚数，实部为幅度，虚部为频率）**
    
    fft=np.fft.fft2(img)
    
    fft_center=np.fft.fftshift(fft)
    
    fft_flect=20*np.log(np.abs(fft_center))
    
- **图像傅里叶逆变换**
    
    fft_left=np.fft.ifftshift(fft_center)
    
    ifft=np.fft.ifft2(fft_left)
    
    img_f=np.abs(ifft)
    
- **高通滤波**
    
    h_c,w_c=round(h/2),round(w/2)
    
    fft_center[h_c-10:h_c+10,w_c-10:w_c+10]=0  #原图操作，低频信号归0
    
- **opencv 傅里叶变换**
    
    dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
    
    fft_center=np.fft.fftshift(dft)
    
- **opencv 低通滤波**
    
    mask=np.zeros((h,w,2),dtype=np.uint8)    #定义掩膜
    
    h_c,w_c=round(h/2),round(w/2)
    
    R=20
    
    mask[h_c-R:h_c+R,w_c-R:w_c+R]=1
    
    dshift=fft_center*mask       #点乘，保留低频信号
    
- **opencv 傅里叶反变换**
    
    fft_left=np.fft.ifftshift(dshift)
    
    ifft=cv2.idft(fft_left)
    
    img_f=cv2.magnitude(ifft[:,:,0],ifft[:,:,1]) #Square(x*2+y*2）