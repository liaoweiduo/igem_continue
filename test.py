import numpy as np
from pylab import *
from PIL import Image
im = array(Image.open("data/20170318_1ul.jpg","r"))


#输出数组的各维度长度以及类型
print (im.shape,im.dtype)

#输出位于坐标100,100，颜色通道为r的像素值
print (im[100,100,0])

#输出坐标100,100的rgb值
print (im[100,100])

print(im.shape[1])

#im.show()

# 一些点
x = [100,100,400,400]
y = [200,500,200,500]

# 使用红色星状标记绘制点

#plot(x,y,'r*')

# 绘制连接前两个点的线

#plot(x[:2],y[:2])

# 添加标题,显示绘制的图像

#title('Plotting: "Test.jpg"')

#show()

