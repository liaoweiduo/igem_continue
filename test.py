import numpy as np
from pylab import *
from PIL import Image
import matplotlib.pyplot as plt
from MyGaussianBlur import MyGaussianBlur
import queue

# generalPath = "/Volumes/Seagate BUP/IGEM_new/20170318/5ul/piezo+5ult001c1.tif"
img = Image.open("piezot001c1.tif")
im = np.array(img)
plt.subplot(1,2,1)
plt.imshow(im)

r = 4  # 模版半径，自己自由调整
s = 2  # sigema数值，自己自由调整
#GBlur = MyGaussianBlur(radius=r, sigema=s)  # 声明高斯模糊类
#temp = GBlur.template()  # 得到滤波模版
#image = GBlur.filter(img, temp)  # 高斯模糊滤波，得到新的图片
#plt.imshow(np.array(image))  # 图片显示

#print(im.shape, im.dtype)

# 输出坐标100,100的rgb值
print(im[1, 2])
print(im[1][2])
print(im[:,:])


print(len(im))

print(im)

# multiply each pixel by 1.2

# im.show()

# 一些点
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

# 使用红色星状标记绘制点

# plot(x,y,'r*')

# 绘制连接前两个点的线

# plot(x[:2],y[:2])

# 添加标题,显示绘制的图像

# title('Plotting: "Test.jpg"')

# show()


a = 1
b = a

b += 1

print(a)
print(b)


class aa:
    def __init__(self, x):
        self.x = x


a = aa(2)
b = a

b.x = 5

print(a.x)
print(b.x)

fifo = queue.Queue()
print(fifo.qsize())
fifo.put((1, 2))
print(fifo.qsize())
fifo.put((3, 2))
print(fifo.qsize())

print(fifo.get()[1])

print(fifo.qsize())

a = zeros([5, 5], int8)

print(a)

plt.imshow(im)  # 显示图片
# plt.axis('off')
# x=ginput(3)

# print(x[1][1])

print(im[0:2,0:3])
print(im[0:2,0:3].mean())


po=(2,1)
a[po]=3
a[1,2]=4
a[1][2]=5
print (a)

print (po+(2,1))