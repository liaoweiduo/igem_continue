# import numpy as np
# from pylab import *
# from PIL import Image
# import matplotlib.pyplot as plt
# from MyGaussianBlur import MyGaussianBlur
# import queue
# # 一些点
# x = [100, 100, 400, 400]
# y = [200, 500, 200, 500]
#
# # 使用红色星状标记绘制点
#
# # plot(x,y,'r*')
#
# # 绘制连接前两个点的线
#
# # plot(x[:2],y[:2])
#
# # 添加标题,显示绘制的图像
#
# # title('Plotting: "Test.jpg"')
#
# # show()
#
# # generalPath = "/Volumes/Seagate BUP/IGEM_new/20170318/5ul/piezo+5ult001c1.tif"
# img = Image.open("piezot001c1.tif")
# im = np.array(img)
# plt.subplot(1,2,1)
# plt.imshow(im)
#
# r = 4  # 模版半径，自己自由调整
# s = 2  # sigema数值，自己自由调整
# #GBlur = MyGaussianBlur(radius=r, sigema=s)  # 声明高斯模糊类
# #temp = GBlur.template()  # 得到滤波模版
# #image = GBlur.filter(img, temp)  # 高斯模糊滤波，得到新的图片
# #plt.imshow(np.array(image))  # 图片显示
#
# #print(im.shape, im.dtype)
#
# # 输出坐标100,100的rgb值
# print(im[1, 2])
# print(im[1][2])
# print(im[:,:])
#
#
# print(len(im))
#
# print(im)
#
# # multiply each pixel by 1.2
#
# # im.show()
#
#
#
#
# a = 1
# b = a
#
# b += 1
#
# print(a)
# print(b)
#
#
# class aa:
#     def __init__(self, x):
#         self.x = x
#
#
# a = aa(2)
# b = a
#
# b.x = 5
#
# print(a.x)
# print(b.x)
#
# fifo = queue.Queue()
# print(fifo.qsize())
# fifo.put((1, 2))
# print(fifo.qsize())
# fifo.put((3, 2))
# print(fifo.qsize())
#
# print(fifo.get()[1])
#
# print(fifo.qsize())
#
# a = zeros([5, 5], int8)
#
# print(a)
#
# plt.imshow(im)  # 显示图片
# # plt.axis('off')
# # x=ginput(3)
#
# # print(x[1][1])
#
# print(im[0:2,0:3])
# print(im[0:2,0:3].mean())
#
#
# po=(2,1)
# a[po]=3
# a[1,2]=4
# a[1][2]=5
# print (a)
#
# print (po+(2,1))
#
# aa = []
#
# plt.cla()
# plt.figure(1)
# plt.plot((1,2,3),(4,5,6),'b')
# plt.show

'''
from multiprocessing import Process, Value, Array
import time

import multiprocessing
import numpy as np
from pylab import *
from PIL import Image
import matplotlib.pyplot as plt
from MyGaussianBlur import MyGaussianBlur
import queue
# 一些点
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

# 使用红色星状标记绘制点

# plot(x,y,'r*')

# 绘制连接前两个点的线

# plot(x[:2],y[:2])
# bgPoint = ginput(2)

# 添加标题,显示绘制的图像

#title('Plotting: "Test.jpg"')


# show()

xxx=np.linspace(0,5,6)
print(xxx)
'''

'''
def f(x):
    time.sleep(2)
    return x*x
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(cores)


def f(x):
    return x * x

if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    # print(pool.map(f,arr))

    print (num.value)
    print (arr[2])

    print(pool.map(f, [1, 2, 3]))
cnt = 0
for _ in pool.imap_unordered(f, [1, 2, 3, 4, 5, 6, 7, 8]):
    print('\rdone %d/%d' % (cnt, 3))
    cnt += 1

print ("你好吗？\n朋友")
print ("——分隔线——")
print ("你好吗？\r朋友")
'''
'''
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0

# 包含了狗，猫和猎豹的最高奔跑速度，还有对应的可视化颜色
speed_map = {
    'dog': (48, '#7199cf'),
    'cheetah': (120, '#e1a7a2'),
    'cat': (45, '#4fc4aa')
}

# 整体图的标题
fig = plt.figure('Bar chart & Pie chart')

# 在整张图上加入一个子图，121的意思是在一个1行2列的子图中的第一张
ax = fig.add_subplot(221)
ax.set_title('Running speed - bar chart')

# 生成x轴每个元素的位置
xticks = np.linspace(0, 2, 3)
xx = np.arange(3)
print(xticks)
print(xx * 0.5)
# 定义柱状图每个柱的宽度
bar_width = 0.5

# 动物名称
animals = speed_map.keys()
print(animals)
# 奔跑速度
speeds = [x[0] for x in speed_map.values()]
print(speeds)

# 对应颜色
colors = [x[1] for x in speed_map.values()]
print(colors)

# 画柱状图，横轴是动物标签的位置，纵轴是速度，定义柱的宽度，同时设置柱的边缘为透明
bars = ax.bar(xticks, speeds, width=bar_width, edgecolor='none')

# 设置y轴的标题
ax.set_ylabel('Speed(km/h)')

# x轴每个标签的具体位置，设置为每个柱的中央
ax.set_xticks(xticks + bar_width / 2)

# 设置每个标签的名字
ax.set_xticklabels(animals)

# 设置x轴的范围
ax.set_xlim([bar_width / 2 - 0.5, 3 - bar_width / 2])

# 设置y轴的范围
ax.set_ylim([0, 125])

# 给每个bar分配指定的颜色
for bar, color in zip(bars, colors):
    bar.set_color(color)
# xx = plt.ginput(2)
# 在122位置加入新的图
ff = plt.figure('sddffsdfdsf')
bx = ff.add_subplot(222)
bx.set_title('Running speed - pie chart')

# 生成同时包含名称和速度的标签
labels = ['{}\n{} km/h'.format(animal, speed) for animal, speed in zip(animals, speeds)]

# 画饼状图，并指定标签和对应颜色
bx.pie(speeds, labels=labels, colors=colors)
# bx.imshow([1,2,3,4,5,6])
# plt.show()

print(np.arange(3).dtype)
print(np.linspace(0, 2, 3))

xTime = np.arange(5) * 0.5
print(xTime)

print(random.randint(1, 5))

ssss = 'sdfdsfsd'
qqqq = ssss + '.ll'
print(qqqq)

bgValue = 5
im = np.array([6, 5, 3, 2, 7, 8, 4, 3])
print(im)
im = np.where(im > bgValue, im - bgValue, 0)

print(im)
from Cell import Cell

cell = Cell((1, 2))
print(cell.cellSize)
'''














import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import cv2
from PIL import Image

# im = cv2.imread('/Volumes/Seagate BUP/IGEM_new/20170409/GECO/geco1ul/geco1ult001c1.tif', flags=cv2.IMREAD_ANYDEPTH)
im = cv2.imread('piezot001c1.tif', flags=cv2.IMREAD_ANYDEPTH)
#im = cv2.imread('1_u012599883.jpg', flags=cv2.IMREAD_ANYDEPTH)





im_t = copy.copy(im)
im_reshape = im_t.reshape(1, -1)
im_reshape.sort()
maxAve = im_reshape[0, -1]

img = Image.fromarray(im * (255.0 / maxAve))
img = img.convert('L')

blur = cv2.GaussianBlur(np.array(img), (3, 3), 0)
# blur = cv2.blur(np.array(img), (3, 3))

hist = cv2.calcHist([blur],[0],None,[256],[0,256]) #获得直方图
mean = hist.mean()
RMS = hist.var()

canny = cv2.Canny(blur, mean + RMS, mean + 3 * RMS, apertureSize = 3)    # two threshold should be set

# erode and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
dilated = canny
for i in range(1,10):
    dilated = cv2.dilate(dilated,kernel)
eroded = dilated
for i in range(1,10):
    eroded = cv2.erode(eroded,kernel)





fig = plt.figure(figsize=(8, 6), dpi=80)
g1 = fig.add_subplot(221)
g1.imshow(blur)
g2 = fig.add_subplot(222)
g2.imshow(canny)
g3 = fig.add_subplot(223)
g3.imshow(eroded)
g4 = fig.add_subplot(224)
g4.imshow(eroded)


plt.show()
