import multiprocessing
from multiprocessing.sharedctypes import Value, Array
from MyGaussianBlur import MyGaussianBlur
from pylab import *
from PIL import Image
import copy
import queue
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import random

generalPath = "/Volumes/Seagate BUP/IGEM_new/20170409/GECO+Piezo/geco+piezo1ul/geco+piezo1ult%03dc1.tif"
dataSave = "data/20170409geco+piezo1ul"
stress = '0.0512 Pa'

photosNum = 300  # the num of photos
delay = 0.5

minCellSize = 50  # a cell is bigger than $ pixels
maxCellSize = 10000  # a cell is smaller than $ pixels
ThreadHoldRate = 0.15  # Cell threadhold rate

mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0

r = 1  # 高斯模糊模版半径，自己自由调整
s = 2  # 高斯模糊sigema数值，自己自由调整

class Cell:
    cellsCount = 0

    def __init__(self, topX, topY):
        self.topPoint = (topX, topY)
        self.downPoint = (topX, topY)
        self.leftPoint = (topX, topY)
        self.rightPoint = (topX, topY)
        self.cellSize = 0  # count pixel of cell
        Cell.cellsCount += 1
        self.cellNo = Cell.cellsCount
        self.reFluorescenceTable = np.zeros(photosNum)
        self.maxFluo = 0.0
        self.maxFluoIndex = 0

img = Image.open(generalPath % (int(photosNum / 2)), 'r')
im = np.array(img)

# print('Im: \n', im)

lenX = int(im.shape[0])
lenY = int(im.shape[1])
lenT = lenX * lenY


figInput = plt.figure('Choose Background')
plt.imshow(im)  # 显示图片

# GaussianBlur
GBlur = MyGaussianBlur(radius=r, sigema=s)  # 声明高斯模糊类
temp = GBlur.template()  # 得到滤波模版
image = GBlur.filter(img, temp)  # 高斯模糊滤波，得到新的图片
im = np.array(image)

bgPoint = plt.ginput(2)
bgValue = int(im[int(bgPoint[0][1]): int(bgPoint[1][1]), int(bgPoint[0][0]): int(bgPoint[1][0])].mean())

fig = plt.figure(dataSave)
g1 = fig.add_subplot(221)
g1.set_title('Without Background')

for i in range(0, lenX):
    for j in range(0, lenY):
        if im[i][j] < bgValue:
            im[i][j] = 0
        else:
            im[i][j] -= bgValue

g1.imshow(im)

label = np.zeros([im.shape[0], im.shape[1]], int16)

# get threadHold  = average(max,min)

sortedIm = copy.copy(im)
sortedIm.shape = [1, -1]
sortedIm = sortedIm[0]
sortedIm.sort()

# print('totalLen: ', lenT)
# print('Im(-): \n', im)
# print('sortedIm: \n', sortedIm)
# print(sortedIm.shape)
minAve = sortedIm[0]
maxAve = sortedIm[int(lenT) - 1]
threadHold = minAve * (1 - ThreadHoldRate) + maxAve * ThreadHoldRate

# print('min: ', minAve, ' max: ', maxAve, ' threadHold: ', threadHold)

# label     0:unVisit  1:cellNo ...   -1:background
def labelExpend(cell, i, j):
    flag = cell.cellNo
    fifo = queue.Queue()
    fifo.put((i, j))

    while fifo.qsize() != 0:
        point = fifo.get()
        if label[point] == 0:
            if im[point] > threadHold:
                label[point] = flag
            else:
                label[point] = -1
                continue
            cell.cellSize += 1
            if label[point[0], point[1] - 1] >= 0:  # left
                fifo.put((point[0], point[1] - 1))
            if label[point[0], point[1] + 1] >= 0:  # right
                fifo.put((point[0], point[1] + 1))
            if label[point[0] + 1, point[1]] >= 0:  # down
                fifo.put((point[0] + 1, point[1]))
            # 更新此point到cell边界中
            if cell.leftPoint[1] > point[1]:
                cell.leftPoint = point
            if cell.rightPoint[1] < point[1]:
                cell.rightPoint = point
            if cell.downPoint[0] < point[0]:
                cell.downPoint = point
        elif label[point] != -1 and label[point] != flag:  # 和已存在细胞相撞 判断是否删掉2个细胞
            cell2No = label[point]
            # print('join occurred, the cell is: ',cell2No)
            if cell.cellSize > minCellSize / 2:
                removeCellByNo(cellList, cell2No)
                # print('2 cell join, remove: ', cell2No, ' for the reason that cellSize = ', cell.cellSize)
                cell.cellSize = 0  # 跳出程序后删
            return


def removeCellByNo(cellList, cellNo):
    for cell in cellList:
        if cell.cellNo == cellNo:
            cellList.remove(cell)
            del (cell)
            return True
    return False


def fluoExpend(cell, im, bgValue):
    result = 0
    visit = np.zeros([lenX, lenY], dtype=bool)
    fifo = queue.Queue()
    fifo.put(cell.topPoint)
    while fifo.qsize() != 0:
        point = fifo.get()
        if label[point] == cell.cellNo and visit[point] == False:
            visit[point] = True
            if im[point] > bgValue:
                result += (im[point] - bgValue)
            fifo.put((point[0], point[1] - 1))
            fifo.put((point[0], point[1] + 1))
            fifo.put((point[0] + 1, point[1]))
    return result / cell.cellSize

cellList = []
cellsForParallel = {}

for i in range(0, lenY):
    for j in range(0, lenX):
        if label[i, j] == 0:
            if im[i, j] < threadHold:
                label[i, j] = - 1
            else:
                cell = Cell(i, j)
                # print('start expend cell: ', cell.cellNo, ' start at position: ', i, '行, ', j, '列')
                cellList.append(cell)
                labelExpend(cell, i, j)  # 函数内给label（i，j）赋值
                # print('finally get pixel: ', cell.cellSize)
                if cell.cellSize < minCellSize or cell.cellSize > maxCellSize:
                    # print('remove from cellList')
                    cellList.remove(cell)
                    del (cell)

'''待加入check cellList，删除圆形细胞及非细胞的代码'''

for cell in cellList:
    cellsForParallel[cell] = {
        'reFluorescenceTable' : Array('f', np.zeros(photosNum), lock = False),
        'maxFluo' : Value('f', 0.0, lock = True),
        'maxFluoIndex' : Value('i', 0, lock = True)}

print('finish init')
g2 = fig.add_subplot(222)
g2.set_title('Chosen Cells')
g2.imshow(label)

# 每张图片
'''没有考虑到细胞的帧偏移'''

def processPhoto(photoIndex):
    # print('处理第', photoIndex+1, '张照片...') # if photoIndex % 10 == 0 else None

    img = Image.open(generalPath % (photoIndex + 1), 'r')
    im = np.array(img)

    bgValue = int(im[int(bgPoint[0][1]): int(bgPoint[1][1]), int(bgPoint[0][0]): int(bgPoint[1][0])].mean())

    '''只给算的区域减背景'''
    # 每个细胞
    for cell in cellList:
        fluo = fluoExpend(cell, im, bgValue)
        cellsForParallel[cell]['reFluorescenceTable'][photoIndex] = fluo
        cellsForParallel[cell]['maxFluo'].value = fluo if fluo > cellsForParallel[cell]['maxFluo'].value else cellsForParallel[cell]['maxFluo'].value
        cellsForParallel[cell]['maxFluoIndex'].value = photoIndex if fluo > cellsForParallel[cell]['maxFluo'].value else cellsForParallel[cell]['maxFluoIndex'].value
    return photoIndex


cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
print('start multiprocessing, coresNum: ', cores)
# for photoIndex in range(0, photosNum):
#     pool.apply_async(processPhoto, photoIndex)
photosIndex = range(0, photosNum)
cnt = 0
for _ in pool.imap_unordered(processPhoto, photosIndex):
    cnt += 1
    print ('处理了 ', cnt, ' 张照片...') if cnt % 10 == 0 else None
#把共享变量里的值赋给cell
for cell in cellList:
    cell.reFluorescenceTable = cellsForParallel[cell]['reFluorescenceTable'][:]
    cell.maxFluo = cellsForParallel[cell]['maxFluo'].value
    cell.maxFluoIndex = cellsForParallel[cell]['maxFluoIndex'].value

# 求细胞related荧光
print('get cell reFluo')

for cell in cellList:
    table = cell.reFluorescenceTable
    for index in range(1, table.__len__()):
        if table[0] != 0:
            table[index] = (table[index] - table[0]) / table[0]
        else:
            table[index] = 0
    table[0] = 0
    cell.maxFluo = table[cell.maxFluoIndex]
    print('cell No: ', cell.cellNo, ' reFluorescenceTable: ', cell.reFluorescenceTable)
print('data save')

with open(dataSave + '.pkl', 'wb') as f:                     # open file with write-mode
    pickle.dump(cellList, f)                   # serialize and save object

# 画图

cellMap = {}

g3 = fig.add_subplot(223)
g3.set_title('Bar Chart')
maxF = 0
minF = 100000
for cell in cellList:
    if maxF < cell.maxFluo:
        maxF = cell.maxFluo
    elif minF > cell.maxFluo:
        minF = cell.maxFluo
# delta = (round(maxF) - round(minF)) / (columnNum + 1)

xTicks = np.arange(round(maxF) + 1)
columnY = np.zeros(round(maxF) + 1, dtype = int)
for cell in cellList:
    columnY[round(cell.maxFluo)] += 1
    if cellMap.get(round(cell.maxFluo)) == None:
        cellMap[round(cell.maxFluo)] = [cell]
    else:
        cellMap[round(cell.maxFluo)].append(cell)

# 定义柱状图每个柱的宽度
bar_width = 1

# 画柱状图，定义柱的宽度，同时设置柱的边缘为透明
bars = g3.bar(xTicks, columnY, width=bar_width, edgecolor='none')

# 设置y轴的标题
g3.set_ylabel('Count')

# x轴每个标签的具体位置，设置为每个柱的中央
g3.set_xticks(xTicks + bar_width / 2)

# 设置x轴的范围
g3.set_xlim([0 - bar_width / 2, round(maxF) + 3 * bar_width / 2])

#画reFluo时间图
g4 = fig.add_subplot(224)
g4.set_title('Time Chart ' + stress)
xTime = np.arange(photosNum) * delay
colorMap = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
colorIndex = 0
for x in xTime:
    if cellMap.get(x) == None:
        continue
    selectedCellIndex = random.randint(0,len(cellMap[x])-1)
    selectedCell = cellMap[x][selectedCellIndex]
    print('select cell No: ', selectedCell.cellNo)
    g4.plot(xTime, selectedCell.reFluorescenceTable, colorMap[colorIndex])
    colorIndex += 1 if colorIndex < 7 else 0

plt.savefig(dataSave + '.png')
plt.show()
