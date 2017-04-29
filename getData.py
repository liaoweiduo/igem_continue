import multiprocessing
from multiprocessing.sharedctypes import Value, Array
from MyGaussianBlur import MyGaussianBlur
from pylab import *
from PIL import Image
import queue
import copy
import matplotlib.pyplot as plt
import pickle
import random
from setting import *
from Cell import Cell

def labelExpend(cell, i, j):
    flag = cell.cellNo
    fifo = queue.Queue()
    fifo.put((i, j))

    while fifo.qsize() != 0:
        point = fifo.get()
        if label[point] >= 0:
            if label[point] > 0 and label[point] != flag and cell.cellSize < minCellSize / 2: # 不单独成为细胞的数据和已存在细胞相撞
                cell.cellSize = 0
                return
            elif label[point] == flag:
                continue
            if im[point] > threshold:
                label[point] = flag
            else:
                label[point] = -1
                continue
            cell.addPoint(point)
            if point[1] - 1 >= 0 and label[point[0], point[1] - 1] >= 0:  # left
                fifo.put((point[0], point[1] - 1))
            if point[1] + 1 < lenX and label[point[0], point[1] + 1] >= 0:  # right
                fifo.put((point[0], point[1] + 1))
            if point[0] + 1 < lenX and label[point[0] + 1, point[1]] >= 0:  # down
                fifo.put((point[0] + 1, point[1]))


def removeCellByNo(cellList, cellNo):
    for cell in cellList:
        if cell.cellNo == cellNo:
            cellList.remove(cell)
            del (cell)
            return True
    return False
'''改'''
def fluoExpend(cell, im, bgValue):
    result = 0
    for point in cell.pointSet:
        result += im[point]
    result -= bgValue
    return result / cell.cellSize

def processPhoto(photoIndex):
    # print('处理第', photoIndex+1, '张照片...') # if photoIndex % 10 == 0 else None

    img = Image.open(generalPath % (photoIndex + 1), 'r')
    im = np.array(img)

    # get bgValue
    im_t = copy.copy(im)
    im_reshape = im_t.reshape(1, lenT)
    im_reshape.sort()
    bgValue = im_reshape[0, int(lenT / 3) : int(lenT / 2)].mean()

    # 每个细胞
    for cell in cellList:
        fluo = fluoExpend(cell, im, bgValue)
        cellsForParallel[cell]['reFluorescenceTable'][photoIndex] = fluo
        cellsForParallel[cell]['maxFluo'].value = fluo if fluo > cellsForParallel[cell]['maxFluo'].value else cellsForParallel[cell]['maxFluo'].value
        cellsForParallel[cell]['maxFluoIndex'].value = photoIndex if fluo > cellsForParallel[cell]['maxFluo'].value else cellsForParallel[cell]['maxFluoIndex'].value
    return photoIndex

cellList = []
cellsForParallel = {}

# get cell list
# open img
img = Image.open(generalPath % (int(photosNum / 2)), 'r')
im = np.array(img)

lenX = int(im.shape[0])
lenY = int(im.shape[1])
lenT = lenX * lenY

label = np.zeros([lenX, lenY], int16) # 0:unVisit  1:cellNo ...   -1:background

# GaussianBlur
GBlur = MyGaussianBlur(radius=r, sigema=s)  # 声明高斯模糊类
temp = GBlur.template()  # 得到滤波模版
image = GBlur.filter(img, temp)  # 高斯模糊滤波，得到新的图片
im = np.array(image)

# get bgValue
im_t = copy.copy(im)
im_reshape = im_t.reshape(1, lenT)
im_reshape.sort()
bgValue = im_reshape[0, int(lenT / 3):int(lenT / 2)].mean()

# divide background
im = np.where(im > bgValue, im - bgValue, 0)

# plot picture without background
fig = plt.figure(dataSave)
g1 = fig.add_subplot(221)
g1.set_title('Without Background')
g1.imshow(im)

# get threshold
minAve = im_reshape[0, 0]
maxAve = im_reshape[0, -1]
threshold = minAve * (1 - thresholdRate) + maxAve * thresholdRate

# cell expend to fill cell list
for i in range(0, lenY):
    for j in range(0, lenX):
        if label[i, j] == 0:
            if im[i, j] < threshold:
                label[i, j] = - 1
            else:
                cell = Cell((i, j))
                cellList.append(cell)
                labelExpend(cell, i, j)
                if cell.cellSize < minCellSize or cell.cellSize > maxCellSize:
                    cellList.remove(cell)
                    del (cell)

'''待加入check cellList，删除圆形细胞及非细胞的代码'''

# init multiprocess
for cell in cellList:
    cellsForParallel[cell] = {
        'reFluorescenceTable' : Array('f', np.zeros(photosNum), lock = False),
        'maxFluo' : Value('f', 0.0, lock = True),
        'maxFluoIndex' : Value('i', 0, lock = True)}

print('finish init')
g2 = fig.add_subplot(222)
g2.set_title('Chosen Cells')
g2.imshow(label)

# data processing per picture
'''没有考虑到细胞的帧偏移'''
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
print('start multiprocessing, coresNum: ', cores)
photosIndex = range(0, photosNum)
cnt = 0

# for each picture
for _ in pool.imap_unordered(processPhoto, photosIndex):
    cnt += 1
    print ('处理了 ', cnt, ' 张照片...') if cnt % 10 == 0 else None

#data in share memory transform to cell
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
xTime = np.arange(photosNum) * timeDelay
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

