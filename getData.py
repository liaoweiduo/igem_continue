import multiprocessing
from multiprocessing.sharedctypes import Value, Array
from pylab import *
from PIL import Image
import queue
import copy
import matplotlib.pyplot as plt
import pickle
import random
import pandas as pd
from setting import *
from Cell import Cell
import cv2
from matplotlib.colors import ListedColormap
import pylab as pl

def labelExpend(cell, i, j):
    flag = cell.cellNo
    fifo = queue.Queue()
    fifo.put((i, j))

    while fifo.qsize() != 0:
        point = fifo.get()
        if label[point] == flag or label[point] == 0:
            continue
        elif label[point] != 255 and cell.cellSize < minCellSize / 2: # 不单独成为细胞的数据和已存在细胞相撞
            cell.cellSize = 0
            return

        label[point] = flag
        cell.addPoint(point)
        if point[1] - 1 >= 0 and label[point[0], point[1] - 1] > 0:  # left
            fifo.put((point[0], point[1] - 1))
        if point[1] + 1 < lenX and label[point[0], point[1] + 1] > 0:  # right
            fifo.put((point[0], point[1] + 1))
        if point[0] + 1 < lenX and label[point[0] + 1, point[1]] > 0:  # down
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
    return result / cell.cellSize

def processPhoto(photoIndex):
    # print('处理第', photoIndex+1, '张照片...') # if photoIndex % 10 == 0 else None

    im = cv2.imread(generalPath % (photoIndex + 1), flags=cv2.IMREAD_ANYDEPTH)

    # get bgValue
    im_t = copy.copy(im)
    im_reshape = im_t.reshape(1, lenT)
    im_reshape.sort()
    bgValue = im_reshape[0, int(lenT * 0.7)]
    # divide background
    im = np.where(im > bgValue, im - bgValue, 0)

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
im = cv2.imread(generalPath % (int(photosNum / 2)), flags=cv2.IMREAD_ANYDEPTH)

lenY = int(im.shape[0])
lenX = int(im.shape[1])
lenT = lenX * lenY

im_t = copy.copy(im)
im_reshape = im_t.reshape(1, -1)
im_reshape.sort()

''' bgValue'''
bgValue = im_reshape[0, int(lenT * 0.7)]
'''
print('bgValue(get 70%) = ', bgValue)
plt.imshow(blur)
bgPoint = plt.ginput(2)
bgValue_2 = int(im[int(bgPoint[0][1]): int(bgPoint[1][1]), int(bgPoint[0][0]): int(bgPoint[1][0])].mean())
print('bgValue(chosen area) = ', bgValue_2)
'''
# divide background
im = np.where(im > bgValue, im - bgValue, 0)

maxAve = im_reshape[0, -1] - bgValue

img = Image.fromarray(im * (255.0 / maxAve))
img = img.convert('L')

blur = cv2.GaussianBlur(np.array(img), (r, r), s)

# canny and adapted threshold —— 梯度直方图法
x = cv2.Sobel(blur, cv2.CV_16S, 1, 0) # Sobel 算子
y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)  # 转回uint8
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

meanForThreshold = dst.mean()
rmsForThreshold = dst.std()

canny = cv2.Canny(blur, meanForThreshold + 5 * rmsForThreshold, meanForThreshold + 10 * rmsForThreshold, apertureSize = 3)

# erode and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
dilated = canny
for _ in range(1,10):
    dilated = cv2.dilate(dilated,kernel)
eroded = dilated
for _ in range(1,10):
    eroded = cv2.erode(eroded,kernel)
label = eroded.astype(int)# 0:background  1:cellNo ...

# plot picture without background
fig = plt.figure(dataSave, figsize = (8,6), dpi = 150)
g1 = fig.add_subplot(221)
g1.set_title('blur')
g1.imshow(blur)

# cell expend to fill cell list
for i in range(0, lenY):
    for j in range(0, lenX):
        if label[i, j] == 255:
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

'''刷新label数组，用于显示选择的细胞'''
label = np.zeros((lenX,lenY), dtype = int)
for cell in cellList:
    for point in cell.pointSet:
        label[point] = cell.cellNo

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

xTicks = np.arange(int(floor(maxF)) + 1)
columnY = np.zeros(int(floor(maxF)) + 1, dtype = int)
for cell in cellList:
    columnY[int(floor(cell.maxFluo))] += 1
    if cellMap.get(int(floor(cell.maxFluo))) == None:
        cellMap[int(floor(cell.maxFluo))] = [cell]
    else:
        cellMap[int(floor(cell.maxFluo))].append(cell)

# 定义柱状图每个柱的宽度
bar_width = 1

# 画柱状图，定义柱的宽度，同时设置柱的边缘为透明
bars = g3.bar(xTicks, columnY, width=bar_width, edgecolor='none')

# 设置y轴的标题
g3.set_ylabel('Count')

# x轴每个标签的具体位置，设置为每个柱的中央
g3.set_xticks(xTicks + bar_width / 2)

# 设置x轴的范围
g3.set_xlim([0 - bar_width / 2, floor(maxF) + 3 * bar_width / 2])

#画reFluo时间图
g4 = fig.add_subplot(224)
g4.set_title('Time Chart ' + stress)
xTime = np.arange(photosNum) * timeDelay

'''colorMap'''
colorMap = [(0.0,0.0,0.0)]
colorMap.extend(pl.cm.jet(np.linspace(0, 1, Cell.cellsCount)))   # color range: 0 -> 1  还没找到对应真实颜色值
selfMap = ListedColormap(colorMap)


# colorIndex = 1
for x in xTime:
    if cellMap.get(x) == None:
        continue
    selectedCellIndex = random.randint(0,len(cellMap[x])-1)
    selectedCell = cellMap[x][selectedCellIndex]
    print('select cell No: ', selectedCell.cellNo)
    g4.plot(xTime, selectedCell.reFluorescenceTable, color = colorMap[selectedCell.cellNo], linewidth = 1, linestyle = "-")
    g2.imshow(label, cmap = selfMap)
    # pointS = pd.DataFrame(selectedCell.pointSet)
    # g2.plot(list(pointS.loc[:, 1]),list(pointS.loc[:, 0]), c = colorMap[colorIndex])
    # colorIndex += 1


plt.savefig(dataSave + '.png', dpi = 200)
plt.show()

