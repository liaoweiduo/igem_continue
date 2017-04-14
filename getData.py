import multiprocessing

from MyGaussianBlur import MyGaussianBlur
from pylab import *
from PIL import Image
import copy
import queue
import matplotlib.pyplot as plt

generalPath = "/Volumes/Seagate BUP/IGEM_new/20170318/5ul/piezo+5ult%03dc1.tif"
#generalPath = "piezot%03dc1.tif"

photosNum = 500 # the num of photos
columnNum = 9 # column graph 's column num

minCellSize = 50  # a cell is bigger than $ pixels
maxCellSize = 1024  # a cell is smaller than $ pixels
ThreadHoldRate = 0.15  # Cell threadhold rate

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
        self.maxFluo = 0  # 不算前1%
        self.maxFluoIndex = 0

img = Image.open(generalPath % (1), 'r')
im = np.array(img)

# print('Im: \n', im)

lenX = int(im.shape[0])
lenY = int(im.shape[1])
lenT = lenX * lenY

plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(im)  # 显示图片
plt.title("Choose Background")

# GaussianBlur
GBlur = MyGaussianBlur(radius = r, sigema = s)  # 声明高斯模糊类
temp = GBlur.template()  # 得到滤波模版
image = GBlur.filter(img, temp)  # 高斯模糊滤波，得到新的图片
im=np.array(image)

bgPoint = ginput(2)
bgValue = int(im[int(bgPoint[0][1]) : int(bgPoint[1][1]), int(bgPoint[0][0]) : int(bgPoint[1][0])].mean())

for i in range(0, lenX):
    for j in range(0, lenY):
        if im[i][j] < bgValue:
            im[i][j] = 0
        else:
            im[i][j] -= bgValue

plt.imshow(im)

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
maxAve = sortedIm[int(lenT)-1]
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
        elif label[point] != -1 and label[point] != flag:     # 和已存在细胞相撞 判断是否删掉2个细胞
            cell2No = label[point]
            # print('join occurred, the cell is: ',cell2No)
            if cell.cellSize > minCellSize / 2:
                removeCellByNo(cellList, cell2No)
                # print('2 cell join, remove: ', cell2No, ' for the reason that cellSize = ', cell.cellSize)
                cell.cellSize = 0                      # 跳出程序后删
            return


def removeCellByNo(cellList, cellNo):
    for cell in cellList:
        if cell.cellNo == cellNo:
            cellList.remove(cell)
            del(cell)
            return True
    return False

def fluoExpend(cell, im, bgValue):
    result = 0
    visit = np.zeros([lenX, lenY], dtype = bool)
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
    return int(result / cell.cellSize)

cellList = []

plt.subplot(2,2,2)

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
                    del(cell)

'''待加入check cellList，删除圆形细胞及非细胞的代码'''
print('finish init')
plt.imshow(label)
# plt.show()
#每张图片
'''没有考虑到细胞的帧偏移'''
def processPhoto(photoIndex):
    print('处理第', photoIndex+1, '张照片...') #if photoIndex % 10 == 0 else None

    img = Image.open(generalPath % (photoIndex + 1), 'r')
    im = np.array(img)

    bgValue = int(im[int(bgPoint[0][1]): int(bgPoint[1][1]), int(bgPoint[0][0]): int(bgPoint[1][0])].mean())

    '''只给算的区域减背景'''
    #每个细胞
    for cell in cellList:
        fluo = fluoExpend(cell, im, bgValue)
        cell.reFluorescenceTable[photoIndex] = fluo
        cell.maxFluo = fluo if fluo > cell.maxFluo else cell.maxFluo
        cell.maxFluoIndex = photoIndex if fluo > cell.maxFluo else cell.maxFluoIndex

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
print('start multiprocessing, coresNum: ',cores)
photosIndex = range(0, photosNum)
cnt = 0
for _ in pool.imap_unordered(processPhoto, photosIndex):
    sys.stdout.write('done %d/%d\r' % (cnt, len(photosIndex)))
    cnt += 1

# 求细胞related荧光
print('test cell reFluo')

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
'''
# 画图
plt.subplot(2,2,3)

maxF = 0
minF = 100000
for cell in cellList:
    if maxF < cell.maxFluo:
        maxF = cell.maxFluo
    elif minF > cell.maxFluo:
        minF = cell.maxFluo

(ceil(maxF) - floor(minF)) / (columnNum + 1)
x = np.linspace(floor(minF), ceil(maxF), columnNum + 2)

print('x轴: ',x)


'''