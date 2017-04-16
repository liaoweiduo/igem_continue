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

dataSave = "data/20170409geco+piezo1ul"
stress = '0.0512 Pa'

photosNum = 300  # the num of photos
delay = 0.5

mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0


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

cellList = []

with open(dataSave + '.pkl', 'rb') as f:
    cellList = pickle.load(f)


fig = plt.figure(dataSave)
g1 = fig.add_subplot(121)
g2 = fig.add_subplot(122)

cellMap = {}

g1.set_title('Bar Chart')
maxF = 0
minF = 100000
for cell in cellList:
    if maxF < cell.maxFluo:
        maxF = cell.maxFluo
    elif minF > cell.maxFluo:
        minF = cell.maxFluo

xTicks = np.arange(round(maxF) + 1)
columnY = np.zeros(round(maxF) + 1, dtype = int)
for cell in cellList:
    columnY[round(cell.maxFluo)] += 1
    if cellMap.get(round(cell.maxFluo)) == None:
        cellMap[round(cell.maxFluo)] = [cell]
    else:
        cellMap[round(cell.maxFluo)].append(cell)

bar_width = 1
bars = g1.bar(xTicks, columnY, width=bar_width, edgecolor='none')
g1.set_ylabel('Count')
g1.set_xticks(xTicks + bar_width / 2)
g1.set_xlim([0 - bar_width / 2, round(maxF) + 3 * bar_width / 2])

#画reFluo时间图
g2.set_title('Time Chart ' + stress)
xTime = np.arange(photosNum) * delay
colorMap = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
colorIndex = 0
for x in xTime:
    if cellMap.get(x) == None:
        continue
    selectedCellIndex = random.randint(0,len(cellMap[x])-1)
    selectedCell = cellMap[x][selectedCellIndex]
    print('select cell No: ', selectedCell.cellNo)
    g2.plot(xTime, selectedCell.reFluorescenceTable, colorMap[colorIndex])
    colorIndex += 1 if colorIndex < 6 else 0

plt.savefig(dataSave + '.png')
plt.show()
