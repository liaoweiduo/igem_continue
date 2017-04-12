import numpy as np
from pylab import *
from PIL import Image
import copy

class Cell:
    cellsCount = 0
    def __init__(self, topX, topY):
        self.topX=topX
        self.topY=topY
        self.centralX=topX
        self.centralY=topY
        self.downX=topX
        self.downY=topY
        self.leftX=topX
        self.leftY=topY
        self.rightX=topX
        self.rightY=topY
        Cell.cellsCount+=1

generalPath="/Volumes/Seagate BUP/IGEM_new/20170318/5ul/piezo+5ult%03dc1.tif"
im = np.array(Image.open(generalPath % (1),"r"))

sign = zeros(im.shape[0], im.shape[1])
# get threadHold  = average(max,min)

lenX=im.shape[0]
lenY=im.shape[1]
lenT=lenX*lenY

sortedIm=copy.copy(im)
sortedIm.sort()

print('totalLen: ',lenT)

min=average(sortedIm[0:lenT*0.1])
max=average(sortedIm[lenT*0.9:lenT])
threadHold=(min+max)/2

print('min: ',min,' max: ',max,' threadHold: ',threadHold)

# sign     0:unVisit  1:cell1 ...   -1:background

def signExpend(x, y):
    Flag = Cell.cellsCount
    if sign[x,y] == -1:
        return
    elif sign[x,y] >0 and sign[x,y]!=Flag:    #与其他细胞相撞

    if im[i,j]<threadHold:
        sign[i,j]=-1
    else:


for i in range(0,lenX):
    for j in range(0,lenY):
        if sign[i,j]==0:
            if im[i,j]<threadHold:
                sign[i,j]=-1
            else:
                newCell=Cell(i,j)
                signExpend(i,j)
