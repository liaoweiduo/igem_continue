import numpy as np
from setting import photosNum
class Cell:
    cellsCount = 0
    pointSet = []
    cellSize = 0
    cellNo = 0
    reFluorescenceTable = np.zeros(photosNum)
    maxFluo = 0.0
    maxFluoIndex = 0

    def __init__(self, point):
        self.pointSet.append(point)
        self.cellSize = 1  # count pixel of cell
        Cell.cellsCount += 1
        self.cellNo = Cell.cellsCount

    def addPoint(self, point):
        self.pointSet.append(point)
        self.cellSize += 1
