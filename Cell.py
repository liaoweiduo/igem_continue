import numpy as np
from setting import photosNum
class Cell:
    cellsCount = 0

    def __init__(self, point):
        self.cellSize = 0  # count pixel of cell
        Cell.cellsCount += 1
        self.cellNo = Cell.cellsCount
        self.pointSet = []
        self.reFluorescenceTable = np.zeros(photosNum)
        self.maxFluo = 0.0
        self.maxFluoIndex = 0

    def __del__(self):
        Cell.cellsCount -= 1

    def addPoint(self, point):
        self.pointSet.append(point)
        self.cellSize += 1
