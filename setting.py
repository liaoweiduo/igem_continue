import matplotlib as mpl

generalPath = "/Volumes/Seagate BUP/IGEM_new/20170409/GECO/geco1ul/geco1ult%03dc1.tif"
dataSave = "data/20170409geco1ul"
stress = '0.0512 Pa'

photosNum = 300  # the num of photos
timeDelay = 0.5 # time delay of each frame

minCellSize = 50  # a cell is bigger than $ pixels
maxCellSize = 10000  # a cell is smaller than $ pixels
'''
thresholdRate = 0.15  # Cell threashold rate
'''
# mpl.rcParams['axes.titlesize'] = 20
# mpl.rcParams['xtick.labelsize'] = 16
# mpl.rcParams['ytick.labelsize'] = 16
# mpl.rcParams['axes.labelsize'] = 16
# mpl.rcParams['xtick.major.size'] = 0
# mpl.rcParams['ytick.major.size'] = 0

r = 3  # 高斯模糊模版半径，自己自由调整
s = 0  # 高斯模糊sigema数值，自己自由调整
'''
cannyThreshold1 = 30
cannyThreshold2 = 80
'''
