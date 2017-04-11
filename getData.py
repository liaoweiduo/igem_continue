import numpy as np
from pylab import *
from PIL import Image
import copy

generalPath="/Volumes/Seagate BUP/IGEM_new/20170318/5ul/piezo+5ult%03dc1.tif"
im = np.array(Image.open(generalPath % (1),"r"))

#get threadHold  = average(max,min)

sortedIm=copy.copy(im)
sortedIm.sort()

min=average(sortedIm[0:100])
max=average(sortedIm[])

print(sortedIm[100000:])
threadHold=average(im)
print(threadHold)


