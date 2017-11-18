
import numpy as np
import cv2

#d = np.loadtxt('depth.txt', delimiter=' ')
#print(d.shape)

#with open('depth.txt', 'r') as f:
#    print(f.readline().split())

raster = cv2.imread("result.tif", cv2.IMREAD_UNCHANGED)
np.savetxt("1111.csv", raster, fmt='%d', delimiter=',')
print(raster.shape)