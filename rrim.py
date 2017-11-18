#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time


# compute zenith or nadir angles
def theta(dh):
    if len(dh):
        x = dh[:,1]/dh[:,0]
        return 90-np.arctan(np.max(x))*180/np.pi
    return 0

# compute openness
def openness(depth, r, c, cellSize, L):
    dc = [1,1,0,-1,-1,-1,0,1]
    dr = [0,1,1,1,0,-1,-1,-1]
    po = []
    no = []
    for dCount in range(8):
        pCount = 1
        pe = []
        ne = []
        while pCount:
            pr = r+dr[dCount]*pCount
            pc = c+dc[dCount]*pCount
            dis = np.sqrt((pr-r)**2+(pc-c)**2)*cellSize
            if dis > L or pr < 0 or pr >= depth.shape[0] or pc < 0 or pc >= depth.shape[1]:
                break

            pe.append([dis, depth[pr, pc]-depth[r, c]])
            ne.append([dis, depth[r, c] - depth[pr, pc]])
            pCount += 1

        po.append(theta(np.array(pe)))
        no.append(theta(np.array(ne)))

    return sum(po)/8, sum(no)/8

# compute slope
def slope(depth, cellSize):
    y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = cv2.filter2D(depth, -1, y_kernel) / (8 * cellSize)
    dx = cv2.filter2D(depth, -1, x_kernel) / (8 * cellSize)
    return np.arctan(np.sqrt(dy ** 2 + dx ** 2)) * 180 / np.pi

# color scheme
def make_RRIM_map():
    size = 90, 50, 3
    img_hsv = np.zeros(size, dtype=np.uint8)

    # saturation
    saturation_values = np.linspace(0, 255, size[0])
    for i in range(0, size[0]):
        img_hsv[i, :, 1] = np.ones(size[1], dtype=np.uint8) * np.uint8(saturation_values[i])

    # value
    V_values = np.linspace(0, 255, size[1])
    for i in range(0, size[1]):
        img_hsv[:, i, 2] = np.ones(size[0], dtype=np.uint8) * np.uint8(V_values[i])

    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

# generate rrim image
def makeRRIMImage(slopedata, openness):
    RRIM_map = make_RRIM_map()

    size = (slopedata.shape[0], slopedata.shape[1], 3)
    result = np.zeros(size, dtype=np.uint8)

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            inc = np.uint8(min(slopedata[y, x], 90-1))
            openness_val = np.uint8(openness[y, x]+25)
            if openness_val<0:
                openness_val=0
            elif openness_val>=50:
                openness_val=49
            result[y, x, :] = RRIM_map[inc, openness_val]
    return result

# decorator: compute time cost
def timer(func):
    def wrapper(*args, **kw):
        startTime = time.clock()
        callback =  func(*args, **kw)
        print('\nTotal running time: %.3f' % ((time.clock() - startTime) / 60.0), 'mins')
        return callback
    return wrapper

# rrim function
@timer
def rrim(depth, cellSize, L, output_fname):
    print('start rrim...')

    # 1. slop step
    slopeMat = slope(depth, cellSize)

    # 2. openness step
    opennessMat = np.zeros(depth.shape)
    for j in range(depth.shape[0]):
        if j % 100 == 0:
            print('  %.2f  finished...' % (j/depth.shape[0]*100))
        for i in range(depth.shape[1]):
            o = openness(depth, j, i, cellSize, L)
            opennessMat[j,i] = (o[0]-o[1])/2

    # 3. img generation step
    img_rrim = makeRRIMImage(slopeMat, opennessMat)
    cv2.imwrite(output_fname, img_rrim)

    print('rrim complete.')


if __name__ == '__main__':
    depthFile = './data/ASTGTM2_N29E111_dem.tif'
    raster = cv2.imread(depthFile, cv2.IMREAD_UNCHANGED)
    cellSize = 30
    L = 600

    '''
    raster = cv2.imread("./data/result.tif", cv2.IMREAD_UNCHANGED)
    cellSize = 0.05
    L = 0.8
    
    depthFile = './data/depth.csv'
    raster = np.loadtxt(depthFile, delimiter=',')
    cellSize = 0.05
    L = 0.5
    '''
    rrimFile = depthFile[:-4]+'_rrim.png'  # output file name
    rrim(raster.astype(np.float), cellSize, L, rrimFile)  # main function of rrim


