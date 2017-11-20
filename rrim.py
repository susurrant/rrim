#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
 Implemented by Xin Yao : https://github.com/susurrant/'
"""

import cv2
import numpy as np
import time
from stl import mesh

# compute zenith and nadir angles
def theta(dh):
    if len(dh):
        x = dh[:,1]/dh[:,0]
        v = np.array([np.max(x), -np.min(x)])
        return 90-np.arctan(v)*180/np.pi
    return 0, 0

# compute openness
def openness(depth, r, c, cell_size, L):
    dc = [1,1,0,-1,-1,-1,0,1]
    dr = [0,1,1,1,0,-1,-1,-1]
    o = []
    for dCount in range(8):
        pCount = 1
        e = []
        while pCount:
            pr = r+dr[dCount]*pCount
            pc = c+dc[dCount]*pCount
            dis = np.sqrt((pr-r)**2+(pc-c)**2)*cell_size
            if dis > L or pr < 0 or pr >= depth.shape[0] or pc < 0 or pc >= depth.shape[1]:
                break
            e.append([dis, depth[pr, pc] - depth[r, c]])
            pCount += 1
        o.append(theta(np.array(e)))

    return np.sum(o, axis=0)/8

# compute openness with numpy, but it failed
def theta_n(dh, dis):
    if dh.size:
        x = dh/dis
        v = np.array([np.max(x), -np.min(x)])
        return 90 - np.arctan(v) * 180 / np.pi
    return 0, 0

def openness_n(depth, j, i, cell_size, L):
    a = np.arange(1, int(L / cell_size))
    row = a[:, np.newaxis] * np.array([0, 1, 1, 1, 0, -1, -1, -1]) + j
    column = a[:, np.newaxis] * np.array([1, 1, 0, -1, -1, -1, 0, 1]) + i

    idx = (row < depth.shape[0]) & (row  >= 0) & (column < depth.shape[1]) & (column >= 0)
    o = []
    for n, y in enumerate(np.transpose(idx)):
        rs = row[np.where(y == True), n]
        cs = column[np.where(y == True), n]
        dis = (np.sqrt((rs - j) ** 2 + (cs - i) ** 2))*cell_size
        dh = depth[rs, cs] - depth[j, i]
        o.append(theta_n(dh, dis))

    return np.sum(o, axis=0) / 8

# compute slope
def slope(depth, cell_size):
    y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = cv2.filter2D(depth, -1, y_kernel) / (8 * cell_size)
    dx = cv2.filter2D(depth, -1, x_kernel) / (8 * cell_size)
    return np.arctan(np.sqrt(dy ** 2 + dx ** 2)) * 180 / np.pi

# color scheme
def colorScheme(size):
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
def genRRIMImage(slopedata, openness, color_size, output_fname):
    RRIM_map = colorScheme(color_size)

    result = np.zeros((slopedata.shape[0], slopedata.shape[1], 3), dtype=np.uint8)

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            inc = np.uint8(min(slopedata[y, x], color_size[0]-1))
            openness_val = np.uint8(openness[y, x]+color_size[1]/2)
            if openness_val < 0:
                openness_val = 0
            elif openness_val >= color_size[1]:
                openness_val = color_size[1]-1
            result[y, x, :] = RRIM_map[inc, openness_val]

    cv2.imwrite(output_fname, result)

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
def rrim(depth, cell_size, L, output_fname, color_size=(50, 50, 3)):
    print('start rrim...')

    # 1. slop step
    slopeMat = slope(depth, cell_size)

    # 2. openness step

    opennessMat = np.zeros(depth.shape)
    for j in range(depth.shape[0]):
        if j % 100 == 0:
            print('  %.2f  finished...' % (j/depth.shape[0]*100))
        for i in range(depth.shape[1]):
            o = openness(depth, j, i, cell_size, L)
            opennessMat[j,i] = (o[0]-o[1])/2

    # 3. img generation step
    genRRIMImage(slopeMat, opennessMat, color_size, output_fname)

    print('rrim complete.')

# read data from an image or a DEM or a file
# cell_size must be mannully set in the main funtion
def readDataFromImg(dem_file):
    return cv2.imread(dem_file, cv2.IMREAD_UNCHANGED)

def readDataFromFile(file_name, delimiter=',', skiprows=0):
    return np.loadtxt(file_name, delimiter=delimiter, skiprows=skiprows)

# read data from a stl file
# A depth map is needed, which can be obtain with MeshLab
# the cell_size is automatically computed
def readDataFromStl(depth_img, stl_name):
    d = cv2.imread(depth_img, cv2.IMREAD_UNCHANGED)[:, :, 0]
    y, x = np.where(d != 255)
    d = 255 - d[np.min(y):np.max(y) + 1, np.min(x):np.max(x) + 1]
    d[np.where(d == 0)] = sorted(list(set(d.flatten())))[1]

    m = mesh.Mesh.from_file(stl_name)
    cell_size = (np.max(m.vectors, axis=0)[0, 1] - np.min(m.vectors, axis=0)[0, 1]) / d.shape[0]

    return d, cell_size



if __name__ == '__main__':
    depth_file = './data/ASTGTM2_N29E111_dem.tif'
    raster = readDataFromImg('./data/ASTGTM2_N29E111_dem.tif')
    cell_size = 30
    L = 600
    
    '''
    depth_file = './data/Bodacious_Snaget-Krunk.png'
    stl_file = './data/Bodacious_Snaget-Krunk.stl'
    raster, cell_size = readDataFromStl(depth_file, stl_file)
    L = 0.6       # usually L is slightly larger than cell_size*10
    '''

    rrimFile = depth_file[:-4]+'_rrim_new.png'  # output file name
    rrim(raster.astype(np.float), cell_size, L, rrimFile, color_size=(90, 50, 3))  # main function of rrim


