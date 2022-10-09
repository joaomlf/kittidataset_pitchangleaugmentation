import numpy as np
import math
import torch
import cv2
import pptk
import utils.config as cnf
import utils.kitti_utils as kitti_utils

import os

currdir = os.getcwd()
os.chdir('data/KITTI/object/training/velodyne')

PointCloud = np.fromfile(str("000835.bin"), dtype=np.float32, count=-1).reshape([-1,4])

BoundaryCond = cnf.boundary
Discretization = cnf.DISCRETIZATION

# Boundary condition
minX = BoundaryCond['minX']
maxX = BoundaryCond['maxX']
minY = BoundaryCond['minY']
maxY = BoundaryCond['maxY']
minZ = BoundaryCond['minZ']
maxZ = BoundaryCond['maxZ']

# Remove the point out of range x,y,z
mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
        PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
PointCloud = PointCloud[mask]

PointCloud[:, 2] = PointCloud[:, 2] - minZ

Height = cnf.BEV_HEIGHT + 1
Width = cnf.BEV_WIDTH + 1

# Discretize Feature Map
PointCloud1 = np.copy(PointCloud)
PointCloud1[:, 0] = np.int_(np.floor(PointCloud1[:, 0] / Discretization))
PointCloud1[:, 1] = np.int_(np.floor(PointCloud1[:, 1] / Discretization) + Width / 2)

# sort-3times
indices = np.lexsort((-PointCloud1[:, 2], PointCloud1[:, 1], PointCloud1[:, 0]))
PointCloud1 = PointCloud1[indices]

# Height Map
heightMap = np.zeros((Height, Width))

_, indices = np.unique(PointCloud1[:, 0:2], axis=0, return_index=True)
PointCloud1_frac = PointCloud1[indices]
# some important problem is image coordinate is (y,x), not (x,y)
max_height = float(np.abs(BoundaryCond['maxZ'] - BoundaryCond['minZ']))
heightMap[np.int_(PointCloud1_frac[:, 0]), np.int_(PointCloud1_frac[:, 1])] = PointCloud1_frac[:, 2] / max_height

# Intensity Map & Density Map
intensityMap = np.zeros((Height, Width))
densityMap = np.zeros((Height, Width))

_, indices, counts = np.unique(PointCloud1[:, 0:2], axis=0, return_index=True, return_counts=True)
PointCloud1_top = PointCloud1[indices]

normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

intensityMap[np.int_(PointCloud1_top[:, 0]), np.int_(PointCloud1_top[:, 1])] = PointCloud1_top[:, 3]
densityMap[np.int_(PointCloud1_top[:, 0]), np.int_(PointCloud1_top[:, 1])] = normalizedCounts

print(PointCloud.shape)
x = PointCloud[:, 0]  # x position of point
y = PointCloud[:, 1]  # y position of point
z = PointCloud[:, 2]  # z position of point
r = PointCloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
#print(z)
#print(r)
#print(d)
#print(np.max(heightMap))
#print(np.min(heightMap))
#print(np.mean(heightMap))
#print(x[12123], y[12123], z[12123])

# Save PCL data in txt
#with open( "PCL_data.txt", mode='w' ) as file:      # open file to save
#    file.write( "X\t\tY\t\tZ\t\tReflectance\n")
#    for i in range( len(PointCloud) ):
#        file.write( "%f\t%f\t%f\t%f\n" % (x[i], y[i], z[i], r[i]) )
