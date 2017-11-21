import numpy as np
import cv2
from stl import mesh

file_name = 'estrella_nijadoble'

d = cv2.imread('./data/'+file_name+'.png', cv2.IMREAD_UNCHANGED)[:,:,0]
y, x = np.where(d != 255)
d = 255 - d[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
d[np.where(d==0)] = sorted(list(set(d.flatten())))[1]
print('shape     :', d.shape)
print('gray range: %d - %d' % (np.min(d), np.max(d)))
cv2.namedWindow("Image")
cv2.imshow("Image", d)
cv2.waitKey (0)
cv2.destroyAllWindows()

mesh = mesh.Mesh.from_file('./data/'+file_name+'.stl')
zmin = np.min(mesh.vectors, axis=0)[0,2]
zmax = np.max(mesh.vectors, axis=0)[0,2]
print('z range   : %f - %f' % (zmin, zmax))

d = zmin + (d-np.min(d))*(zmax-zmin)/(np.max(d)-np.min(d))
cell_size = (np.max(mesh.vectors, axis=0)[0,1]-np.min(mesh.vectors, axis=0)[0,1])/d.shape[0]
print('cell size :', cell_size)

#np.savetxt('./data/'+file_name+'_depth.csv', d, delimiter=',')


