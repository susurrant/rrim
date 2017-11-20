import numpy as np
import cv2
from stl import mesh


d = cv2.imread('./data/Bodacious_Snaget-Krunk.png', cv2.IMREAD_UNCHANGED)[:,:,0]

y, x = np.where(d != 255)
d = 255 - d[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
d[np.where(d==0)] = sorted(list(set(d.flatten())))[1]

mesh = mesh.Mesh.from_file('./data/Bodacious_Snaget-Krunk.stl')
cellSize = (np.max(mesh.vectors, axis=0)[0,1]-np.min(mesh.vectors, axis=0)[0,1])/d.shape[0]
print(cellSize)

cv2.namedWindow("Image")
cv2.imshow("Image", d)
cv2.waitKey(0)
cv2.destroyAllWindows()

