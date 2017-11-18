from stl import mesh
import numpy as np

def compare_s(v):
    vs = [[-72.99987793, -72.99987793], [-72.99987793, 327],
          [327, -72.99987793], [327, 327]]

    for t in vs:
        if abs(v[0] - t[0]) < 0.0001 and abs(v[1] - t[1]) < 0.0001:
            return True
    return False


def compare(vector):
    if compare_s(vector[0]) and compare_s(vector[1]) and compare_s(vector[2]):
        return False
    if vector[0][2] == 0 and vector[1][2] == 0 and vector[2][2] == 0:
        return False
    return True

data = np.zeros(48, dtype=mesh.Mesh.dtype)
your_mesh = mesh.Mesh.from_file('p.stl')

print(np.min(your_mesh.vectors, axis=0))
'''

for ov in your_mesh.vectors:
    for iv in ov:
        if abs(iv[0] - 327)<0.0001 or abs(iv[0]+72.99987793)<0.0001:
            print(iv[:2])

'''
count = 0
for i in range(64):
    if compare(your_mesh.vectors[i]):
        data['vectors'][count] = your_mesh.vectors[i]
        count += 1
    #else:
        #print(your_mesh.vectors[i])

print(count)
newmesh = mesh.Mesh(data)
newmesh.save('newp.stl')

print(your_mesh.vectors.shape)
