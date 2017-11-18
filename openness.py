
import numpy as np

# compute zenith or nadir angles
def theta(dh):
    if len(dh):
        x = dh[:,1]/dh[:,0]
        return 90-np.arctan(np.max(x))*180/np.pi
    return 0

# compute openness
def openness(depth, r, c, cellSize, L):
    np.arange(1, int(L/cellSize))

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
            ne.append([dis, depth[r, c]-depth[pr, pc]])
            pCount += 1

        po.append(theta(np.array(pe)))
        no.append(theta(np.array(ne)))

    return sum(po)/8, sum(no)/8


a = np.arange(1,5)
d = np.array([0,1])
dr = np.array([0,1,1,1,0,-1,-1,-1])
dc = np.array([1,1,0,-1,-1,-1,0,1])
r = a[:,np.newaxis]*d + [1,1]
print(r)


#print(r)
#print(c)
#print(r[ridx])
#print(c[cidx])
