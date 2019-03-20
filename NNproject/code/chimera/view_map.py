from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np



def scatter_cryo_EM_map(mtrx, thr,ax,color='b',mr='o',label='X'):
    #create indexes
    Xr,Yr,Zr = mtrx.shape
    print(Xr,Yr,Zr)
    Xs,Ys,Zs = np.meshgrid(range(Xr),range(Yr),range(Zr),indexing='ij')
    xs = np.reshape(Xs,-1)
    ys = np.reshape(Ys,-1)
    zs = np.reshape(Zs,-1)

    inds = np.where(np.reshape(mtrx,-1)>thr)[0]
    xs = xs[inds]
    ys = ys[inds]
    zs = zs[inds]

    ax.scatter(xs, ys, zs, c=color, marker=mr,label=label)

mtrxO = np.load('O.npy')
mtrxC = np.load('C.npy')
mtrxN = np.load('N.npy')
mtrxS = np.load('S.npy')
mtrxOut = np.load('output.npy')






fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter_cryo_EM_map(mtrxOut, 0.0614,ax,color='y', mr='o',label='map')
scatter_cryo_EM_map(mtrxO, 0.0614,ax,color='b', mr='+',label='O')
scatter_cryo_EM_map(mtrxC, 0.0614,ax,color='r', mr='+',label='C')
scatter_cryo_EM_map(mtrxN, 0.0614,ax,color='g', mr='+',label='N')
scatter_cryo_EM_map(mtrxS, 0.0614,ax,color='m', mr='+',label='S')

ax.legend()

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
