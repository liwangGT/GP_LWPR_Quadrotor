#!/usr/bin/env python
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import sys
from math import pi as PI
from math import atan2, sin, cos, sqrt, fabs, floor, exp
from scipy.misc import factorial as ft
from scipy.misc import comb as nCk
from numpy import linalg as LA

import matplotlib.patches
import matplotlib.pyplot as plt

import time 
import numpy as np
from cvxopt import matrix, solvers
import pickle #save multiple objects

# add quad simulator path
sys.path.insert(0, '/home/li/gitRepo/GP_Quadrotor/Quad_simulator/Interp')
from Spline_interp import *
from Quad_visual import *


if __name__ == "__main__":
    #rospy.init_node('cf_barrier', anonymous = True)
    fig = plt.figure()
    ax  = fig.gca(projection = '3d')
    ax.set_aspect('equal')
    ax.invert_zaxis()

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([3, 2, 1.6]).max()
    Xb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][0].flatten()
    Yb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][1].flatten() 
    Zb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][2].flatten() - 1
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')
    plt.pause(.001)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    # consider 1 agents
    N = 1
    deg = 8 # spline poly degree is set to 8
    # waypoints
    p0 = dict()
    p0[0] = np.array([[-1., -1., 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    # init ploting handles
    Cfplt = dict() 
    CfCoord = dict()
    for i in range(N):
        #Cfplt[i] = Shpere3D(p0[i][0,:], ax)
        CfCoord[i] = Coord3D(ax, p0[i][0,:])

    # initial point
    p0[1] = np.array([[-1., -1., -0.8],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # end point
    p0[2] = np.array([[1.6, 1.3, -1.4],
                   [-0.2, 0.5, 0.2],
                   [-3, 2.5, 1],
                   [0, 0, 0]])

    # end point
    p0[3] = np.array([[-1.2, 1.0, -1.7],
                   [-0.1, -0.3, 0.2],
                   [-3, 2, -1.6],
                   [0, 0, 0]])
    # end point
    p0[4] = np.array([[-1.8, 0.6, -1.5],
                   [-0.3, -0.2, 0.6],
                   [1, -2.2, 3.3],
                   [0.1, 0.3, 0.4]])
    p0[5] = np.array([[1, 0.6, -1.5],
                   [0.5, -0.2, -0.6],
                   [1., 2.0, 1.3],
                   [0, 0.1, 0]])

    # end point
    p0[6] = np.array([[1.7, -1.2, -0.8],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p0[7] = np.array([[1.7, -1.2, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # draw waypoints
    wp = dict()
    for jj in range(len(p0)):
        wp[jj] = Coord3D(ax, p0[jj][0,:], trans=0.2)
        roll, pitch, yaw = Invert_diff_flat_output(p0[jj])
        wp[jj].update(p0[jj][0,:], roll, pitch, yaw)


    # draw a corridor constraint
    delta = 0.1 # corridor width is 0.05m
    x=np.linspace(p0[4][0,0],p0[5][0,0], 10)
    z=np.linspace(-delta, delta, 10)
    Xc, Zc=np.meshgrid(x, z)
    Yc = np.sqrt(delta**2-Zc**2)

    # Draw parameters
    rstride = 4
    cstride = 3
    ax.plot_surface(Xc, Yc+p0[5][0,1], Zc+p0[5][0,2], alpha=0.1, rstride=rstride, cstride=cstride)
    ax.plot_surface(Xc, -Yc+p0[5][0,1], Zc+p0[5][0,2], alpha=0.1, rstride=rstride, cstride=cstride)

    Tp = np.array([0.8, 1.5, 1.5, 1., 1., 1., 0.8,1.,1.])
    # interpolating waypoints
    dt = 0.05
    thist = np.empty((0,1))
    xhist = np.empty((0,9))
    yhist = np.empty((0,3))
    yreal = np.empty((0,3))
    for j in range(len(p0)-1):
        t = 0
        k = j+1
        T = Tp[j]
        if (j==4):
            coeff = Interp_corridor(p0[j], p0[k], T, 15) # interp between p0 and p1
            #coeff = Interp(p0[j], p0[k], T, deg) # interp between p0 and p1
        else:
            coeff = Interp(p0[j], p0[k], T, deg) # interp between p0 and p1
        while (t<Tp[j]):
           t += dt
           pk = ExtractVal(coeff, t, T)
           #print pk
           roll, pitch, yaw = Invert_diff_flat_output(pk)
           CfCoord[0].update(pk[0,:], roll, pitch, yaw)
           #Cfplt[0].update(pk[0,:],ax)
           plt.pause(.001)
           # log data
           thist = np.vstack((thist, t))
           xnew = np.hstack((pk[0,:], pk[1,:], np.array([roll, pitch, yaw])))
           xhist = np.vstack((xhist, xnew))
           ynoise = 2.5 * (np.random.rand(3,) -0.5)
           yhist = np.vstack((yhist, pk[2,:]*0.5+ynoise )) 
           yreal = np.vstack((yreal, pk[2,:]*0.5)) 

    # save ground truth data
    # x value is: (rx,ry,rz, vx,vy,vz, rho, pitch, yaw)
    # y value is: (y1,y2,y3)
    f = open('Sim_ground_truth02.pckl', 'w')
    pickle.dump([dt, xhist, yhist, yreal], f)
    f.close()
    print '----data logging completed!!!----'

    plt.show()


