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

from Spline_interp import *
from Quad_visual import *


def Invert_diff_flat_output(x):
    #note yaw =0 is fixed
    m = 35.89/1000
    g = 9.8
    beta1 = - x[2,0]
    beta2 = - x[2,2] + 9.8
    beta3 = x[2,1]

    roll = atan2(beta3,sqrt(beta1**2+beta2**2))
    pitch = atan2(beta1,beta2)
    yaw = 0
    a_temp = LA.norm([0,0,g]-x[2,:])
    # acc g correspond to 49201
    thrust = int(a_temp/g*49201)
    return roll,pitch,yaw


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


    # consider 1 agents
    N = 10
    deg = 8 # spline poly degree is set to 8
    # waypoints
    p0 = dict()
    p1 = dict()
    p2 = dict()
    p3 = dict()
    p4 = dict()
    p5 = dict()
    p6 = dict()
    p7 = dict()
    p8 = dict()
    p9 = dict()

    # init ploting handles
    Cfplt = dict() 
    CfCoord = dict()


    # waypoints
    p0[0] = np.array([[-1, -1, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p0[1] = np.array([[-1, 1, -0.8],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p0[2] = np.array([[0.5, -1.0, -1.5],
                   [-0.2, 0.2, -0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p0[3] = np.array([[-0.6, -1.0, -1.0],
                   [0.2, -0.8, 0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p0[4] = np.array([[1.5, 1.0, -1.6],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p0[5] = np.array([[1.5, 1.0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # waypoints
    p1[0] = np.array([[-1, 1, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p1[1] = np.array([[1, -1, -0.8],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p1[2] = np.array([[-0.5, 1.0, -1.4],
                   [-0.2, 0.2, -0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p1[3] = np.array([[0.6, 1.0, -1.0],
                   [0.2, -0.8, 0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p1[4] = np.array([[-1.5, -1.0, -1.3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p1[5] = np.array([[-1.5, -1.0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # waypoints
    p2[0] = np.array([[1, -1, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p2[1] = np.array([[1, 1, -0.8],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p2[2] = np.array([[-0.5, -1.0, -1.2],
                   [-0.2, 0.2, -0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p2[3] = np.array([[0.6, -1.0, -1.0],
                   [0.2, -0.8, 0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p2[4] = np.array([[-1.5, 1.0, -0.6],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p2[5] = np.array([[-1.5, 1.0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # waypoints
    p3[0] = np.array([[1, 1, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p3[1] = np.array([[-0.5, -1, -0.6],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p3[2] = np.array([[0.8, 1.0, -1.0],
                   [-0.2, 0.2, -0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p3[3] = np.array([[-0.6, 1.2, -1.2],
                   [0.2, -0.8, 0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p3[4] = np.array([[1.5, -1.1, -0.8],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p3[5] = np.array([[1.0, -0.5, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # waypoints
    p4[0] = np.array([[-1.5, 0.5, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p4[1] = np.array([[-1, -1, -1.3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p4[2] = np.array([[0.8, 1.3, -1.2],
                   [-0.2, 0.2, -0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p4[3] = np.array([[-0.8, 1.2, -1.4],
                   [0.2, -0.8, 0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p4[4] = np.array([[1.2, -1.0, -1.1],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p4[5] = np.array([[-1.5, -0.5, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # waypoints
    p5[0] = np.array([[0.1, 0.2, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p5[1] = np.array([[0.1, 0.2, -1.4],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p5[2] = np.array([[0.2, 1.1, -1.8],
                   [-0.2, 0.2, -0.1],
                   [-4, -3, 0.2],
                   [0, 0, 0]])
    p5[3] = np.array([[-0.2, 1.4, -1.1],
                   [0.2, -0.8, 0.1],
                   [2, 4, 0],
                   [0, 0, 0]])
    p5[4] = np.array([[-0.8, -0.7, -1.2],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p5[5] = np.array([[-0.8, -0.7, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # waypoints
    p6[0] = np.array([[-1.0, -0.8, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p6[1] = np.array([[-1.2, -0.8, -1.3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p6[2] = np.array([[0.3, -1.2, -0.8],
                   [-0.2, 0.2, -0.1],
                   [-2, 4, 0],
                   [0, 0, 0]])
    p6[3] = np.array([[-0.5, 1.0, -1.2],
                   [0.2, -0.8, 0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p6[4] = np.array([[1.2, -1.3, -1.0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p6[5] = np.array([[-1.2, -0.5, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    # waypoints
    p7[0] = np.array([[-1.7, -0.7, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p7[1] = np.array([[-1.3, -0.7, -1.4],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p7[2] = np.array([[0.4, -1.3, -1.8],
                   [-0.2, 0.2, -0.1],
                   [0, 3, 2],
                   [0, 0, 0]])
    p7[3] = np.array([[-0.7, 1.7, -1.2],
                   [0.2, -0.8, 0.1],
                   [2, 4, 0],
                   [0, 0, 0]])
    p7[4] = np.array([[-0.7, 0.7, -1.3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p7[5] = np.array([[-0.7, 0.7, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    # waypoints
    p8[0] = np.array([[-0.2, 1.7, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p8[1] = np.array([[-0.6, 1.2, -1.0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p8[2] = np.array([[0.7, -1.6, -0.8],
                   [-0.2, 0.2, -0.1],
                   [2, 3, 0],
                   [0, 0, 0]])
    p8[3] = np.array([[-0.4, 1.7, -1.2],
                   [0.2, -0.8, 0.1],
                   [2, 0, 3],
                   [0, 0, 0]])
    p8[4] = np.array([[-1.7, 1.2, -1.0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p8[5] = np.array([[-1.7, 1.2, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    # waypoints
    p9[0] = np.array([[-0.8, -0.3, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p9[1] = np.array([[-1.3, 0.5, -1.0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p9[2] = np.array([[1.2, -1.3, -1.8],
                   [-0.2, 0.2, -0.1],
                   [3, 2, 0],
                   [0, 0, 0]])
    p9[3] = np.array([[-0.3, 1.6, -1.2],
                   [0.2, -0.8, 0.1],
                   [-4, 1., 0],
                   [0, 0, 0]])
    p9[4] = np.array([[0.8, -1.3, -1.0],
                   [0.3, 1.2, 0.1],
                   [0, 0, 0],
                   [0, 0, 0]])
    p9[5] = np.array([[0.8, -1.3, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])



    #Cfplt[i] = Shpere3D(p0[i][0,:], ax)
    CfCoord[0] = Coord3D(ax, p0[0][0,:])
    CfCoord[1] = Coord3D(ax, p1[0][0,:])
    CfCoord[2] = Coord3D(ax, p2[0][0,:])
    CfCoord[3] = Coord3D(ax, p3[0][0,:])
    CfCoord[4] = Coord3D(ax, p4[0][0,:])
    CfCoord[5] = Coord3D(ax, p5[0][0,:])
    CfCoord[6] = Coord3D(ax, p6[0][0,:])
    CfCoord[7] = Coord3D(ax, p7[0][0,:])
    CfCoord[8] = Coord3D(ax, p8[0][0,:])
    CfCoord[9] = Coord3D(ax, p9[0][0,:])


    Tp = np.array([1, 2, 2, 2, 1, 2])
    # interpolating waypoints
    dt = 0.03
    coeff = dict()
    time.sleep(4)
    plt.pause(.001)
    time.sleep(3)
    plt.pause(.001)
    time.sleep(3)
    for j in range(len(p0)):
        t = 0
        k = j+1
        if (k==len(p0)):
            k = 0
        T = Tp[j]
        coeff[0] = Interp(p0[j], p0[k], T, deg) # interp between p0 and p1
        coeff[1] = Interp(p1[j], p1[k], T, deg) # interp between p0 and p1
        coeff[2] = Interp(p2[j], p2[k], T, deg) # interp between p0 and p1
        coeff[3] = Interp(p3[j], p3[k], T, deg) # interp between p0 and p1
        coeff[4] = Interp(p4[j], p4[k], T, deg) # interp between p0 and p1
        coeff[5] = Interp(p5[j], p5[k], T, deg) # interp between p0 and p1
        coeff[6] = Interp(p6[j], p6[k], T, deg) # interp between p0 and p1
        coeff[7] = Interp(p7[j], p7[k], T, deg) # interp between p0 and p1
        coeff[8] = Interp(p8[j], p8[k], T, deg) # interp between p0 and p1
        coeff[9] = Interp(p9[j], p9[k], T, deg) # interp between p0 and p1
        while (t<Tp[j]):
           t += dt
           for nn in range(N):
               pk = ExtractVal(coeff[nn], t, T)
               #print pk
               roll, pitch, yaw = Invert_diff_flat_output(pk)
               CfCoord[nn].update(pk[0,:], roll, pitch, yaw)
               #Cfplt[0].update(pk[0,:],ax)
           plt.pause(.001)
           
    plt.show()
