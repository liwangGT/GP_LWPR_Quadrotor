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

# min-snap spline interpolation, generates polynomial basis coefficients
def Interp(p0, p1, T, deg):
    k = deg # highest degree
    dim = p0[0].size
    coeff = np.zeros((k+1, dim))
    # scale the derivaties with T
    p0[1,:] *= T
    p0[2,:] *= T**2
    p0[3,:] *= T**3
    p1[1,:] *= T
    p1[2,:] *= T**2
    p1[3,:] *= T**3

    # H, f, A is the same for each dimension
    hh = 1.0/np.arange(1, k+1) # build Hblock from hh
    Hblock = np.array([[1,     1.0/2, 1.0/3, 1.0/4, 1.0/5],
                       [1.0/2, 1.0/3, 1.0/4, 1.0/5, 1.0/6],
                       [1.0/3, 1.0/4, 1.0/5, 1.0/6, 1.0/7],
                       [1.0/4, 1.0/5, 1.0/6, 1.0/7, 1.0/8],
                       [1.0/5, 1.0/6, 1.0/7, 1.0/8, 1.0/9]])
    Htemp = np.vstack( (np.zeros((4, k+1)), np.hstack((np.zeros((5,4)),Hblock))  ) )
    kscale = np.zeros((k+1,)) # scaling
    for ii in range(k+1):
        if (ii == 4): 
            kscale[ii] = 1*2*3*4.0
        if (ii>4):
            kscale[ii] = kscale[ii-1]*ii/(ii-4)
    H = kscale[:,None]*Htemp*kscale # scaling each dimension, not matrix multiplication
    f = np.zeros((k+1,1))
    A = np.zeros((8,k+1)) # init A matrix
    for i in range(k+1):
        A[0,0] = 1
        A[1,1] = 1
        A[2,2] = 2
        A[3,3] = 6
        A[4,:] = np.ones((1,k+1))
        A[5,:] = np.arange(k+1)
        A[6,:] = A[5,:]*(np.arange(k+1)-1)
        A[7,:] = A[6,:]*(np.arange(k+1)-2)
    A=A.astype(float) # convert to double to avoid errors    

    # b is different for each dimension
    for i in range(dim):
        b = np.append(p0[:,i], p1[:,i])[:,None]    
        # 0.5x^THx+f^Tx, s.t. Gx<=h, Ax=b
        sol = solvers.qp(matrix(H), matrix(f), matrix(np.empty((0,k+1))), matrix(np.empty((0,1))), matrix(A), matrix(b))
        x = sol['x']
        coeff[:,i] = np.reshape(x,(k+1,))
    print H.shape, f.shape, A.shape, b.shape
    return coeff

# extract points using polynomial basis
def ExtractVal(coeff, t, T):
    t = min(t/T,1)
    # automatically extract the degree of polynominal
    deg = coeff[:,0].size - 1
    print "deg is: ", deg
    pk = np.zeros((4, coeff[0,:].size))
    for i in range(coeff[0,:].size):
        # for each dimension
        for j in range(coeff[:,0].size):
           pk[0,i] += coeff[j,i]*t**j
           if j>0:
               pk[1,i] += j*coeff[j,i]*t**(j-1)
           if j>1:
               pk[2,i] += j*(j-1)*coeff[j,i]*t**(j-2)
           if j>2:
               pk[3,i] += j*(j-1)*(j-2)*coeff[j,i]*t**(j-3)
    pk[1,:] /= T
    pk[2,:] /= T**2
    pk[3,:] /= T**3
    return pk

# quiver 3D does not work in python2
def plotQuiver(p, ax, cl):
    ax.plot([p[0,0], p[0,0]+p[1,0]], [p[0,1], p[0,1]+p[1,1]], [p[0,2], p[0,2]+p[1,2]], cl)

if __name__ == "__main__":
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


    # initial point
    p0 = np.array([[-1, 0, -0.5],
                   [0, 0.5, 2.5],
                   [0, 0, 0],
                   [0, 0, 0]])
    plotQuiver(p0, ax, 'g-')

    # end point
    p1 = np.array([[1, 0.5, 0.5],
                   [0, -0.6, -2.5],
                   [0, 0, 0],
                   [0, 0, 0]])
    plotQuiver(p1, ax, 'r-')

    # end point
    p2 = np.array([[0.3, -1.5, -1.5],
                   [2, 1, 1.5],
                   [0, 0, 0],
                   [0, 0, 0]])
    plotQuiver(p2, ax, 'b-')

    # end point
    p3 = np.array([[1.3, 1.5, -1.0],
                   [-2, 1, -1.5],
                   [0, 0, 0],
                   [0, 0, 0]])
    plotQuiver(p3, ax, 'y-')

    # spline interpolations, min-snap splines
    coeff = Interp(p0, p1, 1, 8) # interp between p0 and p1
    t = np.linspace(0, 1, 100, endpoint=True)
    for i in range(t.size):
       pk = ExtractVal(coeff, t[i], 1)
       ax.plot([pk[0,0]], [pk[0,1]], [pk[0,2]], 'g*')

    coeff = Interp(p1, p2, 1, 8) # interp between p1 and p2
    t = np.linspace(0, 1, 100, endpoint=True)
    for i in range(t.size):
       pk = ExtractVal(coeff, t[i], 1)
       ax.plot([pk[0,0]], [pk[0,1]], [pk[0,2]], 'r*')

    coeff = Interp(p2, p3,1, 8) # interp between p2 and p3
    t = np.linspace(0, 1, 100, endpoint=True)
    for i in range(t.size):
       pk = ExtractVal(coeff, t[i],1)
       ax.plot([pk[0,0]], [pk[0,1]], [pk[0,2]], 'b*')

    coeff = Interp(p3, p0,1, 8) # interp between p3 and p0
    t = np.linspace(0, 1, 100, endpoint=True)
    for i in range(t.size):
       pk = ExtractVal(coeff, t[i],1)
       ax.plot([pk[0,0]], [pk[0,1]], [pk[0,2]], 'g*')

    plt.show()



