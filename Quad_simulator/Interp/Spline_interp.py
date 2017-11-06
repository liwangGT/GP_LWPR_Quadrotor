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
    hh = 1.0/np.arange(1, 2*k-6) # build Hblock from hh
    Hblock = np.zeros((k-3, k-3))
    for ii in range(k-3):
        Hblock[ii,:] = hh[ii:ii+k-3] 

    # H is (k-3) by (k-3)
    Htemp = np.vstack( (np.zeros((4, k+1)), np.hstack((np.zeros((k-3,4)),Hblock))  ) )
    kscale = np.zeros((k+1,)) # scaling
    for ii in range(k+1):
        if (ii == 4): 
            kscale[ii] = 1*2*3*4.0
        if (ii>4):
            kscale[ii] = kscale[ii-1]*ii/(ii-4)
    H = Htemp
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
        print sol['primal objective']
        coeff[:,i] = np.reshape(x,(k+1,))
        print coeff[:,i]
        #print 0.5*np.dot(np.dot(H,coeff[:,i]), coeff[:,i])
        #print np.dot(A, coeff[:,i])-b.T[0]
    
    print H.shape, f.shape, A.shape, b.shape
    return coeff

# extract points using polynomial basis
def ExtractVal(coeff, t, T):
    t = min(t/T,1)
    # automatically extract the degree of polynominal
    deg = coeff[:,0].size - 1
    #print "deg is: ", deg
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

# nomalize a vector
def normalize(x):
    temp = sqrt(sum(x**2)*1.0)
    if (temp==0):
        return x
    else: 
        return x/temp

# min-snap spline interpolation with corridor constraints
# corridor is ||(xyz -p0[0,:]) - (xyz -p0[0,:])*ti*ti||_infty < delta
# for simplicity, assume p0[yz] = p1[yz], i.e., ti = [1,0,0], |dy|<delta, |dz|<delta
# take 4 intermediate points: t = 0.2, 0.4, 0.6, 0.8
# note: otherwise, need to combine 3 dimensions into one QP
def Interp_corridor(p0, p1, T, deg):
    #ti = normalize(p1[0,:] - p0[0,:]) # unit vector 
    delta = 0.05 # corridor width is 0.05m
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
    hh = 1.0/np.arange(1, 2*k-6) # build Hblock from hh
    Hblock = np.zeros((k-3, k-3))
    for ii in range(k-3):
        Hblock[ii,:] = hh[ii:ii+k-3] 

    # H is (k-3) by (k-3)
    Htemp = np.vstack( (np.zeros((4, k+1)), np.hstack((np.zeros((k-3,4)),Hblock))  ) )
    kscale = np.zeros((k+1,)) # scaling
    for ii in range(k+1):
        if (ii == 4): 
            kscale[ii] = 1*2*3*4.0
        if (ii>4):
            kscale[ii] = kscale[ii-1]*ii/(ii-4)
    H = Htemp
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
        # inequality constraint
        G = np.empty((0,k+1))
        h = np.empty((0,1))
        if (i>=1):
            # y/z corridor, yt-p0[1]<=delta, p0[1]-yt<=delta
            for tt in np.array([0.2,0.4,0.6,0.8]):
                h = np.vstack((h, np.array([[p0[0,i]+delta]]), np.array([[delta-p0[0,i]]]) ))
                G = np.vstack((G, tt**np.arange(k+1), -tt**np.arange(k+1)))
        b = np.append(p0[:,i], p1[:,i])[:,None]    
        # 0.5x^THx+f^Tx, s.t. Gx<=h, Ax=b
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(A), matrix(b))
        x = sol['x']
        print sol['status']
        print sol['primal objective']
        coeff[:,i] = np.reshape(x,(k+1,))
        print coeff[:,i]
        #print 0.5*np.dot(np.dot(H,coeff[:,i]), coeff[:,i])
        #print np.dot(A, coeff[:,i])-b.T[0]
    
    print H.shape, f.shape, A.shape, b.shape
    return coeff


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


    # waypoints
    p0 = dict()
    p0[0] = np.array([[-1., -1., 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    # initial point
    p0[1] = np.array([[-1., -1., -0.8],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # end point
    p0[2] = np.array([[1.4, 1.0, -1.9],
                   [-0.0, 0.4, -0.2],
                   [-4, 2, 1],
                   [0, 0, 0]])

    # end point
    p0[3] = np.array([[-1.2, 1.0, -1.7],
                   [-0.0, -0.3, 0.2],
                   [4, -2, -1],
                   [0, 0, 0]])
    # end point
    p0[4] = np.array([[-1., 0, -1.3],
                   [0.1, -0.2, 0.6],
                   [1, 1.2, 1.3],
                   [0.1, 0.3, 0.4]])
    p0[5] = np.array([[1, 0, -1.3],
                   [0.5, -0.2, -0.6],
                   [1., 2.0, 1.3],
                   [0, 0.1, 0]])

    # end point
    p0[6] = np.array([[1.5, -1.0, -0.8],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    p0[7] = np.array([[1.5, -1.0, 0],
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
    cstride = 4
    ax.plot_surface(Xc, Yc+p0[5][0,1], Zc+p0[5][0,2], alpha=0.2, rstride=rstride, cstride=cstride)
    ax.plot_surface(Xc, -Yc+p0[5][0,1], Zc+p0[5][0,2], alpha=0.2, rstride=rstride, cstride=cstride)

    Tp = np.array([1., 2., 0.5, 1., 1., 1., 1.,1.,1.])
    # spline interpolations, min-snap splines
    for ii in range(len(p0)-1):
        T = Tp[ii]
        if (ii==4):
            coeff = Interp_corridor(p0[ii], p0[ii+1], T, 15) # interp between p0 and p1
        else:
            coeff = Interp(p0[ii], p0[ii+1], T, 8) # interp between p0 and p1
        t = np.linspace(0, 1, 100, endpoint=True)
        for i in range(t.size):
           pk = ExtractVal(coeff, t[i], 1)
           ax.plot([pk[0,0]], [pk[0,1]], [pk[0,2]], 'g.')


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    



