#!/usr/bin/env python
from control import acker
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import sys
from math import pi as PI
from math import atan2, sin, cos, sqrt, fabs, floor, exp
from scipy.misc import factorial as ft
from scipy.misc import comb as nCk
from numpy import linalg as LA
from numpy.linalg import inv

import matplotlib.patches
import matplotlib.pyplot as plt
import GPy

import time 
import numpy as np
from cvxopt import matrix, solvers
import pickle #save multiple objects

#from geometry_msgs.msg import Twist
import Recursive_GP.GP_pred as gp_cf


if __name__ == '__main__':
    # simulate signal
    # input is 9 dimensional, output is 3 dimensional
    N = 100
    t = np.arange(0, 5, 5.0/N)
    yhist = np.empty((0,3))
    ypredhist = np.empty((0,3))
    xall = np.vstack((np.sin(t), np.cos(t), t**2, t+10, 10*np.exp(-t/10), t**3, 2*t-10, 3*t+3, t**2-t))
    xnew = xall[:,0]
    ynew = np.array([xall[0,0], xall[1,0], xall[2,0]])
    gp = gp_cf.GP_pred(xnew, ynew)
    for i in range(1,N):
        xnew = xall[:,i]
        ynew = np.array([xall[0,i], xall[1,i], xall[2,i]])
        gp.Update(xnew, ynew)
        xpred = np.array([sin(t[i]),cos(t[i]),(t[i])**2, t[i]+10, 10*np.exp(-t[i]/10), t[i]**3, 2*t[i]-10, 3*t[i]+3, t[i]**2-t[i]])
        ypredict, Vy = gp.Predict(xpred)
        yhist = np.vstack((yhist, ynew[None,:]))
        ypredhist = np.vstack((ypredhist, ypredict[None,:]))


    # get optimal parameters using GPy
    kernel = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1.)
    yall = np.hstack((gp.y1[:,None], gp.y2[:,None], gp.y3[:,None]))
    m = GPy.models.GPRegression(gp.xx,gp.y1[:,None],kernel)
    print m
    print m.rbf.gradient
    m.optimize()
    print m.rbf.gradient
    print m.rbf.variance
    print m.rbf.lengthscale
    print m.Gaussian_noise.variance
   



    plt.close('all')
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(range(0,yhist.shape[0]), yhist[:,0])
    axarr[0].plot(range(0,ypredhist.shape[0]), ypredhist[:,0], '*')

    axarr[1].plot(range(0,yhist.shape[0]), yhist[:,1])
    axarr[1].plot(range(0,ypredhist.shape[0]), ypredhist[:,1], '*')

    axarr[2].plot(range(0,yhist.shape[0]), yhist[:,2])
    axarr[2].plot(range(0,ypredhist.shape[0]), ypredhist[:,2], '*')
    plt.show()
