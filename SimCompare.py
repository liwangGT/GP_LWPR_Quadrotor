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
    # load sim data, x is 9 dimensional, y is 3 dimensional
    #f = open('genSimData/Sim_ground_truth01.pckl')
    f = open('Sim_data/genSimData/Sim_ground_truth02.pckl')
    dt, xhist, yhist, yreal =  pickle.load(f)
    f.close()
    
    # time sequence
    tN = len(xhist[:,0])
    t = np.linspace(0, dt*(tN-1), tN)

    """
    Method 0: full GP
    """
    # get optimal parameters using GPy
    kernel0 = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1., ARD=True)
    m0 = GPy.models.GPRegression(xhist,yhist[:,0][:,None],kernel0)
    m0.optimize()

    kernel1 = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1., ARD=True)
    m1 = GPy.models.GPRegression(xhist,yhist[:,1][:,None],kernel1)
    m1.optimize()

    kernel2 = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1., ARD=True)
    m2 = GPy.models.GPRegression(xhist,yhist[:,2][:,None],kernel2)
    m2.optimize()


    print kernel0
    print kernel0.lengthscale
    print kernel1
    print kernel1.lengthscale
    print kernel2
    print kernel2.lengthscale

    #print m
    #print m.rbf.gradient
    #m.optimize()
    #print m.rbf.gradient
    #print m.rbf.variance
    #print m.rbf.lengthscale
    #print m.Gaussian_noise.variance


    """
    Method 1: recursive GP
    """
    yM0 = np.empty((0,3))
    yM1 = np.empty((0,3))
    xnew = xhist[0,:]
    ynew = yhist[0,:]
    gpR  = gp_cf.GP_pred(xnew, ynew)
    for i in range(1,tN):
        # make predictions
        xnew = xhist[i,:]
        ypred, Vy = gpR.Predict(xnew)

        # update model with actual data
        ynew = yhist[i,:]
        gpR.Update(xnew, ynew)

        # store predictions for comparision        
        yM1 = np.vstack((yM1, ypred))        

        # full GP prediction
        ym0,yV0 = m0.predict(xnew[None,:])
        ym1,yV1 = m1.predict(xnew[None,:])
        ym2,yV2 = m2.predict(xnew[None,:])
        print ypred
        ypred = np.array([ym0, ym1, ym2]).reshape((3,))
        print ypred
        yM0 = np.vstack((yM0, ypred))

    # visualize data
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(range(0,yhist.shape[0]), yhist[:,0], 'g-')
    axarr[0].plot(range(0,yhist.shape[0]), yreal[:,0], 'r--')
    axarr[0].plot(range(0,yM0.shape[0]), yM0[:,0], 'b*')

    axarr[1].plot(range(0,yhist.shape[0]), yhist[:,1], 'g-')
    axarr[1].plot(range(0,yhist.shape[0]), yreal[:,1], 'r--')
    axarr[1].plot(range(0,yM0.shape[0]), yM0[:,1], 'b*')

    axarr[2].plot(range(0,yhist.shape[0]), yhist[:,2], 'g-')
    axarr[2].plot(range(0,yhist.shape[0]), yreal[:,2], 'r--')
    axarr[2].plot(range(0,yM0.shape[0]), yM0[:,2], 'b*')




   
    plt.show()
