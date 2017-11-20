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
from SSGP_Uncertainty.ssgpy import SSGPy as SSGP  # change if ssgpy.so is in another directory


if __name__ == '__main__':
    # load sim data, x is 9 dimensional, y is 3 dimensional
    #f = open('genSimData/Sim_ground_truth01.pckl')
    f = open('Sim_data/genSimData/Sim_ground_truth02.pckl')
    dt, xhist, yhist, yreal =  pickle.load(f)
    f.close()
    
    # time sequence
    tN = len(xhist[:,0])
    t = np.linspace(0, dt*(tN-1), tN)
    print "Number of data points is: ", tN




    """
    Method 0: full GP
    """
    # get optimal parameters using GPy
    kernel0 = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1., ARD=True) + GPy.kern.White(1)
    m0 = GPy.models.GPRegression(xhist,yhist[:,0][:,None],kernel0)
    m0.optimize()
    # optimized params [ variance 20.0, l0:3.74, l1:0.95; l2:1122.7; l3:7.31; 
    #                    l4:5.57; l5:25.4; l6:761.1; l7:0.64; l8:1.0 ]

    kernel1 = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1., ARD=True) + GPy.kern.White(1)
    m1 = GPy.models.GPRegression(xhist,yhist[:,1][:,None],kernel1)
    m1.optimize()
    # optimized params [ variance 133.3, l0:489.9, l1:61.0; l2:313.9; l3:542.7; 
    #                    l4:604.9; l5:56.4; l6:1.28; l7:29.4; l8:1.0 ]

    kernel2 = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1., ARD=True) + GPy.kern.White(1)
    m2 = GPy.models.GPRegression(xhist,yhist[:,2][:,None],kernel2)
    m2.optimize()
    # optimized params [ variance 5.97, l0:0.57, l1:141.7; l2:0.26; l3:215.0; 
    #                    l4:96.8; l5:1.02; l6:0.70; l7:0.46; l8:1.0 ]

    print kernel0
    print kernel0.rbf.lengthscale
    print kernel1
    print kernel1.rbf.lengthscale
    print kernel2
    print kernel2.rbf.lengthscale

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

    """
    Method 3: SSGP with uncertainty
    """
    # n is x dimension, k is y dimension, D is number of frequencies, ell is 
    D = 25
    sf20 = kernel0.rbf.variance
    ell0 = kernel0.rbf.lengthscale
    sn20 = kernel0.white.variance
    ssgp0 = SSGP(9, 1, D, ell0, sf20, sn20)

    sf21 = kernel1.rbf.variance
    ell1 = kernel1.rbf.lengthscale
    sn21 = kernel1.white.variance
    ssgp1 = SSGP(9, 1, D, ell1, sf21, sn21)

    sf22 = kernel2.rbf.variance
    ell2 = kernel2.rbf.lengthscale
    sn22 = kernel2.white.variance
    ssgp2 = SSGP(9, 1, D, ell2, sf22, sn22)


    start_update = time.time()
    ssgp0.update(xhist, yhist[:,0][:,None])
    ssgp1.update(xhist, yhist[:,1][:,None])
    ssgp2.update(xhist, yhist[:,2][:,None])
    end_update = time.time()

    start_pred = time.time()
    pred0 = ssgp0.predict_mean(xhist)
    pred1 = ssgp1.predict_mean(xhist)
    pred2 = ssgp2.predict_mean(xhist)
    yM3 = np.hstack((pred0[:,None], pred1[:,None], pred2[:,None]))
    end_pred = time.time()

    """
    plt.plot(range(len(pred)), pred, 'bo')
    plt.plot(range(len(yhist[:,0])), yhist[:,0], 'ro')
    fullgp = matplotlib.patches.Patch(color='green', label='Full GP')
    sparsegp = matplotlib.patches.Patch(color='blue', label='SSGP')
    actual = matplotlib.patches.Patch(color='red', label='Actual')
    plt.legend([fullgp, sparsegp, actual], ['Full GP', 'SSGP', 'Actual'])
    plt.show()

    print "{0} ms to perform {1} updates".format((end_update - start_update) *
                                                 1000, Ntrh)
    print "{0} ms per update".format((end_update - start_update) * 1000 / Ntrh)

    print "{0} ms to perform {1} predictions".format((end_pred - start_pred) *
                                                     1000, Nts)
    print "{0} ms per prediction".format((end_pred - start_pred) * 1000 / Nts)
    """





    # visualize data
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(range(0,yhist.shape[0]), yhist[:,0], 'g-')
    axarr[0].plot(range(0,yhist.shape[0]), yreal[:,0], 'r--')
    axarr[0].plot(range(1,yM0.shape[0]+1), yM0[:,0], 'b*')
    axarr[0].plot(range(0,yM3.shape[0]), yM3[:,0], 'k+')

    axarr[1].plot(range(0,yhist.shape[0]), yhist[:,1], 'g-')
    axarr[1].plot(range(0,yhist.shape[0]), yreal[:,1], 'r--')
    axarr[1].plot(range(1,yM0.shape[0]+1), yM0[:,1], 'b*')
    axarr[1].plot(range(0,yM3.shape[0]), yM3[:,1], 'k+')

    axarr[2].plot(range(0,yhist.shape[0]), yhist[:,2], 'g-')
    axarr[2].plot(range(0,yhist.shape[0]), yreal[:,2], 'r--')
    axarr[2].plot(range(1,yM0.shape[0]+1), yM0[:,2], 'b*')
    axarr[2].plot(range(0,yM3.shape[0]), yM3[:,2], 'k+')



   
    plt.show()
