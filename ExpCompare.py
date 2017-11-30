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
    # load exp data
    f = open('Quadrotor_data/cf1_3rd_data_09202017_PIDdrift_v1GPWind_XY.pckl')
    dt, xhist, yhist, yreal =  pickle.load(f)
    f.close()

    # time sequence
    tN = len(xhist[:,0])
    t = np.linspace(0, dt*(tN-1), tN)
    print "Number of data points is: ", tN

    # store prediciton data for methods: 
    # 0-fullGP, 1-recurGP, 2-SSGPwUncertainty, 
    # 3-SparseGPinduce, 4-SparseSpectrumGP, 5-LWPR
    yM0 = np.empty((0,3))
    yM1 = np.empty((0,3))
    yM2 = np.empty((0,3))
    yM3 = np.empty((0,3))
    yM4 = np.empty((0,3))
    yM5 = np.empty((0,3))

    # only use 500 data points for full GP optimization
    Ntrain = min(1500, tN)

    # construct GP regressors for five different methods
    """
    Method 0: full GP
    """
    print "Method 0: full GP"
    # Optimize hyper parameter with full GP (GPy)
    # get optimal parameters using GPy for dimension 0
    kernel0 = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1., ARD=True) + GPy.kern.White(1)
    m0 = GPy.models.GPRegression(xhist[:Ntrain+1, :],yhist[:Ntrain+1,0][:,None],kernel0)
    m0.optimize()

    # get optimal parameters using GPy for dimension 1
    kernel1 = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1., ARD=True) + GPy.kern.White(1)
    m1 = GPy.models.GPRegression(xhist[:Ntrain+1, :],yhist[:Ntrain+1,1][:,None],kernel1)
    m1.optimize()

    # get optimal parameters using GPy for dimension 2
    kernel2 = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1., ARD=True) + GPy.kern.White(1)
    m2 = GPy.models.GPRegression(xhist[:Ntrain+1, :],yhist[:Ntrain+1,2][:,None],kernel2)
    m2.optimize()

    """
    Method 1: recursive GP
    """
    print "Method 1: recursive GP"
    # TODO: split recursive GP into three, each with different ARD params
    xnew = xhist[0,:][None,:]
    ynew = yhist[0,:]
    gpR  = gp_cf.GP_pred(xnew.flatten(), ynew)


    """
    Method 2: SSGP with uncertainty
    """
    print "Method 2: SSGP with uncertainty"
    # n is x dimension, k is y dimension, D is number of frequencies, ell is 
    D = 25
    # construct kernal for dimension 0
    sf20 = kernel0.rbf.variance
    ell0 = kernel0.rbf.lengthscale
    sn20 = kernel0.white.variance
    ssgp0 = SSGP(9, 1, D, ell0, sf20, sn20)

    # construct kernal for dimension 1
    sf21 = kernel1.rbf.variance
    ell1 = kernel1.rbf.lengthscale
    sn21 = kernel1.white.variance
    ssgp1 = SSGP(9, 1, D, ell1, sf21, sn21)

    # construct kernal for dimension 2
    sf22 = kernel2.rbf.variance
    ell2 = kernel2.rbf.lengthscale
    sn22 = kernel2.white.variance
    ssgp2 = SSGP(9, 1, D, ell2, sf22, sn22)
    
    # init with the first data point
    ssgp0.update(xnew, np.array([[ynew[0]]]))
    ssgp1.update(xnew, np.array([[ynew[1]]]))
    ssgp2.update(xnew, np.array([[ynew[2]]]))



    """
    Method 3: SSGP with inducing points
    """
    print "Method 3: SSGP with inducing points"
    # copy kernel parameter
    kernel30 = kernel0.copy()
    kernel31 = kernel1.copy()
    kernel32 = kernel2.copy()
    # optimize inducing points for dimension 0
    m30 = GPy.models.SparseGPRegression(xhist,yhist[:,0][:,None],kernel=kernel30, num_inducing=20)
    m30.Z.unconstrain()
    m30.optimize('bfgs')
    Z30 = m30.inducing_inputs # remember optimized inducing points
    m30ic = GPy.models.SparseGPRegression(xnew,np.array([[ynew[0]]]),Z=Z30, kernel=kernel30)
    m30ic.Z.fix()
    m30ic.kern.fix()

    # optimize inducing points for dimension 1
    m31 = GPy.models.SparseGPRegression(xhist,yhist[:,1][:,None],kernel=kernel31,num_inducing=20)
    m31.Z.unconstrain()
    m31.optimize('bfgs')
    Z31 = m31.inducing_inputs # remember optimized inducing points
    m31ic = GPy.models.SparseGPRegression(xnew,np.array([[ynew[1]]]),Z=Z31, kernel=kernel31)
    m31ic.Z.fix()
    m31ic.kern.fix()

    # optimize inducing points for dimension 2
    m32 = GPy.models.SparseGPRegression(xhist,yhist[:,2][:,None],kernel=kernel32,num_inducing=20)
    m32.Z.unconstrain()
    m32.optimize('bfgs')
    Z32 = m32.inducing_inputs # remember optimized inducing points
    m32ic = GPy.models.SparseGPRegression(xnew,np.array([[ynew[2]]]),Z=Z32, kernel=kernel32)
    #m32ic = GPy.core.SparseGP_MPI(xnew, np.array([[ynew[2]]]), Z=Z32, kernel=kernel32, likelihood=GPy.likelihoods.Gaussian())
    m32ic.Z.fix()
    m32ic.kern.fix()



    """
    Method 4: Sparse Spectrum Gaussian Process Regression
    """
    print "Method 4: SSGP"
    # code is a simpler version of method 2


    """
    Method 5: Locally Weighted Project Regressor (LWPR)
    """
    print "Method 5: LWPR"
    # TODO: add method here



    print "Start incremental prediction"
    # test incremental prediction
    for i in range(1,tN):
        start_update = time.time()

        # get current state
        xnew = xhist[i,:][None,:]

        # make prediction
        # method 0: full GP
        t0_pred_s = time.time()
        ym0,yV0 = m0.predict(xnew)
        ym1,yV1 = m1.predict(xnew)
        ym2,yV2 = m2.predict(xnew)
        ypred = np.array([ym0, ym1, ym2]).reshape((3,))
        yM0 = np.vstack((yM0, ypred))
        t0_pred_e = time.time()

        # method 1: recurGP
        t1_pred_s = time.time()
        ypred, Vy = gpR.Predict(xnew.flatten())
        yM1 = np.vstack((yM1, ypred))
        t1_pred_e = time.time()
        # method 2: SSGP with uncertainty
        t2_pred_s = time.time()
        pred0 = ssgp0.predict_mean(xnew)
        pred1 = ssgp1.predict_mean(xnew)
        pred2 = ssgp2.predict_mean(xnew)
        ypred = np.hstack((pred0, pred1, pred2))
        yM2 = np.vstack((yM2, ypred))
        t2_pred_e = time.time()
        # method 3: SGP with inducing points
        t3_pred_s = time.time()
        ym0,yV0 = m30ic.predict(xnew)
        ym1,yV1 = m31ic.predict(xnew)
        ym2,yV2 = m32ic.predict(xnew)
        ypred = np.array([ym0, ym1, ym2]).reshape((3,))
        yM3 = np.vstack((yM3, ypred))
        t3_pred_e = time.time()
        # method 4: SSGP origin
        # yM4 = np.vstack((yM4, ypred))
        # method 5: LWPR
        # yM5 = np.vstack((yM5, ypred))

        # update model with actual data
        ynew = yhist[i,:]
        # method 0: full GP
        # no need to update, batch prediction
        t0_update_s = time.time()
        t0_update_e = time.time()
        # method 1: recurGP
        t1_update_s = time.time()
        gpR.Update(xnew.flatten(), ynew)
        t1_update_e = time.time()
        # method 2: SSGP with uncertainty
        t2_update_s = time.time()
        ssgp0.update(xnew, np.array([[ynew[0]]]))
        ssgp1.update(xnew, np.array([[ynew[1]]]))
        ssgp2.update(xnew, np.array([[ynew[2]]]))
        t2_update_e = time.time()
        # method 3: SGP with inducing points
        t3_update_s = time.time()
        m30ic.set_XY(xhist[0:i+1,:],yhist[0:i+1,0][:,None])
        m31ic.set_XY(xhist[0:i+1,:],yhist[0:i+1,1][:,None])
        m32ic.set_XY(xhist[0:i+1,:],yhist[0:i+1,2][:,None])
        #m30ic.set_data(xnew,np.array([[ynew[0]]]))
        #m31ic.set_data(xnew,np.array([[ynew[1]]]))
        #m32ic.set_data(xnew,np.array([[ynew[2]]]))
        t3_update_e = time.time()
     
        end_update = time.time()
        print "FullGP predict {0}ms, update {1}ms".format((t0_pred_e - t0_pred_s) *1000, (t0_update_e - t0_update_s) *1000)
        print "RecurGP predict {0}ms, update {1}ms".format((t1_pred_e - t1_pred_s) *1000, (t1_update_e - t1_update_s) *1000)
        print "SSGP uncertain predict {0}ms, update {1}ms".format((t2_pred_e - t2_pred_s) *1000, (t2_update_e - t2_update_s) *1000)
        print "SGP induce predict {0}ms, update {1}ms".format((t3_pred_e - t3_pred_s) *1000, (t3_update_e - t3_update_s) *1000)

    # visualize data
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(range(0,yhist.shape[0]), yhist[:,0], 'g-', label='Data')
    axarr[0].plot(range(0,yhist.shape[0]), yreal[:,0], 'r--', label='Real')
    axarr[0].plot(range(1,yM0.shape[0]+1), yM0[:,0], 'b*', label='FullGP')
    #axarr[0].plot(range(1,yM1.shape[0]+1), yM1[:,0], 'r.', label='RecurGP')
    axarr[0].plot(range(1,yM2.shape[0]+1), yM2[:,0], 'k+', label='SSGP')
    axarr[0].plot(range(1,yM3.shape[0]+1), yM3[:,0], 'kv', label='SSGP Induce')
    axarr[0].legend()
    axarr[0].set_title('Acc X')

    axarr[1].plot(range(0,yhist.shape[0]), yhist[:,1], 'g-', label='Data')
    axarr[1].plot(range(0,yhist.shape[0]), yreal[:,1], 'r--', label='Real')
    axarr[1].plot(range(1,yM0.shape[0]+1), yM0[:,1], 'b*', label='FullGP')
    #axarr[1].plot(range(1,yM1.shape[0]+1), yM1[:,1], 'r.', label='RecurGP')
    axarr[1].plot(range(1,yM2.shape[0]+1), yM2[:,1], 'k+', label='SSGP')
    axarr[1].plot(range(1,yM3.shape[0]+1), yM3[:,1], 'kv', label='SSGP Induce')
    axarr[1].legend()
    axarr[1].set_title('Acc Y')

    axarr[2].plot(range(0,yhist.shape[0]), yhist[:,2], 'g-', label='Data')
    axarr[2].plot(range(0,yhist.shape[0]), yreal[:,2], 'r--', label='Real')
    axarr[2].plot(range(1,yM0.shape[0]+1), yM0[:,2], 'b*', label='FullGP')
    #axarr[2].plot(range(1,yM1.shape[0]+1), yM1[:,2], 'r.', label='RecurGP')
    axarr[2].plot(range(1,yM2.shape[0]+1), yM2[:,2], 'k+', label='SSGP')
    axarr[2].plot(range(1,yM3.shape[0]+1), yM3[:,2], 'kv', label='SSGP Induce')
    axarr[2].legend()
    axarr[2].set_title('Acc Z')
    plt.show()

    """
    print "{0} ms to perform {1} updates".format((end_update - start_update) *
                                                 1000, Ntrh)
    print "{0} ms per update".format((end_update - start_update) * 1000 / Ntrh)
    print "{0} ms to perform {1} predictions".format((end_pred - start_pred) *
                                                     1000, Nts)
    print "{0} ms per prediction".format((end_pred - start_pred) * 1000 / Nts)
    """
