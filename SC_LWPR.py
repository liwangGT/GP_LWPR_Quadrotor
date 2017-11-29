#!/usr/bin/env python
import numpy as np
from lwpr import *
from random import *

import sys
import matplotlib.patches
import matplotlib.pyplot as plt
import pickle #load/save multiple objects



if __name__ == '__main__':
    # load sim data, x is 9 dimensional, y is 3 dimensional
    #f = open('genSimData/Sim_ground_truth01.pckl')
    f = open('Sim_data/genSimData/Sim_ground_truth01.pckl')
    dt, xhist, yhist, yreal =  pickle.load(f)
    f.close()
    
    print xhist.dtype
    print yhist.dtype

    xhist_d = np.longdouble(xhist)
    yhist_d = np.longdouble(yhist)

    print xhist_d.dtype
    print yhist_d.dtype

    # time sequence
    tN = len(xhist[:,0])
    t = np.linspace(0, dt*(tN-1), tN)
    print "Number of data points is: ", tN



    """
    Method 5: Locally Weighted Projection Regression (LWPR)
    """
    # initialize 3 models (each with input dim = 9, output dim = 1)
    model_xdd = LWPR(9,1)
    model_xdd.init_D = 20*np.eye(9)
    model_xdd.update_D = True
    model_xdd.init_alpha = 40*np.ones([9,9])
    model_xdd.meta = False

    model_ydd = LWPR(9,1)
    model_ydd.init_D = 20*np.eye(9)
    model_ydd.update_D = True
    model_ydd.init_alpha = 40*np.ones([9,9])
    model_ydd.meta = False

    model_zdd = LWPR(9,1)
    model_zdd.init_D = 20*np.eye(9)
    model_zdd.update_D = True
    model_zdd.init_alpha = 40*np.ones([9,9])
    model_zdd.meta = False

    # initialize prediction
    yp = np.zeros([tN, 3])

    # train model
    for k in range(20):
        ind = np.random.permutation(tN) 

        for i in range(tN):
            print xhist_d[ind[i]].dtype
            print np.array([yhist_d[ind[i],0]]).dtype
            
            yp[ind[i],0] = model_xdd.update(xhist[ind[i]], np.array([yhist[ind[i],0]]))
            yp[ind[i],1] = model_ydd.update(xhist[ind[i]], np.array([yhist[ind[i],1]]))
            yp[ind[i],2] = model_zdd.update(xhist[ind[i]], np.array([yhist[ind[i],2]]))
            
    # visualize data
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(range(0,yhist.shape[0]), yhist[:,0], 'g-', label='Data')
    axarr[0].plot(range(0,yhist.shape[0]), yreal[:,0], 'r--', label='Real')
    axarr[0].plot(range(1,yM5.shape[0]+1), yM5[:,0], 'k.', label='LWPR')
    axarr[0].legend()
    axarr[0].set_title('Acc X')

    axarr[1].plot(range(0,yhist.shape[0]), yhist[:,1], 'g-', label='Data')
    axarr[1].plot(range(0,yhist.shape[0]), yreal[:,1], 'r--', label='Real')
    axarr[1].plot(range(1,yM5.shape[0]+1), yM5[:,1], 'k.', label='LWPR')
    axarr[1].legend()
    axarr[1].set_title('Acc Y')

    axarr[2].plot(range(0,yhist.shape[0]), yhist[:,2], 'g-', label='Data')
    axarr[2].plot(range(0,yhist.shape[0]), yreal[:,2], 'r--', label='Real')
    axarr[2].plot(range(1,yM5.shape[0]+1), yM5[:,2], 'k.', label='LWPR')
    axarr[2].legend()
    axarr[2].set_title('Acc Z')
    plt.show()
