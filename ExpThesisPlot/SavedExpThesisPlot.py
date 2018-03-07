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
    #f = open('Quadrotor_data/cf1_3rd_data_09202017_PIDdrift_v1GPWind_XY.pckl')
    f = open('ExpDataSave.pckl')
    yhist, yreal, yM0, yM2, yM3 =  pickle.load(f)
    f.close()

    N0 = 200
    N1 = 550
    yhist = yhist[N0:N1]
    yreal = yreal[N0:N1]
    yM0 = yM0[N0:N1]
    yM2 = yM2[N0:N1]
    yM3 = yM3[N0:N1]

    # visualize data
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    axarr[0].plot(range(0,yhist.shape[0]), yhist[:,0], 'g-', label='Data')
    #axarr[0].plot(range(0,yhist.shape[0]), yreal[:,0], 'r--', label='Real')
    axarr[0].plot(range(1,yM0.shape[0]+1), yM0[:,0], 'b*', label='FullGP')
    #axarr[0].plot(range(1,yM1.shape[0]+1), yM1[:,0], 'r.', label='RecurGP')
    axarr[0].plot(range(1,yM2.shape[0]+1), yM2[:,0], 'k+', label='SSGP')
    #axarr[0].plot(range(1,yM3.shape[0]+1), yM3[:,0], 'kv', label='SSGP Induce')
    #axarr[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axarr[0].set_title('Accelation X')
    axarr[0].set_ylabel(r'Acc $(m/s^2)$')

    axarr[1].plot(range(0,yhist.shape[0]), yhist[:,1], 'g-', label='Data')
    #axarr[1].plot(range(0,yhist.shape[0]), yreal[:,1], 'r--', label='Real')
    axarr[1].plot(range(1,yM0.shape[0]+1), yM0[:,1], 'b*', label='FullGP')
    #axarr[1].plot(range(1,yM1.shape[0]+1), yM1[:,1], 'r.', label='RecurGP')
    axarr[1].plot(range(1,yM2.shape[0]+1), yM2[:,1], 'k+', label='SSGP')
    #axarr[1].plot(range(1,yM3.shape[0]+1), yM3[:,1], 'kv', label='SSGP Induce')
    #axarr[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axarr[1].set_title('Accelation Y')
    axarr[1].set_ylabel(r'Acc $(m/s^2)$')

    axarr[2].plot(range(0,yhist.shape[0]), yhist[:,2], 'g-', label='Flight Data')
    #axarr[2].plot(range(0,yhist.shape[0]), yreal[:,2], 'r--', label='Real')
    axarr[2].plot(range(1,yM0.shape[0]+1), yM0[:,2], 'b*', label='FullGP')
    #axarr[2].plot(range(1,yM1.shape[0]+1), yM1[:,2], 'r.', label='RecurGP')
    axarr[2].plot(range(1,yM2.shape[0]+1), yM2[:,2], 'k+', label='SSGP')
    #axarr[2].plot(range(1,yM3.shape[0]+1), yM3[:,2], 'kv', label='SSGP Induce')
    lgd = axarr[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=3)
    axarr[2].set_title('Accelation Z')
    axarr[2].set_ylabel(r'Acc $(m/s^2)$')
    axarr[2].set_xlabel(r'Time step')
    plt.savefig('expflight.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


    """
    print "{0} ms to perform {1} updates".format((end_update - start_update) *
                                                 1000, Ntrh)
    print "{0} ms per update".format((end_update - start_update) * 1000 / Ntrh)
    print "{0} ms to perform {1} predictions".format((end_pred - start_pred) *
                                                     1000, Nts)
    print "{0} ms per prediction".format((end_pred - start_pred) * 1000 / Nts)
    """
    print '----data logging completed!!!----'

