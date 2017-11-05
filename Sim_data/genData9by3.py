#!/usr/bin/env python
from mpl_toolkits.mplot3d import Axes3D
import sys
from math import pi as PI
from math import atan2, sin, cos, sqrt, fabs, floor, exp
from scipy.misc import factorial as ft
from scipy.misc import comb as nCk
from numpy import linalg as LA
from numpy.linalg import inv

import matplotlib.patches
import matplotlib.pyplot as plt

import time 
import numpy as np
from cvxopt import matrix, solvers
import pickle #save multiple objects




if __name__ == "__main__":
    """
    This script generates simulated data points for quadrotor 
    The dimension of input X is 9 (x,y,z,xd,yd,zd,roll,pitch,yaw); 
    The dimension of output is 3 (xdd, ydd, zdd);
    """
    Ntrain = 100
    Ntest = 100
    # generate xyz, xdydzd data
    Xtrain = np.hstack((4*np.random.rand(Ntrain, 6) - 2, PI/3*2*np.random.rand(Ntrain, 3) - PI/3))  
    Ytrain = np.hstack((np.sin(Xtrain[:,6]) + Xtrain[:,1]*0.3 , np.cos(Xtrain[:,7]) +   )) +  np.random.randn(Ntrain, 9) * 0.25

    Xtest = np.hstack((4*np.random.rand(Ntest, 6) - 2, PI/3*2*np.random.rand(Ntest, 3) - PI/3))  
    Yactual = 1 

    



    # save data into pckl file
    f = open('quadSimData9by5.pckl', 'w')
    pickle.dump([Xtrain, Ytrain, Xtest, Yactual], f)


