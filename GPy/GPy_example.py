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




if __name__ == '__main__':
    # sample inputs and outputs
    X = np.random.uniform(-3.,3.,(50,2))
    #Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05
    Y = np.sin(X[:,0:1]) + 0.01*np.sin(X[:,1:2])+np.random.randn(50,1)*0.05

    # define kernel
    # ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
    ker2 = GPy.kern.RBF(input_dim=2, ARD=True)

    # create simple GP model
    m = GPy.models.GPRegression(X,Y,ker2)

    # optimize and plot
    #m.optimize(messages=True,max_f_eval = 1000)

    print m
    print m.rbf.gradient
    m.optimize()
    print m.rbf.gradient
    print m
    #print m.rbf.variance
    print m.rbf.lengthscale
    #print m.Gaussian_noise.variance
