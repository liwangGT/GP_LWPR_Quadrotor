#!/usr/bin/env python
import sys
import math
from math import pi as PI
from math import atan2, sin, cos, sqrt, fabs, floor, exp
from scipy.misc import factorial as ft
from scipy.misc import comb as nCk
from numpy import linalg as LA
from numpy.linalg import inv
import GPy

import matplotlib.patches
import matplotlib.pyplot as plt

import time 
import numpy as np
from cvxopt import matrix, solvers
import pickle #save multiple objects
from multiprocessing import Pool # multi-prcess for loop


np.random.seed(101)


N = 50
noise_var = 0.05

X = np.linspace(0,10,50)[:,None]
k = GPy.kern.RBF(1)
y = np.random.multivariate_normal(np.zeros(N),k.K(X)+np.eye(N)*np.sqrt(noise_var)).reshape(-1,1)



m_full = GPy.models.GPRegression(X,y)
m_full.optimize('bfgs')
m_full.plot()
print m_full



Z = np.hstack((np.linspace(2.5,4.,3),np.linspace(7,8.5,3)))[:,None]
m = GPy.models.SparseGPRegression(X,y,Z=Z)
m.likelihood.variance = noise_var
m.plot()
print m



m.inducing_inputs.fix()
m.optimize('bfgs')
m.plot()
print m


m.randomize()
m.Z.unconstrain()
m.optimize('bfgs')
m.plot()






