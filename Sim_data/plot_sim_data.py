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

# add quad simulator path
sys.path.insert(0, '/home/li/gitRepo/GP_Quadrotor/Quad_simulator/Interp')
from Spline_interp import *
from Quad_visual import *


if __name__ == "__main__":
    # load sim data
    #f = open('genSimData/Sim_ground_truth01.pckl')
    f = open('genSimData/Sim_ground_truth02.pckl')
    dt, xhist, yhist, yreal =  pickle.load(f)
    f.close()
    
    print xhist[:,0].size, yreal[:,0].size
    # time sequence
    tN = len(xhist[:,0])
    t = np.linspace(0, dt*(tN-1), tN)

    # visualize yhist and yreal
    fig = plt.figure(1)
    plt.subplot(311)
    plt.plot(t, yreal[:,0], 'r-', label='acc x')
    plt.plot(t, yhist[:,0], 'g-', label='acc x')

    plt.subplot(312)
    plt.plot(t, yreal[:,1], 'r-', label='acc y')
    plt.plot(t, yhist[:,1], 'g-', label='acc y')

    plt.subplot(313)
    plt.plot(t, yreal[:,2], 'r-', label='acc z')
    plt.plot(t, yhist[:,2], 'g-', label='acc z')
    plt.show()
