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

# arg parsers
import argparse

# batch load data files
import glob


if __name__ == "__main__":
    filelist =  glob.glob("*.pckl")
    filelist = filelist[0:6]
    for filename in filelist:
        print("Loading: " + filename)
        with open(filename, "r") as f:
            xhist, yhist, yreal, yM0, yM2, yM3 = pickle.load(f)
        
        # plot the "wind field in 3D"
        fig = plt.figure()
        ax  = fig.gca(projection = '3d')
        ax.set_aspect('equal')
        ax.invert_zaxis()


        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([3, 2, 1.6]).max()
        Xb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][0].flatten()
        Yb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][1].flatten() 
        Zb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][2].flatten() - 1
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')
        
        # plot trajectory
        ax.plot(xhist[:,0], xhist[:,1], xhist[:,2], label="Flight trajectory")

        # plot vector field
        Lv = 0.2
        for i in range(1,len(yreal)):
            ax.plot([xhist[i,0], xhist[i,0]+Lv*yM0[i-1,0]], [xhist[i,1], xhist[i,1]+Lv*yM0[i-1,1]], [xhist[i,2], xhist[i,2]+Lv*yM0[i-1,2]], 'g')
        ax.legend()
 
    plt.show()

    # TODO: save GP params, plot wind field in 3D




