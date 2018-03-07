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
    for filename in filelist:
        print("Loading: " + filename)
        with open(filename, "wb") as f:
            xhist, yhist, yreal, yM0, yM2, yM3 = pickle.load(f)
        
        # plot the "wind field in 3D"
        fig = plt.figure()
        ax  = fig.gca(projection = '3d')
        ax.set_aspect('equal')
        ax.invert_zaxis()
        
        ax.plot(yhist[:,0], yhist[:,0], yhist)




