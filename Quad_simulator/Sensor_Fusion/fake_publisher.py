#!/usr/bin/env python
import tf
from tf import TransformListener
import roslib, rospy
from control import acker
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import sys, os
import math
from math import pi as PI
from math import atan2, sin, cos, sqrt, fabs, floor
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

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from crazyflie_demo.msg import Kalman_pred
from sensor_msgs.msg import Imu

if __name__ == '__main__':
    # init all params, publisers, subscrivers
    rospy.init_node('cf_pub', anonymous = True)
    N = rospy.get_param("~cfnumber", 2)  # total number of cfs
    N = 1
    cwd = os.getcwd()
    print cwd
    for i in range(N):
        pub_imu_data = '/crazyflie%d/imu' %i
        pub_pos_data = '/crazyflie%d/pos' %i
    pubimu = rospy.Publisher(pub_imu_data, Imu, queue_size=1)
    pubpos = rospy.Publisher(pub_pos_data, PoseStamped, queue_size=1)

    # load experimental data
    f = open('/home/li/quad_ws/src/crazyflie_ros-master/crazyflie_demo/scripts/cf1_Diff_Flat20170421fbimu10.pckl')
    t_hist, p_hist, phat_hist, ptrack_hist, u_hist, uhat_hist, cmd_hist, cmdreal_hist, rpytrack_hist, imu_hist = pickle.load(f)
    f.close()

    # Optitrack data and time step, convert from dict to np.array
    N = len(t_hist)        # number of data points, data rate 50Hz
    t = np.zeros((N,))
    dt = 0.02
    # tracking data from Optitrack
    x  = np.zeros((N,))    # xpos
    y  = np.zeros((N,))    # ypos
    z  = np.zeros((N,))    # zpos
    xvest = np.zeros((N,))   # xvel
    yvest = np.zeros((N,))   # yvel
    zvest = np.zeros((N,))   # zvel
    xaest = np.zeros((N,))   # xacc
    yaest = np.zeros((N,))   # yacc
    zaest = np.zeros((N,))   # zacc
    # reference data from path planning
    xr = np.zeros((N,))    # xpos
    yr = np.zeros((N,))    # ypos
    zr = np.zeros((N,))    # zpos
    xvr = np.zeros((N,))   # xvel
    yvr = np.zeros((N,))   # yvel
    zvr = np.zeros((N,))   # zvel
    xar = np.zeros((N,))   # xacc
    yar = np.zeros((N,))   # yacc
    zar = np.zeros((N,))   # zacc
    xvrest = np.zeros((N,))   # xvel
    yvrest = np.zeros((N,))   # yvel
    zvrest = np.zeros((N,))   # zvel
    xarest = np.zeros((N,))   # xacc
    yarest = np.zeros((N,))   # yacc
    zarest = np.zeros((N,))   # zacc
    imu = np.zeros((N,3))   # zacc

    # visualize tracking data and reference data
    posdata = PoseStamped()
    imudata = Imu()

    for i in range(N):
        tt0 = rospy.Time.now()
        x[i]  = ptrack_hist[i][0]
        y[i]  = ptrack_hist[i][1]
        z[i]  = ptrack_hist[i][2]
        imu[i] = imu_hist[i] #note x data needs to be flipped
        imudata.header.stamp = rospy.Time.now()
        imudata.linear_acceleration.x = imu[i,0]
        imudata.linear_acceleration.y = imu[i,1]
        imudata.linear_acceleration.z = imu[i,2]
        posdata.header.stamp = rospy.Time.now()
        posdata.pose.position.x = x[i]
        posdata.pose.position.y = y[i]
        posdata.pose.position.z = z[i]
        pubimu.publish(imudata)
        pubpos.publish(posdata)
        while (rospy.Time.now()-tt0).to_sec() < dt:
            aa = 1

