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

def cal_R(c1, c2, c3): # use angle in radiance
    R = np.array([[cos(c1)*cos(c3), sin(c2)*sin(c1)*cos(c3)-cos(c2)*sin(c3), cos(c2)*sin(c1)*cos(c3)+sin(c2)*sin(c3)],
         [cos(c1)*sin(c3), sin(c2)*sin(c1)*sin(c3)+cos(c2)*cos(c3), cos(c2)*sin(c1)*sin(c3)-sin(c2)*cos(c3)],
         [-sin(c1), sin(c2)*cos(c1), cos(c2)*cos(c1)]])
    return R

def cal_Rd(c1, c2,c3): # roll, pitch, yaw
    Rd = np.array([[1, sin(c1)*tan(c2), cos(c1)*tan(c2)],[0, cos(c1), -sin(c1)],[0, sin(c1)/cos(c2), cos(c1)/cos(c2)]])
    return Rd

class Kalmanu():
    def __init__(self, xinit):
        self.Q = np.array([[1e-8, 1e-12, 1e-12],[1e-12, 1e-8, 1e-12],[1e-12, 1e-12, 1e-2]]) # dynamic noise
        self.R = np.array([[1e-6, 1e-12],[1e-12,1e-1]]) # measurement noise
        #self.Q = np.array([[1e-4, 1e-8, 1e-8],[1e-8, 1e-4, 1e-8],[1e-8, 1e-8, 1e-1]]) # dynamic noise
        #self.R = np.array([1e-10]) # measurement noise
        self.S = self.Q # state estimate noise
        self.X = xinit

    def update(self, newdata):
        dt = 0.02
        Phi = np.array([[1, dt, 0],[0, 1, dt],[0,0,1]]) # state transition matrix
        M   = np.array([[1,0,0],[0,0,1]]) # measurement matrix
        # prediction step
        self.X = np.dot(Phi,self.X)
        self.S = Phi.dot(self.S).dot(Phi.transpose()) + self.Q
        # Kalman gain
        Kt = self.S.dot(M.transpose()).dot(inv( M.dot(self.S).dot(M.transpose())+self.R ))

        # update
        self.X = self.X + Kt.dot(newdata-M.dot(self.X))
        self.S = self.S - Kt.dot(M.dot(self.S))
        return self.X



if __name__ == "__main__":
    # load exp data
    f = open('data/cf1_3rd_data_04272018_v3.pckl')
    t_hist, p_hist, phat_hist, ptrack_hist, u_hist, uhat_hist, cmd_hist, cmdreal_hist, rpytrack_hist, imu_hist, gyro_hist, thrust_hover = pickle.load(f)

   # preprocessing all data into (x,y) format
    N = len(t_hist)
    m = 35.89/1000 #kg
    g = 9.8 #kg.m.s2
    # nominal thrust instead!!
    #thrust_hover = 49022
    #thrust_hover = 46189.5107422
    #f1, axarr1 = plt.subplots(2, sharex=True)
    imu = np.empty((0,3))
    gyro = np.empty((0,3))
    r = np.empty((0,3))
    rhat = np.empty((0,3))
    rd = np.empty((0,3))
    rdd = np.empty((0,3))
    angle = np.empty((0,3))
    angled = np.empty((0,3))
    cmd_angle = np.empty((0,3))
    cmd_angleReal = np.empty((0,3))
    fzReal = np.empty((0,1))
    y1all = np.empty((0,3))
    y1all_model = np.empty((0,3))
    y2all = np.empty((0,3))
    yexp = np.empty((0,3))
    for i in range(N):
        #print  np.array([ptrack_hist[i][0], ptrack_hist[i][1], ptrack_hist[i][2]])
        #r = np.vstack((ptrack_hist[i][None,:]))
        angle = np.vstack((angle, 180.0/PI*np.array([rpytrack_hist[i][0], rpytrack_hist[i][1], rpytrack_hist[i][2]])))
        cmd = cmd_hist[i]
        cmdReal = cmdreal_hist[i]
        cmd_angle = np.vstack((cmd_angle, np.array([cmd[0],-cmd[1],0])))
        cmd_angleReal = np.vstack((cmd_angleReal, np.array([cmdReal.linear.y,-cmdReal.linear.x,0])))
        fzReal = np.vstack((fzReal, cmdReal.linear.z))
        imu = np.vstack((imu, imu_hist[i][None,:]))
        gyro = np.vstack((gyro, gyro_hist[i][None,:]))
        r = np.vstack((r, np.array([ptrack_hist[i][0], ptrack_hist[i][1], ptrack_hist[i][2]])))
        rhat = np.vstack((rhat, np.array([phat_hist[i][0,0], phat_hist[i][0,1], phat_hist[i][0,2]])))
        #yexp = np.vstack((yexp, np.array([pred_hist[i][0],pred_hist[i][1],pred_hist[i][2]])))

    for i in range(N):
        if i == 0:
           xKF = Kalmanu(np.array([ptrack_hist[i][0],phat_hist[i][1,0],phat_hist[i][2,0]]))
           yKF = Kalmanu(np.array([ptrack_hist[i][1],phat_hist[i][1,1],phat_hist[i][2,1]]))
           zKF = Kalmanu(np.array([ptrack_hist[i][2],phat_hist[i][1,2],phat_hist[i][2,2]]))
           rd  = np.vstack((rd,np.array([[0,0,0]])))
           rdd = np.vstack((rdd,np.array([[0,0,0]])))
           angled = np.vstack((angled,np.array([[0,0,0]])))
        else:
           xpre = xKF.update(np.array([ptrack_hist[i][0], imu[i,0]]))
           ypre = yKF.update(np.array([ptrack_hist[i][1], imu[i,1]]))
           zpre = zKF.update(np.array([ptrack_hist[i][2], imu[i,2]]))
           #axarr1[0].plot(i,xpre[2],'r.')
           #axarr1[0].plot(i,ypre[2],'g.')
           #axarr1[0].plot(i,zpre[2],'b.')
           rd  = np.vstack((rd,np.array([[xpre[1],ypre[1],zpre[1]]])))
           rdd = np.vstack((rdd,np.array([[xpre[2],ypre[2],zpre[2]]])))
           angled = np.vstack((angled,  (angle[i]-angle[i-1])/0.02))
        R = cal_R(rpytrack_hist[i][1], rpytrack_hist[i][0], rpytrack_hist[i][2])
        # TODO: determine what is thrust_hover, note fzReal should be NEGATIVE
        y1all = np.vstack((y1all, rdd[i,:]-np.array([0,0,g])+fzReal[i]/thrust_hover*g*np.array([cos(angle[i][0]/180*PI)*sin(angle[i][1]/180*PI), -sin(angle[i][0]/180*PI), cos(angle[i][0]/180*PI)*cos(angle[i][1]/180*PI)]) ))
        y1all_model = np.vstack((y1all_model, np.array([0,0,g])-fzReal[i]/thrust_hover*g*np.array([cos(angle[i][0]/180*PI)*sin(angle[i][1]/180*PI), -sin(angle[i][0]/180*PI), cos(angle[i][0]/180*PI)*cos(angle[i][1]/180*PI)]) ))
    # y2 is angle(measured) - angle(model_predicted)
    y2all = angle - cmd_angleReal

    # Simulate GP prediction
    print r.shape, rd.shape, angle.shape, rdd.shape, angled.shape, y1all.shape, y2all.shape
    xall = np.hstack((r,rd,angle))
    # calculate yall, which is ydd-gzw-1/m*R*zw*fz
    # first try an easier way: angle - cmd_angleReal
    yall = np.hstack((y1all, y2all))


    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(range(0,r.shape[0]), r[:,0], 'g-', label='Data')
    axarr[0].plot(range(0,rhat.shape[0]), rhat[:,0], 'r--', label='Ref')
    axarr[0].legend()
    axarr[0].set_title('pos X')

    axarr[1].plot(range(0,r.shape[0]), r[:,1], 'g-', label='Data')
    axarr[1].plot(range(0,rhat.shape[0]), rhat[:,1], 'r--', label='Ref')
    axarr[1].legend()
    axarr[1].set_title('pos Y')

    axarr[2].plot(range(0,r.shape[0]), r[:,2], 'g-', label='Data')
    axarr[2].plot(range(0,rhat.shape[0]), rhat[:,2], 'r--', label='Ref')
    axarr[2].legend()
    axarr[2].set_title('pos Z')

    plt.show()
