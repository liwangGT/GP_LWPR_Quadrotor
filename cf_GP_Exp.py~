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



if __name__ == '__main__':
    # load experimental data
    f = open('Quadrotor_data/Exp20170920/cf1_3rd_data_09202017_PIDdrift_v0Wind.pckl')
    t_hist, p_hist, phat_hist, ptrack_hist, u_hist, uhat_hist, cmd_hist, cmdreal_hist, rpytrack_hist, imu_hist, gyro_hist, thrust_hover, pred_hist, flag_hist = pickle.load(f)
    N = len(t_hist)
    m = 35.89/1000 #kg
    g = 9.8 #kg.m.s2
    # nominal thrust instead!!
    #thrust_hover = 49022
    #thrust_hover = 46189.5107422
    f1, axarr1 = plt.subplots(2, sharex=True)
    imu = np.empty((0,3))
    gyro = np.empty((0,3))
    r = np.empty((0,3))
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
        yexp = np.vstack((yexp, np.array([pred_hist[i][0],pred_hist[i][1],pred_hist[i][2]])))

    axarr1[0].plot(range(0,N), imu[:,0], 'r-')
    axarr1[0].plot(range(0,N), imu[:,1], 'g-')
    axarr1[0].plot(range(0,N), imu[:,2], 'b-')

    axarr1[1].plot(range(0,N), gyro[:,0], 'r-')
    axarr1[1].plot(range(0,N), gyro[:,1], 'g-')
    axarr1[1].plot(range(0,N), gyro[:,2], 'b-')

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
           axarr1[0].plot(i,xpre[2],'r.')
           axarr1[0].plot(i,ypre[2],'g.')
           axarr1[0].plot(i,zpre[2],'b.')
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
    xnew = xall[0,:]
    ynew = yall[0,:]
    gp = gp_cf.GP_pred(xnew, ynew)
    ypredhist = np.zeros((1,6))




####TODO: use # gp2: variance 0.381, lengthscal 3.42, ONLY run gp1, gp2, gp3 for EXP

###############opt kernal value###########
# gp1: variance 0.57, lengthscal 3.25
# gp2: variance 0.14, lengthscal 1.75
# gp3: variance 9163, lengthscal 40
# gp4: variance 3216.83, lengthscal 17.8
# gp5: variance 10602, lengthscal 24.37
# gp6: variance 255318, lengthscal 3900

    # TODO: plot desired xdd, ydd, zdd from invert Diff

    # Two subplots, the axes array is 1-d
    f3, axarr3 = plt.subplots(3, sharex=True)
    axarr3[0].plot(range(0,yall.shape[0]), yall[:,0], 'r-')
    axarr3[0].plot(range(0,rdd.shape[0]), rdd[:,0], 'g.')
    axarr3[0].plot(range(0,y1all_model.shape[0]), y1all_model[:,0], 'b-')
    #axarr3[0].plot(range(0,ypredhist.shape[0]), ypredhist[:,0], 'y*')
    #axarr3[0].plot(range(0,yexp.shape[0]), yexp[:,0], 'k.')

    axarr3[1].plot(range(0,yall.shape[0]), yall[:,1], 'r-')
    axarr3[1].plot(range(0,rdd.shape[0]), rdd[:,1], 'g.')
    axarr3[1].plot(range(0,y1all_model.shape[0]), y1all_model[:,1], 'b-')
    #axarr3[1].plot(range(0,ypredhist.shape[0]), ypredhist[:,1], 'y*')
    #axarr3[1].plot(range(0,yexp.shape[0]), yexp[:,1], 'k.')

    axarr3[2].plot(range(0,yall.shape[0]), yall[:,2], 'r-')
    axarr3[2].plot(range(0,rdd.shape[0]), rdd[:,2], 'g.')
    axarr3[2].plot(range(0,y1all_model.shape[0]), y1all_model[:,2], 'b-')
    #axarr3[2].plot(range(0,ypredhist.shape[0]), ypredhist[:,2], 'y*')
    #axarr3[2].plot(range(0,yexp.shape[0]), yexp[:,2], 'k.')

    xdata = np.empty((0,9))
    ydata = np.empty((0,6))
    for i in range(1,N):
        if flag_hist[i] == False:
            continue
        #if i%200 == 199:
            #gp.Reset()
        xnew = xall[i,:]

        # prediction
        ypredict, Vy = gp.Predict(xnew)
        #ypredict[2] = min(max(ypredict[2], ynew[2]-1), ynew[2]+1)
        ypredhist = np.vstack((ypredhist, ypredict))
        #axarr1[1].plot(i, Vy, 'k.')
        axarr3[0].plot(i, ypredict[0], 'y*')
        axarr3[1].plot(i, ypredict[1], 'y*')
        axarr3[2].plot(i, ypredict[2], 'y*')


        # sampling
        ynew = yall[i,:]
        xdata = np.vstack((xdata, xnew))
        ydata = np.vstack((ydata, ynew))
        gp.Update(xnew, ynew)

    plt.show()
