#!/usr/bin/env python
import tf
from tf import TransformListener
import roslib, rospy
from control import acker
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import sys
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

def norm(p):
    temp = 0
    for a in p:
        temp = temp + a**2
    return sqrt(temp)

class Kalman():
    def __init__(self, i):
        self.worldFrame = rospy.get_param("~worldFrame", "/world")
        self.frame = 'crazyflie%d' %i
        self.index = i
        self.imunew = 0 # see if new data comes
        self.posnew = 0
        self.time0 = rospy.Time.now()
        self.time1 = rospy.Time.now()
        sub_imu_data = '/crazyflie%d/imu' %i
        sub_pos_data = '/crazyflie%d/pos' %i
        self.subImu = rospy.Subscriber(sub_imu_data, Imu, self.ImuCallback)
        self.subpos = rospy.Subscriber(sub_pos_data, PoseStamped, self.posCallback)
        pub_name = '/crazyflie%d/estimator' %i
        self.pubEst = rospy.Publisher(pub_name, Kalman_pred, queue_size=1)
        self.position    = []
        self.orientation = []
        self.Imu = []
        self.X = []
        self.Y = []
        self.Z = []
        self.Xvar = []
        self.Yvar = []
        self.Zvar = []
        self.Q = []
        self.R = []
        self.SX = []
        self.SY = []
        self.SZ = []

    def EstnoImu(self):
        dt = 0.02
        newX = [self.position[0]]
        newY = [self.position[1]]
        newZ = [self.position[2]]
        if len(self.X) == 0:
             self.X = np.array([self.position[0], 0, 0])
             self.Y = np.array([self.position[1], 0, 0])
             self.Z = np.array([self.position[2], 0, 0])
             self.Q = np.array([[1e-4, 1e-6, 1e-6],[1e-6, 1e-4, 1e-6],[1e-6, 1e-6, 1e-1]]) # dynamic noise
             self.R = np.array([1e-8]) # measurement noise
             self.SX = self.Q # state estimate noise
             self.SY = self.Q # state estimate noise
             self.SZ = self.Q # state estimate noise
        Phi = np.array([[1, dt, 0],[0, 1, dt],[0,0,1]]) # state transition matrix
        M   = np.array([1,0,0]) # measurement matrix
        # prediction step
        self.X = np.dot(Phi,self.X)
        self.Y = np.dot(Phi,self.Y)
        self.Z = np.dot(Phi,self.Z)
        self.SX = Phi.dot(self.SX).dot(Phi.transpose()) + self.Q
        self.SY = Phi.dot(self.SY).dot(Phi.transpose()) + self.Q
        self.SZ = Phi.dot(self.SZ).dot(Phi.transpose()) + self.Q
        # Kalman gain
        KX = self.SX.dot(M.transpose())*1/( M.dot(self.SX).dot(M.transpose())+self.R )
        KY = self.SY.dot(M.transpose())*1/( M.dot(self.SY).dot(M.transpose())+self.R )
        KZ = self.SZ.dot(M.transpose())*1/( M.dot(self.SZ).dot(M.transpose())+self.R )
        # update
        self.X = self.X + KX*(newX-M.dot(self.X))
        self.Y = self.Y + KY*(newY-M.dot(self.Y))
        self.Z = self.Z + KZ*(newZ-M.dot(self.Z))
        self.SX = self.SX - np.outer(KX,M.dot(self.SX))
        self.SY = self.SY - np.outer(KY,M.dot(self.SY))
        self.SZ = self.SZ - np.outer(KZ,M.dot(self.SZ))

        data = Kalman_pred()
        data.predX.x = self.X[0]
        data.predX.y = self.X[1]
        data.predX.z = self.X[2]
        data.varX = self.SX.flatten()
        data.predY.x = self.Y[0]
        data.predY.y = self.Y[1]
        data.predY.z = self.Y[2]
        data.varY = self.SY.flatten()
        data.predZ.x = self.Z[0]
        data.predZ.y = self.Z[1]
        data.predZ.z = self.Z[2]
        data.varZ = self.SZ.flatten()
        self.pubEst.publish(data)

    def EstImu(self):
        dt = 0.02
        t0 = rospy.Time.now()
        #while (self.imunew != 1 or self.posnew != 1) and (rospy.Time.now()-t0).to_sec() < 0.010:
        #    aa = 1
        #self.imunew = 0 # set data to be used
        #self.posnew = 0 # set data to be used
        if len(self.position)==0 or len(self.Imu) == 0:
            t0 = rospy.Time.now()
            while (len(self.Imu)==0 or len(self.position)==0) and (rospy.Time.now()-t0).to_sec() < 5: #5s time out
                aa = 1 # wait for Imu data
            if (rospy.Time.now()-t0).to_sec() >= 5: # time out
                rospy.logerr("Can't log Imu data for Crazyflie %d", self.index) 
                return 0      
        newX = np.array([self.position[0], self.Imu[0]])
        newY = np.array([self.position[1], self.Imu[1]])
        newZ = np.array([self.position[2], self.Imu[2]])
        if len(self.X) == 0:
             self.X = np.array([self.position[0], 0, 0])
             self.Y = np.array([self.position[1], 0, 0])
             self.Z = np.array([self.position[2], 0, 0])
             self.Q = np.array([[1e-8, 1e-12, 1e-12],[1e-12, 1e-8, 1e-12],[1e-12, 1e-12, 1e-1]]) # dynamic noise
             self.R = np.array([[1e-6, 1e-12],[1e-12,1e-1]]) # measurement noise
             self.SX = self.Q # state estimate noise
             self.SY = self.Q # state estimate noise
             self.SZ = self.Q # state estimate noise
        Phi = np.array([[1, dt, 0],[0, 1, dt],[0,0,1]]) # state transition matrix
        M   = np.array([[1,0,0],[0,0,1]]) # measurement matrix
        # prediction step
        self.X = np.dot(Phi,self.X)
        self.Y = np.dot(Phi,self.Y)
        self.Z = np.dot(Phi,self.Z)
        self.SX = Phi.dot(self.SX).dot(Phi.transpose()) + self.Q
        self.SY = Phi.dot(self.SY).dot(Phi.transpose()) + self.Q
        self.SZ = Phi.dot(self.SZ).dot(Phi.transpose()) + self.Q
        # Kalman gain
        KX = self.SX.dot(M.transpose()).dot(inv( M.dot(self.SX).dot(M.transpose())+self.R ))
        KY = self.SY.dot(M.transpose()).dot(inv( M.dot(self.SY).dot(M.transpose())+self.R ))
        KZ = self.SZ.dot(M.transpose()).dot(inv( M.dot(self.SZ).dot(M.transpose())+self.R ))
        # update
        self.X = self.X + KX.dot(newX-M.dot(self.X))
        self.Y = self.Y + KY.dot(newY-M.dot(self.Y))
        self.Z = self.Z + KZ.dot(newZ-M.dot(self.Z))
        self.SX = self.SX - KX.dot(M.dot(self.SX))
        self.SY = self.SY - KY.dot(M.dot(self.SY))
        self.SZ = self.SZ - KZ.dot(M.dot(self.SZ))

        data = Kalman_pred()
        data.predX.x = self.X[0]
        data.predX.y = self.X[1]
        data.predX.z = self.X[2]
        data.varX = self.SX.flatten()
        data.predY.x = self.Y[0]
        data.predY.y = self.Y[1]
        data.predY.z = self.Y[2]
        data.varY = self.SY.flatten()
        data.predZ.x = self.Z[0]
        data.predZ.y = self.Z[1]
        data.predZ.z = self.Z[2]
        data.varZ = self.SZ.flatten()
        self.pubEst.publish(data)

    def ImuCallback(self, sdata):
        acc = sdata.linear_acceleration
        self.Imu = np.array([acc.x, -acc.y, 9.8-acc.z])
        self.imunew = 1
        self.time1 = sdata.header.stamp

    def posCallback(self, sdata):
        self.position = np.array([sdata.pose.position.x, sdata.pose.position.y, sdata.pose.position.z])
        self.posnew = 1
        self.time1 = sdata.header.stamp

    def updatepos(self):
        t = 0
        #t = self.listener.getLatestCommonTime(self.worldFrame, self.frame)
        #if self.listener.canTransform(self.worldFrame, self.frame, t):
        #    self.position, quaternion = self.listener.lookupTransform(self.worldFrame, self.frame, t)
        #    rpy = tf.transformations.euler_from_quaternion(quaternion)
        #    self.orientation = rpy


if __name__ == '__main__':
    # init all params, publisers, subscrivers
    rospy.init_node('cf_Estimator', anonymous = True)
    N = rospy.get_param("~cfnumber", 2)  # total number of cfs
    dt = 0.02
    cfs = dict()
    for i in range(N):
        cfs[i] = Kalman(i)

    print '----Kalman filter started!!!----'
    while not (rospy.is_shutdown()):
        tt0 = rospy.Time.now()
        try:
            for i in range(N):
                cfs[i].updatepos() # get pos data from transfomation
                cfs[i].EstImu()

            # debug Kalman
            if fabs(cfs[i].position[0] -0.626817286015)<1e-7 or fabs(cfs[i].position[0] +0.877806127071)<1e-7:
                print cfs[i].position[0], cfs[i].X[1], cfs[i].X[2]

            while (rospy.Time.now()-tt0).to_sec() < dt:
                aa = 1
        except rospy.ROSInterruptException:
            print '----Kalman filter interrupted!!!----'
            break

    print '----Kalman filter stopped!!!----'
