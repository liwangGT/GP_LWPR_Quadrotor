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

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Imu

import GP_quads.GP_pred as gp_cf

def cal_R(c1, c2, c3):
    R = np.array([[cos(c1)*cos(c3), sin(c2)*sin(c1)*cos(c3)-cos(c2)*sin(c3), cos(c2)*sin(c1)*cos(c3)+sin(c2)*sin(c3)],
         [cos(c1)*sin(c3), sin(c2)*sin(c1)*sin(c3)+cos(c2)*cos(c3), cos(c2)*sin(c1)*sin(c3)-sin(c2)*cos(c3)],
         [-sin(c1), sin(c2)*cos(c1), cos(c2)*cos(c1)]])
    return R

class CF():
    def __init__(self, i):
        self.worldFrame = rospy.get_param("~worldFrame", "/world")
        self.frame = 'crazyflie%d' %i
        self.zscale = 3
        self.state = 0
        self.position    = []
        self.orientation = []
        self.pref = []
        self.cmd_vel = []
        self.goal = PoseStamped()
        self.goal.header.seq = 0
        self.goal.header.stamp = rospy.Time.now()
        pub_name = '/crazyflie%d/goal' %i
        sub_name = '/crazyflie%d/CF_state' %i
        pub_cmd_diff = '/crazyflie%d/cmd_diff' %i
        sub_cmd_vel = '/crazyflie%d/cmd_vel' %i
        sub_imu_data = '/crazyflie%d/imu' %i
        self.pubGoal = rospy.Publisher(pub_name, PoseStamped, queue_size=1)
        self.pubCmd_diff = rospy.Publisher(pub_cmd_diff, Twist, queue_size=1)
        self.subCmd_vel = rospy.Subscriber(sub_cmd_vel, Twist, self.cmdCallback)
        self.subGoal = rospy.Subscriber(pub_name, PoseStamped, self.GoalCallback)
        self.subState  = rospy.Subscriber(sub_name, String, self.CfsCallback)
        self.subImu = rospy.Subscriber(sub_imu_data, Imu, self.ImuCallback)
        self.listener = TransformListener()
        self.listener.waitForTransform(self.worldFrame, self.frame, rospy.Time(), rospy.Duration(5.0))
        self.updatepos()
        self.send_cmd_diff()


    def CfsCallback(self, sdata):
        self.state = int(sdata.data)

    def ImuCallback(self, sdata):
        accraw = sdata.linear_acceleration
        imuraw = np.array([accraw.x, -accraw.y, -accraw.z])
        self.updatepos()
        imuraw = cal_R(self.orientation[1],self.orientation[0],self.orientation[2]).dot(imuraw)       
        self.Imu = np.array([imuraw[0], imuraw[1], 9.88+imuraw[2]])
        self.Gyro = np.array([sdata.angular_velocity.x, sdata.angular_velocity.y, sdata.angular_velocity.z])


    def GoalCallback(self, gdata):
        self.goal = gdata

    def cmdCallback(self, cdata):
        self.cmd_vel = cdata

    def hover_init(self, pnext, s):
        goal = PoseStamped()
        goal.header.seq = self.goal.header.seq + 1
        goal.header.frame_id = self.worldFrame
        goal.header.stamp = rospy.Time.now()
        if self.state != 1:
            goal.pose.position.x = pnext[0]
            goal.pose.position.y = pnext[1]
            goal.pose.position.z = pnext[2]
            quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
            goal.pose.orientation.x = quaternion[0]
            goal.pose.orientation.y = quaternion[1]
            goal.pose.orientation.z = quaternion[2]
            goal.pose.orientation.w = quaternion[3]
            self.pubGoal.publish(goal)
            #print "Waiting for crazyflie %d to take off" %i
        else:
            t = self.listener.getLatestCommonTime(self.worldFrame, self.frame)
            if self.listener.canTransform(self.worldFrame, self.frame, t):
                position, quaternion = self.listener.lookupTransform(self.worldFrame, self.frame, t)
                rpy = tf.transformations.euler_from_quaternion(quaternion)
                dx = pnext[0] - position[0]
                dy = pnext[1] - position[1]
                dz = pnext[2] - position[2]
                dq = 0 - rpy[2]
                
                s = max(s,0.5)
                goal.pose.position.x = position[0] + s*dx
                goal.pose.position.y = position[1] + s*dy
                goal.pose.position.z = position[2] + s*dz
                quaternion = tf.transformations.quaternion_from_euler(0, 0, rpy[2]+s*dq)
                goal.pose.orientation.x = quaternion[0]
                goal.pose.orientation.y = quaternion[1]
                goal.pose.orientation.z = quaternion[2]
                goal.pose.orientation.w = quaternion[3]
                self.pubGoal.publish(goal)
                error = sqrt(dx**2+dy**2+dz**2)
                print 'Hovering error is %0.2f' %error
                if error<0.1:
                    return 1
        return 0
        
    def updatepos(self):
        t = self.listener.getLatestCommonTime(self.worldFrame, self.frame)
        if self.listener.canTransform(self.worldFrame, self.frame, t):
            self.position, quaternion = self.listener.lookupTransform(self.worldFrame, self.frame, t)
            rpy = tf.transformations.euler_from_quaternion(quaternion)
            self.orientation = rpy

    def goto(self, pnext):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.seq = self.goal.header.seq + 1
        goal.header.frame_id = self.worldFrame
        goal.pose.position.x = pnext[0]
        goal.pose.position.y = pnext[1]
        goal.pose.position.z = pnext[2]
        quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]
        self.pubGoal.publish(goal)

    def gotoT(self, pnext, s):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.seq = self.goal.header.seq + 1
        goal.header.frame_id = self.worldFrame
        t = self.listener.getLatestCommonTime(self.worldFrame, self.frame)
        if self.listener.canTransform(self.worldFrame, self.frame, t):
            position, quaternion = self.listener.lookupTransform(self.worldFrame, self.frame, t)
            rpy = tf.transformations.euler_from_quaternion(quaternion)
            dx = pnext[0] - position[0]
            dy = pnext[1] - position[1]
            dz = pnext[2] - position[2]
            dq = 0 - rpy[2]

            goal.pose.position.x = position[0] + s*dx
            goal.pose.position.y = position[1] + s*dy
            goal.pose.position.z = position[2] + s*dz
            quaternion = tf.transformations.quaternion_from_euler(0, 0, rpy[2]+s*dq)
            goal.pose.orientation.x = quaternion[0]
            goal.pose.orientation.y = quaternion[1]
            goal.pose.orientation.z = quaternion[2]
            goal.pose.orientation.w = quaternion[3]
            self.pubGoal.publish(goal)
            error = sqrt(dx**2+dy**2+dz**2)
            print 'error is %0.2f' %error
            if error<0.1:
                return 1
            else:
                return 0

    def send_cmd_diff(self, roll=0, pitch=0, yawrate=0, thrust=49000):
        # note theoretical default thrust is 39201 equal to 35.89g lifting force
        # actual 49000 is 35.89
        msg = Twist()
        msg.linear.x = -pitch  #note vx is -pitch, see crazyflie_server.cpp line 165
        msg.linear.y = roll    #note vy is roll
        msg.linear.z = thrust
        msg.angular.z = yawrate
        self.pubCmd_diff.publish(msg)

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

def norm(p):
    temp = 0
    for a in p:
        temp = temp + a**2
    return sqrt(temp)

def BezierInterp(T, p0, p1):
    # Bezier curve interpolation
    # T-conversion factor, dp/dt = dp/ds*1/T
    # p0,p1 are start/end point x,xd,xdd,xddd,xdddd
    d = 3 # number of derivatives of p0,p1
    n = 7 #degree of curve
    Cout = np.zeros((n+1,3))
    
    # note: interpolation is done in [0,1], then converted to [0,T]
    for i in range(3):
        # cal x,y,z dimensions seperately
        CL = np.zeros((d+1,(n+1)/2))
        CR = np.zeros((d+1,(n+1)/2))
        x0 = np.array([p0[0,i],p0[1,i]*T,p0[2,i]*T**2,p0[3,i]*T**3])
        x1 = np.array([p1[0,i],p1[1,i]*T,p1[2,i]*T**2,p1[3,i]*T**3])
        CL[:,0] = np.multiply(ft(n-np.arange(d+1))/ft(n),x0)
        for j in range(d):
            CL[0:d-j,j+1] = CL[0:d-j,j] + CL[1:d-j+1,j]
        CR[:,-1] = np.multiply(ft(n-np.arange(d+1))/ft(n),x1)
        for j in range(d):
            CR[0:d-j,-2-j] = CR[0:d-j,-2-j+1] - CR[1:d-j+1,-2-j+1]
        # control points 10*3 matrix
        Cout[:,i] = np.transpose(np.hstack((CL[0,:],CR[0,:])))
    return Cout

def ref_extract(T, t, Cin):
    #extract nominal state from Bezier control points
    n = 7 #degree of cureve
    d = 3 #derivative degree of Cin
    Cout  = np.zeros((d+1,3))
    tn = min(t/T,1) #nomarlized time, [0,1]

    # calculte basis
    Polybasis = np.zeros((d+1, n+1))
    for k in range(d+1):
        for kk in range(n-k+1):
            Polybasis[k,kk] = ft(n)/ft(n-k)*nCk(n-k,kk)*(tn**kk)*((1-tn)**(n-k-kk))

    for i in range(3):
        Ctemp = np.zeros((d+1,n+1))
        # cal 3 dimensional nominal states
        Ctemp[0,:] = np.transpose(Cin[:,i])
        for j in range(d):
            if j == 0:
                Ctemp[1,0:-1] = Ctemp[j,1:] - Ctemp[j,0:-1]
            else:
                Ctemp[j+1,0:-1-j] = Ctemp[j,1:-j] - Ctemp[j,0:-1-j]
        # cal 0-4th order derivatives
        for k in range(d+1):
            Cout[k,i] = np.dot(Polybasis[k,:], Ctemp[k,:])/T**k

    return Cout


def Safe_Barrier_3D(x, uhat, Kb):
    u = uhat.copy() #init u
    N = len(uhat)
    zscale = 3
    gamma = 1e0
    Ds = 0.3
    H = 2*np.eye(3*N)
    f = -2*np.reshape(np.hstack(uhat.values()),(3*N,1))
    A = np.empty((0,3*N))
    b = np.empty((0,1))
    for i in range(N-1):
        for j in range(i+1,N):
            pr    = np.multiply(x[i][0,:]-x[j][0,:],np.array([1,1,1.0/zscale]))
            prd   = np.multiply(x[i][1,:]-x[j][1,:],np.array([1,1,1.0/zscale]))
            prdd  = np.multiply(x[i][2,:]-x[j][2,:],np.array([1,1,1.0/zscale]))
            prddd = np.multiply(x[i][3,:]-x[j][3,:],np.array([1,1,1.0/zscale]))

            # note elementwise operation
            h     = LA.norm(pr,4)**4 - Ds**4
            hd    = sum(4*pr**3*prd)
            hdd   = sum(12*pr**2*prd**2 + 4*pr**3*prdd)
            hddd  = sum(24*pr*prd**3 + 36*pr**2*prd*prdd + 4*pr**3*prddd)
            # hdddd = Lfh + Lgh*(ui-uj)
            Lfh   = sum(24*pr**4 + 144*pr*prd**2*prdd + 36*pr**2*prdd**2 + 48*pr**2*prd*prddd)
            Lgh   = 4*pr**3*np.array([1,1,1.0/zscale])
            #- Lgh*(ui-uj) < gamma*h + Lfh
            Anew = np.zeros((3*N,))
            Anew[3*i:3*i+3] = - Lgh
            Anew[3*j:3*j+3] = Lgh
            bnew = gamma*np.dot(Kb,[h,hd,hdd,hddd]) + Lfh
            A = np.vstack([A, Anew])
            b = np.vstack([b, bnew])
    sol = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b), matrix(np.empty((0,3*N))), matrix(np.empty((0,1))))
    x = sol['x']
    for i in range(N):
        u[i] = np.reshape(x[3*i:3*i+3],(1,3))
    return u

def Invert_diff_flat_output(x, u, thrust_hover):
    #note yaw =0 is fixed
    m = 35.89/1000
    g = 9.8
    beta1 = - x[2,0]
    beta2 = - x[2,2] + 9.8
    beta3 =  x[2,1]

    roll = atan2(beta3,sqrt(beta1**2+beta2**2))*180/PI
    pitch = atan2(beta1,beta2)*180/PI
    yawrate = 0
    a_temp = LA.norm([0,0,g]-x[2,:])
    # acc g correspond to 49000, at thrust_hover
    thrust = int(a_temp/g*thrust_hover)
    return roll,pitch,yawrate,thrust

def Invert_diff_flat_output_GP(x, u, thrust_hover, ygp):
    #note yaw =0 is fixed
    m = 35.89/1000
    g = 9.8
    beta1 = ygp[0] - x[2,0]
    beta2 = ygp[2] - x[2,2] + 9.8
    beta3 =  x[2,1] - ygp[1]

    roll = atan2(beta3,sqrt(beta1**2+beta2**2))*180/PI
    pitch = atan2(beta1,beta2)*180/PI
    yawrate = 0
    a_temp = LA.norm([0,0,g]-x[2,:])
    # acc g correspond to 49000, at thrust_hover
    thrust = int(a_temp/g*thrust_hover)
    return roll,pitch,yawrate,thrust

if __name__ == '__main__':
    # init all params, publisers, subscrivers
    rospy.init_node('cf_Diff_Flat', anonymous = True)
    N = rospy.get_param("~cfnumber", 2)  # total number of cfs
    #init p,pd,pdd,pddd,pdddd, on a circle
    N = 1
    g = 9.8 #kg.m.s2
    p0 = dict()
    p1 = dict()
    cfs = dict()
    theta_N = np.linspace(0,2*PI,N, endpoint=False)
    for i in range(N):
        cfs[i] = CF(i)
        p0[i] = np.zeros([4,3])
        p1[i] = np.zeros([4,3])
        p0[i][0,:] = [cos(theta_N[i]), sin(theta_N[i]), -0.8]
        p1[i][0,:] = [cos(PI+theta_N[i]), sin(PI+theta_N[i])+0.2*(-1)**i, -0.8]
    p0[0][0,:] = [-1,-1,-0.8]
    p1[0][0,:] = [1,1.1,-0.8]


    # define circle waypoints
    pc = dict()
    Tf = 1.5*np.ones((9,1))
    for ii in range(10):
        pc[ii] = np.zeros([4,3])
    pc[0][0,:] = [-1, 0, -0.8]
    pc[1][0,:] = [0, 1, -0.8]
    pc[2][0,:] = [1, 0, -0.8]#-1.4]
    pc[3][0,:] = [0, -1, -0.8]
    pc[4][0,:] = [-1, 0, -0.8]#-1.4]
    pc[5][0,:] = [0, 1, -0.8]
    pc[6][0,:] = [1, 0, -0.8]#-1.4]
    pc[7][0,:] = [0, -1, -0.8]
    pc[8][0,:] = [-1, 0, -0.8]#-1.4]
    pc[9][0,:] = [0, 1, -0.8]

    pc[0][1,:] = [0, 0, 0]
    pc[1][1,:] = [1.5, 0, 0]
    pc[2][1,:] = [0, -1.5, 0]
    pc[3][1,:] = [-1.5, 0, 0]
    pc[4][1,:] = [0, 1.5, 0]
    pc[5][1,:] = [1.5, 0, 0]
    pc[6][1,:] = [0, -1.5, 0]
    pc[7][1,:] = [-1.5, 0, 0]
    pc[8][1,:] = [0, 1.5, 0]
    pc[9][1,:] = [0, 0, 0]

    Tf = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

    # take off and hover at (xinit, yinit, 0.5)
    phover = dict()
    fflag = np.zeros((N,1))
    for i in range(N):
        phover[i] = [cfs[i].position[0], cfs[i].position[1], -0.8]
        print phover[i]
        
    t0 = rospy.Time.now()
    # use 3s to take off and hover
    t_takeoff = 2.0
    s = 0 #parameterized time [0~1]
    while (s < 1) or (sum(fflag) < N):
        t = rospy.Time.now()
        s = min((t-t0).to_sec()/t_takeoff, 1.0)
        for i in range(N):
            cfs[i].send_cmd_diff()
            if cfs[i].state != 1:
                s = 1
                t0 = rospy.Time.now()
            fflag[i] = cfs[i].hover_init(phover[i],s)    
    time.sleep(2)
    # recording hovering thrust
    thrust_hover = dict()
    for i in range(N):
        thrust_hover[i] = cfs[i].cmd_vel.linear.z 
        cfs[i].send_cmd_diff(0, 0, 0, thrust_hover[i])

    print 'Going to first waypoint'
    # go to first waypoint
    t_wp = 3.0
    t0 = rospy.Time.now()
    s = 0 #parameterized time [0~1]
    while (s < 1) or (sum(fflag) < N):
        cfs[i].send_cmd_diff(0, 0, 0, thrust_hover[i])
        t = rospy.Time.now()
        s = min((t-t0).to_sec()/t_wp, 1.0)
        for i in range(N):
            fflag[i] = cfs[i].gotoT(pc[0][0,:], s)
    print 'Initialization Done! Experiment Starts!'

#-------------------------Actual control experiment--------------------#

    # pole placement for CLF and CBF
    AA=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]) 
    bb=np.array([[0], [0], [1]]) 
    #Kb=acker(AA,bb,[-2.2,-2.4,-2.6,-2.8])
    #Kb=acker(AA,bb,[-5.2,-5.4,-5.6,-5.8])
    Kb=np.asarray(acker(AA,bb,[-8.2,-8.4,-8.6]))

    # start auto mode program
    print 'Auto mode: differential flatness'
    CBF_on = 0
    Cout = dict() # Bezier Control points
    tk = 0
    dt = 0.02
    t_total = sum(Tf)+1
    phat = dict() #nominal state from interpolation
    uhat = dict() #nominal control
    x = dict() #actual states
    xd = dict() #derivative of x
    for i in range(N):
        x[i] = pc[0][0:-1,:]
        xd[i] = np.zeros((3,3))
    solvers.options['show_progress'] = False #suppress display of solver.qp

    t_hist = dict()
    p_hist = dict()
    phat_hist = dict()
    ptrack_hist = dict()
    rpytrack_hist = dict()
    u_hist = dict()
    uhat_hist = dict()
    cmd_hist = dict()
    cmdreal_hist = dict()
    imu_hist = dict()
    gyro_hist = dict()
    
    time.sleep(2)

    thrust = 0 #initialize thrust
    flag_done = 1
    kk = 0
    ttt = rospy.Time.now()
    t_real =0
    while not (rospy.is_shutdown() or t_real*dt>t_total):
        tt0 = rospy.Time.now()
        if flag_done == 1 and kk<9:
            # Bezier Interpolation
            T = Tf[kk] #time to complete, also scaling factor
            for i in range(N):
                Cout[i] = BezierInterp(T, pc[kk], pc[kk+1])
            kk = kk +1
            flag_done = 0
            tk = 0

        t = tk*dt
        try:
            print '----Current time is %0.2f----' %t

            # extract ref trajectories
            for i in range(N):
                cfs[i].updatepos()
                phat[i] = ref_extract(T, t, Cout[i])
               # update real state
                ptemp = cfs[i].position
                if t_real == 0:
                   xKF = Kalmanu(np.array([ptemp[0],phat[i][1,0],phat[i][2,0]]))
                   yKF = Kalmanu(np.array([ptemp[1],phat[i][1,1],phat[i][2,1]]))
                   zKF = Kalmanu(np.array([ptemp[2],phat[i][1,2],phat[i][2,2]]))
                else:
                   xpre = xKF.update(np.array([ptemp[0], cfs[i].Imu[0]]))
                   ypre = yKF.update(np.array([ptemp[1], cfs[i].Imu[1]]))
                   zpre = zKF.update(np.array([ptemp[2], cfs[i].Imu[2]]))
                   #x[i][0,:] = np.array([ptemp[0],ptemp[1],ptemp[2]]) 
                   #x[i][1,:] = np.array([xpre[1],ypre[1],zpre[1]]) 
                   #x[i][2,:] = np.array([xpre[2],ypre[2],zpre[2]]) 
                   #np.array([phat[i][2,0],phat[i][2,1],phat[i][2,2]]) 

                # compute nominal control from xd = AA*x + bb*u, u = ref(4)-Kb*(x-ref)
                uhat[i] = phat[i][3,:] - np.dot(Kb,x[i] - phat[i][0:-1,:])

            if CBF_on == 1:
                u = Safe_Barrier_3D(x, uhat, Kb)
                # ? if LA.norm(u)<LA.norm(uhat)*10: u = Safe_Barrier_3D(x, u*10, Kb)
            else:
                u = uhat.copy()



            # make GP prediction with xnew
            if t_real >= 1:  #KF is started
                xnew = np.array([cfs[0].position[0], cfs[0].position[1], cfs[0].position[2],
                                  xpre[1], ypre[1], zpre[1],
                                  180.0/PI*cfs[0].orientation[0], 
                                  180.0/PI*cfs[0].orientation[1],
                                  180.0/PI*cfs[0].orientation[2] ])
                if t_real > 1:
                    ypredict, Vy = gp.Predict(xnew)
            else: #KF is not started
                ypredict = np.zeros((6,))
                Vy = 0


            # update actual dynamics
            for i in range(N):
                # Compute roll/pitch/yawrate/thrust
                #roll,pitch,yawrate,thrust = Invert_diff_flat_output(x[i], u[i], thrust_hover[i])

                # compute u with GP
                roll,pitch,yawrate,thrust = Invert_diff_flat_output_GP(x[i], u[i], thrust_hover[i], ypredict)
                # send to quads
                cfs[i].goto(x[i][0,:])    # send setpoint
                cfs[i].send_cmd_diff(roll, pitch, yawrate, thrust)  # send feedforward term
                # update corrected ref traj
                xd[i] = np.dot(AA,x[i]) + np.dot(bb,u[i])
                x[i] = x[i] + xd[i]*dt   
            

            # update GP model with xnew ynew
            if t_real >=1:
                fzReal = cfs[i].cmd_vel.linear.z
                y1model = np.array([0,0,g])-fzReal/thrust_hover[0]*g*np.array([cos(cfs[0].orientation[0])*sin(cfs[0].orientation[1]), -sin(cfs[0].orientation[0]), cos(cfs[0].orientation[0])*cos(cfs[0].orientation[1])])
                y1new = np.array([xpre[2], ypre[2], zpre[2]])-y1model
                y2new = np.array([0,0,0])
                ynew = np.append(y1new, y2new)
                
                if t_real == 1:
                    gp = gp_cf.GP_pred(xnew, ynew)
                elif t_real >1:
                    gp.Update(xnew, ynew)


            t_hist[t_real] = (rospy.Time.now()-ttt).to_sec()
            p_hist[t_real] = x[0]
            phat_hist[t_real] = phat[0]
            ptrack_hist[t_real] = cfs[0].position
            rpytrack_hist[t_real] = cfs[0].orientation
            u_hist[t_real] = u[0]
            uhat_hist[t_real] = uhat[0]
            cmd_hist[t_real] = [roll, -pitch, yawrate, thrust]
            cmdreal_hist[t_real] = cfs[i].cmd_vel
            imu_hist[t_real] = cfs[i].Imu
            gyro_hist[t_real] = cfs[i].Gyro

            t_real = t_real+1
            tk = tk+1
            # check waypoint finished
            if t >= T:
                flag_done = 1

            while (rospy.Time.now()-tt0).to_sec() < dt:
                aa = 1
            print '----Actual dt is %0.3f----' %(rospy.Time.now()-tt0).to_sec()

        except rospy.ROSInterruptException:
            print '----Experiment interrupted!!!----'
            break
    # save data
    f = open('cf1_3rd_data_09142019_v0.pckl', 'w')
    pickle.dump([t_hist, p_hist, phat_hist, ptrack_hist, u_hist, uhat_hist, cmd_hist, cmdreal_hist, rpytrack_hist, imu_hist, gyro_hist, thrust_hover[i]], f)
    f.close()
    print '----Experiment completed!!!----'
