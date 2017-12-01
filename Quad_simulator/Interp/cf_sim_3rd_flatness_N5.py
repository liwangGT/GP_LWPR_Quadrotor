#!/usr/bin/env python
import roslib, rospy
from control import acker
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

from geometry_msgs.msg import Twist
from Quad_visual import *

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
    Ctemp = np.zeros((d+1,n+1))
    Cout  = np.zeros((d+1,3))
    tn = min(t/T,1) #nomarlized time, [0,1]

    for i in range(3):
        # cal 3 dimensional nominal states
        Ctemp[0,:] = np.transpose(Cin[:,i])
        for j in range(d):
            if j == 0:
                Ctemp[1,0:-1] = Ctemp[j,1:] - Ctemp[j,0:-1]
            else:
                Ctemp[j+1,0:-1-j] = Ctemp[j,1:-j] - Ctemp[j,0:-1-j]
        # cal 0-4th order derivatives
        for k in range(d+1):
            temp = 0
            for kk in range(n-k+1):
                Btemp =  nCk(n-k,kk)*(tn**kk)*((1-tn)**(n-k-kk))
                temp = temp + ft(n)/ft(n-k)*Ctemp[k,kk]*Btemp
            Cout[k,i] = temp/T**k

    return Cout


class CF():
    def __init__(self, p0, i):
        self.pos = p0
        self.vel = np.zeros((1,3))
        self.zscale = 3
        pub_cmd_diff = '/crazyflie%d/cmd_diff' %i
        self.pubCmd_diff = rospy.Publisher(pub_cmd_diff, Twist, queue_size=1)
        self.send_cmd_diff()

        self.pref = p0
        self.vref = np.zeros((1,3))

    def gotoSmooth(self, p0, p1, t0, t, T):
        #use cosine acc/dacc to get smooth vref curve
        T0 = 2 #s
        v0 = norm(p0-p1)/(T - T0/2) #m/s, note acc/dacc takes T0/2 s
        ii = (p1-p0)/norm(p0-p1)
        kp = 10
        if T <= 2:
            # acc/dacc require at least 2s
            self.goto(p0, p1, t0, t, T)
        else:
            Dt = min(t-t0, T)
            if Dt < 1:
                self.vref = (-v0/2*cos(2*PI/T0*Dt) + v0/2)*ii
                self.pref = p0+(v0*Dt/2-v0*T0/4/PI*sin(2*PI/T0*Dt))*ii
            elif Dt > T-1:
                self.vref = (v0/2*cos(2*PI/T0*(Dt-(T-1))) + v0/2)*ii
                self.pref = p0+(v0*(T-1.5) + v0/2*(Dt-(T-1)) + v0*T0/4/PI*sin(2*PI/T0*(Dt-(T-1))))*ii
            else:
                self.vref = v0*ii
                self.pref = p0+(v0*(Dt-0.5))*ii
            #compensate for tracking error
            self.vref = self.vref + kp*(self.pref - self.pos)

    def goto(self, p0, p1, t0, t, T):
        kp = 10
        v = norm(p0-p1)/T
        ii = (p1-p0)/norm(p0-p1)
        self.vref = v*ii
        self.pref = p0 + v*(t-t0)*ii
        #compensate for tracking error
        self.vref = self.vref + kp*(self.pref - self.pos)

    def send_cmd_diff(self, roll=0, pitch=0, yawrate=0, thrust=39201):
        # note default thrust is 39201 equal to 35.89g lifting force
        msg = Twist()
        msg.linear.x = roll
        msg.linear.y = pitch
        msg.linear.z = thrust
        msg.angular.z = yawrate
        self.pubCmd_diff.publish(msg)

"""
class Shpere3D():
    def __init__(self, initpos, ax):
        u, v = np.mgrid[0:2*PI:9j, 0:PI:5j]
        self.zscale = 3
        self.x = 0.3/2*np.sign(np.cos(u)*np.sin(v))*np.sqrt(np.abs(np.cos(u)*np.sin(v))) 
        self.y = 0.3/2*np.sign(np.sin(u)*np.sin(v))*np.sqrt(np.abs(np.sin(u)*np.sin(v)))
        self.z = 0.3/2*self.zscale*np.sign(np.cos(v))*np.sqrt(np.abs(np.cos(v)))
        self.handle = ax.plot_wireframe(self.x+initpos[0], self.y+initpos[1], self.z+initpos[2], color="k")
        #plt.pause(.001)  #moved outside of the for loop

    def update(self, newpos, ax):
        oldcol = self.handle
        ax.collections.remove(oldcol)
        self.handle = ax.plot_wireframe(self.x+newpos[0], self.y+newpos[1], self.z+newpos[2], color="k")
        #plt.pause(.001)  #moved outside of the for loop
        #plt.draw()

class Coord3D():
    def __init__(self, ax, initpos, roll=0.0, pitch=0.0, yaw=0.0):
        R = self.RotMat()
        L = 0.35
        self.hx, = plt.plot([initpos[0] , initpos[0]+L*R[0,0]], [initpos[1] , initpos[1]+L*R[1,0]], [initpos[2] , initpos[2]+L*R[2,0]], 'r')
        self.hy, = plt.plot([initpos[0] , initpos[0]+L*R[0,1]], [initpos[1] , initpos[1]+L*R[1,1]], [initpos[2] , initpos[2]+L*R[2,1]], 'g')
        self.hz, = plt.plot([initpos[0] , initpos[0]+L*R[0,2]], [initpos[1] , initpos[1]+L*R[1,2]], [initpos[2] , initpos[2]+L*R[2,2]], 'b')

    def RotMat(self, r=0.0,p=0.0,y=0.0):
        R = np.array([[cos(p)*cos(y), sin(r)*sin(p)*cos(y)-cos(r)*sin(y), cos(r)*sin(p)*cos(y)+sin(r)*sin(y)],
                      [cos(p)*sin(y), sin(r)*sin(p)*sin(y)+cos(r)*cos(y), cos(r)*sin(p)*sin(y)-sin(r)*cos(y)],
                      [-sin(p), sin(r)*cos(p), cos(r)*cos(p)]])
        return R

    def update(self, pos, r, p, y=0):
        R = self.RotMat(r, p, y)
        L = 0.35
        self.hx.set_xdata([pos[0] , pos[0]+L*R[0,0]])
        self.hx.set_ydata([pos[1] , pos[1]+L*R[1,0]])
        self.hx.set_3d_properties([pos[2] , pos[2]+L*R[2,0]])
        self.hy.set_xdata([pos[0] , pos[0]+L*R[0,1]])
        self.hy.set_ydata([pos[1] , pos[1]+L*R[1,1]])
        self.hy.set_3d_properties([pos[2] , pos[2]+L*R[2,1]])
        self.hz.set_xdata([pos[0] , pos[0]+L*R[0,2]])
        self.hz.set_ydata([pos[1] , pos[1]+L*R[1,2]])
        self.hz.set_3d_properties([pos[2] , pos[2]+L*R[2,2]])
"""

def Safe_Barrier_3D(x, uhat, Kb):
    u = uhat.copy() #init u
    N = len(uhat)
    zscale = 3
    gamma = 5e-1
    Ds = 0.28
    H = 2*np.eye(3*N)
    f = -2*np.reshape(np.hstack(uhat.values()),(3*N,1))
    A = np.empty((0,3*N))
    b = np.empty((0,1))
    for i in range(N-1):
        for j in range(i+1,N):
            pr    = np.multiply(x[i][0,:]-x[j][0,:],np.array([1,1,1.0/zscale]))
            prd   = np.multiply(x[i][1,:]-x[j][1,:],np.array([1,1,1.0/zscale]))
            prdd  = np.multiply(x[i][2,:]-x[j][2,:],np.array([1,1,1.0/zscale]))

            # note elementwise operation
            h     = LA.norm(pr,4)**4 - Ds**4
            hd    = sum(4*pr**3*prd)
            hdd   = sum(12*pr**2*prd**2 + 4*pr**3*prdd)
            # hddd  = sum(24*pr*prd**3 + 36*pr**2*prd*prdd + 4*pr**3*prddd)
            # hddd = Lfh + Lgh*(ui-uj)
            Lfh   = sum(24*pr*prd**3 + 36*pr**2*prd*prdd)
            Lgh   = 4*pr**3*np.array([1,1,1.0/zscale])
            #- Lgh*(ui-uj) < gamma*h + Lfh
            Anew = np.zeros((3*N,))
            Anew[3*i:3*i+3] = - Lgh
            Anew[3*j:3*j+3] = Lgh
            bnew = gamma*np.dot(Kb,[h,hd,hdd]) + Lfh
            A = np.vstack([A, Anew])
            b = np.vstack([b, bnew])

    G = np.vstack([A, -np.eye(3*N), np.eye(3*N)])
    amax = 1e4
    h = np.vstack([b, amax*np.ones((3*N,1)), amax*np.ones((3*N,1))])
    sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(np.empty((0,3*N))), matrix(np.empty((0,1))))
    #sol = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b), matrix(np.empty((0,3*N))), matrix(np.empty((0,1))))
    x = sol['x']
    for i in range(N):
        u[i] = np.reshape(x[3*i:3*i+3],(1,3))
    return u

def s2textract(phat, x, s, dt):
    ks   = 10
    sd   = exp(-ks*norm(phat[0,:] - x[0,:])**2) #paramed time dynamics
    pd   = phat[1,:]*sd
    sdd  = -ks*exp(-ks*norm(phat[0,:] - x[0,:])**2)*2*np.dot(phat[0,:] - x[0,:],pd-x[1,:]) 
    pdd  = phat[2,:]*sd**2 + phat[1,:]*sdd
    sddd = ks*ks*exp(-ks*norm(phat[0,:] - x[0,:])**2)*4*(np.dot(phat[0,:] - x[0,:],pd-x[1,:]))**2-ks*2*exp(-ks*norm(phat[0,:] - x[0,:])**2)*(np.dot(pd-x[1,:],pd-x[1,:]) + np.dot(phat[0,:] - x[0,:],pdd-x[2,:])) 
    pddd = phat[3,:]*sd**3 + 3*phat[2,:]*sd*sdd + phat[1,:]*sddd
    phatnew = np.vstack([phat[0,:], pd, pdd, pddd])
    s = s + sd*dt
    return phatnew, s


def Invert_diff_flat_output(x, u):
    #note yaw =0 is fixed
    m = 35.89/1000
    g = 9.8
    beta1 = - x[2,0]
    beta2 = - x[2,2] + 9.8
    beta3 = x[2,1]

    roll = atan2(beta3,sqrt(beta1**2+beta2**2))
    pitch = atan2(beta1,beta2)
    yawrate = 0
    a_temp = LA.norm([0,0,g]-x[2,:])
    # acc g correspond to 49201
    thrust = int(a_temp/g*49201)
    return roll,pitch,yawrate,thrust

if __name__ == '__main__':
    rospy.init_node('cf_barrier', anonymous = True)
    fig = plt.figure()
    ax  = fig.gca(projection = '3d')
    ax.set_aspect('equal')
    ax.invert_zaxis()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([3, 2, 1.6]).max()
    Xb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][0].flatten()
    Yb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][1].flatten() 
    Zb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][2].flatten() - 1
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')
    plt.pause(.001)

    N   = 5 #number of agents
    #init p,pd,pdd,pddd,pdddd, on a circle
    p0 = dict()
    p1 = dict()
    pk = dict()
    rr = 0.45
    theta_N = np.linspace(0,2*PI,N-1, endpoint=False)
    
    for i in range(N-1):
        pk[i] = np.zeros([4,3])
        pk[i][0,:] = [rr*cos(theta_N[i]), rr*sin(theta_N[i]), -0.8]
        pk[i][1,:] = [-1.1*sin(theta_N[i]), 1.1*cos(theta_N[i]), 0]


    for i in range(N-1):
        p0[i] = np.zeros([4,3])
        p1[i] = np.zeros([4,3])
        p0[i][0,:] = [rr*cos(theta_N[i]), rr*sin(theta_N[i]), -0.8]
        p1[i][0,:] = [rr*cos(theta_N[(i+1)%4]), rr*sin(theta_N[(i+1)%4]), -0.8]
        p1[i][1,:] = [-1.1*sin(theta_N[(i+1)%4]), 1.1*cos(theta_N[(i+1)%4]), 0]



    # agent 4
    pc = dict()
    for ii in range(10):
        pc[ii] = np.zeros([4,3])
    RR = 0.9
    pc[0][0,:] = np.array([-RR,-RR,-0.8])
    pc[1][0,:] = np.array([RR, RR-0.2,-0.8])
    pc[2][0,:] = np.array([RR, -RR+0.2, -0.8])
    pc[3][0,:] = np.array([-RR, RR, -0.8])
    pc[4][0,:] = np.array([-RR+0.2, -RR, -0.8])
    pc[5][0,:] = np.array([RR+0.4, RR-0.5, -0.8])
    #pc[6][0,:] = np.array([-RR, RR, -0.8])
    #pc[7][0,:] = np.array([RR+0.2, -RR+0.2, -0.8])

    p0[4] = np.zeros([4,3])
    p0[4][0,:] = pc[0][0,:]

    Tf = [5, 3, 5, 3, 5, 3, 5, 3, 5]


    # init ploting handles
    Cfplt = dict() 
    cfs   = dict()
    CfCoord = dict()
    for i in range(N):
        #Cfplt[i] = Shpere3D(p0[i][0,:], ax)
        CfCoord[i] = Coord3D(ax, p0[i][0,:])
        cfs[i] = CF(p0[i][0,:],i)
    plt.draw()

    # pole placement for CLF and CBF
    AA=np.array([[0, 1, 0], [0, 0, 1],[0, 0, 0]]) 
    bb=np.array([[0], [0], [1]]) 
    #Kb=acker(AA,bb,[-2.2,-2.4,-2.6]);
    #Kb=acker(AA,bb,[-5.2,-5.4,-5.6]);
    Kb=np.asarray(acker(AA,bb,[-12.2,-12.4,-12.6]));

    # Bezier Interpolation
    Cout = dict()
    T = 4.0 #time to complete, also scaling factor
    for i in range(N-1):
        Cout[i] = BezierInterp(T, p0[i], p1[i])
    Cout[4] = BezierInterp(T, pc[0], pc[1])

    # start auto mode program
    CBF_on = 1
    tk = 0
    dt = 0.02
    t_total = 30
    phat = dict() #nominal state from interpolation
    uhat = dict() #nominal control
    x = dict() #actual states
    xd = dict() #derivative of x
    s = dict() #virtual time
    ks = 100
    for i in range(N):
        x[i] = p0[i][0:-1,:]
        xd[i] = np.array((4,3))
        s[i] = 0
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
    xx = dict()
    uu = dict()

    flag_done = 1
    flag_done0 = 1
    tk0 = 0
    t_real = 0
    kk0 = 0
    kk = 0 # waypoint index
    while not (rospy.is_shutdown() or t_real==int(t_total/dt)):
        tt0 = rospy.Time.now()
        if flag_done0 == 1 and kk <5:
            T0 = 1.0 #time to complete, also scaling factor
            print 'waypoint %d' %kk0
            for i in range(N-1):
                ktemp = (kk0+i)%(N-1)
                ktemp1 = (kk0+i+1)%(N-1)
                if t_real == 0:
                    Cout[i] = BezierInterp(T0, p0[i], p1[i])
                else:
                    Cout[i] = BezierInterp(T0, pk[ktemp], pk[ktemp1])
                #s[i] = 0
            #time.sleep(1)
            tk0 = 0
            flag_done0 = 0
            kk0 = kk0+1

        if flag_done == 1 and kk <5:
            T = Tf[kk] #time to complete, also scaling factor
            print 'waypoint %d' %kk
            Cout[4] = BezierInterp(T, pc[kk], pc[kk+1])
            s[4] = 0
            #time.sleep(1)
            tk = 0
            flag_done = 0
            kk = kk+1

        t = tk*dt
        try:
            print '----Current time is %0.2f----' %t

            # extract ref trajectories
            for i in range(N):
                if i == 4:
                    phat[i] = ref_extract(T, s[i], Cout[i])
                    # phat[i] = p0[i]
                else:                  
                    om = 1.0/4.0*2*PI
                    th = t_real*dt*om + i*PI/2
                    rr = 0.45
                    phat[i] = np.array([[rr*cos(th), rr*sin(th), -0.8],
                                        [-rr*om*sin(th), rr*om*cos(th), 0],
                                        [-rr*om**2*cos(th), -rr*om**2*sin(th), 0],
                                        [rr*om**3*sin(th), -rr*om**3*cos(th), 0]])
                phat[i], s[i] = s2textract(phat[i], x[i], s[i], dt)
                # compute nominal control from xd = AA*x + bb*u, u = ref(4)-Kb*(x-ref)
                uhat[i] = phat[i][3,:] - np.dot(Kb,x[i] - phat[i][0:-1,:])
                if LA.norm(uhat[i])> 1e4:
                    uhat[i] = uhat[i]/LA.norm(uhat[i])*1e4

            if CBF_on == 1:
                u = Safe_Barrier_3D(x, uhat, Kb)
                # ? if LA.norm(u)<LA.norm(uhat)*10: u = Safe_Barrier_3D(x, u*10, Kb)
            else:
                u = uhat.copy()

            # update actual dynamics
            for i in range(N):
                # Compute roll/pitch/yawrate/thrust
                roll,pitch,yawrate,thrust = Invert_diff_flat_output(x[i], u[i])
                # send to quads
                # cfs[i].goto(x[i][0,:])
                cfs[i].send_cmd_diff(roll, pitch, yawrate, thrust)
                #if i == 0:
                    #print roll*180/PI, pitch*180/PI
                # visualize coordinates
                CfCoord[i].update(x[i][0,:], roll, pitch, 0)
                xd[i] = np.dot(AA,x[i]) + np.dot(bb,u[i])
                x[i] = x[i] + xd[i]*dt   

            #visualize
            for i in range(N):
                cfs[i].pos = x[i][0,:]
                #Cfplt[i].update(cfs[i].pos,ax)
                
            if t_real%5 == 1:
                plt.pause(.001)
            #plt.draw()
            t_hist[t_real] = t
            p_hist[t_real] = x[0]
            phat_hist[t_real] = phat[0]
            ptrack_hist[t_real] = x[0][0,:]
            rpytrack_hist[t_real] = [roll, -pitch, yawrate, thrust]
            u_hist[t_real] = u[0][0,:]
            uhat_hist[t_real] = uhat[0][0,:]
            cmd_hist[t_real] = [roll, -pitch, yawrate, thrust]
            cmdreal_hist[t_real] = [roll, -pitch, yawrate, thrust]
            xx[t_real] = x.copy()
            uu[t_real] = u.copy()
            tk = tk+1
            tk0 = tk0 +1
            t_real = t_real + 1
            # check waypoint finished
            if tk*dt >= T and s[i]>T:
                flag_done = 1
            if tk0*dt >= T0:
                flag_done0 = 1
            print '----Actual dt is %0.3f----' %(rospy.Time.now()-tt0).to_sec()
        except rospy.ROSInterruptException:
            print '----Experiment interrupted!!!----'
            break
    # save data
    #f = open('cf5_sim_3rd.pckl', 'w')
    #pickle.dump([t_hist, p_hist, phat_hist, ptrack_hist, u_hist, uhat_hist, cmd_hist, cmdreal_hist, rpytrack_hist], f)
    #f.close()
    f = open('traj2017041201ctr.pckl', 'w')
    pickle.dump([dt, xx, uu], f)
    f.close()
    print '----Experiment completed!!!----'

    




