#!/usr/bin/env python
import sys
import math
from math import pi as PI
from math import atan2, sin, cos, sqrt, fabs, floor, exp
from scipy.misc import factorial as ft
from scipy.misc import comb as nCk
from numpy import linalg as LA
from numpy.linalg import pinv
import GPy

import matplotlib.patches
import matplotlib.pyplot as plt

import time 
import numpy as np
from cvxopt import matrix, solvers
import pickle #save multiple objects
from multiprocessing import Pool # multi-prcess for loop

class GP_pred():
    def __init__(self, xinit, yinit):
        # gp1: variance 0.57, lengthscal 3.25
        # gp2: variance 0.14, lengthscal 1.75
        # gp3: variance 9163, lengthscal 40
        # gp1: variance 51.8, lengthscal 3.8
        # gp3: variance 9.16, lengthscal 0.8
        self.sigmaf = 10.8#2.0#15.2#2.0
        self.sigmal = 0.6#3.8#3.0#3.8
        self.Nlimit = 300
        self.Kinv = 1/self.sigmaf**2
        # data in GP
        self.xx = np.array([xinit])
        self.y1 = np.array([yinit[0]])
        self.y2 = np.array([yinit[1]])
        self.y3 = np.array([yinit[2]])

        # data to remember
        self.xmem  = np.empty((0,xinit.size))
        self.y1mem = np.empty((0,))
        self.y2mem = np.empty((0,))
        self.y3mem = np.empty((0,))



    def Predict(self, xQuery):
        N1 = self.xx.shape[0]
        print N1
        # use xpred, xx, yy, Kinv to predict ypred
        ktemp = np.sum((self.xx-np.kron(np.ones((N1,1)),xQuery[None,:]))**2,axis=1)
        k = self.sigmaf**2*np.exp(-0.5*ktemp/self.sigmal**2)
        c = self.sigmaf**2
        if N1 == 30:
            print self.xx
            print ktemp
            print k
        my1 = np.dot(np.dot(k,self.Kinv),self.y1)
        my2 = np.dot(np.dot(k,self.Kinv),self.y2)
        my3 = np.dot(np.dot(k,self.Kinv),self.y3)
        Vy = c - np.dot(np.dot(k,self.Kinv),k)
        yQuery = np.array([my1, my2, my3])
        return yQuery, Vy

    def Reset(self):
        N = self.xx.shape[0]
        K = np.empty((N,0))
        for i in range(N):
            kxtemp = np.sum((self.xx-np.kron(np.ones((N,1)), self.xx[i,:]))**2, axis=1)
            kx = self.sigmaf**2*np.exp(-0.5*kxtemp/self.sigmal**2)
            K = np.hstack((K, kx[:,None]))
        self.Kinv = pinv(K)

    def Update(self, xsample, ysample):
        # check add data or not
        # kernal function is squared exponential kernal
        # kse = sigmaf^2*exp(-0.5*(xi-xj)'*(xi-xj)/sigmal^2);
        xcomb = np.vstack((self.xx, self.xmem))
        N = xcomb.shape[0]
        kxtemp = np.sum((xcomb-np.kron(np.ones((N,1)), xsample[None,:]))**2, axis=1)
        kx = self.sigmaf**2*np.exp(-0.5*kxtemp/self.sigmal**2)
        if max(kx) <= self.sigmaf**2*0.90:#0.93: # distinct, add new data
            # gp1 0.9, gp2, 0.9, gp3 
            if self.xx.shape[0]>=self.Nlimit: # remove a data point from GP
                N1 = self.xx.shape[0]
                kxxtemp = np.sum((self.xx-np.kron(np.ones((N1,1)), xsample[None,:]))**2, axis=1)
                kind = kxxtemp.argmin()
                # move to mem
                self.xmem  = np.vstack((self.xmem, self.xx[kind]))
                self.y1mem = np.append(self.y1mem, self.y1[kind])
                self.y2mem = np.append(self.y2mem, self.y2[kind])
                self.y3mem = np.append(self.y3mem, self.y3[kind])
                # delete from GP
                mask = np.ones(len(self.xx), dtype = bool)
                mask[kind] = False
                self.xx = self.xx[mask,...]
                self.y1 = self.y1[mask,...]
                self.y2 = self.y2[mask,...]
                self.y3 = self.y3[mask,...]
                Kperm = np.vstack((self.Kinv[mask,:], self.Kinv[kind,:]))
                Kperm = np.hstack((self.Kinv[:,mask], self.Kinv[:,kind][:,None]))
                Ka = Kperm[0:-1,0:-1]
                Kb = Kperm[0:-1,-1]
                Kc = Kperm[-1,-1] #note Kc is 1by1, thus inverse is 1.0/Kc
                    
                self.Kinv = Ka - np.dot(np.dot(1.0/Kc, Kb), Kb)
            # add new data to GP
            N2 = self.xx.shape[0]
            k2temp = np.sum((self.xx-np.kron(np.ones((N2,1)), xsample[None,:]))**2, axis=1)
            k2 = self.sigmaf**2*np.exp(-0.5*k2temp/self.sigmal**2)
            c2 = self.sigmaf**2
            blockA = self.Kinv + np.dot(self.Kinv, k2[:,None])*np.dot(k2[None,:],self.Kinv)/(c2-np.dot(np.dot(self.Kinv,k2),k2))
            blockB = -np.dot(self.Kinv,k2[:,None])/(c2-np.dot(np.dot(self.Kinv,k2),k2))
            blockC = blockB.T
            blockD = np.array([[1.0/(c2-np.dot(np.dot(self.Kinv, k2), k2))]])
            self.Kinv = np.asarray(np.bmat([[blockA, blockB],[blockC, blockD]]))
            self.xx = np.vstack((self.xx, xsample))
            self.y1= np.append(self.y1, ysample[0])
            self.y2= np.append(self.y2, ysample[1])
            self.y3= np.append(self.y3, ysample[2])




if __name__ == '__main__':
    # simulate signal
    N = 100
    t = np.arange(0, 5, 5.0/N)
    yhist = np.empty((0,3))
    ypredhist = np.empty((0,3))
    xall = np.vstack((np.sin(t), np.cos(t), t**2, t+10, 10*np.exp(-t/10), t**3, 2*t-10, 3*t+3, t**2-t))
    xnew = xall[:,0]
    ynew = np.array([xall[0,0], xall[1,0], xall[2,0]])
    gp = GP_pred(xnew, ynew)
    for i in range(1,N):
        xnew = xall[:,i]
        ynew = np.array([xall[0,i], xall[1,i], xall[2,i]])
        gp.Update(xnew, ynew)
        xpred = np.array([sin(t[i]),cos(t[i]),(t[i])**2, t[i]+10, 10*np.exp(-t[i]/10), t[i]**3, 2*t[i]-10, 3*t[i]+3, t[i]**2-t[i]])
        ypredict, Vy = gp.Predict(xpred)
        yhist = np.vstack((yhist, ynew[None,:]))
        ypredhist = np.vstack((ypredhist, ypredict[None,:]))


    # get optimal parameters using GPy
    kernel = GPy.kern.RBF(input_dim=9, variance=1., lengthscale=1.)
    yall = np.hstack((gp.y1[:,None], gp.y2[:,None], gp.y3[:,None]))
    m = GPy.models.GPRegression(gp.xx,gp.y1[:,None],kernel)
    print m
    print m.rbf.gradient
    m.optimize()
    print m.rbf.gradient
    print m.rbf.variance
    print m.rbf.lengthscale
    print m.Gaussian_noise.variance
   



    plt.close('all')
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(range(0,yhist.shape[0]), yhist[:,0])
    axarr[0].plot(range(0,ypredhist.shape[0]), ypredhist[:,0], '*')

    axarr[1].plot(range(0,yhist.shape[0]), yhist[:,1])
    axarr[1].plot(range(0,ypredhist.shape[0]), ypredhist[:,1], '*')

    axarr[2].plot(range(0,yhist.shape[0]), yhist[:,2])
    axarr[2].plot(range(0,ypredhist.shape[0]), ypredhist[:,2], '*')
    plt.show()
