from lwpr import *
import numpy as np
from random import *

import pickle

import matplotlib.patches
import matplotlib.pyplot as plt



if __name__ == '__main__':
    # load sim data (x is 9 dimensional, y is 3 dimensional)
    f = open('../Sim_data/genSimData/Sim_ground_truth02.pckl')
    dt, xhist, yhist, yreal =  pickle.load(f)
    f.close()
    
    # extract time sequence
    tN = len(xhist[:,0])
    t = np.linspace(0, dt*(tN-1), tN)

    # initialize model (input dim = 9, output dim = 3)
    model = LWPR(9,3)    
    model.init_D = 20*np.eye(9)
    model.update_D = True
    model.init_alpha = 40*np.ones([9,9])
    model.meta = False

    # train model
    for k in range(20):
        ind = np.random.permutation(tN)
        mse = 0 

        for i in range(tN):
            yp = model.update(xhist[ind[i]], yhist[ind[i]])
            mse = mse + np.linalg.norm(yreal[ind[i], :] - yp)

            nMSE = mse/tN/var(yreal)
            print "#Data: %5i  #RFs: %3i  nMSE=%5.3f"%(model.n_data, model.num_rfs, nMSE)

    # show model parameters:
    #print model
    #print model.kernel

    # test the model
    ypred = np.zeros((tN,1))
    Conf = np.zeros((tN,1))
 
    for k in range(tN):
        ypred[k,:], Conf[k,:] = model.predict_conf(array([xhist[k]]))
 
    # visualize yhist, ypred, and yreal
    fig = plt.figure(1)
    plt.subplot(311)
    plt.plot(t, yreal[:,0], 'g-', label='acc x (real)')
    plt.plot(t, yhist[:,0], 'b-', label='acc x (hist)')
    plt.plot(t, ypred[:,0], 'r-', label='acc x (pred)')

    plt.subplot(312)
    plt.plot(t, yreal[:,1], 'g-', label='acc y (real)')
    plt.plot(t, yhist[:,1], 'b-', label='acc y (hist)')
    plt.plot(t, ypred[:,1], 'r-', label='acc y (pred)')

    plt.subplot(313)
    plt.plot(t, yreal[:,2], 'g-', label='acc z (real)')
    plt.plot(t, yhist[:,2], 'b-', label='acc z (hist)')
    plt.plot(t, ypred[:,2], 'r-', label='acc z (pred)')
    
    plt.show()

    # visualize with conf-int
    #plt.plot(Xtr, Ytr, 'r.') 
    #plt.plot(Xtest,Ytest,'b-')
    #plt.plot(Xtest,Ytest+Conf,'c-', linewidth=2)
    #plt.plot(Xtest,Ytest-Conf,'c-', linewidth=2) 
    #plt.show()

    # Display resulting model and save
    #print model
    #print model.kernel
    #model.write_XML("LWPR_model.xml")
