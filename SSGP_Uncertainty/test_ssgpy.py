"""Test Python interface for SSGP."""

import time
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
from ssgpy import SSGPy as SSGP  # change if ssgpy.so is in another directory


def optimize_hyperparameters(ell, sf2, sn2, inputs, targets, test_inputs,
                             test_outputs):
    """Optimize hyperparameters of a full GP with SqARD kernel function."""
    N, n = inputs.shape

    sf2_kernel = ConstantKernel(constant_value=sf2)
    rbf_kernel = RBF(length_scale=ell)
    sn2_kernel = WhiteKernel(noise_level=sn2)

    kernel = sf2_kernel * rbf_kernel + sn2_kernel

    mygp = gp.GaussianProcessRegressor(kernel=kernel, alpha=0.0,
                                       optimizer='fmin_l_bfgs_b',
                                       n_restarts_optimizer=3)
    mygp.fit(X, Y)

    pred_mean = mygp.predict(test_inputs)
    plt.plot(test_inputs, pred_mean, 'go')

    return np.exp(kernel.theta)


if __name__ == '__main__':
    n = 1
    k = 1
    D = 25
    Ntrh = 100
    Nts = 1000

    X = np.random.randn(Ntrh, n)
    Y = np.sinc(X) + np.random.randn(Ntrh, n) * 0.25
    Xtest = np.random.randn(Nts, n)
    Ytest = np.sinc(Xtest)

    hp = optimize_hyperparameters(np.ones(n), 1.0, 0.37, X, Y, Xtest, Ytest)
    ell = np.array(hp[0]).reshape(n, 1)
    sf2 = np.array(hp[1]).reshape(n, 1)
    sn2 = np.array(hp[2]).reshape(n, 1)

    ssgp = SSGP(n, k, D, ell, sf2, sn2)

    start_update = time.time()
    ssgp.update(X, Y)
    end_update = time.time()

    start_pred = time.time()
    pred = ssgp.predict_mean(Xtest)
    end_pred = time.time()

    plt.plot(Xtest, pred, 'bo')
    plt.plot(Xtest, Ytest, 'ro')
    fullgp = mpatches.Patch(color='green', label='Full GP')
    sparsegp = mpatches.Patch(color='blue', label='SSGP')
    actual = mpatches.Patch(color='red', label='Actual')
    plt.legend([fullgp, sparsegp, actual], ['Full GP', 'SSGP', 'Actual'])
    plt.show()

    print "{0} ms to perform {1} updates".format((end_update - start_update) *
                                                 1000, Ntrh)
    print "{0} ms per update".format((end_update - start_update) * 1000 / Ntrh)

    print "{0} ms to perform {1} predictions".format((end_pred - start_pred) *
                                                     1000, Nts)
    print "{0} ms per prediction".format((end_pred - start_pred) * 1000 / Nts)
