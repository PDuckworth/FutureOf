
from __future__ import print_function, absolute_import

import sys
import gpflow
import numpy as np
import tensorflow as tf

from gpflow.models import GPR
# from gpflow.core.autoflow import AutoFlow
from gpflow.decors import autoflow

float_type = gpflow.settings.dtypes.float_type


class GPR_with_grad(GPR):
    def __init__(self, X, Y, kern):
        GPR.__init__(self, X, Y, kern)

    @autoflow((float_type, [None, None]))
    def compute_posterior_grad_at(self, Xnew):
        """
        Compute the gradient of the posterior mean function at a specific valuem, Xnew.
        """
        pred_mean, pred_var = self._build_predict(Xnew)
        return tf.gradients(pred_mean, Xnew)


if __name__ == "__main__":

    k = gpflow.kernels.RBF(2)

    X = np.array([[1.,1.],[2.,2.],[3,3]]).reshape(3,2)
    Y = np.array([1.,2.,1.]).reshape(3,1)

    N = 12
    X = np.random.rand(N,2)*10
    g = np.random.randn(1)
    print("random gradient = ",g)
    Y = g*(X.T)[0]
    Y = Y.reshape(N,1)

    print(X)
    print(Y)

    m = GPR_with_grad(X, Y, k)
    m.likelihood.variance = 0.13

    gpflow.train.ScipyOptimizer().minimize(m)

    X_val = np.array([0.5,0.5]).reshape(1,2)
    print(X_val)
    print(m.compute_posterior_grad_at(X_val))

    print(m)
