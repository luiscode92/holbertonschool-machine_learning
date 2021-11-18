#!/usr/bin/env python3
"""
Performs Bayesian optimization on a noiseless 1D Gaussian process
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init,
                 bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor
        """
        # black-box function to be optimized
        self.f = f

        # instance of the class GaussianProcess
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # array containing all acquisition sample points,
        # evenly spaced between min and max
        # (1x50)
        self.X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)

        # (50x1)
        self.X_s = self.X_s.reshape(-1, 1)

        # exploration-exploitation factor
        self.xsi = xsi
        self.minimize = minimize
