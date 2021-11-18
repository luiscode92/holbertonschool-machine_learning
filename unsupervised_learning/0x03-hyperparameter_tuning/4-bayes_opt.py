#!/usr/bin/env python3
"""
Performs Bayesian optimization on a noiseless 1D Gaussian process
"""
import numpy as np
from scipy.stats import norm
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

        # exploration-exploitation factor for acquisition
        self.xsi = xsi

        # bool determining whether optimization should be performed for
        # minimization (True) or maximization (False)
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        """
        mu_s, sigma_s = self.gp.predict(self.X_s)

        if self.minimize is True:
            Y_s = np.min(self.gp.Y)
            imp = Y_s - mu_s - self.xsi

        else:
            Y_s = np.max(self.gp.Y)
            imp = mu_s - Y_s - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sigma_s
            EI = imp * norm.cdf(Z) + sigma_s * norm.pdf(Z)
            EI[sigma_s == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
