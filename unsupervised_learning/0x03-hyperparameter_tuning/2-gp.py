#!/usr/bin/env python3
"""
http://krasserm.github.io/2018/03/19/gaussian-processes/
Represents a noiseless 1D Gaussian process
"""


import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        """
        # (t, 1) inputs already sampled with the black-box function
        self.X = X_init
        # (t, 1) outputs of the black-box function for each input in X
        self.Y = Y_init
        # length parameter for the kernel
        self.l = l
        # standard deviation given to the output of the black-box function
        self.sigma_f = sigma_f
        # Current covariance kernel matrix for the Gaussian process
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        """
        sqdist = \
            np.sum(X1**2, 1).reshape(-1, 1) \
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points
        in a Gaussian process
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        # mean
        # (10x1)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        # (1x10)
        mu_s = mu_s.reshape(-1)

        # (10x10) covariance
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        # (1x10) variance
        var_s = np.diag(cov_s)

        return mu_s, var_s

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process
        """
        # (3,)
        self.X = np.append(self.X, X_new)
        # (3, 1)
        self.X = self.X[:, np.newaxis]

        # (3,)
        self.Y = np.append(self.Y, Y_new)
        # (3, 1)
        self.Y = self.Y[:, np.newaxis]

        # current covariance kernel matrix for the Gaussian process
        self.K = self.kernel(self.X, self.X)
