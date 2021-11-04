#!/usr/bin/env python3
"""
Calculates a GMM from a dataset
"""


import sklearn.mixture


def gmm(X, k):
    """
    Returns: pi, m, S, clss, bic
    """
    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    # Estimate model parameters with the EM algorithm.
    GMM.fit(X)
    pi = GMM.weights_
    m = GMM.means_
    S = GMM.covariances_
    # Predict the labels for the data samples in X using trained model.
    clss = GMM.predict(X)
    # Bayesian information criterion for the current model on the input X.
    # The lower the better.
    bic = GMM.bic(X)

    return pi, m, S, clss, bic
