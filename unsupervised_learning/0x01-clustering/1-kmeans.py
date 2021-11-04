#!/usr/bin/env python3
"""
Initializes cluster centroids for K-means.
Performs K-means on a dataset.
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Returns: C, clss, or None, None on failure
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(k) is not int or k <= 0:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    C = initialize(X, k)

    for _ in range(iterations):
        C_copy = np.copy(C)
        # new axis for broadcasting
        dist = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        # minimum distance
        clss = np.argmin(dist, axis=-1)
        # move the centroids
        for j in range(k):
            index = np.argwhere(clss == j)
            # If a cluster contains no data points during the update step,
            # reinitialize its centroid
            if not len(index):
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[index], axis=0)

        # If no change in the cluster centroids occurs between iterations,
        # return
        if (C_copy == C).all():
            return C, clss

    # new axis for broadcasting
    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
    clss = np.argmin(dist, axis=-1)

    return C, clss


def initialize(X, k):
    """
    Returns: a numpy.ndarray of shape (k, d)
    containing the initialized centroids for each cluster, or None on failure
    """

    d = X.shape[1]
    mini = np.amin(X, axis=0)
    maxi = np.amax(X, axis=0)

    # Initialize cluster centroids
    init = np.random.uniform(mini, maxi, size=(k, d))

    return init
