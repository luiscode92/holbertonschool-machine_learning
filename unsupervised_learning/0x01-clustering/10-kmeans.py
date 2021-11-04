#!/usr/bin/env python3
"""
Performs K-means on a dataset
"""


import sklearn.cluster


def kmeans(X, k):
    """
    Returns: C, clss
    """
    k_model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = k_model.cluster_centers_
    clss = k_model.labels_

    return C, clss
