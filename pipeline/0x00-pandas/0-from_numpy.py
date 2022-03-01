#!/usr/bin/env python3
"""
Creates a 'pd.DataFrame' from a 'np.ndarray'
"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """
    Returns: the newly created 'pd.DataFrame'
    """
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    columns = labels[0:np.shape(array)[1]]
    df = pd.DataFrame(data=array, columns=columns)
    return df
