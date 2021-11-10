#!/usr/bin/env python3
"""
Performs the backward algorithm for a hidden markov model
"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Returns: P, B, or None, None on failure
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None

    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None

    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]

    N, _ = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None

    if not np.sum(Transition, axis=1).all():
        return None, None

    if not np.sum(Initial) == 1:
        return None, None

    B = np.zeros((N, T))

    # Initialization step
    B[:, T - 1] = np.ones(N)

    # Recursion
    for t in range(T - 2, -1, -1):
        prob = \
            np.sum(B[:, t + 1] *
                   Emission[:, Observation[t + 1]] * Transition, axis=1)
        B[:, t] = prob

    # Likelihood of the observations given the model
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
