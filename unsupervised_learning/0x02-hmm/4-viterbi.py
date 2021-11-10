#!/usr/bin/env python3
"""
Calculates the most likely sequence of hidden states for a hidden markov model
"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Returns: path, P, or None, None on failure
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

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    # Initialization step
    viterbi[:, 0] = Initial.transpose() * Emission[:, Observation[0]]

    # Recursion
    for t in range(1, T):
        a = viterbi[:, t - 1]
        b = Transition.transpose()
        ab = a * b
        ab_max = np.amax(ab, axis=1)
        c = Emission[:, Observation[t]]
        prob = ab_max * c

        viterbi[:, t] = prob
        backpointer[:, t - 1] = np.argmax(ab, axis=1)

    # Path initialization
    path = []
    current = np.argmax(viterbi[:, T - 1])
    path = [current] + path

    # Path backwards traversing
    for t in range(T - 2, -1, -1):
        current = int(backpointer[current, t])
        path = [current] + path

    # Probability of obtaining the path sequence

    P = np.amax(viterbi[:, T - 1], axis=0)

    return path, P
