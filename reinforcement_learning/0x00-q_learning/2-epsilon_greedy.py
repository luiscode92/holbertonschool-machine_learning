#!/usr/bin/env python3
"""
Uses epsilon-greedy to determine the next action
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Returns: the next action index
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        """
        Explore: select a random action
        """
        action = np.random.randint(Q.shape[1])
    else:
        """
        Exploit: select the action with max value (future reward)
        """
        action = np.argmax(Q[state, :])

    return action
