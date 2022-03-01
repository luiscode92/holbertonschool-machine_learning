#!/usr/bin/env python3
"""
policy and policy_gradient functions.
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes to policy with a weight of a matrix.
    matrix: represents the current observation of the environment.
    weight: initial matrix of random weight.
    Returns: policy
    """
    z = matrix @ weight
    # softmax function
    # probabilities of actions: policy
    probs_actions = np.exp(z) / np.sum(np.exp(z))

    return probs_actions


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state
    and a weight matrix.
    Return: the action and the gradient (in this order)
    """
    # probabilities of actions
    probs = policy(state, weight)

    # random action
    action = np.random.choice(len(probs[0]), p=probs[0])

    # softmax gradient
    # probs shape (1, 2)
    # s shape (2, 1)
    s = probs.reshape(-1, 1)

    # softmax matrix
    softmax = np.diagflat(s) - s @ s.T

    # Take the obs for the action taken
    dsoftmax = softmax[action, :]

    # Derivative of natural logarithm
    dlog = dsoftmax / probs[0, action]

    gradient = state.T @ dlog[None, :]

    return action, gradient
