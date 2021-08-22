#!/usr/bin/env python3
"""
Updates a variable in place using the Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm
    - alpha is the learning rate
    - beta1 is the weight used for the first moment
    - beta2 is the weight used for the second moment
    - epsilon is a small number to avoid division by zero
    - var is a numpy.ndarray containing the variable to be updated
    - grad is a numpy.ndarray containing the gradient of var
    - v is the previous first moment of var
    - s is the previous second moment of var
    - t is the time step used for bias correction
    - Returns: the updated variable, the new first moment,
      and the new second moment, respectively
    """
    # Gradient Descent with Momentum
    v = beta1 * v + (1 - beta1) * grad
    # Gradient Descent with RMSProp
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    # bias correction
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)
    # updated variable
    var = var - alpha * v_corrected / ((s_corrected ** 0.5) + epsilon)
    return var, v, s
