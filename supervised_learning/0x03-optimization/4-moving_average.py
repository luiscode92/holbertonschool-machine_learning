#!/usr/bin/env python3
"""
Calculates the weighted moving average of a data set
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set
    """
    moving_avg = []
    value = 0
    # Exponentially Weighted Averages with Bias Correction
    for i in range(len(data)):
        value = beta * value + (1 - beta) * data[i]
        # Bias Correction
        correct = value / (1 - (beta ** (i + 1)))
        moving_avg.append(correct)
    return moving_avg
