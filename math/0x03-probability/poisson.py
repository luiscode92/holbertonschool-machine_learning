#!/usr/bin/env python3
"""
Represents a poisson distribution
"""


e = 2.7182818285


class Poisson:
    """Represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor
        - data is a list of the data to be used to estimate the distribution
        - lambtha is the expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                # save lambtha as a float
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # calculate lambtha of the data
                self.lambtha = float(sum(data)/len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        k is the number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        return ((e**(self.lambtha*(-1))*self.lambtha**k)/self.factorial(k))

    def factorial(self, k):
        """Calculates the factorial of k"""
        if k == 0:
            return 1
        else:
            return (k * self.factorial(k-1))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        k is the number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(0, k+1):
            cdf += ((e**(self.lambtha*(-1))*self.lambtha**i)/self.factorial(i))
        return cdf
