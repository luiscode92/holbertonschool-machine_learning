#!/usr/bin/env python3
"""
Represents a binomial distribution
"""


class Binomial:
    """Represents a binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor
        - data is a list of the data to be used to estimate the distribution
        - n is the number of Bernoulli trials
        - p is the probability of a “success”
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                # Saves n as an integer and p as a float
                self.n = int(n)
                self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # Calculate n and p from data
                # Round n to the nearest integer
                # Calculate p first and then calculate n. Then recalculate p
                mean = float(sum(data)/len(data))
                variance = float(sum((x - mean)**2 for x in data) /
                                 len(data))
                p = 1 - variance/mean
                self.n = round(mean/p)
                # Recalculating p for more precision because of rounding n
                self.p = float(mean/self.n)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        k is the number of “successes”
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        return (self.factorial(self.n)*self.p**k*(1-self.p)**(self.n - k) /
                (self.factorial(self.n - k)*self.factorial(k)))

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
        if k < 0 or k > self.n:
            return 0
        cdf = 0
        for i in range(0, k+1):
            cdf += self.pmf(i)
        return cdf
