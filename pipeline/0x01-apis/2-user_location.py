#!/usr/bin/env python3
"""
GitHub API
Script that prints the location of a specific user.
Your code should not be executed when the file is imported.
"""
import requests
import sys
import time


if __name__ == '__main__':
    url = sys.argv[1]

    # https://api.github.com/users/holbertonschool

    my_status = requests.get(url).status_code

    if my_status == 200:
        print(requests.get(url).json()["location"])

    elif my_status == 404:
        print("Not found")

    elif my_status == 403:
        """
        Reset in X min. X is the number of minutes from now and the value of
        X-Ratelimit-Reset
        """
        Now = int(time.time())
        Reset = int(requests.get(url).headers['X-Ratelimit-Reset'])
        seconds = Reset - Now
        X = seconds / 60
        print("Reset in {} min".format(int(X)))
