#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

# take the last 10 rows
A = df.tail(10)

# columns High and Close and convert them into a numpy.ndarray
A = A.loc[:, ['High', 'Close']].to_numpy()

# YOUR CODE ENDS HERE

print(A)
