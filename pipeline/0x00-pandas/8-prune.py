#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

# remove the entries in the pd.DataFrame where Close is NaN
df = df.dropna(subset=['Close'])

# YOUR CODE ENDS HERE

print(df.head())
