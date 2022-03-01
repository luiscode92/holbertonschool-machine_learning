#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

# The column Weighted_Price should be removed
df = df.drop(columns=['Weighted_Price'])

# missing values in Close should be set to the previous row value
df["Close"].fillna(method='ffill', inplace=True)

# missing values in High, Low, Open
# should be set to the same row’s Close value
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)

# missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

print(df.head())
print(df.tail())
