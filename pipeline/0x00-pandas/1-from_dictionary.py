#!/usr/bin/env python3
"""
Script that creates a pd.DataFrame from a dictionary
"""
import pandas as pd


my_dict = {"First": [0.0, 0.5, 1.0, 1.5],
           "Second": ["one", "two", "three", "four"]}

df = pd.DataFrame(data=my_dict, index=["A", "B", "C", "D"])
