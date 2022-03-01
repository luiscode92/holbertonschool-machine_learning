#!/usr/bin/env python3
"""
Loads data from a file as a pd.DataFrame
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Returns: the loaded pd.DataFrame
    """
    df = pd.read_csv(filepath_or_buffer=filename, delimiter=delimiter)

    return df
