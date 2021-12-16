#!/usr/bin/env python3
"""
Preprocess raw dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def preprocess_raw_data():
    """
    Returns: train_df, val_df, test_df
    """
    # making data frame from csv file
    data = \
        pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

    data = data.drop(["Open", "High", "Low", "Volume_(BTC)",
                     "Volume_(Currency)", "Weighted_Price"], axis=1)

    df = data.dropna()

    df = df[0::60]

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.reset_index(inplace=True, drop=True)

    df = df[df['Timestamp'].dt.year >= 2017]
    df.reset_index(inplace=True, drop=True)

    print(df)

    print(df.describe().transpose())

    date_time = pd.to_datetime(df.pop('Timestamp'), format='%Y-%m-%d %H:%M:%S')
    # print(df)
    plot_features = df['Close']
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)

    column_indices = {name: i for i, name in enumerate(df.columns)}

    # Split the data
    # Training 70%, validation 20%, test 10%
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    # Normalize the data
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df
