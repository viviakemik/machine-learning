import sys
sys.path.append('../')
from src.finance_ml.data_preparation.data_preparation import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import copy


def calculate_SMA(data, window_size):  # SMA function
    series = pd.Series(data)
    sma = series.rolling(window=window_size).mean()
    sma_list = sma.dropna().tolist()
    return sma_list


def calculate_triangular_moving_average(data, window_size):   # TMA function
    SMA = calculate_SMA(data, window_size)

    n = len(SMA)
    moving_average = []

    for i in range(n):
        if i + window_size <= len(SMA):
            subset = SMA[i:i + window_size]
            subset = list(subset)

            mid_point = len(subset) // 2 # calculating weights in a trangle manner
            weights = list(range(1, mid_point + 2)) + list(range(mid_point, 0, -1))

            average = sum(val * weight for val, weight in zip(subset, weights)) / sum(weights)
            moving_average.append(average)
        else:
            moving_average.append(None)

    return moving_average
