import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def remove_outliers(df, n_estimators = 100, freq_period = 7):

    isolation_forest = IsolationForest(n_estimators = n_estimators)

    start_date = df.index.min()
    end_date = df.index.max()

    freq = str(freq_period) + 'D'
    date_ranges = pd.date_range(start=start_date, end=end_date, freq = freq)

    for n, start in enumerate(date_ranges):
        end = start + pd.Timedelta(days = freq_period)
        mask = (df.index >= start) & (df.index < end)
        interval_df = df.loc[mask]
        isolation_forest = IsolationForest(n_estimators = n_estimators)
        anomalies = isolation_forest.fit_predict(interval_df.loc[:, ~interval_df.columns.isin(['YearMonthDay', 'YearMonth', 'Week'])])
        if n == 0:
            df_ = interval_df[anomalies == 1]
        else:
            df_ = pd.concat([df_, interval_df[anomalies == 1]])

    return df_