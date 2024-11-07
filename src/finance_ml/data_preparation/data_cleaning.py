import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def remove_duplicated_values_by_index(df):

    df_ = df[~df.index.duplicated()]

    return df_


def treatments_for_missing(df, type = 1):

    # types of treatments
    #  1 - remove missing
    #  2 - input mean values
    #  3 - input zero

    df_ = df.copy()

    if type == 1:

        # first replace any kind of missing values with NaN
        df_ = df_.replace(r'^\s+$|^\t+$|^$', np.nan, regex = True)
        
        # drop all rows with some NaN values
        df_ = df_.dropna(axis=0)
    
    return df_