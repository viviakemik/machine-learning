import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def new_time_periods(df):

    df_ = df.copy()

    df_['YearMonthDay'] = [dt.date().isoformat() for dt in df_.index]
    
    df_['YearMonth'] = [f"{dt.year}-{dt.month:02d}" for dt in df_.index]
    
    df_['Week'] = [np.where(dt.date().day <= 7,'s1',
                       np.where(dt.date().day <= 14,'s2',
                                np.where(dt.date().day <= 21,'s3',
                                         's4'))) for dt in df_.index]
    
    col_list = df_.columns.tolist()
    first_cols = ['YearMonthDay', 'YearMonth', 'Week']
    columns = list(set(col_list) - set(first_cols))
    new_order = first_cols + columns
    df_ = df_[new_order]
 
    return df_