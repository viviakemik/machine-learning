import numpy as np


def traditional_pivot(df_1, col_high, col_low, col_close):
    df = df_1.copy(deep=True)
    df['Pivot'] = (df[col_high] + df[col_low] + df[col_close]) / 3
    df['S1'] = (df['Pivot'] * 2) - df[col_high]
    df['S2'] = df['Pivot'] - (df[col_high] - df[col_low])
    df['S3'] = df[col_low] - (2 * (df[col_high] - df['Pivot']))
    df['R1'] = (df['Pivot'] * 2) - df[col_low]
    df['R2'] = df['Pivot'] + df[col_high] - df[col_low]
    df['R3'] = df[col_high] + (2 * (df['Pivot'] - df[col_low]))

    return df


def woodie_pivot(df_1, col_high, col_low, col_close):
    df = df_1.copy(deep=True)
    df['Pivot'] = (df[col_high] + df[col_low] + (2 * df[col_close])) / 4
    df['R1'] = (2 * df['Pivot']) - df[col_low]
    df['R2'] = df['Pivot'] + df[col_high] - df[col_low]
    df['S1'] = (2 * df['Pivot']) - df[col_high]
    df['S2'] = df['Pivot'] - df[col_high] + df[col_low]

    return df


def camarilla_pivot(df_1, col_high, col_low, col_close):
    df = df_1.copy(deep=True)
    df['Pivot'] = (df[col_high] + df[col_low] + df[col_close]) / 3
    df['S1'] = df[col_close] - ((df[col_high] - df[col_low]) * 1.083)
    df['S2'] = df[col_close] - ((df[col_high] - df[col_low]) * 1.16)
    df['S3'] = df[col_close] - ((df[col_high] - df[col_low]) * 1.25)
    df['S4'] = df[col_close] - ((df[col_high] - df[col_low]) * 1.5)
    df['R1'] = df[col_close] + ((df[col_high] - df[col_low]) * 1.083)
    df['R2'] = df[col_close] + ((df[col_high] - df[col_low]) * 1.16)
    df['R3'] = df[col_close] + ((df[col_high] - df[col_low]) * 1.25)
    df['R4'] = df[col_close] + ((df[col_high] - df[col_low]) * 1.5)

    return df