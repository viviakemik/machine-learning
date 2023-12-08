import numpy as np
import pandas as pd


def traditional_pivot(df_1,col_high, col_low, col_close):
    df = df_1.copy(deep=True)
    df['Pivot'] = (df[col_high] + df[col_low] + df[col_close]) / 3
    df['S1'] = (df['Pivot'] * 2) - df[col_high]
    df['S2'] = df['Pivot'] - (df[col_high] - df[col_low])
    df['S3'] = df[col_low] - (2 * (df[col_high] - df['Pivot']))
    df['R1'] = (df['Pivot'] * 2) - df[col_low]
    df['R2'] = df['Pivot'] + df[col_high] - df[col_low]
    df['R3'] = df[col_high] + (2 * (df['Pivot'] - df[col_low]))
    df['Pivot'] = df['Pivot'].shift(1)
    df['S1'] = df['S1'].shift(1)
    df['S2'] = df['S2'].shift(1)
    df['S3'] = df['S3'].shift(1)
    df['R1'] = df['R1'].shift(1)
    df['R2'] = df['R2'].shift(1)
    df['R3'] = df['R3'].shift(1)
    df.dropna(inplace=True)
    df = df[['DATE','Pivot','S1','S2','S3','R1','R2','R3']]
    return df


def woodie_pivot(df_1, col_high, col_low, col_close):
    df = df_1.copy(deep=True)
    df['Pivot'] = (df[col_high] + df[col_low] + (2 * df[col_close])) / 4
    df['R1'] = (2 * df['Pivot']) - df[col_low]
    df['R2'] = df['Pivot'] + df[col_high] - df[col_low]
    df['S1'] = (2 * df['Pivot']) - df[col_high]
    df['S2'] = df['Pivot'] - df[col_high] + df[col_low]
    df['S3'] = df['S1'] - (df[col_high] - df[col_low])
    df['R3'] = df['R1'] + (df[col_high] - df[col_low])
    df['S1'] = df['S1'].shift(1)
    df['S2'] = df['S2'].shift(1)
    df['S3'] = df['S3'].shift(1)
    df['R1'] = df['R1'].shift(1)
    df['R2'] = df['R2'].shift(1)
    df['R3'] = df['R3'].shift(1)
    df = df[['DATE','Pivot','S1','S2','S3','R1','R2','R3']]
    df.dropna(inplace=True)

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
    df['S1'] = df['S1'].shift(1)
    df['S2'] = df['S2'].shift(1)
    df['S3'] = df['S3'].shift(1)
    df['S4'] = df['S4'].shift(1)
    df['R1'] = df['R1'].shift(1)
    df['R2'] = df['R2'].shift(1)
    df['R3'] = df['R3'].shift(1)
    df['R4'] = df['R4'].shift(1)
    df.dropna(inplace=True)

    df = df[['DATE','Pivot','S1','S2','S3','S4','R1','R2','R3','R4']]
    return df


def resample_ohlc(raw_data, col_names, timeframe):
    resampled_data = pd.DataFrame()
    open_col = col_names[0]
    high_col = col_names[1]
    low_col = col_names[2]
    close_col = col_names[3]

    resampled_data = raw_data.resample(timeframe).agg({open_col:'first',
                                                             high_col:'max',
                                                             low_col:'min',
                                                             close_col:'last'
                                                             })
    if timeframe == '1W':
        resampled_data.index = resampled_data.index + pd.DateOffset(weekday=0)

    resampled_data.reset_index(inplace=True)
    resampled_data.dropna(inplace=True)
    return resampled_data

def merge_ohlc_pivot(ohlc_data, pivot_data):
    ohlc_data['MERGE_DATE'] = pd.to_datetime(ohlc_data['DATE']).dt.date
    pivot_data['DATE'] = pd.to_datetime(pivot_data['DATE']).dt.date
    pivot_data.rename(columns={'DATE':'MERGE_DATE'}, inplace=True)
    merged_data = pd.merge(ohlc_data, pivot_data, on=['MERGE_DATE'], how='left')
    merged_data.sort_values(by='DATE', ascending=True, inplace=True)
    merged_data = merged_data.ffill()
    merged_data.drop(columns=['MERGE_DATE'], inplace=True)
    merged_data.dropna(inplace=True)
    return merged_data

#
# def generate_buy(merged_data_1,close_col,open_col):
#     merged_data = merged_data_1.copy(deep=True)
#     merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])
#     merged_data.sort_values(by='DATE', ascending=True, inplace=True)
#     merged_data['Buy_sell'] = np.where( (((merged_data[close_col].shift(1) < merged_data['S3']) & (merged_data[close_col] > merged_data['S3']) & (merged_data[close_col] > merged_data[open_col])) |
#                                         ((merged_data[close_col].shift(1) < merged_data['S2']) & (merged_data[close_col] > merged_data['S2']) & (merged_data[close_col] > merged_data[open_col]))  |
#                                         ((merged_data[close_col].shift(1) < merged_data['S1']) & (merged_data[close_col] > merged_data['S1']) & (merged_data[close_col] > merged_data[open_col])) |
#                                         ((merged_data[close_col].shift(1) < merged_data['Pivot']) & (merged_data[close_col] > merged_data['Pivot']) & (merged_data[close_col] > merged_data[open_col])) |
#                                         ((merged_data[close_col].shift(1) < merged_data['R1']) & (merged_data[close_col] > merged_data['R1']) & (merged_data[close_col] > merged_data[open_col])) |
#                                         ((merged_data[close_col].shift(1) < merged_data['R2']) & (merged_data[close_col] > merged_data['R2']) & (merged_data[close_col] > merged_data[open_col])) |
#                                         ((merged_data[close_col].shift(1) < merged_data['R3']) & (merged_data[close_col] > merged_data['R3']) & (merged_data[close_col] > merged_data[open_col]))), 'Buy',None
#                                         )
#
#     merged_data['Profit_exit'] = np.where(merged_data['Buy_sell']=='Buy', merged_data[close_col] + (merged_data[close_col]*0.01),None)
#     merged_data['Stoploss_exit'] = np.where(merged_data['Buy_sell']=='Buy', merged_data[close_col] - (merged_data[close_col]*0.005),None)
#
#     merged_data['Buy_sell_fill'] = merged_data['Buy_sell'].ffill()
#     merged_data['Profit_exit'] = merged_data['Profit_exit'].ffill()
#     merged_data['Stoploss_exit'] = merged_data['Stoploss_exit'].ffill()
#
#     merged_data['Buy_sell'] = np.where(
#                                         (merged_data['Buy_sell_fill'] == 'Buy')
#                                         & (merged_data['Buy_sell'] != 'Buy')
#                                         & ((merged_data[close_col] >= merged_data['Profit_exit'])
#                                         | (merged_data[close_col] <= merged_data['Stoploss_exit'])),'Buy_SL', merged_data['Buy_sell']
#                                     )
#     merged_data.dropna(subset=['Buy_sell'], inplace=True)
#     merged_data = merged_data[merged_data['Buy_sell'] != merged_data['Buy_sell'].shift(1)]
#
#     merged_data.drop(columns=['Buy_sell_fill'], inplace=True)
#     return merged_data
#
# def generate_sell(merged_data_1,close_col,open_col):
#     merged_data = merged_data_1.copy(deep=True)
#     merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])
#     merged_data.sort_values(by='DATE', ascending=True, inplace=True)
#     merged_data['Buy_sell'] = np.where(
#                                 ((merged_data[close_col].shift(1) > merged_data['S3']) & (merged_data[close_col] < merged_data['S3']) & (merged_data[close_col] < merged_data[open_col])) |
#                                  ((merged_data[close_col].shift(1) > merged_data['S2']) & (merged_data[close_col] < merged_data['S2']) & (merged_data[close_col] < merged_data[open_col])) |
#                                  ((merged_data[close_col].shift(1) > merged_data['S1']) & (merged_data[close_col] < merged_data['S1']) & (merged_data[close_col] < merged_data[open_col])) |
#                                  ((merged_data[close_col].shift(1) > merged_data['Pivot']) & (merged_data[close_col] < merged_data['Pivot']) & (merged_data[close_col] < merged_data[open_col])) |
#                                  ((merged_data[close_col].shift(1) > merged_data['R1']) & (merged_data[close_col] < merged_data['R1']) & (merged_data[close_col] < merged_data[open_col])) |
#                                  ((merged_data[close_col].shift(1) > merged_data['R2']) & (merged_data[close_col] < merged_data['R2']) & (merged_data[close_col] < merged_data[open_col])) |
#                                  ((merged_data[close_col].shift(1) > merged_data['R3']) & (merged_data[close_col] < merged_data['R3']) & (merged_data[close_col] < merged_data[open_col])), 'Sell', None
#     )
#
#     merged_data['Profit_exit'] = np.where(merged_data['Buy_sell'] == 'Sell',
#                                           merged_data[close_col] - (merged_data[close_col] * 0.01),None)
#     merged_data['Stoploss_exit'] = np.where(merged_data['Buy_sell'] == 'Sell',
#                                             merged_data[close_col] + (merged_data[close_col] * 0.005),None)
#
#     merged_data['Buy_sell_fill'] = merged_data['Buy_sell'].ffill()
#     merged_data['Profit_exit'] = merged_data['Profit_exit'].ffill()
#     merged_data['Stoploss_exit'] = merged_data['Stoploss_exit'].ffill()
#
#     merged_data['Buy_sell'] = np.where(
#         (merged_data['Buy_sell_fill'] == 'Sell')
#         & (merged_data['Buy_sell'] != 'Sell')
#         & ((merged_data[close_col] <= merged_data['Profit_exit'])
#         | (merged_data[close_col] >= merged_data['Stoploss_exit'])), 'Sell_SL', merged_data['Buy_sell']
#     )
#
#     merged_data.dropna(subset=['Buy_sell'], inplace=True)
#     merged_data = merged_data[merged_data['Buy_sell'] != merged_data['Buy_sell'].shift(1)]
#     merged_data.drop(columns=['Buy_sell_fill'], inplace=True)
#     return merged_data

def generate_buy_pivot_exit(merged_data_1,close_col,open_col):
    merged_data = merged_data_1.copy(deep=True)
    merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])
    merged_data.sort_values(by='DATE', ascending=True, inplace=True)
    merged_data['Buy_sell'] = np.where(
                                        (
                                          ((merged_data[close_col].shift(1) < merged_data['S3']) & (merged_data[close_col] > merged_data['S3']) & (merged_data[close_col] > merged_data[open_col])) |
                                          ((merged_data[close_col].shift(1) < merged_data['S2']) & (merged_data[close_col] > merged_data['S2']) & (merged_data[close_col] > merged_data[open_col])) |
                                          ((merged_data[close_col].shift(1) < merged_data['S1']) & (merged_data[close_col] > merged_data['S1']) & (merged_data[close_col] > merged_data[open_col])) |
                                          ((merged_data[close_col].shift(1) < merged_data['Pivot']) & (merged_data[close_col] > merged_data['Pivot']) & (merged_data[close_col] > merged_data[open_col])) |
                                          ((merged_data[close_col].shift(1) < merged_data['R1']) & (merged_data[close_col] > merged_data['R1']) & (merged_data[close_col] > merged_data[open_col])) |
                                          ((merged_data[close_col].shift(1) < merged_data['R2']) & (merged_data[close_col] > merged_data['R2']) & (merged_data[close_col] > merged_data[open_col])) |
                                          ((merged_data[close_col].shift(1) < merged_data['R3']) & (merged_data[close_col] > merged_data['R3']) & (merged_data[close_col] > merged_data[open_col]))
                                        ), 'Buy', None
                                     )

    merged_data['Profit_exit'] = np.where(merged_data['Buy_sell'] == 'Buy',merged_data[close_col] + (merged_data[close_col] * 0.01), None)
    merged_data['Stoploss_exit'] = np.where(merged_data['Buy_sell'] == 'Buy',merged_data[close_col] - (merged_data[close_col] * 0.005), None)
    merged_data['Stoploss_cpr'] = np.where(
                                            ((merged_data[close_col].shift(1) < merged_data['S3']) & (merged_data[close_col] > merged_data['S3']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['S3'],
                                        np.where(
                                            ((merged_data[close_col].shift(1) < merged_data['S2']) & (merged_data[close_col] > merged_data['S2']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['S2'],
                                            np.where(
                                                ((merged_data[close_col].shift(1) < merged_data['S1']) & (merged_data[close_col] > merged_data['S1']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['S1'],
                                                np.where(
                                                    ((merged_data[close_col].shift(1) < merged_data['Pivot']) & (merged_data[close_col] > merged_data['Pivot']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['Pivot'],
                                                    np.where(
                                                        ((merged_data[close_col].shift(1) < merged_data['R1']) & (merged_data[close_col] > merged_data['R1']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['R1'],
                                                        np.where(
                                                            ((merged_data[close_col].shift(1) < merged_data['R2']) & (merged_data[close_col] > merged_data['R2']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['R2'],
                                                            np.where(
                                                                ((merged_data[close_col].shift(1) < merged_data['R3']) & (merged_data[close_col] > merged_data['R3']) & (merged_data['Buy_sell'] == 'Buy')),merged_data['R3'],None
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
    )

    merged_data['Buy_sell_fill'] = merged_data['Buy_sell'].ffill()
    merged_data['Profit_exit'] = merged_data['Profit_exit'].ffill()
    merged_data['Stoploss_exit'] = merged_data['Stoploss_exit'].ffill()
    merged_data['Stoploss_cpr'] = merged_data['Stoploss_cpr'].ffill()

    merged_data['Buy_sell'] = np.where(
                                        (merged_data['Buy_sell_fill'] == 'Buy') & (merged_data['Buy_sell'] != 'Buy') & ( (merged_data[close_col] >= merged_data['Profit_exit']) |
                                                                                                                         (merged_data[close_col] <= merged_data['Stoploss_exit']) |
                                                                                                                         (merged_data[close_col] <= merged_data['Stoploss_cpr'])
                                                                                                                       ), 'Buy_SL', merged_data['Buy_sell']
    )
    merged_data.dropna(subset=['Buy_sell'], inplace=True)

    merged_data = merged_data[merged_data['Buy_sell'] != merged_data['Buy_sell'].shift(1)]
    merged_data.drop(columns=['Buy_sell_fill'], inplace=True)
    return merged_data

def generate_sell_pivot_exit(merged_data_1,close_col,open_col):
    merged_data = merged_data_1.copy(deep=True)
    merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])
    merged_data.sort_values(by='DATE', ascending=True, inplace=True)
    merged_data['Buy_sell'] = np.where(
                                ((merged_data[close_col].shift(1) > merged_data['S3']) & (merged_data[close_col] < merged_data['S3']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['S2']) & (merged_data[close_col] < merged_data['S2']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['S1']) & (merged_data[close_col] < merged_data['S1']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['Pivot']) & (merged_data[close_col] < merged_data['Pivot']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['R1']) & (merged_data[close_col] < merged_data['R1']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['R2']) & (merged_data[close_col] < merged_data['R2']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['R3']) & (merged_data[close_col] < merged_data['R3']) & (merged_data[close_col] < merged_data[open_col])), 'Sell', None
    )

    merged_data['Profit_exit'] = np.where(merged_data['Buy_sell'] == 'Sell',merged_data[close_col] - (merged_data[close_col] * 0.01),None)
    merged_data['Stoploss_exit'] = np.where(merged_data['Buy_sell'] == 'Sell',merged_data[close_col] + (merged_data[close_col] * 0.005),None)

    merged_data['Stoploss_cpr'] = np.where(
                                            ((merged_data[close_col].shift(1) > merged_data['S3']) & (merged_data[close_col] < merged_data['S3']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['S3'],
                                            np.where(
                                                ((merged_data[close_col].shift(1) > merged_data['S2']) & (merged_data[close_col] < merged_data['S2']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['S2'],
                                                np.where(
                                                    ((merged_data[close_col].shift(1) > merged_data['S1']) & (merged_data[close_col] < merged_data['S1']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['S1'],
                                                    np.where(
                                                        ((merged_data[close_col].shift(1) > merged_data['Pivot']) & (merged_data[close_col] < merged_data['Pivot']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['Pivot'],
                                                        np.where(
                                                            ((merged_data[close_col].shift(1) > merged_data['R1']) & (merged_data[close_col] < merged_data['R1']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['R1'],
                                                            np.where(
                                                                ((merged_data[close_col].shift(1) > merged_data['R2']) & (merged_data[close_col] < merged_data['R2']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['R2'],
                                                                np.where(
                                                                    ((merged_data[close_col].shift(1) > merged_data['R3']) & (merged_data[close_col] < merged_data['R3']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['R3'],None
                                                                )
                                                          )
                                                      )
                                              )
                                          )
                                          )
                                )



    merged_data['Buy_sell_fill'] = merged_data['Buy_sell'].ffill()
    merged_data['Profit_exit'] = merged_data['Profit_exit'].ffill()
    merged_data['Stoploss_exit'] = merged_data['Stoploss_exit'].ffill()
    merged_data['Stoploss_cpr'] = merged_data['Stoploss_cpr'].ffill()

    merged_data['Buy_sell'] = np.where(
                                        (merged_data['Buy_sell_fill'] == 'Sell') & (merged_data['Buy_sell'] != 'Sell') & ((merged_data[close_col] <= merged_data['Profit_exit']) |
                                                                                                                          (merged_data[close_col] >= merged_data['Stoploss_exit']) |
                                                                                                                          (merged_data[close_col] >= merged_data['Stoploss_cpr'])), 'Sell_SL', merged_data['Buy_sell']
    )

    merged_data.dropna(subset=['Buy_sell'], inplace=True)
    merged_data = merged_data[merged_data['Buy_sell'] != merged_data['Buy_sell'].shift(1)]
    merged_data.drop(columns=['Buy_sell_fill'], inplace=True)
    return merged_data

def generate_buy_with_supertrend(merged_data_1, ticker_name,close_col, open_col):
    merged_data = merged_data_1.copy(deep=True)
    merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])
    merged_data.sort_values(by='DATE', ascending=True, inplace=True)
    merged_data['Supertrend_buy_sell'] = np.where(merged_data[f'{ticker_name}_SUPERTREND_Direction']==1,'Buy','Sell')
    merged_data['Buy_sell'] = np.where(
                                        (
                                          (
                                            ((merged_data[close_col].shift(1) < merged_data['S3']) & (merged_data[close_col] > merged_data['S3']) & (merged_data[close_col] > merged_data[open_col])) |
                                            ((merged_data[close_col].shift(1) < merged_data['S2']) & (merged_data[close_col] > merged_data['S2']) & (merged_data[close_col] > merged_data[open_col])) |
                                            ((merged_data[close_col].shift(1) < merged_data['S1']) & (merged_data[close_col] > merged_data['S1']) & (merged_data[close_col] > merged_data[open_col])) |
                                            ((merged_data[close_col].shift(1) < merged_data['Pivot']) & (merged_data[close_col] > merged_data['Pivot']) & (merged_data[close_col] > merged_data[open_col])) |
                                            ((merged_data[close_col].shift(1) < merged_data['R1']) & (merged_data[close_col] > merged_data['R1']) & (merged_data[close_col] > merged_data[open_col])) |
                                            ((merged_data[close_col].shift(1) < merged_data['R2']) & (merged_data[close_col] > merged_data['R2']) & (merged_data[close_col] > merged_data[open_col])) |
                                            ((merged_data[close_col].shift(1) < merged_data['R3']) & (merged_data[close_col] > merged_data['R3']) & (merged_data[close_col] > merged_data[open_col]))
                                          ) & (merged_data['Supertrend_buy_sell']=='Buy')
                                        ), 'Buy', None
                                       )

    merged_data['Profit_exit'] = np.where(merged_data['Buy_sell'] == 'Buy',merged_data[close_col] + (merged_data[close_col] * 0.01), None)
    merged_data['Stoploss_exit'] = np.where(merged_data['Buy_sell'] == 'Buy',merged_data[close_col] - (merged_data[close_col] * 0.005), None)
    merged_data['Stoploss_cpr'] = np.where(
                                            ((merged_data[close_col].shift(1) < merged_data['S3']) & (merged_data[close_col] > merged_data['S3']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['S3'],
                                            np.where(
                                                    ((merged_data[close_col].shift(1) < merged_data['S2']) & (merged_data[close_col] > merged_data['S2']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['S2'],
                                                    np.where(
                                                            ((merged_data[close_col].shift(1) < merged_data['S1']) & (merged_data[close_col] > merged_data['S1']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['S1'],
                                                            np.where(
                                                                    ((merged_data[close_col].shift(1) < merged_data['Pivot']) & (merged_data[close_col] > merged_data['Pivot']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['Pivot'],
                                                                    np.where(
                                                                            ((merged_data[close_col].shift(1) < merged_data['R1']) & (merged_data[close_col] > merged_data['R1']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['R1'],
                                                                            np.where(
                                                                                    ((merged_data[close_col].shift(1) < merged_data['R2']) & (merged_data[close_col] > merged_data['R2']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['R2'],
                                                                                    np.where(
                                                                                            ((merged_data[close_col].shift(1) < merged_data['R3']) & (merged_data[close_col] > merged_data['R3']) & (merged_data['Buy_sell'] == 'Buy')), merged_data['R3'], None
                                                                                            )
                                                                                    )
                                                                            )
                                                                    )
                                                            )
                                                    )
                                            )

    merged_data['Buy_sell_fill'] = merged_data['Buy_sell'].ffill()
    merged_data['Profit_exit'] = merged_data['Profit_exit'].ffill()
    merged_data['Stoploss_exit'] = merged_data['Stoploss_exit'].ffill()
    merged_data['Stoploss_cpr'] = merged_data['Stoploss_cpr'].ffill()

    merged_data['Buy_sell'] = np.where(
                                        (merged_data['Buy_sell_fill'] == 'Buy') & (merged_data['Buy_sell'] != 'Buy') & ((merged_data[close_col] >= merged_data['Profit_exit']) |
                                                                                                                        (merged_data[close_col] <= merged_data['Stoploss_exit']) |
                                                                                                                        (merged_data[close_col] <= merged_data['Stoploss_cpr'])
                                                                                                                        ), 'Buy_SL', merged_data['Buy_sell']
                                        )
    merged_data.dropna(subset=['Buy_sell'], inplace=True)
    merged_data = merged_data[merged_data['Buy_sell'] != merged_data['Buy_sell'].shift(1)]

    merged_data.drop(columns=['Buy_sell_fill'], inplace=True)
    return merged_data

def generate_sell_with_supertrend(merged_data_1,ticker_name,close_col,open_col):
    merged_data = merged_data_1.copy(deep=True)
    merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])
    merged_data.sort_values(by='DATE', ascending=True, inplace=True)
    merged_data['Supertrend_buy_sell'] = np.where(merged_data[f'{ticker_name}_SUPERTREND_Direction'] == 1, 'Buy', 'Sell')
    merged_data['Buy_sell'] = np.where(
                                ((((merged_data[close_col].shift(1) > merged_data['S3']) & (merged_data[close_col] < merged_data['S3']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['S2']) & (merged_data[close_col] < merged_data['S2']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['S1']) & (merged_data[close_col] < merged_data['S1']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['Pivot']) & (merged_data[close_col] < merged_data['Pivot']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['R1']) & (merged_data[close_col] < merged_data['R1']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['R2']) & (merged_data[close_col] < merged_data['R2']) & (merged_data[close_col] < merged_data[open_col])) |
                                 ((merged_data[close_col].shift(1) > merged_data['R3']) & (merged_data[close_col] < merged_data['R3']) & (merged_data[close_col] < merged_data[open_col]))) & (merged_data['Supertrend_buy_sell'] == 'Sell')) , 'Sell', None
    )

    merged_data['Profit_exit'] = np.where(merged_data['Buy_sell'] == 'Sell',
                                          merged_data[close_col] - (merged_data[close_col] * 0.01),None)
    merged_data['Stoploss_exit'] = np.where(merged_data['Buy_sell'] == 'Sell',
                                            merged_data[close_col] + (merged_data[close_col] * 0.005),None)
    merged_data['Stoploss_cpr'] = np.where(
                                            ((merged_data[close_col].shift(1) > merged_data['S3']) & (merged_data[close_col] < merged_data['S3']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['S3'],
                                            np.where(
                                                        ((merged_data[close_col].shift(1) > merged_data['S2']) & (merged_data[close_col] < merged_data['S2']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['S2'],
                                                        np.where(
                                                            ((merged_data[close_col].shift(1) > merged_data['S1']) & (merged_data[close_col] < merged_data['S1']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['S1'],
                                                            np.where(
                                                                ((merged_data[close_col].shift(1) > merged_data['Pivot']) & (merged_data[close_col] < merged_data['Pivot']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['Pivot'],
                                                                np.where(
                                                                    ((merged_data[close_col].shift(1) > merged_data['R1']) & (merged_data[close_col] < merged_data['R1']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['R1'],
                                                                    np.where(
                                                                        ((merged_data[close_col].shift(1) > merged_data['R2']) & (merged_data[close_col] < merged_data['R2']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['R2'],
                                                                        np.where(
                                                                            ((merged_data[close_col].shift(1) > merged_data['R3']) & (merged_data[close_col] < merged_data['R3']) & (merged_data['Buy_sell'] == 'Sell')), merged_data['R3'],None
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                           )
                                          )



    merged_data['Buy_sell_fill'] = merged_data['Buy_sell'].ffill()
    merged_data['Profit_exit'] = merged_data['Profit_exit'].ffill()
    merged_data['Stoploss_exit'] = merged_data['Stoploss_exit'].ffill()
    merged_data['Stoploss_cpr'] = merged_data['Stoploss_cpr'].ffill()

    merged_data['Buy_sell'] = np.where(
                                        (merged_data['Buy_sell_fill'] == 'Sell') & (merged_data['Buy_sell'] != 'Sell') & ((merged_data[close_col] <= merged_data['Profit_exit']) |
                                                                                                                          (merged_data[close_col] >= merged_data['Stoploss_exit']) |
                                                                                                                          (merged_data[close_col] >= merged_data['Stoploss_cpr'])
                                                                                                                          ), 'Sell_SL', merged_data['Buy_sell']
                                     )

    merged_data.dropna(subset=['Buy_sell'], inplace=True)
    merged_data = merged_data[merged_data['Buy_sell'] != merged_data['Buy_sell'].shift(1)]
    merged_data.drop(columns=['Buy_sell_fill'], inplace=True)
    return merged_data


def get_pnl(pnl_data,col_close):

    pnl_data['DATE'] = pd.to_datetime(pnl_data['DATE'])
    pnl_data.sort_values(by=['DATE'], ascending=True, inplace=True)
    pnl_data['PNL'] = np.where(
                                pnl_data['Buy_sell'] == 'Buy_SL',pnl_data[col_close] - pnl_data[col_close].shift(1) ,
                                np.where(pnl_data['Buy_sell']=='Sell_SL', pnl_data[col_close].shift(1) - pnl_data[col_close] ,None)
                              )

    pnl_data['Time_in_market'] = np.where( ((pnl_data['Buy_sell']=='Buy_SL') | (pnl_data['Buy_sell']=='Sell_SL')), pnl_data['DATE'] - pnl_data['DATE'].shift(1), None)

    pnl_data['Time_in_market'] = pd.to_timedelta(pnl_data['Time_in_market'])

    pnl_data['Win_ratio'] = np.where(pnl_data['PNL'] > 0, 1, 0)

    return pnl_data

