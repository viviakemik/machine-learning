import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import talib as ta

'''
1. Stochastic Oscillator
Purpose: Identifies overbought and oversold conditions, showing where the price is in relation to a high-low range over a set period.
Usefulness: Useful for pinpointing potential reversal points. A high reading (above 80) suggests overbought conditions, while a low reading (below 20) suggests oversold conditions. Traders often use the %K line crossing above or below the %D line (its moving average) as a signal to enter or exit trades.

2. Bollinger Bands
Purpose: Measures volatility and helps identify price trends or reversals. It consists of a simple moving average (middle band) and two standard deviation bands (upper and lower).
Usefulness: Useful for spotting when the price is "stretching" beyond its typical range. When the price reaches the upper band, it might be overbought; at the lower band, it might be oversold. Squeezes (when bands are close) indicate low volatility and potential breakouts, while wide bands suggest high volatility and potential trend changes.

3. Relative Strength Index (RSI)
Purpose: Measures the speed and change of price movements, indicating momentum.
Usefulness: Helps detect overbought (typically above 70) and oversold (typically below 30) conditions, signaling potential reversals. RSI is valuable for momentum-based trading strategies and can be used to confirm other trend indicators or spot divergences between the price and RSI.

4. Average Directional Movement Index (ADX)
Purpose: Measures the strength of a trend without indicating its direction.
Usefulness: An ADX above 20 or 25 often indicates a strong trend (bullish or bearish). It’s frequently used alongside other indicators to confirm trends and avoid false signals. A rising ADX confirms trend strength, while a falling ADX suggests a weakening trend or a potential range-bound market.

5. Exponential Moving Average (EMA)
Purpose: A moving average that gives more weight to recent prices, making it more responsive to new information.
Usefulness: Great for spotting trends and reversals. EMAs are especially useful in fast-moving markets, where price trends may change quickly. Shorter EMAs (e.g., 12-period) respond faster to price changes and are useful for shorter time frames, while longer EMAs (e.g., 50-period) work well for long-term trend analysis.

6. Simple Moving Average (SMA)
Purpose: A basic average of prices over a specific period, often used to smooth out price action.
Usefulness: Useful for spotting the overall trend. SMA crossovers (e.g., a 50-day SMA crossing above a 200-day SMA, known as a "golden cross") can signal trend reversals. It provides a general trend view but lags behind the EMA, so it may be better suited for long-term analysis.
'''

def list_groups_of_functions():
    for group, funcs in ta.get_function_groups().items():
        print('-----------------------------------------')
        print('Group:', group)
        print('Functions:')
        for func in funcs:
            # A descrição pode ser obtida assim
            print('    ', func)  # Se não tiver info, apenas exibe o nome da função
        print('\n')



def calculate_tecnical_indicators(df, vars_in = [], method = None, parans = []):
    
    if method == None:
        return df
    
    if method == 'SMA':
        var_out = vars_in[0] + '_' + method + '_' + str(parans[0])
        df[var_out] = ta.__dict__[method](df[vars_in[0]].values, timeperiod = parans[0])
        return df
    
    if method == 'EMA':
        var_out = vars_in[0] + '_' + method + '_' + str(parans[0])
        df[var_out] = ta.__dict__[method](df[vars_in[0]].values, timeperiod = parans[0])
        return df
   
    if method == 'PLUS_DI':
        var_out = method + '_' + str(parans[0])
        df[var_out] = ta.__dict__[method](high  = df[vars_in[0]].values,
                                          low   = df[vars_in[1]].values,
                                          close = df[vars_in[2]].values, 
                                          timeperiod = parans[0])
        return df
    
    if method == 'MINUS_DI':
        var_out = method + '_' + str(parans[0])
        df[var_out] = ta.__dict__[method](high  = df[vars_in[0]].values,
                                          low   = df[vars_in[1]].values,
                                          close = df[vars_in[2]].values, 
                                          timeperiod = parans[0])
        return df
    
    if method == 'ADX':
        var_out = method + '_' + str(parans[0])
        df[var_out] = ta.__dict__[method](high  = df[vars_in[0]].values,
                                          low   = df[vars_in[1]].values,
                                          close = df[vars_in[2]].values, 
                                          timeperiod = parans[0])
        return df
    
    if method == 'RSI':
        var_out = vars_in[0] + '_' + method + '_' + str(parans[0])
        df[var_out] = ta.__dict__[method](df[vars_in[0]].values, timeperiod = parans[0])
        return df

    if method == 'BBANDS':
        var_out_low = vars_in[0] + '_' + method + '_LOW_' + str(parans[0]) + '_' + str(parans[2])
        var_out_mid = vars_in[0] + '_' + method + '_MID_' + str(parans[0])
        var_out_up = vars_in[0] + '_' + method + '_UP_' + str(parans[0]) + '_' + str(parans[1])
        df[var_out_up],df[var_out_mid],df[var_out_low] = ta.__dict__[method](df[vars_in[0]].values, 
                                                                                 timeperiod = parans[0], 
                                                                                 nbdevup = parans[1], 
                                                                                 nbdevdn = parans[2], 
                                                                                 matype = parans[3])
        return df

    if method == 'STOCH':
        var_out = vars_in[2] + '_' + method + '_' + str(parans[0]) + '_' + str(parans[1])
        
        df[var_out], df[var_out] = ta.__dict__[method](high  = df[vars_in[0]].values,
                                                                  low   = df[vars_in[1]].values,
                                                                  close = df[vars_in[2]].values, 
                                                                  fastk_period = parans[0], 
                                                                  slowk_period = parans[1], 
                                                                  slowk_matype = parans[2], 
                                                                  slowd_period = parans[3], 
                                                                  slowd_matype = parans[4])
        return df
        