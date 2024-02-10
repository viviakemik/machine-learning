import sys
sys.path.append('../')
from src.finance_ml.data_preparation.data_preparation import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import copy
from Functions import *

ticker = 'Apple'
fname_USDEUR = 'FX/USDEUR_2020-04-07_2022-04-06.parquet'
N = 10000

dataloader = DataLoader(time_index_col='DATE', keep_cols=['VOLUME', 'OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VW', 'TRANSACTIONS'])

original_data = dataloader.load_dataset({ticker: 'C:\\Users\\acer\\PycharmProjects\\MLF\\data\\' + fname_USDEUR}).iloc[:N]

df_win23 = copy.deepcopy(original_data)


close_prices = df_win23['Apple_CLOSE']
open_prices= df_win23['Apple_OPEN']
high_prices = df_win23['Apple_HIGHT']
low_prices = df_win23['Apple_LOW']

# Calculate TMA
window_size=23
result = calculate_triangular_moving_average(close_prices, window_size) # to get result for close price
print(result)
result_open = calculate_triangular_moving_average(open_prices, window_size) # to get results for open prices
result_high=calculate_triangular_moving_average(high_prices, window_size) # to get results for high prices
result_low=calculate_triangular_moving_average(low_prices, window_size) # to get results for low prices
#print("TMA = ",result)

# Difference btw closing TMA and Opeing TMA
diff=[]
for i in range(min(len(result), len(result_open))):
    if result[i] is not None and result_open[i] is not None:

        difference = result[i] - result_open[i]
        diff.append(difference)

# Plot Difference
plt.figure(figsize=(12, 6))
plt.plot(diff, label='Difference')
plt.ylabel('Price')
plt.title('Difference btw the TMAs of close prices and open prices ')
plt.legend()
plt.grid()
plt.show()

# Plot TMA
plt.figure(figsize=(12, 6))
plt.plot(result, label='Close TMA')
plt.ylabel('Price')
plt.title('Triangular Moving Averages for closing prices')
plt.legend()
plt.grid()
plt.show()

#plot open tma
plt.figure(figsize=(12, 6))
plt.plot(result_open, label='Open TMA')
plt.ylabel('Price')
plt.title('Triangular Moving Averages for opening prices')
plt.legend()
plt.grid()
plt.show()

# Plot high TMA
plt.figure(figsize=(12, 6))
plt.plot(result_high, label='Hight prices TMA')
plt.ylabel('Price')
plt.title('Triangular Moving Averages for hight prices')
plt.legend()
plt.grid()
plt.show()

# Plot low TMA
plt.figure(figsize=(12, 6))
plt.plot(result_low, label='Low prices TMA')
plt.ylabel('Price')
plt.title('Triangular Moving Averages for low prices')
plt.legend()
plt.grid()
plt.show()