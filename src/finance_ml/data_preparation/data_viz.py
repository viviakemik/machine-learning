import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


colors = {'red': '#ff207c', 'grey': '#42535b', 'blue': '#207cff', 'orange': '#ffa320', 'green': '#00ec8b'}
config_ticks = {'size': 14, 'color': colors['grey'], 'labelcolor': colors['grey']}
config_title = {'size': 18, 'color': colors['grey'], 'ha': 'left', 'va': 'baseline'}


def format_borders(plot):
    plot.spines['top'].set_visible(False)
    plot.spines['left'].set_visible(False)
    plot.spines['left'].set_color(colors['grey'])
    plot.spines['bottom'].set_color(colors['grey'])

def get_charts(stock_data, symbol):
    plt.rc('figure', figsize=(15, 7))
    
    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.tight_layout(pad=3)
    
    date = stock_data['date']
    close = stock_data['price']
    vol = stock_data['volume'] / 1000000
    label = stock_data['date_label']
    
    plot_price = axes[0]
    plot_price.plot(date, close, color=colors['blue'], linewidth=1, label='Price')
    
    plot_vol = axes[1]
    plot_vol.bar(date, vol, width=3, color='green')

    xticks = np.arange(0, stock_data.shape[0], 1000)
    
    plot_price.yaxis.tick_right()
    plot_price.set_xticks(xticks)
    plot_price.set_xticklabels(label[xticks])
    plot_price.tick_params(axis='both', **config_ticks)
    plot_price.set_ylabel('Price', fontsize=14)
    plot_price.yaxis.set_label_position("right")
    plot_price.yaxis.label.set_color(colors['grey'])
    plot_price.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)
    plot_price.set_axisbelow(True)

    plot_vol.yaxis.tick_right()
    plot_vol.set_xticks(xticks)
    plot_vol.tick_params(axis='both', **config_ticks)
    plot_vol.set_ylabel('Volume (in millions)', fontsize=14)
    plot_vol.yaxis.set_label_position("right")
    plot_vol.yaxis.label.set_color(colors['grey'])
    plot_vol.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)
    plot_vol.set_axisbelow(True)

    format_borders(plot_price)
    format_borders(plot_vol)

    fig.suptitle(symbol + ' Price (per minutes) and Volume', size=20, color=colors['grey'])

