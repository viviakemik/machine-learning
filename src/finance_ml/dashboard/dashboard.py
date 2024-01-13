'''
Created on 02-Jan-2022
By
- Pratik
- Justins
- Swetha

This file will help to visualize the Profit and loss generated by the strategy.
'''
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output
import plotly.express as px
from datetime import date
import sys
sys.path.append('../')
import os
import pandas_ta as ta
from data_preparation.data_preparation import DataLoader
from trading_strategy.pivot_strategy import *
from indicators.indicators import Indicators


app = dash.Dash(external_stylesheets=[dbc.themes.COSMO])

app.layout = html.Div(children = [
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(['AAPL','FB','TSLA',"RUBEUR","USDBRL","USDEUR","GLD","PDBC","SLV","BTCUSD","DOGEUSD","ETHUSD","IVM","QQQ","RNRG","SPY"],'AAPL',id='Company',placeholder='AAPL')
        ]),
        dbc.Col([
            dcc.DatePickerRange(id='Date_picker',
                                min_date_allowed=date(2020,4,7),
                                max_date_allowed=date(2022,4,6),
                                start_date=date(2020,4,7),
                                end_date=date(2022,4,7)
                                )
        ]),
        dbc.Col([
            dcc.Dropdown(['Supertrend','Without_supertrend'],'Without_supertrend',
                         id='supertrend_selection',
                         placeholder='Without_supertrend'
                         )
        ]),
        dbc.Col([
            dcc.Dropdown(['Traditional_pivot','Camarilla_pivot','Woodie_pivot'],'Traditional_pivot',
                         id='pivot_selection',
                         placeholder='Traditional_pivot'
                         )
        ]),
        dbc.Col([
            dcc.Dropdown(['5min','15min','30min','1h'],'5min',placeholder='5min',id='trades_timeframe')
        ]),
        dbc.Col([
            dcc.Dropdown(['1d','1w'],'1d', placeholder='1d', id='pivot_timeframe')
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='PNL_CHART')

        ]),
        dbc.Col([
            dbc.Row([
                dbc.Card(
                dbc.CardBody([
                    html.H2('Total Profit and Loss', className='card-title'),
                    html.H4(id='PNL_Total',className='card-subtitle')
                    ])
                )
            ]),
            dbc.Row([
                dbc.Card(
                    dbc.CardBody([
                        html.H2('Max Profit', className='card-title'),
                        html.H4(id='Max_profit',className='card-subtitle')
                        ])
                )
            ]),
            dbc.Row([
                    dbc.Card(
                        dbc.CardBody([
                            html.H2('Win Ratio', className='card-title'),
                            html.H4(id='win_ratio',className='card-subtitle')
                        ])
                    )
            ])
        ])
    ]),

])


@app.callback([Output('PNL_CHART','figure'),
               Output('PNL_Total','children'),
               Output('Max_profit','children'),
               Output('win_ratio','children')],

              [Input('Company','value'),
               Input('Date_picker', 'start_date'),
               Input('Date_picker', 'end_date'),
               Input('supertrend_selection', 'value'),
               Input('pivot_selection', 'value'),
               Input('trades_timeframe', 'value'),
               Input('pivot_timeframe', 'value')
               ]
)
def return_company(ticker, start_date, end_date, supertrend_selected, pivot_selected, trades_timeframe, pivot_timeframe):



    clicked_id = ticker

    current_directory = os.getcwd()
    new_directory = os.path.abspath(os.path.join(current_directory, '..', '..', '..'))

    dataloader = DataLoader(time_index_col='DATE',
                            keep_cols=['VOLUME', 'OPEN', 'CLOSE', 'LOW', 'TRANSACTIONS', 'HIGHT'])

    if clicked_id == 'BTCUSD':

        ticker_name = 'BTCUSD'
        data = dataloader.load_dataset({ticker_name: fr"{new_directory}/data/cryptos/BTCUSD_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "RUBEUR":
        ticker_name = 'RUBEUR'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/FX/RUBEUR_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "USDBRL":
        ticker_name = 'USDBRL'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/FX/USDBRL_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "USDEUR":
        ticker_name = 'USDEUR'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/FX/USDEUR_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "GLD":
        ticker_name = 'GLD'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/commodities/GLD_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "PDBC":
        ticker_name = 'PDBC'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/commodities/PDBC_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "SLV":
        ticker_name = 'SLV'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/commodities/SLV_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "DOGEUSD":
        ticker_name = 'DOGEUSD'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/cryptos/DOGEUSD_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "ETHUSD":
        ticker_name = 'ETHUSD'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/cryptos/ETHUSD_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "AAPL":
        ticker_name = 'AAPL'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/equities/AAPL_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "FB":
        ticker_name = 'FB'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/equities/FB_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "TSLA":
        ticker_name = 'TSLA'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/equities/TSLA_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "IWM":
        ticker_name = 'IWM'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/indices/IWM_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "QQQ":
        ticker_name = 'QQQ'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/indices/QQQ_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "RNRG":
        ticker_name = 'RNRG'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/indices/RNRG_2020-04-07_2022-04-06.parquet"})

    elif clicked_id == "SPY":
        ticker_name = 'SPY'
        data = dataloader.load_dataset(
            {ticker_name: fr"{new_directory}/data/indices/SPY_2020-04-07_2022-04-06.parquet"})

    else:
        data = pd.read_excel(fr'D:\FAU\Ml in finance_ex\github_repo\notebooks\report\traditional_merge_15m.xlsx')



    open_col = f'{ticker_name}_OPEN'
    high_col = f'{ticker_name}_HIGHT'
    low_col = f'{ticker_name}_LOW'
    close_col = f'{ticker_name}_CLOSE'

    resampled_data = resample_ohlc(data, [open_col, high_col, low_col, close_col], pivot_timeframe).copy(deep=True)
    resampled_data_trade = resample_ohlc(data, [open_col, high_col, low_col, close_col], trades_timeframe).copy(deep=True)

    print(resampled_data_trade)


    if supertrend_selected == "Supertrend":
        indi = Indicators(ticker=ticker_name, calc_all=False, list_ind=['SUPERTREND'])
        super_trend = indi.fit_transform(resampled_data_trade)

    if pivot_selected == 'Woodie_pivot':
        pivot_values = woodie_pivot(resampled_data, high_col, low_col, close_col)

    elif pivot_selected == 'Traditional_pivot':
        pivot_values = traditional_pivot(resampled_data, high_col, low_col, close_col)

    elif pivot_selected == 'Camarilla_pivot':
        pivot_values = camarilla_pivot(resampled_data, high_col, low_col, close_col)


    if supertrend_selected == "Supertrend":
        mereged_data = merge_ohlc_pivot(super_trend, pivot_values)
        traded_buy = generate_buy_with_supertrend(mereged_data, ticker_name,close_col, open_col)
        traded_sell = generate_sell_with_supertrend(mereged_data, ticker_name,close_col, open_col)
    else:
        mereged_data = merge_ohlc_pivot(resampled_data_trade, pivot_values)
        traded_buy = generate_buy_pivot_exit(mereged_data, close_col, open_col)
        traded_sell = generate_sell_pivot_exit(mereged_data, close_col, open_col)



    pnl_buy = get_pnl(traded_buy, close_col)
    pnl_sell = get_pnl(traded_sell, close_col)

    df = pd.concat([pnl_buy,pnl_sell],axis=0)

    df.sort_values(by="DATE", ascending=True, inplace=True)

    df.dropna(subset=["PNL"], inplace=True)


    if start_date is not None and end_date is not None:
        sorted_df = df[(pd.to_datetime(df['DATE']) >= start_date) & (pd.to_datetime(df['DATE']) <= end_date)]


    fig = px.line(sorted_df,x='DATE',y='PNL', title='PNL')

    total_pnl = sorted_df['PNL'].sum()
    total_pnl = round(total_pnl,2)

    max_profit = sorted_df['PNL'].max()
    max_profit = round(max_profit,2)

    win_ratio = (sorted_df['Win_ratio'].sum() / sorted_df['Win_ratio'].count()) * 100
    win_ratio = round(win_ratio,2)
    win_ratio_per = f'{win_ratio} %'
    return fig, total_pnl, max_profit, win_ratio_per
app.run_server()