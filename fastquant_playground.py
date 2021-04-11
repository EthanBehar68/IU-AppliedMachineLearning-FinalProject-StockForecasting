from fastquant import get_stock_data, backtest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = get_stock_data("AAPL", "2001-03-26", "2021-03-26")
#print(df.head())
#print(df['open'][0])



#res = backtest("smac", df, fast_period=range(15,30,3), slow_period=range(40,55,3), verbose=False)
#print(res[['fast_period', 'slow_period', 'final_value']].head())


stock_tickers = ['AAPL', 'TSLA', 'GOOG']
crypto_tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD']

stock_tickers = {
    'AAPL': {
        'start_date': '2001-03-26',
        'end_date': '2021-03-26'
    },
    'TSLA': {
        'start_date': '2001-03-26',
        'end_date': '2021-03-26'
    },
    'GOOG': {
        'start_date': '2001-03-26',
        'end_date': '2021-03-26'
    },
}

strats = {
    'rsi': {
        'rsi_period': 14,
        'rsi_upper': 70,
        'rsi_lower': 30
    },
    'smac': {
        'slow_period': 40,
        'fast_period': 15 
    },
    'emac': {
        'slow_period': 40,
        'fast_period': 15
    },
    'macd': {
        'slow_period': 40, 
        'fast_period': 15,
        'signal_period': 9,
        'sma_period': 30,
        'dir_period': 10
    },
    'bbands': {
        'period': 20,
        'devfactor': 2.0
    },
}


def rsi(parameters):
    return backtest()



for i, ticker in enumerate(stock_tickers.keys()):
    start_date = stock_tickers[ticker]['start_date']
    end_date = stock_tickers[ticker]['end_date']
    
    df = get_stock_data(ticker, start_date, end_date)

    print(df)

    for i, key in enumerate(strats.keys()):
        parameters = strats[key]
        # python doesn't have a switch statement function which is lame
        if key == 'rsi':
            period = parameters['rsi_period']
            upper = parameters['rsi_upper']
            lower = parameters['rsi_lower']

            res = backtest(key, df, rsi_period=period, rsi_upper=upper, rsi_lower=lower)
        elif key == 'smac' or key == 'emac':
            fast = parameters['fast_period']
            slow = parameters['slow_period']

            res = backtest(key, df, fast_period=fast, slow_period=slow) 
        elif key == 'macd':
            fast = parameters['fast_period']
            slow = parameters['slow_period']
            signal = parameters['signal_period']
            sma = parameters['sma_period']
            dir = parameters['dir_period']

            res = backtest(key, df, fast_period=fast, slow_period=slow, signal_period=signal, sma_period=sma, dir_period=dir)
        elif key == 'bbands':
            period = parameters['period']
            dev = parameters['devfactor']

            res = backtest(key, df, period=period, devfactor=dev) 
        else:
            print('invalid key in the strats dictionary: {}'.format(key))