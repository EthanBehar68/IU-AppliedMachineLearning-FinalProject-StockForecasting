from fastquant import get_stock_data, backtest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = get_stock_data("AAPL", "2001-03-26", "2021-03-26")
#print(df.head())
#print(df['open'][0])



#res = backtest("rsi", df, rsi_period=14, rsi_upper=70, rsi_lower=30)
#res = backtest('macd', df, fast_period=15, slow_period=40, signal_period=9, sma_period=30, dir_period=10)
#res = backtest('bbands', df, period=20, devfactor=2.0) 
#print(res[['fast_period', 'slow_period', 'final_value']].head())
#print(res.info())

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
    'buynhold':{},
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


res = pd.DataFrame()
'''
Note that this is the output for the smac strategy, indexes will be different for different strats

$ -> we're interested in using this attribute for performance analysis

RangeIndex: 1 entries, 0 to 0
Data columns (total 29 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   strat_id           1 non-null      int64  
 1   init_cash          1 non-null      int64  $  
 2   buy_prop           1 non-null      int64  
 3   sell_prop          1 non-null      int64  
 4   commission         1 non-null      float64
 5   stop_loss          1 non-null      int64  
 6   stop_trail         1 non-null      int64  
 7   execution_type     1 non-null      object 
 8   channel            1 non-null      object 
 9   symbol             1 non-null      object 
 10  allow_short        1 non-null      bool   
 11  short_max          1 non-null      float64
 12  add_cash_amount    1 non-null      int64  
 13  add_cash_freq      1 non-null      object 
 14  fast_period        1 non-null      int64  $ 
 15  slow_period        1 non-null      int64  $ 
 16  rtot               1 non-null      float64
 17  ravg               1 non-null      float64
 18  rnorm              1 non-null      float64
 19  rnorm100           1 non-null      float64
 20  len                1 non-null      int64  
 21  drawdown           1 non-null      float64
 22  moneydown          1 non-null      float64
 23  max                1 non-null      object 
 24  maxdrawdown        1 non-null      float64
 25  maxdrawdownperiod  1 non-null      int64  
 26  sharperatio        1 non-null      float64
 27  pnl                1 non-null      float64
 28  final_value        1 non-null      float64  $
'''

buynhold_results = []
rsi_results = []
smac_results = []
emac_results = []
macd_results = []
bbands_results = []


for i, ticker in enumerate(stock_tickers.keys()):
    start_date = stock_tickers[ticker]['start_date']
    end_date = stock_tickers[ticker]['end_date']
    
    df = get_stock_data(ticker, start_date, end_date)

    print(df)
    for i, key in enumerate(strats.keys()):
        parameters = strats[key]
        # python doesn't have a switch statement function which is lame
        if key == 'buynhold':
            res = backtest(key, df)
            
            percent_gain = (res[1]-res[26])/res[1]
        elif key == 'rsi':
            period = parameters['rsi_period']
            upper = parameters['rsi_upper']
            lower = parameters['rsi_lower']

            res = backtest(key, df, rsi_period=period, rsi_upper=upper, rsi_lower=lower)
            percent_gain = (res[29]-res[1])/res[1]

            curr = {
                'percent_gain': percent_gain,
                'rsi_period': period,
                'rsi_upper': upper,
                'rsi_lower': lower
            }

            rsi_results.append(curr)
        elif key == 'smac' or key == 'emac':
            fast = parameters['fast_period']
            slow = parameters['slow_period']

            res = backtest(key, df, fast_period=fast, slow_period=slow) 
            init_cash = res[1]
            final_value = res[28]
            percent_gain = (final_value-init_cash)/init_cash

            curr = {
                'percent_gain': percent_gain,
                'fast_period': fast,
                'slow_period': slow,
                'start': start_date,
                'end': end_date
            }
            
            if key == 'smac':
                smac_results.append(curr)
            else:
                emac_results.append(curr)

        elif key == 'macd':
            fast = parameters['fast_period']
            slow = parameters['slow_period']
            signal = parameters['signal_period']
            sma = parameters['sma_period']
            dir = parameters['dir_period']

            res = backtest(key, df, fast_period=fast, slow_period=slow, signal_period=signal, sma_period=sma, dir_period=dir)

            percent_gain = (res[31]-res[1])/res[1]
            
            curr = {
                'percent_gain': percent_gain,
                'fast_period': fast,
                'slow_period': slow,
                'signal_period': signal,
                'sma_period': sma,
                'dir_period': dir
            }

            macd_results.append(curr)
        elif key == 'bbands':
            period = parameters['period']
            dev = parameters['devfactor']

            res = backtest(key, df, period=period, devfactor=dev) 

            percent_gain = (res[28]-res[1])/res[1]

            curr = {
                'percent_gain': percent_gain,
                'period': period,
                'devfactor': dev
            }
        else:
            print('invalid key in the strats dictionary: {}'.format(key))
