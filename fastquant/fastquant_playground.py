from fastquant import get_stock_data, backtest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





'''
df = get_stock_data("AAPL", "2001-03-26", "2021-03-26")
#print(df.head())
#print(df['open'][0])


res = backtest("buynhold", df, verbose=0, plot=False)

print(res.info())

#res = backtest('macd', df, fast_period=15, slow_period=40, signal_period=9, sma_period=30, dir_period=10)
#res = backtest('bbands', df, period=20, devfactor=2.0) 
#print(res[['fast_period', 'slow_period', 'final_value']].head())
#print(type(res))
#print(res.info())

for key in history.keys():
    print('[][][][][][][][][][][][][][][][][][][]')
    print(key)
    print(history[key].info())


for key in history.keys():
    print(history[key].head())


'''
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
        'rsi1': {
            'rsi_period': 14,
            'rsi_upper': 70,
            'rsi_lower': 30
        },
       'rsi2': {
            'rsi_period': 14,
            'rsi_upper': 60,
            'rsi_lower': 40
        }, 
        'rsi3': {
            'rsi_period': 20,
            'rsi_upper': 70,
            'rsi_lower': 30
        },
        'rsi4': {
            'rsi_period': 20,
            'rsi_upper': 60,
            'rsi_lower': 40
        }, 
        'rsi5': {
            'rsi_period': 16,
            'rsi_upper': 65,
            'rsi_lower': 35
        }, 
    },
    'smac': {
        'smac1': {
            'slow_period': 40,
            'fast_period': 15 
        },
       'smac2': {
            'slow_period': 45,
            'fast_period': 20 
        }, 
        'smac3': {
            'slow_period': 50,
            'fast_period': 25 
        },
        'smac4': {
            'slow_period': 30,
            'fast_period': 10 
        },
        'smac5': {
            'slow_period': 30,
            'fast_period': 15 
        },
    },
    'emac': {
        'emac1':{
            'slow_period': 40,
            'fast_period': 15
        },
        'emac2':{
            'slow_period': 45,
            'fast_period': 20
        },
        'emac3':{
            'slow_period': 50,
            'fast_period': 25
        },
        'emac4':{
            'slow_period': 30,
            'fast_period': 10
        },
        'emac5':{
            'slow_period': 30,
            'fast_period': 15
        },
        
    },
    'macd': {
        'macd1':{
            'slow_period': 40, 
            'fast_period': 15,
            'signal_period': 9,
            'sma_period': 30,
            'dir_period': 10
        },
       'macd2':{
            'slow_period': 45, 
            'fast_period': 20,
            'signal_period': 9,
            'sma_period': 30,
            'dir_period': 10
        }, 
        'macd3':{
            'slow_period': 50, 
            'fast_period': 25,
            'signal_period': 9,
            'sma_period': 30,
            'dir_period': 10
        },
        'macd4':{
            'slow_period': 30, 
            'fast_period': 10,
            'signal_period': 9,
            'sma_period': 30,
            'dir_period': 10
        },
        'macd5':{
            'slow_period': 30, 
            'fast_period': 15,
            'signal_period': 9,
            'sma_period': 30,
            'dir_period': 10
        }
    },
    'bbands': {
        'bbands1': {
            'period': 20,
            'devfactor': 2.0
        },
       'bbands2': {
            'period': 30,
            'devfactor': 2.0
        }, 
        'bbands3': {
            'period': 40,
            'devfactor': 2.0
        },
        'bbands4': {
            'period': 20,
            'devfactor': 1.5
        },
        'bbands5': {
            'period': 20,
            'devfactor': 2.5
        },
        'bbands6': {
            'period': 30,
            'devfactor': 2.5
        },
    }
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
start_date = 0
end_date = 0


def buynhold():
    res, history = backtest('buynhold', 
                            df,
                            plot=False,
                            return_history=True,
                            commission=0,
                            verbose=0)

    final = float(res['final_value'])
    init = float(res['init_cash'])

    percent_gain = (final-init)/init
    #print(len(res.columns))
    #print(res.columns)
    #print(res.info())
    curr = {
        'percent_gain': percent_gain
    }
    
    return curr

def rsi(period, upper, lower):
    res, history = backtest('rsi', 
                            df, 
                            rsi_period=period, 
                            rsi_upper=upper, 
                            rsi_lower=lower, 
                            plot=False, 
                            return_history=True, 
                            commission=0,
                            verbose=0)



    final = float(res['final_value'])
    init = float(res['init_cash'])

    percent_gain = (final-init)/init

    curr = {
        'percent_gain': percent_gain,
        'rsi_period': period,
        'rsi_upper': upper,
        'rsi_lower': lower,
    #   'orders': history['orders']
    }

    return curr

def smac(fast, slow):
    res, history = backtest('smac', 
                            df, 
                            fast_period=fast, 
                            slow_period=slow,
                            plot=False,
                            return_history=True,
                            commission=0,
                            verbose=0) 

    final = float(res['final_value'])
    init = float(res['init_cash'])

    percent_gain = (final-init)/init

    curr = {
        'percent_gain': percent_gain,
        'fast_period': fast,
        'slow_period': slow,
        'start': start_date,
        'end': end_date,
    #    'orders': history['orders']
    }

    return curr

def emac(fast, slow):
    res, history = backtest('emac', 
                            df, 
                            fast_period=fast, 
                            slow_period=slow,
                            plot=False,
                            return_history=True,
                            commission=0,
                            verbose=0
                            ) 


    final = float(res['final_value'])
    init = float(res['init_cash'])

    percent_gain = (final-init)/init

    curr = {
        'percent_gain': percent_gain,
        'fast_period': fast,
        'slow_period': slow,
        'start': start_date,
        'end': end_date,
    #    'orders': history['orders']
    }

    return curr



def macd(fast, slow, signal, sma, dir):
    res, history = backtest('macd', 
                            df, 
                            fast_period=fast, 
                            slow_period=slow, 
                            signal_period=signal, 
                            sma_period=sma, 
                            dir_period=dir,
                            plot=False,
                            return_history=True,
                            commission=0,
                            verbose=0)


    final = float(res['final_value'])
    init = float(res['init_cash'])

    percent_gain = (final-init)/init
    
    curr = {
        'percent_gain': percent_gain,
        'fast_period': fast,
        'slow_period': slow,
        'signal_period': signal,
        'sma_period': sma,
        'dir_period': dir,
    #    'orders': history['orders']
    }

    return curr


def bbands(period, dev):
    res, history = backtest('bbands', 
                            df, 
                            period=period, 
                            devfactor=dev,
                            plot=False,
                            return_history=True,
                            commission=0,
                            verbose=0)

    final = float(res['final_value'])
    init = float(res['init_cash'])

    percent_gain = (final-init)/init

    curr = {
        'percent_gain': percent_gain,
        'period': period,
        'devfactor': dev,
    #    'orders': history['orders']
    }

    return curr

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

    for j, key in enumerate(strats.keys()):
        strategy = strats[key]
        for k, specific_strat in enumerate(strats[key].keys()):
            parameters = strategy[specific_strat]
            # python doesn't have a switch statement function which is lame
            if key == 'buynhold':
                res = buynhold()
                res['ticker'] = ticker
                res['start_date'] = start_date
                res['end_date'] = end_date
                buynhold_results.append(res)
                
            elif key == 'rsi':
                res = rsi(parameters['rsi_period'], parameters['rsi_upper'], parameters['rsi_lower'])
                res['ticker'] = ticker
                res['start_date'] = start_date
                res['end_date'] = end_date
                rsi_results.append(res)

            elif key == 'smac':
                res = smac(parameters['fast_period'], parameters['slow_period'])
                res['ticker'] = ticker
                res['start_date'] = start_date
                res['end_date'] = end_date
                smac_results.append(res) 

            elif key == 'emac':
                fast = parameters['fast_period']
                slow = parameters['slow_period']
                res = emac(fast, slow)
                res['ticker'] = ticker
                res['start_date'] = start_date
                res['end_date'] = end_date
                emac_results.append(res)

            elif key == 'macd':
                fast = parameters['fast_period']
                slow = parameters['slow_period']
                signal = parameters['signal_period']
                sma = parameters['sma_period']
                dir = parameters['dir_period']

                res = macd(fast, slow, signal, sma, dir)
                res['ticker'] = ticker
                res['start_date'] = start_date
                res['end_date'] = end_date
                macd_results.append(res)
                
            elif key == 'bbands':
                period = parameters['period']
                dev = parameters['devfactor']

                res = bbands(period, dev) 
                res['ticker'] = ticker
                res['start_date'] = start_date
                res['end_date'] = end_date
                bbands_results.append(res)
            else:
                print(f'invalid key in the strats dictionary: {key}')


print(type(rsi_results[0]['percent_gain']))

buynhold_results = sorted(buynhold_results, key=lambda x: x['percent_gain'], reverse=True)
rsi_results = sorted(rsi_results, key=lambda x: x['percent_gain'], reverse=True)
smac_results = sorted(smac_results, key=lambda x: x['percent_gain'], reverse=True)
emac_results = sorted(emac_results, key=lambda x: x['percent_gain'], reverse=True)
macd_results = sorted(macd_results, key=lambda x: x['percent_gain'], reverse=True)
bbands_results = sorted(bbands_results, key=lambda x: x['percent_gain'], reverse=True)


print('buynhold =========================================')
print(buynhold_results)
print()
print('rsi =========================================')
print(rsi_results)
print()
print('smac =========================================')
print(smac_results)