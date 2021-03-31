from fastquant import get_stock_data
from hmmlearn import hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# training with apple feb-10-2003 -> sep-10-2004
# testing with apple sep-13-2004 -> jan-21-2005

training_data = get_stock_data("AAPL", "2003-02-10", "2004-09-10")
testing_data = get_stock_data("AAPL", "2004-09-13", "2005-01-21")

def data_prep(data):
    df = pd.DataFrame(data=None, columns=['fracChange','fracHigh','fracLow'])
    df['fracChange'] = (data['close']-data['open'])/data['open']
    df['fracHigh'] = (data['high']-data['open'])/data['open']
    df['fracLow'] = (data['open']-data['low'])/data['open']

    return df

x_train = data_prep(training_data)
x_test = data_prep(testing_data)

print('TEST DATA')
print(x_test.head())

print('TRAIN DATA')
print(x_train.tail())

model = hmm.GMMHMM(n_components=4,n_mix=5,algorithm="map",n_iter=100)
model.fit(x_train)

closes = testing_data['close'].values
opens = testing_data['open'].values
preds = []


observed = x_train.iloc[-10:].values
for i in range(len(x_test)):
    # d = 10
    # create observed data of O_1,...,O_d,O_d+1
    # try 50x10x10 possible values for O_d+1 and find max log lik
    fracChange = np.arange(-0.1,0.1,0.2/50)
    fracHigh = np.arange(0,0.1,0.1/10)
    fracLow = np.arange(0,0.1,0.1/10)
    best = {'next_day': None, 'loglik': -math.inf}
    for change in fracChange:
        for high in fracHigh:
            for low in fracLow:
                next_day = np.array([change,high,low])
                observed_test = np.vstack((observed,next_day))
                log_lik = model.score(observed_test)
                if log_lik > best['loglik']:
                    best['next_day'],best['loglik'] = next_day,log_lik

    best_frac_change = best['next_day']
    observed = np.vstack((observed,next_day))
    # calc predicted close value
    pred_close = best['next_day'][0]*opens[i]+opens[i]
    preds.append(pred_close)

    # drop the O_1 observation to slide the window for the next observation
    observed = observed[1:]

    print(f'predicted close: {pred_close}')
    print(f'actual close   : {closes[i]}\n')


# print(best)
# open_test = testing_data['open'][0]
# pred_close = best['next_day'][0]*open_test+open_test
# print(pred_close)
# print(testing_data['close'][0])
