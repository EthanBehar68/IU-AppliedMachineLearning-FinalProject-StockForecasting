from fastquant import get_stock_data
from hmmlearn import hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class GmmHMM:
    def __init__(self,n_components,n_mix,algorithm,n_iter,d):
        self.n_components = n_components
        self.n_mix = n_mix
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.d = d

        self.model = None
    
    def get_data(self, ticker, start_date, end_date):
        return get_stock_data(ticker, start_date, end_date)

    def data_prep(self, data):
        df = pd.DataFrame(data=None, columns=['fracChange','fracHigh','fracLow'])
        df['fracChange'] = (data['close']-data['open'])/data['open']
        df['fracHigh'] = (data['high']-data['open'])/data['open']
        df['fracLow'] = (data['open']-data['low'])/data['open']

        return df

    def plot_results(self, preds, actual, title):
        fig, ax = plt.subplots(figsize=(15,5))
        ax.set_title(title)
        time = range(len(preds))
        ax.plot(time,preds,color='tab:red',marker='s',markersize=2,linestyle='-',linewidth=1,label='forcast')
        ax.plot(time,actual,color='tab:blue',marker='s',markersize=2,linestyle='-',linewidth=1,label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('stock price ($)')
        ax.set_xticks(np.arange(0,time+10,10))
        ax.set_xlim(0,time+10)
        ax.xaxis.grid(True,ls='--')
        ax.yaxis.grid(True,ls='--')
        ax.legend()
        plt.savefig(f'../imgs/{title}.png')
    
    def mean_abs_percent_error(self, y_pred, y_true):
        return (1.0)/(len(y_pred))*((np.abs(y_pred-y_true)/np.abs(y_true))*100).sum()
    
    def train(self, train_data):
        train_obs = self.data_prep(train_data)
        self.model = hmm.GMMHMM(n_components=self.n_components,
                                n_mix=self.n_mix,
                                algorithm=self.algorithm,
                                n_iter=self.n_iter)

        self.model.fit(train_obs)
    
    def test(self,test_data):
        test_obs = self.data_prep(test_data)
        
        test_close_prices = test_data['close'].values
        test_open_prices = test_data['open'].values

        # use a latency of d days. So observations start as training data
        observed = self.train_obs.iloc[-self.d:].values
        
        # list to hold predictions
        preds = []
        
        # loop through all points we want to test
        for i in range(len(test_data)):
            # try 50x10x10 possible values for O_d+1
            change = np.arange(-0.1,0.1,0.2/50)
            high = np.arange(0,0.1,0.1/10)
            low = np.arange(0,0.1,0.1/10)            
            observations = [np.vstack((observed,np.array([c,h,l]))) \
                for l in low for h in high for c in change]

            best = max(observations, key=self.model.score)

            # actually stack the best day on to the observations to use for next test point
            # drop the first thing in observed to shift our latency window `d`
            observed = np.vstack((observed,best))
            observed = observed[1:]

            #calculate the close value from best
            pred_close = best[0]*test_open_prices[i]+test_open_prices[i]
            preds.append(pred_close)
                
        return preds,test_close_prices
            

if __name__ == "__main__":
    # training with apple feb-10-2003 -> sep-10-2004
    # testing with apple sep-13-2004 -> jan-21-2005

    model = GmmHMM(n_components=4,
                   n_mix=5,
                   algorithm="map",
                   n_iter=100,
                   d=10)
    
    train_data = model.get_data(ticker='AAPL',start_date='2003-02-10',end_date='2004-09-10')
    test_data = model.get_data(ticker='AAPL',start_date='2004-09-13',end_date='2005-01-21')

    model.train(train_data=train_data)

    preds,actual = model.test(test_data=test_data)
    error = self.model.mean_abs_percent_error(y_pred=preds, y_true=actual)
    print(f'AAPL error: {error}')

    model.plot_results(preds=preds, actual=actual, 
                       title='GMM HMM AAPL forcasted vs actual stock prices Sep 2004 - Jan 2005')
    
    # training with IBM feb-10-2003 -> sep-10-2004
    # testing with IBM sep-13-2004 -> jan-21-2005

    model = GmmHMM(n_components=4,
                   n_mix=5,
                   algorithm="map",
                   n_iter=100,
                   d=10)
    
    train_data = model.get_data(ticker='IBM',start_date='2003-02-10',end_date='2004-09-10')
    test_data = model.get_data(ticker='IBM',start_date='2004-09-13',end_date='2005-01-21')

    model.train(train_data=train_data)

    preds,actual = model.test(test_data=test_data)
    error = self.model.mean_abs_percent_error(y_pred=preds, y_true=actual)
    print(f'IBM error: {error}')

    model.plot_results(preds=preds, actual=actual, 
                       title='GMM HMM IBM forcasted vs actual stock prices Sep 2004 - Jan 2005')
    
    # training with dell feb-10-2003 -> sep-10-2004
    # testing with dell sep-13-2004 -> jan-21-2005

    model = GmmHMM(n_components=4,
                   n_mix=5,
                   algorithm="map",
                   n_iter=100,
                   d=10)
    
    train_data = model.get_data(ticker='DELL',start_date='2003-02-10',end_date='2004-09-10')
    test_data = model.get_data(ticker='DELL',start_date='2004-09-13',end_date='2005-01-21')

    model.train(train_data=train_data)

    preds,actual = model.test(test_data=test_data)
    error = self.model.mean_abs_percent_error(y_pred=preds, y_true=actual)
    print(f'DELL error: {error}')

    model.plot_results(preds=preds, actual=actual, 
                       title='GMM HMM DELL forcasted vs actual stock prices Sep 2004 - Jan 2005')





# closes = testing_data['close'].values
# opens = testing_data['open'].values
# preds = []


# observed = x_train.iloc[-10:].values
# for i in range(len(x_test)):
#     # d = 10
#     # create observed data of O_1,...,O_d,O_d+1
#     # try 50x10x10 possible values for O_d+1 and find max log lik
#     fracChange = np.arange(-0.1,0.1,0.2/50)
#     fracHigh = np.arange(0,0.1,0.1/10)
#     fracLow = np.arange(0,0.1,0.1/10)
#     best = {'next_day': None, 'loglik': -math.inf}
#     for change in fracChange:
#         for high in fracHigh:
#             for low in fracLow:
#                 next_day = np.array([change,high,low])
#                 observed_test = np.vstack((observed,next_day))
#                 log_lik = model.score(observed_test)
#                 if log_lik > best['loglik']:
#                     best['next_day'],best['loglik'] = next_day,log_lik

#     best_frac_change = best['next_day']
#     observed = np.vstack((observed,next_day))
#     # calc predicted close value
#     pred_close = best['next_day'][0]*opens[i]+opens[i]
#     preds.append(pred_close)

#     # drop the O_1 observation to slide the window for the next observation
#     observed = observed[1:]

#     print(f'predicted close: {pred_close}')
#     print(f'actual close   : {closes[i]}\n')


# print(best)
# open_test = testing_data['open'][0]
# pred_close = best['next_day'][0]*open_test+open_test
# print(pred_close)
# print(testing_data['close'][0])