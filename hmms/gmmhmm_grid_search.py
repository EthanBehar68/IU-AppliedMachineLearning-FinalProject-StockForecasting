from fastquant import get_stock_data
from hmmlearn import hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from multiprocessing import Pool
import sys


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
        ax.set_xticks(np.arange(0,len(preds)+10,10))
        ax.set_xlim(0,len(preds)+10)
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

        return train_obs
    
    def log_lik_calc(self, observed, observations):
        log_liks = []
        for o in observations:
            obs = np.vstack((observed,o))
            log_lik = self.model.score(obs)
            log_liks.append((o,log_lik))

        return log_liks
    
    def test(self,test_data, train_obs):
        
        test_obs = self.data_prep(test_data)
        
        test_close_prices = test_data['close'].values
        test_open_prices = test_data['open'].values

        # use a latency of d days. So observations start as training data
        observed = train_obs.iloc[-self.d:].values
        
        # list to hold predictions
        preds = []
        
        # loop through all points we want to test
        for i in range(len(test_data)):
            # try 50x10x10 possible values for O_d+1
            change = np.arange(-0.1,0.1,0.2/50)
            high = np.arange(0,0.1,0.1/10)
            low = np.arange(0,0.1,0.1/10)

            observations = [np.array([c,h,l]) for l in low for h in high for c in change]
            
            # compute all log likelihoods w/ their observations in parallel
            jump = int(len(observations)/20)
            with Pool(processes=20) as pool:
                results = pool.starmap(self.log_lik_calc, [(observed, observations[i:i+jump]) for i in range(0,len(observations),jump)])

            best = {'obs':None, 'log_lik':-math.inf}
            for log_liks in results:
                for obs,log_lik in log_liks:
                    if log_lik > best['log_lik']:
                        best['obs'],best['log_lik'] = obs,log_lik

            # actually stack the best day on to the observations to use for next test point
            # drop the first thing in observed to shift our latency window `d`
            observed = np.vstack((observed,best['obs']))
            observed = observed[1:]

            #calculate the close value from best
            pred_close = best['obs'][0]*test_open_prices[i]+test_open_prices[i]
            preds.append(pred_close)

            print(f'{i+1}/{len(test_data)}',end='\r')
        print('DONE')
        return preds,test_close_prices
            

if __name__ == "__main__":
    # pull in data
    train_data_aapl = get_stock_data('AAPL','2003-02-10','2004-09-12')
    test_data_aapl = get_stock_data('AAPL','2004-09-13','2005-01-22')
    
    train_data_ibm = get_stock_data('IBM','2003-02-10','2004-09-12')
    test_data_ibm = get_stock_data('IBM','2004-09-13','2005-01-22')


    # preform grid search to find best model parameters
    best_params = {'n_components':None, 'n_mix':None, 'd':None, 'error':math.inf, 'aapl':None, 'ibm':None}
    
    n_components_vals = [2,3,4,5,6,7,8]
    n_mix_vals = [1,2,3,4,5,6,7]
    d_vals = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    for n_components in n_components_vals:
        for n_mix in n_mix_vals:
            for d in d_vals:
                print(f'n_components: {n_components} n_mix: {n_mix} d: {d}')

                try:
                    # training with apple feb-10-2003 -> sep-10-2004
                    # testing with apple sep-13-2004 -> jan-21-2005
                    
                    model = GmmHMM(n_components=n_components,
                                n_mix=n_mix,
                                algorithm="map",
                                n_iter=100,
                                d=d)

                    train_obs = model.train(train_data=train_data_aapl)

                    start = time.time()
                    preds,actual = model.test(test_data=test_data_aapl, train_obs=train_obs)
                    end = time.time()
                    print(f'model tested in {round((end-start)/60,2)} minutes')
                    error_aapl = model.mean_abs_percent_error(y_pred=preds, y_true=actual)
                    print(f'AAPL error: {error_aapl}')
                    
                    # training with IBM feb-10-2003 -> sep-10-2004
                    # testing with IBM sep-13-2004 -> jan-21-2005

                    model = GmmHMM(n_components=n_components,
                                n_mix=n_mix,
                                algorithm="map",
                                n_iter=100,
                                d=d)

                    train_obs = model.train(train_data=train_data_ibm)

                    start = time.time()
                    preds,actual = model.test(test_data=test_data_ibm, train_obs=train_obs)
                    end = time.time()
                    print(f'model tested in {round((end-start)/60,2)} minutes')
                    error_ibm = model.mean_abs_percent_error(y_pred=preds, y_true=actual)
                    print(f'IBM error: {error_ibm}')

                    error = error_aapl+error_ibm

                    if error < best_params['error']:
                        best_params['n_components'] = n_components
                        best_params['n_mix'] = n_mix
                        best_params['d'] = d
                        best_params['error'] = error
                        best_params['aapl'] = error_aapl
                        best_params['ibm'] = error_ibm

                except ValueError:
                    print('n_samples should be >= n_clusters')
    
    print(best_params)
    print(f'AAPL error: {best_params["aapl"]}')
    print(f'IBM error: {best_params["ibm"]}')