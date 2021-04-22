import sys
from test import *
from model import Model
from fastquant import get_stock_data
from hmmlearn import hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from multiprocessing import Pool


class GmmHMM(Model):
    def __init__(self,params):
        super().__init__(params)
        self.n_components = params['n_components']
        self.n_mix = params['n_mix']
        self.algorithm = params['algorithm']
        self.n_iter = params['n_iter']
        self.d = params['d']
    
    def train(self, train_data):
        self.train_obs = self.data_prep(train_data)
        self.model = hmm.GMMHMM(n_components=self.n_components,
                                n_mix=self.n_mix,
                                algorithm=self.algorithm,
                                n_iter=self.n_iter)

        self.model.fit(self.train_obs)
    
    def predict(self,test_data):        
        test_close_prices = test_data['close'].values
        test_open_prices = test_data['open'].values
        test_obs = self.data_prep(test_data).values

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

            # stack the actual observation on so we can predict next day
            # drop the first thing in observed to shift our latency window `d`
            observed = np.vstack((observed,test_obs[i]))
            observed = observed[1:]

            #calculate the close value from best
            pred_close = best['obs'][0]*test_open_prices[i]+test_open_prices[i]
            preds.append(pred_close)

            print(f'{i+1}/{len(test_data)}',end='\r',flush=True)
        print('DONE')
        return preds,test_close_prices
    
    def log_lik_calc(self, observed, observations):
        log_liks = []
        for o in observations:
            obs = np.vstack((observed,o))
            log_lik = self.model.score(obs)
            log_liks.append((o,log_lik))

        return log_liks
    
    def data_prep(self, data):
        df = pd.DataFrame(data=None, columns=['fracChange','fracHigh','fracLow'])
        df['fracChange'] = (data['close']-data['open'])/data['open']
        df['fracHigh'] = (data['high']-data['open'])/data['open']
        df['fracLow'] = (data['open']-data['low'])/data['open']

        return df


if __name__ == "__main__":
    params = {'n_components': 2, 
              'n_mix': 4, 
              'algorithm': 'map', 
              'n_iter': 100, 
              'd': 5,
              'name':'GMMHMM'}
    
    print('testing best found parameters paper tests')
    test = Test(Model=GmmHMM, params=params, tests=paper_tests, f='gmmhmm-paper-tests.json', plot=True)
    test.fixed_origin_tests()

    print('testing best found parameters own tests')
    test = Test(Model=GmmHMM, params=params, tests=own_tests, f='gmmhmm-own-tests.json', plot=True)
    test.fixed_origin_tests()

    print('testing')
    test = Test(Model=GmmHMM, params=params, tests=rolling_window_tests, f='gmmhmm-rolling-tests.json', plot=True)
    test.rolling_window_test()
