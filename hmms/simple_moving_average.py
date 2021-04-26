import sys
from test import *
from model import Model
from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

# True base line model using simple moving average
# the model is just using the past d days average as the prediction for d+1
class SMA(Model):
    def __init__(self,params):
        super().__init__(params)
        self.d = params['d']
    
    def train(self, train_data):
        # no work needed for training
        self.train_data = train_data['close'].values.reshape(-1,1)
    
    def predict(self,test_data):        
        test_close_prices = test_data['close'].values

        # use a latency of d days. So observations start as training data
        observed = self.train_data[-self.d:]
        
        # list to hold predictions
        preds = []
        
        # loop through all points we want to test
        for i in range(len(test_data)):
            # prediction is average of past d days (the observed days)
            pred_close = observed.mean()

            # stack the actual observation on so we can predict next day
            # drop the first thing in observed to shift our latency window `d`
            observed = np.vstack((observed,test_close_prices[i]))
            observed = observed[1:]

            preds.append(pred_close)

            print(f'{i+1}/{len(test_data)}',end='\r',flush=True)
        print('DONE')
        return preds,test_close_prices


if __name__ == "__main__":
    params = {'d': 10,
              'name': 'SMA-10'}
    
    test = Test(Model=SMA, params=params, tests=paper_tests, f='sma-10-paper-tests.json', plot=True)
    test.fixed_origin_tests()

    test = Test(Model=SMA, params=params, tests=own_tests, f='sma-10-own-tests.json', plot=True)
    test.fixed_origin_tests()

    test = Test(Model=SMA, params=params, tests=rolling_window_tests, f='sma-10-rolling-tests.json', plot=True)
    test.rolling_window_test()
