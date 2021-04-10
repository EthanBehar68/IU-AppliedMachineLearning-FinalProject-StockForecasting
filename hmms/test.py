import json
from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# file to test using multiple tickers with dates
paper_tests = {
    'test1': {
        'train':
            {'ticker':'AAPL', 'start':'2003-02-10', 'end':'2004-09-12'},
        'test':
            {'ticker':'AAPL', 'start':'2004-09-13', 'end':'2005-01-22'}
    },
    'test2': {
        'train':
            {'ticker':'IBM', 'start':'2003-02-10', 'end':'2004-09-12'},
        'test':
            {'ticker':'IBM', 'start':'2004-09-13', 'end':'2005-01-22'}
    }
}

# own tests, more training data, and more recent stocks
own_tests = {
    'test1': {
        'train':
            {'ticker':'AMZN', 'start':'2009-01-01', 'end':'2015-01-01'},
        'test':
            {'ticker':'AMZN', 'start':'2015-01-02', 'end':'2016-01-02'}
    },
    'test2': {
        'train':
            {'ticker':'MSFT', 'start':'2009-01-01', 'end':'2015-01-01'},
        'test':
            {'ticker':'MSFT', 'start':'2015-01-02', 'end':'2016-01-02'}
    },
    'test3': {
        'train':
            {'ticker':'GOOGL', 'start':'2009-01-01', 'end':'2015-01-01'},
        'test':
            {'ticker':'GOOGL', 'start':'2015-01-02', 'end':'2016-01-02'}
    },
    'test4': {
        'train':
            {'ticker':'DPZ', 'start':'2009-01-01', 'end':'2015-01-01'},
        'test':
            {'ticker':'DPZ', 'start':'2015-01-02', 'end':'2016-01-02'}
    },
    'test5': {
        'train':
            {'ticker':'DPZ', 'start':'2009-01-01', 'end':'2015-01-01'},
        'test':
            {'ticker':'DPZ', 'start':'2015-01-02', 'end':'2016-01-02'}
    },
    'test6': {
        'train':
            {'ticker':'TMO', 'start':'2009-01-01', 'end':'2015-01-01'},
        'test':
            {'ticker':'TMO', 'start':'2015-01-02', 'end':'2016-01-02'}
    }
}

class Test:
    def __init__(self, Model, params, tests, f):
        self.Model = Model
        self.params = params
        self.tests = tests
        self.results = {}
        self.f = f
    
    def run_tests(self):
        for test in self.tests.values():
            training_params = test['train']
            testing_params = test['test']

            ticker = training_params['ticker']

            # make the model
            self.model = self.Model(params=self.params)

            # collect data from fastquant
            train_data = self.model.get_data(ticker=ticker,
                                        start_date=training_params['start'],
                                        end_date=training_params['end'])

            test_data = self.model.get_data(ticker=ticker,
                                       start_date=testing_params['start'],
                                       end_date=testing_params['end'])

            # train and predict
            self.model.train(train_data=train_data)
            preds, actuals = self.model.predict(test_data=test_data)

            # get and save error
            error = self.model.mean_abs_percent_error(y_pred=preds, y_true=actuals)

            self.results[f'{model.name}:{ticker}'] = error
        
        # write errors to file
        json = json.dumps(self.results)
        f = open(f, 'w')
        f.write(json)
        f.close()