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
            {'ticker':'DIS', 'start':'2009-01-01', 'end':'2015-01-01'},
        'test':
            {'ticker':'DIS', 'start':'2015-01-02', 'end':'2016-01-02'}
    },
    'test6': {
        'train':
            {'ticker':'TMO', 'start':'2009-01-01', 'end':'2015-01-01'},
        'test':
            {'ticker':'TMO', 'start':'2015-01-02', 'end':'2016-01-02'}
    }
}

rolling_window_tests = {
    'test1': {
        'window':
            {'ticker':'AMZN', 'start':'2015-01-01', 'end':'2020-01-01'},
    },
    'test2': {
        'window':
            {'ticker':'MSFT', 'start':'2015-01-01', 'end':'2020-01-01'},
    },
    'test3': {
        'window':
            {'ticker':'GOOGL', 'start':'2015-01-01', 'end':'2020-01-01'},
    },
    'test4': {
        'window':
            {'ticker':'DPZ', 'start':'2015-01-01', 'end':'2020-01-01'},
    },
    'test5': {
        'window':
            {'ticker':'DIS', 'start':'2015-01-01', 'end':'2020-01-01'},
    },
    'test6': {
        'window':
            {'ticker':'TMO', 'start':'2015-01-01', 'end':'2020-01-01'},
    }
}

class Test:
    def __init__(self, Model, params, tests, f, plot=False):
        self.Model = Model
        self.params = params
        self.tests = tests
        self.results = {}
        self.f = f
        self.plot = plot
    
    def fixed_origin_tests(self):
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

            self.results[f'{self.model.name}:{ticker}'] = error

            # plot results if flag is set
            if self.plot:
                self.model.plot_results(preds=preds, actual=actuals,
                                        title=f'{self.model.name} {ticker} forcasted vs actual stock prices {testing_params["start"]} to {testing_params["end"]}')
        
        # write errors to file
        dump = json.dumps(self.results)
        output_file = open(self.f, 'w')
        output_file.write(dump)
        output_file.close()

    def rolling_window_test(self):
        # train on 1155 points, test on 10 points
        # slide window over by testing_size each time to get 10 tests
        training_size = 1155
        testing_size = 10

        for test in self.tests.values():
            # var to store error and test num
            error = 0
            test_n = 0

            # collect the data for the window
            window_params = test['window']
            ticker = window_params['ticker']

            window = get_stock_data(ticker,window_params['start'],window_params['end'])

            # 10 tests within the window
            for i in range(0,100,10):
                train_data = window.iloc[i:i+training_size]
                test_data = window.iloc[i+training_size:i+training_size+testing_size]

                print(f'window {i+1}')

                # make the model
                self.model = self.Model(params=self.params)

                # train and predict
                self.model.train(train_data=train_data)
                preds, actuals = self.model.predict(test_data=test_data)

                # get error for this window
                error += self.model.mean_abs_percent_error(y_pred=preds, y_true=actuals)
                test_n += 1

                print('DONE')
            
            # use last window for plotting
            if self.plot:
                self.model.plot_continuous(preds=preds, train=train_data, actual=actuals,
                                          title=f'{self.model.name} {ticker} forecasted vs actual continuous stock price')
            
            # store average MAPE error
            avg_error = error/test_n
            self.results[f'{self.model.name}:{ticker}'] = avg_error
        
        # write errors to file
        dump = json.dumps(self.results)
        output_file = open(self.f, 'w')
        output_file.write(dump)
        output_file.close()

if __name__ == "__main__":
    test = rolling_window_tests['test6']['window']
    df = get_stock_data(test['ticker'],test['start'],test['end'])
    print('df')
    print(df)
    for i in range(0,100,10):
        train = df.iloc[i:i+1155]
        test = df.iloc[i+1155:i+1155+10]
        print(i)
        print(train)
        print(test)
