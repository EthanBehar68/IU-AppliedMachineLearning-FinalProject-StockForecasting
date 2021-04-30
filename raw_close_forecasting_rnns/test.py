from fastquant import get_stock_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import math

heavy_hitters_tests = {
    'test1': {
        'train':
            {'ticker':'F', 'start':'2007-01-01', 'end':'2015-01-01'}, # Ford
        'test':
            {'ticker':'F', 'start':'2015-01-02', 'end':'2016-02-23'}
    }
    ,
    'test2': {
        'train':
            {'ticker':'MSFT', 'start':'2007-01-01', 'end':'2015-01-01'}, # Microsoft
        'test':
            {'ticker':'MSFT', 'start':'2015-01-02', 'end':'2016-01-02'}
    },
    'test3': {
        'train':
            {'ticker':'AMZN', 'start':'2007-01-01', 'end':'2015-01-01'}, # Amazon
        'test':
            {'ticker':'AMZN', 'start':'2015-01-02', 'end':'2016-01-02'}
    },
    'test4': {
        'train':
            {'ticker':'MRK', 'start':'2007-01-01', 'end':'2015-01-01'}, # Merck & Co.
        'test':
            {'ticker':'MRK', 'start':'2015-01-02', 'end':'2016-01-02'}
    },
    'test5': {
        'train':
            {'ticker':'NKE', 'start':'2007-01-01', 'end':'2015-01-01'}, # Nike
        'test':
            {'ticker':'NKE', 'start':'2015-01-02', 'end':'2016-01-02'}
    }
}

window_heavy_hitters_tests = {
    'test1': {
        'window':
            {'ticker':'F', 'start':'2007-01-01', 'end':'2016-02-23'} # Ford
    }
    ,
    'test2': {
        'window':
            {'ticker':'AMZN', 'start':'2007-01-01', 'end':'2016-02-23'}, # Amazon
    },
    'test3': {
        'window':
            {'ticker':'MRK', 'start':'2007-01-01', 'end':'2016-02-23'}, # Merck & Co.
    }
}

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
    def __init__(self, Model, Train_Predictor, params, tests, plot=False):
        self.Model = Model
        self.Train_Predictor = Train_Predictor
        self.params = params
        self.tests = tests
        self.results = {}
        self.plot = plot
    
    def fixed_origin_tests(self, folder):
        for test in self.tests.values():
            training_params = test['train']
            testing_params = test['test']

            ticker = training_params['ticker']

            # Make the model and train_predictor
            self.model_class = self.Model(params=self.params)
            self.model = self.model_class.gen_model()
            self.train_predictor = self.Train_Predictor(params=self.params)

            # collect data from fastquant
            train_data = self.model_class.get_data(ticker=ticker,
                                        start_date=training_params['start'],
                                        end_date=training_params['end'])

            test_data = self.model_class.get_data(ticker=ticker,
                                       start_date=testing_params['start'],
                                       end_date=testing_params['end'])

            # train and predict
            self.model, model_history = self.train_predictor.train(model=self.model, train_data=train_data, label_column_index=self.model_class.label_column_index)
            preds, actuals = self.train_predictor.predict(model=self.model, test_data=test_data, label_column_index=self.model_class.label_column_index)

            # print(model_history.history['loss'])
            # print(model_history.history['val_loss'])

            # get error for this window
            if self.params['loss'] == "mean_absolute_percentage_error": # updated to keras' name for the loss
                error = self.model_class.mean_abs_percent_error(y_pred=preds, y_true=actuals)
            elif self.params['loss'] == "root_mean_squared_error":
                error = self.model_class.root_mean_squared_error(y_pred=preds, y_true=actuals)
            else:
                raise ValueError("Loss parameter isn't programmed or incorrect. Loss parameter: " + self.params['loss'])

            self.results[f'{self.model_class.name}:{ticker}'] = error

            # plot results if flag is set
            if self.plot:
                self.model_class.plot_results(preds=preds, actual=actuals, title=f'{self.model_class.name} {ticker} forcasted vs actual stock prices {testing_params["start"]} to {testing_params["end"]}', folder=folder)
                if model_history is not None:
                    self.model_class.plot_loss(t_loss=model_history.history['loss'], v_loss=model_history.history['val_loss'], title=f'{self.model_class.name} {ticker} train vs validation loss {testing_params["start"]} to {testing_params["end"]}', folder=folder)

        # write errors to file
        json_file = f'{folder}{self.model_class.name}.json'
        dump = json.dumps(self.results)
        output_file = open(json_file, 'w')
        output_file.write(dump)
        output_file.close()

    def rolling_window_test(self, folder, windows=10, train_size=1300, test_size=100):
        for test in self.tests.values():
            # var to store error and test num
            error = 0
            test_n = 0

            # collect the data for the window
            window_params = test['window']
            ticker = window_params['ticker']

            window = get_stock_data(ticker, window_params['start'], window_params['end'])
            print('Window shape: ', window.shape)
            data_size = window.shape[0]
            # testing_size = 10

            index = 0
            for i in range(0, windows * test_size, test_size):
                print('Train range: ', i, '-', i+train_size)
                print('Test range: ', i+train_size, '-', i+train_size+test_size)
                train_data = window.iloc[i:i+train_size]
                test_data = window.iloc[i+train_size:i+train_size+test_size]
                print('Window train data shape: ', train_data.shape)
                print('Window test data shape: ', test_data.shape)

                # print(f'window {i+1}')

                # Make the model and train_predictor
                self.model_class = self.Model(params=self.params)
                self.model = self.model_class.gen_model()
                self.train_predictor = self.Train_Predictor(params=self.params)

                # train and predict
                self.model, model_history = self.train_predictor.train(model=self.model, train_data=self.model_class.preprocess_data(train_data), label_column_index=self.model_class.label_column_index)
                preds, actuals = self.train_predictor.predict(model=self.model, test_data=self.model_class.preprocess_data(test_data), label_column_index=self.model_class.label_column_index)

                # get error for this window
                if self.params['loss'] == "mean_absolute_percentage_error": # updated to keras' name for the loss
                    error += self.model_class.mean_abs_percent_error(y_pred=preds, y_true=actuals)
                elif self.params['loss'] == "root_mean_squared_error":
                    error += self.model_class.root_mean_squared_error(y_pred=preds, y_true=actuals)
                # A bit weird - ask Rowan
                elif self.params['loss'] == "mean_squared_error":
                    error += self.model_class.mean_abs_percent_error(y_pred=preds, y_true=actuals)
                else:
                    raise ValueError("Loss parameter isn't programmed or incorrect. Loss parameter: " + self.params['loss'])
                test_n += 1

                print('DONE')
            
                # use last window for plotting
                if self.plot:                        
                    self.model_class.plot_results(preds=preds, actual=actuals, title=f'{self.model_class.name} {ticker} Window {index+1} forcasted vs actual stock prices {window_params["start"]} to {window_params["end"]}', folder=folder)
                    if model_history is not None:
                        self.model_class.plot_loss(t_loss=model_history.history['loss'], v_loss=model_history.history['val_loss'], title=f'{self.model_class.name} {ticker} Window {index+1} train vs validation loss', folder=folder)            

                print('DONE PLOTTING')
                index += 1

            # store average MAPE error
            avg_error = error/test_n
            self.results[f'{self.model_class.name}:{ticker}'] = avg_error

        # write errors to file
        json_file = f'{folder}{self.model_class.name}.json'
        dump = json.dumps(self.results)
        output_file = open(json_file, 'w')
        output_file.write(dump)
        output_file.close()

if __name__ == "__main__":
    # test = rolling_window_tests['test6']['window']
    # df = get_stock_data(test['ticker'],test['start'],test['end'])
    # print('df')
    # print(df)
    # for i in range(0,100,10):
    #     train = df.iloc[i:i+1155]
    #     test = df.iloc[i+1155:i+1155+10]
    #     print(i)
    #     print(train)
    #     print(test)
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 50, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-250-Close-Window', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?

    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=window_heavy_hitters_tests, f='Roondiwala-Std-Win-250-Close-Window-heavy_hitters_tests.json', plot=True)
    test.rolling_window_test()
