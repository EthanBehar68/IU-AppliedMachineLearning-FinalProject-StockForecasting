from fastquant import get_stock_data
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.metrics import RootMeanSquaredError
from model import Model
from test import *

class LSTM_Roonetal(Model):
    def __init__(self, params):
        super().__init__(params)
        self.lr = params['lr']
        self.loss = params['loss']
        self.activation = params['activation']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.d = params['d']
        self.discretization = params['discretization']
        self.fill_method = params['fill_method']

    def train(self, train_data):
        # save train data and scaler obj because we will need it for testing
        self.train_data = train_data
        self.scaler = MinMaxScaler(feature_range=(0,1))

        # pull out the close values and scale them
        train_vals = self.train_data[['close']].values
        train_scale = self.scaler.fit_transform(train_vals)

        # build the x as the observation from (O_i,...,O_i+d)
        # y is O_i+d
        x_train, y_train = [],[]
        for i in range(self.d, len(train_scale)):
            x_train.append(train_scale[i-self.d:i,0])
            y_train.append(train_scale[i,0])
        
        x_train,y_train = np.array(x_train),np.array(y_train)
        x_train = np.reshape(x_train, (*x_train.shape,1))

        # build the model
        self.model = self.gen_model()
        self.model.compile(
            optimizer=RMSprop(learning_rate=self.lr),
            loss='mse',
            metrics=[RootMeanSquaredError()])
        
        # train the model
        self.model.fit(x=x_train, 
                       y=y_train, 
                       epochs=self.epochs, 
                       batch_size=self.batch_size,
                       verbose=1)

    def predict(self, test_data):
        test_data = self.train_data[-(self.d-1):].append(test_data)
        test_data = test_data[['close']].values

        # scale the test data
        test_scale = self.scaler.transform(test_data)

        # build observations like in training
        x_test, actual = [],[]
        for i in range(self.d, len(test_scale)):
            x_test.append(test_scale[i-self.d:i,0])
            # This paper leaves data normalized so we shall too
            # So we need y actual to be normalized.
            # actual.append(test_data[i,0])
            actual.append(test_scale[i,0])

        x_test, actual = np.array(x_test), np.array(actual)
        x_test = np.reshape(x_test, (*x_test.shape,1))

        # predict the points
        preds = self.model.predict(x_test)
        # This paper leaves data normalized so we shall too
        # preds = self.scaler.inverse_transform(preds)

        return preds, actual
    
    def get_data(self, ticker, start_date, end_date):
        # get tickers' from https://www.quandl.com/data/
        # Drop unnecessary columns, rename to volume
        quandl.ApiConfig.api_key = 'NzYdeTcwJ539XMzzwZNS'
        return self.preprocess_data(
            quandl.get(ticker, 
                start_date=start_date, 
                end_date=end_date)
                .drop(columns=['Turnover (Rs. Cr)'], axis=1)
                .rename(columns={'Shares Traded': 'volume', 
                                'Close': 'close', 
                                'Open': 'open', 
                                'Low': 'low', 
                                'High': 'high'}))
    
    def preprocess_data(self, data):
        if self.discretization:
            data = data.round(0)

        if self.fill_method == 'previous':
            data = data.fillna(method='pad')
        
        return data

    # Don't use the activation param - it's a bit limited in that it only uses one 
    # This paper used 3 different activation functions (tanh is default)
    def gen_model(self):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, bias_initializer='glorot_uniform'))
        model.add(LSTM(64, return_sequences=False, bias_initializer='glorot_uniform'))
        model.add(Dense(16, init='uniform', bias_initializer='glorot_uniform'))
        model.add(Dense(1, init='uniform', bias_initializer='glorot_uniform'))
        return model


if __name__ == "__main__":
    params = {'lr': 0.001,
                'loss': 'root_mean_squared_error', # Match paper, this line in compile method metrics=[tf.keras.metrics.RootMeanSquaredError()] makes is RMSE instead of MSE
                'activation': 'tanh',
                'epochs': 250, # Paper uses 250/500
                'batch_size': 150, # Paper doesn't specify batch sizes
                'd': 22,  # Match paper
                'name': 'LSTM_Roonetal',
                'discretization': True,
                'fill_method': 'previous'}
    
    print('Roonetal Tests with rounding')
    test = Test(Model=LSTM_Roonetal, params=params, tests=roonetal_tests, f='LSTM_Roonetal-forecast-tests.json', plot=True)
    test.fixed_origin_tests()

    params = {'lr': 0.001,
                'loss': 'root_mean_squared_error', # Match paper, this line in compile method metrics=[tf.keras.metrics.RootMeanSquaredError()] makes is RMSE instead of MSE
                'activation': 'tanh',
                'epochs': 250, # Paper uses 250/500
                'batch_size': 150, # Paper doesn't specify batch sizes
                'd': 22,  # Match paper
                'name': 'LSTM_Roonetal_NotRounded',
                'discretization': False,
                'fill_method': 'previous'}

    print('Roonetal Tests with no rounding')
    test = Test(Model=LSTM_Roonetal, params=params, tests=roonetal_tests, f='LSTM_Roonetal-NotRounded-forecast-tests.json', plot=True)
    test.fixed_origin_tests()