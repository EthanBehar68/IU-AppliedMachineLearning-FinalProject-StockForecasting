from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout
from keras.optimizers import Adam
from model import Model
from test import *

class LSTMModel(Model):
    def __init__(self, params):
        super().__init__(params)
        self.lr = params['lr']
        self.loss = params['loss']
        self.activation = params['activation']
        self.recurrent_activation = params['recurrent_activation']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.scale_type = params['scale_type']
        self.d = params['d']

    def train(self, train_data):
        # save train data and scaler obj because we will need it for testing
        self.train_obs = self.data_prep(train_data).values

        if self.scale_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0,1))
        else:
            self.scaler = StandardScaler()

        self.scaler = self.scaler.fit(self.train_obs)
        self.train_obs = self.scaler.transform(self.train_obs)

        self.train_labels = self.data_prep(train_data)['fracChange'].values.reshape(-1,1)
        
        if self.scale_type == 'minmax':
            self.scaler_out = MinMaxScaler(feature_range=(0,1))
        else:
            self.scaler_out = StandardScaler()

        self.scaler_out = self.scaler_out.fit(self.train_labels)
        self.train_labels = self.scaler_out.transform(self.train_labels)
        
        # build the x as the observation from (O_i,...,O_i+d)
        # y is O_i+d
        x_train, y_train = [],[]
        for i in range(self.d, len(self.train_obs)):
            x_train.append(self.train_obs[i-self.d:i])
            y_train.append(self.train_labels[i])
        
        x_train,y_train = np.array(x_train), np.array(y_train)
        y_train = np.reshape(y_train, (*y_train.shape, 1))

        # build the model
        self.model = self.gen_model()
        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss)
        
        # train the model
        self.model.fit(x=x_train, 
                       y=y_train, 
                       epochs=self.epochs, 
                       batch_size=self.batch_size,
                       verbose=1)

    def predict(self, test_data):
        # scale the test observations
        test_close_prices = test_data['close'].values
        test_open_prices = test_data['open'].values
        test_obs = self.scaler.transform(self.data_prep(test_data).values)

        # create first set of observations to be passed to the model
        observed = self.train_obs[-self.d:]

        preds = []

        for i in range(len(test_data)):
            # predict and stack the next test point onto the observed list
            pred_frac_change = self.model.predict(observed.reshape(1,self.d,3))
            observed = np.vstack((observed,test_obs[i]))
            observed = observed[1:]
            
            # inverse transform the prediction and solve for close price
            pred_frac_change = self.scaler_out.inverse_transform(pred_frac_change)
            pred_close = pred_frac_change*test_open_prices[i]+test_open_prices[i]
            preds.append(pred_close.reshape(1,))
            
            print(f'{i+1}/{len(test_data)}', end='\r', flush=True)

        return np.array(preds).flatten(), test_close_prices
    
    def data_prep(self, data):
        df = pd.DataFrame(data=None, columns=['fracChange','fracHigh','fracLow'])
        df['fracChange'] = (data['close']-data['open'])/data['open']
        df['fracHigh'] = (data['high']-data['open'])/data['open']
        df['fracLow'] = (data['open']-data['low'])/data['open']

        return df

    # define model structure
    def gen_model(self):
        model = Sequential()
        model.add(LSTM(512,input_shape=(self.d, 3), return_sequences=True,activation=self.activation,recurrent_activation=self.recurrent_activation))
        model.add(Dropout(0.1))
        model.add(LSTM(512,activation=self.activation,recurrent_activation=self.recurrent_activation))
        model.add(Dropout(0.1))
        model.add(Dense(16,activation='sigmoid'))
        model.add(Dense(1))

        return model


if __name__ == "__main__":
    params = {'lr': 0.001,
              'loss': 'mean_squared_error',
              'activation': 'tanh',
              'recurrent_activation': 'sigmoid',
              'epochs': 50,
              'batch_size': 75,
              'd': 20,
              'scale_type': 'minmax',
              'name': 'LSTM-adv-std'}
    
    print('paper tests')
    test = Test(Model=LSTMModel, params=params, tests=paper_tests, f='lstm-std-adv-paper-tests.json', plot=True)
    test.fixed_origin_tests()

    print('own tests')
    test = Test(Model=LSTMModel, params=params, tests=own_tests, f='lstm-adv-own-tests.json', plot=True)
    test.fixed_origin_tests()

    print('testing')
    test = Test(Model=LSTMModel, params=params, tests=rolling_window_tests, f='lstm-std-adv-rolling-tests.json', plot=True)
    test.rolling_window_test()

