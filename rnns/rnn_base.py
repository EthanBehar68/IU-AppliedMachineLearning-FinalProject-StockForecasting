from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from keras.optimizers import RMSprop
from model import Model
from test import *

class RNN(Model):
    def __init__(self,params):
        super().__init__(params)
        self.lr = params['lr']
        self.loss = params['loss']
        self.activation = params['activation']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.d = params['d']


    def train(self, train_data):
        # save train data and scaler obj because we will need it for testing
        self.train_obs = self.data_prep(train_data).values
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaler = self.scaler.fit(self.train_obs)
        self.train_obs = self.scaler.transform(self.train_obs)

        # build the x as the observation from (O_i,...,O_i+d)
        # y is O_i+d
        x_train, y_train = [],[]
        for i in range(self.d, len(self.train_obs)):
            x_train.append(self.train_obs[i-self.d:i,0])
            y_train.append(self.train_obs[i,0])
        
        x_train,y_train = np.array(x_train),np.array(y_train)
        x_train = np.reshape(x_train, (*x_train.shape,1))
        print(x_train.shape)

        # build the model
        self.model = self.gen_model()
        self.model.compile(optimizer=RMSprop(learning_rate=self.lr), loss=self.loss)
        
        # train the model
        self.model.fit(x=x_train, 
                       y=y_train, 
                       epochs=self.epochs, 
                       batch_size=self.batch_size,
                       verbose=1)

    def predict(self, test_data):
        test_close_prices = test_data['close'].values
        test_open_prices = test_data['open'].values

        observed = self.train_obs[-self.d:]

        preds = []
        print('starting predictions')

        for i in range(len(test_data)):
            pred_frac_change = self.model.predict(observed.reshape(1,self.d,1))
            observed = np.vstack((observed,pred_frac_change))
            observed = observed[1:]

            pred_frac_change = self.scaler.inverse_transform(pred_frac_change)
            pred_close = pred_frac_change*test_open_prices[i]+test_open_prices[i]
            preds.append(pred_close.reshape(1,))
            
            print(f'{i+1}/{len(test_data)}',end='\r',flush=True)

        print(preds)
        return preds, test_close_prices
    
    def data_prep(self, data):
        df = pd.DataFrame(data=None, columns=['fracChange'])
        df['fracChange'] = (data['close']-data['open'])/data['open']

        return df

    def gen_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))

        return model


if __name__ == "__main__":
    params = {'lr': 0.01,
              'loss': 'mean_squared_error',
              'activation': 'tanh',
              'epochs': 300,
              'batch_size': 150,
              'd': 10,
              'name': 'RNN-LSTM'}
    
    print('paper tests')
    test = Test(Model=RNN, params=params, tests=paper_tests, f='rnn-lstm-paper-tests.json', plot=True)
    test.run_tests()

    print('own tests')
    test = Test(Model=RNN, params=params, tests=own_tests, f='rnn-lstm-own-tests.json', plot=True)
    test.run_tests()