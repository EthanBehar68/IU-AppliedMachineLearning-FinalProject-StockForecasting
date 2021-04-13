from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
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

        # build the x as the observation from (O_i,...,O_i+d)
        # y is O_i+d
        x_train, y_train = [],[]
        for i in range(self.d, len(self.train_obs)):
            x_train.append(self.train_obs[i-self.d:i,0])
            y_train.append(self.train_obs[i,0])
        
        x_train,y_train = np.array(x_train),np.array(y_train)
        x_train = np.reshape(x_train, (*x_train.shape,1))

        # build the model
        self.model = self.gen_model()
        self.model.compile(optimizer=RMSprop(learning_rate=self.lr), loss=self.loss)
        
        # train the model
        self.model.fit(x=x_train, 
                       y=y_train, 
                       epochs=self.epochs, 
                       batch_size=self.batch_size,
                       verbose=0)

    def predict(self, test_data):
        # test_data = self.train_data[-(self.d-1):].append(test_data)
        # test_data = test_data[['close']].values

        # # scale the test data
        # test_scale = self.scaler.transform(test_data)

        # # build observations like in training
        # x_test,actual = [],[]
        # for i in range(self.d, len(test_scale)):
        #     x_test.append(test_scale[i-self.d:i,0])
        #     actual.append(test_data[i,0])

        # x_test,actual = np.array(x_test),np.array(actual)
        # x_test = np.reshape(x_test, (*x_test.shape,1))

        # # predict the points
        # preds = self.model.predict(x_test)
        # preds = self.scaler.inverse_transform(preds)

        test_close_prices = test_data['close'].values
        test_open_prices = test_data['open'].values

        observed = self.train_obs[-self.d:]

        preds = []

        for i in range(len(test_data)):
            pred_frac_change = self.model.predict(observed)

            observed = np.vstack((observed,pred_frac_change))
            observed = observed[1:]

            pred_close = pred_frac_change*test_open_prices[i]+test_open_prices[i]
            preds.append(pred_close)

        return preds, test_close_prices
    
    def data_prep(self, data):
        df = pd.DataFrame(data=None, columns=['fracChange','fracHigh','fracLow'])
        df['fracChange'] = (data['close']-data['open'])/data['open']
        df['fracHigh'] = (data['high']-data['open'])/data['open']
        df['fracLow'] = (data['open']-data['low'])/data['open']

        return df

    def gen_model(self):
        model = Sequential()
        model.add(SimpleRNN(32, return_sequences=True, activation=self.activation))
        model.add(SimpleRNN(32, return_sequences=True, activation=self.activation))
        model.add(SimpleRNN(32, return_sequences=True, activation=self.activation))
        model.add(SimpleRNN(32, activation=self.activation))
        model.add(Dense(1))

        return model


if __name__ == "__main__":
    params = {'lr': 0.001,
              'loss': 'mean_absolute_percentage_error',
              'activation': 'sigmoid',
              'epochs': 100,
              'batch_size': 1,
              'd': 5,
              'name': 'SimpleRNN'}
    
    print('paper tests')
    test = Test(Model=RNN, params=params, tests=paper_tests, f='rnn-paper-tests.json', plot=True)
    test.run_tests()

    print('own tests')
    test = Test(Model=RNN, params=params, tests=own_tests, f='rnn-own-tests.json', plot=True)
    test.run_tests()