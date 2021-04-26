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
        self.model.compile(optimizer=RMSprop(learning_rate=self.lr), loss=self.loss)
        
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
        x_test,actual = [],[]
        for i in range(self.d, len(test_scale)):
            x_test.append(test_scale[i-self.d:i,0])
            actual.append(test_data[i,0])

        x_test,actual = np.array(x_test),np.array(actual)
        x_test = np.reshape(x_test, (*x_test.shape,1))

        # predict the points
        preds = self.model.predict(x_test)
        preds = self.scaler.inverse_transform(preds)

        return preds, actual
    
    def gen_model(self):
        model = Sequential()
        model.add(SimpleRNN(50, return_sequences=True, activation=self.activation))
        model.add(SimpleRNN(50, return_sequences=True, activation=self.activation))
        model.add(SimpleRNN(50, return_sequences=True, activation=self.activation))
        model.add(SimpleRNN(50, activation=self.activation))
        model.add(Dense(1))

        return model


if __name__ == "__main__":
    params = {'lr': 0.001,
              'loss': 'mean_absolute_percentage_error',
              'activation': 'sigmoid',
              'epochs': 200,
              'batch_size': 150,
              'd': 45,
              'name': 'SimpleRNN'}
    
    print('tests')
    test = Test(Model=RNN, params=params, tests=rolling_window_tests, f='rnn-forecast-tests.json', plot=True)
    test.run_tests()