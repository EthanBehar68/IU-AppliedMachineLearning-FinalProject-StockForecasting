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
        train_vals = self.train_data['close'].values
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
                       verbose=0)

    def predict(self, test_data):
        test_data = self.train_data[-(self.d-1):].append(test_data)
        test_data = test_data['close'].values

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
        model.add(SimpleRNN(32, return_sequences=True, activation=self.activation))
        model.add(SimpleRNN(32, return_sequences=True, activation=self.activation))
        model.add(SimpleRNN(32, return_sequences=True, activation=self.activation))
        model.add(SimpleRNN(32, activation=self.activation))
        model.add(Dense(1))

        return model


if __name__ == "__main__":
    params = {'lr': 0.001,
              'loss': 'mean_absolute_percentage_error',
              'activation': 'tanh',
              'epochs': 100,
              'batch_size': 150,
              'd': 60,
              'name': 'SimpleRNN'}
    
    print('paper tests')
    test = Test(Model=RNN, params=params, tests=paper_tests, f='rnn-paper-tests.json', plot=True)
    test.run_tests()

    print('own tests')
    test = Test(Model=RNN, params=params, tests=own_tests, f='rnn-own-tests.json', plot=True)
    test.run_tests()

# train_data = get_stock_data("IBM", "2006-01-01", "2016-12-31")['high'].values
# test_data = get_stock_data("IBM", "2017-01-01", "2017-12-31")['high'].values

# train_data = train_data.reshape(-1,1)
# test_data = test_data.reshape(-1,1)

# sc = MinMaxScaler(feature_range=(0,1))
# train_scale = sc.fit_transform(train_data)

# test_scale = sc.transform(test_data)

# X_train = []
# Y_train = []

# for i in range(60, len(train_scale)):
#     X_train.append(train_scale[i-60:i,0])
#     Y_train.append(train_scale[i,0])

# x_train,y_train = np.array(X_train),np.array(Y_train)

# x_train = np.reshape(x_train, (*x_train.shape,1))

# X_test = []
# for i in range(60,len(test_scale)):
#     X_test.append(test_scale[i-60:i,0])

# x_test = np.array(X_test)
# x_test = np.reshape(x_test, (*x_test.shape,1))

# def gen_model(x_train,y_train,x_test,sc):
#     model = Sequential()
#     model.add(SimpleRNN(32, return_sequences=True))
#     model.add(SimpleRNN(32, return_sequences=True))
#     model.add(SimpleRNN(32, return_sequences=True))
#     model.add(SimpleRNN(32))
#     model.add(Dense(1))

#     model.compile(optimizer='rmsprop', loss='mean_squared_error')

#     model.fit(x_train,y_train, epochs=100, batch_size=150)

#     scaled_preds = model.predict(x_test)
#     test_preds = sc.inverse_transform(scaled_preds)

#     return model, test_preds


# model, preds = gen_model(x_train,y_train,x_test,sc)

# diff = len(test_data)-len(preds)
# plt.figure(figsize=(20,5))
# plt.plot(preds,label='prediction')
# plt.plot(test_data[diff:],label='actual')
# plt.title(f'predicting IBM daily highs for 2017 based on 2006-2016 daily highs')
# plt.legend()
# plt.savefig('../imgs/graph.png')
# plt.pause(1)
# plt.close()
