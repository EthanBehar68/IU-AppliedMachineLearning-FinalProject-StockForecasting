from fastquant import get_stock_data
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from model import Model
from test import *
from scipy.ndimage.filters import gaussian_filter
from scipy import fft

class LSTM_Rowan(Model):
    def __init__(self, params):
        super().__init__(params)
        self.lr = params['lr']
        self.loss = params['loss']
        self.activation = params['activation']
        self.recurrent_activation = params['recurrent_activation']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.d = params['d']
        self.train_columns = params['train_columns']
        self.label_column = params['label_column']
        self.discretization = params['discretization']
        self.fill_method = params['fill_method']
        self.normalization = params['normalization']
        self.scaler = None
        self.window_scaling = params['window_scaling']
        self.window_scalers = {}

    # If label column is not part of x train
    # This needs to update
    def train(self, train_data, rolling_window_test=False):
        # Save train data and scaler obj because we will need it for testing
        self.train_obs = train_data['close'].values
        #plt.plot(range(0,len(self.train_obs)), self.train_obs)
        #plt.show()

        # gaussian smoothing kernel
        self.train_obs = gaussian_filter(self.train_obs, sigma=10)
        #plt.plot(range(0,len(self.train_obs)), self.train_obs)
        #plt.show()

        self.train_obs = self.train_obs.reshape(-1,1)
        
        # standardize data
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaler = self.scaler.fit(self.train_obs)
        self.train_obs = self.scaler.transform(self.train_obs)
        # self.train_obs = np.log(self.train_obs)

        # Build the x as the observation from (O_i,...,O_i+d), y is O_i+d
        x_train, y_train = [], []
        for i in range(self.d, len(self.train_obs)):
            x_train.append(self.train_obs[i-self.d:i])
            y_train.append(self.train_obs[i])

        x_train, y_train = np.array(x_train), np.array(y_train)
        y_train = y_train.reshape(-1, 1)
        print('x_train shape before training: ', x_train.shape)
        print('y_train shape before training: ', y_train.shape)

        # build the model
        self.model = self.gen_model()
        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss)
        
        # train the model
        return self.model.fit(x=x_train, 
                              y=y_train, 
                              epochs=self.epochs, 
                              batch_size=self.batch_size,
                              validation_split=0.1,
                              verbose=1)

    def predict(self, test_data, rolling_window_test=False):
        test_close_prices = test_data['close'].values

        # Save train data and scaler obj because we will need it for testing
        test_obs = test_data['close'].values

        # gaussian smoothing kernel
        test_obs = gaussian_filter(test_obs, sigma=10)

        # standardize data
        test_obs = self.scaler.transform(test_obs.reshape(-1,1))
        # test_obs = np.log(test_obs)

        # Add self.d amount of days in front of test data so test_data[0] can be first prediction point
        observed = self.train_obs[-self.d:]

        preds = []

        for i in range(len(test_data)):
            pred_std_close = self.model.predict(observed.reshape(1,self.d,1))
            observed = np.vstack((observed,test_obs[i]))
            observed = observed[1:]

            pred_close = self.scaler.inverse_transform(pred_std_close)
            preds.append(pred_close.reshape(1,))
            
            print(f'{i+1}/{len(test_data)}', end='\r', flush=True)

        return np.array(preds).flatten(), test_close_prices

    def gen_model(self):
        model = Sequential()
        model.add(
            LSTM(128, 
                 input_shape=(self.d, len(self.train_columns)),
                 bias_initializer='random_normal',
                 recurrent_dropout=0.1,
                 return_sequences=True
                 )
            )
        model.add(
           LSTM(64, 
                input_shape=(self.d, len(self.train_columns)),
                bias_initializer='random_normal',
                recurrent_dropout=0.1,
                )
           )
        model.add(Dropout(0.1))
        model.add(Dense(16, activation='sigmoid'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='linear'))
        return model

if __name__ == "__main__":

    # Use the tester files for running tests
    # This should be used only to make sure its working.
    
    # ['close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Round/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001,
              'loss': 'mean_squared_error', 
              'activation': 'tanh',
              'recurrent_activation': 'sigmoid',
              'epochs': 100,
              'batch_size': 32,
              'd': 20,
              'train_columns': ['close'],
              'label_column': 'close', 
              'name': 'Rowan-Std-500-HighLowOpenClose', 
              'discretization': False,
              'fill_method': 'previous',
              'normalization': True,
              'window_scaling': False }
    
    test = Test(Model=LSTM_Rowan, params=params, tests=own_tests, f='rowan-forecast-lstm.json', plot=True)
    #test.rolling_window_test('./forecasting_rnns/results/Rowan-Std-500-HighLowOpenClose/')
    test.fixed_origin_tests()