from fastquant import get_stock_data
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.metrics import RootMeanSquaredError
from model import Model
from test import *

class LSTM_RoondiwalaEtAl(Model):
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
        self.window_scaling = params['window_scaling']
        self.obs_window_scalers = {}
        self.label_window_scalers = {}

    def train(self, train_data):
        # Normalization/Standization for whole data set
        # Save train data and scaler obj because we will need it for testing
        self.train_obs = train_data[self.train_columns].values
        self.train_label = train_data[self.label_column].values.reshape(-1,1)
        if not self.window_scaling:
            if self.normalization:
                self.obs_scaler = MinMaxScaler(feature_range=(0,1))
                self.label_scaler = MinMaxScaler(feature_range=(0,1))
            else:
                self.obs_scaler = StandardScaler()
                self.label_scaler = StandardScaler()
            x_train_scale = self.obs_scaler.fit_transform(self.train_obs)
            y_train_scale = self.label_scaler.fit_transform(self.train_label)


        # build the x as the observation from (O_i,...,O_i+d)
        # y is O_i+d
        x_train, y_train = [], []
        for i in range(self.d, len(self.train_obs)):
        # Normalization/Standization for whole data set
            if not self.window_scaling:
                x_train.append(x_train_scale[i-self.d:i])
                y_train.append(y_train_scale[i])
            # Window Normalization/Standization for whole data set
            else:
                if self.normalization:
                    obs_scaler = MinMaxScaler(feature_range=(0,1))
                    label_scaler = MinMaxScaler(feature_range=(0,1))
                else:
                    obs_scaler = StandardScaler()
                    label_scaler = StandardScaler()
                x_window = self.train_obs[i-self.d:i]
                x_scale_window = obs_scaler.fit_transform(x_window)
                x_train.append(x_scale_window)
                self.obs_window_scalers[i] = obs_scaler
                y_window = self.train_label[i].reshape(-1,1)
                y_scale_window = label_scaler.fit_transform(y_window)
                y_train.append(y_scale_window)
                self.label_window_scalers[i] = label_scaler

        x_train, y_train = np.array(x_train), np.array(y_train)
        y_train = y_train.reshape(-1, 1)

        # print(x_train.shape)
        # print(y_train.shape)

        # build the model
        self.model = self.gen_model()
        self.model.compile(optimizer=RMSprop(learning_rate=self.lr), loss=self.loss)
        
        # train the model
        return self.model.fit(  x=x_train, 
                                y=y_train, 
                                epochs=self.epochs, 
                                batch_size=self.batch_size,
                                validation_split=0.1,
                                verbose=1)

    def predict(self, test_data):
        # print(self.train_label[-(self.d-1):].shape)
        # print(test_data[self.label_column].values.reshape(-1, 1).shape)
        test_obs = np.concatenate((self.train_obs[-(self.d-1):], test_data[self.train_columns].values), axis=0)
        test_label =  np.concatenate((self.train_label[-(self.d-1):], test_data[self.label_column].values.reshape(-1, 1)), axis=0)

        # Normalization/Standization for whole data set
        if not self.window_scaling:
            test_scale_obs = self.obs_scaler.transform(test_obs)

        # Build observations like in training
        x_test, labels = [], []
        for i in range(self.d, len(test_data)):
            # Normalization/Standization for whole data set
            if not self.window_scaling:
                x_test.append(test_scale_obs[i-self.d:i])
                labels.append(test_label[i])
            # Window Normalization/Standization for whole data set
            else:
                obs_scaler = self.obs_window_scalers[i]
                test_window_obs = test_obs[i-self.d:i]
                scale_window_obs = obs_scaler.fit_transform(test_window_obs)
                x_test.append(scale_window_obs)
                labels.append(test_label[i])

        x_test, labels = np.array(x_test), np.array(labels)

        # predict the points
        scaled_preds = self.model.predict(x_test)

        # Inverse data set
        preds = []
        if not self.window_scaling:
            preds = self.label_scaler.inverse_transform(scaled_preds)
        else: 
            predictionIndex = 0
            for i in range(self.d, len(test_data)):
                # Window Inverse data set
                label_scaler = self.label_window_scalers[i]
                prediction = label_scaler.inverse_transform(scaled_preds[predictionIndex])
                preds.append(prediction)
                predictionIndex += 1
            preds = np.array(preds)

        # print(preds.shape)
        # print(labels.shape)

        return preds, labels

    # Saving just in case we come back to this    
    # def get_data(self, ticker, start_date, end_date, data_columns=['close']):
    #     # get tickers' from https://www.quandl.com/data/
    #     # Drop unnecessary columns, rename to volume
    #     quandl.ApiConfig.api_key = 'NzYdeTcwJ539XMzzwZNS'
    #     return self.preprocess_data(
    #         quandl.get(ticker, 
    #             start_date=start_date, 
    #             end_date=end_date)
    #             .drop(columns=['Turnover (Rs. Cr)'], axis=1)
    #             .rename(columns={'Shares Traded': 'volume', 
    #                             'Close': 'close', 
    #                             'Open': 'open', 
    #                             'Low': 'low', 
    #                             'High': 'high'}))
    
    def get_data(self, ticker, start_date, end_date):
        return self.preprocess_data(get_stock_data(ticker, start_date, end_date))

    def preprocess_data(self, data):
        if self.discretization:
            data = data.round(0)

        if self.fill_method == 'previous':
            data = data.fillna(method='pad')
        
        return data

    # Faithfully recreating Roondiwala as close as possible
    def gen_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.d, len(self.train_columns)), return_sequences=True))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(16, init='uniform', activation='relu'))
        model.add(Dense(1, init='uniform', activation='linear'))
        return model


if __name__ == "__main__":
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Round/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 10, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-Win-250-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': True} # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=heavy_hitters_tests, f='Roondiwala-Std-Win-250-OpenClose-Full-heavy_hitters_tests.json', plot=True)
    test.fixed_origin_tests()