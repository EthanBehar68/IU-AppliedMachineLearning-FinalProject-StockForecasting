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

class LSTM_PawarEtAl(Model):
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
        # B/c rolling_window_test doesn't use GetData function
        if rolling_window_test:
            train_data = self.preprocess_data(train_data)

        # Save train data and scaler obj because we will need it for testing
        self.train_obs = train_data.values

        # Normalization/Standization for whole data set
        if not self.window_scaling:
            if self.normalization:
                self.scaler = MinMaxScaler(feature_range=(0,1))
            else:
                self.scaler = StandardScaler()
            x_train_scale = self.scaler.fit_transform(self.train_obs)
            y_train_scale = x_train_scale[:, self.label_column_index]        

        # Build the x as the observation from (O_i,...,O_i+d), y is O_i+d
        x_train, y_train = [], []
        for i in range(self.d, len(self.train_obs)):
        # Normalization/Standization for whole data set
            if not self.window_scaling:
                x_train.append(x_train_scale[i-self.d:i])
                y_train.append(y_train_scale[i])
            # Window Normalization/Standization for whole data set
            # else:
            #     if self.normalization:
            #         scaler = MinMaxScaler(feature_range=(0,1))
            #     else:
            #         scaler = StandardScaler()
            #     x_train_window = self.train_obs[i-self.d:i]
            #     x_scale_window = scaler.fit_transform(x_train_window)
            #     x_train.append(x_scale_window)
            #     y_scale_window = x_scale_window[:, self.label_column_index]
            #     y_train.append(y_scale_window[-1])
            #     self.window_scalers[i] = scaler
            #     self.abc.append(i)

        x_train, y_train = np.array(x_train), np.array(y_train)
        y_train = y_train.reshape(-1, 1)
        print('x_train shape before training: ', x_train.shape)
        print('y_train shape before training: ', y_train.shape)

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

    def predict(self, test_data, rolling_window_test=False):
        # B/c rolling_window_test doesn't use GetData function
        if rolling_window_test:
            test_data = self.preprocess_data(test_data)

        # Add self.d amount of days in front of test data so test_data[0] can be first prediction point
        test_obs = np.concatenate((self.train_obs[-(self.d-1):], test_data.values), axis=0)

        # Normalization/Standization for whole data set
        if not self.window_scaling:
            test_scale_obs = self.scaler.transform(test_obs)
            test_scale_label = test_scale_obs[:, self.label_column_index]       

        # Build observations like in training
        x_test, labels = [], []
        for i in range(self.d, len(test_data)):
            # Normalization/Standization for whole data set
            if not self.window_scaling:
                x_test.append(test_scale_obs[i-self.d:i])
                labels.append(test_scale_label[i])
            # Window Normalization/Standization for whole data set
            # else:
            #     # Get the window scaler
            #     scaler = self.window_scalers[i]
            #     # Scale the test data
            #     test_window_obs = test_obs[i-self.d:i]
            #     scale_windows_obs = scaler.transform(test_window_obs)
            #     print(scale_windows_obs)
            #     x_test.append(scale_windows_obs)
            #     # Get the scaled label
            #     scale_label_window = scale_windows_obs[:, self.label_column_index]
            #     labels.append(scale_label_window[-1])
            #     self.xyz.append(i)

        x_test, labels = np.array(x_test), np.array(labels)
        labels = labels.reshape(-1, 1)

        print('x_test shape before prediction: ' , x_test.shape)
        # predict the points
        preds = self.model.predict(x_test)
        print('preds: ' , preds.shape)
        print('labels: ', labels.shape)
        return preds, labels

    def get_data(self, ticker, start_date, end_date):
        return self.preprocess_data(get_stock_data(ticker, start_date, end_date))

    def preprocess_data(self, data):
        if self.discretization:
            data = data.round(0)

        if self.fill_method == 'previous':
            data = data.fillna(method='pad')

        # This breaks if label column is not a training column
        data_columns = data.columns
        # print('data_columns: ', data_columns)
        # print('train_columns: ', self.train_columns)
        [data.drop(c, axis=1, inplace=True) for c in data_columns if c not in self.train_columns]
        # print('data_columns: ', data.columns)
        self.label_column_index = data.columns.get_loc(self.label_column)

        return data

    # Faithfully recreating Roondiwala as close as possible
    def gen_model(self):
        model = Sequential()
        model.add(LSTM(512, input_shape=(self.d, len(self.train_columns)), return_sequences=True))
        model.add(LSTM(512, return_sequences=False))
        model.add(Dense(1))
        return model


if __name__ == "__main__":
    
    # Use the tester files for running tests
    # This should be used only to make sure its working.
    
    # ['close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Round/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 50, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['close'],
                'label_column': 'close', 
                'name': 'Pawar-Std-250-Close-Fixed', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': True } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_PawarEtAl, params=params, tests=heavy_hitters_tests, f='Pawar-Std-250-Close-Fixed-heavy_hitters_tests.json', plot=True)
    test.fixed_origin_tests()