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
from scipy.ndimage.filters import gaussian_filter
from test import *
from vstack_train_predictor import *
from base_model import *

class LSTM_Rowan(Base_Model):
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
        self.label_column_index = None
        self.sigma = params['sigma']

    def preprocess_data(self, train_data):
        return train_data

    def gen_model(self):
        model = Sequential()
        model.add(
            LSTM(128, 
                 input_shape=(self.d, len(self.train_columns)),
                 recurrent_dropout=0.1,
                 return_sequences=True
                 )
            )
        model.add(
           LSTM(64, 
                input_shape=(self.d, len(self.train_columns)),
                recurrent_dropout=0.1,
                )
           )
        model.add(Dropout(0.1))
        model.add(Dense(16, activation='sigmoid'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='linear'))
        return model

if __name__ == "__main__":
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
              'name': 'Rowan-MinMax-Guassian-Smooth-sigma=5', 
              'discretization': False,
              'fill_method': 'previous',
              'normalization': True,
              'window_scaling': False,
              'sigma': 10}
    
    test = Test(Model=LSTM_Rowan, Train_Predictor=Vstack_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Rowan-Std-500-HighLowOpenClose-2/')