from test import *
from vstack_train_predictor import *
from lstm_rowan import *

params = {'lr': 0.001,
            'loss': 'mean_squared_error', 
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'epochs': 100,
            'batch_size': 32,
            'd': 20,
            'train_columns': ['close'],
            'label_column': 'close', 
            'name': 'Std-Guassian-Smooth-sigma=5', 
            'discretization': False,
            'fill_method': 'previous',
            'normalization': True,
            'window_scaling': False,
            'sigma': 5}

test = Test(Model=LSTM_Rowan, Train_Predictor=Vstack_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
test.rolling_window_test('./forecasting_rnns/results/Rowan-Std-500-HighLowOpenClose-2/')