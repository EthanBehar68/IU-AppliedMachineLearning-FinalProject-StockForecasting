from test import *
from forecasting_train_predictor import *
from overfit_train_predictor import *
from lstm_roondiwala import *
from lstm_pawar import *
from lstm_moghar import *
from lstm_behar import *

if __name__ == "__main__":
####################
# Roondiwala Tests #
####################
    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Discr/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001,
                'loss': 'mean_absolute_percentage_error',
                'activation': 'tanh',
                'recurrent_activation': 'sigmoid',
                'epochs': 1,
                'batch_size': 150,
                'd': 22,
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-500-HighLowOpenClose-Rolling',
                'discretization': False,
                'fill_method': 'previous',
                'normalization': False }
    
    test = Test(Model=LSTM_Roondiwala, Train_Predictor=Forecasting_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
    # Make sure this folder is create or MatLibPlot will error out!!
    test.rolling_window_test('./forecasting_rnns/results/Roondiwala-Std-500-HighLowOpenClose/')

####################
# Pawar Tests #
####################
    # ['close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Discr/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001,
                'loss': 'mean_absolute_percentage_error',
                'activation': 'tanh',
                'recurrent_activation': 'sigmoid',
                'epochs': 1,
                'batch_size': 150,
                'd': 22,
                'train_columns': ['close'],
                'label_column': 'close', 
                'name': 'Pawar-Std-100-Close-Rolling',
                'discretization': False,
                'fill_method': 'previous',
                'normalization': False }
    
    test = Test(Model=LSTM_Pawar, Train_Predictor=Forecasting_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
    # Make sure this folder is create or MatLibPlot will error out!!
    test.rolling_window_test('./forecasting_rnns/results/Pawar-Std-100-Close/')

####################
# Moghar Tests #
####################
    # ['open'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Discr/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001,
                'loss': 'mean_absolute_percentage_error',
                'activation': 'tanh',
                'recurrent_activation': 'sigmoid',
                'epochs': 1,
                'batch_size': 150,
                'd': 22,
                'train_columns': ['open'],
                'label_column': 'open', 
                'name': 'Moghar-Std-100-Open-Rolling',
                'discretization': False,
                'fill_method': 'previous',
                'normalization': False }
    
    test = Test(Model=LSTM_Moghar, Train_Predictor=Forecasting_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
    # Make sure this folder is create or MatLibPlot will error out!!
    test.rolling_window_test('./forecasting_rnns/results/Moghar-Std-100-Open/')

####################
# Behar Tests #
####################
    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Discr/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001,
                'loss': 'mean_absolute_percentage_error', 
                'activation': 'tanh',
                'recurrent_activation': 'sigmoid',
                'epochs': 1,
                'batch_size': 150,
                'd': 22,
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Behar-Std-500-HighLowOpenClose-Rolling',
                'discretization': False,
                'fill_method': 'previous',
                'normalization': False}
    
    test = Test(Model=LSTM_Behar, Train_Predictor=Forecasting_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
    # Make sure this folder is create or MatLibPlot will error out!!
    test.rolling_window_test('./forecasting_rnns/results/Behar-Std-500-HighLowOpenClose/')