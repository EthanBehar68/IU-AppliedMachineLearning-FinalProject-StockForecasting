from test import *
from lstm_roondiwala import *
from X_train_predictor import *

if __name__ == "__main__":
####################
# Roondiwala Tests #
####################
    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001,
                'loss': 'mean_absolute_percentage_error',
                'activation': 'tanh',
                'recurrent_activation': 'sigmoid',
                'epochs': 500,
                'batch_size': 150,
                'd': 22,
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-500-HighLowOpenClose', 
                'discretization': False,
                'fill_method': 'previous',
                'normalization': False }
    
    test = Test(Model=LSTM_Roondiwala, Train_Predictor=X_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Roondiwala-Std-500-HighLowOpenClose/')
