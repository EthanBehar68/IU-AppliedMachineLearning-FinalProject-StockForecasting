from test import *
from lstm_moghar import *
from X_train_predictor import *


if __name__ == "__main__":
####################
# Moghar Tests #
####################
    # ['open'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001,
                'loss': 'mean_absolute_percentage_error',
                'activation': 'tanh',
                'recurrent_activation': 'sigmoid',
                'epochs': 100,
                'batch_size': 150,
                'd': 22,
                'train_columns': ['open'],
                'label_column': 'open', 
                'name': 'Moghar-Std-100-Open', 
                'discretization': False,
                'fill_method': 'previous',
                'normalization': False }
    
    test = Test(Model=LSTM_Moghar, Train_Predictor=X_Train_Predictor, params=params, tests=window_heavy_hitters_tests,plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Moghar-Std-100-Open/')