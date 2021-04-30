from test import *
from lstm_behar import *
from forecasting_train_predictor import *

if __name__ == "__main__":
    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Discr/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001,
                'loss': 'mean_absolute_percentage_error', 
                'activation': 'tanh',
                'recurrent_activation': 'sigmoid',
                'epochs': 500,
                'batch_size': 150,
                'd': 22,
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Behar-Std-500-HighLowOpenClose',
                'discretization': False,
                'fill_method': 'previous',
                'normalization': False}
    
    test = Test(Model=LSTM_Behar, Train_Predictor=Forecasting_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
    # Make sure this folder is create or MatLibPlot will error out!!
    test.rolling_window_test('./forecasting_rnns/results/Behar-Std-500-HighLowOpenClose/')