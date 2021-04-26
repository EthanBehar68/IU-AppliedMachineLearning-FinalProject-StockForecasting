from test import *
from lstm_roondiwalaetal import *
from lstm_pawaretal import *
from lstm_mogharetal import *
from lstm_behar import *

if __name__ == "__main__":
####################
# Roondiwala Tests #
####################
# Already ran and have results
    # ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    # params = {'lr': 0.001, # learning rate
    #             'loss': 'mean_absolute_percentage_error', # Loss function
    #             'activation': 'tanh', # Not used
    #             'recurrent_activation': 'sigmoid', # Not used
    #             'epochs': 250, #250/500
    #             'batch_size': 150,
    #             'd': 22, # Taken from Roonwidala et al.
    #             'train_columns': ['open', 'close'],
    #             'label_column': 'close', 
    #             'name': 'Roondiwala-Std-250-OpenClose', 
    #             'discretization': False, # Use value rounding?
    #             'fill_method': 'previous', # fillna method='pa'
    #             'normalization': False, # Normalize or standardization?
    #             'window_scaling': False } # Window scaling or bulk scaling?
    
    # test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    # test.rolling_window_test('./true forecasting results/Roondiwala-Std-250-OpenClose/')

    # ['high', 'low', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-250-HighLowClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Roondiwala-Std-250-HighLowClose/')

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-250-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Roondiwala-Std-250-HighLowOpenClose/')

    # ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-500-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Roondiwala-Std-500-OpenClose/')

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-500-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Roondiwala-Std-500-HighLowOpenClose/')

####################
# Pawar Tests #
####################

    # ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Pawar-Std-250-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_PawarEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Pawar-Std-250-OpenClose/')

    # ['high', 'low', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'close'],
                'label_column': 'close', 
                'name': 'Pawar-Std-250-HighLowClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_PawarEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Pawar-Std-250-HighLowClose/')

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Pawar-Std-250-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_PawarEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Pawar-Std-250-HighLowOpenClose/')

    # ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Pawar-Std-500-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_PawarEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Pawar-Std-500-OpenClose/')

    # ['high', 'low', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'close'],
                'label_column': 'close', 
                'name': 'Pawar-Std-500-HighLowClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_PawarEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Pawar-Std-500-HighLowClose/')

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Pawar-Std-500-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_PawarEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Pawar-Std-500-HighLowOpenClose/')

####################
# Moghar Tests #
####################
    # ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Moghar-Std-250-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_MogharEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Moghar-Std-250-OpenClose/')

    # ['high', 'low', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'close'],
                'label_column': 'close', 
                'name': 'Moghar-Std-250-HighLowClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_MogharEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Moghar-Std-250-HighLowClose/')

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Moghar-Std-250-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_MogharEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Moghar-Std-250-HighLowOpenClose/')

    # ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Moghar-Std-500-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_MogharEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Moghar-Std-500-OpenClose/')

    # ['high', 'low', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'close'],
                'label_column': 'close', 
                'name': 'Moghar-Std-500-HighLowClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_MogharEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Moghar-Std-500-HighLowClose/')

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Moghar-Std-500-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_MogharEtAl, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Moghar-Std-500-HighLowOpenClose/')

####################
# Behar Tests #
####################
    # ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Behar-Std-250-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_Behar, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Behar-Std-250-OpenClose/')

    # ['high', 'low', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'close'],
                'label_column': 'close', 
                'name': 'Behar-Std-250-HighLowClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_Behar, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Behar-Std-250-HighLowClose/')

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Behar-Std-250-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_Behar, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Behar-Std-250-HighLowOpenClose/')

    # ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Behar-Std-500-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_Behar, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Behar-Std-500-OpenClose/')

    # ['high', 'low', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'close'],
                'label_column': 'close', 
                'name': 'Behar-Std-500-HighLowClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_Behar, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Behar-Std-500-HighLowClose/')

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Behar-Std-500-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': False } # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_Behar, params=params, tests=window_heavy_hitters_tests, f='', plot=True)
    test.rolling_window_test('./forecasting_rnns/results/Behar-Std-500-HighLowOpenClose/')