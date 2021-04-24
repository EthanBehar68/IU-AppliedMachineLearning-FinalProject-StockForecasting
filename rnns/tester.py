from test import *
from lstm_roondiwalaetal import *
# File Structure
# Region Roondiwala Tests


if __name__ == "__main__":
    # Region Roondiwala Tests
    #  ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Round/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-Win-250-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': True} # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=heavy_hitters_tests, f='Roondiwala-Std-Win-250-OpenClose-Rolling-heavy_hitters_tests.json', plot=True)
    test.rolling_window_test()

    # ['high', 'low', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Round/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-Win-250-HighLowClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': True} # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=heavy_hitters_tests, f='Roondiwala-Std-Win-250-HighLowClose-Rolling-heavy_hitters_tests.json', plot=True)
    test.rolling_window_test()

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Round/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 250, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-Win-250-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': True} # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=heavy_hitters_tests, f='Roondiwala-Std-Win-250-HighLowOpenClose-Rolling-heavy_hitters_tests.json', plot=True)
    test.rolling_window_test()

        #  ['open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Round/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-Win-250-OpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': True} # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=heavy_hitters_tests, f='Roondiwala-Std-Win-500-OpenClose-Rolling-heavy_hitters_tests.json', plot=True)
    test.rolling_window_test()

    # ['high', 'low', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Round/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-Win-250-HighLowClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': True} # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=heavy_hitters_tests, f='Roondiwala-Std-Win-500-HighLowClose-Rolling-heavy_hitters_tests.json', plot=True)
    test.rolling_window_test()

    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Round/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001, # learning rate
                'loss': 'mean_absolute_percentage_error', # Loss function
                'activation': 'tanh', # Not used
                'recurrent_activation': 'sigmoid', # Not used
                'epochs': 500, #250/500
                'batch_size': 150,
                'd': 22, # Taken from Roonwidala et al.
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-Win-250-HighLowOpenClose', 
                'discretization': False, # Use value rounding?
                'fill_method': 'previous', # fillna method='pa'
                'normalization': False, # Normalize or standardization?
                'window_scaling': True} # Window scaling or bulk scaling?
    
    test = Test(Model=LSTM_RoondiwalaEtAl, params=params, tests=heavy_hitters_tests, f='Roondiwala-Std-Win-500-HighLowOpenClose-Rolling-heavy_hitters_tests.json', plot=True)
    test.rolling_window_test()
    # Endegion Roondiwala Tests