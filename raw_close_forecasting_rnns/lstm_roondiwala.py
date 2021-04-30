# Base Class
from base_model import *
# RNN 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.metrics import RootMeanSquaredError
# Testability
from test import *
from forecasting_train_predictor import *

class LSTM_Roondiwala(Base_Model):
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

    # Return data from fastquant
    def get_data(self, ticker, start_date, end_date):
        return self.preprocess_data(get_stock_data(ticker, start_date, end_date))

    def preprocess_data(self, data):
        # Round data to nearest dollar
        if self.discretization:
            data = data.round(0)

        # Fill NA's with last previous valid row's data
        if self.fill_method == 'previous':
            data = data.fillna(method='pad')

        # Remove unused columns
        # WARN This breaks if label column is not a training column
        data_columns = data.columns
        [data.drop(c, axis=1, inplace=True) for c in data_columns if c not in self.train_columns]
        self.label_column_index = data.columns.get_loc(self.label_column)

        return data


    # Faithfully recreating Roondiwala Et Al as close as possible
    def gen_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.d, len(self.train_columns)), return_sequences=True))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        return model


if __name__ == "__main__":
    # Use the tester files for running tests
    # This should be used only to make sure its working.

    # ['close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Win/''}-{Discr/''}-{epoch}-{train columns}
    params = {'lr': 0.001,
                'loss': 'mean_absolute_percentage_error',
                'activation': 'tanh',
                'recurrent_activation': 'sigmoid',
                'epochs': 250,
                'batch_size': 150,
                'd': 22,
                'train_columns': ['close'],
                'label_column': 'close', 
                'name': 'Roondiwala-Std-250-Close', 
                'discretization': False,
                'fill_method': 'previous',
                'normalization': False }
    
    test = Test(Model=LSTM_Roondiwala, Train_Predictor=Forecasting_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
    test.rolling_window_test('./imgs/4-25-etb/')