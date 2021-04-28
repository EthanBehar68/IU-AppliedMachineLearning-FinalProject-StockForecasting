from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.losses import MeanAbsolutePercentageError
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from keras.optimizers import RMSprop

def build_simple_rnn(useDropout = True):
    scaler = MinMaxScaler(feature_range=(0,1))

    model = Sequential()
    model.add(SimpleRNN(32, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(SimpleRNN(32, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(SimpleRNN(32, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(SimpleRNN(32, return_sequences=False, activation=activation_funct))
    model.add(Dense(1, activation=activation_funct))

    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    return model, scaler

def build_groovy(useDropout = True):
    scaler = MinMaxScaler(feature_range=(0,1))

    model = Sequential()
    model.add(GRU(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(GRU(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(GRU(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(GRU(50, return_sequences=False, activation=activation_funct))
    model.add(Dense(1, activation=activation_funct))

    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    return model, scaler

def build_loovy(useDropout = True):
    scaler = MinMaxScaler(feature_range=(0,1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=False, activation=activation_funct))
    model.add(Dense(1, activation=activation_funct))

    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    return model, scaler

def build_super_groovy(useDropout = True):
    scaler = MinMaxScaler(feature_range=(0,1))

    model = Sequential()
    model.add(GRU(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(GRU(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(GRU(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(GRU(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(GRU(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(GRU(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(GRU(50, return_sequences=False, activation=activation_funct))
    model.add(Dense(1, activation=activation_funct))

    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    return model, scaler

def build_super_loovy(useDropout = True):
    scaler = MinMaxScaler(feature_range=(0,1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=True, activation=activation_funct))
    if useDropout: model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=False, activation=activation_funct))
    model.add(Dense(1, activation=activation_funct))
   
    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    return model, scaler

learning_rate = 0.001
momentum = 0.9
# optimizer = RMSprop(learning_rate=learning_rate) created in each RNN class
loss = 'mean_absolute_percentage_error'
mape = MeanAbsolutePercentageError()
activation_funct = 'elu'
epochs = 100
batch_size = 150
dropout = 0.2
train_data_columns = ['high']
test_data_column = ['high']
day_ranges = [5,10,15,20,25,30,35,40,45,50,55,60]

def get_data(day_range):
    full_apple_train = get_stock_data("AAPL", "2003-02-10", "2004-09-12")
    last_59_train = full_apple_train[-(day_range-1):]
    part_apple_test = get_stock_data("AAPL", "2004-09-13", "2005-01-22")
    full_apple_test = last_59_train.append(part_apple_test)
    full_ibm_train = get_stock_data("IBM", "2003-02-10", "2004-09-12")
    last_59_train = full_ibm_train[-(day_range-1):]
    part_ibm_test = get_stock_data("IBM", "2004-09-13", "2005-01-22")
    full_ibm_test = last_59_train.append(part_ibm_test)
    return full_ibm_train, full_ibm_test, full_apple_train, full_apple_test

def data_prep(in_train_data, in_test_data, scaler, data_columns=['high']):
    train_data = in_train_data[data_columns].values
    train_scale = scaler.fit_transform(train_data)
    test_data = in_test_data[data_columns].values
    test_scale = scaler.transform(test_data)
    return train_scale, test_scale, scaler

def get_xy_train_sets(train_data):
    X_train = []
    Y_train = []
    for i in range(day_range, len(train_data)):
        X_train.append(train_data[i-day_range:i, 0])
        Y_train.append(train_data[i,0])

    x_train, y_train = np.array(X_train), np.array(Y_train)
    x_train = np.reshape(x_train, (*x_train.shape, 1))
    return x_train, y_train

def get_xy_test_sets(scaled_test_data, raw_test_data, data_column='high'):
    X_test = []
    for i in range(day_range, len(scaled_test_data)):
        X_test.append(scaled_test_data[i-day_range:i, 0])
    x_test = np.array(X_test)
    x_test = np.reshape(x_test, (*x_test.shape, 1))
    test_data = raw_test_data[data_column].values
    Y_test = []
    for i in range(day_range, len(test_data)):
        Y_test.append(test_data[i])
    y_test = np.array(Y_test)
    return x_test, y_test

def train(model):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict(model, x_test, scaler):
    scaled_predictions = model.predict(x_test)
    fitted_scaled_predictions = np.zeros(shape=(len(scaled_predictions), len(train_data_columns)))
    fitted_scaled_predictions[:,0] = scaled_predictions[:,0]
    test_predictions = scaler.inverse_transform(fitted_scaled_predictions )[:,0]
    return test_predictions

results = []
current_result = ''
rnn = None
scaler = None
for i in range(5):
    if i == 0:
        for day_range in day_ranges:
            print('SimpleRNN day_range=', day_range,' started')
            current_result = 'SimpleRNN day_range=' + str(day_range) + ' loss: '
            full_apple_train, full_apple_test, full_ibm_train, full_ibm_test = get_data(day_range)
            rnn, scaler = build_simple_rnn()
            train_scale, test_scale, scaler = data_prep(full_apple_train, full_apple_test, scaler, train_data_columns)
            x_train, y_train = get_xy_train_sets(train_scale)
            x_test, y_test = get_xy_test_sets(test_scale, full_apple_test, test_data_column)
            rnn = train(rnn)
            y_preds = predict(rnn, x_test, scaler)
            losses = mape(y_test, y_preds).numpy()
            current_result = current_result + str(losses)
            results.append(current_result)
    elif i == 1:
        for day_range in day_ranges:
            current_result = 'Groovy day_range=' + str(day_range) + ' loss: '
            print('Groovy day_range=', day_range,' started')
            full_apple_train, full_apple_test, full_ibm_train, full_ibm_test = get_data(day_range)
            rnn, scaler = build_groovy()
            train_scale, test_scale, scaler = data_prep(full_apple_train, full_apple_test, scaler, train_data_columns)
            x_train, y_train = get_xy_train_sets(train_scale)
            x_test, y_test = get_xy_test_sets(test_scale, full_apple_test, test_data_column)
            rnn = train(rnn)
            y_preds = predict(rnn, x_test, scaler)
            losses = mape(y_test, y_preds).numpy()
            current_result = current_result + str(losses)
            results.append(current_result)
    elif i == 2:
        for day_range in day_ranges:
            current_result = 'Loovy day_range=' + str(day_range) + ' loss: '
            print('Loovy day_range=', day_range,' started')
            full_apple_train, full_apple_test, full_ibm_train, full_ibm_test = get_data(day_range)
            rnn, scaler = build_loovy()
            train_scale, test_scale, scaler = data_prep(full_apple_train, full_apple_test, scaler, train_data_columns)
            x_train, y_train = get_xy_train_sets(train_scale)
            x_test, y_test = get_xy_test_sets(test_scale, full_apple_test, test_data_column)
            rnn = train(rnn)
            y_preds = predict(rnn, x_test, scaler)
            losses = mape(y_test, y_preds).numpy()
            current_result = current_result + str(losses)
            results.append(current_result)            
    elif i == 3:
        for day_range in day_ranges:
            current_result = 'SGroovy day_range=' + str(day_range) + ' loss: '
            print('SGroovy day_range=', day_range,' started')
            full_apple_train, full_apple_test, full_ibm_train, full_ibm_test = get_data(day_range)
            rnn, scaler = build_super_groovy()
            train_scale, test_scale, scaler = data_prep(full_apple_train, full_apple_test, scaler, train_data_columns)
            x_train, y_train = get_xy_train_sets(train_scale)
            x_test, y_test = get_xy_test_sets(test_scale, full_apple_test, test_data_column)
            rnn = train(rnn)
            y_preds = predict(rnn, x_test, scaler)
            losses = mape(y_test, y_preds).numpy()
            current_result = current_result + str(losses)
            results.append(current_result)
    elif i == 4:
        for day_range in day_ranges:
            current_result = 'SLoovy day_range=' + str(day_range) + ' loss: '
            print('SLoovy day_range=', day_range,' started')
            full_apple_train, full_apple_test, full_ibm_train, full_ibm_test = get_data(day_range)
            rnn, scaler = build_super_loovy()
            train_scale, test_scale, scaler = data_prep(full_apple_train, full_apple_test, scaler, train_data_columns)
            x_train, y_train = get_xy_train_sets(train_scale)
            x_test, y_test = get_xy_test_sets(test_scale, full_apple_test, test_data_column)
            rnn = train(rnn)
            y_preds = predict(rnn, x_test, scaler)
            losses = mape(y_test, y_preds).numpy()
            current_result = current_result + str(losses)
            results.append(current_result)
    else:
        print('how is this possible - something went wrong')

for s in results:
    print(s)
