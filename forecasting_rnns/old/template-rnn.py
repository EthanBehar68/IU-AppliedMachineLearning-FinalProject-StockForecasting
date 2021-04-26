from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# Parameters for fastquant
train_start_date = "2006-01-01"
train_end_date = "2016-12-31"
test_start_date = "2017-01-01"
test_end_date = "2017-12-31"
ticker = "IBM"
inputs = ['open','high','low','close','volume']

# Parameters for data 
previous_days = 60 # Previous day worth of data to train on

# Hyperparameters
hp_optimizer='rmsprop'
hp_loss='mean_squared_error'
hp_epochs=100
hp_batch_size=150

# Scaler
mmc = MinMaxScaler(feature_range=(0,1))
stdc = StandardScaler(feature_range=(0,1))

train_data = []
test_data = []
def load_data(in_inputs):
    train_data = get_stock_data(ticker, train_start_date, train_end_date)[in_inputs].values
    test_data = get_stock_data(ticker, test_start_date, test_end_date)[in_inputs].values

    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

def get_train_test_data(scaler):
    train_scale = scaler.fit_transform(train_data)
    test_scale = scaler.transform(test_data)

    X_train = []
    Y_train = []

    for i in range(previous_days, len(train_scale)):
        X_train.append(train_scale[i-previous_days:i,0])
        Y_train.append(train_scale[i,0])

    x_train,y_train = np.array(X_train),np.array(Y_train)

    x_train = np.reshape(x_train, (*x_train.shape,1))

    X_test = []
    for i in range(previous_days,len(test_scale)):
        X_test.append(test_scale[i-previous_days:i,0])

    x_test = np.array(X_test)
    x_test = np.reshape(x_test, (*x_test.shape,1))

    return x_train, y_train, x_test

def gen_model():
    # Our simple RNN as an example    
    model = Sequential()
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32))
    model.add(Dense(1))

    return model

def train_model(model,x_train,y_train,x_test,inepoch,inbatch_size,sc,optimizer_func,loss_func):
    model.compile(optimizer=optimizer_func,loss=loss_func)

    model.fit(x_train,y_train,epochs=inepoch,batch_size=inbatch_size)

    scaled_preds = model.predict(x_test)
    test_preds = sc.inverse_transform(scaled_preds)

    return model, test_preds

model = gen_model()
model, preds = gen_model(model,x_train,y_train,x_test,hp_epochs,hp_batch_size,mmc,hp_optimizer,hp_loss)

diff = len(test_data) - len(preds)
plt.figure(figsize=(20,5))
plt.plot(preds, label='prediction')
plt.plot(test_data[diff:], label='actual')
plt.title(f'predicting IBM daily highs for 2017 based on 2006-2016 daily highs')
plt.legend()
# plt.savefig('../imgs/graph.png')
plt.savefig('./imgs/graph1.png') # Windows' pleb ass only uses 1 dot.
plt.show()

