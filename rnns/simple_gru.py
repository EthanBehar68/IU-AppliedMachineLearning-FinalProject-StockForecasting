from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU 
import tensorflow as tf
import config as cf




# parameters from config.py

train_data = get_stock_data("IBM", "2006-01-01", "2016-12-31")['high'].values
test_data = get_stock_data("IBM", "2017-01-01", "2017-12-31")['high'].values

train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

sc = MinMaxScaler(feature_range=(0,1))
train_scale = sc.fit_transform(train_data)

test_scale = sc.transform(test_data)

X_train = []
Y_train = []

for i in range(60, len(train_scale)):
    X_train.append(train_scale[i-60:i,0])
    Y_train.append(train_scale[i,0])

x_train,y_train = np.array(X_train),np.array(Y_train)

x_train = np.reshape(x_train, (*x_train.shape,1))

X_test = []
for i in range(60,len(test_scale)):
    X_test.append(test_scale[i-60:i,0])

x_test = np.array(X_test)
x_test = np.reshape(x_test, (*x_test.shape,1))

def gen_model(x_train,y_train,x_test,sc):
    model = Sequential()
    model.add(tf.keras.layers.GRU(32, return_sequences=True))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    model.fit(x_train,y_train, epochs=100, batch_size=150)

    scaled_preds = model.predict(x_test)
    test_preds = sc.inverse_transform(scaled_preds)

    return model, test_preds


model, preds = gen_model(x_train,y_train,x_test,sc)

diff = len(test_data)-len(preds)
plt.figure(figsize=(20,5))
plt.plot(preds,label='prediction')
plt.plot(test_data[diff:],label='actual')
plt.title(f'predicting IBM daily highs for 2017 based on 2006-2016 daily highs')
plt.legend()
plt.savefig('../imgs/graph.png')
plt.pause(1)
plt.close()
