from fastquant import get_stock_data
from hmmlearn import hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# training with apple feb-10-2003 -> sep-10-2004
# testing with apple sep-13-2004 -> jan-21-2005

training_data = get_stock_data("AAPL", "2003-02-10", "2004-09-10")
testing_data = get_stock_data("AAPL", "2004-09-13", "2005-01-21")

print(training_data.head())

def data_prep(data):
    df = pd.DataFrame(data=None, columns=['fracChange','fracHigh','fracLow'])
    df['fracChange'] = (data['close']-data['open'])/data['open']
    df['fracHigh'] = (data['high']-data['open'])/data['open']
    df['fracLow'] = (data['open']-data['low'])/data['low']

    return df

x_train = data_prep(training_data)
x_test = data_prep(testing_data)
print(x_test.head())


model = hmm.GMMHMM(n_components=4,n_mix=5,algorithm="map",n_iter=100)
model.fit(x_train)
post = model.predict_proba([[-0.008082, 0.005296, 0.015855]],lengths=[1])
print(post)
