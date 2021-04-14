from abc import ABC, abstractmethod
from fastquant import get_stock_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#Base model class
class Model(ABC):

    # params should be a dict of your parameters that you want to pass to the model
    # name should be a string (used for saving results)
    # params dict *must* include 'name':name within it 
    def __init__(self, params):
        self.model = None
        self.name = params['name']
    
    # wrapper model function for collecting fastquant data
    def get_data(self, ticker, start_date, end_date):
        return get_stock_data(ticker, start_date, end_date)

    # plotting function for standardized plot results
    def plot_results(self, preds, actual, title):
        fig, ax = plt.subplots(figsize=(15,5))
        ax.set_title(title)
        time = range(len(preds))
        ax.plot(time,preds,color='tab:red',marker='s',markersize=2,linestyle='-',linewidth=1,label='forcast')
        ax.plot(time,actual,color='tab:blue',marker='s',markersize=2,linestyle='-',linewidth=1,label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('stock price ($)')
        ax.set_xticks(np.arange(0,len(preds)+10,10))
        ax.set_xlim(0,len(preds)+10)
        ax.xaxis.grid(True,ls='--')
        ax.yaxis.grid(True,ls='--')
        ax.legend()
        plt.savefig(f'../imgs/{title}.png')
    
    # plotting function for training data + prediction + actual
    def plot_continuous(self, preds, train, actual, title):
        last_50 = train['close'].values[-50:]
        last = np.append(last_50, actual[0])
        fig, ax = plt.subplots(figsize=(15,5))
        ax.set_title(title)
        pred_time = range(len(last_50),len(last_50)+len(preds))
        train_time = range(0,len(last_50)+1)
        ax.plot(pred_time,preds,color='tab:red',marker='s',markersize=2,linestyle='-',linewidth=1,label='forcast')
        ax.plot(train_time,last,color='tab:blue',marker='s',markersize=2,linestyle='-',linewidth=1,label='actual')
        ax.plot(pred_time,actual,color='tab:blue',marker='s',markersize=2,linestyle='-',linewidth=1)
        ax.set_xlabel('time')
        ax.set_ylabel('stock price ($)')
        ax.set_xticks(np.arange(0,len(pred_time)+len(last_50)+10,10))
        ax.set_xlim(0,len(pred_time)+len(last_50)+10)
        ax.xaxis.grid(True,ls='--')
        ax.yaxis.grid(True,ls='--')
        ax.legend()
        plt.savefig(f'../imgs/{title}.png')

    # function to get error of the model based on preds and true values
    def mean_abs_percent_error(self, y_pred, y_true):
        return (1.0)/(len(y_pred))*((np.abs(y_pred-y_true)/np.abs(y_true))*100).sum()

    # training function for the model, should create the model, train it, and store in self.model
    @abstractmethod
    def train(self, train_data):
        pass
    
    # prediction function for the model, should return the preds and y_true given the test data
    @abstractmethod
    def predict(self, test_data):
        # return preds,actual
        pass