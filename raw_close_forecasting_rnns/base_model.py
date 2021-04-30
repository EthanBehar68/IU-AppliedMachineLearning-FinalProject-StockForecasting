# Abstract functionality
from abc import ABC, abstractmethod
# Data Pre-processing
from fastquant import get_stock_data
# General Needed libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os


#Base model class
class Base_Model(ABC):

    # params should be a dict of your parameters that you want to pass to the model
    # name should be a string (used for saving results)
    # params dict *must* include 'name':name within it 
    def __init__(self, params):
        self.model = None
        self.name = params['name']
    
    # wrapper model function for collecting fastquant data
    def get_data(self, ticker, start_date, end_date):
        return get_stock_data(ticker, start_date, end_date)

    # function to get error of the model based on preds and true values
    def mean_abs_percent_error(self, y_pred, y_true):
        return (1.0)/(len(y_pred))*((np.abs(y_pred-y_true)/np.abs(y_true))*100).sum()

    def root_mean_squared_error(self, y_pred, y_true):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    # plotting function for standardized plot results
    def plot_loss(self, t_loss, v_loss, title, folder='./imgs/'):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_title(title, loc='center', pad=15, wrap=True)
        time = range(len(t_loss))
        ax.plot(time, t_loss,color='tab:red',marker='s',markersize=2,linestyle='-',linewidth=1,label='Train')
        ax.plot(time, v_loss,color='tab:blue',marker='s',markersize=2,linestyle='-',linewidth=1,label='Val')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_xticks(np.arange(0, len(t_loss), 50))
        ax.set_xlim(0, len(t_loss))
        ax.xaxis.grid(True,ls='--')
        ax.yaxis.grid(True,ls='--')
        ax.legend()
        # plt.savefig(f'../imgs/{title}.png')
        # Ethan needs this if running from .py file.
        plt.savefig(f'{folder}{title}.png')

    # plotting function for standardized plot results
    def plot_results(self, preds, actual, title, folder='./imgs/'):
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
        # plt.savefig(f'../imgs/{title}.png')
        # Ethan needs this if running from .py file.
        plt.savefig(f'{folder}{title}.png')

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
        # plt.savefig(f'../imgs/{title}.png')
        # Ethan needs this if running from .py file.
        plt.savefig(f'{folder}{title}.png')