import sys
from fastquant import get_stock_data
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from datetime import datetime, timedelta

class SentimentTrendModel():
    def __init__(self, params):
        self.tweets = self.read_data(params['tweets_file'])
        self.ticker = params['ticker'].lower()
    
    def train(self):
        # TODO: when more data is present, we can only train on a certain amount of the dates
        #       the rest need to be reserved for testing... not possible right now
        #       we'll update the end date, and not include all unique dates based on this
        #       all tickers will continue to be used for training, as seen in the 
        #       model_exploration notebook, there seems to be a linear trend between previous
        #       days sentiment and the next days fractional change...

        # start and end date used for generating the training data
        self.start_date = min(self.tweets['date'])
        self.end_date = max(self.tweets['date'])

        # unique dates needed for the average sentiment calculation
        self.dates = sorted(self.tweets['date'].unique())
        print(len(self.dates))
        # ignore the last date since we have no stock for 'next day' (twitter scrapes live)
        self.dates = self.dates[:-1]

        # we need all tickers from the tweets data set to create the training data
        self.tickers = self.tweets['ticker'].unique()
        
        # build the training data set using dates and tickers
        self.train_data = pd.DataFrame(data=None, columns=['Trend','SentPred','ticker'])
        for ticker in self.tickers:
            self.train_data = self.train_data.append(self.gen_data(ticker))

    def predict(self):
        test_data = get_stock_data(self.ticker, self.start_date, self.end_date+timedelta(days=1))
        
        # again we ignore the first day, since we dont have the day before its sentiment
        test_data = test_data[1:]

        actual = test_data['close'].values
        actual = ['UPTREND' if a > 0 else 'DOWNTREND' for a in actual]

        # observations again should come from a test set when we have enough data
        obs = self.train_data[self.train_data['ticker']==self.ticker]['SentPred'].values
        
        # loop through test data and and see how often we are correct
        correct = 0
        for i in range(len(test_data)):
            if obs[i] == actual[i]:
                correct += 1
        
        return correct / len(actual)
    
    # helper function to generate the training data
    def gen_data(self, ticker):
        # grab only tweets corresponding to the given ticker
        ticker_data = self.tweets[self.tweets['ticker']==ticker]

        # collect the stock price data for the given ticker from date range
        ticker_price_data = get_stock_data(ticker,self.start_date,self.end_date+timedelta(days=1))

        # convert data to fractional change
        ticker_frac_change = self.data_prep(ticker_price_data)

        # since we are using prev sentiment to predict next day, first frac change is useless
        ticker_frac_change = ticker_frac_change[1:]
        
        # calculate the average sentiment over unique dates
        avg_sent = [self.calc_avg_sentiment(ticker,date) for date in self.dates]

        # convert average sent to predict uptrend or downtrend
        avg_sent = ['UPTREND' if sent > 0 else 'DOWNTREND' for sent in avg_sent]

        # create cols in the data frame for the average sentiment and the ticker
        ticker_frac_change['SentPred'] = avg_sent
        ticker_frac_change['ticker'] = ticker

        return ticker_frac_change

    # helper function for reading in the tweet csv data
    def read_data(self, f):
        tweets = pd.read_csv(f)
        tweets.drop(columns=['Unnamed: 0'], inplace=True)
        
        # drop NA and reset the index
        tweets = tweets.dropna()
        tweets.reset_index(inplace=True)
        tweets.drop(columns=['index'],inplace=True)

        # make the dates actual date objects so we can use min/max
        tweets['date'] = tweets['date'].apply(lambda date: datetime.strptime(date,'%Y-%m-%d'))

        # drop fridays and saturdays from the data (remove unnecessary cols after)
        tweets['day'] = tweets['date'].apply(lambda date: date.weekday())
        tweets = tweets[(tweets['day']!=4)&(tweets['day']!=5)]
        tweets.reset_index(inplace=True)
        tweets.drop(columns=['index', 'day'], inplace=True)

        return tweets

    # helper function to create fractional change data
    def data_prep(self,data):
        df = pd.DataFrame(data=None, columns=['fracChange'])
        df['fracChange'] = (data['close']-data['open'])/data['open']

        return df
    
    # helper function to get average sentimate for a given ticker and date within the loaded tweets data frame
    def calc_avg_sentiment(self,ticker,date):
        df = self.tweets[(self.tweets['ticker']==ticker) & (self.tweets['date']==date)]
        sentiment = df['sentiment'].values
        avg_sentiment = sum([-1 if s == 'NEGATIVE' else 1 for s in sentiment])/len(sentiment)
        
        return avg_sentiment

if __name__ == "__main__":
    for ticker in ['AMZN','AAPL','TSLA','PLTR']:
        params = {'tweets_file': 'tweets.csv',
                'ticker': f'{ticker}'}
        
        model = SentimentTrendModel(params=params)

        model.train()
        score = model.predict()
       
        print(f'{ticker}: {score}')
