import sys
from fastquant import get_stock_data
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from datetime import datetime, timedelta

class SentimentModel():
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

        # ignore the last date since we have no stock for 'next day' (twitter scrapes live)
        self.dates = self.dates[:-1]

        # we need all tickers from the tweets data set to create the training data
        self.tickers = self.tweets['ticker'].unique()
        
        # build the training data set using dates and tickers
        self.train_data = pd.DataFrame(data=None, columns=['fracChange','avgSent','ticker'])
        for ticker in self.tickers:
            self.train_data = self.train_data.append(self.gen_data(ticker))

        # x and y for training the linear model
        x = np.array(self.train_data['avgSent'])
        y = np.array(self.train_data['fracChange'])

        # build and fit the regression model
        self.model = LinearRegression().fit(x.reshape(-1,1),y.reshape(-1,1))

        # plotting the regression model (uncomment to see)
        # plt.scatter(x,y,c='tab:blue')
        # plt.plot(x,self.model.predict(x_poly),c='tab:green')
        # plt.xlabel('average sentiment')
        # plt.ylabel('fractional chance')
        # plt.show()

    def predict(self):
        # use actual ticker this sent-model is supposed to predict
        # TODO: this is why we need more data... cant be just using the same data as from
        #       from training, when enough is present, we'll use different start and end dates
        #       for both training and testing... issue is also that the observations are from training
        #       we will need to pull observations from some testing date set. In reality this model
        #       is like the others used for just next day prediction... again in the future with more
        #       data we could possibly use multiple previous days of sentiment to predict the next day
        #       we have to add 1 day since fastquant is not inclusive of the end date
        test_data = get_stock_data(self.ticker, self.start_date, self.end_date+timedelta(days=1))
        
        # again we ignore the first day, since we dont have the day before its sentiment
        test_data = test_data[1:]

        # true opening and closing prices... same type of prediction model as ghmm and rnn's
        # however here we cant use our predicted fractional change as part of the new observations
        # we actually have to have the average sentiment for a previous day... tricky model here...
        actual = test_data['close'].values
        opens = test_data['open'].values

        # observations again should come from a test set when we have enough data
        obs = self.train_data[self.train_data['ticker']==self.ticker]['avgSent'].values
        
        # loop through test data and predict closing prices using opening and the model
        preds = []
        for i in range(len(test_data)):
            pred_frac_change = self.model.predict(obs[i].reshape(-1,1))
            pred_close = pred_frac_change[0]*opens[i]+opens[i]
            preds.append(pred_close)
        
        return np.array(preds).flatten(),actual
    
    # plotting function for standardized plot results
    def plot_results(self, preds, actual, title):
        fig, ax = plt.subplots(figsize=(15,5))
        ax.set_title(title)
        time = range(len(preds))
        ax.plot(time,preds,color='tab:red',marker='s',markersize=2,linestyle='-',linewidth=1,label='forcast')
        ax.plot(time,actual,color='tab:blue',marker='s',markersize=2,linestyle='-',linewidth=1,label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('stock price ($)')
        ax.set_xticks(np.arange(0,len(preds),1))
        ax.set_xlim(0,len(preds))
        ax.xaxis.grid(True,ls='--')
        ax.yaxis.grid(True,ls='--')
        ax.legend()
        plt.savefig(f'../imgs/{title}.png')
    
    # function to get error of the model based on preds and true values
    def mean_abs_percent_error(self, y_pred, y_true):
        return (1.0)/(len(y_pred))*((np.abs(y_pred-y_true)/np.abs(y_true))*100).sum()
    
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
        
        # create cols in the data frame for the average sentiment and the ticker
        ticker_frac_change['avgSent'] = avg_sent
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
        
        model = SentimentModel(params=params)

        model.train()
        preds,actual = model.predict()
       
        model.plot_results(preds=preds,actual=actual,
                        title=f'SentimentModel {ticker} forcasted vs actual stock prices 2021-04-13 to 2021-04-19')

        error = model.mean_abs_percent_error(y_pred=preds, y_true=actual)
        print(f'{ticker}: {error}')
