import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import flair
import re
import sys
import os


class TweetCollector:
    def __init__(self, token, ticker):
        self.token = token
        self.ticker = ticker
        self.format = '%Y-%m-%dT%H:%M:%SZ'
        self.model = flair.models.TextClassifier.load('en-sentiment')  
    
    # can only get tweets in the last week so we just use this for total data generation
    # we then split this data to use for training / testing the model
    def get_tweets(self):
        # setup params
        params = {
            'query': f'({self.ticker}) (lang:en)',
            'tweet.fields': 'created_at,lang',
            'max_results': 100,
        }
        headers = {'authorization': f'Bearer {self.token}'}

        # start at current date for data collection, go back 1 week hour by hour
        now = datetime.now()
        last_week = now-timedelta(days=7)
        now = now.strftime(self.format)
        i = 0

        # collect 100 tweets per hour from start to end
        tweets = pd.DataFrame()
        while True:
            if datetime.strptime(now,self.format) < last_week:
                break
            
            # index to make sure we arent in infinite while loop
            print(i, end='\r', flush=True)
            i += 1
            
            # back up 60 minutes
            back = self.backup(now)
            
            params['start_time'] = back
            params['end_time'] = now
            
            response = requests.get(
                'https://api.twitter.com/2/tweets/search/recent',
                params=params,
                headers=headers
            )
            
            # break if we got an error from trying to look to far back (free API kinda sucks for this)
            if 'errors' in response.json():
                break

            now = back
            
            for tweet in response.json()['data']:
                tweets = tweets.append(self.format_tweet(tweet), ignore_index=True)

        # put dates in form yyy-mm-dd
        tweets['date'] = tweets['date'].apply(self.convert_date)

        return tweets

    def format_tweet(self, tweet):
        data = {
            'id': tweet['id'],
            'date': tweet['created_at'],
            'lang': tweet['lang'],
            'text': self.clean_tweet(tweet['text']),
            'ticker': self.ticker
        }
        
        # sentiment analysis using flair
        sentence = flair.data.Sentence(text=data['text'])
        pred = self.model.predict(sentence)

        data['probability'] = sentence.labels[0].score
        data['sentiment'] = sentence.labels[0].value
        
        return data
    
    def convert_date(self, date):
        date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%fZ')
        return date.strftime('%Y-%m-%d')
    
    def backup(self, now):
        now = datetime.strptime(now,self.format)
        back = now - timedelta(minutes=60)
        return back.strftime(self.format)

    def clean_tweet(self, tweet):
        return re.sub(r'@\S+|https?://\S+', '', tweet)


if __name__ == "__main__":    
    file_name = 'tweets.csv'
    
    # get bearer token
    with open('tokens.txt') as f:
        token = f.read()
        token = token.strip('\n')

    tickers = ['tsla', 'amzn', 'msft', 'aapl', 'pltr']

    for ticker in tickers:
        collector = TweetCollector(token=token, ticker=ticker)
        tweets = collector.get_tweets()

        # write csv to file
        if os.path.exists(file_name):
            tweets.to_csv(file_name, mode='a', header=False)
        else:
            tweets.to_csv(file_name, header=True)    

    


