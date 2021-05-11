"""
Description:

This is a script for pulling tweets from Twitter servers.
It uses the tweepy wrapper for the Twitter api.

For use, you will need a Twitter developer account and to have generated consumer and consumer secret keys for said account
"""

import os
import time
import tweepy
import pickle
from the_pickler import *



class Auth:
    def __init__(self, consumer_key, secret_key):
        self.ck = consumer_key
        self.sk = secret_key


class API:
    def __init__(self, key):
        self.key = key
        self.api = None
        self.authorize()


    def authorize(self):
        # set up Tweepy API object using OAuth2
        auth = tweepy.AppAuthHandler(self.key.ck, self.key.sk)
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)



    # search tweets based on a query string
    def search_tweets(self, query, since):
        tweets = list()
        for tweet in tweepy.Cursor(self.api.search, q=query, lang='en', result_type='popular', count=200, since=since).items():
            tweets.extend(tweet.text)
        return tweets




def pkl(filename, tweets):
    f = open(filename, 'ab')
    pickle.dump(tweets, f)
    f.close()



def unpkl(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    return data





# if RateLimitError raised by twitter/tweepy, sleep 15 seconds and continue
# input
# - cursor: tweepy cursor object
#
def handle_limit_error(cursor):
    while True:
        try:
            yield next(cursor)
        except tweepy.RateLimitError:
            time.sleep(15 * 60)
        except tweepy.TweepError:
            time.sleep(15 * 60)




if __name__ == '__main__':


    consumer_key = 'SKlP3MDbx2khU8ncYxTPMyB4N'
    secret_key = '4P6eg3GdAPzdGgZ6JwiYNKPN5t0fNt78DHRdQ0ZYfAlzk7gH3D'


    query = 'btc Btc BTC #btc #Btc #BTC'
    api = tweepy.API(tweepy.AppAuthHandler(consumer_key, secret_key))

    num = unpkl('num.pkl')
    tweets = list()
    for tweet in handle_limit_error(tweepy.Cursor(api.search, q=query, lang='en', since_id='2016-01-01').items()):
        tweets.append(tweet.text)
        if len(tweets) == 1000:
            print('pickling tweets {}'.format(num))
            filename = 'tweets{}.pkl'.format(num)
            pkl(filename, tweets)
            tweets = list()
            num += 1
            if os.path.exists('num.pkl'):
                os.remove('num.pkl')
            pkl('num.pkl', num)
