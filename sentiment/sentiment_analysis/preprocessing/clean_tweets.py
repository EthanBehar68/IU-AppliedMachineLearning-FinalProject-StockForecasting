import re # regular expression library
import os
import emoji # python lib for processing emojis (used in cleaning)
import string
from the_pickler import *
import pandas as pd



# source: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
# source: https://www.geeksforgeeks.org/python-efficient-text-data-cleaning/
def clean_tweet(tweet):
    def give_emoji_free_text(tweet):
        return emoji.get_emoji_regexp().sub(r'', tweet.decode('utf8'))

    def remove_urls(tweet):
        return re.sub(r'https?:\/\/.\S+', "", tweet)

    def remove_hashtags(tweet):
        return re.sub(r'#', '', tweet)

    def remove_contractions(tweet):
        contract_dict = {"'s":" is", "n't":" not", "'m":" am", "'ll":" will", "'d":" would", "'ve":" have", "'re":" are"}
        for key, val in contract_dict.items():
            if key in tweet:
                tweet = tweet.replace(key,val)
        return tweet

    def remove_punc(tweet):
        return re.sub(r'[^\w\s]','',tweet)

    def remove_newline(tweet):
        return re.sub('\n', ' ', tweet)

    def remove_excess_whitespace(tweet):
        return " ".join([word for word in tweet.split()])


    return remove_excess_whitespace(remove_newline(remove_punc(remove_contractions(remove_hashtags(remove_urls(give_emoji_free_text(tweet))))))).lower()




def clean_tweets(tweets):
    cleaned = [clean_tweet(tweet.encode('utf8')) for tweet in tweets]
    return [tweet for tweet in cleaned if not not tweet and not tweet.isspace()]




def collect_and_clean(path, num_files):
    tweets = list()

    # collect all tweets
    for i in range(1, num_files + 1):

        # build path to file
        filename = 'tweets{}.pkl'.format(i)
        path_to_file = os.path.join(path, filename)

        # collect tweets in file
        tweets.extend(unpkl(path_to_file))

    # clean tweets
    return clean_tweets(tweets)



if __name__ == '__main__':

    # collect, clean, and write to csv
    path_to_tweets = '../data/pulled_tweets'
    cleaned = collect_and_clean(path_to_tweets, 168)

    csv_filename = '../data/tweets.csv'
    pd.DataFrame(cleaned, columns=['tweets']).to_csv(csv_filename)
