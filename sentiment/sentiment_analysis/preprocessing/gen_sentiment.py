import flair
import pandas as pd
from the_pickler import *






def get_tweets(path_to_csv):
    return list(pd.read_csv(path_to_csv).tweets)





def gen_sentiments(tweets):

    # for each tweet gen sentiment
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')

    missed = list()
    sentiments = list()
    for i, tweet in enumerate(tweets):
        try:
            # create sentence object
            sent = flair.data.Sentence(tweet)
            sentiment_model.predict(sent)
            sentiments.append(sent.labels[0].value)
        except:
            missed.append(i)
            print('An error has occurred. The sentence responsible is at this index in tweets: ', i)
            print('Here is the tweet: ', tweet)
            print('len of the tweet: ', len(tweet))
            print('-----------------------------------------------------------------------------')


    print('final len of the sentiments list: ', len(sentiments))
    if missed:
        print('number of missed tweets: {}'.format(168000 - len(sentiments)))
        print('the indices of the missed tweets are: ', missed)
        for ind in missed:
            del tweets[ind]


    return tweets, sentiments




if __name__ == '__main__':



    #csv_filename = '../data/tweets.csv'
    #tweets = get_tweets()

    #path_to_csv = '../data/df.csv'
    #df = pd.DataFrame(list(zip(tweets, sentiments)), columns=['tweets', 'sentiments'])
    #df.to_csv(path_to_csv)




    path_to_csv = '../data/df.csv'
    df = pd.read_csv(path_to_csv, index_col=0)

    print(df.head())
    print(df.shape)
    print(df.sentiments.value_counts())
