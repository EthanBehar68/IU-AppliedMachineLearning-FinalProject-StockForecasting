import pickle
from api_wrapper import *
from twarc import Twarc
import re
from stats import Stats


# TODO:
# GET DATA INTO R

# TODO:
#
# 1 file, total number of words per tweet, check normality


if __name__ == "__main__":
    with open("data/annotated.pkl", 'rb') as file:
        dict = pickle.load(file)

    ak = "xyWmHDviqN0R933YO02c99KmD"  # api key
    aks = "TGhQp5kFkXzUueFLsL1DuNLTiVxfZQIFVD1Pqobm5ugQuwaxdu"  # api key secret
    at = "1063516087751503872-e5aNVELMhHx5Y6i5Xy0RJVCkruupu9"  # access token
    ats = "SM0FL1X3ki6m0fj3ehr6GxPnx4YtD8Mgk7VsSRKDF83qk"  # access token secret
    api = API(Auth(ak, aks, at, ats))  # fill the parameters of the Auth object

    replyids = list(dict.keys())
    tweets = []


    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def remove_junk(text):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

    replyids = list(divide_chunks(replyids, 100))
    for l in replyids:
        tweets.extend(api.get_tweets_from_ids(l))
    print(len(tweets))
    print(tweets)

    # data sets share an index for t
    words = []
    opinion = []
    for tweet in tweets:
        # grab text and id from tweet and clean the text
        if "retweeted_status" in dir(tweet):
            text, id = tweet.retweeted_status.full_text, tweet.retweeted_status.id
            text = remove_junk(text)
        else:
            text, id = tweet.full_text, tweet.id
            text = remove_junk(text)

        # save number of words from a tweet
        words.append(len(text.split()))
        opinion.append(dict[tweet.id][1])
        # write(id, text)
    print(len(words))
    print(len(opinion))
    print(words)
    print(opinion)

    # ------------------------ PART 2 ------------------------------ #
    # list of parent tweet numbers, list of child tweet numbers, list of all tweet numbers
    all_tweets = []
    parent_n = []
    child_n = []
    replies_n = []
    # grab all data from tweets.pkl
    dicts = []
    with open("data/tweets.pkl", 'rb') as file:
        while True:
            try:
                dicts.append(pickle.load(file))
            except EOFError:
                break

    print("Collecting tweets by ID...")
    # loop through dictionaries and grab all the parent tweets based on ids

    dictnum = 1
    for dictionary in dicts:
        ids = list(dictionary.keys())
        i = 0
        # grab tweets in chunks of 100
        while i + 100 < len(ids):
            all_tweets.extend(api.get_tweets_from_ids(ids[i:i + 100]))
            i += 100
        else:
            all_tweets.extend(api.get_tweets_from_ids(ids[i:]))
        print("done with dict number " + str(dictnum))
        dictnum += 1
    print("Tweets collected...")

    # loop through all tweets
    for tweet in all_tweets:
        # grab text and id from tweet and clean the text
        if "retweeted_status" in dir(tweet):
            text, id = tweet.retweeted_status.full_text, tweet.retweeted_status.id
            text = remove_junk(text)
        else:
            text, id = tweet.full_text, tweet.id
            text = remove_junk(text)

        # save number of words from a tweet
        parent_n.append(len(text.split()))
        # write(id, text)

        # get dict that contains tweet id
        for dictionary in dicts:
            if id in list(dictionary.keys()):
                d = dictionary
                break

        if d in dicts:
            # save number of replies of the tweet
            replies_n.append(len(d[id]))
            # for each reply get the text and id and clean the text
            for rep in d[id]:
                text = remove_junk(rep['full_text'])
                child_n.append(len(text.split()))
                # write(rep['id'], text)

    # num words per tweet is replies + parents
    print(len(parent_n))
    print(len(child_n))
    print(replies_n)

    # ---------------------------------- WRITE TO FILES ---------------------- #
    def write(file, info):
        writer = csv.writer(open(file, mode='w'), delimiter=',', quotechar='"')
        writer.writerows([info])


    # write("parent.csv", parent_n)
    # write("child.csv", child_n)
    # write("replies.csv", replies_n)
    # write("annotated_words.csv", words)
    # write("opinions.csv", opinion)
