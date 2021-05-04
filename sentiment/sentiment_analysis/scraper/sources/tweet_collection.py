from api_wrapper  import *
from twarc import Twarc
import pickle


class TweetCollection:
    def __init__(self):
        """
        Constructor to set up the tweet collection class. swap out keys with your own
        """
        self.ak = "xyWmHDviqN0R933YO02c99KmD"  # api key
        self.aks = "TGhQp5kFkXzUueFLsL1DuNLTiVxfZQIFVD1Pqobm5ugQuwaxdu"  # api key secret
        self.at = "1063516087751503872-e5aNVELMhHx5Y6i5Xy0RJVCkruupu9"  # access token
        self.ats = "SM0FL1X3ki6m0fj3ehr6GxPnx4YtD8Mgk7VsSRKDF83qk"  # access token secret
        self.api = API(Auth(self.ak, self.aks, self.at, self.ats))  # fill the parameters of the Auth object
        self.dict = {}  # dict to hold current queries tweets and replies
        self.dicts = []  # list to hold all dicts from pickle file
        self.tweets = []  # list to hold Status tweet objects fro query
        self.file = "tweets.pkl"

    def tweepy_to_twarc(self, tweet):
        """
        Convert tweepy status object to a dictionary
        :param tweet: tweet to make into dict
        :return: dict representation of tweet
        """
        tw = dict()
        tw['id'] = tweet.id
        tw['id_str'] = str(tweet.id)
        tw['user'] = dict()
        tw['user']['screen_name'] = tweet.user.screen_name
        return tw

    def query(self, q):
        """
        Create list of tweets based on a query
        :param q: hashtag to search for
        :return:
        """
        # change count to change amount of tweets
        self.tweets = self.api.search_tweets(query=q, count=100, since="2019-03-01")
        print(len(self.tweets), ' tweets were collected')

    def collect_replies(self):
        """
        Collect replies for all tweets from query using twarc
        :return:
        """
        twarc = Twarc(self.ak, self.aks, self.at, self.ats)
        reply_count = 0
        # loop through all parent tweets from query
        for tweet in self.tweets:
            replies = []
            reps = twarc.replies(self.tweepy_to_twarc(tweet), recursive=False)  # get iterator for replies from twarc
            rep = next(reps)  # first "rep" is the parent tweet so we don't use it
            i = 0
            # max 30 replies
            while i<30:
                try:
                    rep = next(reps)  # get next reply and add it to list
                    replies.append(rep)
                    i = i + 1
                except StopIteration: break
                except Exception as e: print('error: ', e)
            self.dict[tweet.id] = replies  # add tweet to dict {id:replies}
            reply_count += len(replies)
        print(reply_count, ' replies were collected')

    def print(self):
        """
        print the dictionary of tweets and replies
        :return:
        """
        for id in list(self.dict.keys()):
            print("===========================")
            print(id)
            print("---------------------------")
            for rep in self.dict[id]: print(rep)
            print("===========================")


    def save(self):
        """
        Save dictionary of tweets and replies
        :return:
        """
        pickle.dump(self.dict, open(self.file, 'ab'))

    def load(self):
        """
        load dictionary's from the pickle file
        :return:
        """
        with open(self.file, 'rb') as file:
            while True:
                try: self.dicts.append(pickle.load(file))
                except EOFError: break


if __name__ == "__main__":
    tc = TweetCollection()

    #  Hastags:
    # ["#MarchForLife", "#MAGA", "#GlobalWarming",
    #  "#FlatEarth", "#FlatEarthSociety", "#StateOfTheUnion",
    #  "#SOTU", "#AOC", "#Trump2020", "#BlackLivesMatter", "#BlueLivesMatter"]

    # replace ... w/ hashtag to collect tweets and replies and save them
    # tc.query("#Trump2020")
    # tc.collect_replies()
    # tc.save()
    #
    # # load in current pickle file / print stats + clean tweets of hashtags and links
    # tc.load()
    # tc.stats()

    tc.load()
    print("Collecting parent tweets by ID...")
    # loop through dictionaries and grab all the parent tweets based on ids
    n = 1
    parent_tweets = []
    for dict in tc.dicts:
        ids = list(dict.keys())
        i = 0
        # grab tweets in chunks of 100
        while i + 100 < len(ids):
            parent_tweets.extend(tc.api.get_tweets_from_ids(ids[i:i + 100]))
            i += 100
        else:
            parent_tweets.extend(tc.api.get_tweets_from_ids(ids[i:]))
        print("done with dict number " + str(n))
        n += 1
    print("Parent tweets collected...")

    # d = {id1 : (tweet1, [replies])}
    d = {}
    for tweet in parent_tweets:
        for dict in tc.dicts:
            if tweet.id in list(dict.keys()):
                d[tweet.id] = (tweet, dict[tweet.id])
                break

    pickle.dump(d, open("data1.pkl", 'wb'))
