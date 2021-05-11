import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit



if __name__ == '__main__':

    # read in the csv
    tweets_df = pd.read_csv('../data/df.csv')
    # make sure have the proper data
    print(tweets_df.head())
    # split into x and y
    x = tweets_df.tweets
    y = tweets_df.sentiments


    # create sss obj
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.2)
    # generate splits
    for train_index, test_index in sss.split(x, y):
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

    # build dfs from training and testing data
    train_df = pd.DataFrame(list(zip(x_train, y_train)), columns=['tweet', 'label'])
    test_df = pd.DataFrame(list(zip(x_test, y_test)), columns=['tweet', 'label'])

    print(train_df.head())
    print(test_df.head())


    # write train and test files
    train_df.to_csv('../data/train.csv')
    test_df.to_csv('../data/test.csv')
