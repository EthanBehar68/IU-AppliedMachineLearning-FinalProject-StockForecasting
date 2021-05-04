import pandas as pd
import numpy as np



if __name__ == '__main__':


    train_df = pd.read_csv('../data/train.csv', index_col=0)
    test_df = pd.read_csv('../data/test.csv', index_col=0)

    # binary np array representing the sentiment present in the corresponding tweets
    train_bin = np.array([1 if sent=='POSITIVE' else 0 for sent in train_df.label])
    test_bin = np.array([1 if sent=='POSITIVE' else 0 for sent in test_df.label])


    train_df.drop(['label'], axis=1)
    train_df['label'] = train_bin

    test_df.drop(['label'], axis=1)
    test_df['label'] = test_bin

    print(train_df.head())
    print(test_df.head())


    train_df.to_csv('../data/train_bin_labels.csv', index=False)
    test_df.to_csv('../data/test_bin_labels.csv', index=False)
