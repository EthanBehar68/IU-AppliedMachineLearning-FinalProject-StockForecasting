import pandas as pd



if __name__ == '__main__':


    train_df = pd.read_csv('../data/train_bin_labels.csv', index_col=0)
    train_df.to_csv('../data/train_bin_labels.csv', index=False)

    test_df = pd.read_csv('../data/test_bin_labels.csv', index_col=0)
    test_df.to_csv('../data/test_bin_labels.csv', index=False)
