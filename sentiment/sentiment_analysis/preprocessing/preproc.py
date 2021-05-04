import re
import random
import matplotlib.pyplot as plt
from torchtext.legacy import data




# clean the tweets before they are passed through the network
def tweet_cleanup(tweets):
    cleaned_tweets = list()
    for tweet in tweets:
        # remove punct
        tweet = re.sub('[!#?,.:";]', ' ', tweet)
        # remove excess spaces
        tweet = re.sub(r' +', ' ', tweet)
        # remove newlines
        tweet = re.sub(r'\n', ' ', tweet)
        cleaned_tweets.append(tweet)
    return cleaned_tweets


def get_files(path, train_size, max_doc_len, seed, tokenizer):
    # including lengths makes the text var a tuple containing the tweet and its length
    Text = data.Field(preprocessing=tweet_cleanup, tokenize=tokenizer, batch_first=True, include_lengths=True, fix_length=max_doc_len, lower=True)
    Label = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

    fields = [('text', Text), ('labels', Label)]

    # builds a pytorch dataset from the given training and testing files
    train_data, test_data = data.TabularDataset.splits(
            path=path,
            train='../data/train_bin_labels.csv',
            test='../data/test_bin_labels.csv',
            format='csv',
            fields=fields,
            skip_header=True
    )

    train_data, val_data = train_data.split(split_ratio=train_size, random_state=random.seed(seed))
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(val_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    return train_data, val_data, test_data, Text, Label
