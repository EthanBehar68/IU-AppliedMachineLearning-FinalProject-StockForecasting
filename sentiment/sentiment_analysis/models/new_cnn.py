import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):

    def __init__(self, vocab_size, embed_size, n_filters, filter_sizes, pool_size, hidden_size, n_classes, dropout_rate):

        super(CNN, self).__init__()

        # the embedding layer
        # creates a lookup table where each row represents a word in a numerical format and converts the integer sequence into a dense vector representation
        # vocab_size: num unique words in the dictionary
        # embed_size: num dimensions to use for representing a single word

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_size))
        self.max_pool1 = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(95*n_filters, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, n_classes, bias=True)






    def forward(self, text, text_lengths):

        e = self.embedding(text).unsqueeze(1)
        c1 = self.conv(e)
        max1 = self.max_pool1(c1)
        c2 = self.conv(max1)
        max2 = self.max_pool1(c2)
        x = self.fc1(nn.ReLU(max2))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
