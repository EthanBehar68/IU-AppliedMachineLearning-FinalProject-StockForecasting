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
        self.convs = nn.ModuleList(
                [nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_size)) for fs in filter_sizes]
        )

        self.max_pool1 = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(95*n_filters, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, n_classes, bias=True)






    def forward(self, text, text_lengths):

        # recieves text dimensions in shape of [batch_size, sentence_len]
        # after passing the embedding layer, changes shape to [batch_size, sentence_len, embedding_dims]
        # unsqueeze: adds a dimension of len 1 to the tensor
        embedded = self.embedding(text).unsqueeze(1)
        convolution = [conv(embedded) for conv in self.convs]
        # squeeze: reduces dimensions of len 1 in a tensor
        max1 = self.max_pool1(convolution[0].squeeze())
        max2 = self.max_pool1(convolution[1].squeeze())
        # cat: concatenates two tensors along a given dimension
        cat = torch.cat((max1, max2), dim=2)
        # the function is used to flatten the tensor for each sample in the batch so that the output shape will be of the following form:
        # [batch_size, (n_filters)*(n_out1/pooling_window_size + n_out2/pooling_window_size)]
        x = cat.view(cat.shape[0], -1)
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
