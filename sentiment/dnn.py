"""
Description:
    stuff

"""


"""
STEPS FOR THIS RNN

Data Prep:
    collect tweets
    preprocess tweets
    convert to bag of words model representation (that will be time consuming)
        - determine number of unique words in dataset
            - depending on the format (whether binary or count), may have to normalize the inputs
        - thsi will be the number of nodes in the input layer

Dnn design:
    have to determine the number of layers and nodes per layer
    - use research articles for "inspiration"
    determine the activation functions for each layer/neuron
    determine the appropriate loss function
    what sort of regularization methods should be used (if any)


General building, training, testing of the DNN
Graphs of the outputs
    Metrics
    - error (training/val/test)

"""



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as Fuctional
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import random
import os
from os import path
from sklearn.preprocessing import normalize
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import confusion_matrix








# split the dataset into training, dev, testing
def split_data(x, y, num_splits, test_size):
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size, random_state=42)
    for traindev, test in sss.split(x, y):
        x_traindev, y_traindev = x[traindev], y[traindev]
        x_test, y_test = x[test], y[test]

    return x_traindev, y_traindev, x_test, y_test


# generate labels for the training/dev tweets using Flair
# transform training/dev/testing sets to bag of words representations
def bow_df(df: pd.DataFrame):
    uniq = set()
    for row in df.itertuples():
        uniq = set.union(uniq, row.Message.split())

    bow = pd.DataFrame(0, index=df.index, columns=uniq)

    for index, row in df.iterrows():
        for word in row.Message.split():
            bow.at[index, word] += 1

    return bow





class dset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # returns num data points
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index,:]
        y = self.y[index]


# define and instantiate DNN w/ proper architecture
class DNN(nn.Module):
    def __init_(self).__init__()

    # set base attr
    self.neurons_in = hyp.neurons_in
    self.neurons_l1 = hyp.neurons_l1
    self.neurons_l2 = hyp.neurons_l2
    self.neurons_l3 = hyp.neurons_l3
    self.neurons_out = hyp.neurons_out

    # create lin transformation objs
    self.first_fpass = nn.Linear(self.neurons_in, self.neurons_l1)
    self.second_fpass = nn.Linear(self.neurons_l1, self.neurons_l2)
    self.third_fpass = nn.Linear(self.neurons_l2, self.neurons_l3)
    self.final_fpass = nn.Linear(self.neurons_l3, self.neurons_out)

    # init weights for each layer
    std1 = 2 / np.sqrt(self.neurons_in + self.neurons_l1)
    nn.init.normal_(self.first_fpass.weight, std=std1)
    nn.init.zeros_(self.first_fpass.bias)

    std2 = 2 / np.sqrt(self.neurons_l1 + self.neurons_l2)
    nn.init.normal_(self.second_fpass.weight, std=std2)
    nn.init.zeros_(self.second_fpass.bias)

    std3 = 2 / np.sqrt(self.neurons_l2, self.neurons_l3)
    nn.init.normal_(self.third_fpass.weight, std=std3)
    nn.init.zeros_(self.third_fpass.bias)

    std4 = 2 / np.sqrt(self.neurons_l3, self.neurons_out)
    nn.init.normal_(self.final_fpass.weight, std=std4)
    nn.init.zeros_(self.final_fpass.bias)


    # define the forward pass
    def forward(self, input):
        out1 = Func.relu(self.first_fpass( input )) # output of first hidden layer
        out2 = Func.relu(self.second_fpass( out1 )) # output of second hidden layer
        out3 = Func.relu(self.third_fpass( out2 )) # output of third hidden layer
        return Func.step(self.final_fpass( out3 )) # out of output layer





if __name__ == '__main__':

    # do stuff

    # import the csv file
    print('retrieving data...')
    df = pd.read_csv('tweets.csv')
    df = pd.DataFrame(df.Message)


    # define hyperparams for the DNN
    class hyp:
        neurons_in = X.shape[1]
        neurons_l1 = 1000
        neurons_l2 = 500
        neurons_l3 = 100
        neurons_out = 1

        lr = .01 # learning rate (look at scheduler)
        num_epochs = 100
        bs = 100




    # define params for DNN
    params = {
            'batch_size':hyp.bs,
            'shuffle':True,
            'num_workers':6
            'drop_last':False,
            'pin_memory':True
            }




    # define the optimizer & loss function
    dnn = DNN()
    opt = torch.optim.SGD(dnn.parameters(), lr=hyp.lr)
    loss = nn.CrossEntropyLoss()

    # perform the training and validation
    train_avg_losses = list()
    train_accuracies = list()
    val_avg_losses = list()
    val_accuracies = list()

    for epoch in range(hyp.num_epochs):

        train_correct_count = 0
        train_sample_count = 0
        train_tot_loss = 0.0

        val_correct_count = 0
        val_sample_count = 0
        val_tot_loss = 0.0

        dnn.train(True)
        with torch.set_grad_enabled(True):
            for batch, batch_labels in train_generator:
                opt.zero_grad()

                batch = batch.float()
                batch_labels = batch_labels.float()
                batch, batch_labels = Variable(batch), Variable(batch_labels)

                # model computations
                out1 = dnn(batch)

                # loss calc
                tloss = loss(out1, batch_labels.long())
                train_tot_loss += tloss * hyp.bs

                # backprop
                tloss.backward() # calc grad
                opt.step() # update weights

                train_correct_count += torch.argmax(out1, dim=1).eq(batch_labels).sum().item()
                train_sample_count += hyp.bs

            train_avg_loss = train_tot_loss / len(train_generator.dataset)
            train_avg_losses.append(train_avg_loss)

            train_accuracy = train_correct_count / train_sample_count




    # test the architecture
    # print results
    # generate appropriate graphs
