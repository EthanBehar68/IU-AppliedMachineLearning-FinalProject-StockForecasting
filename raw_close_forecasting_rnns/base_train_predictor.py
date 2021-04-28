# Abstract functionality
from abc import ABC, abstractmethod

# Base model class
class Base_Train_Predictor(ABC):

    # params should be a dict of your parameters that you want to pass to the model
    # name should be a string (used for saving results)
    # params dict *must* include 'name':name within it 
    def __init__(self, params):
        self.lr = params['lr']
        
     # training function for the model, should create the model, train it, and store in self.model
    @abstractmethod
    def train(self, model, train_data, label_column_index):
        pass
    
    # prediction function for the model, should return the preds and y_true given the test data
    @abstractmethod
    def predict(self, model, test_data, label_column_index):
        # return preds,actual
        pass