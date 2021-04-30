# Base Class
from base_train_predictor import Base_Train_Predictor
# Data Pre-processing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# RNN
from keras.optimizers import RMSprop
# General Needed libraries
import pandas as pd
import numpy as np

# Uh... lazy and didn't split params between
# Base Model and Base Train Predictor
# So both classes get the full params which is overkill.
class Overfit_Train_Predictor(Base_Train_Predictor):
    def __init__(self, params):
        super().__init__(params)
        self.lr = params['lr']
        self.loss = params['loss']
        self.activation = params['activation']
        self.recurrent_activation = params['recurrent_activation']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.d = params['d']
        self.train_columns = params['train_columns']
        self.label_column = params['label_column']
        self.discretization = params['discretization']
        self.fill_method = params['fill_method']
        self.normalization = params['normalization']
        self.scaler = None

    # WARN This breaks if label column is not a training column
    def train(self, model, train_data, label_column_index):
        # Save train data and scaler obj because we will need it for testing
        self.train_obs = train_data.values

        # Normalization/Standization for whole data set
        if self.normalization:
            self.scaler = MinMaxScaler(feature_range=(0,1))
        else:
            self.scaler = StandardScaler()
        x_train_scale = self.scaler.fit_transform(self.train_obs)
        y_train_scale = x_train_scale[:, label_column_index]        

        # Build the x as the observation from (O_i,...,O_i+d), y is O_i+d
        x_train, y_train = [], []
        for i in range(self.d, len(self.train_obs)):
            x_train.append(x_train_scale[i-self.d:i])
            y_train.append(y_train_scale[i])

        x_train, y_train = np.array(x_train), np.array(y_train)
        y_train = y_train.reshape(-1, 1)
        print('x_train shape before training: ', x_train.shape)
        print('y_train shape before training: ', y_train.shape)

        # build the model
        model.compile(optimizer=RMSprop(learning_rate=self.lr), loss=self.loss)
        
        # train the model
        model_history = model.fit(x=x_train, 
                        y=y_train, 
                        epochs=self.epochs, 
                        batch_size=self.batch_size,
                        validation_split=0.1,
                        verbose=1)
        return model, model_history

    def predict(self, model, test_data, label_column_index):
        # Add self.d amount of days in front of test data so test_data[0] can be first prediction point
        test_obs = np.concatenate((self.train_obs[-self.d:], test_data.values), axis=0)

        # Normalization/Standization for whole data set
        scale_train_obs = self.scaler.transform(self.train_obs)
        
        # Predict using model's predictions
        observed = scale_train_obs[-self.d:]
        preds, labels = [], []
        for i in range(len(test_data)):
            prediction = model.predict(observed.reshape(1, self.d, len(self.train_columns)))
            labels.append(scale_train_obs[i, label_column_index])
            observed = np.vstack((observed, prediction))
            observed = observed[1:]
            preds.append(prediction)

        preds, labels = np.array(preds).reshape(-1, 1), np.array(labels).reshape(-1, 1)

        print(preds.shape)
        print(labels.shape)

        return preds, labels