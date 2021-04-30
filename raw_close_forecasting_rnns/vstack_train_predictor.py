# Base Class
from base_train_predictor import Base_Train_Predictor
# Data Pre-processing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# RNN
from keras.optimizers import Adam
# General Needed libraries
import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter


# Uh... lazy and didn't split params between
# Base Model and Base Train Predictor
# So both classes get the full params which is overkill.
class Vstack_Train_Predictor(Base_Train_Predictor):
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
        self.label_column_index = None
        self.sigma = params['sigma']

    # If label column is not part of x train
    # This needs to update
    def train(self, model, train_data, label_column_index=None):
        # Save train data and scaler obj because we will need it for testing
        self.train_obs = train_data['close'].values
        # plt.plot(range(0,len(self.train_obs)), self.train_obs)
        # plt.show()

        self.train_obs = self.train_obs.reshape(-1,1)
        
        # standardize data
        self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(self.train_obs)
        self.train_obs = self.scaler.transform(self.train_obs)

        # gaussian smoothing kernel
        self.train_obs = gaussian_filter(self.train_obs, sigma=self.sigma)
        # plt.plot(range(0,len(self.train_obs)), self.train_obs)
        # plt.show()

        # self.train_obs = np.log(self.train_obs)

        # Build the x as the observation from (O_i,...,O_i+d), y is O_i+d
        x_train, y_train = [], []
        for i in range(self.d, len(self.train_obs)):
            x_train.append(self.train_obs[i-self.d:i])
            y_train.append(self.train_obs[i])

        x_train, y_train = np.array(x_train), np.array(y_train)
        y_train = y_train.reshape(-1, 1)
        print('x_train shape before training: ', x_train.shape)
        print('y_train shape before training: ', y_train.shape)

        # build the model
        model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss)
        
        # train the model
        model_history = model.fit(x=x_train, 
                                  y=y_train, 
                                  epochs=self.epochs, 
                                  batch_size=self.batch_size,
                                  validation_split=0.1,
                                  verbose=1)

        return model, model_history

    def predict(self, model, test_data, label_column_index=None):
        true_vals = test_data['close'].values
        true_scaled_vals = self.scaler.transform(true_vals.reshape(-1,1))
        
        # Save train data and scaler obj because we will need it for testing
        test_obs = test_data['close'].values

        # standardize data
        test_obs = self.scaler.transform(test_obs.reshape(-1,1))
        # test_obs = np.log(test_obs)

        # gaussian smoothing kernel
        test_obs = gaussian_filter(test_obs, sigma=self.sigma)

        # Add self.d amount of days in front of test data so test_data[0] can be first prediction point
        observed = self.train_obs[-self.d:]

        preds = []

        for i in range(len(test_data)):
            pred_std_close = model.predict(observed.reshape(1,self.d,1))
            observed = np.vstack((observed,test_obs[i]))
            observed = observed[1:]

            preds.append(pred_std_close.reshape(1,))
            
            print(f'{i+1}/{len(test_data)}', end='\r', flush=True)

        return np.array(preds).flatten(), true_scaled_vals