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
class Forecasting_Train_Predictor(Base_Train_Predictor):
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
        self.scaler_out = None

    # WARN This breaks if label column is not a training column
    def train(self, model, train_data, label_column_index):
        # Save train data and scaler obj because we will need it for testing
        self.train_obs = train_data.values
        self.train_labels = self.train_obs[:,label_column_index]

        # Normalization/Standization for whole data set
        if self.normalization:
            self.scaler = MinMaxScaler(feature_range=(0,1))
            self.scaler_out = MinMaxScaler(feature_range=(0,1))
        else:
            self.scaler = StandardScaler()
            self.scaler_out = StandardScaler()
        
        # Scale observations
        self.scaler = self.scaler.fit(self.train_obs)
        x_train_scale = self.scaler.transform(self.train_obs)

        # Scale Labels - different scalers due to predict
        self.scaler_out = self.scaler_out.fit(self.train_labels.reshape(-1,1))
        y_train_scale = self.scaler_out.transform(self.train_labels.reshape(-1,1))   

        # Build the x as the observation from (O_i,...,O_i+d), y is O_i+d
        x_train, y_train = [], []
        for i in range(self.d, len(self.train_obs)):
            x_train.append(x_train_scale[i-self.d:i])
            y_train.append(y_train_scale[i])

        # Finalize data/shapes and check its holy
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
        test_scale_obs = self.scaler.transform(test_obs)
        test_label = test_obs[:, label_column_index]       

        # Build observations like in training
        x_test, labels = [], []
        for i in range(self.d, len(test_obs)):
            # Normalization/Standization for whole data set
            x_test.append(test_scale_obs[i-self.d:i])
            labels.append(test_label[i])

        x_test, labels = np.array(x_test), np.array(labels)
        labels = labels.reshape(-1, 1)

        print('x_test shape before prediction: ' , x_test.shape)
        # Predict the points
        preds = model.predict(x_test)
        # Reverse scaling
        preds = self.scaler_out.inverse_transform(preds)
        print('preds: ' , preds.shape)
        print('labels: ', labels.shape)

        return preds, labels