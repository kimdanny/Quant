# Not transforming Data into supervised learning paradigm
# --> keeping the time series format
# Firstly not doing an oop version -> TODO: will reformat to oop later on for automation and management

import os
from pathlib import Path
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
# from Data_preparation import CombineData


class MultivariateLSTM:

    def __init__(self, CombineDataObject=None, data_path=None,
                 train_ratio=0.7, validation_ratio=0.2,
                 lag_days=10, n_predict_features=4, scaler=MinMaxScaler(feature_range=(0, 1)),
                 lstm_units=64, loss_metric='mae', optimizer='adam', epochs=150, batch_size=32,
                 company_code='005930'
                 ):

        if CombineDataObject is None:
            self.company_code = company_code
        else:
            self.company_code = CombineDataObject.company_code

        if data_path is None:
            self.data = CombineDataObject.combine()
        else:
            self.data = read_csv(data_path, header=0, index_col=0)

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio

        self.lag_days = lag_days  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features.
        self.n_features = self.data.values.shape[1]  # number of features
        self.n_predict_features = n_predict_features  # how many columns(feature) you wanna predict

        self.scaler = scaler

        # LSTM param
        self.lstm_units = lstm_units
        self.optimizer = optimizer
        self.loss_metric = loss_metric
        self.epochs = epochs
        self.batch_size = batch_size

        # Path Handling for plots
        plot_dirname = self.company_code + '_plots'
        root_dir = os.path.dirname(__file__)
        self.plots_dir_path = os.path.join(root_dir, plot_dirname)

        # Path handling for model checkpoint files
        model_dirname = "model_checkpoints"
        root_dir = os.path.dirname(__file__)
        self.checkpoints_dir_path = os.path.join(root_dir, model_dirname)

    def train_val_test_split(self, data):
        n_train_rows = int(data.shape[0] * self.train_ratio)
        n_val_rows = int(data.shape[0] * self.validation_ratio)

        train = data[:n_train_rows]
        validation = data[n_train_rows:n_train_rows + n_val_rows]
        test = data[n_train_rows + n_val_rows:]

        return train, validation, test

    def split_target(self, data):
        target = data[['Open_x', 'High_x', 'Low_x', 'Close_x']]
        data_set = data.values
        target_set = target.values

        data_set_scaled = self.scaler.fit_transform(data_set)
        target_set_scaled = self.scaler.fit_transform(target_set)

        return data_set, target_set, data_set_scaled, target_set_scaled

    def split_X_y(self, dataset, targetset):
        X, y = [], []
        for i in range(self.lag_days, len(dataset)):
            X.append(dataset[i - self.lag_days:i, :])
            y.append(targetset[i, :])

        return np.array(X), np.array(y)

    def model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(LSTM(units=self.lstm_units, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add((LSTM(units=self.lstm_units)))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        model.add((Dense(units=16, activation='tanh')))
        model.add(BatchNormalization())
        model.add((Dense(units=4, activation='tanh')))
        model.compile(loss=self.loss_metric, optimizer=self.optimizer, metrics=['accuracy', 'mean_squared_error'])
        model.summary()

        return model

    def plot_history(self, history):
        Path(self.plots_dir_path).mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(20, 10))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.plot(history.history['val_accuracy'], label='validation acc')
        plt.title('Train history')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        plt.legend()
        plt.savefig('./' + self.company_code + '_plots/Multivariate_LSTM_train_history.png')

    def plot_individuals(self, target_test, pred_price):
        Path(self.plots_dir_path).mkdir(parents=True, exist_ok=True)

        for i in range(4):
            if i == 0:
                name = 'Open'
            if i == 1:
                name = 'High'
            if i == 2:
                name = 'Low'
            if i == 3:
                name = 'Close'
            plt.figure(figsize=(20, 10))
            plt.plot(target_test[i], color='green', label='Real Samsung stock')
            plt.plot(pred_price[i], color='red', label='Predicted Samsung Stock Price')
            plt.title('Stock Price Prediction')
            plt.xlabel('Trading Day')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.savefig('./' + self.company_code + '_plots/' + name + '_pred.png')

    def plot_all_in_one(self, target_test, pred_price):
        Path(self.plots_dir_path).mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(20, 10))
        plt.plot(target_test, color='green', label='Real Samsung stock')
        plt.plot(pred_price, color='red', label='Predicted Samsung Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Trading Day')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig('./' + self.company_code + '_plots/AllInOne.png')

    def run(self, load_saved_model=True, plot_history=True, plot_indiv=True, plot_allinone=True):
        train, validation, test = self.train_val_test_split(self.data)
        print("train shape: ", train.shape)
        print("vali shape: ", validation.shape)
        print("test shape: ", test.shape)

        _, _, train_scaled, train_target_scaled = self.split_target(train)
        train_X, train_y = self.split_X_y(train_scaled, train_target_scaled)

        _, _, validation_scaled, validation_target_scaled = self.split_target(validation)
        vali_X, vali_y = self.split_X_y(validation_scaled, validation_target_scaled)

        _, test_target, test_scaled, test_target_scaled = self.split_target(test)
        test_X, test_y = self.split_X_y(test_scaled, test_target_scaled)

        del _

        print("train_X, train_y shape: ", train_X.shape, train_y.shape)
        print("vali_X, vali_y shape: ", vali_X.shape, vali_y.shape)
        print("test_X, test_y shape: ", test_X.shape, test_y.shape)

        import sys
        sys.exit(0)

        if not load_saved_model:
            my_model = self.model(input_shape=(train_X.shape[1], train_X.shape[2]))

            # checkpoints path handling
            Path(self.checkpoints_dir_path).mkdir(parents=True, exist_ok=True)
            callback = ModelCheckpoint(
                filepath='./model_checkpoints/Multivariate_LSTM.h5',
                monitor='mean_squared_error',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='auto'
            )

            history = my_model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size,
                                   callbacks=[callback], validation_data=(vali_X, vali_y), shuffle=False)

        else:
            my_model = load_model('./model_checkpoints/Multivariate_LSTM.h5')

        if plot_history and not load_saved_model:
            self.plot_history(history=history)

        predicted_price = my_model.predict(test_X)
        predicted_price = self.scaler.inverse_transform(predicted_price)

        if plot_indiv:
            self.plot_individuals(test_target, pred_price=predicted_price)

        if plot_allinone:
            self.plot_all_in_one(test_target, pred_price=predicted_price)

        return predicted_price


if __name__ == '__main__':
    path = './005930_final_data/from_2017-08-10.csv'
    multi_lstm = MultivariateLSTM(data_path=path)
    multi_lstm.run(load_saved_model=False)

