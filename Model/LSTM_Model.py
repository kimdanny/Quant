import math
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import time
from datetime import date
import os
from pathlib import Path

from matplotlib import pyplot as plt
from numpy.random import seed
from pylab import rcParams

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import plot_model
import sys
print(sys.path)
from Data_preparation import CombineData


class UnivariateLSTM:

    def __init__(self, CombineDataObject: CombineData, stk_path=None):

        self.company_code = CombineDataObject.company_code

        if stk_path is not None:
            # self.stk_path = stk_path
            self.data = self.load_data(stk_path=stk_path)
        else:
            self.data = CombineDataObject.combine()

        self.test_size = 0.2        # proportion of dataset to be used as test set
        self.cv_size = 0.2          # proportion of dataset to be used as cross-validation set

        self.N = 9                  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features.
        # initial value before tuning
        self.lstm_units = 50        # lstm param. initial value before tuning.
        self.dropout_prob = 1       # lstm param. initial value before tuning.
        self.optimizer = 'adam'     # lstm param. initial value before tuning.
        self.epochs = 1             # lstm param. initial value before tuning.
        self.batch_size = 1         # lstm param. initial value before tuning.

        self.model_seed = 100

        self.fontsize = 14
        self.ticklabelsize = 14

        # Path Handling for plots
        dirname = self.company_code + '_plots'
        root_dir = os.path.dirname(__file__)
        self.plots_dir_path = os.path.join(root_dir, dirname)

    @staticmethod
    def load_data(stk_path):
        """
        Use only if you load data from CSV
        :param stk_path: CSV file path
        :return: pd.DataFrame
        """
        df = pd.read_csv(stk_path, sep=',')
        # Convert Date column to datetime
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        # Change all column headings to be lower case, and remove spacing
        # TODO: What if colname is hngl?
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

        # Get month of each sample --> TODO: is this necessary?
        df['month'] = df['date'].dt.month

        # Sort by datetime
        df = df.sort_values(by='date', ascending=True)

        return df

    def plot_column_against_date(self, colname):
        """
        Not engaged in modelling. Just for plotting for EDA
        """
        rcParams['figure.figsize'] = 10, 5  # width 10, height 5

        ax = self.data.plot(x='date', y=colname, style='b-', grid=True)
        ax.set_xlabel("date")
        ax.set_ylabel("price")

        # path handling --> save in company code directory
        Path(self.plots_dir_path).mkdir(parents=True, exist_ok=True)

        plt.savefig(os.path.join(self.plots_dir_path, colname + '.png'))

    @staticmethod
    def get_mape(y_true, y_pred):
        """
        Get Mean Absolute Percentage Error
        :param y_true: dfdd
        :param y_pred:
        :return: mape
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def get_x_y(data, N, offset):
        """
        Split data into x (features) and y (target)
        :param data:
        :param N:
        :param offset:
        :return:
        """
        x, y = [], []
        for i in range(offset, len(data)):
            x.append(data[i - N:i])
            y.append(data[i])
        x = np.array(x)
        y = np.array(y)

        return x, y

    @staticmethod
    def get_x_scaled_y(data, N, offset):
        """
        Split data into x (features) and y (target)
        We scale x to have mean 0 and std dev 1, and return this.
        We do not scale y here.
        :param data:  pandas series to extract x and y
        :param N:
        :param offset:
        :return: x_scaled : features used to predict y. Scaled such that each element has mean 0 and std dev 1
                 y        : target values. Not scaled
                 mu_list  : list of the means. Same length as x_scaled and y
                 std_list : list of the std devs. Same length as x_scaled and y
        """
        x_scaled, y, mu_list, std_list = [], [], [], []
        for i in range(offset, len(data)):
            mu_list.append(np.mean(data[i - N:i]))
            std_list.append(np.std(data[i - N:i]))
            x_scaled.append((data[i - N:i] - mu_list[i - offset]) / std_list[i - offset])
            y.append(data[i])
        x_scaled = np.array(x_scaled)
        y = np.array(y)

        return x_scaled, y, mu_list, std_list

    def train_pred_eval_model(self,
                              x_train_scaled,
                              y_train_scaled,
                              x_cv_scaled,
                              y_cv,
                              mu_cv_list,
                              std_cv_list,
                              lstm_units=50,
                              dropout_prob=0.5,
                              optimizer='adam',
                              epochs=1,
                              batch_size=1):
        """
        Train model, do prediction, scale back to original range and do evaluation
        Use LSTM here.
        Returns rmse, mape and predicted values

        :param    x_train_scaled  : e.g. x_train_scaled.shape=(451, 9, 1). Here we are using the past 9 values to predict the next value
        :param    y_train_scaled  : e.g. y_train_scaled.shape=(451, 1)
        :param    x_cv_scaled     : use this to do predictions
        :param    y_cv            : actual value of the predictions
        :param    mu_cv_list      : list of the means. Same length as x_scaled and y
        :param    std_cv_list     : list of the std devs. Same length as x_scaled and y
        :param    lstm_units      : lstm param
        :param    dropout_prob    : lstm param
        :param    optimizer       : lstm param
        :param    epochs          : lstm param
        :param    batch_size      : lstm param

        :return
            rmse            : root mean square error
            mape            : mean absolute percentage error
            est             : predictions
        """

        # Create the LSTM network
        model = Sequential()
        model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train_scaled.shape[1], 1)))
        model.add(Dropout(dropout_prob))  # Add dropout with a probability of 0.5
        model.add(LSTM(units=lstm_units))
        model.add(Dropout(dropout_prob))  # Add dropout with a probability of 0.5
        model.add(Dense(1))

        # Compile and fit the LSTM network
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(x_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

        # Do prediction
        est_scaled = model.predict(x_cv_scaled)
        est = (est_scaled * np.array(std_cv_list).reshape(-1, 1)) + np.array(mu_cv_list).reshape(-1, 1)

        # Calculate RMSE and MAPE
        #     print("x_cv_scaled = " + str(x_cv_scaled))
        #     print("est_scaled = " + str(est_scaled))
        #     print("est = " + str(est))
        rmse = math.sqrt(mean_squared_error(y_cv, est))
        mape = self.get_mape(y_cv, est)

        return rmse, mape, est


if __name__  == '__main__':
    combine_data_object = CombineData('005930', years=3)
    csv_path = './005930_final_data/from_2017-07-30.csv'
    lstm = UnivariateLSTM(CombineDataObject=combine_data_object, stk_path=csv_path)
    lstm.plot_column_against_date(colname='close_x')
