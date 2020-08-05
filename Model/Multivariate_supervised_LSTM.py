import os
from pathlib import Path
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import LSTM
from keras.models import load_model
from Data_preparation import CombineData


class MultivariateSupervisedLSTM:

    def __init__(self, CombineDataObject=None, data_path=None,
                 train_ratio=0.7, validation_ratio=0.2,
                 lag_days=9, n_predict_features=4, scaler=MinMaxScaler(feature_range=(0, 1)),
                 lstm_units=50, dropout_prob=0.1, loss_metric='mae', optimizer='adam', epochs=160, batch_size=64,
                 company_code='005930'
                 ):
        """
        VERT IMPORTANT: Place predicting features at the rightmost columns

        :param CombineDataObject:
        :param data_path:
        :param train_ratio:
        :param validation_ratio:
        :param lag_days:
        :param n_predict_features:
        :param lstm_units:
        :param dropout_prob:
        :param optimizer:
        :param epochs:
        :param batch_size:
        :param company_code:
        """
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

        self.n_days = lag_days                  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features.
        self.n_features = self.data.values.shape[1]   # number of features
        self.n_predict_features = n_predict_features  # how many columns(feature) you wanna predict

        self.scaler = scaler

        # LSTM param
        self.lstm_units = lstm_units
        self.dropout_prob = dropout_prob
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

    @staticmethod
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg = agg.dropna()
        return agg

    def reframe_data(self, drop_rest_feature=True):
        """

        :param drop_rest_feature:
        :return:
        """
        values = self.data.values

        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaled = self.scaler.fit_transform(values)

        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, self.n_days, 1)

        if drop_rest_feature:
            # drop columns we dont want to predict
            # MUST!!: place predicting features at the leftmost columns
            reframed = reframed.iloc[:, :-(self.n_features - self.n_predict_features)]

        print("Reframed shape: ", reframed.shape)
        print("Reframed columns: ", reframed.columns)

        return reframed

    def train_val_test_split(self, reframed):
        values = reframed.values
        n_train_rows = int(values.shape[0] * self.train_ratio)
        n_val_rows = int(values.shape[0] * self.validation_ratio)

        train = values[:n_train_rows, :]
        validation = values[n_train_rows:n_train_rows + n_val_rows, :]
        test = values[n_train_rows + n_val_rows:, :]

        return train, validation, test

    def split_X_y(self, data):
        """
        Split into X and y
        data must be formed as a supervised learning paradigm

        :param data: Can be train, validation or test
        :return: Splitted into X and y
        """
        n_obs = self.n_days * self.n_features
        data_X, data_y = data[:, :n_obs], data[:, -self.n_predict_features:]

        return data_X, data_y

    def model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, input_shape=input_shape))
        model.add(Dropout(self.dropout_prob))

        model.add(Dense(16))
        model.add(Dense(4))

        model.compile(loss=self.loss_metric, optimizer=self.optimizer)
        model.summary()

        return model

    def run(self, load_saved_model=True, do_test=True):
        """

        :param load_saved_model:
        :param do_test:
        :return:
        """
        # Scale data and make it as supervised problem
        reframed = self.reframe_data()
        # split into train and test
        train, validation, test = self.train_val_test_split(reframed)

        # split into input and outputs
        n_days, n_features = self.n_days, self.n_features
        train_X, train_y = self.split_X_y(train)
        vali_X, vali_y = self.split_X_y(validation)
        test_X, test_y = self.split_X_y(test)

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
        vali_X = vali_X.reshape((vali_X.shape[0], n_days, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
        print(train_X.shape, train_y.shape, vali_X.shape, vali_y.shape, test_X.shape, test_y.shape)

        if not load_saved_model:  # Train model
            my_model = self.model(input_shape=(train_X.shape[1], train_X.shape[2]))

            # fit network
            history = my_model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size,
                                   validation_data=(vali_X, vali_y), verbose=2, shuffle=False)

            # Save model
            Path(self.checkpoints_dir_path).mkdir(parents=True, exist_ok=True)
            my_model.save("./model_checkpoints/multivariate_supervised_LSTM")

            # Save history plot
            Path(self.plots_dir_path).mkdir(parents=True, exist_ok=True)
            # plot history
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='validation')
            plt.legend()
            plt.savefig(os.path.join(self.plots_dir_path, "train_history.png"))

        else:  # load saved model
            my_model = load_model("./model_checkpoints/multivariate_supervised_LSTM")

        if do_test:
            # Make a prediction with Test data
            yhat = my_model.predict(test_X)
            test_X = test_X.reshape((test_X.shape[0], n_days * n_features))

            # invert scaling for forecast
            inv_yhat = concatenate((yhat, test_X[:, -(n_features - self.n_predict_features):]), axis=1)

            inv_yhat = self.scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:, :self.n_predict_features]
            print("Estimation: \n", inv_yhat)
            print("Estimation shape: ", inv_yhat.shape)

            # invert scaling for actual
            test_y = test_y.reshape((len(test_y), self.n_predict_features))
            inv_y = concatenate((test_y, test_X[:, -(n_features - self.n_predict_features):]), axis=1)
            inv_y = self.scaler.inverse_transform(inv_y)
            inv_y = inv_y[:, 0:self.n_predict_features]
            print("inverse y:\n", inv_y)
            print("inverse y shape: ", inv_y.shape)

            # calculate RMSE
            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
            print('Test RMSE: %.3f' % rmse)

            return rmse, inv_yhat, my_model

        else:
            return my_model

    """
    predict new value
    """
    def predict(self, load_saved_model=True):
        """

        :param load_saved_model: Set True if model is already trained
        :return: prediction values of time t+1
        """
        # reframe like above, but DO NOT drop rest features of time t, cuz those will be inputted into our model
        reframed = self.reframe_data(drop_rest_feature=False)
        # TODO: lag day보다 하루 작게해서 시간 t까지 전부 모델에 넣는다. 그러면 모델은 시간 t+1에 해당하는 prediction features 를 output.

        values = reframed.values
        # cut the first past time frame to keep the input shape the same
        values = values[:, self.n_features:]  # from (n_days-1) to t
        values = values.reshape((values.shape[0], self.n_days, self.n_features))

        if load_saved_model:
            my_model = load_model("./model_checkpoints/multivariate_supervised_LSTM")
        else:
            my_model = self.run(load_saved_model=False, do_test=False)

        prediction = my_model.predict(values)
        # rescaling
        values = values.reshape((values.shape[0], self.n_days * self.n_features))
        inv_yhat = concatenate((prediction, values[:, -(self.n_features - self.n_predict_features):]), axis=1)
        print(prediction.shape, values[:, -(self.n_features - self.n_predict_features):].shape)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        DataFrame(inv_yhat).to_csv('./temp.csv')
        inv_yhat = inv_yhat[:, :self.n_predict_features]

        return inv_yhat


if __name__ == '__main__':
    path = './005930_final_data/from_2017-08-04.csv'
    lstm = MultivariateSupervisedLSTM(data_path=path)
    prediction = lstm.predict(load_saved_model=True)
    print(prediction)


