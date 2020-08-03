# Not doing oop. Procedural programming
import os
from pathlib import Path
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model


# convert series to supervised learning
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

path = './005930_final_data/from_2017-08-03.csv'
dataset = read_csv(path, header=0, index_col=0)

values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag days
n_days = 9
n_features = 19
# frame as supervised learning
reframed = series_to_supervised(scaled, n_days, 1)
# drop columns we dont want to predict
reframed = reframed.iloc[:, :-15]
print(reframed.shape)
# print(reframed)
print(reframed.columns)

# split into train and test
values = reframed.values
n_train_rows = int(values.shape[0] * 0.7)
train = values[:n_train_rows, :]
test = values[n_train_rows:, :]

# split into input and outputs
n_obs = n_days * n_features
train_X, train_y = train[:, :n_obs], train[:, -4]
# print(train_X.shape)
# print(train_y.shape)
test_X, test_y = test[:, :n_obs], test[:, -4]
# print(test_X.shape)
# print(test_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(4))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)


model.save("multivariate_LSTM_50units_1layer")

# plot history
dirname = "005930" + '_plots'
root_dir = os.path.dirname(__file__)
plots_dir_path = os.path.join(root_dir, dirname)
Path(plots_dir_path).mkdir(parents=True, exist_ok=True)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig(os.path.join(plots_dir_path, "train_history.png"))

# load model for constant debugging without training again and again
reloaded_model = load_model("multivariate_LSTM_50units_1layer")
# Let's check
# np.testing.assert_allclose(model.predict(test_X), reloaded_model.predict(test_X))

# make a prediction
yhat = reloaded_model.predict(test_X)
print(yhat)
print(yhat.shape)
print(test_X.shape)
test_X = test_X.reshape((test_X.shape[0], n_days * n_features))
print(test_X)
print(test_X.shape)
print("=====")
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X), axis=1)
print(inv_yhat)
print(inv_yhat.shape)

for i in range(172):
    if inv_yhat[0][0] == inv_yhat[0][i]:
        print(i)
        print("hi")

# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]

"""

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
"""

