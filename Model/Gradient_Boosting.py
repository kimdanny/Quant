# Gradient Boosting --> XGBoost Regression and Light GBM Regression
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from matplotlib import pyplot

# TODO: Fit input shpae, Automate process and Class-iztion

n_predict_features = 4
n_days = 5
train_ratio = 0.8

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one step prediction
def gradient_boosting_forecast(train, testX, model='xgboost'):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    if model=='xgboost':
        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    elif model=='lightgbm':
        model = LGBMRegressor(n_estimators=1000)

    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, model='xgboost'):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = gradient_boosting_forecast(history, testX, model=model)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, 1], predictions)
    return error, test[:, 1], predictions


path = './005930_final_data/from_2017-08-04.csv'

# load the dataset
series = read_csv(path, header=0, index_col=0)
values = series.values
print(values.shape)
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=5)
print(data)
print(data.shape)
# evaluate
mae, y, yhat = walk_forward_validation(data, n_test=20, model='xgboost')
print('MAE: %.3f' % mae)
# plot expected vs preducted
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()


# # load the dataset
# series = read_csv(path, header=0, index_col=0)
# values = series.values
# # transform the time series data into supervised learning
# train = series_to_supervised(values, n_in=3)
# # split into input and output columns
# trainX, trainy = train[:, :-1], train[:, -1]
# # fit model
# model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
# model.fit(trainX, trainy)
# # construct an input for a new preduction
# row = values[-3:].flatten()
# # make a one-step prediction
# yhat = model.predict(asarray([row]))
# print('Input: %s, Predicted: %.3f' % (row, yhat[0]))