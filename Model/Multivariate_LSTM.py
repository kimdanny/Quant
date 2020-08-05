# Not transforming Data into supervised learning paradigm
# --> keeping the time series format
# Firstly not doing an oop version -> TODO: will reformat to oop later on for automation and management

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from matplotlib import pyplot as plt


path = './005930_final_data/from_2017-08-04.csv'
data = pd.read_csv(path, header=0, index_col=0)
print(data.shape)

train_until = int(len(data)*0.7)
train = data[:train_until]
print(train.shape)

scaler = MinMaxScaler(feature_range = (0, 1))
target = train[['Open_x', 'High_x', 'Low_x', 'Close_x']]
target_set = target.values
train_set = train.values
training_set_scaled = scaler.fit_transform(train_set)
target_set_scaled = scaler.fit_transform(target_set)

X_train = []
y_train = []
for i in range(50, len(train_set)):
    X_train.append(training_set_scaled[i - 50:i, :])
    y_train.append(target_set_scaled[i, :])

X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape)  # (462, 50, 19)
print(y_train.shape)  # (462, 4)


def model():
    mod = Sequential()
    mod.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    mod.add(Dropout(0.2))
    mod.add(BatchNormalization())
    mod.add(LSTM(units=64, return_sequences=True))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add((LSTM(units=64)))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())
    mod.add((Dense(units=16, activation='tanh')))
    mod.add(BatchNormalization())
    mod.add((Dense(units=4, activation='tanh')))
    mod.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mean_squared_error'])
    mod.summary()

    return mod


my_model = model()

callback = ModelCheckpoint(filepath='./model_checkpoints/Multivariate_LSTM.h5',
                           monitor='mean_squared_error',
                           verbose=0,
                           save_best_only=True,
                           save_weights_only=False,
                           mode='auto',
                           )

my_model.fit(X_train, y_train, epochs=300, batch_size=32, callbacks=[callback])

test = data[train_until:]
test_target = test[['Open_x', 'High_x', 'Low_x', 'Close_x']]
target_set_test = test_target.values
test_set = test.values
test_set_scaled = scaler.fit_transform(test_set)
target_set_scaled = scaler.fit_transform(target_set_test)

X_test = []
y_test = []
for i in range(50, len(test_set)):
    X_test.append(test_set_scaled[i - 50:i, :])
    y_test.append(target_set_scaled[i, :])

X_test, y_test = np.array(X_test), np.array(y_test)
predicted_stock_price = my_model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Plot
# Open
for i in range(4):
    if i == 0:
        name = 'Open'
    if i == 1:
        name = 'High'
    if i == 2:
        name = 'Low'
    if i == 3:
        name = 'Close'
    plt.figure(figsize=(20,10))
    plt.plot(target_set_test[i], color='green', label='Real Samsung stock')
    plt.plot(predicted_stock_price[i], color='red', label='Predicted Samsung Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Trading Day')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('./005930_plots/'+ name + '_pred.png')


# All in one
plt.figure(figsize=(20,10))
plt.plot(target_set_test, color = 'green', label='Real Samsung stock')
plt.plot(predicted_stock_price, color = 'red', label='Predicted Samsung Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('./005930_plots/AllInOne.png')