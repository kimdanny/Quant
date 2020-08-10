"""
Investigating importance of each feature.
Label is 'Close Price'
"""
from pandas import read_csv
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

path = './005930_final_data/from_2017-08-10.csv'
data = read_csv(path)

print('Total dataset has {} samples, and {} features.'.format(data.shape[0], data.shape[1]))


def get_feature_importance_data(data_income):
    data = data_income.copy()
    # y = data[['Open_x', 'High_x', 'Low_x', 'Close_x']]
    # X = data.iloc[:, 5:]

    y = data['Close_x']
    X = data.iloc[:, 5:]

    train_samples = int(X.shape[0] * 0.7)

    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (X_train, y_train), (X_test, y_test)


# Get training and test data
(X_train, y_train), (X_test, y_test) = get_feature_importance_data(data)

regressor = XGBRegressor(gamma=0.0, n_estimators=200, learning_rate=0.05)
xgbModel = regressor.fit(X_train, y_train,
                         eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))


#########################
# Train Validation Plot #
#########################
plt.scatter(x=training_rounds, y=eval_result['validation_0']['rmse'], label='Training Error')
plt.scatter(x=training_rounds, y=eval_result['validation_1']['rmse'], label='Validation Error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Training Vs. Validation Error')
plt.legend()
plt.savefig('./FE_train_val_history.png')


######################
# Feature Importance #
######################
# TODO: Korean Encoding Problem on plot
fig = plt.figure(figsize=(14, 8))
plt.xticks(rotation='vertical')

features_num = [i for i in range(len(xgbModel.feature_importances_))]
importance = xgbModel.feature_importances_.tolist()
feature_names = X_test.columns

# Sorting
matched = zip(features_num, feature_names, importance)
sorted_importance = sorted(matched, key=lambda x: x[2])[::-1]       # sort by importance

sorted_features_order = list(map(lambda x: x[0], sorted_importance))
sorted_features_names = list(map(lambda x: x[1], sorted_importance))
sorted_features_importance = list(map(lambda x: x[2], sorted_importance))


plt.bar(features_num, sorted_features_importance, tick_label=sorted_features_names)
plt.title('Feature Importance')
plt.savefig('./Feature_Importance.png')





