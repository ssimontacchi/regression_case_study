import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn import preprocessing
import matplotlib.pyplot as plt


def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, X_test, y_train, y_test


def scale(train, test):
    scalar = StandardScaler()
    scalar.fit(train)
    train_scaled = scalar.transform(train)
    test_scaled = scalar.transform(test)
    return train_scaled, test_scaled, scalar


def inv_scale(data, scaler):
    return scaler.inverse_transform(data)


def cv(X, y, k, model):
    kf = KFold(n_splits=k, random_state=20)
    X = np.array(X)
    y = np.array(y)
    rmsle_results = []
    rmse_results = []
    counter = 1
    for train_index, test_index in kf.split(X):
        counter += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        predicted = model(X_train, y_train, X_test)
        rmsle_results.append(rmsle(y_test, predicted))
        rmse_results.append(mean_squared_error(y_test, predicted))
    return np.mean(rmsle_results), np.mean(rmse_results)


def predict_lin_reg(X_train, y_train, X_test):
    '''
    outputs np.array of predictions
    '''
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def ridge(X_train, y_train, X_test):
    ridge = RidgeCV(alphas=(0.1, 0.5, 1.0, 2, 5, 10.0))
    ridge.fit(X_train, y_train)
    return ridge.predict(X_test)


def lasso(X_train, y_train, X_test):
    lasso = LassoCV(alphas=[0.1, 0.5, 1.0, 2, 5, 10.0])
    lasso.fit(X_train, y_train)
    return lasso.predict(X_test)


def rmsle(true, predicted):
    sum = 0.0
    for x in range(len(predicted)):
        if predicted[x] < 0 or true[x] < 0:  # check for negative values
            continue
        p = np.log(predicted[x]+1)
        t = np.log(true[x]+1)
        sum = sum + (p - t)**2
    return (sum/len(predicted))**0.5


def make_csv(X_test, prediction):
    '''
    Takes in X_test df and prediction.
    Creates df of columns: predicted SalesID and SalePrice
    Output: csv file titled 'output.csv'
    '''
    X_test = X_test['SalesID']
    X_test['SalePrice'] = prediction
    return X_test.to_csv('output.csv', index=False)


def plot_res(predictions, y_true, xaxis):
    diff = predictions - y_true
    plt.scatter(xaxis, diff)
    plt.show()

