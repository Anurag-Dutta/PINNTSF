# -*- coding: utf-8 -*-
"""
`2023-05-26 08:21:54`

Code Description: Time Series Forecast using PINNTSF (Physics Informed Neural Network for Time Series Forecasting)

Authors: Anurag Dutta (anuragdutta.research@gmail.com || 1anuragdutta@gmail.com) && Tanujit Chakraborty (tanujit.chakraborty@sorbonne.ae || tanujitisi@gmail.com)

"""

"""
## Gathering Dependencies


_Importing Required Libraries_
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras
import tensorflow as tf
from hampel import hampel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from matplotlib import pyplot
from numpy import array
import os
import math
import requests


def van_der_pol(length: int, freq: int, col_imp: int, data_path: str, compar: bool):
    """## Pretraining

    _Van der Pol intermittancy_ is the Simulated Data

    The `van_der_pol_intermittency.dat` feeds the model with the dynamics of the Van der Pol Oscillator
    """

    data = np.genfromtxt('datasets/van_der_pol_intermittency.dat')
    training_set = pd.DataFrame(data).reset_index(drop=True)
    training_set = training_set.iloc[:, 1]

    """## Computing the Gradients

    _Calculating the value of_ $\frac{dx}{dt}$, _and_ $\frac{d^2x}{dt^2}$
    """

    t_diff = freq
    # print(training_set.max())
    gradient_t = (training_set.diff() / t_diff).iloc[1:]  # dx/dt
    # print(gradient_t)
    gradient_tt = (gradient_t.diff() / t_diff).iloc[1:]  # d2x/dt2
    # print(gradient_tt)

    """## Loading Datasets

    """

    data = pd.read_csv(str(data_path))
    training_set = data.iloc[:, col_imp]
    # training_set

    training_set = training_set.head(training_set.shape[0])
    # training_set

    training_set = training_set.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of training_set as index
    gradient_t = gradient_t.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_t as index
    gradient_tt = gradient_tt.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_tt as index

    df = pd.concat((training_set[:-1], gradient_t), axis=1)
    gradient_tt.columns = ["grad_tt"]
    df = pd.concat((df[:-1], gradient_tt), axis=1)
    df.columns = ['y_t', 'grad_t', 'grad_tt']

    """## Preprocessing the data into supervised learning"""

    # split a sequence into samples
    def Supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n_in, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n_out)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    data = Supervised(df.values, n_in=int(length * 3.5), n_out=length)

    for iter in range(1, length + 1):
        data.drop(['var2(t-' + str(iter) + ')', 'var3(t-' + str(iter) + ')'], axis=1, inplace=True)
    # print(data.head())
    # print(data.columns)
    #
    # data.shape
    #
    # data[0:len(data)-1].shape
    #
    # data.tail(1).shape

    train_1 = np.array(data[0:len(data) - 1])
    test_1 = np.array(data.tail(1))

    scaler = MinMaxScaler(feature_range=(0, 1))  # Transform features by scaling each feature to a given range
    train = scaler.fit_transform(
        train_1)  # Fits transformer to 'train_1' and returns a transformed version of 'train_1'.
    forecast = scaler.transform(test_1)

    trainy = train[:, -int(length * 3.5):]
    trainX = train[:, :-int(length * 3.5)]

    forecasty = forecast[:, -int(length * 3.5):]
    forecastX = forecast[:, :-int(length * 3.5)]

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    forecastX = forecastX.reshape((forecastX.shape[0], 1, forecastX.shape[1]))
    # print(trainX.shape, trainy.shape, forecastX.shape)

    """## Model (Without Monte Carlo Dropout)"""

    mu = tf.Variable(4, name="mu", trainable=True, dtype=tf.float32)
    splitr = 0.5

    def loss_fn(y_true, y_pred):
        squared_difference = tf.square(y_true[:, 0] - y_pred[:, 0])
        squared_difference2 = tf.square(y_true[:, 2] - y_pred[:, 2])
        squared_difference1 = tf.square(y_true[:, 1] - y_pred[:, 1])
        squared_difference3 = tf.square(
            y_pred[:, 2] - mu * (y_pred[:, 1] - (y_pred[:, 0] ** 2 * y_pred[:, 1]) - (1 / mu) * y_pred[:, 0]))
        return tf.reduce_mean(squared_difference, axis=-1) + 0.2 * tf.reduce_mean(squared_difference3, axis=-1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(length * 3.5)))
    model.compile(loss=loss_fn, optimizer='adam', metrics=["mae", "mse"])
    history = model.fit(trainX[:int(splitr * trainX.shape[0])], trainy[:int(splitr * trainX.shape[0])], epochs=500,
                        batch_size=64, validation_data=(
        trainX[int(splitr * trainX.shape[0]):trainX.shape[0]], trainy[int(splitr * trainX.shape[0]):trainX.shape[0]]),
                        shuffle=False)

    """## Prediction (Without Monte Carlo Dropout)"""

    forecast_without_mc = forecastX
    yhat_without_mc = model.predict(forecast_without_mc)  # Step Ahead Prediction ('length' 'freq')
    forecast_without_mc = forecast_without_mc.reshape(
        (forecast_without_mc.shape[0], forecast_without_mc.shape[2]))  # Historical Input

    # forecastX.shape
    #
    # forecast_without_mc.shape

    inv_yhat_without_mc = np.concatenate((forecast_without_mc, yhat_without_mc),
                                         axis=1)  # Concatenation of predicted values with Historical Data
    inv_yhat_without_mc = scaler.inverse_transform(inv_yhat_without_mc)  # Transform labels back to original encoding

    # inv_yhat_without_mc.shape

    # inv_yhat_without_mc

    # inv_yhat_without_mc[:,-int(length*3.0):].shape

    fforecast = inv_yhat_without_mc[:, -int(length * 3.0):]

    # fforecast

    case_forecast = fforecast[:, 0:int((length * 3.0) - 1):3]

    # code to replace all negative value with 0
    case_forecast[case_forecast < 0] = 0

    case_fforecast = np.around(case_forecast)

    print(np.array(case_fforecast))

    if compar:
        training_mae = history.history['val_mae']
        training_mse = history.history['val_mse']
        compar_true(col_imp, case_fforecast, training_mae, training_mse)


def lienard(length: int, freq: int, col_imp: int, data_path: str, compar: bool):
    """## Pretraining

    _Lienard intermittancy_ is the Simulated Data

    The `lienard_intermittency.dat` feeds the model with the dynamics of the Lienard System
    """

    data = np.genfromtxt('datasets/lienard_intermittency.dat')
    training_set = pd.DataFrame(data).reset_index(drop=True)
    training_set = training_set.iloc[:, 1]

    """## Computing the Gradients

    _Calculating the value of_ $\frac{dx}{dt}$, _and_ $\frac{d^2x}{dt^2}$
    """

    t_diff = freq
    # print(training_set.max())
    gradient_t = (training_set.diff() / t_diff).iloc[1:]  # dx/dt
    # print(gradient_t)
    gradient_tt = (gradient_t.diff() / t_diff).iloc[1:]  # d2x/dt2
    # print(gradient_tt)

    """## Loading Datasets

    """

    data = pd.read_csv(str(data_path))
    training_set = data.iloc[:, col_imp]
    # training_set

    training_set = training_set.head(training_set.shape[0])
    # training_set

    training_set = training_set.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of training_set as index
    gradient_t = gradient_t.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_t as index
    gradient_tt = gradient_tt.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_tt as index

    df = pd.concat((training_set[:-1], gradient_t), axis=1)
    gradient_tt.columns = ["grad_tt"]
    df = pd.concat((df[:-1], gradient_tt), axis=1)
    df.columns = ['y_t', 'grad_t', 'grad_tt']

    """## Preprocessing the data into supervised learning"""

    # split a sequence into samples
    def Supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n_in, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n_out)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    data = Supervised(df.values, n_in=int(length * 3.5), n_out=length)

    for iter in range(1, length + 1):
        data.drop(['var2(t-' + str(iter) + ')', 'var3(t-' + str(iter) + ')'], axis=1, inplace=True)
    # print(data.head())
    # print(data.columns)
    #
    # data.shape
    #
    # data[0:len(data)-1].shape
    #
    # data.tail(1).shape

    train_1 = np.array(data[0:len(data) - 1])
    test_1 = np.array(data.tail(1))

    scaler = MinMaxScaler(feature_range=(0, 1))  # Transform features by scaling each feature to a given range
    train = scaler.fit_transform(
        train_1)  # Fits transformer to 'train_1' and returns a transformed version of 'train_1'.
    forecast = scaler.transform(test_1)

    trainy = train[:, -int(length * 3.5):]
    trainX = train[:, :-int(length * 3.5)]

    forecasty = forecast[:, -int(length * 3.5):]
    forecastX = forecast[:, :-int(length * 3.5)]

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    forecastX = forecastX.reshape((forecastX.shape[0], 1, forecastX.shape[1]))
    # print(trainX.shape, trainy.shape, forecastX.shape)

    """## Model (Without Monte Carlo Dropout)"""

    a = tf.Variable(0.45, name="a", trainable=True, dtype=tf.float32)
    b = tf.Variable(0.5, name="b", trainable=True, dtype=tf.float32)
    c = tf.Variable(-0.5, name="c", trainable=True, dtype=tf.float32)
    splitr = 0.5

    def loss_fn(y_true, y_pred):
        squared_difference = tf.square(y_true[:, 0] - y_pred[:, 0])
        squared_difference2 = tf.square(y_true[:, 2] - y_pred[:, 2])
        squared_difference1 = tf.square(y_true[:, 1] - y_pred[:, 1])
        squared_difference3 = tf.square(
            y_pred[:, 2] + a * y_pred[:, 0] * y_pred[:, 1] + c * y_pred[:, 0] + b * y_pred[:, 0] ** 3)
        return tf.reduce_mean(squared_difference, axis=-1) + 0.2 * tf.reduce_mean(squared_difference3, axis=-1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(length * 3.5)))
    model.compile(loss=loss_fn, optimizer='adam', metrics=["mae", "mse"])
    history = model.fit(trainX[:int(splitr * trainX.shape[0])], trainy[:int(splitr * trainX.shape[0])], epochs=500,
                        batch_size=64, validation_data=(
            trainX[int(splitr * trainX.shape[0]):trainX.shape[0]],
            trainy[int(splitr * trainX.shape[0]):trainX.shape[0]]),
                        shuffle=False)

    """## Prediction (Without Monte Carlo Dropout)"""

    forecast_without_mc = forecastX
    yhat_without_mc = model.predict(forecast_without_mc)  # Step Ahead Prediction ('length' 'freq')
    forecast_without_mc = forecast_without_mc.reshape(
        (forecast_without_mc.shape[0], forecast_without_mc.shape[2]))  # Historical Input

    # forecastX.shape
    #
    # forecast_without_mc.shape

    inv_yhat_without_mc = np.concatenate((forecast_without_mc, yhat_without_mc),
                                         axis=1)  # Concatenation of predicted values with Historical Data
    inv_yhat_without_mc = scaler.inverse_transform(inv_yhat_without_mc)  # Transform labels back to original encoding

    # inv_yhat_without_mc.shape

    # inv_yhat_without_mc

    # inv_yhat_without_mc[:,-int(length*3.0):].shape

    fforecast = inv_yhat_without_mc[:, -int(length * 3.0):]

    # fforecast

    case_forecast = fforecast[:, 0:int((length * 3.0) - 1):3]

    # code to replace all negative value with 0
    case_forecast[case_forecast < 0] = 0

    case_fforecast = np.around(case_forecast)

    print(np.array(case_fforecast))

    if compar:
        training_mae = history.history['val_mae']
        training_mse = history.history['val_mse']
        compar_true(col_imp, case_fforecast, training_mae, training_mse)


def lorenz(length: int, freq: int, col_imp: int, data_path: str, compar: bool):
    """## Pretraining

    _Lorenz Intermittency_ is the Simulated Data

    The `lorenz_intermittency.dat` feeds the model with the dynamics of the Lorenz Attractor
    """

    data = np.genfromtxt('datasets/lorenz_intermittency.dat')
    training_set = pd.DataFrame(data).reset_index(drop=True)
    training_set = training_set.iloc[:, 1]

    """## Computing the Gradients

    _Calculating the value of_ $\frac{dx}{dt}$, _and_ $\frac{d^2x}{dt^2}$
    """

    t_diff = freq
    # print(training_set.max())
    gradient_t = (training_set.diff() / t_diff).iloc[1:]  # dx/dt
    # print(gradient_t)
    gradient_tt = (gradient_t.diff() / t_diff).iloc[1:]  # d2x/dt2
    # print(gradient_tt)

    """## Loading Datasets

    """

    data = pd.read_csv(str(data_path))
    training_set = data.iloc[:, col_imp]
    # training_set

    training_set = training_set.head(training_set.shape[0])
    # training_set

    training_set = training_set.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of training_set as index
    gradient_t = gradient_t.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_t as index
    gradient_tt = gradient_tt.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_tt as index

    df = pd.concat((training_set[:-1], gradient_t), axis=1)
    gradient_tt.columns = ["grad_tt"]
    df = pd.concat((df[:-1], gradient_tt), axis=1)
    df.columns = ['y_t', 'grad_t', 'grad_tt']

    """## Preprocessing the data into supervised learning"""

    # split a sequence into samples
    def Supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n_in, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n_out)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    data = Supervised(df.values, n_in=int(length * 3.5), n_out=length)

    for iter in range(1, length + 1):
        data.drop(['var2(t-' + str(iter) + ')', 'var3(t-' + str(iter) + ')'], axis=1, inplace=True)
    # print(data.head())
    # print(data.columns)
    #
    # data.shape
    #
    # data[0:len(data)-1].shape
    #
    # data.tail(1).shape

    train_1 = np.array(data[0:len(data) - 1])
    test_1 = np.array(data.tail(1))

    scaler = MinMaxScaler(feature_range=(0, 1))  # Transform features by scaling each feature to a given range
    train = scaler.fit_transform(
        train_1)  # Fits transformer to 'train_1' and returns a transformed version of 'train_1'.
    forecast = scaler.transform(test_1)

    trainy = train[:, -int(length * 3.5):]
    trainX = train[:, :-int(length * 3.5)]

    forecasty = forecast[:, -int(length * 3.5):]
    forecastX = forecast[:, :-int(length * 3.5)]

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    forecastX = forecastX.reshape((forecastX.shape[0], 1, forecastX.shape[1]))
    # print(trainX.shape, trainy.shape, forecastX.shape)

    """## Model (Without Monte Carlo Dropout)"""

    s = tf.Variable(10, name="sigma", trainable=True, dtype=tf.float32)
    r = tf.Variable(28, name="rhow", trainable=True, dtype=tf.float32)
    splitr = 0.5

    def loss_fn(y_true, y_pred):
        squared_difference = tf.square(y_true[:, 0] - y_pred[:, 0])
        squared_difference2 = tf.square(y_true[:, 2] - y_pred[:, 2])
        squared_difference1 = tf.square(y_true[:, 1] - y_pred[:, 1])
        squared_difference3 = tf.square((y_pred[:, 2] + y_pred[:, 1] * (1 + s) - y_pred[:, 0] * s * (r - 1)))
        return tf.reduce_mean(squared_difference, axis=-1) + 0.2 * tf.reduce_mean(squared_difference3, axis=-1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(length * 3.5)))
    model.compile(loss=loss_fn, optimizer='adam', metrics=["mae", "mse"])
    history = model.fit(trainX[:int(splitr * trainX.shape[0])], trainy[:int(splitr * trainX.shape[0])], epochs=500,
                        batch_size=64, validation_data=(
            trainX[int(splitr * trainX.shape[0]):trainX.shape[0]],
            trainy[int(splitr * trainX.shape[0]):trainX.shape[0]]),
                        shuffle=False)

    """## Prediction (Without Monte Carlo Dropout)"""

    forecast_without_mc = forecastX
    yhat_without_mc = model.predict(forecast_without_mc)  # Step Ahead Prediction ('length' 'freq')
    forecast_without_mc = forecast_without_mc.reshape(
        (forecast_without_mc.shape[0], forecast_without_mc.shape[2]))  # Historical Input

    # forecastX.shape
    #
    # forecast_without_mc.shape

    inv_yhat_without_mc = np.concatenate((forecast_without_mc, yhat_without_mc),
                                         axis=1)  # Concatenation of predicted values with Historical Data
    inv_yhat_without_mc = scaler.inverse_transform(inv_yhat_without_mc)  # Transform labels back to original encoding

    # inv_yhat_without_mc.shape

    # inv_yhat_without_mc

    # inv_yhat_without_mc[:,-int(length*3.0):].shape

    fforecast = inv_yhat_without_mc[:, -int(length * 3.0):]

    # fforecast

    case_forecast = fforecast[:, 0:int((length * 3.0) - 1):3]

    # code to replace all negative value with 0
    case_forecast[case_forecast < 0] = 0

    case_fforecast = np.around(case_forecast)

    print(np.array(case_fforecast))

    if compar:
        training_mae = history.history['val_mae']
        training_mse = history.history['val_mse']
        compar_true(col_imp, case_fforecast, training_mae, training_mse)


def lotka_volterra(length: int, freq: int, col_imp: int, data_path: str, compar: bool):
    """## Pretraining

    _Lotka Volterra Intermittency_ is the Simulated Data

    The `lotka_volterra_intermittency.dat` feeds the model with the dynamics of the Lotka Volterra Equations
    """

    data = np.genfromtxt('datasets/lotka_volterra_intermittency.dat')
    training_set = pd.DataFrame(data).reset_index(drop=True)
    training_set = training_set.iloc[:, 1]

    """## Computing the Gradients

    _Calculating the value of_ $\frac{dx}{dt}$, _and_ $\frac{d^2x}{dt^2}$
    """

    t_diff = freq
    # print(training_set.max())
    gradient_t = (training_set.diff() / t_diff).iloc[1:]  # dx/dt
    # print(gradient_t)
    gradient_tt = (gradient_t.diff() / t_diff).iloc[1:]  # d2x/dt2
    # print(gradient_tt)

    """## Loading Datasets

    """

    data = pd.read_csv(str(data_path))
    training_set = data.iloc[:, col_imp]
    # training_set

    training_set = training_set.head(training_set.shape[0])
    # training_set

    training_set = training_set.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of training_set as index
    gradient_t = gradient_t.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_t as index
    gradient_tt = gradient_tt.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_tt as index

    df = pd.concat((training_set[:-1], gradient_t), axis=1)
    gradient_tt.columns = ["grad_tt"]
    df = pd.concat((df[:-1], gradient_tt), axis=1)
    df.columns = ['y_t', 'grad_t', 'grad_tt']

    """## Preprocessing the data into supervised learning"""

    # split a sequence into samples
    def Supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n_in, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n_out)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    data = Supervised(df.values, n_in=int(length * 3.5), n_out=length)

    for iter in range(1, length + 1):
        data.drop(['var2(t-' + str(iter) + ')', 'var3(t-' + str(iter) + ')'], axis=1, inplace=True)
    # print(data.head())
    # print(data.columns)
    #
    # data.shape
    #
    # data[0:len(data)-1].shape
    #
    # data.tail(1).shape

    train_1 = np.array(data[0:len(data) - 1])
    test_1 = np.array(data.tail(1))

    scaler = MinMaxScaler(feature_range=(0, 1))  # Transform features by scaling each feature to a given range
    train = scaler.fit_transform(
        train_1)  # Fits transformer to 'train_1' and returns a transformed version of 'train_1'.
    forecast = scaler.transform(test_1)

    trainy = train[:, -int(length * 3.5):]
    trainX = train[:, :-int(length * 3.5)]

    forecasty = forecast[:, -int(length * 3.5):]
    forecastX = forecast[:, :-int(length * 3.5)]

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    forecastX = forecastX.reshape((forecastX.shape[0], 1, forecastX.shape[1]))
    # print(trainX.shape, trainy.shape, forecastX.shape)

    """## Model (Without Monte Carlo Dropout)"""

    a = tf.Variable(0.1, name="alpha", trainable=True, dtype=tf.float32)
    b = tf.Variable(0.05, name="beta", trainable=True, dtype=tf.float32)
    c = tf.Variable(1.1, name="gamma", trainable=True, dtype=tf.float32)
    d = tf.Variable(0.1, name="delta", trainable=True, dtype=tf.float32)
    splitr = 0.5

    def loss_fn(y_true, y_pred):
        squared_difference = tf.square(y_true[:, 0] - y_pred[:, 0])
        squared_difference2 = tf.square(y_true[:, 2] - y_pred[:, 2])
        squared_difference1 = tf.square(y_true[:, 1] - y_pred[:, 1])
        squared_difference3 = tf.square(y_pred[:, 2] - a * a * y_pred[:, 0] + 2 * a * b * y_pred[:, 0] * (
                (a * y_pred[:, 0] - y_pred[:, 1]) / (b * y_pred[:, 0])) - b * b * y_pred[:, 0] * (
                                                (a * y_pred[:, 0] - y_pred[:, 1]) / (b * y_pred[:, 0])) * (
                                                (a * y_pred[:, 0] - y_pred[:, 1]) / (b * y_pred[:, 0])) - b * d * (
                                                y_pred[:, 0] ** 2) * ((a * y_pred[:, 0] - y_pred[:, 1]) / (
                b * y_pred[:, 0])) - c * b * y_pred[:, 0] * (
                                                (a * y_pred[:, 0] - y_pred[:, 1]) / (b * y_pred[:, 0])))
        return tf.reduce_mean(squared_difference, axis=-1) + 0.2 * tf.reduce_mean(squared_difference3, axis=-1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(length * 3.5)))
    model.compile(loss=loss_fn, optimizer='adam', metrics=["mae", "mse"])
    history = model.fit(trainX[:int(splitr * trainX.shape[0])], trainy[:int(splitr * trainX.shape[0])], epochs=500,
                        batch_size=64, validation_data=(
            trainX[int(splitr * trainX.shape[0]):trainX.shape[0]],
            trainy[int(splitr * trainX.shape[0]):trainX.shape[0]]),
                        shuffle=False)

    """## Prediction (Without Monte Carlo Dropout)"""

    forecast_without_mc = forecastX
    yhat_without_mc = model.predict(forecast_without_mc)  # Step Ahead Prediction ('length' 'freq')
    forecast_without_mc = forecast_without_mc.reshape(
        (forecast_without_mc.shape[0], forecast_without_mc.shape[2]))  # Historical Input

    # forecastX.shape
    #
    # forecast_without_mc.shape

    inv_yhat_without_mc = np.concatenate((forecast_without_mc, yhat_without_mc),
                                         axis=1)  # Concatenation of predicted values with Historical Data
    inv_yhat_without_mc = scaler.inverse_transform(inv_yhat_without_mc)  # Transform labels back to original encoding

    # inv_yhat_without_mc.shape

    # inv_yhat_without_mc

    # inv_yhat_without_mc[:,-int(length*3.0):].shape

    fforecast = inv_yhat_without_mc[:, -int(length * 3.0):]

    # fforecast

    case_forecast = fforecast[:, 0:int((length * 3.0) - 1):3]

    # code to replace all negative value with 0
    case_forecast[case_forecast < 0] = 0

    case_fforecast = np.around(case_forecast)

    print(np.array(case_fforecast))

    if compar:
        training_mae = history.history['val_mae']
        training_mse = history.history['val_mse']
        compar_true(col_imp, case_fforecast, training_mae, training_mse)


def duffing(length: int, freq: int, col_imp: int, data_path: str, compar: bool):
    """## Pretraining

    _Duffing Intermittency_ is the Simulated Data

    The `duffing_intermittency.dat` feeds the model with the dynamics of the Duffing Equations
    """

    data = np.genfromtxt('datasets/duffing_intermittency.dat')
    training_set = pd.DataFrame(data).reset_index(drop=True)
    training_set = training_set.iloc[:, 1]

    """## Computing the Gradients

    _Calculating the value of_ $\frac{dx}{dt}$, _and_ $\frac{d^2x}{dt^2}$
    """

    t_diff = freq
    # print(training_set.max())
    gradient_t = (training_set.diff() / t_diff).iloc[1:]  # dx/dt
    # print(gradient_t)
    gradient_tt = (gradient_t.diff() / t_diff).iloc[1:]  # d2x/dt2
    # print(gradient_tt)

    """## Loading Datasets

    """

    data = pd.read_csv(str(data_path))
    training_set = data.iloc[:, col_imp]
    # training_set

    training_set = training_set.head(training_set.shape[0])
    # training_set

    training_set = training_set.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of training_set as index
    gradient_t = gradient_t.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_t as index
    gradient_tt = gradient_tt.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_tt as index

    df = pd.concat((training_set[:-1], gradient_t), axis=1)
    gradient_tt.columns = ["grad_tt"]
    df = pd.concat((df[:-1], gradient_tt), axis=1)
    df.columns = ['y_t', 'grad_t', 'grad_tt']

    """## Preprocessing the data into supervised learning"""

    # split a sequence into samples
    def Supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n_in, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n_out)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    data = Supervised(df.values, n_in=int(length * 3.5), n_out=length)

    for iter in range(1, length + 1):
        data.drop(['var2(t-' + str(iter) + ')', 'var3(t-' + str(iter) + ')'], axis=1, inplace=True)
    # print(data.head())
    # print(data.columns)
    #
    # data.shape
    #
    # data[0:len(data)-1].shape
    #
    # data.tail(1).shape

    train_1 = np.array(data[0:len(data) - 1])
    test_1 = np.array(data.tail(1))

    scaler = MinMaxScaler(feature_range=(0, 1))  # Transform features by scaling each feature to a given range
    train = scaler.fit_transform(
        train_1)  # Fits transformer to 'train_1' and returns a transformed version of 'train_1'.
    forecast = scaler.transform(test_1)

    trainy = train[:, -int(length * 3.5):]
    trainX = train[:, :-int(length * 3.5)]

    forecasty = forecast[:, -int(length * 3.5):]
    forecastX = forecast[:, :-int(length * 3.5)]

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    forecastX = forecastX.reshape((forecastX.shape[0], 1, forecastX.shape[1]))
    # print(trainX.shape, trainy.shape, forecastX.shape)

    """## Model (Without Monte Carlo Dropout)"""

    a = tf.Variable(0.1, name="alpha", trainable=True, dtype=tf.float32)
    b = tf.Variable(0.05, name="beta", trainable=True, dtype=tf.float32)
    c = tf.Variable(1.1, name="gamma", trainable=True, dtype=tf.float32)
    d = tf.Variable(0.1, name="delta", trainable=True, dtype=tf.float32)
    splitr = 0.5

    def loss_fn(y_true, y_pred):
        squared_difference = tf.square(y_true[:, 0] - y_pred[:, 0])
        squared_difference2 = tf.square(y_true[:, 2] - y_pred[:, 2])
        squared_difference1 = tf.square(y_true[:, 1] - y_pred[:, 1])
        squared_difference3 = tf.square(y_pred[:, 2] - c + d * y_pred[:, 1] - a * y_pred[:, 0] - b * y_pred[:, 0] ** 3)
        return tf.reduce_mean(squared_difference, axis=-1) + 0.2 * tf.reduce_mean(squared_difference3, axis=-1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(length * 3.5)))
    model.compile(loss=loss_fn, optimizer='adam', metrics=["mae", "mse"])
    history = model.fit(trainX[:int(splitr * trainX.shape[0])], trainy[:int(splitr * trainX.shape[0])], epochs=500,
                        batch_size=64, validation_data=(
            trainX[int(splitr * trainX.shape[0]):trainX.shape[0]],
            trainy[int(splitr * trainX.shape[0]):trainX.shape[0]]),
                        shuffle=False)

    """## Prediction (Without Monte Carlo Dropout)"""

    forecast_without_mc = forecastX
    yhat_without_mc = model.predict(forecast_without_mc)  # Step Ahead Prediction ('length' 'freq')
    forecast_without_mc = forecast_without_mc.reshape(
        (forecast_without_mc.shape[0], forecast_without_mc.shape[2]))  # Historical Input

    # forecastX.shape
    #
    # forecast_without_mc.shape

    inv_yhat_without_mc = np.concatenate((forecast_without_mc, yhat_without_mc),
                                         axis=1)  # Concatenation of predicted values with Historical Data
    inv_yhat_without_mc = scaler.inverse_transform(inv_yhat_without_mc)  # Transform labels back to original encoding

    # inv_yhat_without_mc.shape

    # inv_yhat_without_mc

    # inv_yhat_without_mc[:,-int(length*3.0):].shape

    fforecast = inv_yhat_without_mc[:, -int(length * 3.0):]

    # fforecast

    case_forecast = fforecast[:, 0:int((length * 3.0) - 1):3]

    # code to replace all negative value with 0
    case_forecast[case_forecast < 0] = 0

    case_fforecast = np.around(case_forecast)

    print(np.array(case_fforecast))

    if compar:
        training_mae = history.history['val_mae']
        training_mse = history.history['val_mse']
        compar_true(col_imp, case_fforecast, training_mae, training_mse)


def henon_heiles(length: int, freq: int, col_imp: int, data_path: str, compar: bool):
    """## Pretraining

    _Henon Heiles Intermittency_ is the Simulated Data

    The `henon_heiles_intermittency.dat` feeds the model with the dynamics of the Henon Heiles Equations.
    """

    data = np.genfromtxt('datasets/henon_heiles_intermittency.dat')
    training_set = pd.DataFrame(data).reset_index(drop=True)
    training_set = training_set.iloc[:, 1]

    """## Computing the Gradients

    _Calculating the value of_ $\frac{dx}{dt}$, _and_ $\frac{d^2x}{dt^2}$
    """

    t_diff = freq
    # print(training_set.max())
    gradient_t = (training_set.diff() / t_diff).iloc[1:]  # dx/dt
    # print(gradient_t)
    gradient_tt = (gradient_t.diff() / t_diff).iloc[1:]  # d2x/dt2
    # print(gradient_tt)

    """## Loading Datasets

    """

    data = pd.read_csv(str(data_path))
    training_set = data.iloc[:, col_imp]
    # training_set

    training_set = training_set.head(training_set.shape[0])
    # training_set

    training_set = training_set.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of training_set as index
    gradient_t = gradient_t.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_t as index
    gradient_tt = gradient_tt.reset_index(
        drop=True)  # sets a list of integer ranging from 0 to length of gradient_tt as index

    df = pd.concat((training_set[:-1], gradient_t), axis=1)
    gradient_tt.columns = ["grad_tt"]
    df = pd.concat((df[:-1], gradient_tt), axis=1)
    df.columns = ['y_t', 'grad_t', 'grad_tt']

    """## Preprocessing the data into supervised learning"""

    # split a sequence into samples
    def Supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n_in, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n_out)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    data = Supervised(df.values, n_in=int(length * 3.5), n_out=length)

    for iter in range(1, length + 1):
        data.drop(['var2(t-' + str(iter) + ')', 'var3(t-' + str(iter) + ')'], axis=1, inplace=True)
    # print(data.head())
    # print(data.columns)
    #
    # data.shape
    #
    # data[0:len(data)-1].shape
    #
    # data.tail(1).shape

    train_1 = np.array(data[0:len(data) - 1])
    test_1 = np.array(data.tail(1))

    scaler = MinMaxScaler(feature_range=(0, 1))  # Transform features by scaling each feature to a given range
    train = scaler.fit_transform(
        train_1)  # Fits transformer to 'train_1' and returns a transformed version of 'train_1'.
    forecast = scaler.transform(test_1)

    trainy = train[:, -int(length * 3.5):]
    trainX = train[:, :-int(length * 3.5)]

    forecasty = forecast[:, -int(length * 3.5):]
    forecastX = forecast[:, :-int(length * 3.5)]

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    forecastX = forecastX.reshape((forecastX.shape[0], 1, forecastX.shape[1]))
    # print(trainX.shape, trainy.shape, forecastX.shape)

    """## Model (Without Monte Carlo Dropout)"""

    splitr = 0.5

    def loss_fn(y_true, y_pred):
        squared_difference = tf.square(y_true[:, 0] - y_pred[:, 0])
        squared_difference2 = tf.square(y_true[:, 2] - y_pred[:, 2])
        squared_difference1 = tf.square(y_true[:, 1] - y_pred[:, 1])
        squared_difference3 = tf.square((y_pred[:, 2] - y_pred[:, 1]))
        return tf.reduce_mean(squared_difference, axis=-1) + 0.2 * tf.reduce_mean(squared_difference3, axis=-1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(length * 3.5)))
    model.compile(loss=loss_fn, optimizer='adam', metrics=["mae", "mse"])
    history = model.fit(trainX[:int(splitr * trainX.shape[0])], trainy[:int(splitr * trainX.shape[0])], epochs=500,
                        batch_size=64, validation_data=(
            trainX[int(splitr * trainX.shape[0]):trainX.shape[0]],
            trainy[int(splitr * trainX.shape[0]):trainX.shape[0]]),
                        shuffle=False)

    """## Prediction (Without Monte Carlo Dropout)"""

    forecast_without_mc = forecastX
    yhat_without_mc = model.predict(forecast_without_mc)  # Step Ahead Prediction ('length' 'freq')
    forecast_without_mc = forecast_without_mc.reshape(
        (forecast_without_mc.shape[0], forecast_without_mc.shape[2]))  # Historical Input

    # forecastX.shape
    #
    # forecast_without_mc.shape

    inv_yhat_without_mc = np.concatenate((forecast_without_mc, yhat_without_mc),
                                         axis=1)  # Concatenation of predicted values with Historical Data
    inv_yhat_without_mc = scaler.inverse_transform(inv_yhat_without_mc)  # Transform labels back to original encoding

    # inv_yhat_without_mc.shape

    # inv_yhat_without_mc

    # inv_yhat_without_mc[:,-int(length*3.0):].shape

    fforecast = inv_yhat_without_mc[:, -int(length * 3.0):]

    # fforecast

    case_forecast = fforecast[:, 0:int((length * 3.0) - 1):3]

    # code to replace all negative value with 0
    case_forecast[case_forecast < 0] = 0

    case_fforecast = np.around(case_forecast)

    print(np.array(case_fforecast))

    if compar:
        training_mae = history.history['val_mae']
        training_mse = history.history['val_mse']
        compar_true(col_imp, case_fforecast, training_mae, training_mse)


def pinntsf(choice: str, length: int, freq: int, col_imp: int, data_path: str, compar: bool):
    if choice == "van_der_pol":
        van_der_pol(length, freq, col_imp, data_path, compar)
    elif choice == "lienard":
        lienard(length, freq, col_imp, data_path, compar)
    elif choice == "lorenz":
        lorenz(length, freq, col_imp, data_path, compar)
    elif choice == "lotka_volterra":
        lotka_volterra(length, freq, col_imp, data_path, compar)
    elif choice == "duffing":
        duffing(length, freq, col_imp, data_path, compar)
    elif choice == "henon_heiles":
        henon_heiles(length, freq, col_imp, data_path, compar)
    else:
        print(
            "'" + choice + "' is not a valid physics for PINNTSF; supported values are 'van_der_pol', 'lienard', 'lorenz', 'lotka_volterra', 'duffing', 'henon_heiles'")


def compar_true(col_imp: int, case_fforecast, training_mae, training_mse):
    csv_filepath = str(input("Please enter a valid file path to the test data: "))
    while not os.path.isfile(csv_filepath):
        print("Error: That is not a valid file, try again...")
        csv_filepath = input("Please enter a valid file path to the test data: ")

    try:
        col_imp_choice = int(
            input("Is Column " + str(col_imp) + " still the reference column in the test data(0--No/1--Yes)? "))
        if col_imp_choice:
            col_imp_1 = col_imp
            test_data = pd.read_csv(csv_filepath)
            test_data = test_data.iloc[:, col_imp_1]
            MSE = np.square(np.subtract(np.array(test_data), np.array(case_fforecast))).mean()
            rsme = math.sqrt(MSE)
            MAE = np.abs(np.subtract(np.array(test_data), np.array(case_fforecast))).mean()
            mae = MAE
            print("Root Mean Squared Error (Training): " + str(math.sqrt(sum(training_mse) / len(training_mse))))
            print("Mean Absolute Error (Training): " + str(sum(training_mae) / len(training_mae)))
            print("Root Mean Squared Error (Testing): " + str(rsme))
            print("Mean Absolute Error (Testing): " + str(mae))
        else:
            col_imp_0 = int(input("Enter the referencing column index: "))
            test_data = pd.read_csv(csv_filepath)
            test_data = test_data.iloc[:, col_imp_0]
            MSE = np.square(np.subtract(np.array(test_data), np.array(case_fforecast))).mean()
            rsme = math.sqrt(MSE)
            MAE = np.abs(np.subtract(np.array(test_data), np.array(case_fforecast))).mean()
            mae = MAE
            print("Root Mean Squared Error (Training): " + str(math.sqrt(sum(training_mse) / len(training_mse))))
            print("Mean Absolute Error (Training): " + str(sum(training_mae) / len(training_mae)))
            print("Root Mean Squared Error (Testing): " + str(rsme))
            print("Mean Absolute Error (Testing): " + str(mae))

    except BaseException as exception:
        print(f"An exception occurred: {exception}")


"""
*******************************************************************************************
This 6 parameters, must be addressed
*******************************************************************************************
"""

param0 = "van_der_pol"  # Physical Dynamics
param1 = 13  # Forecast Length
param2 = 7  # Frequency of Data (Hourly -- 1/24, Daily -- 1, Weekly -- 7, and so on...)
param3 = 1  # Attribute of dataset, subjected to forecast.
param4 = "datasets/elnino.csv"  # Data path
param5 = True  # Fetch the metrics, Training + Testing (True -- Yes, False -- No)

pinntsf(param0, param1, param2, param3, param4, param5)

"""
*******************************************************************************************
*******************************************************************************************
"""