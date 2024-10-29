import glob
import os.path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import torch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, ELU
from torch.utils.data import DataLoader, TensorDataset
from Utilities import *
import sys
import io
from PIL import Image


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
#return the last added file into the staticfiles\uploads directory
#a better practice would be to connect the flask api to a relational database.



def train_model(data):
    data.dropna(inplace= True)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index(data['Date'])
    ### Adding indicators to help the model catch trend and seasonality.
    data['SMA'] = SMA(data, 12)
    data['EMA'] = EMA(data, 12)
    data['MACD'] = MACD(data, 12, 26)
    data['RSI'] = RSI(data, 12)
    data['upper_band'], data['lower_band'] = Bollinger_Bands(data, 12)


    target = data['Close'].shift(-1)

    ### Data Normalisation
    Feature_scaler = MinMaxScaler()
    Target_scaler = MinMaxScaler()

    T_Sc = Target_scaler.fit(target.values.reshape(-1, 1))


    Features = data.drop(columns = ['Close', 'Date'], axis = 1)

    F_Sc = Feature_scaler.fit(Features)
    data_scaled = pd.DataFrame(F_Sc.transform(Features.values), columns = Features.columns.tolist()).dropna()
    Target_scaled = pd.DataFrame(T_Sc.transform(target.values.reshape(-1, 1)) , columns = ['target'])


    X, y = createdataset(data_scaled, Target_scaled, 1)

    ###Data Splitting 
    Xtr = X[:int(X.shape[0]*0.8)]
    Xval = X[int(X.shape[0]*0.8):int(X.shape[0]*0.9)]
    Xts = X[int(X.shape[0]*0.9):]
    ytr = y[:int(y.shape[0]*0.8)]
    yval = y[int(y.shape[0]*0.8):int(y.shape[0]*0.9)]
    yts = y[int(y.shape[0]*0.9):]

##Model Architecture
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
    model.add(ELU(alpha = 0.5))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(ELU(alpha = 0.5))


    model.add(Dense(units = 1))
    model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
    try:
         model.fit(Xtr, ytr, epochs=20, batch_size=32, validation_data=(Xval,yval))
    except UnicodeEncodeError:
        model.fit(Xtr, ytr, epochs=20, batch_size=32, validation_data=(Xval,yval), verbose=0)
    return model, Xts, yts, T_Sc, F_Sc

