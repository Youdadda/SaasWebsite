import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def SMA(data, window_size):
    return data['Close'].rolling(window=window_size).mean()

def EMA(data, window_size):
    return data['Close'].ewm(span=window_size).mean()

def MACD(data, short_window, long_window):
    short_EMA = EMA(data, short_window)
    long_EMA = EMA(data, long_window)
    return short_EMA - long_EMA

def RSI(data, window_size):
    delta = data['Close'].diff()
    delta = delta[1:] 
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=window_size-1 , min_periods=window_size).mean()
    ema_down = down.ewm(com=window_size-1 , min_periods=window_size).mean()
    return ema_up/ema_down

def Bollinger_Bands(data, window_size):
    middle_band = SMA(data, window_size)
    std_dev = data['Close'].rolling(window=window_size).std()
    upper_band = middle_band + (std_dev*2)
    lower_band = middle_band - (std_dev*2)
    return upper_band, lower_band




def date_features(df):
    # Create time series features based on time series index.
    df.index = pd.to_datetime(df.index)
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


def createdataset(X,y, time_step):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        Xs.append(X.iloc[i:i+time_step].values)
        ys.append(y.iloc[i:i+time_step].values)
    return np.array(Xs), np.array(ys)

