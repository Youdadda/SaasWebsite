import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import joblib
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, ELU
import matplotlib.pyplot as plt


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



def get_stock_prices(tickers):
    data = yf.download(tickers,  period="max")
    return data
def Prepare_Data(data):
    data.columns = data.columns.droplevel(1)
    data = data.reset_index()
    data.dropna(inplace= True)
    data['Date'] = pd.to_datetime(data['Date'])
    """ data = data.set_index(data['Date']) """
    ### Adding indicators to help the model catch trend and seasonality.
    add_indicators(data) # Ensure only the first two columns are assigned
    data = data.replace(np.nan, 0)

    target = data['Close'].shift(-1)[:-1]
    Features = data.drop(columns = ['Close', 'Date'], axis = 1)

    Feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()


    TS = target_scaler.fit(target.values.reshape(-1,1))
    FS = Feature_scaler.fit(Features)
    F_scaled = pd.DataFrame(FS.transform(Features.values), columns = Features.columns)
    T_scaled = pd.DataFrame(TS.transform(target.values.reshape(-1,1)))
    joblib.dump(TS, "DeepAR/TraderO/TS.save")
    joblib.dump(FS, "DeepAR/TraderO/FS.save")

    X, y = createdataset(F_scaled, T_scaled, 1)


    Xtr = X[:int(X.shape[0]*0.8)]
    Xts = X[int(X.shape[0]*0.8):]
    ytr = y[:int(y.shape[0]*0.8)]
    yts = y[int(y.shape[0]*0.8):]

    return Xtr, ytr, Xts, yts

def DeepARModel( X):
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
    model.add(ELU(alpha = 0.5))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(ELU(alpha = 0.5))


    model.add(Dense(units = 1))
    model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
    
    return model

def Training(Xtr, ytr, Xts, yts):

# ... (previous code, including model definition)

# Convert data to PyTorch tensors
    Xtr_tensor = torch.FloatTensor(Xtr)
    ytr_tensor = torch.FloatTensor(ytr).squeeze(-1)  # Remove the last dimension
    Xts_tensor = torch.FloatTensor(Xts)
    yts_tensor = torch.FloatTensor(yts).squeeze(-1)

    # Create DataLoader for training data
    train_dataset = TensorDataset(Xtr_tensor, ytr_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Assuming X.shape[2] is the number of features
    model = DeepARModel( Xtr)
    try:
         model.fit(Xtr, ytr, epochs=20, batch_size=32, validation_data=(Xts_tensor, yts_tensor))
    except UnicodeEncodeError:
        model.fit(Xtr, ytr, epochs=20, batch_size=32, validation_data=(Xts_tensor, yts_tensor), verbose=0)
    
    
    """ 

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(Xts_tensor).squeeze(-1)
        test_loss = criterion(test_predictions, yts_tensor)
        print(f'Test Loss: {test_loss.item():.4f}') """

    # Save the model
    model.save('DeepAR/TraderO/deepar_model.h5')

def load_model_and_scalers(model_path, feature_scaler_path, target_scaler_path):
    # Load the model
    
    model = tf.keras.models.load_model(model_path)
  # Set the model to evaluation mode

    # Load the scalers
    FS = joblib.load(feature_scaler_path)
    TS = joblib.load(target_scaler_path)

    return model, FS, TS

def add_indicators(data):
    # Assuming these functions are defined in your Utilities module
    data['SMA'] = SMA(data, 12) 
    data['MACD'] = MACD(data, 12, 26) 
    data['RSI'] = RSI(data, 12) 
    data['EMA'] = EMA(data, 12) 
    data['upper_band'], data['lower_band'] = Bollinger_Bands(data, 12)
    return data

def predict_next_day(model, Feature_scaler, target_scaler, latest_data):
    # Ensure the model is in evaluation mode

    # Convert latest_data to DataFrame if it's not already
    if not isinstance(latest_data, pd.DataFrame):
        latest_data_df = pd.DataFrame([latest_data])
    else:
        latest_data_df = latest_data.copy()

    # Add indicators to the latest data
    latest_data_with_indicators = add_indicators(latest_data_df)

    # Instead of dropping NaN values, fill them with the last known value
    latest_data_with_indicators = latest_data_with_indicators.fillna(value=0)

    # Use only the features used during training
    features_to_use = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'EMA', 'MACD', 'RSI', 'upper_band', 'lower_band']
    
    # Check if all required features are present
    missing_features = set(features_to_use) - set(latest_data_with_indicators.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    latest_data_array = latest_data_with_indicators[features_to_use].values

    if latest_data_array.shape[0] == 0:
        raise ValueError("No data available for prediction after processing")

    # Scale the data
    scaled_data = Feature_scaler.transform(latest_data_array)
    
    # Convert to PyTorch tensor and reshape for the model
    input_tensor = torch.FloatTensor(scaled_data).unsqueeze(1)  # Shape: (1, 1, num_features)

    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)

    # Inverse transform the prediction
    unscaled_prediction = target_scaler.inverse_transform(prediction.numpy().reshape(-1, 1))

    return unscaled_prediction[0][0]

