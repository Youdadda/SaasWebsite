import pandas as pd
from DeepAR.ModelFunctionalities import *
import yfinance as yf


data = get_stock_prices('TSLA')


Xtr, ytr, Xts, yts = Prepare_Data(data)
Training(Xtr, ytr, Xts, yts)

data = add_indicators(data)
latest_data = data.iloc[-1]
model, FS , TS  = load_model_and_scalers("DeepAR/TraderO/deepar_model.h5", "DeepAR/TraderO/FS.save", "DeepAR/TraderO/TS.save")
# Usage
try:
    next_day_price = predict_next_day(model, FS, TS, latest_data)
    print(f"Predicted price for the next day: {next_day_price:.2f}")

    # Compare with the last known price
    last_known_price = data['Close'].iloc[-1]
    print(f"Last known price: {last_known_price:.2f}")
    print(f"Predicted change: {next_day_price - last_known_price:.2f}")
except Exception as e:
    print("Error occurred:", str(e))
    print("latest_data:", latest_data)
