from flask import Flask,render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import yfinance as yf
from datetime import datetime, timedelta
from DeepAR.ModelFunctionalities import *
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/Forecasting", methods=["GET", "POST"])
def forecasting():
    if request.method == 'POST':
        stock_name = request.form.get("Stockname").upper()
        if stock_name:
                # Fetch stock data
                data = get_stock_prices(stock_name)
                
                if len(data) == 0:
                    return render_template("Forecasting.html", error=f"No data found for ticker '{stock_name}'. Please check the ticker symbol and try again.")
                
                # For simplicity, we'll just return the last closing price
                Xtr, ytr, Xts, yts = Prepare_Data(data)

                # Ensure the directory exists before saving
                os.makedirs(os.path.dirname("DeepAR/TraderO/TS.save"), exist_ok=True)  # Add this line

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
                    return render_template("Forecasting.html", result=f"Last known price: {last_known_price:.2f} $", rES = f"Predicted change: {next_day_price - last_known_price:.2f}$")
                except Exception as e:
                    print("Error occurred:", str(e))
                    print("latest_data:", latest_data)
                               
                    return render_template("Forecasting.html", error=f"An error occurred: {str(e)}. Please check the ticker symbol and try again.")
                    
    return render_template("Forecasting.html")

@app.route("/FETranslation")
def Transaltion():
    return render_template("FETranslation.html")


UPLOAD_FOLDER = './StyleTransfer/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/Style", methods = ["POST","GET"])
def Style():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('StyleTransf.html')
if __name__ == "__main__":
    app.run(debug=True)

