from flask import Flask,render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import yfinance as yf
from datetime import datetime, timedelta
from DeepAR.ModelFunctionalities import *
import os
from S.src.models.style_transfer import StyleTransfer


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/StyleTransfer/Inputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("main.html")



####################### StyleTransfer ############################

def make_prediction(content_image_path, style_image_path, output_image_path, learning_rate=5.0, style_weight=10, content_weight=1e3, epochs=3000):
    transfer = StyleTransfer(
        content_image_path, style_image_path, output_image_path,
        learning_rate, style_weight, content_weight, epochs
    )

    history = transfer.run_style_transfer()
    return history

@app.route("/ImageTransfer", methods=["POST", "GET"])
def ImageTransfer():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('file')
        if not files or len(files) != 2:
            flash('Please upload exactly two files: an object image and a style image')
            return redirect(request.url)
        
        paths = []
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                filename = 'Object.jpg' if i == 0 else 'Style.jpg'
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                paths.append(filepath)
        
        # Ensure we have exactly two paths and run style transfer
        if len(paths) == 2:
            output_path = 'static/Output.jpg'
            make_prediction(paths[0], paths[1], output_path)
            return render_template('ImageTransfer.html', result=True)
        else:
            flash('Error in saving files')
            return redirect(request.url)
            
    return render_template('ImageTransfer.html')




####################### Forecasting ############################


@app.route("/Forecasting", methods=["GET", "POST"])
def Forecasting():
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

if __name__ == "__main__":
    app.run(debug=True)
