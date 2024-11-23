from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/StyleTransfer/Inputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/StyleTransfer", methods=["POST", "GET"])
def StyleTransfer():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('file')
        if not files or len(files) != 2:
            flash('Please upload exactly two files: an object image and a style image')
            return redirect(request.url)
        
        paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                paths.append(filepath)
        
        
        return render_template("StyleTransfer.html", object_image=paths[0], style_image=paths[1])
    
    return render_template("StyleTransfer.html")


@app.route("/Forecasting", methods=["GET", "POST"])
def Forecasting():
    return render_template("Forecasting.html")

if __name__ == "__main__":
    app.run(debug=True)
