import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
from utils.feature_extraction import extract_features
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Set the upload folder path in the backend directory
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Load the trained model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parkinsons_model.pkl")
with open(model_path, "rb") as file:

    model = pickle.load(file)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file is uploaded
        if 'file' not in request.files:
            flash("No file part!")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash("No selected file!")
            return redirect(request.url)

        # Validate file and save it
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract features and predict
            features = extract_features(filepath).reshape(1, -1)
            prediction = model.predict(features)
            result = ("Your voice patterns show potential signs that may be associated with Parkinson's disease. "
                      "We strongly recommend consulting a healthcare professional for a thorough evaluation and appropriate guidance."
                      if prediction == 1
                      else "Great news! Your voice patterns indicate no signs of Parkinson's disease. "
                           "However, regular checkups and a healthy lifestyle are always recommended for overall well-being.")

            flash(f"Prediction: {result}")
            return redirect(url_for("index"))
        else:
            flash("Invalid file type! Please upload a .wav file.")
            return redirect(request.url)

    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    # Ensure the uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('app.log')
stream_handler = logging.StreamHandler()

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# ...

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file is uploaded
        if 'file' not in request.files:
            logger.warning("No file part!")
            flash("No file part!")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file!")
            flash("No selected file!")
            return redirect(request.url)

        # Validate file and save it
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract features and predict
            features = extract_features(filepath).reshape(1, -1)
            prediction = model.predict(features)
            result = ("Your voice patterns show potential signs that may be associated with Parkinson's disease. "
                      "We strongly recommend consulting a healthcare professional for a thorough evaluation and appropriate guidance."
                      if prediction == 1
                      else "Great news! Your voice patterns indicate no signs of Parkinson's disease. "
                           "However, regular checkups and a healthy lifestyle are always recommended for overall well-being.")

            logger.info(f"Prediction: {result}")
            flash(f"Prediction: {result}")
            return redirect(url_for("index"))
        else:
            logger.warning("Invalid file type! Please upload a .wav file.")
            flash("Invalid file type! Please upload a .wav file.")
            return redirect(request.url)

    return render_template("index.html")

# ...

if __name__ == "__main__":
    # Ensure the uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    logger.info("App started on port {}".format(port))
    app.run(host="0.0.0.0", port=port, debug=True)