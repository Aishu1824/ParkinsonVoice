import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
from utils.feature_extraction import extract_features
from werkzeug.utils import secure_filename

# Get the absolute path to the frontend/templates folder
template_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend/templates")

app = Flask(__name__, template_folder=template_folder_path)
app.secret_key = "supersecretkey"

# Set the upload folder path in the backend directory
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Load the trained model
with open("voicepark/backend/parkinsons_model.pkl", "rb") as file:
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
            result = "Affected by Parkinson's" if prediction == 1 else "Healthy"

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
