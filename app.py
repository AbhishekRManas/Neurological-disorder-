from flask import Flask, redirect, render_template, request, jsonify, send_from_directory, url_for
import joblib
from flask_cors import CORS
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename 

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('brain_stroke.pkl')

@app.route('/', methods=['GET', 'POST'])
def root():
    return redirect(url_for('index'))

@app.route('/index.html', methods=['GET', 'POST'])
def index():
    brain_stroke_image = url_for('uploaded_file', filename="brain_stroke.jpg")
    if request.method == 'POST':
        try:
            # Get the user inputs from the form
            gender_Female= request.json.get('gender_Female')
            gender_Male= request.json.get('gender_Male')
            age= request.json.get('age')
            hypertension= request.json.get('hypertension')
            heart_disease= request.json.get('heart_disease')
            ever_married_no= request.json.get('ever_married_no')
            ever_married_yes= request.json.get('ever_married_yes')
            avg_glucose_level= request.json.get('avg_glucose_level')
            bmi= request.json.get('bmi')
            work_type_Govt_job= request.json.get('work_type_Govt_job')
            work_type_Private= request.json.get('work_type_Private')
            work_type_Self_employed= request.json.get('work_type_Self_employed')
            work_type_children= request.json.get('work_type_children')
            Residence_type_Rural= request.json.get('Residence_type_Rural')
            Residence_type_Urban= request.json.get('Residence_type_Urban')
            smoking_status_formerly_smoked= request.json.get('smoking_status_formerly_smoked')
            smoking_status_never_smoked= request.json.get('smoking_status_never_smoked')
            smoking_status_smokes= request.json.get('smoking_status_smokes')
            smoking_status_unknown = request.json.get('smoking_status_unknown')

            # Make a prediction
            predicted_brain_stroke = model.predict([[age, hypertension, heart_disease, avg_glucose_level, bmi, gender_Female, gender_Male, ever_married_no, ever_married_yes, work_type_Govt_job, work_type_Private, work_type_Self_employed, work_type_children, Residence_type_Rural, Residence_type_Urban, smoking_status_unknown, smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes]])[0]

            # Classify the earthquake based on magnitude
            if predicted_brain_stroke == 0:
                classification = "Brain stroke occurance chance is less"
            else:
                classification = "Brain stroke occurance chance is more"


            # Prepare the response as JSON
            response = {
                'brain_stroke': float(predicted_brain_stroke),
                'classification': classification
            }
            
            # Return the JSON response
            return jsonify(response)
        except Exception as e:
            # If there's an error during prediction, return an error message as JSON
            error_response = {
                'error': str(e)
            }
            return jsonify(error_response), 500
    else:
        # If it's a GET request, render the index.html template
        return render_template('index.html', brain_stroke_image=brain_stroke_image)

# Load the trained model using Keras
Model = tf.keras.models.load_model("trained_model.h5")

# Create the "uploads" directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Function to preprocess the uploaded image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/brain_tumor', methods=["GET", "POST"])
def brain_tumor():
    prediction = None
    uploaded_image = None
    brain_tumor_image = url_for('uploaded_file', filename="brain_tumor.jpg")
    
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("brain_tumor.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("brain_tumor.html", error="No selected file")

        if file:
            file_name = secure_filename(file.filename)
            file_path = os.path.join("uploads", file_name)
            file.save(file_path)
            img_array = preprocess_image(file_path, target_size=(200, 200))
            prediction_probabilities = Model.predict(img_array)[0]
            predicted_class = np.argmax(prediction_probabilities)
            if predicted_class == 1:
                prediction = "TUMOR"
            else:
                prediction = "NO"
            
            uploaded_image = url_for('uploaded_file', filename=file_name)  # Use the route to serve the image
            print(url_for('uploaded_file', filename="brain_tumor.jpg"))

    return render_template("brain_tumor.html", prediction=prediction, uploaded_image=uploaded_image, brain_tumor_image=brain_tumor_image)