from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('brain_stroke.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
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
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
