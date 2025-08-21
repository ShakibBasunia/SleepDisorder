from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('sleep_disorder_model_xgb.pkl')
scaler = joblib.load('scaler1.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and handle missing or invalid data
        try:
            age = float(request.form['age'])
            sleep_duration = float(request.form['sleep_duration'])
            physical_activity = float(request.form['physical_activity'])
            bmi = float(request.form['bmi'])
            heart_rate = int(request.form['heart_rate'])
            daily_steps = int(request.form['daily_steps'])
            sleep_efficiency = float(request.form['sleep_efficiency'])
        except ValueError:
            return render_template('index.html', prediction_text="Error: Please ensure all fields are filled with valid numbers.")

        # Get the categorical fields (assume they are integers in the form)
        gender = int(request.form['gender'])
        occupation = int(request.form['occupation'])
        quality_of_sleep = int(request.form['quality_of_sleep'])
        stress_level = int(request.form['stress_level'])
        bmi_category = int(request.form['bmi_category'])

        # Prepare the feature vector (numeric data)
        features = np.array([[age, gender, occupation, sleep_duration, quality_of_sleep, physical_activity,
                              stress_level, bmi, heart_rate, daily_steps, sleep_efficiency, bmi_category]])

        # Apply the same scaling as the training data
        features_scaled = scaler.transform(features)

        # Predict using the trained model
        prediction = model.predict(features_scaled)

        # Map prediction back to original classes (optional)
        if prediction == 0:
            result = "None"
        elif prediction == 1:
            result = "Insomnia"
        elif prediction == 2:
            result = "Sleep Apnea"
        else:
            result = "Other Sleep Disorder"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
