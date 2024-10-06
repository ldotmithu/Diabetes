from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessor
model_path = 'artifacts/model_train/model.joblib'
preprocess_path = 'artifacts/model_train/preprocess.joblib'
model = joblib.load(model_path)
preprocessor = joblib.load(preprocess_path)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs from the form
    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])

    # Create a DataFrame from input data
    input_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age],
    })

    # Preprocess the input data
    try:
        input_data_preprocessed = preprocessor.transform(input_data)
    except ValueError as e:
        return render_template('results.html', prediction=f"Error in preprocessing: {e}")

    # Predict diabetes risk
    try:
        prediction = model.predict(input_data_preprocessed)[0]
        if prediction == 0:
            result = "No Diabetes"
        else:
            result = "Diabetes"
    except Exception as e:
        result = f"Error in prediction: {e}"

    return render_template('results.html', prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
