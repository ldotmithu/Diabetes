from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction_piprline import PredictionPipeline

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"

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
