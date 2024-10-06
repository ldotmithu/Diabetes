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

@app.route('/predict', methods=['POST', 'GET'])  # Route to show predictions
def predict():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            # Prepare data for prediction
            data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction=predict)

        except Exception as e:
            print('The Exception message is: ', e)
            return render_template('results.html', prediction="Error in prediction.")

    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
