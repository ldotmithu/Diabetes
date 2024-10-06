from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Load the trained model and preprocessor
model = joblib.load(Path('artifacts/model_train/model.joblib'))
churn_preprocess = joblib.load(Path('artifacts/model_train/preprocess.joblib'))

# Create a FastAPI instance
app = FastAPI(title="Diabetes Prediction API")

# Define the input data structure using Pydantic
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Define a route for predictions
@app.post("/predict/")
async def predict_diabetes(data: DiabetesInput):
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Preprocess the input data
        input_data_preprocessed = churn_preprocess.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_preprocessed)[0]

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Define a route for feature explanation (optional)
@app.get("/features/")
async def feature_explanation():
    return {
        "feature_explanations": {
            "Pregnancies": "Number of pregnancies.",
            "Glucose": "Plasma glucose concentration over 2 hours in an oral glucose tolerance test.",
            "BloodPressure": "Diastolic blood pressure (mm Hg).",
            "SkinThickness": "Triceps skinfold thickness (mm).",
            "Insulin": "2-Hour serum insulin (mu U/ml).",
            "BMI": "Body Mass Index (weight in kg/(height in m)^2).",
            "DiabetesPedigreeFunction": "A function that scores the likelihood of diabetes based on family history.",
            "Age": "Age of the individual in years."
        }
    }
