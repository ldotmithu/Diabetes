import joblib
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts\model_train\model.joblib'))
        self.preprocess = joblib.load(Path('artifacts\model_train\preprocess.joblib'))

    def predict(self, data):
        # Ensure data is converted into a DataFrame if it's not already
        if isinstance(data, pd.DataFrame):
            processed_data = self.preprocess.transform(data)
        else:
            data = pd.DataFrame(data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction','Age'])
            processed_data = self.preprocess.transform(data)

        # Perform the prediction
        prediction = self.model.predict(processed_data)

        return prediction