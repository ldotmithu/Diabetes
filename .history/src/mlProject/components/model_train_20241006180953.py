from mlProject.config.configuration import *
from mlProject import logging
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.metrics import accuracy_score

class ModelTrain:
    def __init__(self, config: ModelTrainConfig) -> None:
        self.config = config
        
    def model_preprocess(self):
        num_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Define the pipeline with IsolationForest, PowerTransformer, and StandardScaler
        num_pipeline = Pipeline([
            ('outlier_remover', IsolationForest(contamination=0.1, random_state=42)),
            ('power_transform', PowerTransformer(method='yeo-johnson')),  # PowerTransformer applied first
            ('scale', StandardScaler())  # StandardScaler applied after PowerTransformer
        ])

        # Apply the pipeline using ColumnTransformer
        preprocess = ColumnTransformer([
            ('num_columns', num_pipeline, num_columns)
        ])
        
        return preprocess
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        target_col = 'Outcome'
        
        X_train = train_data.drop(columns=target_col, axis=1)
        X_test = test_data.drop(columns=target_col, axis=1)
        y_train = train_data[target_col]
        y_test = test_data[target_col]
        
        preprocess_obj = self.model_preprocess()
       
        # Fit and transform the training data
        X_train = preprocess_obj.fit_transform(X_train)

        # Get the indices of inliers to filter y_train accordingly
        mask = preprocess_obj.named_transformers_['num_columns'].named_steps['outlier_remover'].predict(X_train) == 1
        y_train = y_train[mask]  # Filter y_train to match inliers in X_train

        # Apply the same transformation to X_test
        X_test = preprocess_obj.transform(X_test)

        # Train the Logistic Regression model
        log = LogisticRegression()
        log.fit(X_train, y_train)

        # Save the model and preprocessing object
        joblib.dump(log, self.config.model_path)
        joblib.dump(preprocess_obj, self.config.preprocess_path)
        logging.info('Model saved')
        logging.info('Preprocessing file saved')
        
        # Make predictions on training data for logging accuracy
        pred = log.predict(X_train)
        logging.info(f'Accuracy on training data: {accuracy_score(y_train, pred)}')
