# Importing necessary libraries
from mlProject.config.configuration import *  # Assuming this contains ModelTrainConfig
from mlProject import logging  # Assuming this handles logging
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.1):
        """Initialize the outlier remover."""
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination, random_state=42)

    def fit(self, X, y=None):
        """Fit the Isolation Forest model."""
        self.model.fit(X)
        return self

    def transform(self, X):
        """Transform the data by removing outliers."""
        # Predict outliers: -1 for outliers, 1 for inliers
        mask = self.model.predict(X) == 1  # Mask to keep only inliers
        return X[mask]  # Keep only inliers

class ModelTrain:
    def __init__(self, config: ModelTrainConfig) -> None:
        """Initialize the ModelTrain class with a configuration object."""
        self.config = config
        
    def model_preprocess(self):
        """Define the preprocessing pipeline for numerical features."""
        num_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Define the pipeline with OutlierRemover, PowerTransformer, and StandardScaler
        num_pipeline = Pipeline([
            ('outlier_remover', OutlierRemover(contamination=0.1)),  # Custom outlier remover
            ('power_transform', PowerTransformer(method='yeo-johnson')),  # Power transformation
            ('scale', StandardScaler())  # Standard scaling
        ])

        # Apply the pipeline using ColumnTransformer
        preprocess = ColumnTransformer([
            ('num_columns', num_pipeline, num_columns)
        ])
        
        return preprocess
    
    def train(self):
        """Load data, preprocess it, train the model, and save the model and preprocessing object."""
        # Load training and testing data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        target_col = 'Outcome'  # Define the target column
        
        # Split data into features and target
        X_train = train_data.drop(columns=target_col, axis=1)
        X_test = test_data.drop(columns=target_col, axis=1)
        y_train = train_data[target_col]
        y_test = test_data[target_col]
        
        # Create the preprocessing object
        preprocess_obj = self.model_preprocess()
       
        # Fit and transform the training data, transform the test data
        X_train = preprocess_obj.fit_transform(X_train)
        # Apply the same transformation to X_test, ensuring consistent preprocessing
        X_test = preprocess_obj.transform(X_test)
        
        # Initialize and train the model
        log = LogisticRegression()
        log.fit(X_train, y_train)
        
        # Save the model and preprocessing object
        joblib.dump(log, self.config.model_path)
        joblib.dump(preprocess_obj, self.config.preprocess_path)
        logging.info('Model saved successfully')
        logging.info('Preprocessing object saved successfully')
        
        # Make predictions and log the accuracy
        pred = log.predict(X_train)
        logging.info(f'Training Accuracy: {accuracy_score(y_train, pred)}')
