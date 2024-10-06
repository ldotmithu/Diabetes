from mlProject.config.configuration import *
from mlProject import logging
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



class ModelTrain:
    def __init__(self,config:ModelTrainConfig) -> None:
        self.config=config
        
    def model_preprocess(self):
        num_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Define the pipeline with PowerTransformer and StandardScaler
        num_pipeline = Pipeline([
            ('power_transform', PowerTransformer(method='yeo-johnson')),  # PowerTransformer applied first
            ('scale', StandardScaler())  # StandardScaler applied after PowerTransformer
        ])

        # Apply the pipeline using ColumnTransformer
        preprocess = ColumnTransformer([
            ('num_columns', num_pipeline, num_columns)
        ])
            
        return preprocess
    
    def train(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data=pd.read_csv(self.config.test_data_path)
        
        target_col='Outcome'
        
        X_train=train_data.drop(columns=target_col,axis=1)
        X_test=test_data.drop(columns=target_col,axis=1)
        y_train=train_data[target_col]
        y_test=test_data[target_col]
        