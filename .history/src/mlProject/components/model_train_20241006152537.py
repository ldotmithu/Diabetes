from mlProject.config.configuration import *
from mlProject import logging
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer



class ModelTrain:
    def __init__(self,config:ModelTrainConfig) -> None:
        self.config=config
        
    def model_preprocess(self):
        num_columns=['Pregnancies',
 'Glucose',
 'BloodPressure',
 'SkinThickness',
 'Insulin',
 'BMI',
 'DiabetesPedigreeFunction',
 'Age']
        
        num_pipeline=Pipeline([
    ('scale',StandardScaler())
])

Preproess=ColumnTransformer([
    ('num_columns',num_pipeline,num_columns)
])
            