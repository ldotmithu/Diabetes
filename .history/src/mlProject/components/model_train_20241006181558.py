from mlProject.config.configuration import *
from mlProject import logging
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.metrics import accuracy_score


class ModelTrain:
    def __init__(self,config:ModelTrainConfig) -> None:
        self.config=config
        
    def model_preprocess(self):
        num_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age']

     
        num_pipeline = Pipeline([
            ('power_transform', PowerTransformer(method='yeo-johnson')),  
            ('scale', StandardScaler())  
        ])

        
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
        
        preprocess_obj=self.model_preprocess()
       
        X_train=preprocess_obj.fit_transform(X_train)
        X_test=preprocess_obj.transform(X_test)
        
        log=LogisticRegression(class_weight='balanced',C=2)
        log.fit(X_train,y_train)
        
        joblib.dump(log,self.config.model_path)
        joblib.dump(preprocess_obj,self.config.preprocess_path)
        logging.info('Model save')
        logging.info('Preprocess file save')
        
        pred=log.predict(X_train)
        logging.info(accuracy_score(y_train,pred))
        
        
        