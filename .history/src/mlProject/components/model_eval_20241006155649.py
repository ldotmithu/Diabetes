from mlProject import logging
import pandas as pd 
import joblib,os
from mlProject.config.configuration import *
from sklearn.metrics import accuracy_score
from mlProject.utils.common import save_json

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationconfig) -> None:
        self.config=config
        
    def evaluation(self):
        test_data=pd.read_csv(self.config.test_data_path)
        model=joblib.load(self.config.model_path)
        preprocess_obj=joblib.load(self.config.preprocess_path)
        
        target_col='Outcome'
        
        
        X_test=test_data.drop(columns=target_col,axis=1)
        y_test=test_data[target_col]   
        X_test=preprocess_obj.transform(X_test)
        
        pred=model.predict(X_test)
        
        acc_score=accuracy_score(y_test,pred)
        
        score={'accuracy_score':acc_score}
        save_json(path=os.path.join(self.config.root_dir,self.config.metrics_path),data=score)
        
        