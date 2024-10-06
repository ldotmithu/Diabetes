from mlProject.config.configuration import *
from mlProject import logging
import pandas as pd 
from sklearn.linear_model import LogisticRegression



class ModelTrain:
    def __init__(self,config:ModelTrainConfig) -> None:
        self.config=config
        
    def model_preprocess(self):
            