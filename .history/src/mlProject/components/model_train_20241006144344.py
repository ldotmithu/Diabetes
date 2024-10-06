from mlProject.config.configuration import *
from mlProject import logging


class ModelTrain:
    def __init__(self,config:ModelTrainConfig) -> None:
        self.config=config
        
    def model_preprocess(self):
            