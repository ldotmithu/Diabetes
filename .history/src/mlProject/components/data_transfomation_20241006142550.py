from mlProject.config.configuration import *
from sklearn.model_selection import train_test_split
import pandas as pd 
import os 


class DataTransfomation:
    def __init__(self,config:DataTransfomationConfig) -> None:
        self.config=config
        
    def Split_data(self):
            