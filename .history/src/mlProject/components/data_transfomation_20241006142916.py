from mlProject.config.configuration import *
from sklearn.model_selection import train_test_split
import pandas as pd 
import os 


class DataTransfomation:
    def __init__(self,config:DataTransfomationConfig) -> None:
        self.config=config
        
    def Split_data(self):
        data=pd.read_csv(self.config.data_path)
        logging.info('Data loed through Pandas')
        train_data,test_data=train_test_split(data,test_size=0.2)
        
        train_data.to_csv(os.path.join(self.config.root_dir,'train.csv'),index=False)
        test_data.to_csv(os.path.join(self.config.root_dir,'test.csv'),index=False)    
        logging.info('')