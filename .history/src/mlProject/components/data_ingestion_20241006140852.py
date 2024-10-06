from mlProject import logging
from mlProject.config.configuration import *
import os,zipfile
import urllib.request as request


class DataIngestion:
    def __init__(self,config:DataIngestionConfig) -> None:
        self.config=config
    
    def download_zip_data(self):
        try:
            if not os.path.exists(self.config.local_data_path):
                    request.urlretrieve(self.config.URL,self.config.local_data_path)
                    logging.info('Zip Data Doenloaded')
                    
            else:
                logging.info('File Alredy Exists')    
                
        except Exception as e:
            logging.exception(e)
            raise e
        
    def Extract_File(self):
        try:
            unzip_path=self.config.unzip_dir
            os.makedirs(unzip_path,exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_path,'r')as f :
                f.extractall(unzip_path)
                logging.info('Data Extract ')  
        except Exception as e:
            logging.exception(e)
            raise e                      