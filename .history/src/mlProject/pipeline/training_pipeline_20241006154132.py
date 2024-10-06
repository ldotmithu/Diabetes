from mlProject.components.data_ingestion import DataIngestion
from mlProject.components.data_transfomation import DataTransfomation
from mlProject.components.model_train import ModelTrain
from mlProject.config.configuration import ConfigurationManager
from mlProject import logging


class DataIngestionPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=data_ingestion_config)
        data_ingestion.download_zip_data()
        data_ingestion.Extract_File()
        
class DataTransfomationPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config=ConfigurationManager()
        data_transfomation_config=config.get_data_transfomation_config()
        data_transfomation=DataTransfomation(config=data_transfomation_config)
        data_transfomation.Split_data()   
        
class ModelTrainConfig:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config=ConfigurationManager()
        model_train_config=config.get_model_train_congif()
        model_train=ModelTrain(config=model_train_config)
        model_train.model_preprocess()
        model_train.train()        
        
             
        
        