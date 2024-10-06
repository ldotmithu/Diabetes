from mlProject.components.data_ingestion import DataIngestion
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