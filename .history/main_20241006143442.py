from mlProject.pipeline.training_pipeline import *
from mlProject import logging

Stage_Name= ' Data Ingestion'
try:
    data_ingestion=DataIngestionPipeline()
    data_ingestion.main()
    logging.info(f'{Stage_Name} Completed')
    logging.info('........................')
except Exception as e:
    logging.exception(e)
    raise e

Stage_Name=' Data Transfomation'
try:
    data_transfomation=DataTransfomationPipeline()
    data_transfomation.main()
    logging.info(f'{Stage_Name} Completed')
    logging.info('........................')
except Exception as e:
    logging.exception(e)
    raise e            