from mlProject.pipeline.training_pipeline import *
from mlProject import logging
import os
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

Stage_Name='Model Train'
try:
    model_train=ModelTrainPipeline()
    model_train.main()
    logging.info(f'{Stage_Name} Completed')
    logging.info('........................')
except Exception as e:
    logging.exception(e)
    raise e  

Stage_Name='Model Evaluation'
try:
    model_eval=ModelEvaluationPipeline()
    model_eval.main()
    logging.info(f'{Stage_Name} Completed')
    logging.info('........................')
except Exception as e:
    logging.exception(e)
    raise e  

try:
    os.system('python app.py')
    os.system('streamlit run app.py')    
    logging.info('web app runing')
except Exception as e:
    logging.exception(e)
    raise e    


     