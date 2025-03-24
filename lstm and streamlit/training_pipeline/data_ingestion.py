from google.cloud import storage
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
#from src.custom_exception import CustomException
from utils.common_function import read_yaml
from config.paths_config import *
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ['GOOGLE_APPLICATION_CREDENTIALS']= "gcp-key.json"

logger= get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config= config["data_ingestion"]
        self.bucket_name= self.config["bucket_name"]
        self.train_file= self.config["train_file"]
        self.test_file= self.config["test_file"]

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info("Data Ingestion started")

    def download_data_from_gcp(self):
        try:
            client= storage.Client()
            bucket= client.bucket(self.bucket_name)
            train_blob= bucket.blob(self.train_file)
            test_blob= bucket.blob(self.test_file)

            train_blob.download_to_filename(TRAIN_FILE_PATH)
            test_blob.download_to_filename(TEST_FILE_PATH)
            
            logger.info("Data downloaded from GCP Bucket to {RAW_FILE_PATH} successfully...")

        except Exception as e:
            logger.error(f"Error raised while downloading data from GCP, {e}")
            #raise CustomException("Failed to download data from GCP Bucket", e)
  
    def run(self):
        try:
            self.download_data_from_gcp()
            #self.split_data()

            logger.info("DataIngestion completed successfully...")

        except Exception as e:
            logger.error(f"Error raised while running DataIngestion, {e}")
            #raise CustomException("Failed to run DataIngestion", e)
    
if __name__=="__main__":
    data_ingestion= DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()