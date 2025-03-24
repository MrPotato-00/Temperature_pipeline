import os
from src.logger import get_logger
from utils.common_function import read_yaml
import numpy as np
from config.paths_config import *
import traceback
import pandas as pd

logger= get_logger(__name__)

class CleanData:
    def __init__(self):
        logger.info("Data Cleaning Started")
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    def process_pressure(self, df, col):
        df_errors = df[(df[col] < 900) | (df[col]>1083)]

        df_cleaned = df.drop(df_errors.index , axis = 0)
        mean_pressure_modified = df_cleaned[col].mean()
        df.loc[df_errors.index , col] = mean_pressure_modified

    def process(self):
        try:

            data_train= pd.read_csv(TRAIN_FILE_PATH, index_col='date', parse_dates= ['date'])
            data_test= pd.read_csv(TEST_FILE_PATH, index_col="date", parse_dates= ["date"])

            data_train.rename(columns = {'meantemp':'temp'} , inplace = True)
            data_test.rename( columns =  {'meantemp':'temp'} , inplace = True)

            self.process_pressure(data_train, "meanpressure")
            self.process_pressure(data_test, "meanpressure")

            data_train.to_csv(PROCESSED_TRAIN_DATA_PATH)
            data_test.to_csv(PROCESSED_TEST_DATA_PATH)

            logger.info("Data cleaning completed...")
            logger.info(f"Processed data for train saved to {PROCESSED_TRAIN_DATA_PATH} and test file saved to {PROCESSED_TEST_DATA_PATH}")
        
        except Exception:
            logger.error(traceback.format_exc())
            
if __name__=="__main__":
    process= CleanData()
    process.process()

    

        
    
