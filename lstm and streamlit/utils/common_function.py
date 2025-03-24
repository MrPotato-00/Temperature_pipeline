import os
import yaml
from src.logger import get_logger
#from src.custom_exception import CustomException
import pandas as pd

logger= get_logger(__name__)

def read_yaml(file):
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File path {file} does not exist")
        
        with open(file, "r") as f:
            content= yaml.safe_load(f)

        logger.info(f"File {file} loaded successfully.")
        return content

    except Exception as e:
        logger.error(f"Error raised while reading yaml file, {e}")
        #raise CustomException("Yaml file read failed", e)


    