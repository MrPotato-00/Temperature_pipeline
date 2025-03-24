import logging
import os
from datetime import datetime


LOGS_DIR= "logs"
os.makedirs(LOGS_DIR, exist_ok= True)

LOG_FILE= os.path.join(LOGS_DIR, f'log_{datetime.now().strftime("%d-%m-%Y")}.log')

logging.basicConfig(
    filename= LOG_FILE,
    level= logging.INFO, ## only show info, warning and error messages
    format= "%(asctime)s - %(levelname)s - %(message)s"
)

def get_logger(name):
    logger= logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger