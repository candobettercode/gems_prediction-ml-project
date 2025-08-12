import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

# Constants for log configuration
LOG_DIR = 'logs'
#LOG_FILE = f"{datetime.now().strftime('%d-%m-%Y')}.log"
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024 
BACKUP_COUNT = 3 # Numbers of backup log files to keep

# Construct log file path
log_dir_path = os.path.join(from_root(),LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path,LOG_FILE)

def configure_logger():
    """
    Configure logging with a rorating file handle and a console handler.
    """

    # create custome logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Define formatter
    formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

    # File handler with rotation
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # COnsole Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# COnfigure the logger
configure_logger()