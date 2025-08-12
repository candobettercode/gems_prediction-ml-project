from src.logger import logging
from src.exception import *

#logging.info("Logging has started")
#logging.debug("This is a debug message.")
#logging.warning("This is a warning message.")
#logging.error("This is an error message.")
#logging.critical("This is a critical message.")


if __name__=='__main__':
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero error")
        raise MyException(e,sys)
    