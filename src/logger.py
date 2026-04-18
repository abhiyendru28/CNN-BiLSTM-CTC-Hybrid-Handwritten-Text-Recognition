import logging
import os
from src.config import LOG_DIRECTORY

def initialize_logger(module_name: str) -> logging.Logger:
    """
    Initializes a highly robust, dual-output logger for production tracking.
    Captures INFO, WARNING, ERROR, and CRITICAL exceptions.
    """
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent hierarchical duplicate logging
    if not logger.handlers:
        file_trace = logging.FileHandler(os.path.join(LOG_DIRECTORY, 'htr_execution.log'))
        file_trace.setLevel(logging.DEBUG)
        
        console_trace = logging.StreamHandler()
        console_trace.setLevel(logging.INFO)
        
        format_standard = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
        file_trace.setFormatter(format_standard)
        console_trace.setFormatter(format_standard)
        
        logger.addHandler(file_trace)
        logger.addHandler(console_trace)
        
    return logger