# options_screening/utils/logger.py

import logging
import sys

def setup_logger(name, level=logging.INFO):
    """Set up a logger that can be imported and used in any module."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Create a logger for the entire application
app_logger = setup_logger('options_screening')

def set_log_level(level):
    """Set the log level for the application logger."""
    app_logger.setLevel(level)

# Function to toggle debug logging
def toggle_debug_logging(enable=True):
    if enable:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(logging.INFO)