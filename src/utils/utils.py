import os
import sys
import logging


def set_logger(log_dir, log_file_name):
    """
    Set logger to log the training and evaluation process.
    Args:
        log_dir (str): path to save log file
        log_file_name (str): name of log file
    Returns:
        logger (logging.Logger): logger instance
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # if log file exists, then delete it
    if os.path.exists(os.path.join(log_dir, log_file_name)):
        os.remove(os.path.join(log_dir, log_file_name))
    logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO)
    logger = logging.getLogger()
    return logger