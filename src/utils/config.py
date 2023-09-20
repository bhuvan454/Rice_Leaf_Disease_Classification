import logging


def config_logging(log_dir, log_file_name):
    """
    Set logger to log the training and evaluation process.
    Args:
        log_dir (str): path to save log file
        log_file_name (str): name of log file
    Returns:
        logger (logging.Logger): logger instance
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    return logger