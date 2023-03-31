import logging
from logging.handlers import RotatingFileHandler
import sys
import os

DEFAULT_LOG_DIR = "/var/log/islr/"


def get_logger(name, log_path=""):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Define log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if log_path == "":
        os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
        log_path = os.path.join(DEFAULT_LOG_DIR, "islr-run.log")

    # Define file handler for log rotation
    file_handler = RotatingFileHandler(log_path, maxBytes=1000000, backupCount=10)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Define console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
