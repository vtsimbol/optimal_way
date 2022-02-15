import logging
import sys


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    return console_handler


def init_logger(level=logging.INFO):
    logger = logging.getLogger('')
    logger.setLevel(level)
    logger.addHandler(get_console_handler())


def get_logger(logger_name):
    if len(logger_name) == 0:
        raise ValueError("Not use 'root' logger!")

    logger = logging.getLogger(logger_name)
    return logger
