import logging
import json
import sys


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


class CustomFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'override_funcName'):
            record.funcName = record.override_funcName
        return super(CustomFormatter, self).format(record)


log_format = {
    "Timestamp": "%(asctime)s",
    "Level": "%(levelname)s",
    "Message": "%(message)s",
    "Name": "%(name)s",
    "Function": "%(funcName)s"
}
FORMATTER = CustomFormatter(fmt=json.dumps(log_format), datefmt="%Y-%m-%dT%H:%M:%S%z")


def init_logger(level=logging.INFO):
    logger = logging.getLogger('')
    logger.setLevel(level)
    logger.addHandler(get_console_handler())


def get_logger(logger_name):
    if len(logger_name) == 0:
        raise ValueError("Not use 'root' logger!")

    logger = logging.getLogger(logger_name)
    return logger
