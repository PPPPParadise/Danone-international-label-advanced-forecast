# coding: utf-8
import logging

from cfg.paths import LOG_FORMAT


def setup_loggers(tag: str, log_filepath: str, log_format: str = LOG_FORMAT, level=logging.DEBUG) -> None:
    """ Setup of loggers
    """
    # set level
    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.setLevel(level=level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_filepath % tag)
    fh.setLevel(level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(log_format)
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to root logger
    logger = logging.getLogger()
    logger.addHandler(ch)
    logger.addHandler(fh)
