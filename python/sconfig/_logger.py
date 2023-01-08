__all__ = ["get_logger"]

import logging


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)
        # Create formatters and add it to handlers
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.setLevel(level or logging.INFO)

    return logger
