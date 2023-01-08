__all__ = ["get_logger"]

import logging


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler("file.log")
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.ERROR)
        # Create formatters and add it to handlers
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(c_format)
        # f_handler.setFormatter(f_format)
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        logger.setLevel(level or logging.DEBUG)

    return logger
