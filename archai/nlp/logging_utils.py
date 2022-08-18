# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Logging-based methods and helpers.
"""

import logging
import os
import sys
from logging import Filter, Formatter, Logger, LogRecord, StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "archai.nlp.log"
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


class RankFilter(Filter):
    """Filters internal loggings based on the rank of the process."""

    def __init__(self, rank: int) -> None:
        """Overrides initialization method.

        Args:
            rank: Rank of the process.

        """

        self.rank = rank

    def filter(self, record: LogRecord) -> bool:
        """Filters a logging record.

        Args:
            record: Logging record.

        Returns:
            (bool): Whether record should be logged or not.

        """

        return self.rank == 0


def get_console_handler() -> StreamHandler:
    """Gets a console handler to handle logging into console.

    Returns:
        (StreamHandler): Output information into console.

    """

    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Gets a timed file handler to handle logging into files.

    Returns:
        (TimedRotatingFileHandler): Output information into timed files.

    """

    file_handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight", encoding="utf-8")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name: str) -> Logger:
    """Gets a log and make it avaliable for further use.

    Args:
        logger_name: Name of the logger.

    Returns:
        (Logger): Logging-based object.

    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_timed_file_handler())

    logger.addFilter(RankFilter(LOCAL_RANK))

    # Defines whether logs should be propagated or not
    logger.propagate = False

    return logger
