# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import sys
from logging import Filter, Formatter, Logger, LogRecord, StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "archai.log"
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


class RankFilter(Filter):
    """A filter for logging records based on the rank of the process.

    Only log records from the process with rank 0 will be logged,
    while log records from other processes will be filtered out.

    """

    def __init__(self, rank: int) -> None:
        """Initialize the filter with the rank of the process.

        Args:
            rank: The rank of the process that will generate log records.

        """

        self.rank = rank

    def filter(self, record: LogRecord) -> bool:
        """Filter a logging record based on the process rank.

        Args:
            record: The logging record to be filtered.

        Returns:
            `True` if the record should be logged, `False` otherwise.

        """

        return self.rank == 0


def get_console_handler() -> StreamHandler:
    """Get a `StreamHandler` for logging to the console.

    The `StreamHandler` can be used to log messages to the
    console (i.e., `sys.stdout`) and is configured with a formatter.

    Returns:
        A `StreamHandler` for logging to the console.

    """

    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Get a `TimedRotatingFileHandler` for logging to timestamped files.

    Returns:
        A `TimedRotatingFileHandler` for logging to timestamped files.

    """

    file_handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight", encoding="utf-8")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name: str) -> Logger:
    """Get a logger with the specified name and default settings.

    Args:
        logger_name: The name of the logger.

    Returns:
        A `Logger` instance with the specified name and default settings.

    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_timed_file_handler())

    logger.addFilter(RankFilter(LOCAL_RANK))

    logger.propagate = False

    return logger
