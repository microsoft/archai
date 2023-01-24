# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from logging import LogRecord, StreamHandler
from logging.handlers import TimedRotatingFileHandler

from archai.nlp.logging_utils import (
    FORMATTER,
    LOG_FILE,
    RankFilter,
    get_console_handler,
    get_logger,
    get_timed_file_handler,
)


def test_rank_filter():
    # Assert that the filter accepts records from rank 0
    rank_filter = RankFilter(0)
    assert rank_filter.filter(LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None))

    # Assert that the filter rejects records from rank 1
    rank_filter = RankFilter(1)
    assert not rank_filter.filter(LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None))


def test_get_console_handler():
    # Assert that the handler is a StreamHandler instance
    handler = get_console_handler()
    assert isinstance(handler, StreamHandler)

    # Assert that the handler has the correct formatter
    assert handler.formatter == FORMATTER


def test_get_timed_file_handler():
    # Assert that the handler is an instance of TimedRotatingFileHandler
    handler = get_timed_file_handler()
    assert isinstance(handler, TimedRotatingFileHandler)

    # Assert that the handler uses the correct log file and formatter
    assert os.path.basename(handler.baseFilename) == LOG_FILE
    assert handler.formatter == FORMATTER


def test_get_logger():
    # Assert that the logger has the expected name
    logger = get_logger("test")
    assert logger.name == "test"

    # Assert that the logger has a console and file handler
    assert any(isinstance(h, StreamHandler) for h in logger.handlers)
    assert any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers)

    # Assert that the logger has a RankFilter
    assert any(isinstance(f, RankFilter) for f in logger.filters)

    # Assert that the logger does not propagate messages
    assert not logger.propagate
