# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List
from logging import StreamHandler, Handler
import os
import time

from overrides import overrides

from threading import Lock

class AtomicFileHandler(Handler):
    """
    This class opens and writes entire file instead of appending one line at a time
    """

    terminator = '\n'

    def __init__(self, filename, encoding=None, save_delay=30.0):
        super().__init__()
        filename = os.fspath(filename)
        self.baseFilename = os.path.abspath(filename)
        self.encoding = encoding
        self._buffer:List[str] = []
        self.mutex = Lock()
        self._last_flush = 0.0
        self.save_delay = save_delay

    @overrides
    def flush(self):
        self._flush_buffer(force=True)

    @overrides
    def close(self):
        try:
            self._flush_buffer(force=True)
        finally:
            super().close()

    def _open(self):
        return open(self.baseFilename, 'a', encoding=self.encoding)

    def _flush_buffer(self, force=False)->None:
        if force or (time.time() - self._last_flush >= self.save_delay):
            try:
                self.mutex.acquire()
                with self._open() as f:
                    f.writelines(self._buffer)
                self._buffer.clear()
                self._last_flush = time.time()
            finally:
                self.mutex.release()

    def emit(self, record):
        msg = self.format(record)
        try:
            self.mutex.acquire()
            self._buffer.append(msg + self.terminator)
        finally:
            self.mutex.release()

        self._flush_buffer()


