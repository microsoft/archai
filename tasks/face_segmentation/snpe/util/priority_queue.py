# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from threading import Lock


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.lock = Lock()

    def enqueue(self, priority, data):
        self.lock.acquire()
        inserted = False
        for i in range(len(self.queue)):
            item = self.queue[i]
            if item[0] > priority:
                self.queue.insert(i, (priority, data))
                inserted = True
                break
        if not inserted:
            self.queue += [(priority, data)]
        self.lock.release()

    def dequeue(self):
        """ returns a tuple containing the (priority, data) that was enqueued. """
        self.lock.acquire()
        item = None
        if len(self.queue) > 0:
            item = self.queue[0]
            del self.queue[0]
        self.lock.release()
        return item

    def peek(self):
        """ returns a tuple containing the (priority, data) that was enqueued. """
        self.lock.acquire()
        item = None
        if len(self.queue) > 0:
            item = self.queue[0]
        self.lock.release()
        return item

    def size(self):
        self.lock.acquire()
        rc = len(self.queue)
        self.lock.release()
        return rc
