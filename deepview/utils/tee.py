# Standard
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return getattr(self.streams[0], "isatty", lambda: False)()

    def fileno(self):
        return getattr(self.streams[0], "fileno", lambda: -1)()
