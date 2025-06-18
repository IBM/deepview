# /*******************************************************************************
#  * Copyright 2025 IBM Corporation
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
# *******************************************************************************/

# Standard
import sys


class Tee:
    """Duplicates output to multiple streams.

    Args:
        *streams: Variable length argument list of streams to write to.
    """

    def __init__(self, *streams):
        """Initialize Tee with multiple output streams.

        Args:
            *streams: Streams to duplicate writes to (e.g., sys.stdout, a file).
        """
        self.streams = streams

    def write(self, data):
        """Write data to all streams.

        Args:
            data (str): The string data to write.
        """
        for s in self.streams:
            s.write(data)

    def flush(self):
        """Flush all streams to ensure all buffered data is written out."""
        for s in self.streams:
            s.flush()

    def isatty(self):
        """Return True if the first stream is connected to a terminal.

        Returns:
            bool: True if first stream is a tty, False otherwise.
        """
        return getattr(self.streams[0], "isatty", lambda: False)()

    def fileno(self):
        """Return the file descriptor of the first stream.

        Returns:
            int: File descriptor if available, else -1.
        """
        return getattr(self.streams[0], "fileno", lambda: -1)()
