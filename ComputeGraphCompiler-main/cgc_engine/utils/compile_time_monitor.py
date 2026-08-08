# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

from .logger import logger
from .singleton_meta import SingletonMeta


class CompileMonitor(metaclass=SingletonMeta):
    """
    Compile time monitor (singleton pattern).

    This class tracks the compilation time for torch.compile operations.
    It uses the SingletonMeta metaclass to ensure only one instance exists
    throughout the application lifecycle.
    """

    def __init__(self):
        """Initialize the compile monitor with default values."""
        self._start_time: float = 0.0

    def start(self):
        """Start monitoring compilation time."""
        self._start_time = time.time()

    def mark(self, prefix: str = "") -> float:
        """
        Mark the current time point and return elapsed time since start.

        Args:
            prefix: Optional prefix string for the log message

        Returns:
            Elapsed time in seconds since monitoring started
        """
        time_collapsed = time.time() - self._start_time
        logger.debug(f"{prefix} collapsed time: {time_collapsed:.2f} s")
        return time_collapsed

    def end(self):
        """End compilation time monitoring and log the total time."""
        logger.debug(f"torch.compile takes {time.time() - self._start_time:.2f} s in total")
