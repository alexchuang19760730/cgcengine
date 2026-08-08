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


import threading


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass implementation.

    This metaclass ensures that only one instance of a class is created,
    even in a multi-threaded environment. It uses a class-level lock to
    synchronize instance creation across threads.

    Usage:
        class MyClass(metaclass=SingletonMeta):
            pass

        # Both instances will be the same object
        instance1 = MyClass()
        instance2 = MyClass()
        assert instance1 is instance2  # True
    """

    # Dictionary to store singleton instances for each class
    _instances = {}
    # Class-level lock to ensure thread-safe instance creation
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Override __call__ to implement singleton pattern.

        Uses double-checked locking pattern to minimize lock contention:
        1. First check if instance exists (no lock needed)
        2. If not, acquire lock and check again
        3. Create instance only if still doesn't exist

        Args:
            *args: Positional arguments passed to the class constructor
            **kwargs: Keyword arguments passed to the class constructor

        Returns:
            The singleton instance of the class
        """
        # Fast path: check if instance already exists (no lock needed)
        if cls not in cls._instances:
            # Slow path: acquire lock and check again (double-checked locking)
            with cls._lock:
                # Check again inside the lock to prevent race condition
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
