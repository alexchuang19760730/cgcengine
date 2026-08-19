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

import logging
import os
import sys

import torch.distributed as dist

from cgc_engine.utils.envs import MAGI_LOGGING_LEVEL

_FMT = "[%(asctime)s - %(levelname)s] [Rank %(rank)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        try:
            return dist.get_rank()
        except Exception:
            pass
    return int(os.getenv("RANK", 0))


def _get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        try:
            return dist.get_world_size()
        except Exception:
            pass
    return 1


def _should_log(rank: int | str) -> bool:
    """rank: int (only that rank), 'all' (every rank)."""
    if rank == "all":
        return True
    current = _get_rank()
    if isinstance(rank, int):
        return current == rank
    return False


class _RankFormatter(logging.Formatter):
    """Inject ``rank`` into every log record so ``%(rank)s`` works in the format string."""

    def format(self, record: logging.LogRecord) -> str:
        record.rank = _get_rank()  # type: ignore[attr-defined]
        return super().format(record)


def _build_logger() -> logging.Logger:
    """Create and configure the ``cgc_engine`` logger exactly once."""
    lg = logging.getLogger("cgc_engine")
    lg.propagate = False
    lg.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_RankFormatter(fmt=_FMT, datefmt=_DATEFMT))
    lg.addHandler(handler)
    lg.setLevel(getattr(logging, MAGI_LOGGING_LEVEL, logging.WARNING))
    return lg


_std_logger = _build_logger()


class MagiLogger:
    """
    Logger for MagiCompiler, backed by the standard-library ``logging`` module
    with a dedicated named logger ``"cgc_engine"``.

    Fully isolated from loguru / other logging instances in the same process.
    Level is set once at import time via ``MAGI_LOGGING_LEVEL`` (see ``envs.py``).
    API: ``.info / .debug / .warning / .error(msg, *args, rank=0)``
    ``rank`` defaults to 0 (only rank-0 logs). Pass an explicit value to override:
    int = only that rank, "all" = every rank.
    """

    def info(self, message: str, *args, rank: int | str = 0, **kwargs) -> None:
        if not _should_log(rank):
            return
        _std_logger.info(message, *args, **kwargs)

    def debug(self, message: str, *args, rank: int | str = 0, **kwargs) -> None:
        if not _should_log(rank):
            return
        _std_logger.debug(message, *args, **kwargs)

    def warning(self, message: str, *args, rank: int | str = 0, **kwargs) -> None:
        if not _should_log(rank):
            return
        _std_logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, rank: int | str = 0, **kwargs) -> None:
        if not _should_log(rank):
            return
        _std_logger.error(message, *args, **kwargs)


magi_logger = MagiLogger()
logger = magi_logger  # alias
