#!/usr/bin/env python3
"""Wrapper: load OrthoKDA patch, then start sglang server."""
import sys
import os
import warnings

# Import patch loader (applies monkey-patch on import)
import orthokda_patch_loader

# Start sglang server (same as python3 -m sglang.launch_server)
from sglang.srt.plugins import load_plugins
from sglang.launch_server import prepare_server_args, run_server, kill_process_tree

warnings.warn(
    "'python -m sglang.launch_server' is still supported, but "
    "'sglang serve' is the recommended entrypoint.",
    UserWarning,
    stacklevel=1,
)

load_plugins()
server_args = prepare_server_args(sys.argv[1:])

try:
    run_server(server_args)
finally:
    kill_process_tree(os.getpid(), include_parent=False)
