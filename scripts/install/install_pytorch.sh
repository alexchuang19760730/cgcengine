#!/bin/bash
echo "Starting PyTorch install..."
/tmp/cuda_env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
echo "Exit code: $?"
/tmp/cuda_env/bin/pip list | grep torch || echo "Torch not found"