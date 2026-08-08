#!/bin/bash
cd /tmp
/tmp/cuda_env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
/tmp/cuda_env/bin/pip list | grep torch