#!/bin/bash
cd /tmp
ls -la /tmp/cuda_env/bin/ | head -20
echo "---"
/tmp/cuda_env/bin/pip list | grep torch