#!/bin/bash
cd /home/gs01
ls -la MagiCompiler 2>/dev/null || echo "MagiCompiler not found"
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"