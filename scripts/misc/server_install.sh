#!/bin/bash
cd /home/gs01/MagiCompiler-main
echo "Starting vllm installation at $(date)"
pip3 install vllm 2>&1
echo "Installation finished at $(date)"
pip3 show vllm | head -3