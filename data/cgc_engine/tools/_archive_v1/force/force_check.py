#!/usr/bin/env python
"""Force check versions"""
import subprocess
import sys

# Run the check
result = subprocess.run([
    'ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=30',
    'gs01@10.100.200.65',
    'echo "SSH works" && pip list | grep -iE "torch|vllm"'
], capture_output=True, text=True, timeout=60)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
