#!/bin/bash
# HDC wrapper for Git Bash (avoids path conversion)
# Usage: ./hdc_wrap.sh shell <command>
#        ./hdc_wrap.sh file send <local> <remote>
#        ./hdc_wrap.sh file recv <remote> <local>

HDC="D:\\Program Files\\Huawei\\DevEco Studio\\sdk\\default\\openharmony\\toolchains\\hdc.exe"
DEV_ID="3DK0224920000525"

# Use Python to avoid Git Bash path issues
python -c "
import subprocess, sys
hdc = r'$HDC'
args = [hdc, '-t', '$DEV_ID'] + sys.argv[1:]
r = subprocess.run(args, capture_output=True, text=True, timeout=300)
print(r.stdout, end='')
if r.stderr:
    print(r.stderr, end='')
sys.exit(r.returncode)
" "$@"
