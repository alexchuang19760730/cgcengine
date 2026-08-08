import os
import subprocess

output = subprocess.check_output(['lsof', '-F', 'pfn']).decode('utf-8')
lines = output.strip().split('\n')

pid = None
fd = None
for line in lines:
    if line.startswith('p'):
        pid = line[1:]
    elif line.startswith('f'):
        fd = line[1:]
    elif line.startswith('n'):
        name = line[1:]
        if '(deleted)' in name and ('/tmp/ray' in name or '/tmp/tmax' in name or '/tmp/uitars' in name):
            try:
                path = f"/proc/{pid}/fd/{fd}"
                if os.path.exists(path):
                    print(f"Truncating {path} for {name}")
                    with open(path, 'w') as f:
                        f.truncate(0)
            except Exception as e:
                print(f"Failed on {path}: {e}")
