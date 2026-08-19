#!/usr/bin/env python3
"""pressure_holder.py — hold N GB of INCOMPRESSIBLE anonymous memory (real pressure).
Usage: python3 pressure_holder.py <gb> [seconds]
Verifies pressure is real by sampling compressor pages.
"""
import sys, time, os, subprocess

gb = float(sys.argv[1])
secs = int(sys.argv[2]) if len(sys.argv) > 2 else 600
n = int(gb * 1024**3)
buf = bytearray(n)
mv = memoryview(buf)
with open('/dev/urandom', 'rb') as f:
    off = 0
    while off < n:
        chunk = min(1 << 28, n - off)  # 256MB per read
        got = f.readinto(mv[off:off+chunk])
        off += got
print(f"pressure: holding {n/1e9:.1f}GB incompressible", flush=True)

def vm():
    out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().split('.')[0]
            if v.lstrip('-').isdigit():
                d[k.strip()] = int(v)
    return d

t0 = time.time()
while time.time() - t0 < secs:
    d = vm()
    free = (d.get('Pages free', 0) + d.get('Pages speculative', 0)) * 16384 / 16e9 * 100
    comp = d.get('Pages occupied by compressor', 0) * 16384 / 16e9
    print(f"free_pct={free:.1f}% compressor={comp:.2f}GB", flush=True)
    time.sleep(2)
