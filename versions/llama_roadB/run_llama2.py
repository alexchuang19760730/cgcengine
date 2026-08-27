import subprocess, sys, time, os
limit = float(sys.argv[1]); args = sys.argv[2:]
logf = open("/tmp/llama_live.log", "w")
t0 = time.time()
p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
def pump():
    line = p.stdout.readline()
    if not line: return None
    logf.write(line.decode(errors="replace"))
    logf.flush()
    return line.decode(errors="replace").rstrip()
last = None
while time.time() - t0 < limit:
    import select
    r,_,_ = select.select([p.stdout], [], [], 20)
    if r:
        last = pump()
    else:
        sz = os.path.getsize("/tmp/llama_live.log")
        print(f"[{time.time()-t0:6.0f}s] log={sz} last={last!r}", flush=True)
    if p.poll() is not None:
        while pump() is not None: pass
        print(f"=== FINISHED exit={p.returncode} in {time.time()-t0:.0f}s ===", flush=True)
        break
else:
    p.kill()
    print(f"=== TIMEOUT {limit}s ===", flush=True)
print("--- last 30 lines of output ---")
lines = open("/tmp/llama_live.log").read().splitlines()
print("\n".join(lines[-30:]))
