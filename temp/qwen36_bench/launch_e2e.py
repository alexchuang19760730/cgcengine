import os, subprocess, sys
CLI = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI"
MODEL = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo"
MSG = "/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench/msg_smoke.json"
LOG = sys.argv[1] if len(sys.argv) > 1 else "/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench/e2e.log"
MAXNEW = sys.argv[2] if len(sys.argv) > 2 else "16"
env = dict(os.environ)
env.update({"TURBO_FIELDFARE_MODEL": MODEL, "TURBO_FIELDFARE_EXPERT_SLOTS": "16",
            "TURBO_FIELDFARE_EXPERT_PREFETCH": "0", "TURBO_FIELDFARE_MISS_PREFETCH": "0", "MTP_MODEL": ""})
cmd = [CLI, "--model", MODEL, "--trust-receipt", "--messages-file", MSG,
       "--max-new", MAXNEW, "--temperature", "0", "--repetition-penalty", "1.0", "--max-context", "256"]
with open(LOG, "w") as f:
    subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
print("launched, log =", LOG)
