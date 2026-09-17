#!/usr/bin/env python3
"""knifeedge.host — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from .anchor import ROOT, SERVER_MATCH
from .constants import FACT_PATTERNS, PROBE_PROMPT



def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout



def running_servers():
    """PIDs of REAL llama-server processes (not bash wrappers that merely mention it).

    `pgrep -f` also matches the desktop-client wrappers whose command line happens to contain
    the binary path, which would make the exclusivity gate refuse forever. So verify that the
    first argv token actually is the server binary before counting a pid as a server.
    """
    pids = []
    for p in sh("pgrep", "-f", SERVER_MATCH).split():
        if not p.strip():
            continue
        cmd = sh("ps", "-o", "command=", "-p", p).strip()
        if cmd and cmd.split(" ", 1)[0].endswith(SERVER_MATCH):
            pids.append(p)
    return pids



def kill_servers():
    subprocess.run(["pkill", "-9", "-f", SERVER_MATCH], capture_output=True)
    time.sleep(3)



def mem_free_pct():
    m = re.search(r"free percentage:\s*(\d+)%", sh("memory_pressure", "-Q"))
    return int(m.group(1)) if m else -1



def server_pid():
    pids = running_servers()
    return pids[0] if pids else None



def rss_gb():
    p = server_pid()
    if not p:
        return 0.0
    out = sh("ps", "-o", "rss=", "-p", p).strip()
    return round(int(out) / 1048576, 2) if out else 0.0



def health(port, timeout=3):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return b"ok" in r.read()
    except Exception:  # noqa: BLE001 - harness
        return False



def ask_probe(port, timeout=180.0):
    payload = {"model": "local", "messages": [{"role": "user", "content": PROBE_PROMPT}],
               "temperature": 0.0, "max_tokens": 48}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except Exception:  # noqa: BLE001 - harness
        pass



def launch_pool_bytes(args, gb):
    """Expert-cache budget, in bytes, that `launch()` should hand to run_server.sh.

    `--no-expert-cache` is the ground-truth arm: budget 0 means `-expert-cache 0`, which leaves
    `model->expert_index` empty, so no cache object is created and no hook is installed -- the
    stock full-resident forward pass. Every other launch keeps the pool at `gb` GiB.
    """
    if getattr(args, "no_expert_cache", False):
        return 0
    try:
        return int(gb) * 1024 ** 3
    except (TypeError, ValueError):
        return 0



def launch(model_file, mtp, pool_bytes, port, kwargs, extra_env, log_path):
    """Start run_server.sh in its own session so it survives this driver dying.

    `model_file` is an ABSOLUTE path (see model_path) -- run_server.sh accepts a model outside
    its MODEL_ROOT, which is what lets the external-drive IQ3 be measured without a 13 GB copy.
    """
    env = dict(os.environ)
    env.update({
        "CGC_DETACHED": "1",
        "_CGC_DETACHED_MARKER": "1",          # stop run_server.sh from forking again
        "CGC_SERVER_MODEL": model_file,
        "CGC_SERVER_EXPERT_CACHE_BYTES": str(pool_bytes),
        "CGC_SERVER_PORT": str(port),
        "CGC_SERVER_MTP": mtp,
    })
    if kwargs is not None:
        env["CGC_SERVER_CHAT_TEMPLATE_KWARGS"] = kwargs
    for kv in extra_env or []:
        k, _, v = kv.partition("=")
        env[k.strip()] = v.strip()
    log = open(log_path, "wb")
    subprocess.Popen(["bash", "scripts/run_server.sh"], cwd=ROOT, env=env,
                     stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                     start_new_session=True)



def server_procs():
    """[(pid, cmdline)] for real llama-server processes (see running_servers for the filter)."""
    out = []
    for p in sh("pgrep", "-f", SERVER_MATCH).split():
        if not p.strip():
            continue
        cmd = sh("ps", "-o", "command=", "-p", p).strip()
        if cmd and cmd.split(" ", 1)[0].endswith(SERVER_MATCH):
            try:
                out.append((int(p), cmd))
            except ValueError:
                pass
    return out



def port_owner(port):
    """PID holding a socket on `port`, or None when it is free / ambiguous."""
    pids = sorted({p for p in sh("lsof", "-ti", f":{port}").split() if p.strip()})
    if len(pids) != 1:
        return None
    try:
        return int(pids[0])
    except ValueError:
        return None



def wait_health(port, model_file, timeout, grace=60.0, pre_pids=()):
    """Seconds until OUR server answers /health, or None if it never came up.

    "No server process yet" is NOT death: `run_server.sh` needs seconds before its child is
    exec'd, and an early poll would abort the wait immediately -- which once made every combo
    report "server died" while the server was still loading.

    Identity is taken from the COMMAND LINE (exact model path + --port), never from a pid.
    Two measured failures forced that: `run_server.sh` prints `[detach] server PID=N` where N is
    not always the server's own pid, so a pid check rejected our own healthy servers; and on a
    shared checkout another agent respawns servers, so a FOREIGN already-loaded server can
    answer /health seconds after our launch and silently be measured instead of ours.
    """
    abs_model = model_file            # absolute; see launch()
    pre = set(pre_pids)
    t0 = time.time()
    seen = False
    while time.time() - t0 < timeout:
        procs = server_procs()
        ours = [p for p, c in procs if abs_model in c and f"--port {port}" in c]
        # A foreign llama-server anywhere means we cannot attribute /health to our config.
        foreign = [p for p, _ in procs if p not in ours]
        owner = port_owner(port)
        if not foreign and len(ours) == 1 and owner == ours[0] and ours[0] not in pre \
                and health(port):
            # Four independent facts: nobody else is loaded, exactly one process matches our
            # model AND port, that process OWNS the port, and it is newer than our launch.
            # Command-line matching alone was not enough on a shared checkout: the other agent
            # runs this same model on this same port, and a probe accepted their already-loaded
            # server 10 s after launch while our launch log still said "模型載入中".
            return time.time() - t0
        if procs:
            seen = True
        elif seen or (time.time() - t0) > grace:
            return None
        time.sleep(5)
    return None



def _read_text(path):
    try:
        return open(path, "rb").read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - harness
        return ""



def read_launch_facts(log_path):
    """Pull the actually-chosen template/model_kind/pool geometry out of the launch log.

    Two logs, one fact set. `run_server.sh` writes the [chat]/[log] headers to its own stdout,
    while the C++ side (n_slots, LAYER_CAPS min) goes to the log file it names in `[log]`. Both
    are needed: the pool path gate is arithmetic on those C++ numbers, so missing them made the
    union-fit gate answer "unknown" for every combo.

    The `[log]` path is cut at the first non-ASCII byte: the line ends with a Chinese
    parenthetical, and a naive `\S+` swallowed it, so the nil scan opened a nonexistent file and
    reported a clean `[]` forever.
    """
    facts = {}
    txt = _read_text(log_path)
    m = re.search(r"\[log\]\s+(\S+)", txt)
    if m:
        facts["log_path"] = re.split(r"[^\x20-\x7e]", m.group(1))[0]

    for source in (txt, _read_text(facts.get("log_path", ""))):
        if not source:
            continue
        for pat, key in FACT_PATTERNS:
            if key in facts:
                continue
            mm = re.search(pat, source)
            if mm:
                facts[key] = mm.groups() if len(mm.groups()) > 1 else mm.group(1)
    return facts
