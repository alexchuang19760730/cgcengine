#!/usr/bin/env python3
"""Pool-size sweep: restart the server at one expert-pool budget, probe, record one row.

Motivation: every pool-size A/B taken before 2026-09-11 measured the legacy ZERO-slot path
*and* a `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=0` flag that the code read as ENABLED (non-null
gate), so blk.0 was out of the pool and got an identity remap. With both fixed the small-pool
numbers change completely (4GB went from "10/10 echoes the prompt" to "10/10 answers 42"), so
the whole memory/quality frontier has to be re-measured.

One pool size per invocation (a full sweep exceeds a single command's time budget):

  python3 scripts/check/pool_curve.py --pool 4 --shots 6 --out /tmp/pool_curve.json

Rows accumulate in --out keyed by pool GB, so the sweep can be run pool-by-pool and the
final table printed with --report.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_skip0 import PROMPTS, ask  # noqa: E402

LOG_DIR = os.path.join(ROOT, "Backup", "cgc_logs")
SERVER_MATCH = "build/bin/llama-server"
DEFAULT_MODEL = os.path.join(ROOT, "models", "gguf", "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")
SERVER_BIN = os.path.join(ROOT, "src", "llama.cpp", "build", "bin", "llama-server")


def _file_digest(path, sample_size=65536):
    """Size + first/last 64KiB hash. Same sampling philosophy as knifeedge_matrix:
    a full sha256 of a 13GB file is too slow for every row, and the head/tail sample
    catches every real-world swap (different quantization, different metadata)."""
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(sample_size))
        if size > sample_size * 2:
            f.seek(-sample_size, 2)
            h.update(f.read(sample_size))
    return {"size": size, "head_hash": h.hexdigest()[:16]}


def record_provenance(pool_gb, model_path, extra_env, facts=None):
    """Simplified provenance for pool_curve rows. The full version lives in
    knifeedge_matrix.record_provenance; this one covers the fields that make
    two pool_curve rows comparable (or not): model identity, binary identity,
    launch env, and the predicted-vs-launched slot geometry.

    Stored inside every row so a curve taken last week cannot be silently
    compared with one taken today after a loader change.
    """
    pool = {"gb": int(pool_gb), "bytes": int(pool_gb) * 1024 ** 3}
    # Predicted slots: try the shared feasibility cell; fall back to None if
    # knifeedge_matrix cannot be imported (it has heavy deps).
    predicted = None
    try:
        from knifeedge_matrix import feasibility_cell  # noqa: WPS433
        cell = feasibility_cell("generic", pool_gb, None, list(extra_env or []))
        predicted = cell.get("min_layer_slots")
        pool.update({"min_layer_slots": predicted,
                     "usable_slots": cell.get("usable_slots"),
                     "cap_max": cell.get("cap_max"),
                     "verdict": cell.get("verdict")})
    except Exception:  # noqa: BLE001 - provenance must never kill a measurement
        pass
    if facts:
        launched = facts.get("min_layer_slots")
        pool["launched_min_layer_slots"] = launched
        pool["launched_n_slots"] = facts.get("n_slots")
        if predicted is not None and launched is not None:
            mismatch = int(predicted) != int(launched)
            pool["geometry_mismatch"] = mismatch
            if mismatch:
                pool["geometry_mismatch_detail"] = (
                    f"predicted {predicted} vs launched {launched}")
        else:
            pool["geometry_mismatch"] = None
    return {"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pool": pool,
            "model": {"realpath": os.path.realpath(model_path),
                      **(_file_digest(model_path) or {})},
            "binary": {"path": SERVER_BIN, **(_file_digest(SERVER_BIN) or {})},
            "launch": {"extra_env": sorted(extra_env or [])}}


def kill_server():
    subprocess.run(["pkill", "-9", "-f", SERVER_MATCH], capture_output=True)
    time.sleep(3)


def mem_free_pct():
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else -1


def pid():
    out = subprocess.run(["pgrep", "-f", SERVER_MATCH], capture_output=True, text=True).stdout.split()
    return out[0] if out else None


def rss_gb():
    p = pid()
    if not p:
        return 0.0
    out = subprocess.run(["ps", "-o", "rss=", "-p", p], capture_output=True, text=True).stdout.strip()
    return int(out) / 1048576 if out else 0.0


def start(pool_bytes, model, port, extra_env=None):
    env = dict(os.environ)
    env.update({
        "CGC_DETACHED": "1",
        "CGC_SERVER_EXPERT_CACHE_BYTES": str(pool_bytes),
        "CGC_SERVER_MODEL": model,
        "CGC_SERVER_PORT": str(port),
    })
    for kv in (extra_env or []):
        k, _, v = kv.partition("=")
        env[k.strip()] = v.strip()
    with open("/tmp/pool_curve_launch.log", "ab") as logf:
        # start_new_session: own session so a SIGINT/SIGHUP aimed at us (or the caller reaping our
        # process group) cannot take the server down mid-measurement. Same reason run_server.sh
        # grew `--detach`; see the cross-kill notes in docs/PROFILE_DUO_2026-09-18.md §5.
        subprocess.run(["bash", "scripts/run_server.sh"], cwd=ROOT, env=env,
                       stdout=logf, stderr=logf, start_new_session=True)
    time.sleep(3)


def wait_health(port, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if b"ok" in r.read():
                    return True
        except Exception:  # noqa: BLE001 - polling
            pass
        time.sleep(5)
    return False


def latest_log():
    if not os.path.isdir(LOG_DIR):
        return None
    files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.startswith("llama_server_")]
    return max(files, key=os.path.getmtime) if files else None


def log_facts(path):
    facts = {}
    if not path or not os.path.exists(path):
        return facts
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"L4 metal pool capacity=(\d+)", txt)
    facts["capacity"] = int(m.group(1)) if m else None
    m = re.search(r"Soft Pool init: L0=(\d+) L1=(\d+) \(n_slots=(\d+)", txt)
    if m:
        facts.update(l0=int(m.group(1)), l1=int(m.group(2)), n_slots=int(m.group(3)))
    facts["pool_lines"] = txt.count("-> GPU pool buffer")
    facts["blk0_pool_lines"] = len(re.findall(r"blk\.0\.ffn_\w+_exps\.weight -> GPU pool buffer", txt))
    facts["skip_load_lines"] = txt.count("out of GPU buffers (skip-load")
    facts["ident_verify"] = bool(re.search(r"CGC-IDENT-VERIFY", txt))
    facts["exact_verify_clean"] = bool(re.search(r"CGC-EXACT-VERIFY:.*mismatched=0", txt))
    # last decode timing block emitted by the server for the probe requests
    evals = re.findall(r"eval time =\s*[\d.]+ ms /\s*\d+ tokens \(\s*([\d.]+) tokens per second\)", txt)
    facts["decode_tps_server"] = float(evals[-1]) if evals else None
    return facts


def probe(url, prompt, shots, max_tokens):
    outs, speeds, finishes = [], [], []
    for _ in range(shots):
        try:
            content, ct, dt, finish = ask(url, prompt, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 - probe tool
            outs.append(f"<ERROR {exc}>")
            continue
        outs.append(content)
        speeds.append(ct / dt if dt > 0 else 0.0)
        finishes.append(finish)
    return outs, speeds, finishes


def run_row(args):
    pool_bytes = args.pool * (1 << 30)
    kill_server()
    free_before = mem_free_pct()
    start(pool_bytes, args.model, args.port, args.extra_env)
    up = wait_health(args.port)
    log_path = latest_log()
    row = {"pool_gb": args.pool, "row_key": args.row_key or str(args.pool),
           "up": up, "free_before_pct": free_before, "extra_env": args.extra_env}
    if not up:
        row["error"] = "server never became healthy"
        kill_server()
        return row

    zh_outs, zh_speeds, zh_fin = probe(f"http://127.0.0.1:{args.port}", PROMPTS["zh"], args.shots, args.max_tokens)
    en_outs, en_speeds, en_fin = probe(f"http://127.0.0.1:{args.port}", PROMPTS["en"], max(2, args.shots // 3), args.max_tokens)

    row.update({
        "rss_gb": round(rss_gb(), 2),
        "free_after_pct": mem_free_pct(),
        "zh_shots": len(zh_outs),
        "zh_correct": sum(1 for o in zh_outs if "42" in o),
        "zh_unique": len(set(zh_outs)),
        "zh_echo_prompt": sum(1 for o in zh_outs if o.strip() in ("15+27", "'15+27'", PROMPTS["zh"])),
        "zh_mean_tps": round(sum(zh_speeds) / len(zh_speeds), 2) if zh_speeds else None,
        "zh_finish_reasons": sorted(set(zh_fin)),
        "zh_sample": zh_outs[0] if zh_outs else None,
        "en_correct": sum(1 for o in en_outs if "42" in o),
        "en_shots": len(en_outs),
        "en_sample": en_outs[0] if en_outs else None,
    })
    facts = log_facts(log_path)
    row.update(facts)
    row["log"] = os.path.basename(log_path) if log_path else None
    # Provenance: stamped AFTER the facts are read so it can carry both the
    # predicted and launched slot counts -- a geometry mismatch is then visible
    # per row, not just via a one-shot cap probe.
    row["provenance"] = record_provenance(args.pool, args.model, args.extra_env, facts=facts)
    kill_server()
    row["free_teardown_pct"] = mem_free_pct()
    return row


def save(out_path, row):
    data = {}
    if os.path.exists(out_path):
        try:
            data = json.load(open(out_path, encoding="utf-8"))
        except Exception:  # noqa: BLE001 - tolerate a partial file
            data = {}
    data[row.get("row_key") or str(row["pool_gb"])] = row
    json.dump(data, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def report(out_path):
    data = json.load(open(out_path, encoding="utf-8"))
    print(f"{'row':>8} {'n_slots':>8} {'RSS GB':>7} {'free%':>6} {'zh ok':>6} {'uniq':>5} "
          f"{'echo':>5} {'t/s(srv)':>9} {'geom':>6} {'finish':>12}")

    def sort_key(k):
        m = re.match(r"\d+", k)
        return (int(m.group()) if m else 999, k)

    for key in sorted(data, key=sort_key):
        r = data[key]
        if not r.get("up"):
            print(f"{key:>8} {'—':>8} {'—':>7} {'—':>6} {'DOWN':>6}")
            continue
        prov = r.get("provenance") or {}
        gm = (prov.get("pool") or {}).get("geometry_mismatch")
        gms = "n/a" if gm is None else ("MIS" if gm else "ok")
        print(f"{key:>8} {str(r.get('n_slots')):>8} {r.get('rss_gb'):>7} {r.get('free_after_pct'):>6} "
              f"{str(r.get('zh_correct')) + '/' + str(r.get('zh_shots')):>6} {r.get('zh_unique'):>5} "
              f"{r.get('zh_echo_prompt'):>5} {str(r.get('decode_tps_server')):>9} "
              f"{gms:>6} {','.join(r.get('zh_finish_reasons') or []):>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, help="expert pool budget in GiB (single-pool run)")
    ap.add_argument("--pools", help="comma list, e.g. 2,3,4,6,8 (runs them sequentially)")
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="/tmp/pool_curve.json")
    ap.add_argument("--row-key", help="override the row key (e.g. '8-exact'); default = pool GB")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="extra KEY=VAL for run_server.sh (repeatable), e.g. CGC_SERVER_WARM_NPAST=999999")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        report(args.out)
        return

    pools = [args.pool] if args.pool else [int(x) for x in (args.pools or "4").split(",")]
    for gb in pools:
        args.pool = gb
        row = run_row(args)
        save(args.out, row)
        print(json.dumps({k: v for k, v in row.items() if k != "log"}, ensure_ascii=False), flush=True)
    print("\n--- accumulated ---")
    report(args.out)


if __name__ == "__main__":
    main()
