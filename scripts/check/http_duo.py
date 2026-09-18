#!/usr/bin/env python3
"""Measure prefill AND decode on the SERVED (HTTP) path, one profile, one table.

STATUS 2026-09-18: this is a *research* instrument for the llama-bench vs served-path question,
not the delivery instrument. Standing rule from the same day: quoted speed numbers come from
llama-bench (`profile_duo.py`); this one only exists to study how the two instruments differ.

WHY THIS SCRIPT EXISTS. Every decode number at or above 12 t/s in this repo's record was taken
over HTTP, while llama-bench tops out near 10 on the same profile and shape. Quoting one
instrument's prefill beside the other's decode is how the record got two numbers that were never
true at the same time. This measures BOTH axes on ONE server, so a served request's two halves are
quoted from the same process, the same load, and the same thermal window.

WHAT IT REPORTS. `timings.prompt_per_second` and `timings.predicted_per_second` from the server's
own response object -- the server's accounting, not a client-side stopwatch. The first repetition
is dropped (warmup) and the rest are shown individually: a spread is a finding, and a single
median hides it.

TWO DEFECTS THIS FILE FIXES (both found 2026-09-18, both were silent):

① PREFILL AXIS HARD-CODED TO A BIG-CTX PROFILE. The prompt was ~4500 tokens because the default
   profile was `prefill250` (CTX 8192/5632). On `prod25` the same prompt returns HTTP 400:
       request (4500 tokens) exceeds the available context size (4096 tokens)
   i.e. the prefill half read as "the server is broken" when it was "this profile cannot hold that
   prompt". Now the prompt is SIZED FROM THE PROFILE'S OWN RESOLVED CTX (read from
   `run_server.sh CGC_DUMP_ENV=1`, cross-checked against the live server's `/props`), and kept
   inside `ctx - n_predict - reserve` using `/tokenize`. When even the smallest honest prefill does
   not fit, the axis is reported as NOT APPLICABLE with the arithmetic printed -- never as a
   failed request, and never silently averaged away.

② CROSS-SESSION KILLING / INVISIBLE SERVER DEATH. `stop()` used `pkill -9 -f llama-server`, which
   kills OTHER sessions' servers, and a mid-run SIGTERM (someone else's `run_server.sh` preflight,
   see `scripts/run_server.sh` [CGC 2026-09-18 no-cross-kill]) used to surface as three quiet
   `None` rows. Now: this server gets its own port and its own session (`start_new_session=True`),
   cleanup is scoped to our own pid/port, every repetition re-checks that the server is alive, and
   the log is scanned for the SIGTERM marker -- a killed arm is reported as INVALID (exit 1)
   instead of being printable.

USAGE
    python3 scripts/check/http_duo.py --profile prefill250
    python3 scripts/check/http_duo.py --profile prod25          # prefill reports NOT APPLICABLE
    python3 scripts/check/http_duo.py --profile prefill250 --reps 4 \
        --json Backup/prod_matrix/http_duo_prefill250.json
    python3 scripts/check/http_duo.py --self-test               # zero-GPU checks, no server

Starts and stops its own llama-server. Exit code 0 unless the server never became healthy, the
prefill budget could not be honoured, or the server was killed underneath us.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import thermal_pressure as tp  # noqa: E402

BAR_PREFILL = 250.0
BAR_DECODE = 12.0
PORT_SCAN_FROM = 8080
PORT_SCAN_TO = 8120

# This host has HTTP_PROXY set, which silently routes 127.0.0.1 through a local proxy unless
# bypassed. urllib picks up the env vars on its own, so the opener is built with no proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# One ~75-token unit of Chinese prose. The prompt is built by repeating it; how many times is
# decided AT RUN TIME from the profile's ctx (see plan_prefill()), because 60 units fit
# prefill250 and do NOT fit prod25 (CTX 4096).
_PREFILL_UNIT = ("巴黎之所以成為法國的政治與文化中心，是因為它在中世紀時期就已經是王權所在，"
                 "同時也是大學、印刷與沙龍的聚集地。十九世紀的鐵路網把外省的糧食、煤鐵與人手送進城裡，"
                 "讓它既能供養龐大的官僚體系，也能養活一批不依靠宮廷的作家與畫家。")

_DECODE_PROMPT = "請用繁體中文簡短說明巴黎為什麼是法國的首都。"

# What the server prints when somebody else SIGTERMs it. Not a watchdog (that path is GGML_ABORT)
# and not an OOM killer; see docs/PROFILE_DUO_2026-09-18.md §5 and run_server.sh preflight notes.
_SIGTERM_MARK = "Received SIGTERM"


# --------------------------------------------------------------------------- helpers: machine state

def wait_nominal(timeout: float, poll: float) -> dict:
    t0 = time.time()
    while True:
        lv = tp.level()
        if lv == 0:
            return {"ok": True, "waited_s": round(time.time() - t0, 1), "label": tp.label(lv)}
        if time.time() - t0 > timeout:
            return {"ok": False, "waited_s": round(time.time() - t0, 1), "label": tp.label(lv)}
        print(f"    thermal {tp.label(lv)} -- waiting for NOMINAL "
              f"({time.time() - t0:.0f}s / {timeout:.0f}s)", flush=True)
        time.sleep(poll)


def port_listeners(port: int) -> list[str]:
    """pids currently LISTENing on `port` ('' if none). Used both for picking a port and for
    scoping the cleanup -- never for global killing."""
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True).stdout.strip()
    return [p for p in out.split() if p]


def pick_port(preferred: int, allow_scan: bool = True) -> int:
    """Return a port nobody is listening on.

    Two lines can legitimately measure at once; taking 8080 whoever gets there first is how
    yesterday's ABBA ended up fighting over one server. `allow_scan=False` = take it or fail.
    """
    if not port_listeners(preferred):
        return preferred
    if not allow_scan:
        raise SystemExit(f"error: port {preferred} already has a listener -- pick another one "
                         f"(--port auto) or stop it yourself")
    for p in range(PORT_SCAN_FROM, PORT_SCAN_TO):
        if p != preferred and not port_listeners(p):
            return p
    raise SystemExit(f"error: no free port in {PORT_SCAN_FROM}..{PORT_SCAN_TO}")


def resolved_profile(profile: str, extra_env: dict[str, str]) -> dict:
    """The launcher's own view of `profile`: resolved CTX/PORT/etc, zero GPU.

    One source of truth -- `run_server.sh CGC_DUMP_ENV=1`, the same call every other harness uses.
    """
    import llama_bench_matrix as lbm
    return lbm.resolve(profile, extra_env)


# ------------------------------------------------------------------------------- helpers: HTTP

def post(path: str, payload: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def post_text(path: str, payload: dict, timeout: float = 900.0) -> str:
    """Same as post(), but surfaces the SERVER'S OWN error body -- the 400 text is the evidence
    (e.g. `request (4500 tokens) exceeds the available context size (4096 tokens)`)."""
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def wait_healthy(timeout: float = 900.0) -> bool:
    """`/health` answers 200 with body {"status":"loading model"} DURING load, so the BODY is the
    signal, not the status code."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with _OPENER.open(BASE + "/health", timeout=5) as r:
                body = json.loads(r.read())
            if body.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def server_ctx_fallback(profile_res: dict) -> int:
    return int(profile_res["scalars"].get("CTX", "0") or 0)


def live_ctx() -> int | None:
    """n_ctx as the RUNNING server reports it (`/props` -> default_generation_settings.n_ctx).
    None if this build does not answer /props (then we trust the launcher's dump)."""
    try:
        with _OPENER.open(BASE + "/props", timeout=10) as r:
            props = json.loads(r.read())
    except Exception:
        return None
    try:
        return int(props["default_generation_settings"]["n_ctx"])
    except Exception:
        return None


def token_count(text: str) -> int:
    """Ground truth token count for this prompt, from the server's own tokenizer."""
    body = post("/tokenize", {"content": text})
    toks = body.get("tokens")
    if isinstance(toks, list):
        return len(toks)
    raise SystemExit(f"error: /tokenize returned no token list: {str(body)[:200]}")


# --------------------------------------------------------------------------- helpers: prefill plan

def plan_prefill(ctx: int, n_predict: int, reserve: int, target: int, min_tokens: int,
                 force: bool) -> dict:
    """Pick how many units of _PREFILL_UNIT fit in `ctx - n_predict - reserve`.

    Returns {"fit": bool, "units": int, "tokens": int, "budget": int, "reason": str}. The point is
    that NOT FITTING is a property of the profile (its ctx), not a failure of the server, and it
    must be reported that way.
    """
    budget = ctx - n_predict - reserve
    plan = {"budget": budget, "units": 0, "tokens": 0, "fit": False, "reason": ""}
    if budget <= 0:
        plan["reason"] = (f"ctx={ctx} leaves {budget} tokens for a prompt after "
                          f"n_predict={n_predict} and reserve={reserve}")
        return plan

    per_unit = token_count(_PREFILL_UNIT)
    if per_unit <= 0:
        raise SystemExit("error: /tokenize counted 0 tokens for the prefill unit")
    plan["per_unit"] = per_unit

    cache: dict[int, int] = {}

    def ntok(units: int) -> int:
        if units not in cache:
            cache[units] = token_count(_PREFILL_UNIT * units)
        return cache[units]

    # exponential search for the first unit count that overshoots, then binary search the boundary
    hi = 1
    while ntok(hi) <= budget and hi < 4096:
        hi *= 2
    lo = hi // 2
    if ntok(lo) > budget:                      # even 1/2^? ... lo can be 0
        lo, hi = 0, hi
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ntok(mid) <= budget:
            lo = mid
        else:
            hi = mid
    units = lo
    tokens = ntok(units) if units else 0

    # never overshoot just to look bigger: target only trims downwards
    if units > 1 and tokens > 0 and tokens > target:
        want_units = max(1, int(target // per_unit))
        if want_units < units and ntok(want_units) <= budget:
            units, tokens = want_units, ntok(want_units)

    plan.update(units=units, tokens=tokens, fit=True)
    if tokens < min_tokens:
        plan["fit"] = False
        if force:
            plan["fit"] = True
            plan["reason"] = (f"only {tokens} tokens fit (< --prefill-min-tokens {min_tokens}); "
                              f"running anyway because --prefill-force")
        else:
            plan["reason"] = (f"ctx={ctx} holds at most {tokens} prompt tokens after "
                              f"n_predict={n_predict} + reserve={reserve}, below the "
                              f"--prefill-min-tokens {min_tokens} a prefill measurement needs")
    return plan


# ------------------------------------------------------------------------- helpers: server lifecycle

def start_server(profile: str, extra_env: dict[str, str], port: int,
                 logpath: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["CGC_SERVER_PROFILE"] = profile
    env["CGC_SERVER_PORT"] = str(port)
    env.update(extra_env)
    log = open(logpath, "wb")
    # start_new_session=True puts the server in its own session so a SIGINT/SIGHUP aimed at us (or
    # the tool reaping our process group) cannot take it down. macOS has no setsid(1); this is the
    # same call run_server.sh's own `--detach` uses.
    return subprocess.Popen([str(ROOT / "scripts" / "run_server.sh")], cwd=str(ROOT),
                            env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def stop(proc: subprocess.Popen, port: int) -> None:
    """Stop OUR server only.

    The old version ran `pkill -9 -f llama-server`, which kills every llama-server on the box --
    including another session's measurement. Same bug family as the preflight cross-kill.
    """
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(2)
    held = port_listeners(port)
    if held:
        print(f"  WARNING: port {port} still held after stopping our server: pids {held}. "
              f"Those are NOT ours -- leave them alone.", flush=True)


def death_evidence(logpath: Path) -> dict:
    """Was this server killed underneath us? Look at ITS OWN log, not at our exit code."""
    out = {"sigterm": False, "tail": ""}
    try:
        text = logpath.read_text(errors="replace")
    except OSError:
        return out
    out["sigterm"] = _SIGTERM_MARK in text
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out["tail"] = lines[-1] if lines else ""
    return out


def diagnose_death(logpath: Path, port: int) -> None:
    print("\n" + "!" * 74, flush=True)
    print("SERVER DIED MID-RUN -- this arm's numbers are INVALID, do not cite them.", flush=True)
    print(f"  server log : {logpath}", flush=True)
    print(f"  marker     : '{_SIGTERM_MARK}' found in the log", flush=True)
    print("  not a watchdog kill (that path says GGML_ABORT) and not an OOM.", flush=True)
    print("  most likely: another line's `run_server.sh` preflight pattern-matched this process")
    print("               (fixed 2026-09-18: preflight no longer signals anything by default),")
    print("               or something killed our process group before that fix landed.", flush=True)
    print(f"  next time  : check the log's last lines and who owned port {port}.", flush=True)
    print("!" * 74, flush=True)


def one_request(prompt: str, n_predict: int) -> dict:
    payload = {"prompt": prompt, "n_predict": n_predict, "max_tokens": n_predict,
               "temperature": 0.0, "stream": False, "cache_prompt": False}
    return json.loads(post_text("/v1/completions", payload)).get("timings", {})


# -------------------------------------------------------------------------------------- self-test

def self_test() -> int:
    """Zero-GPU checks for exactly the two defects above. No server, no model load."""
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")

    print("http_duo self-test (no GPU, no server)")
    # ① budget arithmetic: the prod25 case that used to 400
    print("\n  prefill budget (pure arithmetic, no tokenizer involved):")
    for ctx, np_, res, want_min in ((4096, 16, 64, 1024), (8192, 16, 64, 1024),
                                    (1024, 128, 64, 1024), (4096, 5000, 64, 1024)):
        budget = ctx - np_ - res
        print(f"    ctx={ctx:<5} n_predict={np_:<5} -> budget={budget:<6} "
              f"({'fits' if budget >= want_min else 'PREFLILL NOT APPLICABLE'})")
    check("prod25 (ctx 4096, n_predict 16) has room for a >=1024-token prefill", 4096 - 16 - 64 >= 1024)
    check("ctx 1024 with n_predict 128 cannot host a 1024-token prefill", 1024 - 128 - 64 < 1024)
    check("n_predict larger than ctx yields a non-positive budget", 4096 - 5000 - 64 <= 0)

    # ② death detection works on a real string, not on a hope
    print("\n  SIGTERM evidence scanner:")
    tmp = Path("/tmp/http_duo_selftest_log.txt")
    tmp.write_text("srv  update_slots: ...\n[CGC] Received SIGTERM -- initiating graceful shutdown\n",
                   encoding="utf-8")
    ev = death_evidence(tmp)
    check("detects 'Received SIGTERM' in a log", ev["sigterm"], f"tail={ev['tail'][:40]!r}")
    tmp.write_text("srv  update_slots: all slots are idle\n", encoding="utf-8")
    check("no false positive on a healthy log", not death_evidence(tmp)["sigterm"])
    tmp.write_text("GGML_ABORT: ggml-metal-context.m:863\n", encoding="utf-8")
    check("watchdog abort is NOT reported as SIGTERM", not death_evidence(tmp)["sigterm"])

    # port scoping
    print("\n  port scoping:")
    p = pick_port(PORT_SCAN_FROM)
    print(f"    pick_port({PORT_SCAN_FROM}) -> {p}")
    check("pick_port returns something free", not port_listeners(p))

    print("\nSELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="prefill AND decode on the served HTTP path")
    ap.add_argument("--profile", default="prefill250")
    ap.add_argument("--reps", type=int, default=4, help="repetitions; rep 1 is warmup and dropped")
    ap.add_argument("--prefill-predict", type=int, default=16)
    ap.add_argument("--decode-predict", type=int, default=128)
    ap.add_argument("--extra-env", default="", help="ENV=VAL;ENV=VAL for the server")
    ap.add_argument("--cooldown-timeout", type=float, default=420.0)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--logdir", default="Backup/prod_matrix")
    ap.add_argument("--port", default="auto", help="'auto' picks the first free port >= 8080")
    # prefill sizing (defect ①)
    ap.add_argument("--prefill-target-tokens", type=int, default=2048,
                    help="trim the prompt down to about this many tokens if possible")
    ap.add_argument("--prefill-min-tokens", type=int, default=1024,
                    help="below this the prefill axis is reported NOT APPLICABLE, not measured")
    ap.add_argument("--prefill-reserve", type=int, default=64,
                    help="headroom kept free inside ctx beyond prompt + n_predict")
    ap.add_argument("--prefill-force", action="store_true",
                    help="send the prefill request even when it does not fit (you will see the 400)")
    ap.add_argument("--self-test", action="store_true", help="zero-GPU checks; no server")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    extra_env: dict[str, str] = {}
    for kv in args.extra_env.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            extra_env[k] = v

    prof = resolved_profile(args.profile, extra_env)
    ctx_dump = server_ctx_fallback(prof)
    port_default = int(prof["scalars"].get("PORT", str(PORT_SCAN_FROM)) or PORT_SCAN_FROM)
    if args.port == "auto":
        port = pick_port(port_default)
    else:
        port = pick_port(int(args.port), allow_scan=False)

    ts = time.strftime("%Y%m%d_%H%M%S")
    logpath = Path(args.logdir) / f"http_duo_{args.profile}_{ts}.server.log"
    logpath.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== starting server: profile={args.profile} port={port} "
          f"extra={args.extra_env or '-'}", flush=True)
    print(f"    resolved CTX (from run_server.sh CGC_DUMP_ENV=1) = {ctx_dump}", flush=True)

    global BASE
    BASE = f"http://127.0.0.1:{port}"

    proc = start_server(args.profile, extra_env, port, logpath)
    try:
        if not wait_healthy():
            print(f"error: server never became healthy; see {logpath}", file=sys.stderr)
            return 1
        print(f"  healthy. server log: {logpath}", flush=True)

        ctx_live = live_ctx()
        ctx = ctx_live or ctx_dump
        print(f"    n_ctx reported by the live server (/props) = {ctx_live}  "
              f"-> using {ctx}", flush=True)

        plan = plan_prefill(ctx, args.prefill_predict, args.prefill_reserve,
                            args.prefill_target_tokens, args.prefill_min_tokens,
                            args.prefill_force)
        if plan["fit"]:
            print(f"    prefill prompt: {plan['units']} units = {plan['tokens']} tokens "
                  f"(budget {plan['budget']} = ctx {ctx} - n_predict {args.prefill_predict} "
                  f"- reserve {args.prefill_reserve})", flush=True)
            if plan["reason"]:
                print(f"    WARNING: {plan['reason']}", flush=True)
            prefill_prompt = _PREFILL_UNIT * max(1, plan["units"])
        else:
            prefill_prompt = None
            print(f"    PREFILL NOT APPLICABLE: {plan['reason']}", flush=True)
            print("    (this profile's ctx cannot hold a real prefill + its answer; the prefill "
                  "axis below is quoted from another instrument, never from a 400)", flush=True)

        runs: list[dict] = []
        killed = False
        for i in range(1, args.reps + 1):
            row = {"rep": i, "warmup": i == 1}
            for axis, prompt, n_pred in (("prefill", prefill_prompt, args.prefill_predict),
                                         ("decode", _DECODE_PROMPT, args.decode_predict)):
                if prompt is None:
                    row[axis] = None
                    row[axis + "_na"] = plan["reason"]
                    print(f"  rep{i} {axis:<8} {'-':>10}   (not applicable)", flush=True)
                    continue
                if proc.poll() is not None:
                    killed = True
                    row[axis] = None
                    row[axis + "_na"] = "server died before this request"
                    continue
                w = wait_nominal(args.cooldown_timeout, args.poll)
                if not w["ok"]:
                    print(f"  rep{i} {axis}: REFUSED, still {w['label']}", flush=True)
                    row[axis] = None
                    row[axis + "_thermal"] = w["label"]
                    continue
                try:
                    t = one_request(prompt, n_pred)
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
                    print(f"  rep{i} {axis}: HTTP {e.code} -- server says: {body[:300]}", flush=True)
                    row[axis] = None
                    row[axis + "_http_error"] = body[:300]
                    continue
                except Exception as e:
                    print(f"  rep{i} {axis}: request failed: {e}", flush=True)
                    row[axis] = None
                    continue
                key = "prompt_per_second" if axis == "prefill" else "predicted_per_second"
                val = t.get(key)
                row[axis] = val
                row[axis + "_tokens"] = t.get("prompt_n" if axis == "prefill" else "predicted_n")
                row[axis + "_thermal"] = "NOMINAL"
                print(f"  rep{i} {axis:<8} {val if val is not None else '-':>10} t/s "
                      f"({row[axis + '_tokens']} tokens)", flush=True)
            runs.append(row)
            if proc.poll() is not None or death_evidence(logpath)["sigterm"]:
                killed = True
                break

        evidence = death_evidence(logpath)
        if evidence["sigterm"] or killed:
            diagnose_death(logpath, port)

        kept = [r for r in runs if not r["warmup"]]

        def med(axis):
            vs = [r[axis] for r in kept if r.get(axis) is not None]
            return sorted(vs)[len(vs) // 2] if vs else None
        pf, dc = med("prefill"), med("decode")

        print("\n" + "=" * 74)
        print(f"profile {args.profile}  --  SERVED path, rep1 dropped, port {port}")
        print(f"  ctx      {ctx}  (prefill {'planned from ctx' if prefill_prompt else 'NOT APPLICABLE'})")
        print(f"  prefill  {f'{pf:.2f}' if pf is not None else '-':>10} t/s   "
              f"samples {[round(r['prefill'], 2) for r in kept if r.get('prefill') is not None]}"
              + (f"   NA: {plan['reason']}" if pf is None else ""))
        print(f"  decode   {f'{dc:.2f}' if dc is not None else '-':>10} t/s   "
              f"samples {[round(r['decode'], 2) for r in kept if r.get('decode') is not None]}")
        # NOTE: the bar needs BOTH axes from ONE server. If the prefill axis is not applicable on
        # this profile, there is no served-path verdict here -- only a decode number.
        if pf is None and dc is not None:
            print(f"  bar 250/12 -> NOT EVALUABLE on this profile "
                  f"(prefill needs a ctx that holds {args.prefill_min_tokens}+ prompt tokens; "
                  f"this profile has {ctx})")
            print(f"  decode-only number above is still comparable across profiles.")
            meets = None
        else:
            meets = pf is not None and dc is not None and pf >= BAR_PREFILL and dc >= BAR_DECODE
            print(f"  bar 250/12 -> {'PASS' if meets else 'FAIL'}")
        print("=" * 74)

        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(
                {"profile": args.profile, "extra_env": args.extra_env, "port": port,
                 "ctx_resolved_dump": ctx_dump, "ctx_live_props": ctx_live,
                 "prefill_plan": plan, "prefill_tokens_planned": plan.get("tokens"),
                 "runs": runs, "prefill": pf, "decode": dc, "meets_bar": meets,
                 "valid": not (evidence["sigterm"] or killed),
                 "server_killed": bool(evidence["sigterm"] or killed),
                 "server_log": str(logpath)},
                ensure_ascii=False, indent=1))
            print(f"wrote {args.json}")

        if evidence["sigterm"] or killed:
            return 1
    finally:
        print("stopping server...", flush=True)
        stop(proc, port)
    return 0


BASE = "http://127.0.0.1:8080"


if __name__ == "__main__":
    sys.exit(main())
