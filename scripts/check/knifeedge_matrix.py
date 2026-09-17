#!/usr/bin/env python3
#!/usr/bin/env python3
"""Anti-knife-edge evaluation matrix: (model) x (pool size).

WHY THIS EXISTS
---------------
Single greedy questions on this stack are NOT a measurement. Two runs of the same question
can land in different trajectories (answer vs echo/loop) because a ~0.02 logit offset —
which comes from which MoE path/layout a step took — is enough to flip a near-tie about ten
tokens in. Ranking pool sizes on one greedy question measures luck.

`flip_rate.py` already makes that uncertainty explicit (N greedy repeats + M fixed seeds +
per-question flip statistics + suite variance). What was missing is the **model dimension**:
the same harness must be runnable for iq3 and iq4 without hand-editing anything, because the
quality and the correct chat template differ per model.

This driver adds that, plus the operational hygiene the earlier round was missing:

  * EXCLUSIVITY - refuses to measure while another llama-server is running. Earlier pool
    numbers were taken while other servers were live, which corrupts both the memory and the
    speed readings. Use --kill-existing to clear them explicitly.
  * RESUME      - a combo whose result JSON already exists is skipped, so a long sweep
    survives an interruption instead of restarting from zero.
  * TEMPLATE    - launches with `enable_thinking=false` (see below) and records the
    `[chat] model_kind=` line the server actually chose, so a silent template mismatch is
    visible in the results rather than assumed away.
  * SPEED       - records RSS / free% / decode tok/s next to the pass rate, because the pool
    floor is a quality x speed trade, not a quality alone.

TEMPLATE NOTE (2026-09-11, verified): the GGUF embedded template gates its no-think scaffold
on `enable_thinking`:

    {%- if enable_thinking is defined and enable_thinking is false %}
        '<think>\n\n</think>\n\n'   <- closed   |   {%- else %} '<think>\n'  <- unclosed

If the launcher does not pass it, every generation prompt ends with an UNCLOSED marker and
the model degenerates into a deterministic prompt echo that never stops
(`finish=length`, no answer). Verified on a live IQ4_XS 8GB-pool server: content was the user
text repeated to the token cap.

`--chat-template-kwargs '{"enable_thinking": false}'` is therefore MANDATORY, and it must be
set at LAUNCH: this build does NOT read `chat_template_kwargs` from the request body -- three
request variants (none / enable_thinking=false / +assistant_prefill) came back byte-identical.
A client cannot repair this; see docs/CHAT_SCAFFOLD_ROOTCAUSE_2026-09-11.md §7.

`assistant_prefill` is MANDATORY TOO (2026-09-11) and is why the default kwargs carry both keys.
`run_server.sh` only applies its own anchor when chat-template-kwargs is EMPTY, so a harness that
passes `enable_thinking` alone silently DISABLES the anchor. On iq4 that produced
`completion_tokens: 1` with empty content at every pool size -- which reads like a pool-quality
bug and is not one.

Override with --kwargs '' only if a template genuinely does not need it.

Usage:
    # smoke test the harness end-to-end on both models, one question each
    python3 scripts/check/knifeedge_matrix.py --pilot

    # real sweep
    python3 scripts/check/knifeedge_matrix.py --models iq3,iq4 --pools 4,6,8 \\
        --per-profile 2 --greedy-repeats 3 --repeats 5 --temperature 0.4

    # add a config dimension to the A/B (repeatable)
    python3 scripts/check/knifeedge_matrix.py --models iq3 --pools 8 \\
        --extra-env CGC_LOOP_GUARD=1
"""

from __future__ import annotations

# ---- 這個檔案是 shim ------------------------------------------------
# 內容自 2026-09-17（E4 item 2）起住在 ./knifeedge/，分成 10 個模組。這裡只做三件事：
#   1. 把自己所在的目錄放進 sys.path —— 呼叫者用 import、用 spec_from_file_location、
#      或用 `python3 <這個路徑>` 都必須能解析到 ./knifeedge/。
#   2. 逐一 re-export（含 _ 開頭的私有名），讓 dir() 與原檔一致。
#   3. 轉發 __main__。
# 這一層是「錨點」而不是轉運站：__doc__ 保留原文（呼叫者的字串斷言才不會漂）。
# --------------------------------------------------------------------
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from knifeedge import (  # noqa: E402,F401
    BINARY_DIR,
    CAP_INVARIANCE_DEFAULT_CAPS,
    DEFAULT_PORT,
    ENTRY_FILE,
    EXPERT_CACHE_NOHOOK,
    EXPERT_CACHE_OFF,
    EXPERT_CACHE_OFF_NOGATHER,
    EXPERT_CACHE_ON,
    FACT_PATTERNS,
    GATHER_RE,
    GGUF,
    GROUND_TRUTH_STATES,
    KEY_FIELDS,
    MIN_CAP,
    MODELS,
    MODEL_SAMPLE_BYTES,
    NIL_RE,
    NUMERIC_SOURCES,
    POOL_GEOMETRY_SOURCES,
    PROBE_PROMPT,
    RESULT_DIR,
    ROOT,
    SERVER_MATCH,
    _FEAS_GEOM,
    _FEAS_MOD,
    _as_int,
    _cap_guard,
    _cap_probe_provenance,
    _cap_sidecar,
    _capinv_guard,
    _env_int,
    _feasibility,
    _first_divergence,
    _git_out,
    _model_digest_full,
    _model_guard,
    _oracle_guard,
    _oracle_meta,
    _oracle_rows,
    _read_text,
    _realign_decode,
    _recorded_slots,
    _run_oracle_compare,
    _short_val,
    _stamp_guard,
    _store_capinv,
    _truth_guard,
    _write_oracle_meta,
    apply_template_probe,
    ask_probe,
    binary_stamp,
    cap_for,
    cap_invariance_dump,
    cap_invariance_gate,
    cap_probe_reusable,
    combo_label,
    comparability_verdict,
    effective_launch_env,
    expert_cache_state,
    feasibility_cell,
    feasibility_gate,
    gate_model_identity,
    geometry_conflict_error,
    health,
    key_fingerprint,
    kill_servers,
    launch,
    launch_pool_bytes,
    main,
    mem_free_pct,
    model_identity,
    model_path,
    model_stamp,
    oracle_gate,
    oracle_recompare,
    pool_geometry,
    pool_geometry_stamp,
    port_owner,
    preflight,
    print_capinv_table,
    print_feasibility_matrix,
    print_matrix,
    probe_pool_cap,
    read_launch_facts,
    reconcile_cap_probe,
    record_geometry_key,
    record_provenance,
    rss_gb,
    run_combo,
    running_servers,
    scan_gather_evidence,
    scan_nil,
    server_pid,
    server_procs,
    sh,
    source_stamp,
    source_text,
    step_partition,
    summary_table,
    union_fit_gate,
    wait_health,
)

# 原檔的 stdlib 綁定（`km.os` 這種讀法）—— 為了讓 dir() 與搬家前一致。
from knifeedge import (  # noqa: E402,F401
    argparse,
    hashlib,
    json,
    os,
    re,
    subprocess,
    sys,
    time,
    urllib,
)

# ---- 子模組可以直接取用 ------------------------------------------
# `km.<module>.<name> = x` 是路徑載入（spec_from_file_location）時**唯一**可行的
# 替換方式：那種載入不會把模組放進 sys.modules，所以沒有人拿得到「entry 那個物件」，
# 也就沒有 __setattr__ 可以攔。因此把子模組綁在 entry 上。
from knifeedge import (anchor, _source, constants, identity, host, oracle, feasibility, provenance, caps, matrix, capinv, cli)  # noqa: E402,F401

# ---- 屬性寫入轉發（僅對已註冊的 entry 生效）------------------------
# 這些離線自測會用屬性替換注入替身，例如 `km.pool_geometry = lambda kind: ...`。
# 單體時代那必然生效（只有一個命名空間）。拆檔之後 `feasibility_cell` 讀的是
# `feasibility.pool_geometry`，而 km 上的那個只是 shim 的獨立綁定。
# 正常 import 的 entry 就在 sys.modules 裡，可以換掉它的類別來攔 __setattr__；
# 路徑載入的 entry 不在 sys.modules 裡，Python 沒有給任何物件回指的辦法 ——
# 那種情況請改用 `km.<module>.<name> = x`（見上一段）。
import types as _types

_PKG = "knifeedge"
_OWNER = {
    "BINARY_DIR": "identity",
    "CAP_INVARIANCE_DEFAULT_CAPS": "constants",
    "DEFAULT_PORT": "anchor",
    "ENTRY_FILE": "anchor",
    "EXPERT_CACHE_NOHOOK": "constants",
    "EXPERT_CACHE_OFF": "constants",
    "EXPERT_CACHE_OFF_NOGATHER": "constants",
    "EXPERT_CACHE_ON": "constants",
    "FACT_PATTERNS": "constants",
    "GATHER_RE": "constants",
    "GGUF": "anchor",
    "GROUND_TRUTH_STATES": "constants",
    "KEY_FIELDS": "constants",
    "MIN_CAP": "constants",
    "MODELS": "constants",
    "MODEL_SAMPLE_BYTES": "constants",
    "NIL_RE": "constants",
    "NUMERIC_SOURCES": "constants",
    "POOL_GEOMETRY_SOURCES": "constants",
    "PROBE_PROMPT": "constants",
    "RESULT_DIR": "anchor",
    "ROOT": "anchor",
    "SERVER_MATCH": "anchor",
    "_FEAS_GEOM": "feasibility",
    "_FEAS_MOD": "feasibility",
    "_as_int": "identity",
    "_cap_guard": "oracle",
    "_cap_probe_provenance": "caps",
    "_cap_sidecar": "identity",
    "_capinv_guard": "capinv",
    "_env_int": "feasibility",
    "_feasibility": "feasibility",
    "_first_divergence": "capinv",
    "_git_out": "identity",
    "_model_digest_full": "identity",
    "_model_guard": "identity",
    "_oracle_guard": "oracle",
    "_oracle_meta": "identity",
    "_oracle_rows": "oracle",
    "_read_text": "host",
    "_realign_decode": "capinv",
    "_recorded_slots": "caps",
    "_run_oracle_compare": "oracle",
    "_short_val": "provenance",
    "_stamp_guard": "oracle",
    "_store_capinv": "capinv",
    "_truth_guard": "oracle",
    "_write_oracle_meta": "oracle",
    "apply_template_probe": "caps",
    "ask_probe": "host",
    "binary_stamp": "identity",
    "cap_for": "feasibility",
    "cap_invariance_dump": "capinv",
    "cap_invariance_gate": "capinv",
    "cap_probe_reusable": "caps",
    "combo_label": "matrix",
    "comparability_verdict": "caps",
    "effective_launch_env": "feasibility",
    "expert_cache_state": "oracle",
    "feasibility_cell": "feasibility",
    "feasibility_gate": "feasibility",
    "gate_model_identity": "identity",
    "geometry_conflict_error": "caps",
    "health": "host",
    "key_fingerprint": "provenance",
    "kill_servers": "host",
    "launch": "host",
    "launch_pool_bytes": "host",
    "main": "cli",
    "mem_free_pct": "host",
    "model_identity": "identity",
    "model_path": "identity",
    "model_stamp": "identity",
    "oracle_gate": "oracle",
    "oracle_recompare": "oracle",
    "pool_geometry": "feasibility",
    "pool_geometry_stamp": "identity",
    "port_owner": "host",
    "preflight": "caps",
    "print_capinv_table": "capinv",
    "print_feasibility_matrix": "feasibility",
    "print_matrix": "cli",
    "probe_pool_cap": "caps",
    "read_launch_facts": "host",
    "reconcile_cap_probe": "caps",
    "record_geometry_key": "provenance",
    "record_provenance": "provenance",
    "rss_gb": "host",
    "run_combo": "matrix",
    "running_servers": "host",
    "scan_gather_evidence": "caps",
    "scan_nil": "caps",
    "server_pid": "host",
    "server_procs": "host",
    "sh": "host",
    "source_stamp": "identity",
    "source_text": "_source",
    "step_partition": "capinv",
    "summary_table": "matrix",
    "union_fit_gate": "caps",
    "wait_health": "host",
}


class _Forward(_types.ModuleType):
    """把 `entry.N = x` 轉發到定義 N 的那個模組，其餘照常。"""

    def __setattr__(self, name, value):
        mod = _OWNER.get(name)
        if mod is not None:
            setattr(_sys.modules[f"{_PKG}.{mod}"], name, value)
        super().__setattr__(name, value)


_self = _sys.modules.get(__name__)
if _self is not None:
    _self.__class__ = _Forward

if __name__ == "__main__":
    sys.exit(main())

