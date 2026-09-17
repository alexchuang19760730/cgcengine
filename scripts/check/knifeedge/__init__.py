#!/usr/bin/env python3
"""knifeedge — scripts/check/knifeedge_matrix.py 的內容，按能力拆成 10 個模組。

公開面（`__all__`）與原檔的頂層名字**逐一相同**，所以任何既有的
`import knifeedge_matrix` / `from ... import X` 都不必改一行：
舊路徑 knifeedge_matrix.py 是薄 shim，re-export 的就是這裡。

下面那排 stdlib import 不是 API，是為了讓 `dir()` 與原檔一致 —— 原檔 import 過
`os`/`json`/… 之後，那些名字就是它的屬性；搬家不該讓 `km.os` 消失。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .anchor import (DEFAULT_PORT, ENTRY_FILE, GGUF, RESULT_DIR, ROOT, SERVER_MATCH)
from ._source import (source_text)
from .constants import (CAP_INVARIANCE_DEFAULT_CAPS, EXPERT_CACHE_NOHOOK, EXPERT_CACHE_OFF, EXPERT_CACHE_OFF_NOGATHER, EXPERT_CACHE_ON, FACT_PATTERNS, GATHER_RE, GROUND_TRUTH_STATES, KEY_FIELDS, MIN_CAP, MODELS, MODEL_SAMPLE_BYTES, NIL_RE, NUMERIC_SOURCES, POOL_GEOMETRY_SOURCES, PROBE_PROMPT)
from .identity import (BINARY_DIR, _as_int, _cap_sidecar, _git_out, _model_digest_full, _model_guard, _oracle_meta, binary_stamp, gate_model_identity, model_identity, model_path, model_stamp, pool_geometry_stamp, source_stamp)
from .host import (_read_text, ask_probe, health, kill_servers, launch, launch_pool_bytes, mem_free_pct, port_owner, read_launch_facts, rss_gb, running_servers, server_pid, server_procs, sh, wait_health)
from .oracle import (_cap_guard, _oracle_guard, _oracle_rows, _run_oracle_compare, _stamp_guard, _truth_guard, _write_oracle_meta, expert_cache_state, oracle_gate, oracle_recompare)
from .feasibility import (_FEAS_GEOM, _FEAS_MOD, _env_int, _feasibility, cap_for, effective_launch_env, feasibility_cell, feasibility_gate, pool_geometry, print_feasibility_matrix)
from .provenance import (_short_val, key_fingerprint, record_geometry_key, record_provenance)
from .caps import (_cap_probe_provenance, _recorded_slots, apply_template_probe, cap_probe_reusable, comparability_verdict, geometry_conflict_error, preflight, probe_pool_cap, reconcile_cap_probe, scan_gather_evidence, scan_nil, union_fit_gate)
from .matrix import (combo_label, run_combo, summary_table)
from .capinv import (_capinv_guard, _first_divergence, _realign_decode, _store_capinv, cap_invariance_dump, cap_invariance_gate, print_capinv_table, step_partition)
from .cli import (main, print_matrix)

__all__ = [
    "BINARY_DIR",
    "CAP_INVARIANCE_DEFAULT_CAPS",
    "DEFAULT_PORT",
    "ENTRY_FILE",
    "EXPERT_CACHE_NOHOOK",
    "EXPERT_CACHE_OFF",
    "EXPERT_CACHE_OFF_NOGATHER",
    "EXPERT_CACHE_ON",
    "FACT_PATTERNS",
    "GATHER_RE",
    "GGUF",
    "GROUND_TRUTH_STATES",
    "KEY_FIELDS",
    "MIN_CAP",
    "MODELS",
    "MODEL_SAMPLE_BYTES",
    "NIL_RE",
    "NUMERIC_SOURCES",
    "POOL_GEOMETRY_SOURCES",
    "PROBE_PROMPT",
    "RESULT_DIR",
    "ROOT",
    "SERVER_MATCH",
    "_FEAS_GEOM",
    "_FEAS_MOD",
    "_as_int",
    "_cap_guard",
    "_cap_probe_provenance",
    "_cap_sidecar",
    "_capinv_guard",
    "_env_int",
    "_feasibility",
    "_first_divergence",
    "_git_out",
    "_model_digest_full",
    "_model_guard",
    "_oracle_guard",
    "_oracle_meta",
    "_oracle_rows",
    "_read_text",
    "_realign_decode",
    "_recorded_slots",
    "_run_oracle_compare",
    "_short_val",
    "_stamp_guard",
    "_store_capinv",
    "_truth_guard",
    "_write_oracle_meta",
    "apply_template_probe",
    "ask_probe",
    "binary_stamp",
    "cap_for",
    "cap_invariance_dump",
    "cap_invariance_gate",
    "cap_probe_reusable",
    "combo_label",
    "comparability_verdict",
    "effective_launch_env",
    "expert_cache_state",
    "feasibility_cell",
    "feasibility_gate",
    "gate_model_identity",
    "geometry_conflict_error",
    "health",
    "key_fingerprint",
    "kill_servers",
    "launch",
    "launch_pool_bytes",
    "main",
    "mem_free_pct",
    "model_identity",
    "model_path",
    "model_stamp",
    "oracle_gate",
    "oracle_recompare",
    "pool_geometry",
    "pool_geometry_stamp",
    "port_owner",
    "preflight",
    "print_capinv_table",
    "print_feasibility_matrix",
    "print_matrix",
    "probe_pool_cap",
    "read_launch_facts",
    "reconcile_cap_probe",
    "record_geometry_key",
    "record_provenance",
    "rss_gb",
    "run_combo",
    "running_servers",
    "scan_gather_evidence",
    "scan_nil",
    "server_pid",
    "server_procs",
    "sh",
    "source_stamp",
    "source_text",
    "step_partition",
    "summary_table",
    "union_fit_gate",
    "wait_health",
]
