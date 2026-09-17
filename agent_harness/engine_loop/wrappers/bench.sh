#!/bin/bash
# wrappers/bench.sh — 基準讀數類生產腳本的入口 (PLAN §3)
#
#   bench.sh --list / --self-test / --dry-run <script> [args...] / <script> [args...]
#
# 這一類的共同形狀：**產生一個可以互相比較的吞吐量或品質讀數**。
# 它們的產物是「數字」，而這個 repo 對數字的規則是：**沒有 build 指紋的數字不可引用**
# （`traces/emit_episodes.py` 對 `build == null` 的列直接標 `usable_as_evidence: false`）。
# 所以 `_common.sh` 的 `w_fingerprint()` 對這一類是**硬前置**：沒有 `build.json` 就拒跑
# （`ENGINE_ALLOW_NO_FINGERPRINT=1` 可以明知故犯，但那就會留下 `(no-fingerprint)` 的痕跡）。

set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
w_class bench
w_main "$@"
