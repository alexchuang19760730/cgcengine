#!/bin/bash
# wrappers/gate.sh — 判對錯類生產腳本的入口 (PLAN §3)
#
#   gate.sh --list / --self-test / --dry-run <script> [args...] / <script> [args...]
#
# 這一類的共同形狀：**回答合格／不合格、可比較／不可比較**。
# 它們的輸出是一個判斷，而判斷必須看**結束碼**而不是看文字：
# 這裡的 `w_run` 保留被呼叫腳本的 rc 並寫進 run manifest，所以
#   gate.sh m123_oracle_gate.py   && echo PASS || echo FAIL
# 是可靠的。★ 這也是為什麼這一類不可以被包成「一律吞掉 rc 的包裝」。
#
# 注意一個真實的坑：`m123_oracle_gate.py` 是 D5 的提交前閘門，而它裡面有
# `pkill -9 -f "llama-server"`（**沒有路徑前綴**）⇒ 它會殺掉任何命令列含那個字串的行程，
# 包括別條線正在跑的整臂。這一輪**沒有動它**（在 `scripts/`，屬另一條線），
# 但跑閘門之前要知道這件事。見 .workbuddy/memory/2026-09-17.md §Z6 的完整清單。

set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
w_class gate
w_main "$@"
