#!/bin/bash
# wrappers/env.sh — 環境與服務健康檢查類的入口（**PLAN §3 沒有這一類**）
#
#   env.sh --list / --self-test / --dry-run <script> [args...] / <script> [args...]
#
# ★ 這一支是對計畫的**偏差**，寫在這裡而不是藏起來：
#   PLAN §3 只列了 5 個 wrapper（sweep / ab / bench / gate / triage），而 `scripts/check/`
#   實測有 3 支是環境／服務健康檢查（`check_env.sh`、`check_torch.sh`、`check_server.sh`），
#   加上 `check_server_profiles.py`（後者是 `check_server.sh` 轉呼叫的實作）。
#   它們不是掃描、不是 A/B、不是 benchmark、不是判對錯的閘門、也不是鑑識。
#   把 3 支硬塞進那 5 類的任一個，那一類就會變成雜物桶 —— 而雜物桶的失效方式是
#   「分類看起來很整齊，但沒有人能從分類回答問題」。
#
#   所以新增第 6 類。`wrappers/classify.py` 的 docstring 記了同一個偏差，
#   而 `classes.tsv` 是它的產物（那才是分類的權威；這一支只是那一類的入口）。
#
#   這一類的特性，也說明它為什麼不該和另外 5 類混在一起：
#   它們**不產生量測讀數**，所以它們不需要獨占機器，也不需要 build 指針。
#   它們的典型用途是「在我啟動任何東西之前，先確認環境沒有壞」。

set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
w_class env
w_main "$@"
