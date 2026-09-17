#!/bin/bash
# wrappers/sweep.sh — 參數掃描類生產腳本的入口 (PLAN §3)
#
#   sweep.sh --list                       # 這一類有哪些腳本（含 MANIFEST 的 role 與它們自己的說明）
#   sweep.sh --self-test                  # 分類表與磁碟的一致性
#   sweep.sh --dry-run <script> [args...] # 只印出將執行的命令
#   sweep.sh <script> [args...]           # 執行並落一份 run manifest
#
# 這一類的共同形狀：**對一個旋鈕做掃描，逐格重啟伺服器，每格記一列**。
# 所以它們的關鍵前置不是「腳本在不在」，而是「機器現在能不能被獨占」（見 runners/preflight.sh）。
# 這支 wrapper 不代替你判斷那個 —— 它只保證你跑的版本、參數與二進位檔身分被記下來。

set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
w_class sweep
w_main "$@"
