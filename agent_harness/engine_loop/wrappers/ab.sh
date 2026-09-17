#!/bin/bash
# wrappers/ab.sh — 對照類生產腳本的入口 (PLAN §3)
#
#   ab.sh --list / --self-test / --dry-run <script> [args...] / <script> [args...]
#
# 這一類的共同形狀：**拿兩份設定或兩份輸出比，回答「有沒有差」**。
# 它們最容易出的錯不是跑不起來，而是「兩臂其實不同時只差一個變數」：
#   * 兩臂跨了兩個 build（`scripts/check/ab_interleave.py` 的整個存在理由）
#   * 兩臂的命令列文字被判成同一件事（同一族 bug：行程身分 ≠ 命令列文字）
#   * 兩臂的答案摘要不穩（`answer_md5_set` 多於一個 ⇒ 那一列不可引用）
# 所以 `_common.sh` 對這一類的 run manifest 會記 build 指紋；而 `--dry-run` 的用處是
# 讓你在啟動伺服器之前**看到**那一對到底差在哪。

set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
w_class ab
w_main "$@"
