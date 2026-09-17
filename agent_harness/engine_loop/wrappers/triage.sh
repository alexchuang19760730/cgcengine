#!/bin/bash
# wrappers/triage.sh — 鑑識類生產腳本的入口 (PLAN §3)
#
#   triage.sh --list / --self-test / --dry-run <script> [args...] / <script> [args...]
#
# 這一類的共同形狀：**對一個已經出現的警報或讀數追它的來源**。
# 它們和 `gate` 的差別是：gate 回答「合不合格」，triage 回答「這個數字為什麼長這樣」。
#
# ★ 這一類有 9 支，而 `MANIFEST.jsonl` 把其中大部分都標成 `probe`（整個目錄 44 支裡有 34 支是
#   probe）。這正是需要第二個座標的理由：34 個 `probe` 之間無法回答「我現在該跑哪一支」。
#   這一支替那一類提供的是**鑑識的入口**，而不是「其他」。
#
#   判準上的一個提醒：這一類的產物幾乎都是**解釋**而不是**讀數**，所以它們最容易犯的錯是
#   「拿一個沒有 build 指紋的數字去解釋另一個沒有 build 指紋的數字」。`_common.sh` 因此仍然
#   要求 build.json —— 不是為了記帳，是因為解釋的效力不能超過它引用的讀數。

set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
w_class triage
w_main "$@"
