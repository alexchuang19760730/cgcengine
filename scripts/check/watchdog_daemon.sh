#!/bin/bash
# 看門狗常駐守護：每 600 s 巡檢一次（由有 Documents 權限的環境啟動，避開 launchd 的 TCC 限制）。
# 停止：kill 本腳本進程；日誌見 Backup/lane_watchdog/daemon.log
REPO=/Users/alexchuang/Documents/flashkv-devserver
LOG="$REPO/Backup/lane_watchdog/daemon.log"
mkdir -p "$(dirname "$LOG")"

while true; do
  echo "=== $(date '+%F %T') tick ===" >> "$LOG"
  /usr/bin/python3 "$REPO/scripts/check/lane_watchdog.py" --kill --quarantine >> "$LOG" 2>&1
  echo "rc=$?" >> "$LOG"
  sleep 600
done
