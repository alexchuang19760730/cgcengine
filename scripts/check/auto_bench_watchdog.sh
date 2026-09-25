#!/bin/bash
# 抓「重開機後自動拉起 llama-bench/server」的現行 + 止血
LOG=/tmp/auto_bench_watchdog.log
echo "=== watchdog start $(date '+%H:%M:%S') ===" > "$LOG"
while true; do
  for name in llama-bench llama-server; do
    pgrep -x "$name" | while read pid; do
      ppid=$(ps -o ppid= -p "$pid" | tr -d ' ')
      pcmd=$(ps -o comm= -p "$ppid" 2>/dev/null)
      cmd=$(ps -o command= -p "$pid" | cut -c1-160)
      echo "[$(date '+%H:%M:%S')] $name pid=$pid ppid=$ppid ($pcmd)" >> "$LOG"
      echo "    CMD: $cmd" >> "$LOG"
      # 止血：殺掉自動拉起的測量進程
      kill -9 "$pid" 2>/dev/null && echo "    -> killed $pid" >> "$LOG"
    done
  done
  sleep 4
done
