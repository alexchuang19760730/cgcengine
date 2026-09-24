#!/bin/bash
# Memory / swap watch - record current state to a log file
# Usage: ./scripts/check/memory_watch.sh "label"

LABEL="${1:-unnamed}"
LOG="/Users/alexchuang/Documents/flashkv-devserver/.workbuddy/memory/swap_log.tsv"

# Create log header if first run
if [ ! -f "$LOG" ]; then
  echo -e "timestamp\tlabel\tswap_used_mb\tpages_free\tpages_inactive\tpages_wired\tllama_procs" > "$LOG"
fi

SWAP_USED=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/')
PAGES_FREE=$(vm_stat | awk '/Pages free/ {gsub(/\./, "", $3); print $3}')
PAGES_INACTIVE=$(vm_stat | awk '/Pages inactive/ {gsub(/\./, "", $3); print $3}')
PAGES_WIRED=$(vm_stat | awk '/Pages wired down/ {gsub(/\./, "", $4); print $4}')
LLAMA_PROCS=$(ps aux | grep -i llama | grep -v grep | wc -l | tr -d ' ')

echo -e "$(date +%s)\t$LABEL\t$SWAP_USED\t$PAGES_FREE\t$PAGES_INACTIVE\t$PAGES_WIRED\t$LLAMA_PROCS" >> "$LOG"

echo "Recorded: $LABEL"
echo "  Swap used: ${SWAP_USED} MB"
echo "  Pages free: $PAGES_FREE ($(echo "$PAGES_FREE * 16 / 1024" | bc) MB)"
echo "  Pages inactive: $PAGES_INACTIVE ($(echo "$PAGES_INACTIVE * 16 / 1024" | bc) MB)"
echo "  Pages wired: $PAGES_WIRED ($(echo "$PAGES_WIRED * 16 / 1024" | bc) MB)"
echo "  Llama processes: $LLAMA_PROCS"
echo "  Log: $LOG"
