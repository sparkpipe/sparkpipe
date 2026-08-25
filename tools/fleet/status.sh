#!/usr/bin/env bash
# status.sh — one-shot fleet+lane status. Run any time; reads only.
echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
shopt -s nullglob
for s in /Users/mac/sparkpipe/tools/fleet/lanes/*.state.md; do
  n=$(basename "$s" .state.md)
  pid=$(pgrep -f "lane_runner.sh $n" | head -1)
  echo "-- lane $n [$([ -n "$pid" ] && echo RUNNING pid=$pid || echo stopped)]"
  tail -2 "$s" 2>/dev/null
done
echo "-- bands"
for h in spark4 spark5 spark6 spark7 spark8; do
  printf '%s: %s\n' "$h" "$(ssh -o BatchMode=yes -o ConnectTimeout=4 $h 'pgrep -c -f "^bin/sparkpipe_model_residentd" 2>/dev/null' 2>/dev/null || echo down)"
done
echo "-- glm rank04 pack ship"
ssh -o BatchMode=yes -o ConnectTimeout=4 spark1 'ls -l ~/srcdata/glm52_tp8_packs/ 2>/dev/null | rg rank04 || echo not-started' 2>/dev/null
echo "-- perf HWM (from docs/PERF_DASHBOARD.md)"
rg "^| \*\*4[0-9]\.\d\d\*\*|8\.03|40\." ~/sparkpipe/docs/PERF_DASHBOARD.md 2>/dev/null | head -0
grep -m1 "DSV4 Flash" ~/sparkpipe/docs/PERF_DASHBOARD.md | cut -c1-160
grep -m1 "Qwen 3.8 27B" ~/sparkpipe/docs/PERF_DASHBOARD.md | cut -c1-160
