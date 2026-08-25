#!/usr/bin/env bash
# lane_runner.sh — API-flake-resilient agent lane runner.
#
# Each lane is a prompt file (lanes/<name>.md) plus an append-only state log
# (lanes/<name>.state.md). The runner executes the lane prompt through the
# local agent CLI; on ANY failure (nonzero exit, timeout, empty output) it
# re-launches with backoff. Because every agent instance is told to FIRST
# read its state log and LAST append its progress, work survives arbitrary
# API failures: a killed attempt loses at most one step.
#
# Usage: tools/fleet/lane_runner.sh <lane-name> [base-dir]
#   LANE_MAX_ATTEMPTS   default 50
#   LANE_TIMEOUT_SECS   default 3600
set -uo pipefail
LANE="${1:?usage: lane_runner.sh LANE}"
BASE="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LNDIR="$BASE/tools/fleet/lanes"
PROMPT="$LNDIR/$LANE.md"
STATE="$LNDIR/$LANE.state.md"
MAX="${LANE_MAX_ATTEMPTS:-50}"
TMO="${LANE_TIMEOUT_SECS:-3600}"
[[ -f "$PROMPT" ]] || { echo "lane_runner: no prompt $PROMPT" >&2; exit 2; }
touch "$STATE"
LOCK="/tmp/sparkpipe-lane-$LANE.lock"
exec 9>"$LOCK"; flock -n 9 || { echo "lane $LANE already running" >&2; exit 0; }
for attempt in $(seq 1 "$MAX"); do
    echo "[lane $LANE] attempt $attempt $(date -u +%H:%M:%S)" >> "$STATE"
    PROMPT_FULL="$(cat <<EOF
You are resuming lane '$LANE' for the SparkPipe project.
FIRST: read $STATE — continue from the last completed step; never redo retained work.
RULES: append one line to $STATE after EVERY completed step (format: '[timestamp] STEP n: what was done + artifact path').
Work autonomously until the GOAL in the lane file is met or you hit a hard blocker; then record BLOCKER and exit 0.
$BASE is the canonical checkout unless the lane file says otherwise.
--- LANE DEFINITION ---
$(cat "$PROMPT")
EOF
)"
    if timeout "$TMO" opencode run "$PROMPT_FULL" >> "$LNDIR/$LANE.log" 2>&1; then
        echo "[lane $LANE] attempt $attempt exited clean" >> "$STATE"
        grep -q "GOAL MET\|BLOCKER" "$STATE" && break
    else
        echo "[lane $LANE] attempt $attempt FAILED rc=$? — relaunching" >> "$STATE"
        sleep $(( attempt < 5 ? 15 : 120 ))
    fi
done
echo "[lane $LANE] runner done $(date -u)" >> "$STATE"
