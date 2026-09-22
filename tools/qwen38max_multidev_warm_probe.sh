#!/usr/bin/env bash
# qwen38max_multidev_warm_probe.sh — 30-second health probe of this host's
# warm client against the qwen38_max checkpoint, taken BEFORE any solo
# emission run (the coordinator's managed-window rule: the fast-band
# number must be attributable to a healthy client; per-node client
# health varies wildly - spark7 measured ~190 MB/s for lane 4 while
# spark1 was D-dead).
#
# Queue cmd stays BARE: no env prefixes or other shell syntax.
# Prints a one-line PROBE receipt: bytes, seconds, MB/s, and the client
# state via /proc/self/mountstats is left to the caller's log.
set -euo pipefail

CHECKPOINT="${QMAX_PROBE_CHECKPOINT:-/mnt/model-warm/qwen3.8-max-nvfp4-radixark-bf16-spine}"
TARGET="$CHECKPOINT/model.safetensors.index.json"
BLOCK=$((16 * 1024 * 1024))   # 16 MiB blocks
COUNT=30                      # ~480 MiB, tens of seconds on a healthy client
[ -f "$TARGET" ] || { echo "qwen38max-warm-probe: missing $TARGET" >&2; exit 1; }

START="$(date +%s)"
BYTES=$((BLOCK * COUNT))
dd if="$TARGET" of=/dev/null bs="$BLOCK" count="$COUNT" 2>&1 | tail -1
END="$(date +%s)"
SECONDS_SPENT=$((END - START))
[ "$SECONDS_SPENT" -gt 0 ] || SECONDS_SPENT=1
MBPS=$((BYTES / SECONDS_SPENT / (1024 * 1024)))
echo "PROBE host=$(hostname) target=$TARGET bytes=$BYTES seconds=$SECONDS_SPENT mib_per_s=$MBPS"
