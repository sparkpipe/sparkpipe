#!/usr/bin/env bash
# snapshot_live.sh — record the live serving stack launch state on this rank.
set -u
pid=$(pgrep -f 'bin/lean_residentd' | head -1)
echo "pid=$pid"
if [ -n "$pid" ]; then
  echo "cwd=$(readlink /proc/$pid/cwd)"
  echo "cmdline=$(tr '\0' ' ' < /proc/$pid/cmdline)"
  tr '\0' '\n' < /proc/$pid/environ | grep -E 'SPARKPIPE|LD_LIBRARY_PATH'
fi
