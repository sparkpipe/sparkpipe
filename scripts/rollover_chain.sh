#!/usr/bin/env bash
H="$(cd "$(dirname "$0")" && pwd)"
LOCK="/usr/local/bin/flock -x -w 10800 /Users/mac/sparkpipe-coord/locks/tp16.lock"
for tag in "$@"; do
  case "$tag" in k3) script="$H/rollover_k3.sh" ;; full|flash) script="$H/rollover_glm.sh" ;; spot) script="$H/rollover_spotcheck.sh" ;; *) script="$H/rollover_stage.sh" ;; esac
  echo "#### $tag lock wait $(date -u +%FT%TZ)"
  $LOCK bash "$script" "$tag"
  rc=$?
  echo "#### $tag rc=$rc $(date -u +%FT%TZ)"
  [ $rc = 9 ] && { echo "NODE DOWN: chain stopped"; exit 9; }
done
