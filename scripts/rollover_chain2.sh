#!/usr/bin/env bash
H="$(cd "$(dirname "$0")" && pwd)"
LOCK="/usr/local/bin/flock -x -w 10800 /Users/mac/sparkpipe-coord/locks/tp16.lock"
for tag in "$@"; do
  case "$tag" in final) cmd="$H/rollover_finalize.sh" ;; spot) cmd="$H/rollover_spotcheck.sh" ;; *) cmd="$H/rollover_attrib.sh $tag" ;; esac
  echo "#### $tag lock wait $(date -u +%FT%TZ)"
  $LOCK bash $cmd
  rc=$?
  echo "#### $tag rc=$rc $(date -u +%FT%TZ)"
  [ $rc = 9 ] && { echo "NODE DOWN: chain stopped"; exit 9; }
done
