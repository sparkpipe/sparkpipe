#!/bin/bash
# Diagnostic wrapper: logs TERM delivery (with the signalling parent),
# then lets the real validator run to completion.
LOG=/mnt/model-warm/packbuild/dsv4pro/term-watch.log
on_term() {
    echo "VALIDATOR-TERM at $(date) mypid=$$ ppid=$PPID ppid-cmd=$(ps -o args= -p $PPID 2>/dev/null)" >> "$LOG"
}
trap on_term TERM
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
setsid bash "$DIR/validate_dsv4_resident_decode_stage_cuda.sh" "$@"
rc=$?
echo "validator-exit=$rc at $(date)" >> "$LOG"
exit $rc
