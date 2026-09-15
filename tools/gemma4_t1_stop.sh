#!/bin/bash
# Stop this node's gemma4-26b T1 lane processes (scoped: only cwd under this
# lane root; never touches other lanes, production roots, or daemons).
set -u
ROOT="$HOME/sparkdata/gemma4_26b.tp4pp4.t1"
for pattern in sparkpipe_model_api sparkpipe_model_residentd; do
    for pid in $(pgrep -f "$pattern" 2>/dev/null); do
        cwd=$(readlink /proc/$pid/cwd 2>/dev/null || echo unknown)
        case "$cwd" in
            "$ROOT"|"$ROOT"/*)
                kill -TERM "$pid" 2>/dev/null
                echo "stopped pid=$pid cwd=$cwd"
                ;;
            *)
                echo "skipped pid=$pid cwd=$cwd (not mine)"
                ;;
        esac
    done
done
