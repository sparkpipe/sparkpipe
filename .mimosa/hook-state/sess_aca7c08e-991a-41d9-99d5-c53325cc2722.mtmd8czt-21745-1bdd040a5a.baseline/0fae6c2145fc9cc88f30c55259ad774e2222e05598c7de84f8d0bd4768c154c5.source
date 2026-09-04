#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 OVERLAY_ROOT PHASE10_SOURCE_ROOT" >&2
    exit 2
fi

exec python3 "$1/tools/runtime/apply_runtime_completion_overlay.py" "$1" "$2"
