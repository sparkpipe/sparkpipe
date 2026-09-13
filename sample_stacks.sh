#!/usr/bin/env bash
pid=$(pgrep -f bin/sparkpipe_model_residentd | head -1)
for i in $(seq 1 40); do
    gdb -p "$pid" -batch -ex "thread apply all bt 4" 2>/dev/null
    sleep 0.15
done | awk '/^Thread/{t=$0} /^#0|^#1/{print t" | "$0}' | sort | uniq -c | sort -rn | head -25
