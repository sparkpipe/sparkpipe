#!/usr/bin/env bash
# fleet_node_agent.sh — runs ON a spark: the whole release process.
# Pulls every managed runtime root from a reference tree (ceph or any
# reachable path), and on the convention file-changed -> TERM -> start,
# restarts exactly what changed. After any (re)start it reports the
# running binary versions to the fleet view on the hub (RTX5090 host).
#
# usage: fleet_node_agent.sh REFERENCE_BASE ROOTS_CSV [HUB]  (under setsid)
#   REFERENCE_BASE  directory containing <root-name>/ per deployment
#   ROOTS_CSV       e.g. glm53flash.fp8.tp16,dsv4flash.tp16
set -uo pipefail
REF_BASE="${1:?reference base dir}"
ROOTS="${2:?comma-separated runtime root names}"
HUB="${3:-sparkf}"
HOST=$(hostname)
RANK=$((16#${HOST#spark}))
PID_FILE="$HOME/.fleet_agent.pid"
VIEW="$HOME/current"          # local copy of the report
mkdir -p "$VIEW"

sha16() { sha256sum < "$1" 2>/dev/null | cut -c1-16; }

report() {
    {
        printf '{"host":"%s","time":"%s"' "$HOST" "$(date -Is)"
        printf ',"weightd":"%s"' "$(sha16 "$HOME/sparkdata/weightd/sparkpipe_weightd" 2>/dev/null)"
        local r first=1
        IFS=, read -ra RA <<< "$ROOTS"
        for r in "${RA[@]}"; do
            local rr="$HOME/sparkdata/$r"
            [ -d "$rr" ] || continue
            printf '%s"%s":{"residentd":"%s","driver":"%s","adapter":"%s","transport":"%s"}' \
                "$([ $first = 1 ] && echo ,roots:{ || echo ,)" "$r" \
                "$(sha16 "$rr/bin/sparkpipe_model_residentd")" \
                "$(sha16 "$rr/stages/stage_000/model_driver.so")" \
                "$(sha16 "$rr/lib/model_serving_adapter.so")" \
                "$(sha16 "$rr/lib/hidden_transport.so")"
            first=0
        done
        [ $first = 0 ] && printf '}'
        printf '}\n'
    } > "$VIEW/$HOST.json"
    scp -q -o BatchMode=yes -o ConnectTimeout=4 "$VIEW/$HOST.json" \
        "$HUB:current/" 2>/dev/null || true
}

restart_root() {
    local name="$1" rr="$HOME/sparkdata/$1" p
    for p in $(pgrep -f "bin/sparkpipe_model_(residentd|api)"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && kill -TERM "$p"
    done
    sleep 1
    cd "$rr" || return 1
    ln -sf "stage_$(printf %02d "$RANK").json" config/stage.json
    mv residentd.log residentd.log.prev 2>/dev/null
    LD_LIBRARY_PATH="$rr/lib" nohup ./bin/sparkpipe_model_residentd \
        --deployment model_resident.json --rank-index "$RANK" \
        > residentd.log 2>&1 < /dev/null &
    report
}

sync_root() {
    local name="$1" out="" p
    REF="$REF_BASE/$name"
    [ -d "$REF" ] || return 0
    ROOT="$HOME/sparkdata/$name"
    mkdir -p "$ROOT"
    for p in lib bin stages config model_resident.json model_package.json; do
        [ -e "$REF/$p" ] || continue
        local o
        o=$(rsync -a --out-format=%n "$REF/$p" "$ROOT/" 2>/dev/null) \
            && [ -n "$o" ] && out=1
    done
    [ -n "$out" ] && { echo "$(date +%T) $name changed; restarting"; restart_root "$name"; }
}

echo "$$" > "$PID_FILE"
echo "agent: rank=$RANK roots=$ROOTS ref=$REF_BASE hub=$HUB"
report
while true; do
    IFS=, read -ra RA <<< "$ROOTS"
    for r in "${RA[@]}"; do sync_root "$r"; done
    sleep 5
done
