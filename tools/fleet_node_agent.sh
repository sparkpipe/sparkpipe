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

sha16() { [ -f "$1" ] && sha256sum < "$1" | cut -c1-16 || echo none; }

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

unload_root() {
    local rr="$HOME/sparkdata/$1" p t gone
    for p in $(pgrep -f "bin/sparkpipe_model_(residentd|api)"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && kill -TERM "$p"
    done
    for p in $(pgrep -f "sparkpipe_weightd"); do
        c=$(readlink /proc/$p/cwd 2>/dev/null)
        [ "$c" = "$rr" ] || case "$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null)" in
            *"$rr"*) kill -TERM "$p" ;; esac
    done
    for t in $(seq 1 30); do
        gone=1
        for p in $(pgrep -f "bin/sparkpipe_model_residentd"); do
            [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && gone=0
        done
        [ "$gone" = 1 ] && break
        sleep 1
    done
    [ "$gone" = 1 ] || { echo "$(date +%T) $1: prior residentd not exited; NOT starting new" >&2; return 1; }
    local pack_gb=$(du -sBG "$rr/packs" 2>/dev/null | cut -dG -f1)
    pack_gb=${pack_gb:-0}
    for t in $(seq 1 30); do
        local avail=$(awk "/MemAvailable/ {print int(\$2/1048576)}" /proc/meminfo)
        [ "$avail" -ge $((pack_gb + 8)) ] && return 0
        sleep 2
    done
    echo "$(date +%T) $1: MemAvailable never reached $((pack_gb + 8))GB; NOT starting new" >&2
    return 1
}

restart_root() {
    local name="$1" rr="$HOME/sparkdata/$1"
    cd "$rr" || return 1
    unload_root "$name" || return 1
    ln -sf "stage_$(printf %02d "$RANK").json" config/stage.json
    mv residentd.log residentd.log.prev 2>/dev/null
    LD_LIBRARY_PATH="$rr/lib" nohup ./bin/sparkpipe_model_residentd \
        --deployment model_resident.json --rank-index "$RANK" \
        > residentd.log 2>&1 < /dev/null &
    report
}

sync_root() {
    local name="$1" out="" p
    local ref="$REF_BASE/$name"
    local root="$HOME/sparkdata/$name"
    mkdir -p "$root"
    for p in lib bin stages config model_resident.json model_package.json; do
        local o
        o=$(rsync -a --out-format=%n "$ref/$p" "$root/" 2>>"$HOME/fleet_agent_rsync.log") \
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
