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
LAST_REPORT=""
LAST_START=0
mkdir -p "$VIEW"

sha16() { [ -f "$1" ] && sha256sum < "$1" | cut -c1-16 || echo none; }

root_state() {
    local rr="$HOME/sparkdata/$1"
    if pgrep -f "bin/sparkpipe_model_residentd" >/dev/null && \
       [ "$(readlink /proc/$(pgrep -f 'bin/sparkpipe_model_residentd' | head -1)/cwd 2>/dev/null)" = "$rr" ]; then
        local last
        last=$(tail -1 "$rr/residentd.log" 2>/dev/null | cut -c1-90)
        case "$last" in
            *"model_residentd ready"*) echo "ready" ;;
            *) echo "starting: $last" ;;
        esac
    else
        echo "down"
    fi
}

report() {
    {
        printf '{"host":"%s","time":"%s"' "$HOST" "$(date -Is)"
        printf ',"weightd":"%s"' "$(sha16 "$HOME/sparkdata/weightd/sparkpipe_weightd" 2>/dev/null)"
        local r first=1 states=""
        IFS=, read -ra RA <<< "$ROOTS"
        for r in "${RA[@]}"; do
            local rr="$HOME/sparkdata/$r"
            [ -d "$rr" ] || continue
            local st; st=$(root_state "$r")
            states="$states$r=$st;"
            printf '%s"%s":{"state":"%s","residentd":"%s","driver":"%s"}' \
                "$([ $first = 1 ] && echo ,roots:{ || echo ,)" "$r" \
                "${st//\"/\\\"}" \
                "$(sha16 "$rr/bin/sparkpipe_model_residentd")" \
                "$(sha16 "$rr/stages/stage_000/model_driver.so")"
            first=0
        done
        [ $first = 0 ] && printf '}'
        printf '}\n'
    } > "$VIEW/$HOST.json"
    LAST_REPORT="$states"
    scp -q -o BatchMode=yes -o ConnectTimeout=4 "$VIEW/$HOST.json" \
        "$HUB:current/" 2>/dev/null || true
}

report_if_changed() {
    local r states=""
    IFS=, read -ra RA <<< "$ROOTS"
    for r in "${RA[@]}"; do
        [ -d "$HOME/sparkdata/$r" ] || continue
        states="$states$r=$(root_state "$r");"
    done
    [ "$states" != "$LAST_REPORT" ] && report
}

unload_root() {
    local rr="$HOME/sparkdata/$1" p t gone
    for p in $(pgrep -f "bin/sparkpipe_model_(residentd|api)"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && kill -TERM "$p"
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

start_root() {
    local name="$1" rr="$HOME/sparkdata/$1" p
    for p in $(pgrep -f "bin/sparkpipe_model_(residentd|api)"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && return 0
    done
    cd "$rr" || return 1
    ln -sf "stage_$(printf %02d "$RANK").json" config/stage.json
    mv residentd.log residentd.log.prev 2>/dev/null
    LD_LIBRARY_PATH="$rr/lib" nohup ./bin/sparkpipe_model_residentd \
        --deployment model_resident.json --rank-index "$RANK" \
        > residentd.log 2>&1 < /dev/null &
    report
}

restart_root() {
    local name="$1"
    cd "$HOME/sparkdata/$1" || return 1
    unload_root "$name" || return 1
    start_root "$name"
}

FLEET_SIZE=16
HUBSSH="ssh -o BatchMode=yes -o ConnectTimeout=5"

hub_has() {
    $HUBSSH "${REF_BASE%%:*}" "test -f '${REF_BASE#*:}/$1' && echo yes" 2>/dev/null || true
}

sync_root() {
    local name="$1" p upd
    local refhost="${REF_BASE%%:*}"
    local refdir="${REF_BASE#*:}/$name"
    local root="$HOME/sparkdata/$name"
    mkdir -p "$root"
    local exists
    exists=$(hub_has "$name/UPDATE")
    if [ "$exists" != yes ] && [ -x "$root/bin/sparkpipe_model_residentd" ]; then
        return 0
    fi
    for p in lib bin stages config model_resident.json; do
        rsync -a -e "$HUBSSH" --checksum --omit-dir-times --exclude=stage.json "$REF_BASE/$name/$p" "$root/" 2>>"$HOME/fleet_agent_rsync.log" || true
    done
    [ "$exists" = yes ] || return 0
    upd=$($HUBSSH "$refhost" "cat '$refdir/UPDATE'") || upd=""
    if ! printf '%s\n' "$upd" | grep -qx "down:$HOST"; then
        unload_root "$name" || return 0
        $HUBSSH "$refhost" "echo down:$HOST >> '$refdir/UPDATE'" 2>/dev/null || return 0
        upd=$(printf '%s\ndown:%s\n' "$upd" "$HOST")
    fi
    local gate_wait=0
    while [ "$(printf '%s\n' "$upd" | grep -c '^down:')" -lt "$FLEET_SIZE" ] &&
          [ "$gate_wait" -lt 120 ]; do
        sleep 1
        gate_wait=$((gate_wait + 1))
        upd=$($HUBSSH "$refhost" "cat '$refdir/UPDATE' 2>/dev/null") || upd=""
    done
    [ "$(printf '%s\n' "$upd" | grep -c '^down:')" -ge "$FLEET_SIZE" ] || return 0
    if ! printf '%s\n' "$upd" | grep -qx "up:$HOST"; then
        start_root "$name" || return 0
        $HUBSSH "$refhost" "echo up:$HOST >> '$refdir/UPDATE'" 2>/dev/null || return 0
        upd=$(printf '%s\nup:%s\n' "$upd" "$HOST")
    fi
    [ "$(printf '%s\n' "$upd" | grep -c '^up:')" -ge "$FLEET_SIZE" ] || return 0
    local c
    c=$($HUBSSH "$refhost" "ls '$refdir' 2>/dev/null | sed -n 's/^UPDATE\.\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1")
    $HUBSSH "$refhost" "mv '$refdir/UPDATE' '$refdir/UPDATE.$(( ${c:-0} + 1 ))'" 2>/dev/null || true
}

sync_weightd() {
    local home="$HOME/sparkdata/weightd"
    mkdir -p "$home"
    if [ -x "$home/sparkpipe_weightd" ] && [ "$(hub_has weightd/UPDATE)" != yes ]; then
        return 0
    fi
    rsync -a -e "$HUBSSH" --checksum "$REF_BASE/weightd/" "$home/" 2>>"$HOME/fleet_agent_rsync.log" || true
    $HUBSSH "${REF_BASE%%:*}" "mv '${REF_BASE#*:}/weightd/UPDATE' '${REF_BASE#*:}/weightd/UPDATE.$(date +%s)'" 2>/dev/null || true
}

ensure_weightd() {
    pgrep -f "sparkpipe_weightd" >/dev/null && return 0
    local home="$HOME/sparkdata/weightd"
    [ -x "$home/sparkpipe_weightd" ] || return 0
    echo "$(date +%T) weightd: starting"
    setsid nohup "$home/sparkpipe_weightd" --socket /tmp/spark_weightd.sock \
        > "$HOME/weightd.log" 2>&1 < /dev/null &
}

echo "$$" > "$PID_FILE"
echo "agent: rank=$RANK roots=$ROOTS ref=$REF_BASE hub=$HUB"
report
sync_weightd
ensure_root() {
    local name="$1"
    local st; st=$(root_state "$name")
    [ "$st" = "down" ] || return 0
    local now=$(date +%s)
    [ $((now - LAST_START)) -lt 15 ] && return 0
    LAST_START=$now
    echo "$(date +%T) $name: down; starting"
    restart_root "$name"
}

while true; do
    sync_weightd
    ensure_weightd
    IFS=, read -ra RA <<< "$ROOTS"
    for r in "${RA[@]}"; do sync_root "$r"; done
    for r in "${RA[@]}"; do ensure_root "$r"; done
    report_if_changed
    sleep 5
done
