#!/usr/bin/env bash
set -uo pipefail
ROOTS="${1:?comma-separated runtime root names}"
HUB="${2:-sparkf}"
HOST=$(hostname)
RANK=$((16#${HOST#spark}))
PID_FILE="$HOME/.fleet_agent.pid"
VIEW="$HOME/current"
LAST_REPORT=""
LAST_PIDS=""
LAST_API_START=0

sha16() {
    local s=""
    [ -f "$1" ] && s=$(sha256sum < "$1" 2>/dev/null | cut -c1-16)
    [ -n "$s" ] || s=none
    echo "$s"
}

START_SHA=$(sha16 "$0")
AGENT_BLOCKED=""
mkdir -p "$VIEW"

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
        printf ',"weightd":"%s","agent":"%s"' \
            "$(sha16 "$HOME/sparkdata/weightd/sparkpipe_weightd" 2>/dev/null)" \
            "$(sha16 "$0" 2>/dev/null)"
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
    local r states="" pids=""
    IFS=, read -ra RA <<< "$ROOTS"
    for r in "${RA[@]}"; do
        [ -d "$HOME/sparkdata/$r" ] || continue
        states="$states$r=$(root_state "$r");"
        p="$HOME/sparkdata/$r"
        local pid
        pid=$(for l in $(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model_residentd | sed "s|.*/proc/\([0-9]*\)/exe.*|\1|"); do
            [ "$(readlink /proc/$l/cwd 2>/dev/null)" = "$p" ] && echo "$l"
        done | head -1)
        pids="$pids$r=${pid:-0};"
    done
    { [ "$states" != "$LAST_REPORT" ] || [ "$pids" != "$LAST_PIDS" ]; } && {
        LAST_PIDS="$pids"
        report
    }
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
    [ -f "$rr/env.local" ] && set -a && . "$rr/env.local" && set +a
    ln -sf "stage_$(printf %02d "$RANK").json" config/stage.json
    mv residentd.log residentd.log.prev 2>/dev/null
    local cap="${FLEET_RESIDENTD_MEMORY_MAX:-8589934592}"
    if [ "$cap" != 0 ] && command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --quiet --scope -p MemoryMax="$cap" \
            env LD_LIBRARY_PATH="$rr/lib" ./bin/sparkpipe_model_residentd \
            --deployment model_resident.json --rank-index "$RANK" \
            > residentd.log 2>&1 < /dev/null &
    else
        LD_LIBRARY_PATH="$rr/lib" nohup ./bin/sparkpipe_model_residentd \
            --deployment model_resident.json --rank-index "$RANK" \
            > residentd.log 2>&1 < /dev/null &
    fi
    report
}

ensure_api() {
    [ "$RANK" = 0 ] || return 0
    local rr="$HOME/sparkdata/glm53flash.fp8.tp16"
    [ -x "$rr/bin/sparkpipe_model_api" ] || return 0
    local ready_count now
    ready_count=$(ssh -o BatchMode=yes -o ConnectTimeout=4 "$HUB" \
        "grep -l '\"state\":\"ready' current/*.json 2>/dev/null | wc -l" 2>/dev/null)
    [ "${ready_count:-0}" -ge 16 ] || return 0
    local p
    for p in $(pgrep -f "bin/sparkpipe_model_api"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && return 0
    done
    now=$(date +%s)
    [ $((now - LAST_API_START)) -lt 15 ] && return 0
    LAST_API_START=$now
    echo "$(date +%T) api: starting"
    cd "$rr" || return 1
    LD_LIBRARY_PATH="$rr/lib" setsid nohup ./bin/sparkpipe_model_api \
        --deployment model_resident.json --runtime-root "$rr" --port "${G5_API_PORT:-8433}" \
        > api.log 2>&1 < /dev/null &
}

restart_root() {
    local name="$1"
    cd "$HOME/sparkdata/$1" || return 1
    unload_root "$name" || return 1
    start_root "$name"
}

FLEET_SIZE=16
HUBSSH="ssh -o BatchMode=yes -o ConnectTimeout=5 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-agent-%r@%h:%p -o ControlPersist=600"
RELEASE_HTTP="${FLEET_HTTP_RELEASE:-http://10.10.100.25:8802}"

apply_manifest() {
    local name="$1" root="$2" manifest_cur manifest_applied
    manifest_cur="/tmp/fleet_manifest_$name.txt"
    manifest_applied="$root/.applied_manifest"
    if ! curl -sf --max-time 8 "$RELEASE_HTTP/$name/MANIFEST" -o "$manifest_cur"; then
        [ -f "$manifest_applied" ] || echo "$(date +%T) $name: manifest unreachable" >&2
        return 1
    fi
    cmp -s "$manifest_cur" "$manifest_applied" 2>/dev/null && return 1
    echo "$(date +%T) $name: manifest changed; syncing"
    local fetch_errors=0 line
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        local want="${line%% *}"
        local rel="${line#*  }"
        local tmp="$root/.fetch.tmp"
        if ! curl -sf --max-time 300 "$RELEASE_HTTP/$name/$rel" -o "$tmp"; then
            echo "$(date +%T) $name: fetch failed: $rel" >&2
            fetch_errors=$((fetch_errors+1))
            continue
        fi
        local got
        got=$(sha256sum "$tmp" | cut -d' ' -f1)
        if [ "$got" != "$want" ]; then
            echo "$(date +%T) $name: checksum failed: $rel" >&2
            fetch_errors=$((fetch_errors+1))
            continue
        fi
        mkdir -p "$(dirname "$root/$rel")"
        mv "$tmp" "$root/$rel"
        case "$rel" in
            bin/*) chmod 755 "$root/$rel" ;;
        esac
    done < "$manifest_cur"
    [ "$fetch_errors" != 0 ] && { echo "$(date +%T) $name: $fetch_errors fetch errors; retrying next cycle" >&2; return 1; }
    cp "$manifest_cur" "$manifest_applied"
    return 0
}

sync_root() {
    local name="$1" upd
    local root="$HOME/sparkdata/$name"
    mkdir -p "$root"
    apply_manifest "$name" "$root" || return 0
    local refdir="release/$name"
    upd=$($HUBSSH "$HUB" "cat '$refdir/UPDATE' 2>/dev/null") || upd=""
    [ -n "$upd" ] || return 0
    if ! printf '%s\n' "$upd" | grep -qx "down:$HOST"; then
        unload_root "$name" || return 0
        $HUBSSH "$HUB" "echo down:$HOST >> '$refdir/UPDATE'" 2>/dev/null || return 0
        upd=$(printf '%s\ndown:%s\n' "$upd" "$HOST")
    fi
    local gate_wait=0
    while [ "$(printf '%s\n' "$upd" | grep -c '^down:')" -lt "$FLEET_SIZE" ] &&
          [ "$gate_wait" -lt 120 ]; do
        sleep 1
        gate_wait=$((gate_wait + 1))
        upd=$($HUBSSH "$HUB" "cat '$refdir/UPDATE' 2>/dev/null") || upd=""
    done
    [ "$(printf '%s\n' "$upd" | grep -c '^down:')" -ge "$FLEET_SIZE" ] || return 0
    if ! printf '%s\n' "$upd" | grep -qx "up:$HOST"; then
        start_root "$name" || return 0
        $HUBSSH "$HUB" "echo up:$HOST >> '$refdir/UPDATE'" 2>/dev/null || return 0
        upd=$(printf '%s\nup:%s\n' "$upd" "$HOST")
    fi
    [ "$(printf '%s\n' "$upd" | grep -c '^up:')" -ge "$FLEET_SIZE" ] || return 0
    local c
    c=$($HUBSSH "$HUB" "ls '$refdir' 2>/dev/null | sed -n 's/^UPDATE\.\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1")
    $HUBSSH "$HUB" "mv '$refdir/UPDATE' '$refdir/UPDATE.$(( ${c:-0} + 1 ))'" 2>/dev/null || true
}

sync_core() {
    local core="$HOME/sparkdata/core"
    mkdir -p "$core/bin"
    apply_manifest core "$core"
}

install_core() {
    local core="$HOME/sparkdata/core" wd="$HOME/sparkdata/weightd" p
    [ -x "$core/bin/sparkpipe_weightd" ] || return 0
    local cand installed
    cand=$(sha16 "$core/bin/sparkpipe_weightd")
    installed=$(sha16 "$wd/sparkpipe_weightd")
    [ "$cand" != "$installed" ] || return 0
    echo "$(date +%T) core: weightd $installed -> $cand; deliberate restart"
    for p in $(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_weightd | sed "s|.*/proc/\([0-9]*\)/exe.*|\1|"); do
        kill -9 "$p" 2>/dev/null
    done
    sleep 1
    mkdir -p "$wd"
    install -m 755 "$core/bin/sparkpipe_weightd" "$wd/sparkpipe_weightd.new"
    mv "$wd/sparkpipe_weightd.new" "$wd/sparkpipe_weightd"
    setsid nohup "$wd/sparkpipe_weightd" --socket /tmp/spark_weightd.sock \
        > "$HOME/weightd.log" 2>&1 < /dev/null &
    sleep 1
}

self_update() {
    local new="$HOME/sparkdata/core/bin/fleet_node_agent.sh"
    [ -f "$new" ] || return 0
    local disk
    disk=$(sha16 "$new")
    { [ "$disk" != none ] && [ -n "$START_SHA" ] && [ "$START_SHA" != none ]; } || return 0
    [ "$disk" != "$START_SHA" ] || return 0
    echo "$(date +%T) agent: self-updating $START_SHA -> $disk ($0)"
    exec bash "$new" "$ROOTS" "$HUB"
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
echo "agent: rank=$RANK roots=$ROOTS hub=$HUB http=$RELEASE_HTTP self=$(sha16 "$0")"
report
ensure_weightd
ensure_root() {
    local name="$1"
    local st; st=$(root_state "$name")
    [ "$st" = "down" ] || return 0
    local up
    up=$(awk '{printf "%d", $1}' /proc/uptime)
    [ "$up" -ge 900 ] || {
        [ -n "$AGENT_BLOCKED" ] || { echo "$(date +%T) $name: node up ${up}s (<15min); autospawn blocked"; AGENT_BLOCKED=1; }
        return 0
    }
    echo "$(date +%T) $name: down; starting"
    restart_root "$name"
}

while true; do
    sync_core
    install_core
    self_update
    IFS=, read -ra RA <<< "$ROOTS"
    for r in "${RA[@]}"; do sync_root "$r"; done
    for r in "${RA[@]}"; do ensure_root "$r"; done
    ensure_api
    report_if_changed
    sleep 5
done
