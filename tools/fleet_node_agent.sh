#!/usr/bin/env bash
set -uo pipefail
REF_BASE="${1:?reference base dir}"
ROOTS="${2:?comma-separated runtime root names}"
HUB="${3:-sparkf}"
HOST=$(hostname)
RANK=$((16#${HOST#spark}))
PID_FILE="$HOME/.fleet_agent.pid"
VIEW="$HOME/current"          # local copy of the report
LAST_REPORT=""
LAST_PIDS=""
LAST_START=0
LAST_API_START=0
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
    local boot_id
    boot_id=$(awk '{print $1}' /proc/sys/kernel/random/boot_id 2>/dev/null | cut -c1-8)
    {
        printf '{"host":"%s","time":"%s","boot":"%s","epoch":%s' \
            "$HOST" "$(date -Is)" "${boot_id:-?}" "$(date +%s)"
        printf ',"load":%.2f,"mem_avail_gb":%d' \
            "$(awk '{print $1}' /proc/loadavg)" \
            "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)"
        printf ',"weightd":"%s"' "$(sha16 "$HOME/sparkdata/weightd/sparkpipe_weightd" 2>/dev/null)"
        local r first=1 states=""
        IFS=, read -ra RA <<< "$ROOTS"
        for r in "${RA[@]}"; do
            local rr="$HOME/sparkdata/$r"
            [ -d "$rr" ] || continue
            local st pid etime rss_mb log_age
            st=$(root_state "$r")
            pid=0; etime="-"; rss_mb=0; log_age=-1
            for l in $(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model_residentd | sed "s|.*/proc/\([0-9]*\)/exe.*|\1|"); do
                [ "$(readlink /proc/$l/cwd 2>/dev/null)" = "$rr" ] || continue
                pid=$l
                etime=$(ps -o etimes= -p "$l" 2>/dev/null | tr -d ' ')
                rss_mb=$(awk '/VmRSS/ {print int($2/1024)}' "/proc/$l/status" 2>/dev/null)
                break
            done
            if [ "$pid" != 0 ] && [ -f "$rr/residentd.log" ]; then
                log_age=$(( $(date +%s) - $(stat -c %Y "$rr/residentd.log" 2>/dev/null || echo 0) ))
            fi
            states="$states$r=$st;"
            printf '%s"%s":{"state":"%s","pid":%s,"age_s":%s,"rss_mb":%s,"log_age_s":%s,"residentd":"%s","driver":"%s"}' \
                "$([ $first = 1 ] && echo ,roots:{ || echo ,)" "$r" \
                "${st//\"/\\\"}" "$pid" "${etime:--1}" "$rss_mb" "$log_age" \
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
    LD_LIBRARY_PATH="$rr/lib" nohup ./bin/sparkpipe_model_residentd \
        --deployment model_resident.json --rank-index "$RANK" \
        > residentd.log 2>&1 < /dev/null &
    report
}

ensure_api() {
    [ "$RANK" = 0 ] || return 0
    local rr="$HOME/sparkdata/glm53flash.fp8.tp16"
    [ -x "$rr/bin/sparkpipe_model_api" ] || return 0
    local ready_count now
    ready_count=$(ssh -o BatchMode=yes -o ConnectTimeout=4 sparkf \
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
    ensure_api
    report_if_changed
    sleep 5
done
