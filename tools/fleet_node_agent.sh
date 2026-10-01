#!/usr/bin/env bash
set -uo pipefail
ulimit -c unlimited
ROOTS="${1:?comma-separated runtime root names}"
HUB="${2:-sparkf}"
[ "$HUB" = "sparkf" ] && HUB="spec@100.123.97.61"
HOST=$(hostname)
FLEET_HOSTS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
MESH_INTERFACE="rocep1s0f1"
MESH_SGID_INDEX=3
MESH_PAIR_INTERFACE="rocep1s0f0"
MESH_PAIR_SGID_INDEX=3
MESH_TRAFFIC_CLASS=106
RANK=""
_idx=0
for _host in $FLEET_HOSTS; do
    if [ "$_host" = "$HOST" ]; then
        RANK=$_idx
        break
    fi
    _idx=$((_idx + 1))
done
[ -n "$RANK" ] || {
    echo "fleet agent: host '$HOST' is not in FLEET_HOSTS; refusing to start with a guessed rank" >&2
    exit 2
}
PID_FILE="$HOME/.fleet_agent.pid"
VIEW="$HOME/current"
ROOTS_FILE="${FLEET_AGENT_ROOTS_FILE:-$HOME/.fleet_agent_roots}"
HEADROOM_GIB="${FLEET_AGENT_HEADROOM_GIB:-20}"
case "$HEADROOM_GIB" in
    ''|*[!0-9]*)
        echo "fleet agent: FLEET_AGENT_HEADROOM_GIB='$HEADROOM_GIB' is not a whole number of GiB" >&2
        exit 2 ;;
esac
LAST_REPORT=""
LAST_PIDS=""
ROOT_LIST=""
LAST_ROOTS_ERROR=""
declare -A ROOT_NOTE ROOT_REASON

sha16() {
    local s=""
    [ -f "$1" ] && s=$(sha256sum < "$1" 2>/dev/null | cut -c1-16)
    [ -n "$s" ] || s=none
    echo "$s"
}

START_SHA=$(sha16 "$0")
AGENT_BLOCKED=""
mkdir -p "$VIEW"

load_roots() {
    local r list="" bad="" candidates="${ROOTS//,/ }"
    local -a entries=()
    [ -f "$ROOTS_FILE" ] && candidates="$candidates $(tr '\n' ' ' < "$ROOTS_FILE")"
    read -ra entries <<< "$candidates"
    for r in "${entries[@]}"; do
        if [[ ! "$r" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
            bad="$bad $r"
            continue
        fi
        case " $list " in *" $r "*) continue ;; esac
        list="${list:+$list }$r"
    done
    if [ "$bad" != "$LAST_ROOTS_ERROR" ]; then
        [ -n "$bad" ] && echo "$(date +%T) agent: ignoring invalid root names:$bad" >&2
        LAST_ROOTS_ERROR="$bad"
    fi
    ROOT_LIST="$list"
}
load_roots

root_dir() {
    echo "$HOME/sparkdata/$1"
}

root_path() {
    local d="$HOME/sparkdata/$1"
    (cd -P "$d" 2>/dev/null && pwd) || echo "$d"
}

match_pids() {
    local names="$1" rr="$2" p exe
    for p in $(pgrep -f "bin/$names"); do
        [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$rr" ] || continue
        exe=$(readlink "/proc/$p/exe" 2>/dev/null) || continue
        exe="${exe% (deleted)}"
        [[ "${exe##*/}" =~ ^$names$ ]] && echo "$p"
    done
}

root_pid() {
    local p
    p=$(match_pids "sparkpipe_model_residentd" "$(root_path "$1")" | head -1)
    echo "${p:-0}"
}

note_root() {
    local name="$1" state="$2" reason="${3:-}"
    if [ "${ROOT_NOTE[$name]:-}" != "$state" ] && [ -n "$state" ]; then
        echo "$(date +%T) $name: $state${reason:+ ($reason)}" >&2
    fi
    ROOT_NOTE[$name]="$state"
    ROOT_REASON[$name]="$reason"
}

root_state() {
    local rr
    rr=$(root_dir "$1")
    [ -d "$rr" ] || { echo "missing"; return; }
    if [ "$(root_pid "$1")" != 0 ]; then
        if grep -q "model_residentd ready" "$rr/residentd.log" 2>/dev/null; then
            echo "ready"
        else
            echo "starting: $(tail -1 "$rr/residentd.log" 2>/dev/null | cut -c1-90)"
        fi
    elif [ -n "${ROOT_NOTE[$1]:-}" ]; then
        echo "${ROOT_NOTE[$1]}"
    else
        echo "down"
    fi
}

rc_fail() {
    RC_ERROR="$1"
    return 1
}

root_config() {
    local rr n f line key value
    local -a files=()
    rr=$(root_dir "$1")
    printf -v n '%02d' "$RANK"
    RC_LAYOUT=0
    RC_ROLE=production
    RC_INDEX=$RANK
    RC_STAGE="stage_$n.json"
    RC_MEMORY_MAX=""
    RC_NEED_GIB=""
    RC_SYNC=release
    RC_ERROR=""
    RC_ENV=()
    RC_FILES=()
    [ -f "$rr/agent.env" ] && files+=("$rr/agent.env")
    [ -f "$rr/config/env_$n.env" ] && files+=("$rr/config/env_$n.env")
    if [ ! -f "$rr/config/rank_index_$n" ] && [ ${#files[@]} = 0 ]; then
        case ",$ROOTS," in *",$1,"*) return 0 ;; esac
        RC_ROLE=dev
        rc_fail "roots listed in $ROOTS_FILE need agent.env" || return 1
    fi
    RC_LAYOUT=1
    RC_ROLE=dev
    RC_FILES=("${files[@]}")
    if [ -f "$rr/config/rank_index_$n" ]; then
        RC_FILES+=("$rr/config/rank_index_$n")
        value=""
        read -r value < "$rr/config/rank_index_$n"
        [[ "$value" =~ ^[0-9]{1,4}$ ]] || rc_fail "config/rank_index_$n must hold one rank index, found '$value'" || return 1
        RC_INDEX=$((10#$value))
        printf -v RC_STAGE 'stage_%02d.json' "$RC_INDEX"
    fi
    for f in "${files[@]}"; do
        while IFS= read -r line || [ -n "$line" ]; do
            [ -n "$line" ] || continue
            key="${line%%=*}"
            value="${line#*=}"
            [[ "$line" == *=* && "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || rc_fail "${f#"$rr"/}: line is not KEY=VALUE: '${line:0:40}'" || return 1
            case "$key" in
                AGENT_ROLE)
                    [[ "$value" =~ ^(production|dev)$ ]] || rc_fail "AGENT_ROLE must be production or dev, found '$value'" || return 1
                    RC_ROLE=$value ;;
                AGENT_MEMORY_MAX)
                    [[ "$value" =~ ^[1-9][0-9]*[KMGT]?$ ]] || rc_fail "AGENT_MEMORY_MAX must be a systemd size such as 40G, found '$value'" || return 1
                    RC_MEMORY_MAX=$value ;;
                AGENT_MEMORY_NEED_GIB)
                    [[ "$value" =~ ^[0-9]+$ ]] || rc_fail "AGENT_MEMORY_NEED_GIB must be whole GiB, found '$value'" || return 1
                    RC_NEED_GIB=$((10#$value)) ;;
                AGENT_SYNC)
                    [[ "$value" =~ ^(release|local)$ ]] || rc_fail "AGENT_SYNC must be release or local, found '$value'" || return 1
                    RC_SYNC=$value ;;
                AGENT_*)
                    rc_fail "unknown agent key $key in ${f#"$rr"/}" || return 1 ;;
                *)
                    RC_ENV+=("$key=$value") ;;
            esac
        done < "$f"
    done
    [ -f "$rr/config/$RC_STAGE" ] || rc_fail "config/$RC_STAGE missing for rank index $RC_INDEX" || return 1
    if [ "$RC_ROLE" = dev ]; then
        [ -n "$RC_MEMORY_MAX" ] || rc_fail "a dev root needs AGENT_MEMORY_MAX" || return 1
        [ -n "$RC_NEED_GIB" ] || rc_fail "a dev root needs AGENT_MEMORY_NEED_GIB" || return 1
    fi
    return 0
}

root_role() {
    root_config "$1" >/dev/null 2>&1
    echo "$RC_ROLE"
}

root_unit() {
    echo "sp-agent-$1"
}

mem_available_gib() {
    awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo
}

node_uptime_s() {
    awk '{printf "%d", $1}' /proc/uptime
}

root_gate() {
    check_root_gate "$1" || return 1
    note_root "$1" "" ""
}

check_root_gate() {
    local name="$1" r avail
    if [ -f "$(root_dir "$name")/agent.hold" ]; then
        note_root "$name" held "agent.hold present"
        return 1
    fi
    if ! root_config "$name"; then
        note_root "$name" blocked-config "$RC_ERROR"
        return 1
    fi
    [ "$RC_ROLE" = production ] && return 0
    for r in $ROOT_LIST; do
        [ "$r" = "$name" ] && continue
        [ -d "$(root_dir "$r")" ] || continue
        [ -f "$(root_dir "$r")/agent.hold" ] && continue
        [ "$(root_role "$r")" = production ] || continue
        [ "$(root_state "$r")" = ready ] && continue
        note_root "$name" waiting-production "production root $r is not ready"
        return 1
    done
    root_config "$name"
    avail=$(mem_available_gib)
    avail=${avail:-0}
    if [ $((avail - RC_NEED_GIB)) -lt "$HEADROOM_GIB" ]; then
        note_root "$name" blocked-headroom "needs ${RC_NEED_GIB} GiB, MemAvailable ${avail} GiB, headroom ${HEADROOM_GIB} GiB"
        return 1
    fi
    return 0
}

root_rss_mb() {
    local v
    v=$(awk '/^VmRSS/ {print int($2/1024)}' "/proc/$1/status" 2>/dev/null)
    echo "${v:-0}"
}

root_log_age_s() {
    echo $(( $(date +%s) - $(stat -c %Y "$HOME/sparkdata/$1/residentd.log" 2>/dev/null || echo 0) ))
}

json_text() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//[$'\t\r\n']/ }"
    echo "$s"
}

root_json() {
    local r="$1" rr st pid env_sha unit
    rr=$(root_dir "$r")
    st=$(root_state "$r")
    pid=$(root_pid "$r")
    root_config "$r" >/dev/null 2>&1
    env_sha=none
    [ ${#RC_FILES[@]} -gt 0 ] && env_sha=$(cat "${RC_FILES[@]}" 2>/dev/null | sha256sum | cut -c1-16)
    unit=fleet-agent
    [ -n "$RC_MEMORY_MAX" ] && unit=$(root_unit "$r")
    printf '"%s":{"state":"%s","reason":"%s","pid":%s,"rss_mb":%s,"log_age_s":%s,"residentd":"%s","running":"%s","driver":"%s","role":"%s","rank_index":%s,"stage":"%s","unit":"%s","memory_max":"%s","need_gib":"%s","env":"%s"}' \
        "$r" "$(json_text "$st")" "$(json_text "${ROOT_REASON[$r]:-}")" "$pid" "$(root_rss_mb "$pid")" \
        "$(root_log_age_s "$r")" "$(sha16 "$rr/bin/sparkpipe_model_residentd")" \
        "$([ "$pid" != 0 ] && sha16 "/proc/$pid/exe" || echo none)" \
        "$(sha16 "$rr/stages/stage_000/model_driver.so")" "$RC_ROLE" "$RC_INDEX" "$RC_STAGE" \
        "$unit" "$RC_MEMORY_MAX" "$RC_NEED_GIB" "$env_sha"
}

report() {
    local now load mem r states="" sep=',"roots":{'
    now=$(date +%s)
    load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)
    mem=$(awk '/MemAvailable/ {printf "%d", int($2/1048576)}' /proc/meminfo 2>/dev/null)
    {
        printf '{"host":"%s","time":"%s","epoch":%d,"load":"%s","mem_avail_gb":%s,"headroom_gib":%s' \
            "$HOST" "$(date -Is)" "$now" "$load" "${mem:-0}" "$HEADROOM_GIB"
        printf ',"weightd":"%s","agent":"%s"' \
            "$(sha16 "$HOME/sparkdata/weightd/sparkpipe_weightd" 2>/dev/null)" \
            "$(sha16 "$0" 2>/dev/null)"
        for r in $ROOT_LIST; do
            states="$states$r=$(root_state "$r");"
            printf '%s' "$sep"
            root_json "$r"
            sep=','
        done
        [ "$sep" = ',' ] && printf '}'
        printf '}\n'
    } > "$VIEW/$HOST.json"
    LAST_REPORT="$states"
    scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=4 "$VIEW/$HOST.json" \
        "$HUB:current/" 2>/dev/null || true
}

report_if_changed() {
    local r states="" pids=""
    for r in $ROOT_LIST; do
        states="$states$r=$(root_state "$r");"
        pids="$pids$r=$(root_pid "$r");"
    done
    { [ "$states" != "$LAST_REPORT" ] || [ "$pids" != "$LAST_PIDS" ]; } && {
        LAST_PIDS="$pids"
        report
    }
}

drain_match() {
    local names="$1" rr="$2" grace="$3" p t
    for p in $(match_pids "$names" "$rr"); do
        kill -TERM "$p" 2>/dev/null
    done
    for t in $(seq 1 "$grace"); do
        [ -z "$(match_pids "$names" "$rr")" ] && return 0
        sleep 1
    done
    return 1
}

kill_match() {
    local names="$1" rr="$2" p
    for p in $(match_pids "$names" "$rr"); do
        kill -9 "$p" 2>/dev/null
    done
}

drain_root() {
    local name="$1" rr
    rr=$(root_path "$name")
    drain_match "sparkpipe_model_(residentd|api)" "$rr" 15 || {
        echo "$(date +%T) $name: drain deadline hit; kill -9 fallback" >&2
        kill_match "sparkpipe_model_(residentd|api)" "$rr"
        sleep 1
    }
}

yield_dev_roots() {
    local production="$1" r yielded=0
    for r in $ROOT_LIST; do
        [ "$r" = "$production" ] && continue
        [ "$(root_role "$r")" = dev ] || continue
        [ "$(root_pid "$r")" != 0 ] || continue
        echo "$(date +%T) $r: stopping dev root to give memory to production root $production" >&2
        drain_root "$r"
        note_root "$r" yielded "production root $production needed memory"
        yielded=1
    done
    [ "$yielded" = 1 ]
}

wait_memory() {
    local want="$1" t avail
    for t in $(seq 1 30); do
        avail=$(mem_available_gib)
        [ "${avail:-0}" -ge "$want" ] && return 0
        sleep 2
    done
    return 1
}

unload_root() {
    local name="$1" rr pack_gb
    rr=$(root_dir "$name")
    drain_root "$name"
    pack_gb=$(du -sBG "$rr/packs" 2>/dev/null | cut -dG -f1)
    pack_gb=${pack_gb:-0}
    wait_memory $((pack_gb + 8)) && return 0
    if [ "$(root_role "$name")" = production ] && yield_dev_roots "$name"; then
        wait_memory $((pack_gb + 8)) && return 0
    fi
    echo "$(date +%T) $name: MemAvailable never reached $((pack_gb + 8))GB; NOT starting new" >&2
    return 1
}

declare -A BACKOFF NEXT_OK
LAST_ANY_RESTART=0

restart_ok() {
    local cls="$1" now b n
    now=$(date +%s)
    b=${BACKOFF[$cls]:-1}
    n=${NEXT_OK[$cls]:-0}
    [ "$now" -lt "$n" ] && return 1
    [ $(( now - LAST_ANY_RESTART )) -lt 5 ] && return 1
    NEXT_OK[$cls]=$(( now + b ))
    BACKOFF[$cls]=$(( b < 60 ? b * 2 : 60 ))
    LAST_ANY_RESTART=$now
    return 0
}

restart_healthy() {
    BACKOFF[$1]=1
}

start_layout_root() {
    local name="$1" rr="$2" unit kv status
    local -a environment=(CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0 "LD_LIBRARY_PATH=$rr/lib" "${RC_ENV[@]}")
    if [ -z "$RC_MEMORY_MAX" ]; then
        env "${environment[@]}" nohup stdbuf -o0 -e0 ./bin/sparkpipe_model_residentd \
            --deployment model_resident.json --rank-index "$RC_INDEX" \
            > residentd.log 2>&1 < /dev/null &
        return 0
    fi
    unit=$(root_unit "$name")
    local -a command=(systemd-run --user --unit="$unit" --collect -p MemoryMax="$RC_MEMORY_MAX"
        -p MemorySwapMax=0 --working-directory="$rr")
    for kv in "${environment[@]}"; do
        command+=(--setenv="$kv")
    done
    : > residentd.log
    command+=(-p StandardOutput=append:"$rr/residentd.log" -p StandardError=append:"$rr/residentd.log"
        "$rr/bin/sparkpipe_model_residentd" --deployment model_resident.json --rank-index "$RC_INDEX")
    systemctl --user reset-failed "$unit" >/dev/null 2>&1
    "${command[@]}" >/dev/null 2>&1
    status=$?
    if [ "$status" != 0 ]; then
        note_root "$name" failed-start "systemd-run $unit exited $status"
        return 1
    fi
    return 0
}

start_root() {
    local name="$1" rr
    rr=$(root_path "$name")
    [ -n "$(match_pids "sparkpipe_model_(residentd|api)" "$rr")" ] && return 0
    root_gate "$name" || return 1
    if [ "$(grep -c '"rank_index"' "$rr/model_resident.json" 2>/dev/null)" -gt 1 ] && \
       [ ! -f /tmp/weightd-mesh/.ready ]; then
        echo "$(date +%T) $name: waiting for weightd mesh"
        return 0
    fi
    cd "$rr" || return 1
    ln -sf "$RC_STAGE" config/stage.json
    sha16 "$rr/stages/stage_000/model_driver.so" > "$rr/.driver_sha_at_boot" 2>/dev/null
    [ -s residentd.log ] && mv residentd.log "residentd-$(date +%Y%m%d-%H%M%S).log" 2>/dev/null
    note_root "$name" "" ""
    if [ "$RC_LAYOUT" = 1 ]; then
        start_layout_root "$name" "$rr" || return 1
    else
        env CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0 \
            ${G5_LAUNCH_BLOCKING:+CUDA_LAUNCH_BLOCKING=$G5_LAUNCH_BLOCKING} \
            SPARK_WEIGHTD_EXPERT_POOL_BYTES="${G5_EXPERT_POOL_BYTES:-34359738368}" \
            ${G5_PIN_EXPERTS:+SPARK_GLM5_NEXT_PIN_EXPERTS=$G5_PIN_EXPERTS} \
            ${G5_GRAPH_PATH:+SPARK_GLM5_NEXT_GRAPH_PATH=$G5_GRAPH_PATH} \
            LD_LIBRARY_PATH="$rr/lib" nohup stdbuf -o0 -e0 ./bin/sparkpipe_model_residentd \
            --deployment model_resident.json --rank-index "$RANK" \
            > residentd.log 2>&1 < /dev/null &
    fi
    report
}

restart_root() {
    local name="$1"
    cd "$HOME/sparkdata/$1" || return 1
    unload_root "$name" || return 1
    start_root "$name"
}

HUBSSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-agent-%r@%h:%p -o ControlPersist=600"
if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=4 "$HUB" true 2>/dev/null; then
    ssh-keyscan -H "${HUB#*@}" >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
fi
RELEASE_HTTP="${FLEET_HTTP_RELEASE:-http://100.123.97.61:8802}"

sync_rendezvous() {
    local name="$1" host rd
    host=$(hostname -s)
    rd="$HOME/sparkdata/$name/rendezvous"
    if [ -d "$rd" ]; then
        if [ ! -f "$rd/.shipped" ] || [ -n "$(find "$rd" -name '*.rec' -newer "$rd/.shipped" 2>/dev/null | head -1)" ]; then
            [ -f "$rd/.upload_lock" ] && [ $(( $(date +%s) - $(stat -c %Y "$rd/.upload_lock") )) -lt 3 ] && return 0
            touch "$rd/.upload_lock"
            $HUBSSH "$HUB" "mkdir -p release/qpn/$host/$name" 2>/dev/null
            scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=4 "$rd"/*.rec \
                "$HUB:release/qpn/$host/$name/" 2>/dev/null
            $HUBSSH "$HUB" "cd release/qpn/$host/$name && sha256sum *.rec > index.txt.\$\$ 2>/dev/null && mv index.txt.\$\$ index.txt" 2>/dev/null
            touch "$rd/.shipped"
        fi
    fi
    local mesh_dir="/tmp/weightd-mesh"
    if [ -d "$mesh_dir" ]; then
        local own_rank
        printf -v own_rank '%x' "$RANK"
        local own_rec="$mesh_dir/mesh-$own_rank.rec"
        if [ -f "$own_rec" ]; then
            local sum
            sum=$(sha256sum "$own_rec" | cut -d' ' -f1)
            if [ ! -f "$mesh_dir/.shipped_sha" ] || \
               [ "$(cat "$mesh_dir/.shipped_sha" 2>/dev/null)" != "$sum" ]; then
                $HUBSSH "$HUB" "mkdir -p release/qpn/$host/mesh" 2>/dev/null
                scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=4 "$own_rec" \
                    "$HUB:release/qpn/$host/mesh/" 2>/dev/null && \
                    echo "$sum" > "$mesh_dir/.shipped_sha"
            fi
        fi
        local pr pn fn now age
        now=$(date +%s)
        for pr in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
            pn="spark$pr"
            [ "$pn" = "$host" ] && continue
            fn="$mesh_dir/mesh-$pr.rec"
            if [ -f "$fn" ]; then
                age=$(( now - $(stat -c %Y "$fn" 2>/dev/null || echo "$now") ))
                [ "$age" -lt 2 ] && continue
            fi
            if curl -sf --max-time 2 "$RELEASE_HTTP/qpn/$pn/mesh/mesh-$pr.rec" \
                -o "$fn.tmp" 2>/dev/null; then
                mv "$fn.tmp" "$fn"
            else
                rm -f "$fn.tmp"
            fi
        done
    fi
}

apply_manifest() {
    local name="$1" root="$2" scope="${3:-/dev/null}" manifest_cur manifest_applied
    manifest_cur="/tmp/fleet_manifest_$name.txt"
    manifest_applied="$root/.applied_manifest"
    if ! curl -sf --max-time 8 "$RELEASE_HTTP/$name/MANIFEST" -o "$manifest_cur"; then
        [ -f "$manifest_applied" ] || echo "$(date +%T) $name: manifest unreachable" >&2
        return 1
    fi
    cmp -s "$manifest_cur" "$manifest_applied" 2>/dev/null && return 1
    echo "$(date +%T) $name: manifest changed; syncing diff"
    : > "$scope"
    local fetch_errors=0 rel want tmp got old_f="$manifest_applied"
    [ -f "$old_f" ] || old_f=/dev/null
    while IFS=' ' read -r rel want; do
        [ -n "$rel" ] || continue
        tmp="$root/.fetch.tmp"
        if ! curl -sf --max-time 300 "$RELEASE_HTTP/$name/$rel" -o "$tmp"; then
            echo "$(date +%T) $name: fetch failed: $rel" >&2
            fetch_errors=$((fetch_errors+1))
            continue
        fi
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
        echo "$rel" >> "$scope"
    done < <(awk 'NR==FNR { old[$2]=$1; next } $2 != "" && old[$2] != $1 { print $2, $1 }' \
        "$old_f" "$manifest_cur")
    [ "$fetch_errors" != 0 ] && { echo "$(date +%T) $name: $fetch_errors fetch errors; retrying next cycle" >&2; return 1; }
    cp "$manifest_cur" "$manifest_applied"
    return 0
}

restart_scope() {
    local scope="$1" rel kind="none"
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        case "$rel" in
            bin/sparkpipe_model_residentd|lib/*|stages/*|config/model_resident.json|agent.env|config/rank_index_*|config/env_*.env)
                echo "root"
                return ;;
            bin/sparkpipe_model_api)
                kind="api" ;;
            config/stage_*.json)
                [ "$kind" = "none" ] && kind="stage" ;;
        esac
    done < "$scope"
    echo "$kind"
}

drain_api() {
    local name="$1"
    drain_match "sparkpipe_model_api" "$(root_path "$name")" 10 || kill_match "sparkpipe_model_api" "$(root_path "$name")"
}

sync_root() {
    local name="$1" kind="none"
    local root="$HOME/sparkdata/$name" scope="$HOME/sparkdata/$name/.changed_scope"
    root_config "$name" >/dev/null 2>&1
    [ "$RC_SYNC" = local ] && return 0
    mkdir -p "$root"
    if apply_manifest "$name" "$root" "$scope"; then
        kind=$(restart_scope "$scope")
    fi
    rm -f "$scope"
    case "$kind" in
        root)
            unload_root "$name" || return 0
            start_root "$name" ;;
        api) drain_api "$name" ;;
        stage)
            root_config "$name" >/dev/null 2>&1
            ln -sf "$RC_STAGE" "$root/config/stage.json" ;;
    esac
}

sync_core() {
    local core="$HOME/sparkdata/core"
    mkdir -p "$core/bin"
    apply_manifest core "$core"
}

install_core() {
    local core="$HOME/sparkdata/core" wd="$HOME/sparkdata/weightd" announced
    [ -x "$core/bin/sparkpipe_weightd" ] || return 0
    announced=$(curl -sf --max-time 5 "$RELEASE_HTTP/core/WEIGHTSD_BIN" 2>/dev/null | tr -d "[:space:]") || return 0
    [ -n "$announced" ] || return 0
    [ "$(sha16 "$core/bin/sparkpipe_weightd")" = "$announced" ] || return 0
    [ "$(sha16 "$wd/sparkpipe_weightd")" != "$announced" ] || return 0
    echo "$(date +%T) core: installing announced weightd $announced; weightsd owns the restart"
    mkdir -p "$wd"
    install -m 755 "$core/bin/sparkpipe_weightd" "$wd/sparkpipe_weightd.new"
    mv "$wd/sparkpipe_weightd.new" "$wd/sparkpipe_weightd"

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

node_doctor() {
    local state netdev
    state=$(ibv_devinfo "$MESH_INTERFACE" 2>/dev/null | awk '/^[[:space:]]*state:/ {print $2; exit}')
    case "$state" in
        PORT_ACTIVE) ;;
        *)
            netdev=$(ibdev2netdev 2>/dev/null | awk -v d="$MESH_INTERFACE" '$1==d {print $NF; exit}')
            [ -n "$netdev" ] || return 0
            echo "$(date +%T) doctor: $MESH_INTERFACE state=${state:-missing}; flapping $netdev" >&2
            sudo -n ip link set "$netdev" down 2>/dev/null
            sleep 2
            sudo -n ip link set "$netdev" up 2>/dev/null
            ;;
    esac
}

janitor() {
    local name q a youngest pids
    for name in $ROOT_LIST; do
        pids=$(match_pids "sparkpipe_model_residentd" "$(root_path "$name")")
        youngest=0
        for q in $pids; do
            a=$(ps -o etimes= -p "$q" 2>/dev/null | tr -d ' ')
            [ -n "$a" ] && [ "$a" -gt "$youngest" ] && youngest=$a
        done
        for q in $pids; do
            a=$(ps -o etimes= -p "$q" 2>/dev/null | tr -d ' ')
            [ -n "$a" ] && [ "$a" -lt "$youngest" ] && [ "$a" -gt 1800 ] && {
                echo "$(date +%T) janitor: $name: killing stale residentd pid=$q age=${a}s (current is younger)" >&2
                kill -9 "$q" 2>/dev/null
            }
        done
    done
}

ensure_weightd() {
    local wdd="$HOME/sparkdata/weightd" q owner="" executable exe_sha disk_sha
    for q in $(pgrep -f "sparkpipe_weightd"); do
        executable=$(readlink "/proc/$q/exe" 2>/dev/null) || continue
        case "$executable" in
            "$wdd/sparkpipe_weightd"|"$wdd/sparkpipe_weightd (deleted)")
                if [ -n "$owner" ]; then
                    echo "weightd: multiple production owners ($owner, $q); refusing automatic cleanup" >&2
                    return 1
                fi
                owner=$q ;;
            */sparkpipe_weightd|*/"sparkpipe_weightd (deleted)")
                echo "weightd: unknown owner $q at $executable; refusing automatic startup" >&2
                return 1 ;;
        esac
    done
    if [ -n "$owner" ]; then
        exe_sha=$(sha16 "/proc/$owner/exe")
        disk_sha=$(sha16 "$wdd/sparkpipe_weightd")
        if [ "$exe_sha" = none ] || [ "$disk_sha" = none ]; then
            echo "weightd: cannot verify running/installed identity; owner $owner retained" >&2
            return 1
        fi
        if [ "$exe_sha" != "$disk_sha" ]; then
            if pgrep -f "bin/sparkpipe_model_residentd" >/dev/null; then
                echo "weightd: update $exe_sha -> $disk_sha requires dependent engines to drain; owner $owner retained" >&2
                return 1
            fi
            echo "weightd: stopping owned pid $owner for installed update; waiting for process exit"
            kill -TERM "$owner" 2>/dev/null
            return 1
        fi
        if [ -S /tmp/spark_weightd.sock ] && python3 -c 'import socket; s=socket.socket(socket.AF_UNIX); s.settimeout(3); s.connect("/tmp/spark_weightd.sock"); s.close()' 2>/dev/null; then
            return 0
        fi
        echo "weightd: owned pid $owner alive, control socket not ready; process and mappings retained" >&2
        return 1
    fi
    local home="$HOME/sparkdata/weightd"
    [ -x "$home/sparkpipe_weightd" ] || {
        echo "weightd: missing executable $home/sparkpipe_weightd; dependent startup blocked" >&2
        return 1
    }
    systemctl is-active -q sparkpipe-roce-qos || {
        echo "weightd: sparkpipe-roce-qos is not active; traffic class $MESH_TRAFFIC_CLASS needs DSCP trust and PFC on $MESH_INTERFACE; dependent startup blocked" >&2
        return 1
    }
    restart_ok weightd || return 1
    rm -f /tmp/weightd-mesh/mesh-*.rec /tmp/weightd-mesh/.ready 2>/dev/null
    echo "$(date +%T) weightd: starting (backoff ${BACKOFF[weightd]:-1}s)"
    [ -s "$HOME/weightd.log" ] && mv "$HOME/weightd.log" "$HOME/weightd-$(date +%Y%m%d-%H%M%S).log" 2>/dev/null
    setsid nohup "$home/sparkpipe_weightd" --socket /tmp/spark_weightd.sock \
        --mesh-rank "$RANK" --mesh-rank-mask 0xffff --mesh-interface "$MESH_INTERFACE" \
        --mesh-sgid-index "$MESH_SGID_INDEX" \
        --mesh-pair-interface "$MESH_PAIR_INTERFACE" --mesh-pair-sgid-index "$MESH_PAIR_SGID_INDEX" \
        --mesh-traffic-class "$MESH_TRAFFIC_CLASS" \
        > "$HOME/weightd.log" 2>&1 < /dev/null &
    rm -f /tmp/weightd-mesh/.shipped_sha 2>/dev/null
    ( sleep 2; sync_rendezvous "glm53flash.fp8.tp16" ) >/dev/null 2>&1 &
    return 1
}

prune_older() {
    find "$1" -maxdepth 1 -name "$2" -printf '%T@ %p\n' 2>/dev/null | sort -rn | tail -n +21 | \
        cut -d' ' -f2- | xargs -r -d '\n' rm -f
}

prune_logs() {
    prune_older "$1" 'residentd-2*.log'
    prune_older "$1" 'api-2*.log'
    prune_older "$HOME" 'weightd-2*.log'
}

ensure_root() {
    local name="$1" rr st p exe_sha disk_sha drv_sha start_s up_s up
    rr=$(root_dir "$name")
    [ -d "$rr" ] || return 0
    if [ -f "$rr/agent.hold" ]; then
        [ "$(root_pid "$name")" != 0 ] && drain_root "$name"
        note_root "$name" held "agent.hold present"
        return 0
    fi
    st=$(root_state "$name")
    p=$(root_pid "$name")
    if [ "$p" != 0 ]; then
        note_root "$name" "" ""
        drv_sha=$(sha16 "$rr/stages/stage_000/model_driver.so")
        exe_sha=$(sha16 "/proc/$p/exe")
        disk_sha=$(sha16 "$rr/bin/sparkpipe_model_residentd")
        if [ "$exe_sha" != "$disk_sha" ]; then
            echo "$(date +%T) $name: running residentd $exe_sha != disk $disk_sha; recycling"
            st="down"
        elif [ -f "$rr/.driver_sha_at_boot" ] && [ "$drv_sha" != "$(cat "$rr/.driver_sha_at_boot")" ]; then
            echo "$(date +%T) $name: driver changed since engine boot; recycling"
            st="down"
        fi
    fi
    [ "$p" = 0 ] || [ "$st" = "down" ] || {
        start_s=$(awk '{print $22}' "/proc/$p/stat" 2>/dev/null)
        up_s=$(awk '{printf "%d", $1}' /proc/uptime)
        [ -n "$start_s" ] && [ $(( up_s - start_s / 100 )) -gt 120 ] && restart_healthy "engine-$name"
        return 0
    }
    up=$(node_uptime_s)
    [ "$up" -ge 900 ] || {
        [ -n "$AGENT_BLOCKED" ] || { echo "$(date +%T) $name: node up ${up}s (<15min); autospawn blocked"; AGENT_BLOCKED=1; }
        return 0
    }
    [ "$p" != 0 ] || root_gate "$name" || return 0
    restart_ok "engine-$name" || return 0
    echo "$(date +%T) $name: down; starting (backoff ${BACKOFF[engine-$name]:-1}s)"
    restart_root "$name"
}

LAST_WARM_GEN=""
LAST_WARM_TS=0
warmup_hook() {
    [ "${RANK:-1}" = "0" ] || return 0
    [ "${G5_WARMUP:-1}" = "1" ] || return 0
    local gen now
    [ "$(root_state glm53flash.fp8.tp16 2>/dev/null)" = "ready" ] || return 0
    gen=$(root_pid glm53flash.fp8.tp16 2>/dev/null)
    [ -n "$gen" ] || return 0
    now=$(date +%s)
    if [ -s /tmp/fleet-warmup.pid ]; then
        local wpid; wpid=$(cat /tmp/fleet-warmup.pid 2>/dev/null)
        if [ -n "$wpid" ] && grep -q "v1/completions" "/proc/$wpid/cmdline" 2>/dev/null; then
            return 0
        fi
        rm -f /tmp/fleet-warmup.pid
    fi
    if [ "$gen" = "$LAST_WARM_GEN" ]; then
        grep -q '"tokens"' /tmp/fleet-warmup.out 2>/dev/null && return 0
        [ $(( now - LAST_WARM_TS )) -lt 240 ] && return 0
    fi
    LAST_WARM_GEN=$gen
    LAST_WARM_TS=$now
    (
      sleep 45
      curl -sf --max-time 900 -X POST "http://${G5_API_HOST:-100.123.97.61}:${G5_API_PORT:-8433}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"prompt_token_ids":[1,2,3,4,5,6,7,8],"max_tokens":4,"temperature":0}' \
        > /tmp/fleet-warmup.out 2>&1
    ) &
    echo $! > /tmp/fleet-warmup.pid
    echo "$(date +%T) warmup: fired for engine pid $gen (one cold pass, 900s budget)"
}

echo "$$" > "$PID_FILE"
echo "agent: rank=$RANK roots=$ROOT_LIST roots_file=$ROOTS_FILE headroom=${HEADROOM_GIB}GiB hub=$HUB http=$RELEASE_HTTP self=$(sha16 "$0")"
report

while true; do
    load_roots
    sync_core
    install_core
    self_update
    node_doctor
    janitor
    if ! ensure_weightd; then
        report_if_changed
        sleep 1
        continue
    fi
    for r in $ROOT_LIST; do sync_root "$r"; done
    for r in $ROOT_LIST; do sync_rendezvous "$r"; done
    for r in $ROOT_LIST; do ensure_root "$r"; done
    for r in $ROOT_LIST; do prune_logs "$HOME/sparkdata/$r"; done
    warmup_hook
    report_if_changed
    sleep 1
done
