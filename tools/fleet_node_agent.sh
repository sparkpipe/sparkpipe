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
        if grep -q "model_residentd ready" "$rr/residentd.log" 2>/dev/null; then
            echo "ready"
        else
            echo "starting: $(tail -1 "$rr/residentd.log" 2>/dev/null | cut -c1-90)"
        fi
    else
        echo "down"
    fi
}

root_pid() {
    local rr="$HOME/sparkdata/$1" l
    for l in $(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model_residentd | sed "s|.*/proc/\([0-9]*\)/exe.*|\1|"); do
        [ "$(readlink /proc/$l/cwd 2>/dev/null)" = "$rr" ] && { echo "$l"; return; }
    done
    echo 0
}

root_rss_mb() {
    local v
    v=$(awk '/^VmRSS/ {print int($2/1024)}' "/proc/$1/status" 2>/dev/null)
    echo "${v:-0}"
}

root_log_age_s() {
    echo $(( $(date +%s) - $(stat -c %Y "$HOME/sparkdata/$1/residentd.log" 2>/dev/null || echo 0) ))
}

report() {
    local now load mem r rr st pid states="" first=1
    now=$(date +%s)
    load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)
    mem=$(awk '/MemAvailable/ {printf "%d", int($2/1048576)}' /proc/meminfo 2>/dev/null)
    {
        printf '{"host":"%s","time":"%s","epoch":%d,"load":"%s","mem_avail_gb":%s' \
            "$HOST" "$(date -Is)" "$now" "$load" "${mem:-0}"
        printf ',"weightd":"%s","agent":"%s"' \
            "$(sha16 "$HOME/sparkdata/weightd/sparkpipe_weightd" 2>/dev/null)" \
            "$(sha16 "$0" 2>/dev/null)"
        IFS=, read -ra RA <<< "$ROOTS"
        for r in "${RA[@]}"; do
            rr="$HOME/sparkdata/$r"
            [ -d "$rr" ] || continue
            st=$(root_state "$r")
            states="$states$r=$st;"
            pid=$(root_pid "$r")
            printf '%s"%s":{"state":"%s","pid":%s,"rss_mb":%s,"log_age_s":%s,"residentd":"%s","driver":"%s"}' \
                "$([ $first = 1 ] && echo ',"roots":{' || echo ',')" "$r" \
                "${st//\"/\\\"}" "$pid" "$(root_rss_mb "$pid")" "$(root_log_age_s "$r")" \
                "$(sha16 "$rr/bin/sparkpipe_model_residentd")" \
                "$(sha16 "$rr/stages/stage_000/model_driver.so")"
            first=0
        done
        [ $first = 0 ] && printf '}'
        printf '}\n'
    } > "$VIEW/$HOST.json"
    LAST_REPORT="$states"
    scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=4 "$VIEW/$HOST.json" \
        "$HUB:current/" 2>/dev/null || true
}

report_if_changed() {
    local r states="" pids="" pid
    IFS=, read -ra RA <<< "$ROOTS"
    for r in "${RA[@]}"; do
        [ -d "$HOME/sparkdata/$r" ] || continue
        states="$states$r=$(root_state "$r");"
        pid=$(root_pid "$r")
        pids="$pids$r=${pid:-0};"
    done
    { [ "$states" != "$LAST_REPORT" ] || [ "$pids" != "$LAST_PIDS" ]; } && {
        LAST_PIDS="$pids"
        report
    }
}

drain_match() {
    local pattern="$1" rr="$2" grace="$3" p t gone=0
    for p in $(pgrep -f "$pattern"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && kill -TERM "$p" 2>/dev/null
    done
    for t in $(seq 1 "$grace"); do
        gone=1
        for p in $(pgrep -f "$pattern"); do
            [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && gone=0
        done
        [ "$gone" = 1 ] && return 0
        sleep 1
    done
    return 1
}

kill_match() {
    local pattern="$1" rr="$2" p
    for p in $(pgrep -f "$pattern"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && kill -9 "$p" 2>/dev/null
    done
}

unload_root() {
    local name="$1" rr="$HOME/sparkdata/$1" pack_gb t avail
    drain_match "bin/sparkpipe_model_(residentd|api)" "$rr" 15 || {
        echo "$(date +%T) $name: drain deadline hit; kill -9 fallback" >&2
        kill_match "bin/sparkpipe_model_(residentd|api)" "$rr"
        sleep 1
    }
    pack_gb=$(du -sBG "$rr/packs" 2>/dev/null | cut -dG -f1)
    pack_gb=${pack_gb:-0}
    for t in $(seq 1 30); do
        avail=$(awk "/MemAvailable/ {print int(\$2/1048576)}" /proc/meminfo)
        [ "$avail" -ge $((pack_gb + 8)) ] && return 0
        sleep 2
    done
    echo "$(date +%T) $1: MemAvailable never reached $((pack_gb + 8))GB; NOT starting new" >&2
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

start_root() {
    local name="$1" rr="$HOME/sparkdata/$1" p
    for p in $(pgrep -f "bin/sparkpipe_model_(residentd|api)"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && return 0
    done
    if [ "$(grep -c '"rank_index"' "$rr/model_resident.json" 2>/dev/null)" -gt 1 ] && \
       [ ! -f /tmp/weightd-mesh/.ready ]; then
        echo "$(date +%T) $name: waiting for weightd mesh"
        return 0
    fi
    cd "$rr" || return 1
    ln -sf "stage_$(printf %02d "$RANK").json" config/stage.json
    sha16 "$rr/stages/stage_000/model_driver.so" > "$rr/.driver_sha_at_boot" 2>/dev/null
    [ -s residentd.log ] && mv residentd.log "residentd-$(date +%Y%m%d-%H%M%S).log" 2>/dev/null
    env CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0 \
        ${G5_LAUNCH_BLOCKING:+CUDA_LAUNCH_BLOCKING=$G5_LAUNCH_BLOCKING} \
        SPARK_WEIGHTD_EXPERT_POOL_BYTES="${G5_EXPERT_POOL_BYTES:-34359738368}" \
        ${G5_PIN_EXPERTS:+SPARK_GLM5_NEXT_PIN_EXPERTS=$G5_PIN_EXPERTS} \
    ${G5_GRAPH_PATH:+SPARK_GLM5_NEXT_GRAPH_PATH=$G5_GRAPH_PATH} \
    LD_LIBRARY_PATH="$rr/lib" nohup stdbuf -o0 -e0 ./bin/sparkpipe_model_residentd \
        --deployment model_resident.json --rank-index "$RANK" \
        > residentd.log 2>&1 < /dev/null &
    report
}

api_root() {
    local r
    [ -n "${G5_API_ROOT:-}" ] && { echo "$G5_API_ROOT"; return; }
    IFS=, read -ra RA <<< "$ROOTS"
    for r in "${RA[@]}"; do
        [ -x "$HOME/sparkdata/$r/bin/sparkpipe_model_api" ] && { echo "$r"; return; }
    done
    echo "${RA[0]}"
}

ensure_api() {
    [ "$RANK" = 0 ] || return 0
    [ -n "$G5_API_DISABLED" ] && return 0
    local rr="$HOME/sparkdata/$(api_root)"
    [ -x "$rr/bin/sparkpipe_model_api" ] || return 0
    local ready_count now
    ready_count=$(ssh -o BatchMode=yes -o ConnectTimeout=4 "$HUB" \
        "grep -l '\"state\":\"ready' current/*.json 2>/dev/null | wc -l" 2>/dev/null)
    [ "${ready_count:-0}" -ge 16 ] || return 0
    local p rpid
    proc_start() { awk '{print $22}' "/proc/$1/stat" 2>/dev/null || echo 0; }
    for p in $(pgrep -f "bin/sparkpipe_model_api"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] || continue
        rpid=$(pgrep -f "bin/sparkpipe_model_residentd" | head -1)
        if [ -n "$rpid" ] && [ "$(proc_start "$rpid")" -gt "$(proc_start "$p")" ]; then
            echo "$(date +%T) api: predates residentd; draining"
            drain_match "bin/sparkpipe_model_api" "$rr" 10 || kill_match "bin/sparkpipe_model_api" "$rr"
            return 0
        fi
        return 0
    done
    now=$(date +%s)
    [ $((now - LAST_API_START)) -lt 15 ] && return 0
    LAST_API_START=$now
    echo "$(date +%T) api: starting"
    cd "$rr" || return 1
    [ -s api.log ] && mv api.log "api-$(date +%Y%m%d-%H%M%S).log" 2>/dev/null
    ${G5_MAX_PREFILL_ROWS:+SPARK_MODEL_API_MAX_PREFILL_ROWS="$G5_MAX_PREFILL_ROWS"} \
    ${G5_INFLIGHT_BUDGET_NS:+SPARK_BATCH_INFLIGHT_BUDGET_NS="$G5_INFLIGHT_BUDGET_NS"} \
    LD_LIBRARY_PATH="$rr/lib" setsid nohup stdbuf -o0 -e0 ./bin/sparkpipe_model_api \
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
                [ "$age" -lt 10 ] && continue
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
    local fetch_errors=0 line rel want tmp got old_f="$manifest_applied"
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
            bin/sparkpipe_model_residentd|lib/*|stages/*|config/model_resident.json)
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
    local name="$1" rr="$HOME/sparkdata/$1"
    drain_match "bin/sparkpipe_model_api" "$rr" 10 || kill_match "bin/sparkpipe_model_api" "$rr"
}

sync_root() {
    local name="$1" kind="none"
    local root="$HOME/sparkdata/$name" scope="$HOME/sparkdata/$name/.changed_scope"
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
        stage) ln -sf "stage_$(printf %02d "$RANK").json" "$root/config/stage.json" ;;
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
    local q youngest a holder_exe
    for name in ${ROOTS//,/ }; do
        local rr="$HOME/sparkdata/$name" youngest=0
        for q in $(pgrep -f "bin/sparkpipe_model_residentd"); do
            [ "$(readlink /proc/$q/cwd 2>/dev/null)" = "$rr" ] || continue
            a=$(ps -o etimes= -p "$q" 2>/dev/null | tr -d ' ')
            [ -n "$a" ] && [ "$a" -gt "$youngest" ] && youngest=$a
        done
        for q in $(pgrep -f "bin/sparkpipe_model_residentd"); do
            [ "$(readlink /proc/$q/cwd 2>/dev/null)" = "$rr" ] || continue
            a=$(ps -o etimes= -p "$q" 2>/dev/null | tr -d ' ')
            [ -n "$a" ] && [ "$a" -lt "$youngest" ] && [ "$a" -gt 1800 ] && {
                echo "$(date +%T) janitor: killing stale residentd pid=$q age=${a}s (current is younger)" >&2
                kill -9 "$q" 2>/dev/null
            }
        done
    done
    for q in $(pgrep -f "sparkpipe_weightd"); do
        a=$(ps -o etimes= -p "$q" 2>/dev/null | tr -d ' ')
        [ -n "$a" ] && [ "$a" -gt 1800 ] || continue
        holder_exe=$(sudo -n fuser /tmp/spark_weightd.singleton 2>/dev/null | tr -s ' ' | cut -d: -f2 | tr -d ' ')
        [ "$q" = "$holder_exe" ] && continue
        echo "$(date +%T) janitor: killing stale weightd pid=$q age=${a}s (not the singleton holder)" >&2
        kill -9 "$q" 2>/dev/null
    done
}

ensure_weightd() {
    if pgrep -f "sparkpipe_weightd" >/dev/null; then
        local wdd="$HOME/sparkdata/weightd" q exe_sha disk_sha
        for q in $(pgrep -f "sparkdata/weightd/sparkpipe_weightd"); do
            exe_sha=$(sha16 "$(readlink /proc/$q/exe 2>/dev/null)")
            disk_sha=$(sha16 "$wdd/sparkpipe_weightd")
            if [ -n "$exe_sha" ] && [ "$exe_sha" != "$disk_sha" ]; then
                echo "$(date +%T) weightd: running $exe_sha != installed $disk_sha; recycling"
                kill -TERM "$q" 2>/dev/null
                sleep 3
                kill -9 "$q" 2>/dev/null
            fi
        done
        local youngest=0 p start_s up_s
        for p in $(pgrep -f "sparkpipe_weightd"); do
            start_s=$(awk '{print $22}' "/proc/$p/stat" 2>/dev/null)
            [ -n "$start_s" ] && [ "$start_s" -gt "$youngest" ] && youngest=$start_s
        done
        up_s=$(awk '{printf "%d", $1}' /proc/uptime)
        if [ "$youngest" -gt 0 ] && [ $(( up_s - youngest / 100 )) -lt 120 ]; then
            return 0
        fi
        local probe_ok=0 probe_i
        for probe_i in 1 2 3; do
            if [ -S /tmp/spark_weightd.sock ] && python3 -c "import socket,sys; s=socket.socket(socket.AF_UNIX); s.settimeout(3); s.connect(\"/tmp/spark_weightd.sock\"); s.close()" 2>/dev/null; then
                probe_ok=1
                break
            fi
            sleep 2
        done
        [ "$probe_ok" = 1 ] && return 0
        echo "$(date +%T) weightd: stale or unresponsive instance(s); clearing"
        for p in $(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_weightd | sed "s|.*/proc/\([0-9]*\)/exe.*|\1|"); do
            kill -9 "$p" 2>/dev/null
        done
        sleep 2
        rm -f /tmp/spark_weightd.sock
    fi
    local home="$HOME/sparkdata/weightd"
    [ -x "$home/sparkpipe_weightd" ] || return 0
    restart_ok weightd || return 0
    rm -f /tmp/weightd-mesh/mesh-*.rec /tmp/weightd-mesh/.ready 2>/dev/null
    echo "$(date +%T) weightd: starting (backoff ${BACKOFF[weightd]:-1}s)"
    [ -s "$HOME/weightd.log" ] && mv "$HOME/weightd.log" "$HOME/weightd-$(date +%Y%m%d-%H%M%S).log" 2>/dev/null
    setsid nohup "$home/sparkpipe_weightd" --socket /tmp/spark_weightd.sock \
        --mesh-rank "$RANK" --mesh-interface "$MESH_INTERFACE" \
        --mesh-sgid-index "$MESH_SGID_INDEX" \
        > "$HOME/weightd.log" 2>&1 < /dev/null &
}

prune_logs() {
    ls -t "$1"/residentd-2*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
    ls -t "$1"/api-2*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
    ls -t "$HOME"/weightd-2*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
}

echo "$$" > "$PID_FILE"
echo "agent: rank=$RANK roots=$ROOTS hub=$HUB http=$RELEASE_HTTP self=$(sha16 "$0")"
report
ensure_weightd
ensure_root() {
    local name="$1"
    local st; st=$(root_state "$name")
    [ "$st" = "down" ] || {
        local rr="$HOME/sparkdata/$name" p exe_sha disk_sha drv_sha
        drv_sha=$(sha16 "$rr/stages/stage_000/model_driver.so")
        for p in $(pgrep -f "bin/sparkpipe_model_residentd"); do
            [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] || continue
            exe_sha=$(sha16 "$(readlink /proc/$p/exe)")
            disk_sha=$(sha16 "$rr/bin/sparkpipe_model_residentd")
            if [ "$exe_sha" != "$disk_sha" ]; then
                echo "$(date +%T) $name: running residentd $exe_sha != disk $disk_sha; recycling"
                st="down"
            elif [ -f "$rr/.driver_sha_at_boot" ] && [ "$drv_sha" != "$(cat "$rr/.driver_sha_at_boot")" ]; then
                echo "$(date +%T) $name: driver changed since engine boot; recycling"
                st="down"
            fi
            break
        done
    }
    [ "$st" = "down" ] || {
        local age_p
        for age_p in $(pgrep -f "bin/sparkpipe_model_residentd"); do
            local rr_age="$HOME/sparkdata/$name"
            [ "$(readlink /proc/$age_p/cwd 2>/dev/null)" = "$rr_age" ] || continue
            local start_s up_s
            start_s=$(awk '{print $22}' "/proc/$age_p/stat" 2>/dev/null)
            up_s=$(awk '{printf "%d", $1}' /proc/uptime)
            [ -n "$start_s" ] && [ $(( up_s - start_s / 100 )) -gt 120 ] && restart_healthy "engine-$name"
        done
        return 0
    }
    local up
    up=$(awk '{printf "%d", $1}' /proc/uptime)
    [ "$up" -ge 900 ] || {
        [ -n "$AGENT_BLOCKED" ] || { echo "$(date +%T) $name: node up ${up}s (<15min); autospawn blocked"; AGENT_BLOCKED=1; }
        return 0
    }
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
    echo "$(date +%T) warmup: fired for engine pid $gen (one cold pass, 900s budget)"
}

while true; do
    sync_core
    install_core
    self_update
    node_doctor
    janitor
    ensure_weightd
    IFS=, read -ra RA <<< "$ROOTS"
    for r in "${RA[@]}"; do sync_root "$r"; done
    for r in "${RA[@]}"; do sync_rendezvous "$r"; done
    for r in "${RA[@]}"; do ensure_root "$r"; done
    for r in "${RA[@]}"; do prune_logs "$HOME/sparkdata/$r"; done
    ensure_api
    warmup_hook
    report_if_changed
    sleep 1
done
