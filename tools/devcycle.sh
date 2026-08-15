#!/usr/bin/env bash
#
# devcycle.sh — fast DSV4 TP4 B1 dev cycle (MacBook <-> sparks)
#
# Purpose: one compact command per step of the fail-fast loop, so an agent
# spends one call + a few output lines instead of re-deriving the ssh/make/
# deploy/benchmark commands every time.
#
#   devcycle status              one line per rank: residentd + driver sha
#   devcycle setup NAME          create /tmp/dsv4-NAME-runtime on 4 ranks
#                                (rank-local pack + configs copied from the
#                                pinned lean control runtime)
#   devcycle deploy NAME DIR     install artifacts from local DIR and restart
#                                residentd on all ranks. DIR must contain:
#                                sparkpipe_model_residentd,
#                                sparkpipe_model_batch, model_driver.so,
#                                model_serving_adapter.so, hidden_transport.so
#   devcycle driver NAME SO      deploy only lib/model_driver.so from SO and
#                                restart (the common spot-test path)
#   devcycle stop NAME           stop residentd on all ranks
#   devcycle start NAME          start residentd on all ranks
#   devcycle ready NAME          block until all four ranks report ready
#   devcycle gate24 NAME         O24 exact-output gate (hash-checked)
#   devcycle run NAME [N]        N O128 runs, one line each + hash check
#   devcycle spot A B [PAIRS]    alternating A/B O128 pairs (A = candidate,
#                                B = control; "lean" is the pinned control).
#                                Candidate driver must be at
#                                tools/devcycle/drivers/A/model_driver.so
#                                Prints a compact VERDICT.
#
# Pinned identities (verified 2026-08-15 against committed receipts):
#   O24 gate hash  : 6f2cfd844a2c296feaf9dd05a04d7888b87c906dcfee947d5cc6de28f541e538
#   O128 gate hash : a9385d0b296ca083e577e715d2f6335067691dce0e0dd5ab1394a102a3d3631f
#   lean driver sha: 3d962820608fbad251aa50b7650dba2ab4b1d19ec378251c0e0ee36922e7fce4
#
# A spot-test ACCEPT requires: every run emits the pinned O128 token stream
# (exact hash) and the candidate mean beats the control mean. Anything else
# prints REJECT. This is the fail-fast gate; full CI/accuracy runs follow
# only for accepted candidates.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RANKS=(spark4 spark5 spark6 spark7)
RANK_INDEX=(0 1 2 3)

BASE_RUNTIME="/tmp/dsv4-integrated-lean-3d962820-runtime"  # pinned lean control
RUNTIME_ROOT_PREFIX="/tmp/dsv4-"

O24_HASH="6f2cfd844a2c296feaf9dd05a04d7888b87c906dcfee947d5cc6de28f541e538"
O128_HASH="a9385d0b296ca083e577e715d2f6335067691dce0e0dd5ab1394a102a3d3631f"
LEAN_DRIVER_SHA="3d962820608fbad251aa50b7650dba2ab4b1d19ec378251c0e0ee36922e7fce4"

BATCH_O24_LOCAL="$SCRIPT_DIR/devcycle/batches/o24_batch.json"
BATCH_O128_LOCAL="$SCRIPT_DIR/devcycle/batches/o128_batch.json"
BATCH_O24_REMOTE="/tmp/devcycle-o24-batch.json"
BATCH_O128_REMOTE="/tmp/devcycle-o128-batch.json"

TEMPLATE_RESIDENT="$SCRIPT_DIR/devcycle/templates/model_resident.template.json"
TEMPLATE_STAGE="$SCRIPT_DIR/devcycle/templates/dsv4_flash_tp4_stage.template.json"

RELEASE_GIT_COMMIT="da7f91090c0d40729352b4e4180ad231971c90a2"
RELEASE_GENERATION="20260815000000"

BUILD_HOST="sparkf"
BUILD_CHECKOUT="/tmp/sparkpipe-devcycle"

# ---------------------------------------------------------------------------
# sync / build — MacBook source -> build spark -> driver artifact
# ---------------------------------------------------------------------------

cmd_sync() {
    rsync -az --delete \
        --exclude '.git' --exclude 'build' --exclude 'docs' --exclude 'qualification' \
        --exclude 'tools/devcycle/drivers' \
        "$REPO_ROOT/" "$BUILD_HOST:$BUILD_CHECKOUT/" || die "rsync source to $BUILD_HOST"
    echo "sync ok -> $BUILD_HOST:$BUILD_CHECKOUT (head=$(git -C "$REPO_ROOT" rev-parse --short HEAD))"
}

cmd_build() {
    local name="$1" bucket="${2:-1}" local_sha
    [[ -n "$name" ]] || die "usage: devcycle build NAME [BUCKET]"
    local_sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    cmd_sync >/dev/null || die "sync failed"
    ssh_rank "$BUILD_HOST" "cd '$BUILD_CHECKOUT' && bash tools/devcycle/build_remote.sh '$name' '$local_sha' '$bucket'" \
        || die "remote build failed"
    mkdir -p "$SCRIPT_DIR/devcycle/drivers/$name"
    for artifact in model_driver.so model_serving_adapter.so hidden_transport.so; do
        scp -q -o BatchMode=yes "$BUILD_HOST:/tmp/devcycle-build-$name/$artifact" \
            "$SCRIPT_DIR/devcycle/drivers/$name/$artifact" || die "fetch $artifact"
    done
    echo "build $name ok driver=$(shasum -a 256 "$SCRIPT_DIR/devcycle/drivers/$name/model_driver.so" | cut -d' ' -f1 | cut -c1-16)"
}

usage() {
    sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
}

die() { echo "devcycle: ERROR: $*" >&2; exit 1; }

runtime_path() {
    if [[ "$1" == "lean" ]]; then
        echo "$BASE_RUNTIME"
    else
        echo "$RUNTIME_ROOT_PREFIX$1-runtime"
    fi
}

ssh_rank() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; }

residentd_pid() {
    # $1 host  $2 rank-index  $3 runtime-root
    ssh_rank "$1" "pgrep -f '^bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $2\$' | head -1"
}

require_files() {
    [[ -f "$BATCH_O24_LOCAL" ]] || die "missing $BATCH_O24_LOCAL"
    [[ -f "$BATCH_O128_LOCAL" ]] || die "missing $BATCH_O128_LOCAL"
    [[ -f "$TEMPLATE_RESIDENT" ]] || die "missing $TEMPLATE_RESIDENT"
    [[ -f "$TEMPLATE_STAGE" ]] || die "missing $TEMPLATE_STAGE"
}

ensure_remote_batches() {
    scp -q -o BatchMode=yes "$BATCH_O24_LOCAL" spark4:"$BATCH_O24_REMOTE" || die "scp o24 batch"
    scp -q -o BatchMode=yes "$BATCH_O128_LOCAL" spark4:"$BATCH_O128_REMOTE" || die "scp o128 batch"
}

token_csv_hash() {
    jq -r '[.events[] | select(.event.event == "token") | .event.token_id] | join(",") + "\n"' "$1" \
        | shasum -a 256 | cut -d' ' -f1
}

decode_rate() { jq -r '.decode_tokens_per_second' "$1"; }

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

cmd_status() {
    local name="lean"
    [[ -n "${1:-}" ]] && name="$1"
    local root host idx pid sha i
    root="$(runtime_path "$name")"
    for i in 0 1 2 3; do
        host="${RANKS[$i]}"
        idx="${RANK_INDEX[$i]}"
        pid="$(residentd_pid "$host" "$idx" "$root" 2>/dev/null)"
        sha="$(ssh_rank "$host" "sha256sum $root/lib/model_driver.so 2>/dev/null | cut -d' ' -f1" 2>/dev/null)"
        printf 'rank%s %s residentd=%s driver=%s\n' "$idx" "$host" "${pid:-stopped}" "${sha:-MISSING}"
    done
}

# ---------------------------------------------------------------------------
# setup NAME — create runtime dir on all ranks (rank-local pack + configs)
# ---------------------------------------------------------------------------

cmd_setup() {
    local name="$1"
    [[ -n "$name" ]] || die "usage: devcycle setup NAME"
    [[ "$name" == "lean" ]] && die "refusing to clobber the pinned lean runtime; pick another name"
    local root i host idx
    root="$(runtime_path "$name")"
    for i in 0 1 2 3; do
        host="${RANKS[$i]}"
        idx="${RANK_INDEX[$i]}"
        ssh_rank "$host" "
            set -e
            test -d '$BASE_RUNTIME/packs' || { echo missing-control-pack; exit 1; }
            rm -rf '$root'
            mkdir -p '$root'/bin '$root'/lib '$root'/config '$root'/kv '$root'/packs
            for f in '$BASE_RUNTIME'/packs/*; do
                ln -s "\$f" '$root'/packs/\$(basename "\$f")
            done
            cp -a '$BASE_RUNTIME'/bin/. '$root'/bin/
            cp -a '$BASE_RUNTIME'/lib/. '$root'/lib/
        " >/dev/null || die "setup dirs on $host"
        echo "setup $host rank$idx dirs+pack-symlink ok"
    done
    local tmp
    tmp="$(mktemp -d)"
    sed "s|$BASE_RUNTIME|$root|g" "$TEMPLATE_RESIDENT" >"$tmp/model_resident.json"
    cp "$TEMPLATE_STAGE" "$tmp/dsv4_flash_tp4_stage.json"
    for host in "${RANKS[@]}"; do
        scp -q -o BatchMode=yes "$tmp/model_resident.json" "$tmp/dsv4_flash_tp4_stage.json" \
            "$host:$root/config/" || die "scp configs to $host"
    done
    rm -rf "$tmp"
    echo "setup configs ok"
}

# ---------------------------------------------------------------------------
# stop / start / ready
# ---------------------------------------------------------------------------

cmd_stop() {
    local name="$1"
    [[ -n "$name" ]] || die "usage: devcycle stop NAME"
    local root i host idx pid
    root="$(runtime_path "$name")"
    for i in 0 1 2 3; do
        host="${RANKS[$i]}"
        idx="${RANK_INDEX[$i]}"
        pid="$(residentd_pid "$host" "$idx" "$root" 2>/dev/null)"
        if [[ -n "$pid" ]]; then
            ssh_rank "$host" "kill $pid" >/dev/null 2>&1 || true
        fi
        echo "stop $host rank$idx ${pid:-none}"
    done
}

cmd_start() {
    local name="$1"
    [[ -n "$name" ]] || die "usage: devcycle start NAME"
    local root i host idx
    root="$(runtime_path "$name")"
    for i in 0 1 2 3; do
        host="${RANKS[$i]}"
        idx="${RANK_INDEX[$i]}"
        if [[ -n "$(residentd_pid "$host" "$idx" "$root" 2>/dev/null)" ]]; then
            echo "start $host rank$idx already-running"
            continue
        fi
        ssh_rank "$host" "
            cd '$root' &&
            export LD_LIBRARY_PATH='$root/lib':\$LD_LIBRARY_PATH \
                   SPARKPIPE_RELEASE_GENERATION='$RELEASE_GENERATION' \
                   SPARKPIPE_RELEASE_GIT_COMMIT='$RELEASE_GIT_COMMIT' \
                   SPARKPIPE_RELEASE_ID='devcycle-$name-rank$idx'
            setsid -f bin/sparkpipe_model_residentd \
                --deployment config/model_resident.json --rank-index '$idx' \
                >/tmp/devcycle-$name-rank$idx.log 2>&1 </dev/null
        " || die "start on $host"
        echo "start $host rank$idx launched"
    done
}

cmd_ready() {
    local name="$1"
    [[ -n "$name" ]] || die "usage: devcycle ready NAME"
    local root i host idx attempts ready_count port
    root="$(runtime_path "$name")"
    port="$(jq -r '.nodes[0].control_endpoint.port' "$TEMPLATE_RESIDENT" 2>/dev/null)"
    [[ -n "$port" && "$port" != "null" ]] || port=18480
    attempts=0
    while (( attempts < 120 )); do
        ready_count=0
        for i in 0 1 2 3; do
            host="${RANKS[$i]}"
            idx="${RANK_INDEX[$i]}"
            if [[ -z "$(residentd_pid "$host" "$idx" "$root" 2>/dev/null)" ]]; then
                echo "ready $host rank$idx DEAD"
                return 1
            fi
            # control listener binds only after adapter init completes
            if ssh_rank "$host" "ss -ltnH | awk '{print \$4}' | grep -q ":$port\$" 2>/dev/null"; then
                ready_count=$((ready_count + 1))
            fi
        done
        (( ready_count == 4 )) && { echo "ready all-4-ranks ok (port $port)"; return 0; }
        sleep 2
        attempts=$((attempts + 1))
    done
    echo "ready TIMEOUT ready=$ready_count/4 (port $port)"
    return 1
}

# ---------------------------------------------------------------------------
# deploy NAME DIR — full artifact install + restart on all ranks
# driver NAME SO — model_driver.so-only swap + restart (spot-test path)
# ---------------------------------------------------------------------------

cmd_deploy() {
    local name="$1" dir="$2"
    [[ -n "$name" && -n "$dir" ]] || die "usage: devcycle deploy NAME ARTIFACT_DIR"
    local artifact
    for artifact in sparkpipe_model_residentd sparkpipe_model_batch \
                    model_driver.so model_serving_adapter.so hidden_transport.so; do
        [[ -f "$dir/$artifact" ]] || die "missing artifact $dir/$artifact"
    done
    local root i host idx
    root="$(runtime_path "$name")"
    for i in 0 1 2 3; do
        host="${RANKS[$i]}"
        idx="${RANK_INDEX[$i]}"
        ssh_rank "$host" "
            pid=\$(pgrep -f '^bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $idx\$' | head -1 || true)
            if [[ -n "\$pid" ]]; then kill "\$pid"; fi
            for k in \$(seq 1 100); do kill -0 "\$pid" 2>/dev/null || break; sleep 0.1; done
        " || die "stop residentd on $host"
        scp -q -o BatchMode=yes "$dir/sparkpipe_model_residentd" "$dir/sparkpipe_model_batch" \
            "$host:$root/bin/" || die "scp bins to $host"
        scp -q -o BatchMode=yes "$dir/model_driver.so" "$dir/model_serving_adapter.so" "$dir/hidden_transport.so" \
            "$host:$root/lib/" || die "scp libs to $host"
        ssh_rank "$host" "
            printf '%s\n' '$RELEASE_GIT_COMMIT' >'$root/SOURCE_COMMIT'
            cd '$root' &&
            export LD_LIBRARY_PATH='$root/lib':\$LD_LIBRARY_PATH \
                   SPARKPIPE_RELEASE_GENERATION='$RELEASE_GENERATION' \
                   SPARKPIPE_RELEASE_GIT_COMMIT='$RELEASE_GIT_COMMIT' \
                   SPARKPIPE_RELEASE_ID='devcycle-$name-rank$idx'
            setsid -f bin/sparkpipe_model_residentd \
                --deployment config/model_resident.json --rank-index '$idx' \
                >/tmp/devcycle-$name-rank$idx.log 2>&1 </dev/null
        " || die "restart residentd on $host"
        echo "deploy $host rank$idx restarted"
    done
    echo "deploy $name done (run: devcycle ready $name)"
}

cmd_driver() {
    local name="$1" so="$2"
    [[ -n "$name" && -n "$so" ]] || die "usage: devcycle driver NAME MODEL_DRIVER_SO"
    [[ -f "$so" ]] || die "no such driver: $so"
    local root i host idx
    root="$(runtime_path "$name")"
    for i in 0 1 2 3; do
        host="${RANKS[$i]}"
        idx="${RANK_INDEX[$i]}"
        # self-heal: seed bins/libs from the pinned control if the runtime is bare
        ssh_rank "$host" "
            if [[ ! -x '$root/bin/sparkpipe_model_residentd' ]]; then
                mkdir -p '$root/bin' '$root/lib'
                cp -a '$BASE_RUNTIME'/bin/. '$root'/bin/
                cp -a '$BASE_RUNTIME'/lib/. '$root'/lib/
            fi
            pid=\$(pgrep -f '^bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $idx\$' | head -1 || true)
            if [[ -n "\$pid" ]]; then kill "\$pid"; fi
            for k in \$(seq 1 100); do kill -0 "\$pid" 2>/dev/null || break; sleep 0.1; done
        " || die "stop residentd on $host"
        scp -q -o BatchMode=yes "$so" "$host:$root/lib/model_driver.so" || die "scp driver to $host"
        # the adapter and transport must travel with the driver: a stale
        # adapter .so from the seeded control runtime rejects the candidate
        local so_dir
        so_dir="$(dirname "$so")"
        [[ -f "$so_dir/model_serving_adapter.so" ]] && \
            scp -q -o BatchMode=yes "$so_dir/model_serving_adapter.so" "$host:$root/lib/model_serving_adapter.so"
        [[ -f "$so_dir/hidden_transport.so" ]] && \
            scp -q -o BatchMode=yes "$so_dir/hidden_transport.so" "$host:$root/lib/hidden_transport.so"
        ssh_rank "$host" "
            cd '$root' &&
            export LD_LIBRARY_PATH='$root/lib':\$LD_LIBRARY_PATH \
                   SPARKPIPE_RELEASE_GENERATION='$RELEASE_GENERATION' \
                   SPARKPIPE_RELEASE_GIT_COMMIT='$RELEASE_GIT_COMMIT' \
                   SPARKPIPE_RELEASE_ID='devcycle-$name-rank$idx'
            setsid -f bin/sparkpipe_model_residentd \
                --deployment config/model_resident.json --rank-index '$idx' \
                >/tmp/devcycle-$name-rank$idx.log 2>&1 </dev/null
        " || die "restart residentd on $host"
        echo "driver $host rank$idx $(shasum -a 256 "$so" | cut -d' ' -f1 | cut -c1-16) restarted"
    done
}

# ---------------------------------------------------------------------------
# gate24 NAME — exact O24 output gate
# ---------------------------------------------------------------------------

cmd_gate24() {
    local name="$1"
    [[ -n "$name" ]] || die "usage: devcycle gate24 NAME"
    local root out hash rate
    root="$(runtime_path "$name")"
    ensure_remote_batches
    out="/tmp/devcycle-$name-o24-$(date +%H%M%S).json"
    python3 "$SCRIPT_DIR/model_stream_decode_benchmark.py" --output "$out" \
        ssh -o BatchMode=yes spark4 "$root/bin/sparkpipe_model_batch" \
            --deployment "$root/config/model_resident.json" \
            --runtime-root "$root" \
            --batch "$BATCH_O24_REMOTE" >/dev/null 2>&1 || die "o24 benchmark failed"
    hash="$(token_csv_hash "$out")"
    rate="$(decode_rate "$out")"
    if [[ "$hash" == "$O24_HASH" ]]; then
        echo "gate24 $name PASS hash=${hash:0:16} decode=$rate tok/s"
        return 0
    fi
    echo "gate24 $name FAIL hash=${hash:0:16} expected=${O24_HASH:0:16} receipt=$out"
    return 1
}

# ---------------------------------------------------------------------------
# run NAME [N] — O128 runs, one line each, hash-checked
# ---------------------------------------------------------------------------

cmd_run() {
    local name="$1" n=1
    [[ -n "$name" ]] || die "usage: devcycle run NAME [N]"
    [[ -n "${2:-}" ]] && n="$2"
    local root out hash rate k ok=0
    root="$(runtime_path "$name")"
    ensure_remote_batches
    # warmup: one discarded run so weights/KV/L2 reach steady state
    python3 "$SCRIPT_DIR/model_stream_decode_benchmark.py" \
        --output "/tmp/devcycle-$name-o128-warm-$(date +%H%M%S).json" \
        ssh -o BatchMode=yes spark4 "$root/bin/sparkpipe_model_batch" \
            --deployment "$root/config/model_resident.json" \
            --runtime-root "$root" \
            --batch "$BATCH_O128_REMOTE" >/dev/null 2>&1 || die "o128 warmup failed"
    for ((k=1; k<=n; k++)); do
        out="/tmp/devcycle-$name-o128-r$k-$(date +%H%M%S).json"
        python3 "$SCRIPT_DIR/model_stream_decode_benchmark.py" --output "$out" \
            ssh -o BatchMode=yes spark4 "$root/bin/sparkpipe_model_batch" \
                --deployment "$root/config/model_resident.json" \
                --runtime-root "$root" \
                --batch "$BATCH_O128_REMOTE" >/dev/null 2>&1 || die "o128 run $k failed"
        hash="$(token_csv_hash "$out")"
        rate="$(decode_rate "$out")"
        if [[ "$hash" == "$O128_HASH" ]]; then
            echo "run $name #$k exact $rate tok/s"
            ok=$((ok + 1))
        else
            echo "run $name #$k WRONG-TOKENS $rate tok/s hash=${hash:0:16}"
        fi
    done
    echo "run $name exact=$ok/$n"
    (( ok == n ))
}

# ---------------------------------------------------------------------------
# spot A B [PAIRS] — alternating pairs; compact VERDICT
#
# A = candidate name; its driver must exist at
#     tools/devcycle/drivers/A/model_driver.so  (place it there after build)
# B = control name; use "lean" for the pinned control runtime.
# ---------------------------------------------------------------------------

cmd_spot() {
    local a="$1" b="$2" pairs=3
    [[ -n "$a" && -n "$b" ]] || die "usage: devcycle spot CANDIDATE CONTROL [PAIRS]"
    [[ -n "${3:-}" ]] && pairs="$3"
    [[ "$a" == "lean" ]] && die "candidate A cannot be 'lean'; use: devcycle spot CANDIDATE lean"
    require_files
    ensure_remote_batches

    local root_a root_b so_a so_b i host idx
    root_a="$(runtime_path "$a")"
    root_b="$(runtime_path "$b")"
    if [[ "$b" == "lean" ]]; then
        root_b="$BASE_RUNTIME"
        so_b="$(mktemp -d)/model_driver.so"
        scp -q -o BatchMode=yes spark4:"$BASE_RUNTIME/lib/model_driver.so" "$so_b" || die "fetch lean driver"
    else
        so_b="$SCRIPT_DIR/devcycle/drivers/$b/model_driver.so"
        [[ -f "$so_b" ]] || die "missing control driver $so_b"
    fi
    so_a="$SCRIPT_DIR/devcycle/drivers/$a/model_driver.so"
    [[ -f "$so_a" ]] || die "missing candidate driver $so_a (build first, then copy to $so_a)"

    # ensure runtime dirs exist
    for i in 0 1 2 3; do
        host="${RANKS[$i]}"
        ssh_rank "$host" "test -f '$root_a/config/model_resident.json'" 2>/dev/null \
            || { cmd_setup "$a" >/dev/null || die "setup $a"; break; }
    done
    ssh_rank spark4 "test -f '$root_b/config/model_resident.json'" 2>/dev/null \
        || { cmd_setup "$b" >/dev/null || die "setup $b"; }

    cmd_stop "$a" >/dev/null 2>&1 || true
    cmd_stop "$b" >/dev/null 2>&1 || true

    local rates_a=() rates_b=() exact_a=0 exact_b=0 out hash rate p warm
    for ((p=1; p<=pairs; p++)); do
        # A: candidate
        cmd_driver "$a" "$so_a" >/dev/null || die "deploy A pair $p"
        cmd_ready "$a" >/dev/null || die "ready A pair $p"
        warm="/tmp/devcycle-spot-warm-$(date +%H%M%S%N).json"
        python3 "$SCRIPT_DIR/model_stream_decode_benchmark.py" --output "$warm" \
            ssh -o BatchMode=yes spark4 "$root_a/bin/sparkpipe_model_batch" \
                --deployment "$root_a/config/model_resident.json" \
                --runtime-root "$root_a" \
                --batch "$BATCH_O128_REMOTE" >/dev/null 2>&1 || die "A warm pair $p failed"
        out="/tmp/devcycle-spot-$a-p$p.json"
        python3 "$SCRIPT_DIR/model_stream_decode_benchmark.py" --output "$out" \
            ssh -o BatchMode=yes spark4 "$root_a/bin/sparkpipe_model_batch" \
                --deployment "$root_a/config/model_resident.json" \
                --runtime-root "$root_a" \
                --batch "$BATCH_O128_REMOTE" >/dev/null 2>&1 || die "A pair $p failed"
        hash="$(token_csv_hash "$out")"
        rate="$(decode_rate "$out")"
        [[ "$hash" == "$O128_HASH" ]] && exact_a=$((exact_a + 1))
        rates_a+=("$rate")
        echo "spot p$p $a $rate tok/s exact=$([[ "$hash" == "$O128_HASH" ]] && echo yes || echo NO)"

        # B: control
        if [[ "$b" == "lean" ]]; then
            cmd_start "$b" >/dev/null 2>&1 || true   # no-op: lean root is BASE_RUNTIME
        else
            cmd_driver "$b" "$so_b" >/dev/null || die "deploy B pair $p"
        fi
        cmd_ready "$b" >/dev/null || die "ready B pair $p"
        warm="/tmp/devcycle-spot-warm-$(date +%H%M%S%N).json"
        python3 "$SCRIPT_DIR/model_stream_decode_benchmark.py" --output "$warm" \
            ssh -o BatchMode=yes spark4 "$root_b/bin/sparkpipe_model_batch" \
                --deployment "$root_b/config/model_resident.json" \
                --runtime-root "$root_b" \
                --batch "$BATCH_O128_REMOTE" >/dev/null 2>&1 || die "B warm pair $p failed"
        out="/tmp/devcycle-spot-$b-p$p.json"
        python3 "$SCRIPT_DIR/model_stream_decode_benchmark.py" --output "$out" \
            ssh -o BatchMode=yes spark4 "$root_b/bin/sparkpipe_model_batch" \
                --deployment "$root_b/config/model_resident.json" \
                --runtime-root "$root_b" \
                --batch "$BATCH_O128_REMOTE" >/dev/null 2>&1 || die "B pair $p failed"
        hash="$(token_csv_hash "$out")"
        rate="$(decode_rate "$out")"
        [[ "$hash" == "$O128_HASH" ]] && exact_b=$((exact_b + 1))
        rates_b+=("$rate")
        echo "spot p$p $b $rate tok/s exact=$([[ "$hash" == "$O128_HASH" ]] && echo yes || echo NO)"
    done

    local mean_a mean_b delta
    mean_a="$(printf '%s\n' "${rates_a[@]}" | awk '{s+=$1} END {printf "%.4f", s/NR}')"
    mean_b="$(printf '%s\n' "${rates_b[@]}" | awk '{s+=$1} END {printf "%.4f", s/NR}')"
    delta="$(awk -v a="$mean_a" -v b="$mean_b" 'BEGIN {printf "%+.4f", (a-b)/b*100}')"
    echo "VERDICT $a=$mean_a $b=$mean_b delta=$delta% exact $a=$exact_a/$pairs $b=$exact_b/$pairs"
    if (( exact_a == pairs && exact_b == pairs )) && \
       awk -v a="$mean_a" -v b="$mean_b" 'BEGIN {exit !(a > b)}'; then
        echo "VERDICT ACCEPT $a"
        return 0
    fi
    echo "VERDICT REJECT $a"
    return 1
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

require_files
case "${1:-}" in
    status)  cmd_status "${2:-lean}" ;;
    setup)   cmd_setup "$2" ;;
    stop)    cmd_stop "$2" ;;
    start)   cmd_start "$2" ;;
    ready)   cmd_ready "$2" ;;
    sync)    cmd_sync ;;
    build)   cmd_build "$2" "${3:-}" ;;
    deploy)  cmd_deploy "$2" "$3" ;;
    driver)  cmd_driver "$2" "$3" ;;
    gate24)  cmd_gate24 "$2" ;;
    run)     cmd_run "$2" "${3:-1}" ;;
    spot)    cmd_spot "$2" "$3" "${4:-3}" ;;
    *)       usage ;;
esac
