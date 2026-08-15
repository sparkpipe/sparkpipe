#!/usr/bin/env bash
#
# bbench.sh B — no-spec B-batch decode benchmark for the lean DSV4 TP4 driver.
#
# Expects the B-variant driver at tools/devcycle/drivers/lean-bB/model_driver.so
# (built with: devcycle build lean-bB B). Deploys a B-width runtime on
# spark4-7, runs one discarded warmup + one measured O128 run with B requests,
# and prints the aggregate decode rate.
#
# usage: tools/devcycle/bbench.sh B
set -euo pipefail

B="${1:?usage: bbench.sh B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RANKS=(spark4 spark5 spark6 spark7)
BASE_RUNTIME="/tmp/dsv4-integrated-lean-3d962820-runtime"
ROOT="/tmp/dsv4-b${B}-runtime"
DRIVER="$SCRIPT_DIR/drivers/lean-b${B}/model_driver.so"
ADAPTER="$SCRIPT_DIR/drivers/lean-b${B}/model_serving_adapter.so"
BATCH_LOCAL="/tmp/bbench-b${B}-batch.json"
TEMPLATE="$SCRIPT_DIR/templates/model_resident.template.json"
STAGE_TEMPLATE="$SCRIPT_DIR/templates/dsv4_flash_tp4_stage.template.json"

[[ -f "$DRIVER" ]] || { echo "missing driver $DRIVER (build first: devcycle build lean-b$B $B)" >&2; exit 2; }
[[ -f "$TEMPLATE" ]] || { echo "missing template" >&2; exit 2; }

# ---- generate the B-request batch file from the pinned O128 batch ----
python3 - "$SCRIPT_DIR/batches/o128_batch.json" "$BATCH_LOCAL" "$B" <<'PYEOF'
import json, sys
src, dst, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
b = json.load(open(src))
req = b["requests"][0]
b["request_capacity"] = B
b["max_prefill_rows_per_submission"] = B
b["requests"] = []
for i in range(B):
    r = json.loads(json.dumps(req))
    r["request_id"] = 900000 + i
    r["sequence_id"] = 900000 + i
    b["requests"].append(r)
json.dump(b, open(dst, "w"), indent=2)
print("batch", B, "requests ->", dst)
PYEOF

# ---- per-rank runtime: bins/libs seeded, pack symlinked, B-width config ----
KV_PAGES=$(( B * 2 < 32 ? 32 : B * 2 ))
sed "s|$BASE_RUNTIME|$ROOT|g" "$TEMPLATE" | \
    jq --argjson b "$B" --argjson kv "$KV_PAGES" \
    '.runtime_limits.max_active_sequences = $b | .runtime_limits.max_input_rows = $b | .runtime_limits.resident_sequence_capacity = $b | .runtime_limits.kv_logical_page_capacity = $kv | .runtime_limits.kv_physical_page_capacity = $kv' \
    > /tmp/bbench-mr.json

i=0
for h in "${RANKS[@]}"; do
    ssh -o BatchMode=yes "$h" "
        set -e
        rm -rf '$ROOT'
        mkdir -p '$ROOT'/bin '$ROOT'/lib '$ROOT'/config '$ROOT'/kv '$ROOT'/packs
        cp -a '$BASE_RUNTIME'/bin/. '$ROOT'/bin/
        cp -a '$BASE_RUNTIME'/lib/. '$ROOT'/lib/
        for f in '$BASE_RUNTIME'/packs/*; do ln -s "\$f" '$ROOT'/packs/\$(basename "\$f"); done
    " || { echo "setup $h failed" >&2; exit 3; }
    scp -q -o BatchMode=yes /tmp/bbench-mr.json "$h:$ROOT/config/model_resident.json"
    scp -q -o BatchMode=yes "$STAGE_TEMPLATE" "$h:$ROOT/config/dsv4_flash_tp4_stage.json"
    scp -q -o BatchMode=yes "$DRIVER" "$h:$ROOT/lib/model_driver.so"
    [[ -f "$ADAPTER" ]] && scp -q -o BatchMode=yes "$ADAPTER" "$h:$ROOT/lib/model_serving_adapter.so"
    ssh -o BatchMode=yes "$h" "
        cd '$ROOT' &&
        export LD_LIBRARY_PATH='$ROOT/lib':\$LD_LIBRARY_PATH \
               SPARKPIPE_RELEASE_GENERATION=20260816000000 \
               SPARKPIPE_RELEASE_GIT_COMMIT=lean-bench \
               SPARKPIPE_RELEASE_ID='bbench-b$B'
        setsid -f bin/sparkpipe_model_residentd \
            --deployment config/model_resident.json --rank-index '$i' \
            >/tmp/bbench-b$B-rank$i.log 2>&1 </dev/null
    " || { echo "start $h failed" >&2; exit 4; }
    i=$((i + 1))
    sleep 6
done

# ---- wait for readiness on the control port ----
for attempt in $(seq 1 60); do
    ready=0
    for h in "${RANKS[@]}"; do
        if ssh -o BatchMode=yes "$h" "ss -ltnH | grep -q ':18480 '"; then
            ready=$((ready + 1))
        fi
    done
    [[ "$ready" == 4 ]] && break
    sleep 3
done
[[ "$ready" == 4 ]] || { echo "ready timeout ($ready/4)" >&2; exit 5; }
echo "b$B all-4-ranks ready"

scp -q -o BatchMode=yes "$BATCH_LOCAL" spark4:/tmp/bbench-batch.json
sleep 45

# ---- warm + measured ----
python3 "$REPO_ROOT/tools/model_stream_decode_benchmark.py" \
    --output "/tmp/bbench-b$B-warm.json" \
    ssh -o BatchMode=yes spark4 "$ROOT/bin/sparkpipe_model_batch" \
        --deployment "$ROOT/config/model_resident.json" \
        --runtime-root "$ROOT" \
        --batch /tmp/bbench-batch.json >/dev/null 2>&1 || { echo "warmup failed" >&2; exit 6; }

python3 "$REPO_ROOT/tools/model_stream_decode_benchmark.py" \
    --output "/tmp/bbench-b$B-run.json" \
    ssh -o BatchMode=yes spark4 "$ROOT/bin/sparkpipe_model_batch" \
        --deployment "$ROOT/config/model_resident.json" \
        --runtime-root "$ROOT" \
        --batch /tmp/bbench-batch.json >/dev/null 2>&1 || { echo "run failed" >&2; exit 7; }

python3 - "/tmp/bbench-b$B-run.json" "$B" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
B = int(sys.argv[2])
print(f"bbench B{B}: decode={r['decode_tokens_per_second']:.2f} tok/s "
      f"tokens={r['token_count']} decode_s={r['decode_seconds_after_first']:.3f} "
      f"ttft={r['ttft_seconds']:.3f}")
PYEOF
