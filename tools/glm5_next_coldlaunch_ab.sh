#!/bin/bash
# glm5_next cold-launch preload A/B (lane 0, milestone 3).
#
# Measures time-to-launch (residentd start -> "model_residentd ready") and
# TTFT (first token of the qualified smoke request) for the glm53flash.fp8.tp16
# TP16 resident through the SHARED weightd socket, in two arms submitted as
# two sequential queue jobs under identical lane budgets/ports:
#
#   ARM=A  cold launch, no preload (submit FIRST: the shared daemon must not
#          already hold the pack warm for this arm to measure the cold path)
#   ARM=B  preload exactly the smoke expert set through the shared daemon
#          (weightd_warm --wset model-families/glm5_next/glm53flash.fp8.tp16.smoke.wset,
#          the 336-key set recorded in model-families/glm5_next/smoke_experts.json),
#          then the same launch + request
#
# Mode claim: the resident runs the GPU-qualified configuration
# (SPARK_GLM5_NEXT_GRAPH_PATH=1 + SPARK_GLM5_NEXT_PIN_EXPERTS=1, full pool
# pinning at graph capture). Numbers are smoke-relative stdout timings under
# shared-lane conditions, not isolated-fleet receipts.
#
# Runs inside an admitted queue job (gpu-shared, spark0..sparkf, --per-node):
#   sync lane branch -> add ... --cmd 'ARM=A bash tools/glm5_next_coldlaunch_ab.sh'
#   (then a second job with ARM=B)
#
# The measurement uses the verified PR #1082 stack already deployed on each
# node's NVMe runtime root (/home/<host>/sparkdata/glm53flash.fp8.tp16/bin);
# binary sha256s are recorded in the receipt. Numerical validation of tokens
# remains the qualified harness (tools/inference_smoke.py).

set -euo pipefail

ARM="${ARM:-A}"
HOST="$(hostname)"
EXEC_ROOT="${SPARK_EXEC_ROOT:?set SPARK_EXEC_ROOT to one verified release directory}"
(cd "$EXEC_ROOT" && sha256sum --quiet --strict --check SHA256SUMS)
FAMILY_ROOT="${SPARK_FAMILY_ROOT:-/home/$HOST/sparkdata/glm53flash.fp8.tp16}"
SHARED_SOCKET="${SPARK_WEIGHTD_SOCKET:-/run/sparkpipe-weightd-shared/weightd.sock}"
LANE=0
CHECKOUT="$(pwd)"                      # synced lane checkout (repo-owned command)
WSET="$CHECKOUT/model-families/glm5_next/glm53flash.fp8.tp16.smoke.wset"
BATCH="$CHECKOUT/model-families/glm5_next/glm53flash.smoke.batch.json"
DEPLOYMENT_SOURCE="$CHECKOUT/deployment/glm5_next_tp16/model_resident.json"

CONTROL_BASE=$((23000 + 16 * LANE))
COLLECTIVE_BASE=$((53000 + 16 * LANE))
TRANSPORT_BASE=$((64000 + 16 * LANE))
SESSION_BASE=$((23168 + 64 * LANE))

ATTEMPT="${SPARK_QUEUE_ATTEMPT:?run through the authoritative spark queue}"
ROOT="${SPARK_QUEUE_RUNTIME_ROOT:?}"
mkdir -p "$ROOT"                 # the queue hands us the path, not the directory
RANK="${SPARK_QUEUE_RANK:?}"; SIZE="${SPARK_QUEUE_SIZE:?}"
[ "$SIZE" -eq 16 ] || { echo "job expects the full TP16 fleet (got SIZE=$SIZE)" >&2; exit 2; }
[ -S "$SHARED_SOCKET" ] || { echo "shared weightd socket missing: $SHARED_SOCKET" >&2; exit 2; }
[ -f "$WSET" ] || { echo "wset missing in checkout: $WSET" >&2; exit 2; }
[ -f "$BATCH" ] || { echo "batch missing in checkout: $BATCH" >&2; exit 2; }

reserved_ok() {
  local port="$1" entry first last
  for entry in ${SPARK_QUEUE_PORTS//,/ }; do
    first="${entry%%:*}"; last="${entry##*:}"
    if [ "$port" -ge "$first" ] && [ "$port" -le "$last" ]; then return 0; fi
  done
  echo "listener port $port not reserved" >&2; return 1
}
for r in $(seq 0 15); do
  reserved_ok $((CONTROL_BASE + r)) || exit 2
  reserved_ok $((COLLECTIVE_BASE + r)) || exit 2
done
reserved_ok "$TRANSPORT_BASE" || exit 2

census() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
elapsed() { python3 -c "import sys; print(f'{float(sys.argv[1])-float(sys.argv[2]):.3f}')" "$1" "$2"; }

# receipt: props accumulate as KEY=VALUE lines; finalize merges into JSON
PROPS="$ROOT/receipt.props"
: > "$PROPS"
receipt_set() { printf '%s=%s\n' "$1" "$2" >> "$PROPS"; }
receipt_finalize() {
python3 - "$ROOT" "$PROPS" <<'PYEOF'
import json, sys
from pathlib import Path
root, props_path = Path(sys.argv[1]), Path(sys.argv[2])
value = {"schema_version": 1, "arm": None, "rank": None, "attempt": None, "status": "FAIL"}
for line in props_path.read_text().splitlines():
    if "=" not in line:
        continue
    key, item = line.split("=", 1)
    if item.replace(".", "", 1).isdigit():
        try:
            item = int(item) if item.isdigit() else float(item)
        except ValueError:
            pass
    value[key] = item
(root / "coldlaunch.json").write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
print(json.dumps(value))
PYEOF
}

receipt_set arm "$ARM"
receipt_set rank "$RANK"
receipt_set attempt "$ATTEMPT"
# smoke-set byte bases per node (from the .experts spans; lane-5 chunk finding):
# raw 519.8 MiB; 2 MiB ceil-per-span = 1,344 spans = 2,688 MiB (5.17x raw)
receipt_set smoke_set_raw_bytes_per_node 544997376
receipt_set smoke_set_chunked_bytes_per_node 2818572288
receipt_set chunk_basis "2 MiB ceil per expert span; GLM per-rank spans are all sub-2MiB (5.17x raw)"
receipt_set residentd_sha256 "$(sha_of "$EXEC_ROOT/bin/sparkpipe_model_residentd")"
receipt_set weightd_warm_sha256 "$(sha_of "$EXEC_ROOT/bin/weightd_warm")"
receipt_set model_batch_sha256 "$(sha_of "$EXEC_ROOT/bin/sparkpipe_model_batch")"
receipt_set exec_bundle_source_commit "$(cat "$EXEC_ROOT/SOURCE_COMMIT")"
receipt_set wset_sha256 "$(sha_of "$WSET")"
receipt_set daemon_census_before "$(census)"

# ------------------------- private deployment prep -------------------------
# Template-derived (tools/devcycle/templates/run-family-job.sh.template) with
# the qualified smoke runtime limits (B1, context 512, 128 pages, 2 GiB cap).
python3 - "$ROOT" "$RANK" "$CONTROL_BASE" "$COLLECTIVE_BASE" "$TRANSPORT_BASE" \
         "$SESSION_BASE" "$ATTEMPT" "$LANE" "$DEPLOYMENT_SOURCE" "$FAMILY_ROOT" "$EXEC_ROOT" <<'PYEOF'
import json, sys
from pathlib import Path
root, rank = Path(sys.argv[1]), int(sys.argv[2])
control_base, collective_base, transport_base, session_base = (int(a) for a in sys.argv[3:7])
attempt, lane, deployment_source, family_root, exec_root = sys.argv[7], int(sys.argv[8]), sys.argv[9], sys.argv[10], sys.argv[11]

def fail(msg):
    raise SystemExit("coldlaunch prep: FAIL: " + msg)

host_names = [f"spark{index:x}" for index in range(16)]
deployment = json.loads(Path(deployment_source).read_text())
if len(deployment["nodes"]) != 16:
    fail("deployment is not TP16")
value = json.loads(json.dumps(deployment))
value["transport"]["control_port_base"] = transport_base
value["weightd"] = {"socket_path": "/run/sparkpipe-weightd-shared/weightd.sock"}
value["runtime_limits"] = {
    "max_inflight_submissions": 1, "max_active_sequences": 1, "max_input_rows": 1,
    "resident_sequence_capacity": 1, "kv_logical_page_capacity": 128,
    "kv_physical_page_capacity": 128}
for r, node in enumerate(value["nodes"]):
    if node["rank_index"] != r:
        fail("deployment ranks unordered")
    node["runtime_root"] = str(root / "runtime")
    node["transport_host"] = host_names[r]
    node["control_endpoint"] = {"kind": "tcp", "host": host_names[r], "port": control_base + r}
    node["kv_backing_directory"] = str(root / "kv")
    node["kv_backing_maximum_bytes"] = 2 * 1024 * 1024 * 1024

config_name = value["nodes"][rank]["adapter_configuration_path"]
config = json.loads((Path(family_root) / config_name).read_text())

PORT_NAMES = {"listen_port", "peer_ports", "session_ports", "session_ports_hc", "draft_bridge_port"}
def remap(item):
    if isinstance(item, list):
        return [remap(entry) for entry in item]
    if not isinstance(item, int):
        fail("non-integer listener")
    return 0 if item == 0 else session_base + (item % 4096)
def rewrite(node):
    out = {}
    for key, item in node.items():
        if key.endswith("_port") or key.endswith("_ports") or key.startswith("session_ports"):
            if key not in PORT_NAMES:
                fail(f"unhandled listener key {key}")
            out[key] = remap(item)
        elif isinstance(item, dict):
            out[key] = rewrite(item)
        else:
            out[key] = item
    return out
config = rewrite(config)
tp = config["tp_collective"]
tp["collective_identifier"] = int(attempt[:15], 16) + 1 + lane
tp["listen_port"] = collective_base + rank
tp["peer_ports"] = [collective_base + peer for peer in range(16)]
tp["peer_hosts"] = host_names
if "rail_peer_hosts" in tp:
    tp["rail_peer_hosts"] = [host_names for _ in tp["rail_peer_hosts"]]

runtime = root / "runtime"
runtime.mkdir(parents=True, exist_ok=True)
pack = config["stage_pack_path"]
if Path(pack).is_absolute() or ".." in Path(pack).parts:
    fail("pack escapes runtime root")
source_pack = Path(family_root) / pack
if not source_pack.is_file():
    fail(f"pack missing: {source_pack}")
packs = runtime / "packs"
packs.mkdir()
packs.joinpath(source_pack.name).symlink_to(source_pack.resolve())
for suffix in (".sha256", ".experts", ".ck128"):
    sidecar = Path(str(source_pack) + suffix)
    if not sidecar.is_file():
        if suffix == ".ck128":
            fail("missing .ck128 whole-pack sidecar - run tools/glm5_next_ck128_stamp.sh once per node (release shared weightd requirement)")
        fail(f"missing sidecar {suffix}")
    packs.joinpath(sidecar.name).write_bytes(sidecar.read_bytes())
if len(list(packs.glob("*.sha256"))) != 1:
    fail("packs/ must hold exactly one digest")
# symlink every binary the deployment loads (adapter, driver, transport and
# the collective backend module) from the qualified NVMe runtime root; the
# resident resolves these RELATIVE to runtime_root (a2 r2 postmortem:
# adapter_load not_found with a packs/-only runtime)
assets = {deployment["adapter"]["shared_object_path"], deployment["driver"]["shared_object_path"],
          deployment["transport"]["shared_object_path"]}
backend = config.get("tp_collective", {}).get("backend_module_path")
if backend:
    assets.add(backend)
for relative in sorted(assets):
    source = Path(exec_root) / relative
    if not source.is_file():
        fail(f"runtime asset missing in the selected release: {relative}")
    target = runtime / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())
(runtime / config_name).parent.mkdir(parents=True, exist_ok=True)
(runtime / config_name).write_text(json.dumps(config) + "\n")
(root / "kv").mkdir(exist_ok=True)
(root / "deployment.json").write_text(json.dumps(value) + "\n")
print("coldlaunch prep: rank", rank, "pack", source_pack.name)
PYEOF

config_rel="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["nodes"][int(sys.argv[2])]["adapter_configuration_path"])' "$ROOT/deployment.json" "$RANK")"
pack_rel="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage_pack_path"])' "$ROOT/runtime/$config_rel")"
revision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model_revision","0"))' "$ROOT/runtime/$config_rel")"

# the module and the warmer both require explicit finite pool/spine budgets
# (qualified PR #1082 values: 24 GiB pool / 4 GiB spine; the shared daemon
# enforces its own arena bounds on top)
export SPARK_WEIGHTD_EXPERT_POOL_BYTES=25769803776
export SPARK_WEIGHTD_SPINE_BUDGET_BYTES=4294967296
export SPARK_WEIGHTD_KV_RESERVE_BYTES=0

# ------------------------------ arm B preload -------------------------------
if [ "$ARM" = "B" ]; then
  PACK="$ROOT/runtime/$pack_rel"
  SHA="$(cut -d' ' -f1 "$PACK.sha256")"
  t0=$(date +%s.%N)
  "$EXEC_ROOT/bin/weightd_warm" "$SHARED_SOCKET" "$PACK" "$SHA" "$revision" 16 \
    --wset "$WSET" 300 >"$ROOT/warm.log" 2>&1
  t1=$(date +%s.%N)
  grep -q "WSET-WARM keys=" "$ROOT/warm.log" || { echo "wset warm failed"; cat "$ROOT/warm.log" >&2; exit 2; }
  receipt_set warm_seconds "$(elapsed "$t1" "$t0")"
  receipt_set warm_log "$(grep -E 'WSET-WARM|WSET-ONE-SHOT' "$ROOT/warm.log" | tr '\n' ';')"
  receipt_set daemon_cold_warm_seconds "$(elapsed "$t1" "$t0")"
fi

# ------------------------------ resident launch -----------------------------
export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_SOCKET="$SHARED_SOCKET"
export SPARK_WEIGHTD_LANE="$LANE"
export SPARK_TP_MESH_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export SPARK_TP_WAIT_MODE=hardware
export CUDA_MODULE_LOADING=LAZY
export CUDA_MODULE_DATA_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=32
export SPARK_GLM5_NEXT_GRAPH_PATH=1
export SPARK_GLM5_NEXT_PIN_EXPERTS=1

t0=$(date +%s.%N)
setsid "$EXEC_ROOT/bin/sparkpipe_model_residentd" \
  --deployment "$ROOT/deployment.json" --rank-index "$RANK" \
  >"$ROOT/residentd.log" 2>&1 &
RESIDENTD_PID=$!
ready=0
for _ in $(seq 1 2400); do
  if grep -q "model_residentd ready rank=$RANK " "$ROOT/residentd.log" 2>/dev/null; then ready=1; break; fi
  kill -0 "$RESIDENTD_PID" 2>/dev/null || break
  sleep 0.25
done
t1=$(date +%s.%N)
if [ "$ready" -ne 1 ]; then
  receipt_set error "residentd not ready"
  receipt_finalize >/dev/null
  tail -20 "$ROOT/residentd.log" >&2 || true
  kill -TERM "$RESIDENTD_PID" 2>/dev/null || true
  exit 2
fi
receipt_set time_to_ready_seconds "$(elapsed "$t1" "$t0")"
receipt_set instance_ready_seconds "$(elapsed "$t1" "$t0")"
receipt_set ready_line "$(grep -m1 "model_residentd ready rank=$RANK " "$ROOT/residentd.log")"

# cross-node readiness barrier + coordinator request
printf '{"attempt":"%s","rank":%s,"arm":"%s"}\n' "$ATTEMPT" "$RANK" "$ARM" > "$ROOT/ready.json"
if [ "$RANK" -eq 0 ]; then
  ok=0
  for _ in $(seq 1 480); do
    ok=1
    for r in $(seq 1 15); do
      h=$(printf "spark%x" "$r")
      ssh -o BatchMode=yes -o ConnectTimeout=3 "$h" "cat /tmp/sparkqueue-$ATTEMPT/ready.json" 2>/dev/null | grep -q "\"attempt\":\"$ATTEMPT\"" || { ok=0; break; }
    done
    [ "$ok" -eq 1 ] && break
    sleep 1
  done
  if [ "$ok" -ne 1 ]; then
    receipt_set error "peer barrier timeout"
  else
    t2=$(date +%s.%N)
    python3 "$CHECKOUT/tools/glm5_next_bench_wrap.py" --timeout 120 -- \
      "$EXEC_ROOT/bin/sparkpipe_model_batch" --deployment "$ROOT/deployment.json" \
      --runtime-root "$ROOT/runtime" --batch "$BATCH" >"$ROOT/bench.json" 2>"$ROOT/bench.stderr" || true
    t3=$(date +%s.%N)
    receipt_set request_seconds "$(elapsed "$t3" "$t2")"
    python3 - "$ROOT" "$PROPS" <<'PYEOF'
import json, sys
from pathlib import Path
root, props_path = Path(sys.argv[1]), Path(sys.argv[2])
value = {"schema_version": 1}
for line in props_path.read_text().splitlines():
    if "=" in line:
        key, item = line.split("=", 1)
        if item.replace(".", "", 1).isdigit():
            try:
                item = int(item) if item.isdigit() else float(item)
            except ValueError:
                pass
        value[key] = item
try:
    bench = json.loads((root / "bench.json").read_text())
except (OSError, ValueError):
    bench = {"valid": False, "errors": ["bench output unreadable"]}
value["bench"] = bench
value["steady_state_decode_tokens_per_second"] = bench.get("decode_tokens_per_second")
value["ttft_seconds"] = bench.get("ttft_seconds")
value["status"] = "PASS" if bench.get("valid") else "FAILED"
(root / "coldlaunch.json").write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
print(json.dumps({"ttft_seconds": bench.get("ttft_seconds"), "tokens": bench.get("token_count"),
                  "decode_tokens_per_second": bench.get("decode_tokens_per_second"), "valid": bench.get("valid")}))
PYEOF
    printf '{"attempt":"%s"}\n' "$ATTEMPT" > "$ROOT/inference-done.json"
  fi
else
  done_seen=0
  for _ in $(seq 1 600); do
    if ssh -o BatchMode=yes -o ConnectTimeout=3 spark0 "cat /tmp/sparkqueue-$ATTEMPT/inference-done.json" 2>/dev/null | grep -q attempt; then done_seen=1; break; fi
    sleep 1
  done
  [ "$done_seen" -eq 1 ] || receipt_set error "coordinator completion not observed"
  receipt_set status PASS
fi

# clean teardown of OUR resident only (this job's process group)
kill -TERM "$RESIDENTD_PID" 2>/dev/null || true
for _ in $(seq 1 120); do kill -0 "$RESIDENTD_PID" 2>/dev/null || break; sleep 0.5; done
if kill -0 "$RESIDENTD_PID" 2>/dev/null; then
  kill -KILL "$RESIDENTD_PID" 2>/dev/null || true
  receipt_set teardown killed
else
  receipt_set teardown term
fi
receipt_set daemon_census_after "$(census)"
receipt_finalize
exit 0
