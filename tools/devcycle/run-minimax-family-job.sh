#!/bin/bash
# minimax text-tower TP4 shared-socket family wrapper (multidev lane 10,
# spark8-sparkb).
#
# Ported from tools/devcycle/templates/run-family-job.sh.template (the
# extracted VALIDATED GLM5.3-flash wrapper pattern, PR #1082/#1084), with
# the queue-invocation discipline of tools/devcycle/run-dsv4-pro-family-job.sh
# (submit with --cmd 'bash tools/devcycle/run-minimax-family-job.sh'; the
# dispatcher routes command text through systemd-run, which pre-expands
# $-syntax, so this file keeps its variables inside real bash).
#
# Provenance: the family module is the adopted PR #1080 text-tower driver
# (modules/minimax_resident_decode_stage); the behavioral contract is pinned
# by the template, docs/MULTIDEV_QUICKSTART.md and docs/PARALLEL_DRIVER_DEBUG.md.
#
# MINIMAX ADAPTATIONS (everything beyond FAMILY PARAMETERS is marked):
#   1. DENSE INTERIM PATH (manager ruling 2026-09-23, dense-attach issue
#      #1138): the text tower has no routed experts, so it cannot ride the
#      weightd expert arena - the module mmaps the node-local pack and
#      uploads resident device weights (~17.5 GiB/rank honest declaration at
#      submission; the build-job exception class, bounded TTL). The wrapper
#      still runs shared-socket mode (socket must be present) but exports no
#      preload hook and requires no .experts sidecar (dense packs have none).
#   2. The qwen38-lineage serving adapter takes a five-key stage config
#      WITHOUT a tp_collective section: the module's TP collective is driven
#      by SPARK_MINIMAX_STAGE_TP_* environment this wrapper exports (rail
#      hosts of the spark8-b quartet, attempt-derived identifier, transport
#      block as the control port base, session grid from the lane session
#      block).
#   3. Runtime lib/stages artifacts resolve against the verified firmware
#      root (tools/module_build_release.sh output: SOURCE_COMMIT + SHA256SUMS
#      + bin/ + lib/ + stages/), the qwen38_27b lane pattern.
#   4. IN-JOB DECODE CELL (the k3 M3 cell precedent): rank 0 drives the
#      release-built sparkpipe_model_batch from the same firmware root with
#      the canonical T1 fixture (qualification/t1_reference/minimax/
#      prompts.json) at B1 and prints DECODE-BATCH receipts, then tears down
#      and exits; peer ranks drain on the coordinator control-listener
#      transition and exit 0. The whole serve+decode cell lives inside one
#      queue job and its TTL (the --after dependency sequences job
#      COMPLETION, not readiness, so a cross-job handshake cannot express
#      "decode starts when serve is up").
#
# Laws inherited from the template (each paid for with an incident):
#   - Fail closed: missing/malformed pack identity (.sha256), unreserved
#     listener port, missing shared socket, unbounded KV backing.
#   - The wrapper never starts daemons in shared-socket mode.
#   - No nested SSH GPU launches; every child stays in this queue cgroup.
#   - Logical rank = index in --nodes = rank_index in the deployment.

set -euo pipefail

# ----------------------------- FAMILY PARAMETERS -----------------------------

FAMILY="minimax"
LANE=10
TOPOLOGY="TP4"
HOSTS=(spark8 spark9 sparka sparkb)
MESH_RANKS="8,9,10,11"
EXEC_PREFIX="${MINIMAX_FIRMWARE_ROOT:-}"

# Lane port math (lane_assignments.json, PR #1094 amendment): lane L owns
#   control 23000+16L..+15, collective 53000+16L..+15, transport 64000+16L..+15
CONTROL_BASE=$((23000 + 16 * LANE))       # 23160..23175
COLLECTIVE_BASE=$((53000 + 16 * LANE))    # 53160..53175 (reserved; the module
                                          # TP sessions use the session block)
TRANSPORT_BASE=$((64000 + 16 * LANE))     # 64160..64175
SESSION_BASE=$((23168 + 64 * LANE))       # 23808..23871 (TP session grid)

# Fabric rail 0 of the spark8-b quartet (host-rdma transport, rank order).
RAIL_HOSTS="${MINIMAX_RAIL_HOSTS:-10.10.200.8,10.10.200.9,10.10.200.10,10.10.200.11}"

DEVICE_MIB=18905   # dense interim (lane_budget_calc on smoke_experts.json):
                    # declared honestly at submission (NOT the 6400 lane table
                    # serving budget; bounded-TTL exception per manager ruling)

WEIGHTD_MODE="${MINIMAX_WEIGHTD_MODE:-shared-socket}"
SHARED_SOCKET="${MINIMAX_WEIGHTD_SOCKET:-/run/sparkpipe-weightd-shared/weightd.sock}"

# Family model environment: the pinned shared-lane CUDA trio. The minimax
# module has no whole-chain graph flags - never invent new ones.
FAMILY_ENV=(
  "CUDA_MODULE_LOADING=LAZY"
  "CUDA_MODULE_DATA_LOADING=LAZY"
  "CUDA_DEVICE_MAX_CONNECTIONS=32"
)

# MINIMAX ADAPTATION 1: dense family, no routed experts, no .wset working set
# until #1138 lands dense-attach. The preload hook stays empty.
WORKING_SET=""

DEPLOYMENT_SOURCE="${MINIMAX_DEPLOYMENT_SOURCE:-}"

# ------------------------------ QUEUE CONTRACT -------------------------------

ATTEMPT="${SPARK_QUEUE_ATTEMPT:?run through the authoritative spark queue}"
case "$ATTEMPT" in
  *[!0-9a-f]*|""|?????????????????????????????????*) echo "bad attempt id" >&2; exit 2 ;;
esac
[ "${#ATTEMPT}" -eq 32 ] || { echo "bad attempt id length" >&2; exit 2; }

ROOT="${SPARK_QUEUE_RUNTIME_ROOT:?missing queue runtime root}"
[ "$ROOT" = "/tmp/sparkqueue-$ATTEMPT" ] || { echo "unexpected job namespace: $ROOT" >&2; exit 2; }

RANK="${SPARK_QUEUE_RANK:?}"; SIZE="${SPARK_QUEUE_SIZE:?}"
[ "$SIZE" -eq "${#HOSTS[@]}" ] || { echo "queue size ${SIZE} != deployment hosts ${#HOSTS[@]}" >&2; exit 2; }
[ "$RANK" -ge 0 ] && [ "$RANK" -lt "$SIZE" ] || { echo "rank out of range" >&2; exit 2; }
# Rank hygiene: logical rank R must run on HOSTS[R].
if [ "${MINIMAX_SKIP_HOST_CHECK:-0}" != "1" ]; then
  ACTUAL_HOST="$(hostname)"
  [ "$ACTUAL_HOST" = "${HOSTS[$RANK]}" ] || {
    echo "rank/host mismatch: rank $RANK belongs on ${HOSTS[$RANK]}, this is $ACTUAL_HOST" >&2; exit 2; }
fi

reserved_ok() {
  local port="$1" entry first last
  [ -n "${SPARK_QUEUE_PORTS:-}" ] || { echo "queue reserved no ports" >&2; exit 2; }
  for entry in ${SPARK_QUEUE_PORTS//,/ }; do
    first="${entry%%:*}"; last="${entry##*:}"
    if [ "$port" -ge "$first" ] && [ "$port" -le "$last" ]; then return 0; fi
  done
  echo "listener port $port is not inside a queue-reserved range" >&2; return 1
}

for r in $(seq 0 $((SIZE - 1))); do
  reserved_ok $((CONTROL_BASE + r)) || exit 2
  reserved_ok $((COLLECTIVE_BASE + r)) || exit 2
  reserved_ok $((TRANSPORT_BASE + r)) || exit 2
done
# The TP session grid lives in the lane session block (MINIMAX ADAPTATION 2).
for r in $(seq 0 $((SIZE * SIZE - 1))); do
  reserved_ok $((SESSION_BASE + SIZE + r)) || exit 2
done

# MINIMAX ADAPTATION 3: verified firmware root (module_build_release output).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "$EXEC_PREFIX" ] || EXEC_PREFIX="$REPO/build/minimax-text-tp4-fw"
[ -x "$EXEC_PREFIX/bin/sparkpipe_model_residentd" ] || {
  echo "missing verified firmware root bin/sparkpipe_model_residentd: $EXEC_PREFIX (build via tools/module_build_release.sh in a queue job)" >&2; exit 2; }
[ -f "$EXEC_PREFIX/SOURCE_COMMIT" ] && [ -f "$EXEC_PREFIX/SHA256SUMS" ] || {
  echo "firmware root is not a verified release build (SOURCE_COMMIT/SHA256SUMS): $EXEC_PREFIX" >&2; exit 2; }

[ -n "$DEPLOYMENT_SOURCE" ] || DEPLOYMENT_SOURCE="$REPO/deployment/minimax_text_tp4/model_resident.json"
[ -f "$DEPLOYMENT_SOURCE" ] || { echo "missing family deployment: $DEPLOYMENT_SOURCE" >&2; exit 2; }

# --------------------------- PRIVATE RUNTIME PREP ----------------------------

python3 - "$ROOT" "$RANK" "$SIZE" "$CONTROL_BASE" "$COLLECTIVE_BASE" \
         "$TRANSPORT_BASE" "$SESSION_BASE" "$ATTEMPT" "$LANE" \
         "$DEPLOYMENT_SOURCE" "$EXEC_PREFIX" <<'PYEOF'
import json, sys
from pathlib import Path

root, rank, size = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
control_base, collective_base, transport_base, session_base = (int(a) for a in sys.argv[4:8])
attempt, lane, deployment_source, exec_prefix = sys.argv[8], int(sys.argv[9]), sys.argv[10], sys.argv[11]

def fail(msg):
    raise SystemExit("minimax wrapper prepare: FAIL: " + msg)

def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        fail(f"cannot read {path}: {error}")

deployment = load(deployment_source)
if len(deployment["nodes"]) != size:
    fail("deployment node count differs from queue topology")
value = json.loads(json.dumps(deployment))
value.pop("@comment", None)
value["transport"]["control_port_base"] = transport_base
host_names = []
for r, node in enumerate(value["nodes"]):
    if node["rank_index"] != r:
        fail("deployment ranks must be ordered")
    host_names.append(node["transport_host"])
    node["runtime_root"] = str(root / "runtime")
    node["control_endpoint"] = {"kind": "tcp", "host": node["transport_host"], "port": control_base + r}
    if node.get("kv_backing_directory") is not None:
        if not node.get("kv_backing_maximum_bytes", 0) > 0:
            fail("unbounded kv backing store")
        node["kv_backing_directory"] = str(root / "kv")

config_name = value["nodes"][rank]["adapter_configuration_path"]
source_root = Path(deployment["nodes"][rank]["runtime_root"])
config = load(source_root / config_name)

# The minimax stage config is the qwen38-lineage five-key schema; it carries
# no listener keys, so the template's remap pass reduces to a member check.
ALLOWED_KEYS = {"schema_version", "model_revision", "stage_pack_path", "max_sequence_positions", "tp_degree"}
for key in config:
    if key not in ALLOWED_KEYS:
        fail(f"unhandled stage configuration key: {key}")

runtime = root / "runtime"
runtime.mkdir(parents=True, exist_ok=True)
pack_path = config["stage_pack_path"]
if Path(pack_path).is_absolute() or ".." in Path(pack_path).parts:
    fail("pack escapes runtime root")
source_pack = source_root / pack_path
if not source_pack.is_file():
    fail(f"stage pack missing: {source_pack}")
pack_dir = runtime / "packs"
pack_dir.mkdir()
pack_dir.joinpath(source_pack.name).symlink_to(source_pack.resolve())
sidecar = Path(str(source_pack) + ".sha256")
if not sidecar.is_file():
    fail("missing pack .sha256 sidecar - the family identity fails closed without it")
pack_dir.joinpath(sidecar.name).write_bytes(sidecar.read_bytes())
digests = sorted(pack_dir.glob("*.sha256"))
if len(digests) != 1:
    fail(f"packs/ must contain exactly one *.sha256 digest, found {len(digests)}")
# MINIMAX ADAPTATION 1: no .experts sidecar for the dense interim path (#1138).

# MINIMAX ADAPTATION 3: adapter/driver/transport resolve against
# runtime_root; provide the verified coherent set from the firmware root
# (replacing one .so alone is not a coherent release).
lib_dir = runtime / "lib"
lib_dir.mkdir()
for relative in (deployment["adapter"]["shared_object_path"],
                 deployment["driver"]["shared_object_path"],
                 deployment["transport"]["shared_object_path"]):
    source_lib = Path(exec_prefix) / relative
    if not source_lib.is_file():
        fail(f"firmware artifact missing: {source_lib}")
    target = lib_dir / Path(relative).name
    target.symlink_to(source_lib.resolve())
stages_dir = runtime / "stages" / "stage_000"
stages_dir.mkdir(parents=True)
driver_source = Path(exec_prefix) / deployment["driver"]["shared_object_path"]
stages_dir.joinpath(driver_source.name).symlink_to(driver_source.resolve())

(runtime / config_name).parent.mkdir(parents=True, exist_ok=True)
(runtime / config_name).write_text(json.dumps(config) + "\n")
(root / "kv").mkdir(exist_ok=True)
(root / "deployment.json").write_text(json.dumps(value) + "\n")
print("minimax wrapper prepare: rank", rank, "pack", source_pack.name,
      "digest", digests[0].name)
PYEOF

# ------------------------------ WEIGHTD MODE ---------------------------------

if [ "$WEIGHTD_MODE" = "shared-socket" ]; then
  [ -n "$SHARED_SOCKET" ] || { echo "shared-socket mode requires MINIMAX_WEIGHTD_SOCKET" >&2; exit 2; }
  [ -S "$SHARED_SOCKET" ] || { echo "shared weightd socket is not present: $SHARED_SOCKET (operator-level; do not start daemons by hand)" >&2; exit 2; }
  WEIGHTD_SOCKET="$SHARED_SOCKET"
elif [ "$WEIGHTD_MODE" = "private" ]; then
  [ "$SIZE" -eq 1 ] || { echo "private-daemon mode is single-node; use tools/inference_smoke.py for the qualified multi-node smoke" >&2; exit 2; }
  WEIGHTD_SOCKET="$ROOT/weightd.sock"
  "$EXEC_PREFIX/bin/sparkpipe_weightd" --socket "$WEIGHTD_SOCKET" \
    --device-bytes-max $((DEVICE_MIB * 1024 * 1024)) \
    >"$ROOT/weightd.log" 2>&1 &
  WEIGHTD_PID=$!
  for _ in $(seq 1 300); do
    grep -q "spark_weightd ready " "$ROOT/weightd.log" 2>/dev/null && break
    kill -0 "$WEIGHTD_PID" 2>/dev/null || { echo "private weightd exited early" >&2; exit 2; }
    sleep 0.1
  done
  grep -q "spark_weightd ready " "$ROOT/weightd.log" || { echo "private weightd not ready in 30s" >&2; exit 2; }
else
  echo "unknown WEIGHTD_MODE: $WEIGHTD_MODE" >&2; exit 2
fi

# ------------------- MINIMAX ADAPTATION 2: TP ENVIRONMENT --------------------

IFS=',' read -ra RAIL_ARRAY <<< "$RAIL_HOSTS"
[ "${#RAIL_ARRAY[@]}" -eq "$SIZE" ] || {
  echo "rail hosts (${#RAIL_ARRAY[@]}) must match topology size $SIZE" >&2; exit 2; }

# Off-diagonal session grid inside the lane session block: SIZE*(SIZE-1)
# ports above the first SIZE slots, row-major, diagonal exactly zero (the
# module validates the shape).
SESSION_MATRIX=""
for row in $(seq 0 $((SIZE - 1))); do
  for column in $(seq 0 $((SIZE - 1))); do
    if [ "$row" -eq "$column" ]; then cell=0; else
      cell=$((SESSION_BASE + SIZE + row * SIZE + column))
    fi
    if [ -z "$SESSION_MATRIX" ]; then SESSION_MATRIX="$cell"; else SESSION_MATRIX="$SESSION_MATRIX,$cell"; fi
  done
done

export SPARK_MINIMAX_TP_DEGREE="$SIZE"
export SPARK_MINIMAX_TP_RANK="$RANK"
export SPARK_MINIMAX_STAGE_TP_BACKEND_PATH="$EXEC_PREFIX/lib/hidden_transport.so"
export SPARK_MINIMAX_STAGE_TP_IDENTIFIER="$((0x$(printf '%.15s' "$ATTEMPT") + 1 + LANE))"
export SPARK_MINIMAX_STAGE_TP_PORT_BASE="$TRANSPORT_BASE"
export SPARK_MINIMAX_STAGE_TP_HOSTS="$RAIL_HOSTS"
export SPARK_MINIMAX_STAGE_TP_LOCAL_HOST="${RAIL_ARRAY[$RANK]}"
export SPARK_MINIMAX_STAGE_TP_TIMEOUT_MS="120000"
export SPARK_MINIMAX_STAGE_TP_SESSION_PORTS="$SESSION_MATRIX"

# ----------------------------- RESIDENT LAUNCH -------------------------------

export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_SOCKET="$WEIGHTD_SOCKET"
# The weightd MESH LANE id space is 0..15 since the mesh16 table (PR #1176;
# SPARK_WEIGHTD_MESH_MAX_LANES 16) and is distinct from the port lane index:
# the dense module takes the env acquire path (no lazy pack to borrow a lane
# from), and the daemon binds a lane profile (degree-4 physical ranks
# 8,9,10,11) on first reservation.
MESH_LANE_ID="${MINIMAX_WEIGHTD_MESH_LANE:-10}"
export SPARK_WEIGHTD_LANE="$MESH_LANE_ID"
export SPARK_TP_MESH_RANKS="$MESH_RANKS"
export LD_LIBRARY_PATH="$EXEC_PREFIX/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
for entry in "${FAMILY_ENV[@]}"; do
  case "$entry" in *=*) export "$entry" ;; *) echo "bad FAMILY_ENV entry: $entry" >&2; exit 2 ;; esac
done

if [ "$WEIGHTD_MODE" = "private" ]; then
  trap 'kill -TERM "$WEIGHTD_PID" 2>/dev/null || true' EXIT
fi

# ------------------------ IN-JOB DECODE CELL (rank 0) ------------------------
# The k3 M3 cell precedent (tools/devcycle/batches/k3_t1_*_b1.json driven
# through the release-built sparkpipe_model_batch): the coordinator rank runs
# the batch binary from the same verified firmware root against this attempt's
# private deployment with the canonical T1 fixture at B1, all inside this
# job's cgroup and TTL. The qwen38-lineage adapter accepts the batch-tool
# submission shape; the model_api /v1/completions prefill shape is rejected
# (model_extension_bytes, serve-run13 evidence). Peer ranks drain on the
# rank-0 control listener transition instead of idling to the TTL deadline
# (the residentd never exits on client disconnect).

RESIDENTD_LOG="$ROOT/residentd_rank$RANK.log"

"$EXEC_PREFIX/bin/sparkpipe_model_residentd" \
  --deployment "$ROOT/deployment.json" \
  --rank-index "$RANK" >"$RESIDENTD_LOG" 2>&1 &
RESIDENTD_PID=$!

RESIDENTD_READY_TIMEOUT_S="${MINIMAX_RESIDENTD_READY_TIMEOUT_S:-540}"

residentd_wait_ready() {
  local deadline=$((SECONDS + RESIDENTD_READY_TIMEOUT_S))
  while [ "$SECONDS" -lt "$deadline" ]; do
    grep -q "model_residentd ready" "$RESIDENTD_LOG" 2>/dev/null && return 0
    if ! kill -0 "$RESIDENTD_PID" 2>/dev/null; then
      tail -5 "$RESIDENTD_LOG" >&2
      echo "minimax wrapper: residentd rank $RANK exited before ready" >&2
      return 1
    fi
    sleep 2
  done
  tail -5 "$RESIDENTD_LOG" >&2
  echo "minimax wrapper: residentd rank $RANK not ready in ${RESIDENTD_READY_TIMEOUT_S}s" >&2
  return 1
}

control_port_up() {
  (exec 3<>"/dev/tcp/${HOSTS[0]}/$((CONTROL_BASE))") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
  return 1
}

if [ "$RANK" -ne 0 ]; then
  UP_SEEN=""
  for _ in $(seq 1 $((RESIDENTD_READY_TIMEOUT_S / 5))); do
    if control_port_up; then
      UP_SEEN=1
      break
    fi
    if ! kill -0 "$RESIDENTD_PID" 2>/dev/null; then
      tail -5 "$RESIDENTD_LOG" >&2
      echo "minimax wrapper: peer rank $RANK residentd exited before the decode cell came up" >&2
      exit 2
    fi
    sleep 5
  done
  if [ -n "$UP_SEEN" ]; then
    for _ in $(seq 1 60); do
      control_port_up || break
      sleep 5
    done
    kill -TERM "$RESIDENTD_PID" 2>/dev/null || true
    wait "$RESIDENTD_PID" 2>/dev/null || true
    echo "minimax wrapper: peer rank $RANK drained after decode cell"
    exit 0
  fi
  kill -TERM "$RESIDENTD_PID" 2>/dev/null || true
  echo "minimax wrapper: peer rank $RANK never saw the coordinator control listener" >&2
  exit 1
fi

residentd_wait_ready || { kill -TERM "$RESIDENTD_PID" 2>/dev/null || true; exit 2; }

python3 - "$REPO" "$ROOT" > "$ROOT/t1-batch.json" <<'PYBATCH'
import json, sys
prompts = json.load(open(sys.argv[1] + "/qualification/t1_reference/minimax/prompts.json"))["prompts"]
requests = []
for index, prompt in enumerate(prompts, 1):
    requests.append({
        "request_id": index,
        "sequence_id": index,
        "priority": 0,
        "output_token_budget": prompt["new_tokens"],
        "prompt_token_ids": prompt["prompt_token_ids"],
    })
batch = {
    "schema_version": 1,
    "connect_timeout_ms": 480000,
    "request_capacity": len(requests),
    "max_context_tokens": 128,
    "max_prefill_rows_per_submission": 1,
    "maximum_messages_per_rank_per_progress": 8,
    "maximum_new_submissions_per_progress": 1,
    "stop_token_ids": [],
    "requests": requests,
}
open(sys.argv[2] + "/t1-batch.json", "w").write(json.dumps(batch, indent=1) + "\n")
PYBATCH

BATCH_LOG="$ROOT/batch.log"
set +e
"$EXEC_PREFIX/bin/sparkpipe_model_batch" \
  --deployment "$ROOT/deployment.json" \
  --runtime-root "$ROOT" \
  --batch "$ROOT/t1-batch.json" >"$BATCH_LOG" 2>&1
BATCH_RC=$?
set -e
sed 's/^/DECODE-BATCH /' "$BATCH_LOG"
echo "DECODE-BATCH-RC rc=$BATCH_RC"

kill -TERM "$RESIDENTD_PID" 2>/dev/null || true
wait "$RESIDENTD_PID" 2>/dev/null || true
echo "DECODE-DONE lane=$LANE attempt=$ATTEMPT mesh_lane=$MESH_LANE_ID"
exit "$BATCH_RC"
