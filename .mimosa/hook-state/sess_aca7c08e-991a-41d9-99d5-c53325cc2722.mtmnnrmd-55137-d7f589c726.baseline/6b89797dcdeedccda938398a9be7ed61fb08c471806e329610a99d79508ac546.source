#!/usr/bin/env bash
# Stage + launch the qwen3.8 27B TP1 dev instance WITH the DFlash2 drafter
# on any spark that already holds the incumbent qwen38.fp8.tp1 deployment
# and the incoai drafter pack.
#
# THE DFLASH2 LAUNCH ENV IS MANDATORY (lane README): a spec run missing it
# produces degenerate output. It is emitted verbatim below.
#
# Usage:  SPARK_HOST=spark9 tools/qwen38_27b_dflash2_serve.sh stage|launch|api|smoke|stop
#   stage  - create ~/sparkdata/qwen38.fp8.tp1.dflash2 (binaries hardlinked,
#            pack by absolute path, host-correct configs, KV halved)
#   launch - residentd with the mandatory env (spawn-captured pid recorded)
#   api    - the model_api once "model_residentd ready" is in the log
#   smoke  - one greedy completion + the acceptance lines from the log
#   stop   - TERM only the pids this script recorded (spawn-captured, never
#            fuzzy-matched)
set -euo pipefail

SPARK_HOST="${SPARK_HOST:?set SPARK_HOST, e.g. spark9}"
ACTION="${1:?usage: $0 stage|launch|api|smoke|stop}"
INCUMBENT="/home/${SPARK_HOST}/sparkdata/qwen38.fp8.tp1"
DEPLOY="/home/${SPARK_HOST}/sparkdata/qwen38.fp8.tp1.dflash2"
DRAFTER="/home/${SPARK_HOST}/sparkdata/qwen38-dflash2-drafter-incoai.qwen38_27bsp"
CONTROL_PORT=17480
TRANSPORT_BASE=58700
API_PORT=17490
PID_FILE="/tmp/qwen38_dflash2_${SPARK_HOST}.pids"

run_remote() { ssh -o BatchMode=yes "$SPARK_HOST" "$@"; }

stage() {
	run_remote "test -f ${DRAFTER} || { echo 'drafter pack missing'; exit 1; }"
	run_remote "test -d ${INCUMBENT}/bin || { echo 'incumbent deployment missing'; exit 1; }"
	run_remote "mkdir -p ${DEPLOY}/config"
	run_remote "cp -a ${INCUMBENT}/bin ${INCUMBENT}/lib ${INCUMBENT}/link_units ${DEPLOY}/"
	# the pack stays where it is - read-only, referenced by absolute path
	run_remote "cat > ${DEPLOY}/config/model_resident.json" <<JSON
{
  "schema_version": 2,
  "coordinator_rank_index": 0,
  "adapter": { "shared_object_path": "lib/model_serving_adapter.so" },
  "driver": {
    "shared_object_path": "lib/model_driver.so",
    "program_name": "resident_decode"
  },
  "transport": {
    "shared_object_path": "lib/hidden_transport.so",
    "mode": "host-rdma",
    "control_port_base": ${TRANSPORT_BASE}
  },
  "runtime_limits": {
    "max_inflight_submissions": 2,
    "max_active_sequences": 64,
    "max_input_rows": 128,
    "resident_sequence_capacity": 64,
    "kv_logical_page_capacity": 256,
    "kv_physical_page_capacity": 64
  },
  "nodes": [
    {
      "rank_index": 0,
      "stage_index": 0,
      "runtime_root": "${DEPLOY}",
      "node_target": "cuda.sm121.qwen38_27b.resident_decode_stage.bf16",
      "transport_host": "${SPARK_HOST}",
      "adapter_configuration_path": "config/qwen38_27b_tp1_rank0.json",
      "kv_backing_directory": null,
      "kv_backing_maximum_bytes": 0,
      "control_endpoint": { "kind": "tcp", "host": "${SPARK_HOST}", "port": ${CONTROL_PORT} }
    }
  ]
}
JSON
	# max_sequence_positions 4096 (not the incumbent's 8192): co-resident
	# sizing per the 110 GiB ceiling - weights 29.9G + ~35G KV pool +
	# drafter 3.6G + the glm5_next rank 21.7G stays well under the line
	run_remote "cat > ${DEPLOY}/config/qwen38_27b_tp1_rank0.json" <<JSON
{
  "schema_version": 3,
  "model_revision": "bf16-h5120-l64-gdn48-full16-v248320-mtp1-v1",
  "stage_pack_path": "${INCUMBENT}/packs/qwen38-fp8.tp1.qwen36sp",
  "max_sequence_positions": 4096,
  "speculative_draft_count": 8
}
JSON
	run_remote "ls ${DEPLOY}/bin ${DEPLOY}/config && echo staged"
}

launch() {
	# pre-truncate the log (the stale-log grep race), then the daemon; the
	# pid is captured AT SPAWN and recorded - stop() TERMs exactly these
	run_remote "cat > /tmp/launch_dflash2_${SPARK_HOST}.sh" <<LAUNCH
#!/usr/bin/env bash
set -euo pipefail
cd ${DEPLOY}
: > /tmp/qwen38_dflash2_${SPARK_HOST}.log
env LD_LIBRARY_PATH=\${PWD}/lib:\${LD_LIBRARY_PATH:-} \\
  SPARK_QWEN38_27B_SPECULATORS=0x4 \\
  SPARK_QWEN38_27B_DSPARK_PACK_PATH=${DRAFTER} \\
  SPARK_QWEN38_27B_DFLASH2_STATE_SELECT=1 \\
  SPARK_QWEN38_27B_DFLASH2_BONUS_FOLD=2 \\
  SPARK_QWEN38_27B_DFLASH2_BLOCK_KV=0 \\
  SPARK_QWEN38_27B_DFLASH2_WINDOW=2048 \\
  SPARK_QWEN38_27B_DFLASH2_CTX_CACHE=1 \\
  setsid nohup bin/sparkpipe_model_residentd \\
    --deployment config/model_resident.json --rank-index 0 \\
    > /tmp/qwen38_dflash2_${SPARK_HOST}.log 2>&1 < /dev/null &
echo \$! > /tmp/qwen38_dflash2_${SPARK_HOST}.residentd.pid
LAUNCH
	run_remote "bash /tmp/launch_dflash2_${SPARK_HOST}.sh"
	echo "residentd pid: $(run_remote "cat /tmp/qwen38_dflash2_${SPARK_HOST}.residentd.pid")"
	echo "waiting for the ready line..."
	run_remote "for i in \$(seq 1 120); do grep -q 'model_residentd ready' /tmp/qwen38_dflash2_${SPARK_HOST}.log && break; sleep 5; done; grep -e 'ready' -e 'refused' -e 'failed' /tmp/qwen38_dflash2_${SPARK_HOST}.log | tail -5"
}

api() {
	run_remote "cd ${DEPLOY} && env LD_LIBRARY_PATH=\${PWD}/lib:\${LD_LIBRARY_PATH:-} setsid nohup bin/sparkpipe_model_api --deployment config/model_resident.json --runtime-root ${DEPLOY} --port ${API_PORT} > /tmp/qwen38_dflash2_api_${SPARK_HOST}.log 2>&1 < /dev/null & echo \$! > /tmp/qwen38_dflash2_${SPARK_HOST}.api.pid; sleep 3; cat /tmp/qwen38_dflash2_api_${SPARK_HOST}.log | tail -3"
	echo "api pid: $(run_remote "cat /tmp/qwen38_dflash2_${SPARK_HOST}.api.pid")"
}

smoke() {
	run_remote "curl -s --max-time 300 http://${SPARK_HOST}:${API_PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen38-27b\",\"prompt\":\"The capital of France is\",\"max_tokens\":\"64\",\"temperature\":0}' | head -c 600; echo"
	run_remote "grep -e 'qwen38_27b_spec' /tmp/qwen38_dflash2_${SPARK_HOST}.log | tail -8"
}

stop() {
	# TERM only the pids THIS script recorded at spawn; never a fuzzy match
	run_remote "for f in /tmp/qwen38_dflash2_${SPARK_HOST}.residentd.pid /tmp/qwen38_dflash2_${SPARK_HOST}.api.pid; do if [ -f \$f ]; then pid=\$(cat \$f); kill -TERM \$pid 2>/dev/null && echo \"TERM \$pid (\$f)\" || echo \"pid \$pid already gone\"; fi; done"
}

case "$ACTION" in
	stage) stage ;;
	launch) launch ;;
	api) api ;;
	smoke) smoke ;;
	stop) stop ;;
	*) echo "unknown action $ACTION"; exit 2 ;;
esac
