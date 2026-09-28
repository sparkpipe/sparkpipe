#!/usr/bin/env bash
set -euo pipefail

HOSTS=(spark0 spark1 spark2 spark5)
MESH_RANKS=0,1,2,5
LANE=3
PACK_NAME=qwen27b.mx2.tp4
ROOT_TEMPLATE='/home/{host}/sparkdata/qwen27b.mx2.tp4'
STAGE_HOST=spark8
STAGE_PACKS=/home/${STAGE_HOST}/sparkdata/qwen27b.mx2.tp4/packs
STAGE_TP1_ROOT=/home/${STAGE_HOST}/sparkdata/qwen27b.fp8.tp1
BUILD_BASE=/home/${STAGE_HOST}/build-qwen27b
WEIGHTD_SOCKET=/tmp/spark_weightd.sock
CONTROL_PORT_BASE=$((23000 + 16 * LANE))
COLLECTIVE_PORT_BASE=$((53000 + 16 * LANE))
TRANSPORT_PORT_BASE=$((64000 + 16 * LANE))
SEQUENCES=8
MAX_POSITIONS=8192
KV_LOGICAL_PAGES=256
KV_PHYSICAL_PAGES=64
DRAFT_COUNT=8
UNIT=sp-qwen4-rd
HUB=rtx5090
API_PORT=8435
API_CHANNEL=/home/spec/qwen27b-api-channel
CHECKPOINT=/mnt/model-warm/qwen3.8-27b-fp8
EOS_TOKEN_IDS=248046,248044
TOKENIZER_VOCABULARY_SIZE=248077
GATE_FILE=/Users/mac/sparkpipe-coord/lanes/AGENT_FIX_STATUS
MIN_AVAILABLE_GIB=20
HERE=$(cd "$(dirname "$0")/.." && pwd)

usage() {
	cat >&2 <<USAGE
usage: $0 ACTION [ARG]
  build SHA               archive SHA to ${STAGE_HOST}:${BUILD_BASE}/src-SHA; host binaries, TP4 adapter, module archive, weightd tests
  publish SHA             module GPU validation on ${STAGE_HOST} (TP1 pack, lane ${LANE}, arena reclaimed after) and driver compile
  install SHA             ${HOSTS[*]}: rank pack from ${STAGE_HOST}, binaries/driver/TP4 adapter from src-SHA, drafter from the TP1 root
  configs                 write the TP4 deployment and per-rank adapter configs to every node and build/api.tp4.model_resident.json
  launch nospec|dflash2   residentd rank R on HOSTS[R] in unit ${UNIT} (lane ${LANE}, mesh ${MESH_RANKS}); refuses unless ${GATE_FILE} says DEPLOYED
  status | stop | fresh MODE | reclaim
  refcheck MODE FILE      reference prompts through sparkpipe_model_batch on rank 0 (fresh ranks per prompt)
  repeat MODE CASE N OUT  N sequential requests of one bench case on ONE engine (one batch file), token hashes per request
  perf MODE CASES REPS OUT tools/qwen38_27b_tp1_bench.py cases on rank 0, fresh ranks per run
  api-install SHA         x86 API + TP4 adapter on ${HUB}, stage ${API_CHANNEL}, write qwen27b-api.service
  api-start | api-stop | smoke
  compsec SHA RUN_ID TOK  COMPSEC-17 (qwen template, thinking off, TOK max tokens) against :${API_PORT}
USAGE
	exit 2
}
ACTION="${1:-}"
[ -n "$ACTION" ] || usage
on() { ssh -o BatchMode=yes "$@"; }
root_of() { printf '%s' "${ROOT_TEMPLATE//\{host\}/$1}"; }

gate() {
	grep -q '^DEPLOYED' "$GATE_FILE" || { echo "refused: $(head -c 120 "$GATE_FILE")" >&2; exit 3; }
	local h avail
	for h in "${HOSTS[@]}"; do
		avail=$(on "$h" "free -g | awk '/Mem:/{print \$7}'")
		[ "$avail" -ge "$MIN_AVAILABLE_GIB" ] || { echo "refused: $h MemAvailable ${avail} GiB < ${MIN_AVAILABLE_GIB}" >&2; exit 3; }
	done
}

build() {
	local sha=${1:?SHA}
	git -C "$HERE" archive --format=tar --prefix="src-${sha}/" "$sha" | on "$STAGE_HOST" "mkdir -p ${BUILD_BASE} && rm -rf ${BUILD_BASE}/src-${sha} && tar -xf - -C ${BUILD_BASE}"
	on "$STAGE_HOST" "cd ${BUILD_BASE}/src-${sha} && systemd-run --user --wait --collect --unit=sp-qwen4-build-${sha} --same-dir -p Nice=10 -p StandardOutput=file:${BUILD_BASE}/src-${sha}/build-qwen27b.log -p StandardError=file:${BUILD_BASE}/src-${sha}/build-qwen27b.log bash -c '
export PATH=/usr/local/cuda/bin:\$PATH
set -e
make -j8 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a build/sparkpipe_weightd build/weightd_warm build/sparkpipe_model_residentd build/sparkpipe_model_batch build/sparkpipe_model_compile build/sparkpipe_module_publish build/sparkpipe_driver_inspect hidden_transport_spark_host_rdma_verbs
make -j8 build/test_stage_module_weightd build/test_weightd_attach build/test_weightd_map
./build/test_stage_module_weightd && ./build/test_weightd_attach && ./build/test_weightd_map
make -j8 QWEN38_27B_SERVING_TOPOLOGY_FLAGS=-DSPARK_QWEN38_27B_SERVING_TP_DEGREE=4u build/libqwen38_27b_serving_adapter.so
make -j8 -C modules/qwen38_27b_resident_decode_stage CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a archive
'; echo build_rc=\$?; tail -3 build-qwen27b.log | cut -c1-200; strings build/libqwen38_27b_serving_adapter.so | grep -a 'serving-adapter.tp4'"
}

publish() {
	local sha=${1:?SHA} src=${BUILD_BASE}/src-${1}
	on "$STAGE_HOST" "test -S ${WEIGHTD_SOCKET} || { echo 'weightd socket missing'; exit 2; }
cd ${src} && digest=\$(cut -d' ' -f1 ${STAGE_TP1_ROOT}/packs/qwen38-mx2.tp1.qwen36sp.sha256) && systemd-run --user --wait --collect --unit=sp-qwen4-publish --same-dir -p StandardOutput=file:${src}/publish-unit.log -p StandardError=file:${src}/publish-unit.log -E SPARK_WEIGHTD_ATTACH=1 -E SPARK_WEIGHTD_SOCKET=${WEIGHTD_SOCKET} -E SPARK_WEIGHTD_PACK_SHA256=\$digest -E SPARK_WEIGHTD_LANE=${LANE} bash -c '
export PATH=/usr/local/cuda/bin:\$PATH
set -e
make -C modules/qwen38_27b_resident_decode_stage CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a ALLOW_UNQUALIFIED_EXECUTION=1 TP_DEGREE=1 TP_RANK=0 TP_STANDALONE=1 STAGE_COUNT=1 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=64 MTP_LAYER_COUNT=1 GDN_SNAPSHOT_SLOT_COUNT=16 MAX_ACTIVE_SEQUENCES=8 KV_BLOCK_COUNT=8 STAGE_PACK_PATH=${STAGE_TP1_ROOT}/packs/qwen38-mx2.tp1.qwen36sp publish > publish.log 2>&1
grep -q -E \"qwen38_27b_validation PASS|validation=reused\" publish.log
rm -rf build/tp-driver
build/sparkpipe_model_compile --model examples/model_descriptions/qwen38_27b_resident_decode_stage_firmware.json --library build/module_library --output build/tp-driver --cc /usr/bin/cc --include include --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl --cc-arg -pthread > driver-link.log 2>&1
build/sparkpipe_driver_inspect build/tp-driver/stages/stage_000/model_driver.so cuda.sm121.qwen38_27b.resident_decode_stage.bf16 > driver-inspect.log 2>&1
! ldd -r build/tp-driver/stages/stage_000/model_driver.so | grep -E \"not found|undefined symbol\"
'; echo publish_rc=\$?; grep -a -E 'validation (PASS|FAIL)|validation=' publish.log | tail -3; ~/weightd_warm_09fdad6 ${WEIGHTD_SOCKET} --reclaim 2>&1 | tail -1; free -g | head -2"
}

install() {
	local src=${1:?SHA} rank host root
	for rank in "${!HOSTS[@]}"; do
		host=${HOSTS[$rank]}
		root=$(root_of "$host")
		on "$host" "mkdir -p ${root}/packs ${root}/bin ${root}/lib ${root}/stages ${root}/drafter ${root}/config ${root}/logs && find ${root}/packs -name '*.sha256' ! -name '${PACK_NAME}.rank${rank}.qwen36sp.sha256' -delete"
		on "$STAGE_HOST" "rsync -a ${STAGE_PACKS}/${PACK_NAME}.rank${rank}.qwen36sp ${STAGE_PACKS}/${PACK_NAME}.rank${rank}.qwen36sp.sha256 ${host}:${root}/packs/ && rsync -a ${BUILD_BASE}/src-${src}/build/sparkpipe_model_residentd ${BUILD_BASE}/src-${src}/build/sparkpipe_model_batch ${host}:${root}/bin/ && rsync -a --delete ${BUILD_BASE}/src-${src}/build/tp-driver/stages/ ${host}:${root}/stages/ && rsync -a ${STAGE_TP1_ROOT}/drafter/ ${host}:${root}/drafter/ && rsync -a ${BUILD_BASE}/src-${src}/build/libhidden_transport_spark_host_rdma_verbs.so ${host}:${root}/lib/hidden_transport.so && rsync -a ${BUILD_BASE}/src-${src}/build/libqwen38_27b_serving_adapter.so ${host}:${root}/lib/model_serving_adapter.so" &
	done
	wait
	for rank in "${!HOSTS[@]}"; do
		host=${HOSTS[$rank]}
		root=$(root_of "$host")
		on "$host" "cd ${root} && echo ${src} > SOURCE_COMMIT && (cd packs && sha256sum -c ${PACK_NAME}.rank${rank}.qwen36sp.sha256) && (cd drafter && sha256sum -c *.sha256) && find bin lib stages -type f | sort | xargs sha256sum > SHA256SUMS && echo ${host} rank${rank} \$(sha256sum lib/model_serving_adapter.so stages/stage_000/model_driver.so | cut -c1-16 | tr '\n' ' ')"
	done
}

configs() {
	local tmp rank host
	tmp=$(mktemp -d)
	on "$HUB" "cat ${CHECKPOINT}/tokenizer.json" > "$tmp/tokenizer.json"
	python3 "$HERE/tools/qwen38_27b_tp4_deployment.py" --hosts "$(IFS=,; echo "${HOSTS[*]}")" \
		--runtime-root "$ROOT_TEMPLATE" --pack-name "$PACK_NAME" --weightd-socket "$WEIGHTD_SOCKET" \
		--control-port-base "$CONTROL_PORT_BASE" --collective-port-base "$COLLECTIVE_PORT_BASE" \
		--transport-port-base "$TRANSPORT_PORT_BASE" --attempt "$(date -u +%Y%m%d%H%M%S)0" \
		--sequences "$SEQUENCES" --max-positions "$MAX_POSITIONS" --kv-logical-pages "$KV_LOGICAL_PAGES" \
		--kv-physical-pages "$KV_PHYSICAL_PAGES" --draft-count "$DRAFT_COUNT" --eos-token-ids "$EOS_TOKEN_IDS" \
		--tokenizer "$tmp/tokenizer.json" --tokenizer-vocabulary-size "$TOKENIZER_VOCABULARY_SIZE" --output "$tmp/config"
	for rank in "${!HOSTS[@]}"; do
		host=${HOSTS[$rank]}
		cp "$tmp/config/stage.rank${rank}.json" "$tmp/config/stage.json"
		cp "$tmp/config/stage.dflash2.rank${rank}.json" "$tmp/config/stage.dflash2.json"
		scp -q "$tmp/config/model_resident.nospec.json" "$tmp/config/model_resident.dflash2.json" "$tmp/config/stage.json" "$tmp/config/stage.dflash2.json" "${host}:$(root_of "$host")/config/"
	done
	mkdir -p "$HERE/build"
	cp "$tmp/config/api.model_resident.json" "$HERE/build/api.tp4.model_resident.json"
	rm -r "$tmp"
	echo configs written to "${HOSTS[*]}"
}

spec_env() {
	case "$1" in
		nospec) printf '%s' "-E SPARK_QWEN38_27B_SPECULATORS=0" ;;
		dflash2) printf '%s' "-E SPARK_QWEN38_27B_SPECULATORS=0x4 -E SPARK_QWEN38_27B_DSPARK_PACK_PATH=$2/drafter/qwen38-dflash2-drafter.qwen36sp -E SPARK_QWEN38_27B_DFLASH2_STATE_SELECT=1 -E SPARK_QWEN38_27B_DFLASH2_BONUS_FOLD=2 -E SPARK_QWEN38_27B_DFLASH2_BLOCK_KV=0 -E SPARK_QWEN38_27B_DFLASH2_WINDOW=2048 -E SPARK_QWEN38_27B_DFLASH2_CTX_CACHE=1" ;;
		*) echo "mode must be nospec or dflash2" >&2; exit 2 ;;
	esac
}

launch() {
	local mode=${1:?nospec|dflash2} rank host root
	gate
	for rank in "${!HOSTS[@]}"; do
		host=${HOSTS[$rank]}
		root=$(root_of "$host")
		on "$host" "set -e
systemctl --user is-active --quiet ${UNIT} && { echo '${host}: ${UNIT} already running'; exit 2; }
test -S ${WEIGHTD_SOCKET} || { echo '${host}: weightd socket missing'; exit 2; }
cd ${root} && log=${root}/logs/residentd-${mode}-\$(date -u +%Y%m%dT%H%M%SZ).log && ln -sfn \$log ${root}/logs/current.log && echo ${mode} > ${root}/logs/current.mode
systemd-run --user --collect --unit=${UNIT} --same-dir -p MemoryMax=24G -p MemorySwapMax=0 -p StandardOutput=file:\$log -p StandardError=file:\$log -E LD_LIBRARY_PATH=${root}/lib -E SPARK_WEIGHTD_LANE=${LANE} -E SPARK_TP_MESH_RANKS=${MESH_RANKS} -E SPARK_QWEN38_27B_TP_STANDALONE=0 -E CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0 $(spec_env "$mode" "$root") ${QWEN27B_DIAG_ENV:-} ${root}/bin/sparkpipe_model_residentd --deployment ${root}/config/model_resident.${mode}.json --rank-index ${rank} > /dev/null" &
	done
	wait
	for rank in "${!HOSTS[@]}"; do
		host=${HOSTS[$rank]}
		root=$(root_of "$host")
		on "$host" "for i in \$(seq 1 300); do grep -aq 'model_residentd ready' ${root}/logs/current.log && break; systemctl --user is-active --quiet ${UNIT} || break; sleep 1; done
echo ${host} rank${rank}: \$(grep -a -E 'model_residentd ready|refused|failed|ERRSITE|mismatch' ${root}/logs/current.log | tail -3) avail=\$(free -g | awk '/Mem:/{print \$7}')G"
	done
}

status() {
	local rank host root
	for rank in "${!HOSTS[@]}"; do
		host=${HOSTS[$rank]}
		root=$(root_of "$host")
		on "$host" "echo ${host} rank${rank} \$(systemctl --user is-active ${UNIT}) \$(cat ${root}/logs/current.mode 2>/dev/null) \$(grep -a -c 'model_residentd ready' ${root}/logs/current.log 2>/dev/null) avail=\$(free -g | awk '/Mem:/{print \$7}')G" || true
	done
}

stop() {
	local host
	for host in "${HOSTS[@]}"; do
		on "$host" "systemctl --user stop ${UNIT} 2>/dev/null; systemctl --user reset-failed ${UNIT} 2>/dev/null; echo ${host} \$(systemctl --user is-active ${UNIT})" &
	done
	wait
}

reclaim() {
	local host
	for host in "${HOSTS[@]}"; do
		on "$host" "systemctl --user is-active --quiet ${UNIT} && { echo '${host}: ${UNIT} running, not reclaiming'; exit 0; }; ~/weightd_warm_09fdad6 ${WEIGHTD_SOCKET} --reclaim 2>&1 | tail -1; echo ${host} avail=\$(free -g | awk '/Mem:/{print \$7}')G"
	done
}

fresh() {
	stop > /dev/null
	launch "${1:?mode}"
}

coordinator() { printf '%s' "${HOSTS[0]}"; }

refcheck() {
	local mode=${1:?nospec|dflash2} file=${2:?reference prompts json} name host root
	host=$(coordinator)
	root=$(root_of "$host")
	scp -q "$file" "${host}:${root}/logs/refprompts.json"
	for name in $(python3 -c "import json,sys; print(' '.join(p['name'] for p in json.load(open(sys.argv[1]))['prompts']))" "$file"); do
		fresh "$mode" > /dev/null
		on "$host" "cd ${root} && python3 - ${name} ${mode} <<'EOF'
import json, subprocess, os, sys
name, mode = sys.argv[1], sys.argv[2]
p = [q for q in json.load(open('logs/refprompts.json'))['prompts'] if q['name'] == name][0]
env = dict(os.environ, LD_LIBRARY_PATH='${root}/lib')
batch = {'schema_version': 1, 'connect_timeout_ms': 30000, 'request_capacity': 2, 'max_context_tokens': 4096, 'max_prefill_rows_per_submission': 8, 'maximum_messages_per_rank_per_progress': 8, 'maximum_new_submissions_per_progress': 2, 'stop_token_ids': [], 'requests': [{'request_id': 900, 'sequence_id': 900, 'priority': 0, 'output_token_budget': p['new_tokens'], 'prompt_token_ids': p['prompt_token_ids']}]}
json.dump(batch, open('logs/ref-%s.json' % name, 'w'))
run = subprocess.run(['bin/sparkpipe_model_batch', '--deployment', 'config/model_resident.%s.json' % mode, '--runtime-root', '${root}', '--batch', 'logs/ref-%s.json' % name], capture_output=True, text=True, env=env)
got = [json.loads(l)['token_id'] for l in run.stdout.splitlines() if l.startswith('{') and json.loads(l).get('event') == 'token']
want = p.get('reference_token_ids', [])
same = 0
while same < min(len(got), len(want)) and got[same] == want[same]:
    same += 1
print(json.dumps({'topology': 'tp4', 'mode': mode, 'prompt': name, 'engine': got, 'reference': want, 'matching_prefix': same, 'exact': got == want, 'batch_rc': run.returncode, 'stderr_tail': run.stderr.strip().splitlines()[-1:]}))
EOF"
	done
}

repeat_case() {
	local mode=${1:?mode} case=${2:?case} count=${3:?N} out=${4:?local jsonl} host root
	host=$(coordinator)
	root=$(root_of "$host")
	scp -q "$HERE/qualification/qwen38_27b/bench_prompts_tp1.json" "${host}:${root}/logs/bench_prompts_tp1.json"
	fresh "$mode" > /dev/null
	on "$host" "cd ${root} && python3 - ${mode} ${case} ${count} <<'EOF'
import hashlib, json, os, subprocess, sys
mode, case, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts = json.load(open('logs/bench_prompts_tp1.json'))
key = {'o128': ('prose', 128), 'o512': ('prose', 512), 'code512': ('code', 512), 'rep512': ('repetitive', 512)}[case]
reqs = [{'request_id': 5000 + i, 'sequence_id': 5000 + i, 'priority': 0, 'output_token_budget': key[1], 'prompt_token_ids': prompts[key[0]]} for i in range(count)]
batch = {'schema_version': 1, 'connect_timeout_ms': 30000, 'request_capacity': 2, 'max_context_tokens': 4096, 'max_prefill_rows_per_submission': 128, 'maximum_messages_per_rank_per_progress': 8, 'maximum_new_submissions_per_progress': 1, 'stop_token_ids': [], 'requests': reqs}
json.dump(batch, open('logs/repeat.json', 'w'))
env = dict(os.environ, LD_LIBRARY_PATH='${root}/lib')
run = subprocess.run(['bin/sparkpipe_model_batch', '--deployment', 'config/model_resident.%s.json' % mode, '--runtime-root', '${root}', '--batch', 'logs/repeat.json'], capture_output=True, text=True, env=env)
toks = {}
for line in run.stdout.splitlines():
    if line.startswith('{'):
        e = json.loads(line)
        if e.get('event') == 'token':
            toks.setdefault(e['request_id'], []).append(e['token_id'])
for rid in sorted(toks):
    t = toks[rid]
    print(json.dumps({'topology': 'tp4', 'mode': mode, 'case': case, 'request_id': rid, 'tokens': len(t), 'stream16': hashlib.sha256(str(t).encode()).hexdigest()[:16], 'head': t[:24]}))
print(json.dumps({'batch_rc': run.returncode, 'requests': len(toks), 'distinct_streams': len({str(v) for v in toks.values()}), 'stderr_tail': run.stderr.strip().splitlines()[-1:]}))
EOF" | tee -a "$out"
}

perf() {
	local mode=${1:?mode} cases=${2:?comma list} reps=${3:?reps} out=${4:?local jsonl} prefill_rows=${5:-128} case rep host root
	host=$(coordinator)
	root=$(root_of "$host")
	scp -q "$HERE/tools/qwen38_27b_tp1_bench.py" "${host}:${root}/logs/qwen38_27b_tp1_bench.py"
	scp -q "$HERE/qualification/qwen38_27b/bench_prompts_tp1.json" "${host}:${root}/logs/bench_prompts_tp1.json"
	for case in ${cases//,/ }; do
		for rep in $(seq 1 "$reps"); do
			fresh "$mode" > /dev/null 2>&1
			on "$host" "cd ${root} && LD_LIBRARY_PATH=${root}/lib python3 logs/qwen38_27b_tp1_bench.py ${case} --root ${root} --deployment ${root}/config/model_resident.${mode}.json --prompts logs/bench_prompts_tp1.json --prefill-rows ${prefill_rows} | python3 -c '
import json, sys, re
r = json.loads(sys.stdin.read())
acc = [int(m.group(1)) for m in re.finditer(r\"qwen38_27b_spec accepted=(\\d+)\", open(\"${root}/logs/current.log\", errors=\"replace\").read())]
r.update(topology=\"tp4\", mode=\"${mode}\", rep=${rep}, prefill_rows=${prefill_rows}, source_commit=open(\"${root}/SOURCE_COMMIT\").read().strip(), spec_rounds=len(acc), mean_accepted_drafts=round(sum(acc) / len(acc), 3) if acc else None, mean_tokens_per_round=round(sum(a + 1 for a in acc) / len(acc), 3) if acc else None)
print(json.dumps(r))
'" | tee -a "$out"
		done
	done
	stop > /dev/null 2>&1
}

api_install() {
	local sha=${1:?SHA} tmp
	test -f "$HERE/build/api.tp4.model_resident.json" || { echo "run configs first" >&2; exit 2; }
	git -C "$HERE" archive --format=tar --prefix="api-build-qwen27b-${sha}/" "$sha" | on "$HUB" "tar -xf - -C ~"
	on "$HUB" "cd ~/api-build-qwen27b-${sha} && export PATH=/usr/local/cuda/bin:\$PATH && make -j6 build/sparkpipe_model_api > build-api.log 2>&1 && rm -f build/libqwen38_27b_serving_adapter.so && make -j6 QWEN38_27B_SERVING_TOPOLOGY_FLAGS=-DSPARK_QWEN38_27B_SERVING_TP_DEGREE=4u build/libqwen38_27b_serving_adapter.so > build-adapter.log 2>&1 && ! ldd -r build/libqwen38_27b_serving_adapter.so | grep -E 'not found|undefined' && strings build/libqwen38_27b_serving_adapter.so | grep -q serving-adapter.tp4 && mkdir -p ${API_CHANNEL}/bin ${API_CHANNEL}/runtime/lib ${API_CHANNEL}/runtime/tokenizer && cp build/sparkpipe_model_api ${API_CHANNEL}/bin/.api.tmp && mv ${API_CHANNEL}/bin/.api.tmp ${API_CHANNEL}/bin/sparkpipe_model_api && cp build/libqwen38_27b_serving_adapter.so ${API_CHANNEL}/runtime/lib/.ad.tmp && mv ${API_CHANNEL}/runtime/lib/.ad.tmp ${API_CHANNEL}/runtime/lib/model_serving_adapter.so && cp ${CHECKPOINT}/tokenizer.json ${API_CHANNEL}/runtime/tokenizer/tokenizer.json && echo ${sha}-tp4 > ${API_CHANNEL}/SOURCE_COMMIT"
	scp -q "$HERE/build/api.tp4.model_resident.json" "${HUB}:${API_CHANNEL}/model_resident.json"
	tmp=$(mktemp)
	cat > "$tmp" <<UNIT
[Unit]
Description=SparkPipe Qwen3.8-27B API (internal use only; TP4 on ${HOSTS[*]}, lane ${LANE})
After=network.target
[Service]
WorkingDirectory=${API_CHANNEL}
ExecStart=${API_CHANNEL}/bin/sparkpipe_model_api --deployment ${API_CHANNEL}/model_resident.json --runtime-root ${API_CHANNEL}/runtime --port ${API_PORT}
Restart=no
TimeoutStopSec=30
MemoryMax=2G
MemorySwapMax=0
StandardOutput=append:${API_CHANNEL}/api.log
StandardError=append:${API_CHANNEL}/api.log
[Install]
WantedBy=default.target
UNIT
	scp -q "$tmp" "${HUB}:.config/systemd/user/qwen27b-api.service"
	rm "$tmp"
	on "$HUB" "cd ${API_CHANNEL} && sha256sum bin/sparkpipe_model_api runtime/lib/model_serving_adapter.so runtime/tokenizer/tokenizer.json model_resident.json SOURCE_COMMIT > SHA256SUMS && cat SHA256SUMS"
}

case "$ACTION" in
	build) build "${2:-}" ;;
	publish) publish "${2:-}" ;;
	install) install "${2:-}" ;;
	configs) configs ;;
	launch) launch "${2:-}" ;;
	status) status ;;
	stop) stop ;;
	reclaim) reclaim ;;
	fresh) fresh "${2:-}" ;;
	refcheck) refcheck "${2:-}" "${3:-}" ;;
	repeat) repeat_case "${2:-}" "${3:-}" "${4:-}" "${5:-}" ;;
	perf) perf "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-128}" ;;
	api-install) api_install "${2:-}" ;;
	api-start) on "$HUB" "systemctl --user daemon-reload && systemctl --user start qwen27b-api && sleep 2 && systemctl --user is-active qwen27b-api && curl -s --max-time 5 http://127.0.0.1:${API_PORT}/health; echo" ;;
	api-stop) on "$HUB" "systemctl --user stop qwen27b-api; systemctl --user is-active qwen27b-api || true" ;;
	smoke) on "$HUB" "curl -s --max-time 300 http://127.0.0.1:${API_PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"prompt\":\"<|im_start|>user\\nWhat is the capital of France?<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n\",\"max_tokens\":32,\"temperature\":0}'; echo" ;;
	compsec) on "$HUB" "cd ~/api-build-qwen27b-${2:?SHA} && python3 tools/compsec17_chat.py --endpoint http://127.0.0.1:${API_PORT} --template qwen --thinking off --max-tokens ${4:?TOK} --model-tokenizer ${API_CHANNEL}/runtime/tokenizer/tokenizer.json --out ~/qwen27b-runs/${3:?RUN_ID}/compsec17 2>&1 | tail -24" ;;
	*) usage ;;
esac
