#!/usr/bin/env bash
set -euo pipefail

NODE=spark8
LANE=3
ROOT=/home/${NODE}/sparkdata/qwen27b.fp8.tp1
PACK_REL=packs/qwen38-mx2.tp1.qwen36sp
DRAFTER=${ROOT}/drafter/qwen38-dflash2-drafter.qwen36sp
WEIGHTD_SOCKET=/tmp/spark_weightd.sock
CONTROL_PORT=$((23000 + 16 * LANE))
TRANSPORT_PORT_BASE=$((64000 + 16 * LANE))
SEQUENCES=8
MAX_POSITIONS=8192
KV_LOGICAL_PAGES=256
KV_PHYSICAL_PAGES=64
DRAFT_COUNT=8
BUILD_HOST=sparkf
BUILD_BASE=/home/${BUILD_HOST}/build-qwen27b
NODE_BUILD_BASE=/home/${NODE}/build-qwen27b
HUB=rtx5090
API_PORT=8435
API_CHANNEL=/home/spec/qwen27b-api-channel
CHECKPOINT=/mnt/model-warm/qwen3.8-27b-fp8
EOS_TOKEN_IDS=248046,248044
TOKENIZER_VOCABULARY_SIZE=248077
HERE=$(cd "$(dirname "$0")/.." && pwd)

usage() {
	cat >&2 <<USAGE
usage: $0 ACTION [ARG]
  build SHA               archive SHA to ${BUILD_HOST}:${BUILD_BASE}/src-SHA; host binaries, TP1 adapter, module archive, weightd tests
  publish SHA             copy that tree to ${NODE}; module GPU validation attached to ${WEIGHTD_SOCKET}; driver compile; install into ${ROOT}
  configs                 write ${ROOT}/config (deployments, adapter configs) and build/api.model_resident.json
  launch nospec|mtp|dflash2 residentd in unit sp-qwen27b-residentd (control ${CONTROL_PORT}, transport ${TRANSPORT_PORT_BASE}, lane ${LANE})
  status | stop           unit state, ready line, memory | stop that unit
  bench nospec|dflash2    canonical 128-token prompt, 512 greedy tokens through sparkpipe_model_batch
  refcheck MODE FILE      greedy continuations for a reference prompt file (fresh residentd per prompt), compared with its reference_token_ids
  perf MODE CASES REPS OUT [ROWS] tools/qwen38_27b_tp1_bench.py cases (o128,o512,code512,rep512,ttft,streams8), fresh residentd per run, JSON lines to OUT
  fresh MODE              stop, then launch MODE (residentd serves one client connection; see lane notes)
  api-install SHA         build the x86 API + TP1 adapter on ${HUB}, stage ${API_CHANNEL}, write qwen27b-api.service (not started)
  api-start | api-stop | smoke
  compsec SHA RUN_ID      COMPSEC-17 (qwen template, thinking off, 512 tokens) from ${HUB} against :${API_PORT}
USAGE
	exit 2
}
ACTION="${1:-}"
[ -n "$ACTION" ] || usage
on() { ssh -o BatchMode=yes "$@"; }

build() {
	local sha=${1:?SHA}
	git -C "$HERE" archive --format=tar --prefix="src-${sha}/" "$sha" | on "$BUILD_HOST" "mkdir -p ${BUILD_BASE} && tar -xf - -C ${BUILD_BASE}"
	on "$BUILD_HOST" "cd ${BUILD_BASE}/src-${sha} && systemd-run --user --wait --collect --unit=sp-qwen27b-build-${sha} --same-dir -p StandardOutput=file:${BUILD_BASE}/src-${sha}/build-qwen27b.log -p StandardError=file:${BUILD_BASE}/src-${sha}/build-qwen27b.log bash -c '
export PATH=/usr/local/cuda/bin:\$PATH
set -e
make -j8 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a build/sparkpipe_weightd build/weightd_warm build/sparkpipe_model_residentd build/sparkpipe_model_batch build/sparkpipe_model_compile build/sparkpipe_module_publish build/sparkpipe_driver_inspect hidden_transport_spark_host_rdma_verbs
make -j8 build/test_stage_module_weightd build/test_weightd_attach build/test_weightd_map
./build/test_stage_module_weightd && ./build/test_weightd_attach && ./build/test_weightd_map
make -j8 QWEN38_27B_SERVING_TOPOLOGY_FLAGS=-DSPARK_QWEN38_27B_SERVING_TP_DEGREE=1u build/libqwen38_27b_serving_adapter.so
make -j8 -C modules/qwen38_27b_resident_decode_stage CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a archive
' > build-unit.log 2>&1; echo build_rc=\$?; grep -a -E 'ALL PASS|green|Error' build-qwen27b.log | tail -4"
}

publish_env() {
	printf '%s' "ALLOW_UNQUALIFIED_EXECUTION=1 TP_DEGREE=1 TP_RANK=0 TP_STANDALONE=1 STAGE_COUNT=1 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=64 MTP_LAYER_COUNT=1 GDN_SNAPSHOT_SLOT_COUNT=16 MAX_ACTIVE_SEQUENCES=8 KV_BLOCK_COUNT=8 STAGE_PACK_PATH=${ROOT}/${PACK_REL}"
}

publish() {
	local sha=${1:?SHA} src=${NODE_BUILD_BASE}/src-${1}
	on "$NODE" "mkdir -p ${NODE_BUILD_BASE}"
	on "$BUILD_HOST" "rsync -a ${BUILD_BASE}/src-${sha} ${NODE}:${NODE_BUILD_BASE}/"
	on "$NODE" "test -S ${WEIGHTD_SOCKET} || { echo 'weightd socket ${WEIGHTD_SOCKET} missing on ${NODE}'; exit 2; }
cd ${src} && digest=\$(cut -d' ' -f1 ${ROOT}/${PACK_REL}.sha256) && systemd-run --user --wait --collect --unit=sp-qwen27b-publish --same-dir -p StandardOutput=file:${src}/publish-unit.log -p StandardError=file:${src}/publish-unit.log -E SPARK_WEIGHTD_ATTACH=1 -E SPARK_WEIGHTD_SOCKET=${WEIGHTD_SOCKET} -E SPARK_WEIGHTD_PACK_SHA256=\$digest -E SPARK_WEIGHTD_LANE=${LANE} bash -c '
export PATH=/usr/local/cuda/bin:\$PATH
set -e
make -C modules/qwen38_27b_resident_decode_stage CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a $(publish_env) publish > publish.log 2>&1
grep -q -E \"qwen38_27b_validation PASS|validation=reused\" publish.log
rm -rf build/tp1-driver
build/sparkpipe_model_compile --model examples/model_descriptions/qwen38_27b_resident_decode_stage_firmware.json --library build/module_library --output build/tp1-driver --cc /usr/bin/cc --include include --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl --cc-arg -pthread > driver-link.log 2>&1
build/sparkpipe_driver_inspect build/tp1-driver/stages/stage_000/model_driver.so cuda.sm121.qwen38_27b.resident_decode_stage.bf16 > driver-inspect.log 2>&1
! ldd -r build/tp1-driver/stages/stage_000/model_driver.so | grep -E \"not found|undefined symbol\"
'; echo publish_rc=\$?; grep -a -E 'validation (PASS|FAIL)|validation=' publish.log | tail -3
mkdir -p ${ROOT}/bin ${ROOT}/lib ${ROOT}/stages ${ROOT}/logs
for f in sparkpipe_model_residentd sparkpipe_model_batch; do cp build/\$f ${ROOT}/bin/.\$f.tmp && mv ${ROOT}/bin/.\$f.tmp ${ROOT}/bin/\$f; done
cp build/libhidden_transport_spark_host_rdma_verbs.so ${ROOT}/lib/.ht.tmp && mv ${ROOT}/lib/.ht.tmp ${ROOT}/lib/hidden_transport.so
cp build/libqwen38_27b_serving_adapter.so ${ROOT}/lib/.ad.tmp && mv ${ROOT}/lib/.ad.tmp ${ROOT}/lib/model_serving_adapter.so
rm -rf ${ROOT}/stages/stage_000 && cp -a build/tp1-driver/stages/stage_000 ${ROOT}/stages/
echo ${sha} > ${ROOT}/SOURCE_COMMIT
cd ${ROOT} && find bin lib stages -type f | sort | xargs sha256sum > SHA256SUMS && cat SHA256SUMS"
}

configs() {
	local tmp
	tmp=$(mktemp -d)
	on "$HUB" "cat ${CHECKPOINT}/tokenizer.json" > "$tmp/tokenizer.json"
	python3 "$HERE/tools/qwen38_27b_tp1_deployment.py" --host "$NODE" --runtime-root "$ROOT" \
		--stage-pack "$PACK_REL" --weightd-socket "$WEIGHTD_SOCKET" --control-port "$CONTROL_PORT" \
		--transport-port-base "$TRANSPORT_PORT_BASE" --sequences "$SEQUENCES" --max-positions "$MAX_POSITIONS" \
		--kv-logical-pages "$KV_LOGICAL_PAGES" --kv-physical-pages "$KV_PHYSICAL_PAGES" --draft-count "$DRAFT_COUNT" \
		--eos-token-ids "$EOS_TOKEN_IDS" --tokenizer "$tmp/tokenizer.json" \
		--tokenizer-vocabulary-size "$TOKENIZER_VOCABULARY_SIZE" --output "$tmp/config"
	on "$NODE" "mkdir -p ${ROOT}/config"
	scp -q "$tmp"/config/*.json "${NODE}:${ROOT}/config/"
	mkdir -p "$HERE/build"
	cp "$tmp/config/api.model_resident.json" "$HERE/build/"
	on "$NODE" "ls -la ${ROOT}/config"
	rm -r "$tmp"
}

launch() {
	local mode=${1:?nospec|dflash2} spec_env=""
	case "$mode" in
		nospec) spec_env="-E SPARK_QWEN38_27B_SPECULATORS=0" ;;
		mtp) spec_env="-E SPARK_QWEN38_27B_SPECULATORS=0x1" ;;
		dflash2) spec_env="-E SPARK_QWEN38_27B_SPECULATORS=0x4 -E SPARK_QWEN38_27B_DSPARK_PACK_PATH=${DRAFTER} -E SPARK_QWEN38_27B_DFLASH2_STATE_SELECT=1 -E SPARK_QWEN38_27B_DFLASH2_BONUS_FOLD=2 -E SPARK_QWEN38_27B_DFLASH2_BLOCK_KV=0 -E SPARK_QWEN38_27B_DFLASH2_WINDOW=2048 -E SPARK_QWEN38_27B_DFLASH2_CTX_CACHE=1" ;;
		*) echo "mode must be nospec, mtp or dflash2" >&2; exit 2 ;;
	esac
	local deployment=${mode}
	[ "$mode" = mtp ] && deployment=nospec
	on "$NODE" "set -e
systemctl --user is-active --quiet sp-qwen27b-residentd && { echo 'sp-qwen27b-residentd already running'; exit 2; }
test -S ${WEIGHTD_SOCKET} || { echo 'weightd socket missing'; exit 2; }
free -g | head -2
cd ${ROOT} && log=${ROOT}/logs/residentd-${mode}-\$(date -u +%Y%m%dT%H%M%SZ).log && ln -sfn \$log ${ROOT}/logs/current.log && echo ${mode} > ${ROOT}/logs/current.mode
systemd-run --user --collect --unit=sp-qwen27b-residentd --same-dir -p StandardOutput=file:\$log -p StandardError=file:\$log -E LD_LIBRARY_PATH=${ROOT}/lib -E SPARK_WEIGHTD_LANE=${LANE} -E CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0 ${spec_env} ${QWEN27B_DIAG_ENV:-} ${ROOT}/bin/sparkpipe_model_residentd --deployment ${ROOT}/config/model_resident.${deployment}.json --rank-index 0
for i in \$(seq 1 300); do grep -aq 'model_residentd ready' \$log && break; systemctl --user is-active --quiet sp-qwen27b-residentd || break; sleep 1; done
grep -a -E 'ready|refused|failed|ERRSITE|capacity' \$log | tail -8
free -g | head -2"
}

status() { on "$NODE" "systemctl --user status sp-qwen27b-residentd --no-pager | head -5; grep -a -E 'ready' ${ROOT}/logs/current.log | tail -2; free -g | head -2"; }
stop() { on "$NODE" "systemctl --user stop sp-qwen27b-residentd; systemctl --user is-active sp-qwen27b-residentd || true"; }

bench() {
	local mode=${1:?nospec|dflash2}
	on "$NODE" "cd ${ROOT} && python3 - > ${ROOT}/logs/o512.json <<'EOF'
import json
PROMPT = [0, 3476, 477, 18068, 260, 3375, 35312, 3417, 16, 38074, 13254, 16, 455, 4087, 3287, 2231, 1605, 270, 21361, 8786, 9045, 16, 128803, 79418, 2317, 566, 8130, 345, 14866, 3312, 2019, 16, 983, 1142, 469, 1142, 554, 6242, 260, 31191, 603, 19905, 418, 270, 4031, 2455, 2562, 1167, 1479, 270, 6074, 15398, 344, 10097, 16, 2052, 270, 15398, 344, 1353, 4521, 538, 260, 2395, 2740, 294, 18885, 6243, 14, 20430, 418, 270, 19904, 50098, 5898, 1789, 638, 1341, 294, 6319, 2562, 3737, 603, 25529, 223, 18, 855, 270, 2019, 344, 7681, 1202, 270, 10844, 22283, 339, 671, 2019, 109029, 260, 716, 15, 10554, 30347, 112566, 1936, 14327, 436, 304, 270, 489, 5927, 7104, 339, 9945, 1137, 9854, 69, 201, 223, 19, 28, 1823, 11006, 334, 30557, 32684, 16617]
print(json.dumps({'schema_version': 1, 'connect_timeout_ms': 30000, 'request_capacity': 2, 'max_context_tokens': 4096, 'max_prefill_rows_per_submission': 8, 'maximum_messages_per_rank_per_progress': 8, 'maximum_new_submissions_per_progress': 2, 'stop_token_ids': [], 'requests': [{'request_id': 760130, 'sequence_id': 760130, 'priority': 0, 'output_token_budget': 512, 'prompt_token_ids': PROMPT}]}))
EOF
t0=\$(date +%s.%N); LD_LIBRARY_PATH=${ROOT}/lib bin/sparkpipe_model_batch --deployment config/model_resident.${mode}.json --runtime-root ${ROOT} --batch ${ROOT}/logs/o512.json > ${ROOT}/logs/o512-${mode}.out 2>${ROOT}/logs/o512-${mode}.err; t1=\$(date +%s.%N)
python3 - ${ROOT}/logs/o512-${mode}.out \$t0 \$t1 <<'EOF'
import json, hashlib, sys
t = [json.loads(l)['token_id'] for l in open(sys.argv[1]) if l.startswith('{') and json.loads(l).get('event') == 'token']
wall = float(sys.argv[3]) - float(sys.argv[2])
print(json.dumps({'tokens': len(t), 'wall_s': round(wall, 2), 'tok_s_end_to_end': round(len(t) / wall, 2), 'stream16': hashlib.sha256(str(t).encode()).hexdigest()[:16]}))
EOF
grep -ac spec_diag ${ROOT}/logs/current.log || true"
}

fresh() {
	stop > /dev/null
	launch "${1:?mode}" | grep -a -E 'model_residentd ready|refused|failed|ERRSITE' | tail -3
}

refcheck() {
	local mode=${1:?nospec|mtp|dflash2} file=${2:?reference prompts json} name
	scp -q "$file" "${NODE}:${ROOT}/logs/refprompts.json"
	for name in $(python3 -c "import json,sys; print(' '.join(p['name'] for p in json.load(open(sys.argv[1]))['prompts']))" "$file"); do
		fresh "$mode" > /dev/null
		on "$NODE" "cd ${ROOT} && python3 - ${name} ${mode} <<'EOF'
import json, subprocess, os, sys
name, mode = sys.argv[1], sys.argv[2]
p = [q for q in json.load(open('logs/refprompts.json'))['prompts'] if q['name'] == name][0]
env = dict(os.environ, LD_LIBRARY_PATH='${ROOT}/lib')
batch = {'schema_version': 1, 'connect_timeout_ms': 30000, 'request_capacity': 2, 'max_context_tokens': 4096, 'max_prefill_rows_per_submission': 8, 'maximum_messages_per_rank_per_progress': 8, 'maximum_new_submissions_per_progress': 2, 'stop_token_ids': [], 'requests': [{'request_id': 900, 'sequence_id': 900, 'priority': 0, 'output_token_budget': p['new_tokens'], 'prompt_token_ids': p['prompt_token_ids']}]}
json.dump(batch, open('logs/ref-%s.json' % name, 'w'))
deployment = 'dflash2' if mode == 'dflash2' else 'nospec'
run = subprocess.run(['bin/sparkpipe_model_batch', '--deployment', 'config/model_resident.%s.json' % deployment, '--runtime-root', '${ROOT}', '--batch', 'logs/ref-%s.json' % name], capture_output=True, text=True, env=env)
got = [json.loads(l)['token_id'] for l in run.stdout.splitlines() if l.startswith('{') and json.loads(l).get('event') == 'token']
want = p.get('reference_token_ids', [])
same = 0
while same < min(len(got), len(want)) and got[same] == want[same]:
    same += 1
print(json.dumps({'mode': mode, 'prompt': name, 'engine': got, 'reference': want, 'matching_prefix': same, 'exact': got == want, 'batch_rc': run.returncode, 'stderr_tail': run.stderr.strip().splitlines()[-1:]}))
EOF"
	done
}

perf() {
	local mode=${1:?nospec|mtp|dflash2} cases=${2:?comma list} reps=${3:?reps} out=${4:?local jsonl} prefill_rows=${5:-128} case rep deployment=nospec
	[ "$mode" = dflash2 ] && deployment=dflash2
	scp -q "$HERE/tools/qwen38_27b_tp1_bench.py" "${NODE}:${ROOT}/logs/qwen38_27b_tp1_bench.py"
	scp -q "$HERE/qualification/qwen38_27b/bench_prompts_tp1.json" "${NODE}:${ROOT}/logs/bench_prompts_tp1.json"
	for case in ${cases//,/ }; do
		for rep in $(seq 1 "$reps"); do
			fresh "$mode" > /dev/null 2>&1
			on "$NODE" "cd ${ROOT} && python3 logs/qwen38_27b_tp1_bench.py ${case} --root ${ROOT} --deployment ${ROOT}/config/model_resident.${deployment}.json --prompts logs/bench_prompts_tp1.json --prefill-rows ${prefill_rows} | python3 -c '
import json, sys, re
r = json.loads(sys.stdin.read())
acc = [int(m.group(1)) for m in re.finditer(r\"qwen38_27b_spec accepted=(\\d+)\", open(\"${ROOT}/logs/current.log\", errors=\"replace\").read())]
r.update(mode=\"${mode}\", rep=${rep}, prefill_rows=${prefill_rows}, source_commit=open(\"${ROOT}/SOURCE_COMMIT\").read().strip(), spec_rounds=len(acc), mean_accepted_drafts=round(sum(acc) / len(acc), 3) if acc else None, mean_tokens_per_round=round(sum(a + 1 for a in acc) / len(acc), 3) if acc else None)
print(json.dumps(r))
'" | tee -a "$out"
		done
	done
	stop > /dev/null 2>&1
}

api_install() {
	local sha=${1:?SHA} tmp
	test -f "$HERE/build/api.model_resident.json" || { echo "run configs first" >&2; exit 2; }
	git -C "$HERE" archive --format=tar --prefix="api-build-qwen27b-${sha}/" "$sha" | on "$HUB" "tar -xf - -C ~"
	on "$HUB" "cd ~/api-build-qwen27b-${sha} && export PATH=/usr/local/cuda/bin:\$PATH && make -j6 build/sparkpipe_model_api > build-api.log 2>&1 && make -j6 QWEN38_27B_SERVING_TOPOLOGY_FLAGS=-DSPARK_QWEN38_27B_SERVING_TP_DEGREE=1u build/libqwen38_27b_serving_adapter.so > build-adapter.log 2>&1 && ldd -r build/libqwen38_27b_serving_adapter.so | grep -E 'not found|undefined' ; mkdir -p ${API_CHANNEL}/bin ${API_CHANNEL}/runtime/lib ${API_CHANNEL}/runtime/tokenizer && cp build/sparkpipe_model_api ${API_CHANNEL}/bin/.api.tmp && mv ${API_CHANNEL}/bin/.api.tmp ${API_CHANNEL}/bin/sparkpipe_model_api && cp build/libqwen38_27b_serving_adapter.so ${API_CHANNEL}/runtime/lib/.ad.tmp && mv ${API_CHANNEL}/runtime/lib/.ad.tmp ${API_CHANNEL}/runtime/lib/model_serving_adapter.so && cp ${CHECKPOINT}/tokenizer.json ${API_CHANNEL}/runtime/tokenizer/tokenizer.json && echo ${sha} > ${API_CHANNEL}/SOURCE_COMMIT"
	scp -q "$HERE/build/api.model_resident.json" "${HUB}:${API_CHANNEL}/model_resident.json"
	tmp=$(mktemp)
	cat > "$tmp" <<UNIT
[Unit]
Description=SparkPipe Qwen3.8-27B API (internal use only; TP1 on ${NODE}, lane ${LANE})
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
	configs) configs ;;
	launch) launch "${2:-}" ;;
	status) status ;;
	stop) stop ;;
	bench) bench "${2:-}" ;;
	refcheck) refcheck "${2:-}" "${3:-}" ;;
	fresh) fresh "${2:-}" ;;
	perf) perf "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-128}" ;;
	api-install) api_install "${2:-}" ;;
	api-start) on "$HUB" "systemctl --user daemon-reload && systemctl --user start qwen27b-api && sleep 2 && systemctl --user is-active qwen27b-api && curl -s --max-time 5 http://127.0.0.1:${API_PORT}/health; echo" ;;
	api-stop) on "$HUB" "systemctl --user stop qwen27b-api; systemctl --user is-active qwen27b-api || true" ;;
	smoke) on "$HUB" "curl -s --max-time 300 http://127.0.0.1:${API_PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"prompt\":\"<|im_start|>user\\nWhat is the capital of France?<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n\",\"max_tokens\":32,\"temperature\":0}'; echo" ;;
	compsec) on "$HUB" "cd ~/api-build-qwen27b-${2:?SHA} && python3 tools/compsec17_chat.py --endpoint http://127.0.0.1:${API_PORT} --template qwen --thinking off --model-tokenizer ${API_CHANNEL}/runtime/tokenizer/tokenizer.json --out ~/qwen27b-runs/${3:?RUN_ID}/compsec17 2>&1 | tail -20" ;;
	*) usage ;;
esac
