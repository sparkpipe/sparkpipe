#!/bin/sh
# Q3F-T1 orchestrator: build on spark7, ship to 8 TP8 hosts, hold the mesh
# lease, run the decode wave per prompt, harvest rank dumps, pack, compare.
# Laws: poll never blind-sleep, sparkcap on every run, shared fleet weightsd
# only, fail loud, no rerun-until-pass.
set -eu

REPO=/Users/mac/q3ft1
BUILD_NODE=spark7
LOCAL_SCRATCH="$REPO/.rt"
REMOTE_RT='$HOME/q3ft1_rt'
REMOTE_STAGE='$HOME/q3ft1_rt_stage'
PROMPTS="$REPO/qualification/t1_reference/qwen4_flash/prompts_qwen4flash.json"
PORT_BASE=19456
TP_IDENTIFIER=194561
HOSTS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7"
EXPERT_POOL_BYTES=17179869184
SPINE_BUDGET_BYTES=34359738368
WEIGHTD_SOCKET=/run/sparkpipe-weightsd/weightsd.sock
EXPORT_ROOT=/Users/mac/q3ft1/runs/q3f-t1

say() {
	printf '[t1q3f %s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

rank_host() {
	echo $HOSTS | cut -d' ' -f$(($1 + 1))
}

prompt_field() {
	python3 -c "
import json
prompts = json.load(open('$PROMPTS'))['prompts']
value = prompts[$1]['$2']
if isinstance(value, list):
    print(','.join(str(int(v)) for v in value))
else:
    print(value)"
}

write_rank_env() {
	ports=$(python3 -c "
base = $PORT_BASE
tp = 8
print(','.join(str(0 if a == b else base + a * tp + b) for a in range(tp) for b in range(tp)))")
	{
		echo "PROMPT_IDS=$2"
		echo "NEW_TOKENS=$3"
		echo "TIMING=${TIMING:-0}"
		echo "PORT_BASE=$PORT_BASE"
		echo "TP_IDENTIFIER=$TP_IDENTIFIER"
		echo "TP_HOSTS=\"$(echo $HOSTS | tr ' ' ',')\""
		echo "LOCAL_HOST=$(rank_host "$1")"
		echo "EXPERT_POOL_BYTES=$EXPERT_POOL_BYTES"
		echo "SPINE_BUDGET_BYTES=$SPINE_BUDGET_BYTES"
		echo "WEIGHTD_SOCKET=$WEIGHTD_SOCKET"
		echo "SESSION_PORTS=\"$ports\""
	} > "$LOCAL_SCRATCH/rank$1.env"
}

wave_prompt() {
	prompt_name=$1
	prompt_ids=$2
	new_tokens=$3
	pids=""
	for rank in 0 1 2 3 4 5 6 7; do
		host=$(rank_host "$rank")
		say "launch rank $rank on $host ($prompt_name)"
		ssh -o BatchMode=yes "$host" "rm -rf $REMOTE_RT/dump && mkdir -p $REMOTE_RT/dump"
		write_rank_env "$rank" "$prompt_ids" "$new_tokens"
		scp -q "$LOCAL_SCRATCH/rank$rank.env" "$host:q3ft1_rt/rank$rank.env"
		ssh -o BatchMode=yes "$host" "sh $REMOTE_RT/t1_q3f_rank.sh $rank $REMOTE_RT/rank$rank.env" > "$EXPORT_ROOT/$prompt_name-rank$rank.log" 2>&1 &
		pids="$pids $!"
	done
	fail=0
	for pid in $pids; do
		wait "$pid" || fail=1
	done
	[ "$fail" = 0 ] || { say "WAVE INCOMPLETE for $prompt_name - inspect $EXPORT_ROOT/$prompt_name-rank*.log"; return 1; }
	say "8/8 ranks done ($prompt_name)"
	mkdir -p "$EXPORT_ROOT/$prompt_name-dump"
		for rank in 0 1 2 3 4 5 6 7; do
		host=$(rank_host "$rank")
		scp -q -r "$host:q3ft1_rt/dump/." "$EXPORT_ROOT/$prompt_name-dump/" 2>/dev/null || true
	done
	say "$prompt_name dump files: $(ls "$EXPORT_ROOT/$prompt_name-dump" | wc -l | tr -d ' ')"
}

poll_and_take_lease() {
	while ! grep -q "Q3F-T1" /Users/mac/sparkpipe-coord/MESH_LEASE; do
		printf '%s\n' "HELD | Q3F-T1 | /Users/mac/q3ft1 (lane/q3f-t1) | $(date -u +%Y-%m-%dT%H:%M:%SZ) | qwen3flash TP8 T1 decode on fp8.tp8 via the shared weightsd; co-tenancy respected; slow-but-reliable collectives; T1 wave window then release | (continues)" >> /Users/mac/sparkpipe-coord/MESH_LEASE
		say "mesh lease HELD claimed"
	done
	say "lease present:"
	grep "Q3F-T1" /Users/mac/sparkpipe-coord/MESH_LEASE | tail -1
}

release_lease() {
	sed -i '' "s/^HELD | Q3F-T1/RELEASED | Q3F-T1/" /Users/mac/sparkpipe-coord/MESH_LEASE
	say "lease released"
}

stage_build() {
	rm -rf "$LOCAL_SCRATCH"
	mkdir -p "$LOCAL_SCRATCH"
	ssh "$BUILD_NODE" "rm -rf $REMOTE_STAGE $REMOTE_RT && mkdir -p $REMOTE_STAGE $REMOTE_RT"
	git -C "$REPO" bundle create "$LOCAL_SCRATCH/q3ft1.bundle" lane/q3f-t1
	scp -q "$LOCAL_SCRATCH/q3ft1.bundle" "$BUILD_NODE:q3ft1_rt_stage/"
	ssh "$BUILD_NODE" "rm -rf $REMOTE_STAGE/tree && git clone -q -b lane/q3f-t1 $REMOTE_STAGE/q3ft1.bundle $REMOTE_STAGE/tree"
	say "building on $BUILD_NODE"
	scp -q "$REPO/tools/t1_q3f_build.sh" "$BUILD_NODE:q3ft1_rt_stage/"
	ssh "$BUILD_NODE" "sh $REMOTE_STAGE/t1_q3f_build.sh"
	say "build green on $BUILD_NODE"
	for rank in 0 1 2 3 4 5 6 7; do
		host=$(rank_host "$rank")
		ssh "$host" "rm -rf $REMOTE_RT && mkdir -p $REMOTE_RT"
		scp -q "$BUILD_NODE:q3ft1_rt_stage/t1_q3f_harness" "$BUILD_NODE:q3ft1_rt_stage/libhidden_transport_spark_host_rdma_verbs.so" "$REPO/tools/t1_q3f_rank.sh" "$host:q3ft1_rt/"
	done
	say "staged artifacts on 8 hosts"
}

compare_prompt() {
	prompt_name=$1
	prompt_ids=$2
	new_tokens=$3
	fixture="$REPO/qualification/t1_reference/qwen4_flash/$prompt_name.t1r"
	python3 "$REPO/tools/t1_q3f_pack.py" \
		--reference "$fixture" \
		--dump-dir "$EXPORT_ROOT/$prompt_name-dump" \
		--prompt-ids "$prompt_ids" --new-tokens "$new_tokens" \
		--output "$EXPORT_ROOT/qwen4flash_driver_$prompt_name.t1r"
	set +e
	python3 "$REPO/tools/t1_reference_compare.py" compare \
		--reference "$fixture" \
		--candidate "$EXPORT_ROOT/qwen4flash_driver_$prompt_name.t1r"
	compare_rc=$?
	set -e
	say "$prompt_name compare exit=$compare_rc (0 = PASS, 1 = FAIL reported honestly)"
	return "$compare_rc"
}

negative_control() {
	fixture="$REPO/qualification/t1_reference/qwen4_flash/$(prompt_field 0 name).t1r"
	python3 "$REPO/tools/t1_reference_compare.py" corrupt-fixture \
		--source "$fixture" --target "$EXPORT_ROOT/negative.t1r" \
		--array pos0000_layer0000_streams --offset 9
	set +e
	python3 "$REPO/tools/t1_reference_compare.py" compare \
		--reference "$fixture" --candidate "$EXPORT_ROOT/negative.t1r"
	rc=$?
	set -e
	if [ "$rc" = 0 ]; then
		say "NEGATIVE CONTROL FAILED TO CONVICT"
		exit 1
	fi
	say "negative control convicted (expected)"
}

case "${1:-}" in
build) stage_build ;;
wave)
	poll_and_take_lease
	wave_rc=0
	set +e
	wave_prompt "$(prompt_field 0 name)" "$(prompt_field 0 prompt_token_ids)" "$(prompt_field 0 new_tokens)" || wave_rc=1
	wave_prompt "$(prompt_field 1 name)" "$(prompt_field 1 prompt_token_ids)" "$(prompt_field 1 new_tokens)" || wave_rc=1
	set -e
	release_lease
	say "wave rc=$wave_rc"
	[ "$wave_rc" = 0 ] || exit "$wave_rc"
	;;
compare)
	compare_rc=0
	set +e
	compare_prompt "$(prompt_field 0 name)" "$(prompt_field 0 prompt_token_ids)" "$(prompt_field 0 new_tokens)" || compare_rc=$?
	compare_prompt "$(prompt_field 1 name)" "$(prompt_field 1 prompt_token_ids)" "$(prompt_field 1 new_tokens)" || compare_rc=$?
	negative_control
	set -e
	exit "$compare_rc"
	;;
*)
	echo "usage: $0 build|wave|compare" >&2
	exit 2
	;;
esac
