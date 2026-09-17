#!/bin/sh
# T1-QMAX orchestrator: build on spark7, ship to 16 hosts, take the mesh lease,
# run the decode wave, harvest rank0 dumps, pack, compare, rates, negative control.
# Laws: poll never blind-sleep, sparkcap on every pack/compute step, fail loud.
set -eu

REPO=/Users/mac/t1qmaxw5
BUILD_NODE=spark7
SCRATCH=/tmp/t1qmax
STAGE_DIR=/tmp/t1qmax_stage
RUNTIME_PACKS='$HOME/sparkdata/qwenmax.nvfp4.tp16/packs'
PROMPT_IDS="${PROMPT_IDS:-760,6511,314,9338,369}"
NEW_TOKENS="${NEW_TOKENS:-2}"
PORT_BASE=21504
TP_IDENTIFIER=2141723
HOSTS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
EXPERT_POOL_BYTES=8589934592
SPINE_BUDGET_BYTES=17179869184
EXPORT_ROOT=/Users/mac/t1qmaxw5/runs/t1qmax
REVISION=d2dc35658bcf77e66643428cb52e774cc3b5bd29

say() {
	printf '[t1qmax %s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

rank_host() {
	echo $HOSTS | cut -d' ' -f$(($1 + 1))
}

write_rank_env() {
	ports=$(python3 -c "
base = $PORT_BASE
tp = 16
print(','.join(str(0 if a == b else base + a * tp + b) for a in range(tp) for b in range(tp)))")
	{
		echo "PROMPT_IDS=$PROMPT_IDS"
		echo "NEW_TOKENS=$NEW_TOKENS"
		echo "TIMING=${TIMING:-0}"
		echo "PORT_BASE=$PORT_BASE"
		echo "TP_IDENTIFIER=$TP_IDENTIFIER"
		echo "TP_HOSTS=\"$(echo $HOSTS | tr ' ' ',')\""
		echo "LOCAL_HOST=$(rank_host "$1")"
		echo "EXPERT_POOL_BYTES=$EXPERT_POOL_BYTES"
		echo "SPINE_BUDGET_BYTES=$SPINE_BUDGET_BYTES"
		echo "WEIGHTD_SOCKET=$WEIGHTD_SOCKET"
		echo "SESSION_PORTS=\"$ports\""
	} > "$SCRATCH/rank$1.env"
}

lease_state() {
	grep -E "^(ACTIVE|HELD-FIRST|QUEUED)" /Users/mac/sparkpipe-coord/MESH_LEASE || true
}

lease_take() {
	sed -i '' 's/^QUEUED | T1-QMAX/ACTIVE | T1-QMAX/' /Users/mac/sparkpipe-coord/MESH_LEASE
}

lease_release() {
	sed -i '' 's/^ACTIVE | T1-QMAX/RELEASED | T1-QMAX/' /Users/mac/sparkpipe-coord/MESH_LEASE
}

stage_build() {
	say "bundling lane/t1-qmax-wave4 for $BUILD_NODE"
	rm -rf "$SCRATCH"
	mkdir -p "$SCRATCH"
	ssh "$BUILD_NODE" "rm -rf $STAGE_DIR && mkdir -p $STAGE_DIR $SCRATCH"
	git -C "$REPO" bundle create "$SCRATCH/t1qmax.bundle" lane/t1-qmax-wave4
	scp -q "$SCRATCH/t1qmax.bundle" "$BUILD_NODE:$STAGE_DIR/"
	ssh "$BUILD_NODE" "rm -rf $STAGE_DIR/tree && git clone -q -b lane/t1-qmax-wave4 $STAGE_DIR/t1qmax.bundle $STAGE_DIR/tree"
	say "building on $BUILD_NODE"
	scp -q "$REPO/tools/t1_qmax_build.sh" "$BUILD_NODE:$STAGE_DIR/"
	ssh "$BUILD_NODE" "sh $STAGE_DIR/t1_qmax_build.sh"
	say "build green on $BUILD_NODE"
}

stage_ship() {
	for rank in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
		host=$(rank_host "$rank")
		ssh "$host" "rm -rf $SCRATCH && mkdir -p $SCRATCH"
		write_rank_env "$rank"
		scp -q "$BUILD_NODE:$STAGE_DIR/t1_qmax_harness" "$BUILD_NODE:$STAGE_DIR/libhidden_transport_spark_host_rdma_verbs.so" \
			"$BUILD_NODE:$STAGE_DIR/sparkpipe_weightd" "$REPO/tools/t1_qmax_rank.sh" "$SCRATCH/rank$rank.env" "$host:$SCRATCH/"
	done
	say "staged artifacts on 16 hosts"
}

wave() {
	pids=""
	for rank in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
		host=$(rank_host "$rank")
		say "launch rank $rank on $host"
		ssh -o BatchMode=yes "$host" "sh $SCRATCH/t1_qmax_rank.sh $rank" > "$EXPORT_ROOT/rank$rank.log" 2>&1 &
		pids="$pids $!"
	done
	fail=0
	for pid in $pids; do
		wait "$pid" || fail=1
	done
	if [ "$fail" != 0 ]; then
		say "WAVE INCOMPLETE - inspect $EXPORT_ROOT/rank*.log"
		return 1
	fi
	say "16/16 ranks done"
}

harvest_compare() {
	say "harvesting rank0 dumps + logs"
	mkdir -p "$EXPORT_ROOT/dump"
	scp -q -r "spark0:$SCRATCH/dump/." "$EXPORT_ROOT/dump/"
	scp -q "spark0:$SCRATCH/harness.log" "$EXPORT_ROOT/harness-rank0.log"
	say "dump files: $(ls "$EXPORT_ROOT/dump" | wc -l | tr -d ' ')"
	say "packing candidate T1R1"
	python3 "$REPO/tools/t1_qmax_pack.py" \
		--reference "$REPO/qualification/t1_reference/qwen38_max/capital_of_france.t1r" \
		--dump-dir "$EXPORT_ROOT/dump" \
		--prompt-ids "$PROMPT_IDS" --new-tokens "$NEW_TOKENS" \
		--output "$EXPORT_ROOT/qwen38max_driver.t1r"
	say "compare (defaults rel=0.02 abs=1e-3)"
	set +e
	python3 "$REPO/tools/t1_reference_compare.py" compare \
		--reference "$REPO/qualification/t1_reference/qwen38_max/capital_of_france.t1r" \
		--candidate "$EXPORT_ROOT/qwen38max_driver.t1r"
	compare_rc=$?
	set -e
	say "per-kind rates"
	python3 "$REPO/tools/t1_qmax_rates.py" \
		--reference "$REPO/qualification/t1_reference/qwen38_max/capital_of_france.t1r" \
		--candidate "$EXPORT_ROOT/qwen38max_driver.t1r"
	say "negative control"
	python3 "$REPO/tools/t1_reference_compare.py" corrupt-fixture \
		--source "$REPO/qualification/t1_reference/qwen38_max/capital_of_france.t1r" \
		--target "$EXPORT_ROOT/negative.t1r" --array pos0000_layer0000_streams --offset 9
	set +e
	python3 "$REPO/tools/t1_reference_compare.py" compare \
		--reference "$REPO/qualification/t1_reference/qwen38_max/capital_of_france.t1r" \
		--candidate "$EXPORT_ROOT/negative.t1r"
	neg_rc=$?
	set -e
	if [ "$neg_rc" = 0 ]; then
		say "NEGATIVE CONTROL FAILED TO CONVICT"
		return 1
	fi
	say "negative control convicted (expected)"
	say "compare exit=$compare_rc (0 = T1 PASS, 1 = FAIL reported honestly)"
	return "$compare_rc"
}

poll_and_take_lease() {
	lease_taken=0
	while [ "$lease_taken" = 0 ]; do
		if grep -q "^ACTIVE | T1-G53" /Users/mac/sparkpipe-coord/MESH_LEASE; then
			say "mesh ACTIVE by T1-G53; polling every 20s"
			sleep 20
		elif grep -q "^ACTIVE | T1-QMAX" /Users/mac/sparkpipe-coord/MESH_LEASE; then
			lease_taken=1
		elif grep -q "^QUEUED | T1-QMAX" /Users/mac/sparkpipe-coord/MESH_LEASE; then
			lease_take
			lease_taken=1
		else
			say "lease file unexpected state:"; lease_state
			return 1
		fi
	done
	say "lease ACTIVE for T1-QMAX"
}

case "${1:-}" in
build) stage_build ;;
ship) stage_ship ;;
wave) wave ;;
compare) harvest_compare ;;
lease) poll_and_take_lease ;;
release) lease_release ;;
all)
	mkdir -p "$EXPORT_ROOT"
	stage_build
	stage_ship
	poll_and_take_lease
	set +e
	wave
	wave_rc=$?
	set -e
	lease_release
	say "lease released"
	[ "$wave_rc" = 0 ] || exit "$wave_rc"
	harvest_compare
	;;
*)
	echo "usage: $0 build|ship|wave|compare|lease|release|all" >&2
	exit 2
	;;
esac
