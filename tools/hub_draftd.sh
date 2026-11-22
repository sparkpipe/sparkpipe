#!/usr/bin/env bash
set -eu
UNIT=${DRAFTD_UNIT:-sparkpipe-draftd}
LOG=${DRAFTD_LOG:-/srv/workspace/sparkpipe/logs/draftd.log}
DRAFTER=${DRAFTD_DRAFTER:-}
SHARD=${DRAFTD_TARGET_SHARD:-}
RESTART=${DRAFTD_RESTART:-on-failure}
TAP_HOST=${DRAFTD_TAP_HOST:-10.10.250.2}
TAP_PORT=${DRAFTD_TAP_PORT:-7701}
DRAFT_PORT=${DRAFTD_DRAFT_PORT:-7702}
CONTEXT_BYTES=${DRAFTD_CONTEXT_BYTES:-8589934592}
WINDOW=${DRAFTD_WINDOW:-4096}
RECORD_DIR=${DRAFTD_RECORD_DIR:-}
TARGET_EMBED=${DRAFTD_TARGET_EMBED:-}
TARGET_HEAD=${DRAFTD_TARGET_HEAD:-}
TARGET_NORM=${DRAFTD_TARGET_NORM:-}
SERVER=${DRAFTD_SERVER:-block_draftd}
ENGINE_ROOT=${DRAFTD_ENGINE_ROOT:-}
ENGINE_LANE=${DRAFTD_ENGINE_LANE:-}
ENGINE_UNIT=${DRAFTD_ENGINE_UNIT:-sparkpipe-draftd-engine}
WEIGHTD_UNIT=${DRAFTD_WEIGHTD_UNIT:-sparkpipe-weightd-hub}
CUDA_RUNTIME=${DRAFTD_CUDA_RUNTIME:-/usr/local/cuda-13.1/lib64}
listening() { ss -ltn | grep -c "$TAP_HOST:$TAP_PORT \|127.0.0.1:$DRAFT_PORT " || true; }
engine_setting() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$ENGINE_ROOT/config/draftd.json" "$1"; }
engine_wait() { for _ in $(seq 1 "$2"); do [ -S "$1" ] && return 0; sleep 1; done; return 1; }
engine_check() {
	case "$ENGINE_ROOT" in /srv/workspace/*|/home/*) ;; *) echo "hub_draftd: the engine root $ENGINE_ROOT is not on the hub's local NVMe" >&2; exit 1;; esac
	case "$ENGINE_ROOT" in */../*|*/..) echo "hub_draftd: $ENGINE_ROOT climbs out of its directory" >&2; exit 1;; esac
	test -n "$ENGINE_LANE" || { echo "hub_draftd: DRAFTD_ENGINE_LANE names the lane in the engine root's config" >&2; exit 1; }
	for path in bin/sparkpipe_draftd bin/sparkpipe_weightd config/draftd.json; do
		test -e "$ENGINE_ROOT/$path" || { echo "hub_draftd: $ENGINE_ROOT lacks $path; stage the engine root from merged main first" >&2; exit 1; }
	done
	CONTROL=$(engine_setting control_socket)
	WEIGHTD_SOCKET=$(engine_setting weightd_socket)
}
engine_start() {
	engine_check
	mkdir -p "$ENGINE_ROOT/run/kv" "$(dirname "$LOG")" "$(dirname "$CONTROL")" "$(dirname "$WEIGHTD_SOCKET")"
	if ! systemctl --user is-active -q "$WEIGHTD_UNIT"; then
		systemctl --user reset-failed "$WEIGHTD_UNIT" 2>/dev/null || true
		systemd-run --user --unit="$WEIGHTD_UNIT" -p Restart=on-failure -p RestartSec=10 -E LD_LIBRARY_PATH="$CUDA_RUNTIME" \
			-E SPARK_KV_NODE_ROOT="$ENGINE_ROOT/run/kv" --working-directory="$ENGINE_ROOT" bash -c \
			"exec ./bin/sparkpipe_weightd --socket '$WEIGHTD_SOCKET' --device-bytes-max 1073741824 >> '$ENGINE_ROOT/run/weightd.log' 2>&1"
	fi
	engine_wait "$WEIGHTD_SOCKET" 60 || { echo "hub_draftd: weightd did not open $WEIGHTD_SOCKET; see $ENGINE_ROOT/run/weightd.log" >&2; exit 1; }
	if ! systemctl --user is-active -q "$ENGINE_UNIT"; then
		if [ "$(listening)" != 0 ]; then echo "hub_draftd: port $TAP_PORT or $DRAFT_PORT is already in use; stop the lane's block_draftd unit first" >&2; exit 1; fi
		systemctl --user reset-failed "$ENGINE_UNIT" 2>/dev/null || true
		systemd-run --user --unit="$ENGINE_UNIT" -p Restart="$RESTART" -p RestartSec=10 -E LD_LIBRARY_PATH="$CUDA_RUNTIME" \
			--working-directory="$ENGINE_ROOT" bash -c "exec ./bin/sparkpipe_draftd --config config/draftd.json >> '$LOG' 2>&1"
	fi
	engine_wait "$CONTROL" 60 || { echo "hub_draftd: $ENGINE_UNIT did not open $CONTROL; see $LOG" >&2; exit 1; }
	"$ENGINE_ROOT/bin/sparkpipe_draftd" --control "$CONTROL" activate "$ENGINE_LANE"
}
engine_stop() {
	engine_check
	if [ -S "$CONTROL" ]; then
		"$ENGINE_ROOT/bin/sparkpipe_draftd" --control "$CONTROL" deactivate "$ENGINE_LANE" || true
	fi
	echo "hub_draftd: engine lane $ENGINE_LANE deactivated; $ENGINE_UNIT and $WEIGHTD_UNIT keep running"
}
case "$SERVER" in block_draftd|engine) ;; *) echo "hub_draftd: DRAFTD_SERVER is block_draftd or engine, not $SERVER" >&2; exit 1;; esac
case "${1:-}" in
start)
	if [ "$SERVER" = engine ]; then engine_start; exit $?; fi
	SOURCE=${2:?usage: hub_draftd.sh start SOURCE_DIR}
	test -n "$DRAFTER" || { echo "hub_draftd: DRAFTD_DRAFTER is required; start it from its lane record with tools/lane_config.py draftd RECORD" >&2; exit 1; }
	for path in "$SOURCE" "${DRAFTER#dspark:}" ${SHARD:+"$SHARD"}; do
		case "$path" in /srv/workspace/*|/home/*) ;; *) echo "hub_draftd: $path is not on the hub's local NVMe; inference never reads CephFS or cold storage" >&2; exit 1;; esac
		case "$path" in */../*|*/..) echo "hub_draftd: $path climbs out of its directory" >&2; exit 1;; esac
	done
	test -f "$SOURCE/block_draftd.py" -a -f "$SOURCE/spec_offline/block_drafter.py" || { echo "hub_draftd: $SOURCE lacks block_draftd.py or spec_offline/" >&2; exit 1; }
	NAMES=""
	if [ -n "$TARGET_EMBED$TARGET_HEAD$TARGET_NORM" ]; then
		test -n "$TARGET_EMBED" -a -n "$TARGET_HEAD" -a -n "$TARGET_NORM" -a -n "$SHARD" || { echo "hub_draftd: DRAFTD_TARGET_EMBED, DRAFTD_TARGET_HEAD and DRAFTD_TARGET_NORM name tensors of DRAFTD_TARGET_SHARD; set all three and the shard, or none" >&2; exit 1; }
		NAMES="--embed '$TARGET_EMBED' --head '$TARGET_HEAD' --norm '$TARGET_NORM'"
	fi
	if systemctl --user is-active -q "$UNIT"; then echo "hub_draftd: $UNIT is already active" >&2; exit 1; fi
	if [ "$(listening)" != 0 ]; then echo "hub_draftd: port $TAP_PORT or $DRAFT_PORT is already in use" >&2; exit 1; fi
	RECORD=""
	if [ -n "$RECORD_DIR" ]; then
		case "$RECORD_DIR" in /srv/workspace/*) ;; *) echo "hub_draftd: DRAFTD_RECORD_DIR must be on the hub's local NVMe (a recording holds about 172 KB per tap row)" >&2; exit 1;; esac
		mkdir -p "$RECORD_DIR" || exit 1
		RECORD="--record-dir '$RECORD_DIR'"
	fi
	mkdir -p "$(dirname "$LOG")"
	systemctl --user reset-failed "$UNIT" 2>/dev/null || true
	systemd-run --user --unit="$UNIT" -p Restart="$RESTART" -p RestartSec=10 --working-directory="$SOURCE" bash -c \
		"exec nice -n 5 python3 block_draftd.py --drafter '$DRAFTER' ${SHARD:+--target-shard '$SHARD'} $NAMES --tap-host $TAP_HOST --tap-port $TAP_PORT --draft-port $DRAFT_PORT --window $WINDOW --context-bytes $CONTEXT_BYTES $RECORD >> '$LOG' 2>&1"
	for _ in $(seq 1 180); do [ "$(listening)" = 2 ] && break; sleep 2; done
	if [ "$(listening)" != 2 ]; then echo "hub_draftd: $UNIT did not open both ports; see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; fi
	grep -a 'DRAFTD-READY' "$LOG" | tail -1
	;;
stop)
	if [ "$SERVER" = engine ]; then engine_stop; exit 0; fi
	systemctl --user stop "$UNIT" 2>/dev/null || true
	systemctl --user reset-failed "$UNIT" 2>/dev/null || true
	echo "hub_draftd: $UNIT stopped"
	;;
status)
	if [ "$SERVER" = engine ]; then engine_check; "$ENGINE_ROOT/bin/sparkpipe_draftd" --control "$CONTROL" status; exit $?; fi
	echo "unit $(systemctl --user is-active "$UNIT" || true) listening $(listening)/2"
	grep -a 'TAP-CONNECTED\|TAP-DISCONNECTED\|DRAFTD-TIMING\|ERROR' "$LOG" 2>/dev/null | tail -3 || true
	;;
*)
	echo "usage: hub_draftd.sh start SOURCE_DIR | stop | status" >&2
	exit 2
	;;
esac
