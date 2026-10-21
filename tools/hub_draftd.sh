#!/usr/bin/env bash
set -eu
UNIT=${DRAFTD_UNIT:-sparkpipe-draftd}
LOG=${DRAFTD_LOG:-/srv/workspace/sparkpipe/logs/draftd.log}
DRAFTER=${DRAFTD_DRAFTER:-/mnt/cold-raid6/models/kimi-k3-dflash2-lightseek-e77935fb}
SHARD=${DRAFTD_TARGET_SHARD-/mnt/cold-raid6/models/kimi-k3/model-00094-of-000096.safetensors}
TAP_HOST=${DRAFTD_TAP_HOST:-10.10.210.3}
TAP_PORT=${DRAFTD_TAP_PORT:-7701}
DRAFT_PORT=${DRAFTD_DRAFT_PORT:-7702}
CONTEXT_BYTES=${DRAFTD_CONTEXT_BYTES:-8589934592}
WINDOW=${DRAFTD_WINDOW:-4096}
RECORD_DIR=${DRAFTD_RECORD_DIR:-}
listening() { ss -ltn | grep -c "$TAP_HOST:$TAP_PORT \|127.0.0.1:$DRAFT_PORT " || true; }
case "${1:-}" in
start)
	SOURCE=${2:?usage: hub_draftd.sh start SOURCE_DIR}
	test -f "$SOURCE/block_draftd.py" -a -f "$SOURCE/spec_offline/block_drafter.py" || { echo "hub_draftd: $SOURCE lacks block_draftd.py or spec_offline/" >&2; exit 1; }
	if systemctl --user is-active -q "$UNIT"; then echo "hub_draftd: $UNIT is already active" >&2; exit 1; fi
	if [ "$(listening)" != 0 ]; then echo "hub_draftd: port $TAP_PORT or $DRAFT_PORT is already in use" >&2; exit 1; fi
	RECORD=""
	if [ -n "$RECORD_DIR" ]; then
		case "$RECORD_DIR" in /mnt/cold-raid6/*) ;; *) echo "hub_draftd: DRAFTD_RECORD_DIR must be on /mnt/cold-raid6 (a recording holds about 172 KB per tap row)" >&2; exit 1;; esac
		mkdir -p "$RECORD_DIR" || exit 1
		RECORD="--record-dir '$RECORD_DIR'"
	fi
	mkdir -p "$(dirname "$LOG")"
	systemctl --user reset-failed "$UNIT" 2>/dev/null || true
	systemd-run --user --unit="$UNIT" -p Restart=on-failure -p RestartSec=10 --working-directory="$SOURCE" bash -c \
		"exec nice -n 5 python3 block_draftd.py --drafter '$DRAFTER' ${SHARD:+--target-shard '$SHARD'} --tap-host $TAP_HOST --tap-port $TAP_PORT --draft-port $DRAFT_PORT --window $WINDOW --context-bytes $CONTEXT_BYTES $RECORD >> '$LOG' 2>&1"
	for _ in $(seq 1 180); do [ "$(listening)" = 2 ] && break; sleep 2; done
	if [ "$(listening)" != 2 ]; then echo "hub_draftd: $UNIT did not open both ports; see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; fi
	grep -a 'DRAFTD-READY' "$LOG" | tail -1
	;;
stop)
	systemctl --user stop "$UNIT" 2>/dev/null || true
	systemctl --user reset-failed "$UNIT" 2>/dev/null || true
	echo "hub_draftd: $UNIT stopped"
	;;
status)
	echo "unit $(systemctl --user is-active "$UNIT" || true) listening $(listening)/2"
	grep -a 'TAP-CONNECTED\|TAP-DISCONNECTED\|DRAFTD-TIMING\|ERROR' "$LOG" 2>/dev/null | tail -3 || true
	;;
*)
	echo "usage: hub_draftd.sh start SOURCE_DIR | stop | status" >&2
	exit 2
	;;
esac
