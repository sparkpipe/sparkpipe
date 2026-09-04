#!/usr/bin/env bash
# R3 flash-decode exact-cell, glm52 TP8 fleet (lane-r3flashdecode). Runs ON
# spark8; drives the eight glm52.tp8.fp8 ranks on spark8..sparkf (the fleet
# band, ranks 0..7 per tools/glm52_gen_deployment.py). Self-contained and
# idempotent: distributes the lane's published b1 driver + serving adapter,
# re-stages the per-rank runtime roots (packs stay in place - each rank
# reads its own sparkdata packs dir through a symlink), runs the O128 decode
# gate on BOTH deployment variants, then the long-context decode timing cell
# on the split path.
#
# The variants differ in ONE deployment key: decode_split_context_threshold.
#   phase A  threshold 0    - the shipped byte-for-byte single-pass attention
#   phase B  threshold 2048 - R3 flash-decode above the selection width
#   phase C  timing at 32K context on the split path vs the phase A control
#
# The key lives in the ADAPTER stage config (config/glm52_stage.json - the
# serving adapter parses it and rejects the config without it), and the
# cell's runtime roots point at /tmp/r3flash-glm52 so the staged r3 driver
# .so is the one that loads. The stage configs are rendered with
# max_sequence_positions=32768 so the 32K leg is admissible (the deployed
# fleet configs cap 4096; the O128 gates never reach it; the KV pool admits
# or refuses the 32K sequence loudly at admission).
#
# KILL-SWITCH (before any timing): the phase B O128 token stream must equal
# the phase A stream bit for bit. The split path's combine is a fixed-order
# deterministic merge of the same softmax - in practice the argmax stream
# cannot move - and a mismatch is a RED light for integration regardless.
#
# Receipts land in /tmp/r3flash-results/ on spark8:
#   o128_threshold0.json o128_threshold2048.json exact32k_split.json
#   summary.json

set -uo pipefail

LANE=/home/spark8/lane-r3flash
# durable roots: the wave deployments clean /tmp mid-flight (the t4 run
# lost its runtime root to a concurrent wave's /tmp sweep) - live in the
# lane home, which survived every wave today
# per-rank runtime roots (the driver host is spark8, ranks live on
# spark8..sparkf - a single spark8-root mkdir'd on every host is wrong
# by construction); RES is the driver host's own results dir
ROOT=/home/spark8/lane-r3flash/cell/runtime
RES=/home/spark8/lane-r3flash/cell/results
root_of() { printf '/home/%s/lane-r3flash/cell/runtime' "$1"; }
# fleet root per rank host: /home/<host>/sparkdata/glm52.tp8.fp8
# (NEVER printf a 'sparkRANK' template - 'sparkRANK' is not a printf
# directive, so it never substitutes; that latent bug shipped in the
# original lane script and survived every retry until t7)
PACK_TEMPLATE='glm52_tp8_rank%02d.fp8.glms52sp'
O128_BATCH=${O128_BATCH:-/tmp/r3flash-o128-batch.json}
B32K_BATCH=${B32K_BATCH:-/tmp/r3flash-32k-batch.json}
SPLIT_THRESHOLD=${SPLIT_THRESHOLD:-2048}
MAX_POSITIONS=${MAX_POSITIONS:-32768}
RUN_SECONDS_32K=${RUN_SECONDS_32K:-5400}
CONTROL_PORT_BASE=19480

HOSTS="spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
declare -A RANKOF=( [spark8]=0 [spark9]=1 [sparka]=2 [sparkb]=3 \
	[sparkc]=4 [sparkd]=5 [sparke]=6 [sparkf]=7 )

mkdir -p "$RES"

log() { printf '%s r3cell %s\n' "$(date -u +%H:%M:%S)" "$*"; }

token_hash() {
	jq -r '[.events[] | select(.event.event == "token") | .event.token_id] | join(",") + "\n"' "$1" \
		| sha256sum | cut -d' ' -f1
}

rank_pid() { # rank_pid HOST RANK - residentd of that host's cell root (by cwd)
	ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "
		for p in \$(pgrep -f 'bin/sparkpipe_model_residentd'); do
			[ \"\$(readlink /proc/\$p/cwd)\" = '$(root_of "$1")' ] && { echo \$p; break; }
		done"
}

fleet() { # fleet stop|start|ready
	local action="$1" h r
	case "$action" in
	stop)
		for h in $HOSTS; do
			r=${RANKOF[$h]}
			pid=$(rank_pid "$h" "$r")
			[[ -n "$pid" ]] && ssh -o BatchMode=yes "$h" "kill $pid" >/dev/null 2>&1
		done
		sleep 2
		;;
	start)
		for h in $HOSTS; do
			r=${RANKOF[$h]}
			[[ -z "$(rank_pid "$h" "$r")" ]] || continue
			ssh -o BatchMode=yes "$h" "
				cd '$(root_of "$h")' &&
				export LD_LIBRARY_PATH='$(root_of "$h")/lib':\$LD_LIBRARY_PATH \
					SPARKPIPE_RELEASE_ID='r3flash-rank$r'
				setsid -f bin/sparkpipe_model_residentd \
					--deployment config/model_resident.json --rank-index $r \
					>>/tmp/r3flash-rank$r.log 2>&1 </dev/null"
		done
		;;
	ready)
		local attempt h r ok port
		for attempt in $(seq 1 120); do
			ok=8
			for h in $HOSTS; do
				r=${RANKOF[$h]}
				port=$((CONTROL_PORT_BASE + r))
				[[ -n "$(rank_pid "$h" "$r")" ]] || { ok=$((ok - 1)); continue; }
				ssh -o BatchMode=yes "$h" "ss -ltnH | awk '{print \$4}' | grep -q ':$port\$'" \
					2>/dev/null || ok=$((ok - 1))
			done
			[[ "$ok" == 8 ]] && { log "fleet ready"; return 0; }
			sleep 5
		done
		log "fleet NOT ready (timeout)"
		return 1
		;;
	esac
}

bench() { # bench BATCH_JSON OUT_JSON TAG [TIMEOUT_S]
	local batch="$1" out="$2" tag="$3" tmo="${4:-0}" t0 t1 rc
	t0=$(date +%s)
	cd "$ROOT"
	if [[ "$tmo" != 0 ]]; then
		timeout "$tmo" ./bin/sparkpipe_model_batch \
			--deployment config/model_resident.json \
			--runtime-root "$ROOT" --batch "$batch" \
			>"$out" 2>"$out.stderr"
	else
		./bin/sparkpipe_model_batch \
			--deployment config/model_resident.json \
			--runtime-root "$ROOT" --batch "$batch" \
			>"$out" 2>"$out.stderr"
	fi
	rc=$?
	t1=$(date +%s)
	log "bench $tag rc=$rc wall=$((t1 - t0))s"
	return "$rc"
}

write_config() { # write_config THRESHOLD - render every rank's two configs
	local threshold="$1" h r
	for h in $HOSTS; do
		r=${RANKOF[$h]}
		ssh -o BatchMode=yes "$h" "mkdir -p $(root_of "$h")/config $(root_of "$h")/kv /home/$h/kvcache/r3flash-cell" &
	done
	wait
	for h in $HOSTS; do
		fleet_root="/home/$h/sparkdata/glm52.tp8.fp8"
		ssh -o BatchMode=yes "$h" "
			jq '.nodes |= map(.runtime_root = (\"/home/\" + .transport_host + \"/lane-r3flash/cell/runtime\")
				| .kv_backing_directory = (\"/home/\" + .transport_host + \"/kvcache/r3flash-cell\"))' \
				$fleet_root/config/model_resident.json > $(root_of "$h")/config/model_resident.json &&
			jq --argjson threshold '$threshold' --argjson positions '$MAX_POSITIONS' \
				'.decode_split_context_threshold = \$threshold
				| .max_sequence_positions = \$positions' \
				$fleet_root/config/glm52_stage.json > $(root_of "$h")/config/glm52_stage.json" \
			|| { log "FATAL config render failed on $h"; exit 8; }
	done
}

# --- phase 0: distribute binaries, libs, config template -------------------
# the fleet proxy flakes under load (banner timeouts, broken pipes): every
# remote step here retries, and a still-failing host is FATAL - silent
# distribute failures surface later as pack/config fatals (the t5 lesson)
retry_ssh() { # retry_ssh HOST CMD - up to 4 tries, 5s apart
	local h="$1" cmd="$2" attempt
	for attempt in 1 2 3 4; do
		ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "$cmd" && return 0
		log "retry $attempt failed on $h"
		sleep 5
	done
	return 1
}
log "phase 0: distribute"
[[ -f "$LANE/build/sparkpipe_model_residentd" && -f "$LANE/build/sparkpipe_model_batch" ]] \
	|| { log "FATAL lane binaries missing in $LANE/build (run the publish first)"; exit 9; }
for h in $HOSTS; do
	root="$(root_of "$h")"
	retry_ssh "$h" "mkdir -p $root/bin $root/lib $root/config $root/kv" \
		|| { log "FATAL mkdir failed on $h"; exit 8; } &
done
wait
for h in $HOSTS; do
	fleet_root="/home/$h/sparkdata/glm52.tp8.fp8"
	root="$(root_of "$h")"
	rsync -aq -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
		"$LANE/build/sparkpipe_model_residentd" "$LANE/build/sparkpipe_model_batch" "$h:$root/bin/" \
		|| { log "FATAL bin rsync failed on $h"; exit 8; }
	rsync -aq -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
		"$LANE/build/model_driver.so" "$LANE/build/model_serving_adapter.so" \
		"$LANE/build/hidden_transport.so" "$h:$root/lib/" \
		|| { log "FATAL lib rsync failed on $h"; exit 8; }
	# packs stay at the rank's own fleet root - symlink them into the cell
	# runtime root (NOTE: expand fleet_root here, not $HOME - the script
	# runs on one host and ssh targets the rest)
	root="$(root_of "$h")"
	retry_ssh "$h" "ln -sfn $fleet_root/packs $root/packs" \
		|| { log "FATAL packs symlink failed on $h"; exit 8; }
done
for h in $HOSTS; do
	r=${RANKOF[$h]}
	pack="$(printf "$PACK_TEMPLATE" "$r")"
	root="$(root_of "$h")"
	retry_ssh "$h" "test -s $root/packs/$pack" \
		|| { log "FATAL pack missing on $h ($pack)"; exit 9; }
done
# the batch payloads: prefer retained files, else synthesize the O128 decode
# batch from the fleet's own generator contract
if [[ ! -f "$O128_BATCH" ]]; then
	log "FATAL O128 batch json missing at $O128_BATCH (stage it from the retained glm52 cell files)"
	exit 9
fi
log "phase 0 done"

# --- phase A: threshold 0 (shipped single-pass path) + O128 gate -----------
log "phase A: threshold-0 O128"
write_config 0
fleet stop
fleet start
fleet ready || { log "FATAL threshold-0 fleet not ready"; exit 8; }
bench "$O128_BATCH" "$RES/o128_threshold0.json" o128_threshold0 1800
HA=$(token_hash "$RES/o128_threshold0.json")
log "o128_threshold0 hash=$HA"

# --- phase B: threshold engaged (R3 split path) + the KILL-SWITCH gate -----
log "phase B: threshold-$SPLIT_THRESHOLD O128 kill-switch"
write_config "$SPLIT_THRESHOLD"
fleet stop
fleet start
fleet ready || { log "FATAL split fleet not ready"; exit 8; }
bench "$O128_BATCH" "$RES/o128_threshold2048.json" o128_threshold2048 1800
HB=$(token_hash "$RES/o128_threshold2048.json")
log "o128_threshold2048 hash=$HB"
if [[ "$HA" == "$HB" ]]; then
	log "PASS kill-switch: split path reproduces the single-pass token stream bit for bit"
else
	log "RED LIGHT: token-stream mismatch (control $HA vs split $HB) - no timing may be trusted"
	python3 - "$RES" "$HA" "$HB" <<'PYEOF'
import json, os, sys
res, ha, hb = sys.argv[1], sys.argv[2], sys.argv[3]
with open(os.path.join(res, "summary.json"), "w") as output:
	json.dump({"kill_switch": "FAIL", "control_hash": ha, "split_hash": hb},
		output, indent=1)
PYEOF
	exit 6
fi

# --- phase C: long-context decode timing on the split path -----------------
log "phase C: 32K decode split timing"
if [[ -f "$B32K_BATCH" ]]; then
	bench "$B32K_BATCH" "$RES/exact32k_split.json" exact32k_split "$RUN_SECONDS_32K"
	# the control for the timing comparison: same batch, threshold 0
	write_config 0
	fleet stop
	fleet start
	fleet ready || { log "FATAL control fleet not ready"; exit 8; }
	bench "$B32K_BATCH" "$RES/exact32k_control.json" exact32k_control "$RUN_SECONDS_32K"
else
	log "SKIP exact-32K (batch json unavailable at $B32K_BATCH)"
fi

# --- summary ---------------------------------------------------------------
python3 - "$RES" "$HA" "$HB" <<'PYEOF'
import json, os, sys
res, ha, hb = sys.argv[1], sys.argv[2], sys.argv[3]
summary = {"kill_switch": "PASS" if ha == hb else "FAIL",
           "control_hash": ha, "split_hash": hb}
for name in ("o128_threshold0", "o128_threshold2048",
             "exact32k_control", "exact32k_split"):
    path = os.path.join(res, name + ".json")
    entry = {}
    try:
        with open(path) as source:
            data = json.load(source)
        for field in ("decode_tokens_per_second", "total_tokens",
                      "prefill_seconds", "ttft_seconds",
                      "wall_seconds", "elapsed_seconds", "requests"):
            if field in data:
                entry[field] = data[field]
        events = data.get("events", [])
        entry["event_count"] = len(events)
    except Exception as error:
        entry["error"] = repr(error)
    summary[name] = entry
with open(os.path.join(res, "summary.json"), "w") as output:
	json.dump(summary, output, indent=1)
print(json.dumps(summary, indent=1))
PYEOF
