#!/usr/bin/env bash
# R2c dsv4 TP4 exact-32K prefill cell (lane-r2-prefill). Runs ON spark4;
# drives the four ranks on spark4/5/6/7. Self-contained: distributes the
# r2-prefill devcycle builds (b1 = rows-1 kernel-neutral phase, b8 =
# width-8 R2c bulk phase), re-stages rank packs from the retained bisect
# lane files, runs the pinned O128 exact-output gate on BOTH phases, then
# the exact-32K cell on the bulk path.
#
# Receipts land in /tmp/r2prefill-results/ on spark4:
#   o128_b1.json o128_b8.json exact32k_b8.json summary.json
#
# Gate: the O128 token stream must equal the pinned lean hash on both
# phases (b8 exercises the R2c bulk prefill; b1 exercises the rows==1
# decode-identical path). A mismatch fails the phase loudly and is
# recorded in summary.json.

set -uo pipefail

ROOT=/tmp/dsv4r2prefill
RES=/tmp/r2prefill-results
B1=/tmp/devcycle-build-r2c-b1
B8=/tmp/devcycle-build-r2c-b8
BISECT_FIX=/tmp/dsv4bisect-fix
BISECT_MAIN=/tmp/dsv4bisect-main
SPARK5_PACKS=/home/spark5/lane-dsv4bisect/packs
O128_BATCH=/tmp/devcycle-o128-batch.json
B32K_SRC=/tmp/dsv4-exact32k-b1.json
B32K_B8=/tmp/dsv4-exact32k-b8.json
O128_PINNED=a9385d0b296ca083e577e715d2f6335067691dce0e0dd5ab1394a102a3d3631f
RUN_SECONDS_32K=${RUN_SECONDS_32K:-5400}
CONTROL_PORT=18480

HOSTS="spark4 spark5 spark6 spark7"
declare -A RANKOF=( [spark4]=0 [spark5]=1 [spark6]=2 [spark7]=3 )

mkdir -p "$RES"

log() { printf '%s r2cell %s\n' "$(date -u +%H:%M:%S)" "$*"; }

token_hash() {
	jq -r '[.events[] | select(.event.event == "token") | .event.token_id] | join(",") + "\n"' "$1" \
		| sha256sum | cut -d' ' -f1
}

rank_pid() { # rank_pid HOST RANK - residentd of $ROOT on that host (by cwd)
	ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "
		for p in \$(pgrep -f 'bin/sparkpipe_model_residentd'); do
			[ \"\$(readlink /proc/\$p/cwd)\" = '$ROOT' ] && { echo \$p; break; }
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
				cd '$ROOT' &&
				export LD_LIBRARY_PATH='$ROOT/lib':\$LD_LIBRARY_PATH \
					SPARKPIPE_RELEASE_ID='r2prefill-rank$r'
				setsid -f bin/sparkpipe_model_residentd \
					--deployment config/model_resident.json --rank-index $r \
					>>/tmp/r2prefill-rank$r.log 2>&1 </dev/null"
		done
		;;
	ready)
		local attempt h r ok
		for attempt in $(seq 1 120); do
			ok=4
			for h in $HOSTS; do
				r=${RANKOF[$h]}
				[[ -n "$(rank_pid "$h" "$r")" ]] || { ok=$((ok - 1)); continue; }
				ssh -o BatchMode=yes "$h" "ss -ltnH | awk '{print \$4}' | grep -q ':$CONTROL_PORT\$'" \
					2>/dev/null || ok=$((ok - 1))
			done
			[[ "$ok" == 4 ]] && { log "fleet ready"; return 0; }
			sleep 5
		done
		log "fleet NOT ready (timeout)"
		return 1
		;;
	esac
}

bench() { # bench BATCH_JSON OUT_JSON TAG [TIMEOUT_S]
	local batch="$1" out="$2" tag="$3" tmo="${4:-0}"
	local t0 t1 rc
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

# --- phase 0: distribute binaries, libs, configs, packs ------------------
log "phase 0: distribute"
for h in $HOSTS; do
	ssh -o BatchMode=yes "$h" "mkdir -p $ROOT/bin $ROOT/lib $ROOT/config $ROOT/kv $ROOT/packs" &
done
wait
for h in $HOSTS; do
	rsync -aq "$BISECT_FIX/config/" "$h:$ROOT/config/"
	scp -q "$B1/sparkpipe_model_residentd" "$B1/sparkpipe_model_batch" "$h:$ROOT/bin/"
	scp -q "$B1/model_driver.so" "$B1/model_serving_adapter.so" "$B1/hidden_transport.so" "$h:$ROOT/lib/"
	ssh -o BatchMode=yes "$h" "sed -i 's|/tmp/dsv4bisect-fix|$ROOT|g' $ROOT/config/model_resident.json"
done
# packs: rank0 on spark4, rank3 on spark7 (retained from the bisect lane);
# rank1 is a same-node copy on spark5; rank2 goes spark5-home -> spark6 via nc
scp -q "$BISECT_FIX/packs/dsv4_flash_stage.spstage" spark4:"$ROOT/packs/" 2>/dev/null
scp -q "$BISECT_MAIN/packs/dsv4_flash_stage.spstage" spark7:"$ROOT/packs/" 2>/dev/null
ssh -o BatchMode=yes spark5 "cp $SPARK5_PACKS/dsv4_flash_v3_tp4rank1.spstage $ROOT/packs/dsv4_flash_stage.spstage 2>/dev/null || true"
if ! ssh -o BatchMode=yes spark6 "test -f $ROOT/packs/dsv4_flash_stage.spstage"; then
	log "copying rank2 pack spark5 -> spark6 (40GB)"
	ssh -o BatchMode=yes spark6 "nc -l -p 4040 > $ROOT/packs/.pack.tmp && mv $ROOT/packs/.pack.tmp $ROOT/packs/dsv4_flash_stage.spstage" &
	ncpid=$!
	sleep 2
	ssh -o BatchMode=yes spark5 "nc -q1 spark6 4040 < $SPARK5_PACKS/dsv4_flash_v3_tp4rank2.spstage"
	wait $ncpid
fi
for pair in "spark4:$BISECT_FIX" "spark5:$SPARK5_PACKS/dsv4_flash_v3_tp4rank1.spstage" \
	"spark6:$ROOT/packs" "spark7:$BISECT_MAIN"; do
	h=${pair%%:*}
	rest=${pair#*:}
	ssh -o BatchMode=yes "$h" "test -s $ROOT/packs/dsv4_flash_stage.spstage" \
		|| { log "FATAL pack missing on $h ($rest)"; exit 9; }
done
log "phase 0 done"

# --- phase A: b1 rows-1 (kernel-neutral path) + O128 gate ----------------
log "phase A: b1 O128 gate"
fleet stop
fleet start
fleet ready || { log "FATAL b1 fleet not ready"; exit 8; }
bench "$O128_BATCH" "$RES/o128_b1.json" o128_b1 1800
H1=$(token_hash "$RES/o128_b1.json")
log "o128_b1 hash=$H1 (pinned $O128_PINNED)"
[[ "$H1" == "$O128_PINNED" ]] && log "PASS o128_b1 exact" || log "FAIL o128_b1 hash mismatch"

# --- phase B: b8 width-8 (R2c bulk path) + O128 gate ---------------------
log "phase B: b8 swap + O128 gate"
fleet stop
for h in $HOSTS; do
	scp -q "$B8/model_driver.so" "$B8/model_serving_adapter.so" "$h:$ROOT/lib/"
	ssh -o BatchMode=yes "$h" "sed -i 's/\"max_input_rows\": 1,/\"max_input_rows\": 8,/' $ROOT/config/model_resident.json"
done
python3 - <<'PYEOF'
import json
with open("/tmp/dsv4-exact32k-b1.json") as source:
    batch = json.load(source)
batch["max_prefill_rows_per_submission"] = 8
with open("/tmp/dsv4-exact32k-b8.json", "w") as output:
    json.dump(batch, output)
PYEOF
fleet start
fleet ready || { log "FATAL b8 fleet not ready"; exit 8; }
bench "$O128_BATCH" "$RES/o128_b8.json" o128_b8 1800
H8=$(token_hash "$RES/o128_b8.json")
log "o128_b8 hash=$H8"
[[ "$H8" == "$O128_PINNED" ]] && log "PASS o128_b8 exact (bulk path reproduces the pinned stream)" \
	|| log "FAIL o128_b8 hash MISMATCH - bulk path diverged"

# --- phase C: exact-32K on the bulk path ---------------------------------
log "phase C: exact-32K b8"
[[ -f "$B32K_SRC" ]] || scp -q spark5:"$SPARK5_PACKS/../dsv4-exact32k-b1.json" "$B32K_SRC" 2>/dev/null
if [[ -f "$B32K_SRC" && -f "$B32K_B8" ]]; then
	bench "$B32K_B8" "$RES/exact32k_b8.json" exact32k_b8 "$RUN_SECONDS_32K"
else
	log "SKIP exact-32K (batch json unavailable)"
fi

# --- summary -------------------------------------------------------------
python3 - "$RES" "$H1" "$H8" "$O128_PINNED" <<'PYEOF'
import json, os, sys
res, h1, h8, pinned = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
summary = {"o128_b1_hash": h1, "o128_b8_hash": h8, "pinned": pinned,
           "o128_b1_exact": h1 == pinned, "o128_b8_exact": h8 == pinned}
for name in ("o128_b1", "o128_b8", "exact32k_b8"):
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
        if events:
            entry["first_event"] = events[0]
    except Exception as error:
        entry["error"] = repr(error)
    summary[name] = entry
with open(os.path.join(res, "summary.json"), "w") as output:
    json.dump(summary, output, indent=1)
print(json.dumps(summary, indent=1))
PYEOF
log "cell done - receipts in $RES"