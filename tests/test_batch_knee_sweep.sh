#!/bin/bash
# CB4 batch-size knee (B*) sweep for a single-spark deployment.
#
# For each batch size B the deployment's own sparkpipe_model_batch runs
# twice (two output budgets); the decode-only rate comes from the budget
# difference, so connect + prefill overhead cancel out.
#
# Parameters (parameterization rule: no hardcoded nodes):
#   SPARK_HOST     spark node to run on                (required)
#   DEPLOY         deployment dir with bin/ + config/  (required)
#   BATCHES        batch sizes to sweep                (default "1 2 4 8 16 32 64")
#   BUDGETS        the two output budgets              (default "128 256")
#   PROMPT_TOKENS  prompt length per request           (default 64)
#   CONFIG         deployment config file              (default config/model_resident.json)
#
# Usage (from anywhere with the repo checked out):
#   SPARK_HOST=spark3 DEPLOY=/home/spark3/sparkdata/qwen38.fp8.tp1 \
#       tests/test_batch_knee_sweep.sh
#
# stdout: CSV "B,budget_hi,wall_hi_ms,tokens_hi,aggregate_tok_s,decode_tok_s"
# stderr: residentd lifecycle chatter. Starts and TERMs the residentd.

set -u

if [ "${KNEE_ON_HOST:-0}" != "1" ]; then
	: "${SPARK_HOST:?SPARK_HOST required}"
	: "${DEPLOY:?DEPLOY required}"
	exec ssh -o BatchMode=yes "$SPARK_HOST" \
		"KNEE_ON_HOST=1 KNEE_DEPLOY='$DEPLOY' KNEE_CONFIG='${CONFIG:-config/model_resident.json}' \
		 KNEE_BATCHES='${BATCHES:-1 2 4 8 16 32 64}' KNEE_BUDGETS='${BUDGETS:-128 256}' KNEE_PROMPT='${PROMPT_TOKENS:-64}' \
		 bash -s" < "$0"
fi

DEPLOY="${KNEE_DEPLOY:?KNEE_DEPLOY required}"
CONFIG="${KNEE_CONFIG:-config/model_resident.json}"
BATCHES="${KNEE_BATCHES:-1 2 4 8 16 32 64}"
BUDGETS="${KNEE_BUDGETS:-128 256}"
PROMPT_TOKENS="${KNEE_PROMPT:-64}"

WORK="/tmp/sparkpipe-knee-sweep-$$"
mkdir -p "$WORK"
LOG="/tmp/sparkpipe-knee-residentd.log"
RESIDENT_PID=""

cleanup() {
	if [ -n "$RESIDENT_PID" ]; then
		kill -TERM "$RESIDENT_PID" 2>/dev/null || true
		wait "$RESIDENT_PID" 2>/dev/null || true
	fi
	rm -rf "$WORK"
}
trap cleanup EXIT

if pgrep -f sparkpipe_model_residentd > /dev/null; then
	echo "a residentd is already running here - refusing to start" >&2
	exit 1
fi
if pgrep -f sparkpipe_model_api > /dev/null; then
	echo "a model_api holds the client slot here - refusing to start" >&2
	exit 1
fi

"$DEPLOY/bin/sparkpipe_model_residentd" --deployment "$DEPLOY/$CONFIG" \
	--rank-index 0 > "$LOG" 2>&1 &
RESIDENT_PID=$!
READY=0
for _ in $(seq 1 180); do
	if grep -q "model_residentd ready" "$LOG" 2>/dev/null; then READY=1; break; fi
	kill -0 "$RESIDENT_PID" 2>/dev/null || { echo "residentd died:" >&2; cat "$LOG" >&2; exit 1; }
	sleep 1
done
[ "$READY" = "1" ] || { echo "residentd not ready in 180s" >&2; cat "$LOG" >&2; exit 1; }
echo "residentd ready pid=$RESIDENT_PID" >&2

PROMPT_JSON=""
for ((I=0; I<PROMPT_TOKENS; I++)); do
	PROMPT_JSON+="$((1000 + (I * 137) % 50000)),"
done
PROMPT_JSON="${PROMPT_JSON%,}"

LO_BUDGET=$(echo "$BUDGETS" | awk '{print $1}')
HI_BUDGET=$(echo "$BUDGETS" | awk '{print $2}')

write_batch() {
	local path="$1" size="$2" budget="$3" req_id
	{
		printf '{"schema_version":1,"connect_timeout_ms":10000,"request_capacity":%u,"max_context_tokens":4096,"max_prefill_rows_per_submission":64,"maximum_messages_per_rank_per_progress":64,"maximum_new_submissions_per_progress":8,"stop_token_ids":[],"requests":[' "$size"
		for ((req_id=0; req_id<size; req_id++)); do
			[ "$req_id" -gt 0 ] && printf ','
			printf '{"request_id":%u,"sequence_id":%u,"priority":10,"output_token_budget":%u,"prompt_token_ids":[%s]}' \
				"$((5000 + req_id))" "$((9000 + req_id))" "$budget" "$PROMPT_JSON"
		done
		printf ']}\n'
	} > "$path"
}

run_case() {
	local size="$1" budget="$2" tag="$3" start end wall tokens status
	write_batch "$WORK/batch-$tag.json" "$size" "$budget"
	start=$(date +%s%N)
	"$DEPLOY/bin/sparkpipe_model_batch" \
		--deployment "$DEPLOY/$CONFIG" \
		--runtime-root "$DEPLOY" \
		--batch "$WORK/batch-$tag.json" \
		> "$WORK/out-$tag.ndjson" 2> "$WORK/err-$tag.log"
	status=$?
	end=$(date +%s%N)
	if [ "$status" -ne 0 ]; then
		echo "batch tool failed status=$status (B=$size budget=$budget)" >&2
		tail -5 "$WORK/err-$tag.log" >&2
		exit 1
	fi
	wall=$(( (end - start) / 1000000 ))
	tokens=$(grep -c '"event":"token"' "$WORK/out-$tag.ndjson" || true)
	echo "$size,$budget,$wall,$tokens" >> "$WORK/rows.csv"
}

for B in $BATCHES; do
	run_case "$B" "$LO_BUDGET" "lo"
	run_case "$B" "$HI_BUDGET" "hi"
done

echo "B,budget_hi,wall_hi_ms,tokens_hi,aggregate_tok_s,decode_tok_s"
awk -F, -v hi="$HI_BUDGET" '
	$2 == hi { hb[$1] = $3; ht[$1] = $4 }
	$2 != hi { lb[$1] = $3; lt[$1] = $4 }
	END {
		for (b in hb)
			printf "%s,%s,%d,%d,%.2f,%.2f\n", b, hi, hb[b], ht[b], \
				(b * ht[b] * 1000) / hb[b], \
				(b * (ht[b] - lt[b]) * 1000) / (hb[b] - lb[b])
	}
' "$WORK/rows.csv" | sort -t, -k1,1n
