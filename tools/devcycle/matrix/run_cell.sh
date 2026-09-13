#!/usr/bin/env bash
# run_cell.sh ROOT BATCH_LOCAL TAG REPS [REMOTE_BATCH]
# Runs 1 discarded warmup + REPS measured runs of one matrix cell against the
# TP4 runtime whose batch client lives at ROOT/bin/sparkpipe_model_batch.
set -euo pipefail
ROOT="$1"; BATCH_LOCAL="$2"; TAG="$3"; REPS="$4"
REMOTE_BATCH="${5:-/tmp/matrix-current-batch.json}"
RECEIPT_DIR=/tmp/dsv4-matrix-receipts
mkdir -p "$RECEIPT_DIR"
scp -q -o BatchMode=yes "$BATCH_LOCAL" spark4:"$REMOTE_BATCH"
run_one() { # $1 = receipt path
  python3 /Users/mac/dsh.sparkpipe/tools/model_stream_decode_benchmark.py --output "$1" \
    ssh -o BatchMode=yes spark4 "$ROOT/bin/sparkpipe_model_batch" \
      --deployment "$ROOT/config/model_resident.json" \
      --runtime-root "$ROOT" \
      --batch "$REMOTE_BATCH" >/dev/null 2>"$1.stderr"
}
run_one "$RECEIPT_DIR/$TAG-warm.json"
echo "warm ok: $TAG"
k=1
while [ $k -le $REPS ]; do
  run_one "$RECEIPT_DIR/$TAG-r$k.json"
  python3 /Users/mac/dsh.sparkpipe/tools/devcycle/matrix/summarize_run.py "$RECEIPT_DIR/$TAG-r$k.json"
  k=$((k+1))
done
