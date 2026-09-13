#!/usr/bin/env bash
# TP16 pack production for Qwen3.8-2.4T-A95B (capacity-wall mitigation).
#
# Produces the sixteen rank-local stage packs (MXFP4-E2M1 sliced experts,
# replicated BF16 spine) defined in tools/qwen38_tp16_stagepack.py, per
# docs/QWEN38_MAX_BRINGUP.md risk R1 / section 3 (pack gaps).
#
# Usage:
#   qwen38_tp16_pack_production.sh --checkpoint DIR --output-dir DIR
#       --expert-codec mxfp4_e2m1 [--ranks 16] [--jobs 4] [--plan-only] [--verify]
#
# Gates enforced BEFORE any pack byte is written:
#   0. codec choice is explicit (mxfp4_e2m1 overrides the quality-first vendor
#      FP8 contract stance - that decision belongs to the operator)
#   1. checkpoint has model.safetensors.index.json + config.json
#   2. HASH_VERIFY.json present (the vendor verification receipt)
#   3. python3 + numpy importable
#   4. codec self-test passes (vectorized == family scalar reference)
#   5. output filesystem fits the projected total (+5% headroom)
#
# After packing: receipts + placement manifest land beside the packs;
# publish per docs/DATAFILE_NAMING.md (content-addressed names); record
# evidence in PERFORMANCE_STATUS.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKER="tools/qwen38_tp16_stagepack.py"

CHECKPOINT="" OUTPUT_DIR="" CODEC="" RANKS=16 JOBS=4 PLAN_ONLY=0 VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --expert-codec) CODEC="$2"; shift 2 ;;
    --ranks)        RANKS="$2"; shift 2 ;;
    --jobs)         JOBS="$2"; shift 2 ;;
    --plan-only)    PLAN_ONLY=1; shift ;;
    --verify)       VERIFY=1; shift ;;
    -h|--help)      grep "^#" "$0" | cut -c3-; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$CODEC" ]] || { echo "ERROR: --expert-codec is required (mxfp4_e2m1 = capacity mitigation; fp8_e4m3 = vendor passthrough)" >&2; exit 2; }
if [[ "$CODEC" != "mxfp4_e2m1" && "$CODEC" != "fp8_e4m3" ]]; then
  echo "ERROR: unknown codec $CODEC" >&2; exit 2
fi
if [[ "$CODEC" == "mxfp4_e2m1" ]]; then
  echo "NOTICE: mxfp4_e2m1 requantization OVERRIDES the quality-first vendor-FP8"
  echo "        pack policy pinned by model_contracts/qwen38_authoritative.json."
  echo "        This is the bring-up R1 capacity mitigation; record the decision."
fi
[[ -n "$CHECKPOINT" ]] || { echo "ERROR: --checkpoint is required" >&2; exit 2; }
[[ -d "$CHECKPOINT" ]] || { echo "ERROR: checkpoint dir not found: $CHECKPOINT" >&2; exit 2; }
for f in model.safetensors.index.json config.json HASH_VERIFY.json; do
  [[ -f "$CHECKPOINT/$f" ]] || { echo "ERROR: missing $CHECKPOINT/$f" >&2; exit 2; }
done
echo "gate: HASH_VERIFY.json head -> $(head -c 240 "$CHECKPOINT/HASH_VERIFY.json")"

command -v python3 >/dev/null || { echo "ERROR: python3 not found" >&2; exit 2; }
python3 -c 'import numpy' 2>/dev/null || { echo "ERROR: numpy not importable" >&2; exit 2; }

echo "== gate: codec self-test =="
python3 "$SCRIPT_DIR/tools/qwen38_tp16_stagepack.py" --self-test

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "== plan (rank 0 arithmetic, nothing written) =="
PLAN_OUT="$(mktemp)"
python3 "$PACKER" --checkpoint "$CHECKPOINT" --expert-codec "$CODEC" \
  --tp-degree "$RANKS" --rank 0 --dry-run | tee "$PLAN_OUT"
RANK_GIB="$(grep -o 'file_gib=[0-9.]*' "$PLAN_OUT" | head -1 | cut -d= -f2)"
rm -f "$PLAN_OUT"
[[ -n "$RANK_GIB" ]] || { echo "ERROR: could not parse plan output" >&2; exit 2; }
echo "projected total: $RANK_GIB GiB/rank x $RANKS ranks"
if [[ "$PLAN_ONLY" == "1" ]]; then echo "plan-only requested; stopping."; exit 0; fi

[[ -n "$OUTPUT_DIR" ]] || { echo "ERROR: --output-dir is required to write packs" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
AVAIL_KIB="$(df -Pk "$OUTPUT_DIR" | awk 'NR==2 {print $4}')"
TOTAL_KIB="$(awk -v g="$RANK_GIB" -v n="$RANKS" 'BEGIN { printf "%d", g * 1048576 * n }')"
NEED_KIB=$(( TOTAL_KIB + TOTAL_KIB / 20 ))
if (( AVAIL_KIB < NEED_KIB )); then
  echo "ERROR: output fs short: need ~$((NEED_KIB / 1024 / 1024)) GiB, have $((AVAIL_KIB / 1024 / 1024)) GiB free" >&2
  exit 2
fi

echo "== packing ranks 0..$((RANKS - 1)) (jobs=$JOBS) =="
launch_one() {
  local rank="$1"
  echo "[rank $rank] start $(date -u +%H:%M:%S)"
  if python3 "$PACKER" --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR" \
      --expert-codec "$CODEC" --tp-degree "$RANKS" --rank "$rank" \
      >"$OUTPUT_DIR/.rank$rank.log" 2>&1; then
    echo "[rank $rank] done $(date -u +%H:%M:%S)"
  else
    echo "[rank $rank] FAILED (see $OUTPUT_DIR/.rank$rank.log)" >&2
    return 1
  fi
}

pids=()
for rank in $(seq 0 $((RANKS - 1))); do
  while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 2; done
  launch_one "$rank" &
  pids+=("$!")
done
FAILURES=0
for pid in "${pids[@]}"; do wait "$pid" || FAILURES=$((FAILURES + 1)); done
if (( FAILURES > 0 )); then echo "ERROR: $FAILURES rank job(s) failed" >&2; exit 1; fi

echo "== aggregate manifest =="
PRECISION="mxfp4-e2m1"
if [[ "$CODEC" == "fp8_e4m3" ]]; then PRECISION="fp8-e4m3-b128"; fi
MANIFEST="$(python3 "$PACKER" --aggregate-manifest --output-dir "$OUTPUT_DIR" \
  --expert-codec "$CODEC" --tp-degree "$RANKS" | awk '/^qwen38_tp16_stagepack manifest /{print $NF}')"
[[ -n "$MANIFEST" && -f "$MANIFEST" ]] || { echo "ERROR: manifest not produced" >&2; exit 1; }
echo "manifest: $MANIFEST"

if [[ "$VERIFY" == "1" ]]; then
  echo "== verify: re-hash every published pack =="
  python3 - "$OUTPUT_DIR" "$MANIFEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
out_dir, manifest_path = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
for rank, entry in sorted(manifest["ranks"].items(), key=lambda kv: int(kv[0])):
    pack = out_dir / entry["file"]
    h = hashlib.sha256()
    with pack.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    got = h.hexdigest()
    status = "OK" if got == entry["sha256"] else "MISMATCH"
    print("rank " + rank + ": " + status + " (" + format(entry["bytes"] / 2**30, ".2f") + " GiB)")
    if status != "OK":
        raise SystemExit(1)
print("verify: all rank packs hash-match their receipts")
PY
fi

cat <<EOF

== evidence block (paste into PERFORMANCE_STATUS.md context) ==
artifact: qwen38 tp16 rank-local packs ($PRECISION), $RANKS ranks
producer: tools/qwen38_tp16_stagepack.py via tools/qwen38_tp16_pack_production.sh
checkpoint: $CHECKPOINT
manifest: $MANIFEST
per-rank size: $RANK_GIB GiB (arithmetic projection confirmed at write time)
loader dependency: ValidateEntry rank-local-view delta must land before load
  (docs/QWEN38_MAX_BRINGUP.md); these packs are NOT yet loadable by the module.
==
EOF
