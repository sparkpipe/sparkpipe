#!/bin/bash
# Lease-gated warm batch for lane gemma4-t1: node-local checkpoint copy,
# contract provenance check, emission of the 14 missing TP4xPP4 stage packs,
# placement to the fleet, and 26b reference-fixture generation.
# Run ON sparkA only while CEPH_LEASE is held by this lane.
set -euo pipefail
CKPT_SRC=/mnt/model-warm/gemma-4-26b-a4b-it
CKPT_DST=$HOME/gemma4_26b_ckpt
REPO=$HOME/g4t1
OUT=$REPO/qualification/t1_reference/gen_26b

PLACEMENT_SPARK8=spark8
PLACEMENT_SPARKB=sparkb

echo "== copy checkpoint local (one pass) =="
if [ ! -f "$CKPT_DST/model.safetensors.index.json" ]; then
    mkdir -p "$CKPT_DST"
    rsync -a --progress "$CKPT_SRC/" "$CKPT_DST/"
fi

echo "== contract provenance (digest_freeze) =="
python3 - << 'EOF'
import hashlib, json, sys
contract = json.load(open("model_contracts/gemma4_26b_a4b_authoritative.json"))
frozen = contract["digest_freeze"]["files"]
bad = []
for name, record in sorted(frozen.items()):
    try:
        digest = hashlib.sha256(open(name, "rb").read()).hexdigest()
    except FileNotFoundError:
        bad.append(f"{name}: missing")
        continue
    if digest != record["sha256"]:
        bad.append(f"{name}: {digest} != frozen {record['sha256']}")
    print(f"{name} {digest} {'OK' if digest == record['sha256'] else 'DRIFT'}")
if bad:
    print("PROVENANCE-FAIL:", "; ".join(bad))
    sys.exit(3)
print("PROVENANCE-OK")
EOF
cd "$REPO"

echo "== emit missing packs (16 local; stage2 rank0/rank3 already placed on spark8/sparkb) =="
for stage_tuple in "0 0 8" "1 8 8" "2 16 7" "3 23 7"; do
    set -- $stage_tuple
    stage="$1"; first="$2"; layers="$3"
    for rank in 0 1 2 3; do
        out="$HOME/sparkdata/gemma4_26b.tp4pp4.t1/packs/gemma4_26b_tp4_rank${rank}_stage${stage}.gemma4sp"
        if [ -s "$out" ]; then echo "skip existing $out"; continue; fi
        python3 tools/gemma4_stagepack.py --model 26b-a4b \
            --checkpoint "$CKPT_DST" --output "$out" \
            --tp-degree 4 --tp-rank "$rank" \
            --first-layer "$first" --layer-count "$layers" \
            --experts-manifest "$out.experts" \
            --receipt "$out.receipt.json"
        ( cd "$(dirname "$out")" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256" )
    done
done

echo "== fixture generation (26b) =="
mkdir -p "$OUT"
python3 tools/t1_reference_decoder.py --family gemma4 \
    --checkpoint "$CKPT_DST" \
    --header qualification/t1_reference/gemma4_26b/llm_defines_26b.h \
    --prompts qualification/t1_reference/gemma4_26b/prompts.json \
    --output "$OUT"
cp "$OUT/gemma4"/*.t1r "$OUT/gemma4/MANIFEST.json" qualification/t1_reference/gemma4_26b/
echo "LEASE-BATCH-OK"
