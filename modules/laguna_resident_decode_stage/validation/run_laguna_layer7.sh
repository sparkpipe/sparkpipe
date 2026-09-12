#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
	echo "usage: run_laguna_layer7.sh REPO_ROOT WORK_DIR PACKS_DIR CHECKPOINT_DIR" >&2
	exit 2
fi

REPO=$(cd "$1" && pwd)
WORK=$(mkdir -p "$2" && cd "$2" && pwd)
PACKS=$(cd "$3" && pwd)
CKPT=$(cd "$4" && pwd)
DUMPS="$WORK/dumps"
LOG="$WORK/run.log"
mkdir -p "$DUMPS"
: > "$LOG"

gpu() {
	sudo -n sparkcap --mem 8192 "$@" 2>&1 | tee -a "$LOG"
}

nvcc -O2 -std=c++17 -gencode arch=compute_121a,code=sm_121a -DLAGUNA_EXPERT_WEIGHT_CODEC=1 \
	-DLAGUNA_EXPERT_CODEC_NAME='"bf16"' \
	-I"$REPO" -I"$REPO/include" \
	-I"$REPO/model-families/laguna/include" \
	-I"$REPO/model-families/common/include" \
	-I"$REPO/modules/laguna_resident_decode_stage/include" \
	-o "$WORK/laguna_layer7" \
	"$REPO/modules/laguna_resident_decode_stage/validation/laguna_layer7_realpack.cu" -lcuda 2>&1 | tee -a "$LOG"
echo "BUILD OK" | tee -a "$LOG"

python3 - "$WORK" <<'PYEOF'
import json
import sys

work = sys.argv[1]
ids = [(i * 7919 + 13) % 12544 for i in range(520)]
json.dump({"token_ids": ids}, open(work + "/tokens.json", "w"))
open(work + "/tokens_t10.txt", "w").write(",".join(str(i) for i in ids[:10]))
open(work + "/tokens_t520.txt", "w").write(",".join(str(i) for i in ids))
PYEOF

T10=$(cat "$WORK/tokens_t10.txt")
T520=$(cat "$WORK/tokens_t520.txt")
PACK0="$PACKS/laguna_stage.tp8.pp2.stage0"
PACK1="$PACKS/laguna_stage.tp8.pp2.stage1"

gpu "$WORK/laguna_layer7" embed --pack "$PACK0.rank0.lgsp" --out "$WORK/hidden_t10.bin" --positions "$T10"
gpu "$WORK/laguna_layer7" embed --pack "$PACK0.rank0.lgsp" --out "$WORK/hidden_t520.bin" --positions "$T520"

attn_run() {
	local stage=$1 layer=$2 rank=$3 tokens=$4
	local pack tag
	if (( stage == 0 )); then pack="$PACK0.rank$rank.lgsp"; else pack="$PACK1.rank$rank.lgsp"; fi
	tag="s${stage}l${layer}r${rank}_t${tokens}"
	if (( tokens == 10 )); then hidden="$WORK/hidden_t10.bin"; else hidden="$WORK/hidden_t520.bin"; fi
	gpu "$WORK/laguna_layer7" attn --pack "$pack" --prefix "$DUMPS/$tag" --hidden "$hidden" \
		--layer "$layer" --tokens "$tokens" --rank "$rank" --stage "$stage"
}

for rank in 0 1 2 3 4 5 6 7; do
	attn_run 0 1 "$rank" 10
	attn_run 0 4 "$rank" 10
	attn_run 1 24 "$rank" 10
	attn_run 1 25 "$rank" 10
done
attn_run 0 0 0 10
attn_run 0 0 7 10
attn_run 0 23 0 10
attn_run 0 23 7 10
attn_run 0 1 0 520
attn_run 0 4 0 520
attn_run 1 25 0 520
attn_run 1 24 0 520

python3 - "$WORK" <<'PYEOF'
import numpy as np
import sys

work = sys.argv[1]


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16_from_f32(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


HIDDEN = 3072
for stage, layer in ((0, 1), (0, 4), (1, 24), (1, 25)):
    total = np.zeros((10, HIDDEN), dtype=np.float64)
    for rank in range(8):
        bits = np.fromfile("%s/dumps/s%dl%dr%d_t10_opart.bin" % (work, stage, layer, rank), dtype=np.uint16)
        assert bits.size == 10 * HIDDEN, (stage, layer, rank, bits.size)
        total += bf16_to_f32(bits).reshape(total.shape)
    bf16_from_f32(total.astype(np.float32)).tofile("%s/attnfull_s%dl%d_t10.bin" % (work, stage, layer))
    print("attnfull s%dl%d combined" % (stage, layer))
PYEOF

moe_run() {
	local stage=$1 layer=$2 rank=$3
	local pack
	if (( stage == 0 )); then pack="$PACK0.rank$rank.lgsp"; else pack="$PACK1.rank$rank.lgsp"; fi
	gpu "$WORK/laguna_layer7" moe --pack "$pack" --prefix "$DUMPS/s${stage}l${layer}r${rank}_t10" \
		--hidden "$WORK/hidden_t10.bin" --attn "$WORK/attnfull_s${stage}l${layer}_t10.bin" \
		--layer "$layer" --tokens 10 --rank "$rank" --stage "$stage"
}

for rank in 0 1 2 3 4 5 6 7; do
	moe_run 0 1 "$rank"
	moe_run 0 4 "$rank"
	moe_run 1 24 "$rank"
	moe_run 1 25 "$rank"
done

gpu "$WORK/laguna_layer7" rope --prefix "$DUMPS/rope_full" \
	--positions "0,1,511,512,513,1023,1024,4095,4096,8191,8192,8193,16383,16384,65536,1048575"
gpu "$WORK/laguna_layer7" rope --prefix "$DUMPS/rope_sliding" \
	--positions "0,1,511,512,513,1023,1024,4095,4096"

python3 - "$WORK" <<'PYEOF'
import json
import sys

work = sys.argv[1]
rope_full_positions = [0, 1, 511, 512, 513, 1023, 1024, 4095, 4096, 8191, 8192, 8193, 16383, 16384, 65536, 1048575]
rope_sliding_positions = [0, 1, 511, 512, 513, 1023, 1024, 4095, 4096]
plan = {
    "token_ids": json.load(open(work + "/tokens.json"))["token_ids"],
    "attn_runs": (
        [{"stage": 0, "layer": 1, "rank": r, "count": 10} for r in range(8)] +
        [{"stage": 0, "layer": 4, "rank": r, "count": 10} for r in range(8)] +
        [{"stage": 1, "layer": 24, "rank": r, "count": 10} for r in range(8)] +
        [{"stage": 1, "layer": 25, "rank": r, "count": 10} for r in range(8)] +
        [{"stage": 0, "layer": 0, "rank": 0, "count": 10}, {"stage": 0, "layer": 0, "rank": 7, "count": 10},
         {"stage": 0, "layer": 23, "rank": 0, "count": 10}, {"stage": 0, "layer": 23, "rank": 7, "count": 10},
         {"stage": 0, "layer": 1, "rank": 0, "count": 520}, {"stage": 0, "layer": 4, "rank": 0, "count": 520},
         {"stage": 1, "layer": 25, "rank": 0, "count": 520}, {"stage": 1, "layer": 24, "rank": 0, "count": 520}]
    ),
    "combine_sources": [
        {"stage": 0, "layer": 1, "count": 10}, {"stage": 0, "layer": 4, "count": 10},
        {"stage": 1, "layer": 24, "count": 10}, {"stage": 1, "layer": 25, "count": 10}],
    "moe_runs": [
        {"stage": 0, "layer": 1, "count": 10}, {"stage": 0, "layer": 4, "count": 10},
        {"stage": 1, "layer": 24, "count": 10}, {"stage": 1, "layer": 25, "count": 10}],
    "rope_runs": [
        {"regime": "full", "positions": rope_full_positions},
        {"regime": "sliding", "positions": rope_sliding_positions}],
}
json.dump(plan, open(work + "/plan.json", "w"), indent=1)
print("plan written")
PYEOF

python3 "$REPO/modules/laguna_resident_decode_stage/validation/laguna_layer7_reference.py" \
	--checkpoint "$CKPT" --dumps "$DUMPS" --plan "$WORK/plan.json" 2>&1 | tee -a "$LOG"

mkdir -p "$WORK/retained"
cp "$LOG" "$WORK/plan.json" "$WORK/tokens.json" "$WORK/retained/" 2>/dev/null || true
echo "RUNNER DONE" | tee -a "$LOG"
