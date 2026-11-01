#!/usr/bin/env bash
set -eu
cd ~/drafter-train
SRC=src/tools/spec_offline
BASE=weights/kimi-k3-dflash2-lightseek-e77935fb
SHARD=weights/kimi-k3/model-00094-of-000096.safetensors
TAPS=data/k3-edits-20261016.spt2
RUN=runs/${1:?run name}
STEPS=${STEPS:-300}
LR=${LR:-2e-5}
ANCHORS=${ANCHORS:-48}
mkdir -p "$RUN"
score() {
	python3 $SRC/block_drafter.py --drafter "$1" --target-shard $SHARD --taps $TAPS --requests "$3" --region "$4" --max-anchors 400 \
		--out "$RUN/score-$2-$5-$4.json" > "$RUN/score-$2-$5-$4.log" 2>&1
	python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], sys.argv[3], sys.argv[4], 'tokens_per_round', round(r['tokens_per_round'], 3), 'anchors', sum(q['anchors'] for q in r['requests']))" \
		"$RUN/score-$2-$5-$4.json" "$2" "$5" "$4" | tee -a "$RUN/summary.txt"
}
arms() {
	score "$1" "$2" 100011-100020 reply test
	score "$1" "$2" 100011-100020 prompt test
	score "$1" "$2" 100001-100010 reply train
	score "$1" "$2" 100001-100010 prompt train
	score "$1" "$2" 100022-100031 reply story
}
if [ ! -f runs/base-summary.txt ]; then
	RUN_SAVED=$RUN; RUN=runs/base; mkdir -p $RUN; arms $BASE base; cp $RUN/summary.txt runs/base-summary.txt; RUN=$RUN_SAVED
fi
python3 $SRC/latent_block_train.py train --drafter $BASE --target-shard $SHARD --taps $TAPS --requests 100001-100010 \
	--steps "$STEPS" --anchors "$ANCHORS" --lr "$LR" --out "$RUN/tuned" 2>&1 | tee "$RUN/train.log" | grep -E "TRAIN-(DATA|DONE)|TRAIN-STEP [0-9]*0 "
arms "$RUN/tuned" tuned
cat runs/base-summary.txt "$RUN/summary.txt"
