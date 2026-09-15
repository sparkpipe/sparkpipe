#!/bin/sh
# Rebuild ONE qwen38max TP16 nvfp4 rank pack from the warm checkpoint with the
# fixed builder (NVFP4_PACKED stamp + per-expert f32 tails), then verify it.
# Laws: sparkcap on the warm reads and the verify, fail loud, sums last.
set -eu
rank=$1
checkpoint=/mnt/model-warm/qwen3.8-max-nvfp4-radixark-bf16-spine
tools=/tmp/t1qmax_stage/tools
out=/tmp/t1qmax_rebuild/qwenmax.nvfp4.tp16.rank$rank.sp
mkdir -p /tmp/t1qmax_rebuild
cd "$tools"
sudo -n /usr/local/sbin/sparkcap python3 tools/qwen38_stagepack.py \
	--checkpoint "$checkpoint" \
	--output "$out" \
	--first-layer 0 --layer-count 92 \
	--strip-mtp --expert-codec nvfp4 \
	--tp-degree 16 --tp-rank "$rank" 2>&1 | tail -3
	sudo -n /usr/local/sbin/sparkcap python3 tools/qwen38max_tp16_rank_verify.py \
		--pack "$out" --tp-degree 16 --tp-rank "$rank" \
		--receipt "$out.receipt.json" --recompute-file-hash 2>&1 | tail -12
	/tmp/t1qmax_stage/qwen38max_experts_manifest "$out" || \
		sudo -n /usr/local/sbin/sparkcap /tmp/t1qmax_stage/qwen38max_experts_manifest "$out"
	ls -la "$out" "$out.experts" "$out.receipt.json"
