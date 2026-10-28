#!/bin/sh
out=$HOME/build/cost
rm -rf $out; mkdir -p $out
before=$HOME/build/revs/rev-ff51be851
after=$HOME/build/g31ns
for round in 1 2; do
	for model in qwen gemma26 gemma31; do
		case $model in
			qwen) args="--pack $HOME/models/packs/qwen38_27b_tp1_bf16/qwen38_27b.tp1.rank00.pack"; prompts=qualification/metal/qwen38_27b/gate_prompts.json;;
			gemma26) args="--model gemma4-26b-a4b --pack $HOME/models/packs/gemma4_26b_tp1_bf16/gemma4_26b.tp1.rank00.pack"; prompts=qualification/metal/gemma4_26b/gate_prompts.json;;
			gemma31) args="--model gemma4-31b --pack $HOME/models/packs/gemma4_31b_tp1_bf16/gemma4.tp1.rank00.pack"; prompts=qualification/metal/gemma4_31b/gate_prompts.json;;
		esac
		for build in before after; do
			eval dir=\$$build
			cd $dir
			./build/metal_stage_run $args --prompts $prompts --mode bench --out $out/$model.$build.$round --new-tokens 256 --prefill-rows 2048 --repeats 3 > $out/$model.$build.$round.log 2>&1
			echo "$model $build round $round: $(grep -E 'decode median|prefill 8192 run 3|prefill 1024 run 3' $out/$model.$build.$round.log | tr '\n' ' ')" >> $out/summary.txt
		done
	done
done
echo done >> $out/summary.txt
