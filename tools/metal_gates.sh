#!/bin/sh
set -eu

if [ "$#" -ne 5 ]; then
	echo "usage: [SPARK_GATE_TP=N [SPARK_GATE_STREAM=1]] [SPARK_GATE_DECODE=N] [SPARK_GATE_GREEDY=N] $0 gemma4-26b-a4b|gemma4-31b|muse-glimmer-30b|mimo-v2.6-flash|mimo-v2.6-pro|qwen3.8-flash|laguna-s-2.1|k-exaone-2.0|ling-3.0-flash|ling-3.0-flash-fin|solar-open-2|hy4-preview|glm-5.3-flash PACK REFERENCE_LOGITS_DIR OUT_DIR REVISION" >&2
	exit 2
fi
model=$1
pack=$2
reference=$3
out=$4
revision=$5
case $model in
	gemma4-26b-a4b) routed=1 prompt_dir=gemma4 vocab=262144 probe=ffn_down layer=11;;
	gemma4-31b) routed=0 prompt_dir=gemma4 vocab=262144 probe=ffn_down layer=11;;
	muse-glimmer-30b) routed=0 prompt_dir=muse_glimmer vocab=202048 probe=ffn_down layer=11;;
	mimo-v2.6-flash) routed=0 prompt_dir=mimo26_flash vocab=152576 probe=attn_out layer=11;;
	mimo-v2.6-pro) routed=0 prompt_dir=mimo26_flash vocab=152576 probe=attn_out layer=7;;
	qwen3.8-flash) routed=0 prompt_dir=qwen38_27b vocab=248320 probe=moe_shared_down layer=11 greedy_tokens=8 bench_prompt=design spec=0 bench_long=2048;;
	laguna-s-2.1) routed=0 prompt_dir=laguna vocab=100352 probe=attn_out layer=11;;
	k-exaone-2.0) routed=0 prompt_dir=kexaone2 vocab=153600 probe=attn_out layer=11;;
	ling-3.0-flash|ling-3.0-flash-fin) routed=0 prompt_dir=ling vocab=157184 probe=kda_out layer=12;;
	solar-open-2) routed=0 prompt_dir=solar_open2 vocab=196608 probe=kda_out layer=13;;
	hy4-preview) routed=0 prompt_dir=hy4 vocab=120832 probe=gate layer=11;;
	glm-5.3-flash) routed=0 prompt_dir=glm53flash vocab=154880 probe=kda_out layer=12;;
	*) echo "unknown Metal model $model" >&2; exit 2;;
esac
greedy_tokens=${SPARK_GATE_GREEDY:-${greedy_tokens:-128}}
teacher_flags=${SPARK_GATE_DECODE:+--decode-positions $SPARK_GATE_DECODE}
bench_prompt=${bench_prompt:-}
bench_long=${bench_long:-8192}
spec=${spec:-1}
root=$(cd "$(dirname "$0")/.." && pwd)
prompts=$root/qualification/metal/$prompt_dir/gate_prompts.json
spec_prompts=$root/qualification/metal/$prompt_dir/spec_prompts.json
batches=${SPARK_LADDER_BATCHES:-1,2,4,8,16,32,64}
run="./build/metal_stage_run --model $model --prompts $prompts"
tp=${SPARK_GATE_TP:-1}
control_dir=${SPARK_GATE_CONTROL_DIR:-$out/controlpack}
rank_pack() {
	if [ "$1" = 0 ]; then echo "$pack"; else echo "$(dirname "$pack")/$(basename "$pack" | sed "s/rank00/rank$(printf %02d "$1")/")"; fi
}
if [ "$tp" -gt 1 ]; then
	run="$run --tp $tp"
	[ "${SPARK_GATE_STREAM:-0}" = 1 ] && run="$run --stream-weights"
	spec=0
fi
mkdir -p "$control_dir"
control_pack=$control_dir/$(basename "$pack")
mkdir -p "$out/metal" "$out/exact" "$out/compare"
ln -sfn "$reference" "$out/ref"
cd "$root"
make -s build/test_device_contract_metal build/test_metal_bf16_convert build/test_metal_kernels build/device_bandwidth build/metal_stage_run build/logit_compare
./build/test_device_contract_metal > "$out/device_contract.txt"
./build/test_metal_bf16_convert > "$out/bf16_convert.txt"
./build/test_metal_kernels "$out/kernels.json" > "$out/kernels.txt"
./build/device_bandwidth 16 "$pack" > "$out/bandwidth.json" 2> "$out/bandwidth.log"
$run --pack "$pack" --mode teacher --out "$out/metal" --verify-pack $teacher_flags > "$out/metal_teacher.log"
controls="control:layers.$layer.$probe:swap-rows control_layers:layers.*.$probe:swap-rows control_rotate:layers.$layer.$probe:rotate-rows"
pin=""
if [ "$routed" = 1 ]; then
	mkdir -p "$out/pinned" "$out/pinned64"
	pin="--pin-routes $out/ref"
	$run --pack "$pack" --mode teacher --out "$out/pinned" $pin $teacher_flags > "$out/pinned_teacher.log"
	$run --pack "$pack" --mode teacher --out "$out/pinned64" $pin --prefill-rows 64 $teacher_flags > "$out/pinned64_teacher.log"
	controls="$controls control_experts:layers.*.moe_w2:swap-rows"
fi
$run --pack "$pack" --mode greedy --out "$out/metal" --new-tokens "$greedy_tokens" > "$out/metal_greedy.log"
set -f
for control in $controls; do
	label=${control%%:*}
	rest=${control#*:}
	mkdir -p "$out/$label"
	rank=0
	while [ "$rank" -lt "$tp" ]; do
		receipt=$out/$label/control.json
		[ "$rank" = 0 ] || receipt=$out/$label/control.rank$(printf %02d "$rank").json
		python3 -I tools/pack_tensor_control.py --pack "$(rank_pack "$rank")" --tensor "${rest%:*}" --corruption "${rest##*:}" \
			--output "$control_dir/$(basename "$(rank_pack "$rank")")" --receipt "$receipt"
		rank=$((rank + 1))
	done
	$run --pack "$control_pack" --mode teacher --out "$out/$label" $pin $teacher_flags > "$out/${label}_teacher.log"
	rank=0
	while [ "$rank" -lt "$tp" ]; do
		rm -f "$control_dir/$(basename "$(rank_pack "$rank")")"
		rank=$((rank + 1))
	done
done
set +f
if [ "$tp" = 1 ]; then
	$run --pack "$pack" --mode ladder --batches "$batches" --new-tokens 128 --repeats 3 --out "$out/metal" > "$out/metal_ladder.log"
	$run --pack "$pack" --mode ladder --batches "$batches" --new-tokens 128 --repeats 1 --exact-rows --out "$out/exact" > "$out/exact_ladder.log"
fi
rm -f "$out/compare/compare.jsonl"
for name in $(python3 -I -c 'import json,sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"]))' "$prompts"); do
	split=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["prefill_positions"])' "$out/metal/$name.metal.json")
	./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_a.logits" "$out/ref/$name.ref_b.logits" floor "$split" >> "$out/compare/compare.jsonl"
	if [ "$routed" = 1 ]; then
		./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_a.logits" "$out/metal/$name.metal.logits" metal "$split" "$out/ref/$name.ref_b.logits" >> "$out/compare/compare.jsonl"
		./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_b.logits" "$out/metal/$name.metal.logits" metal_vs_b "$split" "$out/ref/$name.ref_a.logits" >> "$out/compare/compare.jsonl"
		./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_a.logits" "$out/ref/$name.ref_bp.logits" pinned_floor "$split" >> "$out/compare/compare.jsonl"
		for pass in pinned:metal pinned64:metal_rows64 control:control control_layers:control_layers control_rotate:control_rotate control_experts:control_experts; do
			dir=${pass%%:*}
			tag=pinned_${pass#*:}
			./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_a.logits" "$out/$dir/$name.metal.logits" "$tag" "$split" "$out/ref/$name.ref_bp.logits" >> "$out/compare/compare.jsonl"
			./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_bp.logits" "$out/$dir/$name.metal.logits" "${tag}_vs_b" "$split" "$out/ref/$name.ref_a.logits" >> "$out/compare/compare.jsonl"
		done
	else
		for label in metal control control_layers control_rotate; do
			./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_a.logits" "$out/$label/$name.metal.logits" "$label" "$split" "$out/ref/$name.ref_b.logits" >> "$out/compare/compare.jsonl"
			./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_b.logits" "$out/$label/$name.metal.logits" "${label}_vs_b" "$split" "$out/ref/$name.ref_a.logits" >> "$out/compare/compare.jsonl"
		done
	fi
done
if [ "$tp" = 1 ]; then
	$run --pack "$pack" --mode bench --out "$out/metal" --new-tokens 256 --prefill-rows 2048 --repeats 3 ${bench_prompt:+--bench-prompt "$bench_prompt"} \
		--bench-long-tokens "$bench_long" > "$out/metal_bench.log"
fi
if [ "$spec" = 1 ]; then
	$run --pack "$pack" --spec-prompts "$spec_prompts" --mode spec --new-tokens 256 --repeats 3 --draft-max 31 --out "$out/metal" > "$out/metal_spec.log"
fi
python3 -I tools/metal_gate_summary.py --out "$out" --revision "$revision" --prompts "$prompts" --tp-degree "$tp"
