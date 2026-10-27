#!/bin/sh
set -eu

if [ "$#" -ne 5 ]; then
	echo "usage: $0 gemma4-26b-a4b|gemma4-31b PACK REFERENCE_LOGITS_DIR OUT_DIR REVISION" >&2
	exit 2
fi
model=$1
pack=$2
reference=$3
out=$4
revision=$5
case $model in
	gemma4-26b-a4b) receipts=gemma4_26b; routed=1; draws="";;
	gemma4-31b) receipts=gemma4_31b; routed=0; draws="b c d";;
	*) echo "unknown model $model" >&2; exit 2;;
esac
root=$(cd "$(dirname "$0")/.." && pwd)
prompts=$root/qualification/metal/$receipts/gate_prompts.json
spec_prompts=$root/qualification/metal/$receipts/spec_prompts.json
control_pack=$out/control.pack
batches=${SPARK_LADDER_BATCHES:-1,2,4,8,16,32,64}
run="./build/metal_stage_run --model $model --prompts $prompts"
pin=""
[ "$routed" = 1 ] && pin="--pin-routes $out/ref"
mkdir -p "$out/metal" "$out/exact" "$out/compare"
ln -sfn "$reference" "$out/ref"
for name in $(python3 -I -c 'import json,sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"]))' "$prompts"); do
	for draw in $draws; do
		if [ ! -f "$out/ref/$name.ref_$draw.logits" ]; then
			echo "the $model gate scores floor draws $draws: $out/ref/$name.ref_$draw.logits is missing (tools/gemma4_reference.py teacher --variants)" >&2
			exit 2
		fi
	done
done
cd "$root"
make -s build/test_device_contract_metal build/test_metal_bf16_convert build/test_metal_math_accuracy build/test_metal_kernels build/device_bandwidth build/metal_stage_run build/logit_compare
./build/test_device_contract_metal > "$out/device_contract.txt"
./build/test_metal_bf16_convert > "$out/bf16_convert.txt"
./build/test_metal_math_accuracy > "$out/math_accuracy.txt"
./build/test_metal_kernels "$out/kernels.json" > "$out/kernels.txt"
./build/device_bandwidth 16 "$pack" > "$out/bandwidth.json" 2> "$out/bandwidth.log"
$run --pack "$pack" --mode teacher --out "$out/metal" --verify-pack > "$out/metal_teacher.log"
if [ "$routed" = 1 ]; then
	mkdir -p "$out/pinned" "$out/pinned64"
	$run --pack "$pack" --mode teacher --out "$out/pinned" $pin > "$out/pinned_teacher.log"
	$run --pack "$pack" --mode teacher --out "$out/pinned64" $pin --prefill-rows 64 > "$out/pinned64_teacher.log"
else
	mkdir -p "$out/metal64"
	$run --pack "$pack" --mode teacher --out "$out/metal64" --prefill-rows 64 > "$out/metal64_teacher.log"
fi
$run --pack "$pack" --mode greedy --out "$out/metal" --new-tokens 128 > "$out/metal_greedy.log"
controls="control:layers.11.ffn_down:swap-rows control_layers:layers.*.ffn_down:swap-rows control_rotate:layers.11.ffn_down:rotate-rows"
[ "$routed" = 1 ] && controls="$controls control_experts:layers.*.moe_w2:swap-rows"
set -f
for control in $controls; do
	label=${control%%:*}
	rest=${control#*:}
	mkdir -p "$out/$label"
	python3 -I tools/pack_tensor_control.py --pack "$pack" --tensor "${rest%:*}" --corruption "${rest##*:}" --output "$control_pack" --receipt "$out/$label/control.json"
	$run --pack "$control_pack" --mode teacher --out "$out/$label" $pin > "$out/${label}_teacher.log"
	rm -f "$control_pack"
done
set +f
$run --pack "$pack" --mode ladder --batches "$batches" --new-tokens 128 --repeats 3 --out "$out/metal" > "$out/metal_ladder.log"
$run --pack "$pack" --mode ladder --batches "$batches" --new-tokens 128 --repeats 1 --exact-rows --out "$out/exact" > "$out/exact_ladder.log"
rm -f "$out/compare/compare.jsonl"
for name in $(python3 -I -c 'import json,sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"]))' "$prompts"); do
	split=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["prefill_positions"])' "$out/metal/$name.metal.json")
	./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/ref/$name.ref_b.logits" floor "$split" >> "$out/compare/compare.jsonl"
	if [ "$routed" = 1 ]; then
		free="metal:metal"
		pinned="pinned:metal pinned64:metal_rows64 control:control control_layers:control_layers control_rotate:control_rotate control_experts:control_experts"
		./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/ref/$name.ref_bp.logits" pinned_floor "$split" >> "$out/compare/compare.jsonl"
	else
		free="metal:metal metal64:metal_rows64 control:control control_layers:control_layers control_rotate:control_rotate"
		pinned=""
	fi
	for draw in $draws; do
		for floor in $draws; do
			[ "$draw" = "$floor" ] && continue
			./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/ref/$name.ref_$draw.logits" "draw_${draw}_vs_floor_$floor" "$split" "$out/ref/$name.ref_$floor.logits" >> "$out/compare/compare.jsonl"
		done
	done
	for pass in $free; do
		dir=${pass%%:*}
		tag=${pass#*:}
		./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/$dir/$name.metal.logits" "$tag" "$split" "$out/ref/$name.ref_b.logits" >> "$out/compare/compare.jsonl"
		./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_b.logits" "$out/$dir/$name.metal.logits" "${tag}_vs_b" "$split" "$out/ref/$name.ref_a.logits" >> "$out/compare/compare.jsonl"
		for floor in $draws; do
			./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/$dir/$name.metal.logits" "${tag}_vs_floor_$floor" "$split" "$out/ref/$name.ref_$floor.logits" >> "$out/compare/compare.jsonl"
		done
	done
	for pass in $pinned; do
		dir=${pass%%:*}
		tag=pinned_${pass#*:}
		./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/$dir/$name.metal.logits" "$tag" "$split" "$out/ref/$name.ref_bp.logits" >> "$out/compare/compare.jsonl"
		./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_bp.logits" "$out/$dir/$name.metal.logits" "${tag}_vs_b" "$split" "$out/ref/$name.ref_a.logits" >> "$out/compare/compare.jsonl"
	done
done
$run --pack "$pack" --mode bench --out "$out/metal" --new-tokens 256 --prefill-rows 2048 --repeats 3 > "$out/metal_bench.log"
$run --pack "$pack" --spec-prompts "$spec_prompts" --mode spec --new-tokens 256 --repeats 3 --draft-max 31 --out "$out/metal" > "$out/metal_spec.log"
python3 -I tools/metal_gate_summary.py --out "$out" --revision "$revision" --prompts "$prompts"
