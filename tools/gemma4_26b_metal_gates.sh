#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
	echo "usage: $0 PACK REFERENCE_LOGITS_DIR OUT_DIR REVISION" >&2
	exit 2
fi
pack=$1
reference=$2
out=$3
revision=$4
root=$(cd "$(dirname "$0")/.." && pwd)
prompts=$root/qualification/metal/gemma4_26b/gate_prompts.json
spec_prompts=$root/qualification/metal/gemma4_26b/spec_prompts.json
control_pack=$out/control.pack
batches=${SPARK_LADDER_BATCHES:-1,2,4,8,16,32,64}
run="./build/qwen38_27b_metal_run --model gemma4-26b-a4b --prompts $prompts"
mkdir -p "$out/metal" "$out/pinned" "$out/pinned64" "$out/exact" "$out/compare"
ln -sfn "$reference" "$out/ref"
cd "$root"
make -s build/test_device_contract_metal build/test_metal_kernels build/device_bandwidth build/qwen38_27b_metal_run build/logit_compare
./build/test_device_contract_metal > "$out/device_contract.txt"
./build/test_metal_kernels "$out/kernels.json" > "$out/kernels.txt"
./build/device_bandwidth 16 "$pack" > "$out/bandwidth.json" 2> "$out/bandwidth.log"
$run --pack "$pack" --mode teacher --out "$out/metal" --verify-pack > "$out/metal_teacher.log"
$run --pack "$pack" --mode teacher --out "$out/pinned" --pin-routes "$out/ref" > "$out/pinned_teacher.log"
$run --pack "$pack" --mode teacher --out "$out/pinned64" --pin-routes "$out/ref" --prefill-rows 64 > "$out/pinned64_teacher.log"
$run --pack "$pack" --mode greedy --out "$out/metal" --new-tokens 128 > "$out/metal_greedy.log"
set -f
for control in control:layers.11.ffn_down:swap-rows control_layers:layers.*.ffn_down:swap-rows control_rotate:layers.11.ffn_down:rotate-rows; do
	label=${control%%:*}
	rest=${control#*:}
	mkdir -p "$out/$label"
	python3 -I tools/pack_tensor_control.py --pack "$pack" --tensor "${rest%:*}" --corruption "${rest##*:}" --output "$control_pack" --receipt "$out/$label/control.json"
	$run --pack "$control_pack" --mode teacher --out "$out/$label" --pin-routes "$out/ref" > "$out/${label}_teacher.log"
	rm -f "$control_pack"
done
set +f
$run --pack "$pack" --mode ladder --batches "$batches" --new-tokens 128 --repeats 3 --out "$out/metal" > "$out/metal_ladder.log"
$run --pack "$pack" --mode ladder --batches "$batches" --new-tokens 128 --repeats 1 --exact-rows --out "$out/exact" > "$out/exact_ladder.log"
rm -f "$out/compare/compare.jsonl"
for name in $(python3 -I -c 'import json,sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"]))' "$prompts"); do
	split=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["prefill_positions"])' "$out/metal/$name.metal.json")
	./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/ref/$name.ref_b.logits" floor "$split" >> "$out/compare/compare.jsonl"
	./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/metal/$name.metal.logits" metal "$split" "$out/ref/$name.ref_b.logits" >> "$out/compare/compare.jsonl"
	./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_b.logits" "$out/metal/$name.metal.logits" metal_vs_b "$split" "$out/ref/$name.ref_a.logits" >> "$out/compare/compare.jsonl"
	./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/ref/$name.ref_bp.logits" pinned_floor "$split" >> "$out/compare/compare.jsonl"
	for pass in pinned:metal pinned64:metal_rows64 control:control control_layers:control_layers control_rotate:control_rotate; do
		dir=${pass%%:*}
		tag=pinned_${pass#*:}
		./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/$dir/$name.metal.logits" "$tag" "$split" "$out/ref/$name.ref_bp.logits" >> "$out/compare/compare.jsonl"
		./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_bp.logits" "$out/$dir/$name.metal.logits" "${tag}_vs_b" "$split" "$out/ref/$name.ref_a.logits" >> "$out/compare/compare.jsonl"
	done
done
$run --pack "$pack" --mode bench --out "$out/metal" --new-tokens 256 --prefill-rows 2048 --repeats 3 > "$out/metal_bench.log"
$run --pack "$pack" --spec-prompts "$spec_prompts" --mode spec --new-tokens 256 --repeats 3 --draft-max 31 --out "$out/metal" > "$out/metal_spec.log"
python3 -I tools/qwen38_27b_metal_summary.py --out "$out" --revision "$revision" --prompts "$prompts"
