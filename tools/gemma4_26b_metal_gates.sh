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
control_pack=$out/control.pack
batches=${SPARK_LADDER_BATCHES:-1,2,4,8,16,32,64}
run="./build/qwen38_27b_metal_run --model gemma4-26b-a4b --prompts $prompts"
mkdir -p "$out/metal" "$out/compare"
ln -sfn "$reference" "$out/ref"
cd "$root"
make -s build/test_device_contract_metal build/test_metal_kernels build/device_bandwidth build/qwen38_27b_metal_run build/logit_compare
./build/test_device_contract_metal > "$out/device_contract.txt"
./build/test_metal_kernels "$out/kernels.json" > "$out/kernels.txt"
./build/device_bandwidth 16 "$pack" > "$out/bandwidth.json" 2> "$out/bandwidth.log"
$run --pack "$pack" --mode teacher --out "$out/metal" --verify-pack > "$out/metal_teacher.log"
$run --pack "$pack" --mode greedy --out "$out/metal" --new-tokens 128 > "$out/metal_greedy.log"
set -f
for control in control:layers.11.ffn_down:swap-rows control_layers:layers.*.ffn_down:swap-rows control_rotate:layers.11.ffn_down:rotate-rows; do
	label=${control%%:*}
	rest=${control#*:}
	mkdir -p "$out/$label"
	python3 -I tools/pack_tensor_control.py --pack "$pack" --tensor "${rest%:*}" --corruption "${rest##*:}" --output "$control_pack" --receipt "$out/$label/control.json"
	$run --pack "$control_pack" --mode teacher --out "$out/$label" > "$out/${label}_teacher.log"
	rm -f "$control_pack"
done
set +f
$run --pack "$pack" --mode ladder --batches "$batches" --new-tokens 128 --repeats 3 --out "$out/metal" > "$out/metal_ladder.log"
rm -f "$out/compare/compare.jsonl"
for name in $(python3 -I -c 'import json,sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"]))' "$prompts"); do
	split=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["prefill_positions"])' "$out/metal/$name.metal.json")
	./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/ref/$name.ref_b.logits" floor "$split" >> "$out/compare/compare.jsonl"
	for label in metal control control_layers control_rotate; do
		./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_a.logits" "$out/$label/$name.metal.logits" "$label" "$split" "$out/ref/$name.ref_b.logits" >> "$out/compare/compare.jsonl"
		./build/logit_compare "$prompts" "$name" 262144 "$out/ref/$name.ref_b.logits" "$out/$label/$name.metal.logits" "${label}_vs_b" "$split" "$out/ref/$name.ref_a.logits" >> "$out/compare/compare.jsonl"
	done
done
$run --pack "$pack" --mode bench --out "$out/metal" --new-tokens 256 --prefill-rows 2048 --repeats 3 > "$out/metal_bench.log"
python3 -I tools/qwen38_27b_metal_summary.py --out "$out" --revision "$revision" --prompts "$prompts"
