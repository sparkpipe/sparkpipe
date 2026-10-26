#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
	echo "usage: $0 PACK CHECKPOINT_DIR OUT_DIR REVISION" >&2
	exit 2
fi
pack=$1
checkpoint=$2
out=$3
revision=$4
root=$(cd "$(dirname "$0")/.." && pwd)
prompts=$root/qualification/metal/qwen38_27b/gate_prompts.json
control_pack=$out/control.pack
batches=${SPARK_LADDER_BATCHES:-1,2,4,8,16,32,64}
mkdir -p "$out/ref" "$out/metal" "$out/compare"
cd "$root"
make -s build/test_device_contract_metal build/test_metal_kernels build/device_bandwidth build/qwen38_27b_reference build/qwen38_27b_metal_run build/logit_compare
./build/test_device_contract_metal > "$out/device_contract.txt"
./build/test_metal_kernels "$out/kernels.json" > "$out/kernels.txt"
./build/device_bandwidth 16 "$pack" > "$out/bandwidth.json" 2> "$out/bandwidth.log"
if [ ! -f "$out/ref/greedy.ref.json" ]; then
	./build/qwen38_27b_reference "$checkpoint" "$prompts" "$out/ref" 128 2> "$out/ref.log"
	./build/qwen38_27b_reference "$checkpoint" "$prompts" "$out/ref" 0 fp32-activations 2> "$out/ref_c.log"
fi
./build/qwen38_27b_metal_run --pack "$pack" --prompts "$prompts" --mode teacher --out "$out/metal" --verify-pack > "$out/metal_teacher.log"
./build/qwen38_27b_metal_run --pack "$pack" --prompts "$prompts" --mode greedy --out "$out/metal" --new-tokens 128 > "$out/metal_greedy.log"
set -f
for control in control:layers.31.ffn_down:swap-rows control_layers:layers.*.ffn_down:swap-rows control_rotate:layers.31.ffn_down:rotate-rows; do
	label=${control%%:*}
	rest=${control#*:}
	mkdir -p "$out/$label"
	python3 -I tools/pack_tensor_control.py --pack "$pack" --tensor "${rest%:*}" --corruption "${rest##*:}" --output "$control_pack" --receipt "$out/$label/control.json"
	./build/qwen38_27b_metal_run --pack "$control_pack" --prompts "$prompts" --mode teacher --out "$out/$label" > "$out/${label}_teacher.log"
	rm -f "$control_pack"
done
set +f
./build/qwen38_27b_metal_run --pack "$pack" --prompts "$prompts" --mode ladder --batches "$batches" --new-tokens 128 --repeats 3 --out "$out/metal" > "$out/metal_ladder.log"
rm -f "$out/compare/compare.jsonl"
for name in $(python3 -I -c 'import json,sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"]))' "$prompts"); do
	split=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["prefill_positions"])' "$out/metal/$name.metal.json")
	./build/logit_compare "$prompts" "$name" 248320 "$out/ref/$name.ref_a.logits" "$out/ref/$name.ref_b.logits" floor "$split" >> "$out/compare/compare.jsonl"
	for label in metal control control_layers control_rotate; do
		./build/logit_compare "$prompts" "$name" 248320 "$out/ref/$name.ref_a.logits" "$out/$label/$name.metal.logits" "$label" "$split" "$out/ref/$name.ref_b.logits" >> "$out/compare/compare.jsonl"
		./build/logit_compare "$prompts" "$name" 248320 "$out/ref/$name.ref_b.logits" "$out/$label/$name.metal.logits" "${label}_vs_b" "$split" "$out/ref/$name.ref_a.logits" >> "$out/compare/compare.jsonl"
	done
	if [ -f "$out/ref/$name.ref_c.logits" ]; then
		./build/logit_compare "$prompts" "$name" 248320 "$out/ref/$name.ref_c.logits" "$out/ref/$name.ref_a.logits" fp32_activations_vs_a "$split" >> "$out/compare/compare.jsonl"
		./build/logit_compare "$prompts" "$name" 248320 "$out/ref/$name.ref_c.logits" "$out/metal/$name.metal.logits" fp32_activations_vs_metal "$split" >> "$out/compare/compare.jsonl"
	fi
done
./build/qwen38_27b_metal_run --pack "$pack" --prompts "$prompts" --mode bench --out "$out/metal" --new-tokens 256 --prefill-rows 2048 --repeats 3 > "$out/metal_bench.log"
python3 -I tools/qwen38_27b_metal_summary.py --out "$out" --revision "$revision" --prompts "$prompts"
