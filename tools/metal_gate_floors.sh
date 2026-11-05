#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
	echo "usage: $0 GATE_OUT_DIR PROMPTS_JSON VOCAB" >&2
	exit 2
fi
out=$1
prompts=$2
vocab=$3
names=$(python3 -I -c 'import json,sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"]))' "$prompts")
variants=""
for variant in a b h; do
	present=1
	for name in $names; do
		[ -f "$out/ref/$name.ref_$variant.logits" ] || present=0
	done
	[ "$present" = 1 ] && variants="$variants $variant"
done
set -- $variants
if [ "$#" -lt 3 ]; then
	echo "multi-floor test: reference variants a, b and h are needed, found:$variants" >&2
	exit 0
fi
mkdir -p "$out/compare/kl"
python3 -I -c 'import sys; path = sys.argv[1]; lines = [line for line in open(path) if "\"label\":\"mf:" not in line]; open(path, "w").writelines(lines)' "$out/compare/compare.jsonl"
for name in $names; do
	split=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["prefill_positions"])' "$out/metal/$name.metal.json")
	for reference in $variants; do
		for other in $variants; do
			[ "$other" = "$reference" ] && continue
			./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_$reference.logits" "$out/ref/$name.ref_$other.logits" \
				"mf:$reference:$other" "$split" - "$out/compare/kl/$name.$reference.$other.f64" >> "$out/compare/compare.jsonl"
		done
		for label in metal metal_rows64 control control_layers control_rotate control_experts; do
			[ -f "$out/$label/$name.metal.logits" ] || continue
			./build/logit_compare "$prompts" "$name" "$vocab" "$out/ref/$name.ref_$reference.logits" "$out/$label/$name.metal.logits" \
				"mf:$reference:$label" "$split" - "$out/compare/kl/$name.$reference.$label.f64" >> "$out/compare/compare.jsonl"
		done
	done
done
