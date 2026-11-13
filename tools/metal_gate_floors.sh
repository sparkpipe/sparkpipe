#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
	echo "usage: $0 GATE_OUT_DIR PROMPTS_JSON VOCAB REFERENCE_DIR" >&2
	exit 2
fi
out=$1
prompts=$2
vocab=$3
ref_dir=$4
names=$(python3 -I -c 'import json,sys; print(" ".join(p["name"] for p in json.load(open(sys.argv[1]))["prompts"]))' "$prompts")
present() {
	for name in $names; do
		[ -f "$ref_dir/$name.ref_$1.logits" ] || return 1
	done
	return 0
}
floor_set() {
	tag=$1
	kl=$2
	set_variants=$3
	candidates=$4
	mkdir -p "$out/compare/$kl"
	for name in $names; do
		split=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["prefill_positions"])' "$out/metal/$name.metal.json")
		for reference in $set_variants; do
			for other in $set_variants; do
				[ "$other" = "$reference" ] && continue
				./build/logit_compare "$prompts" "$name" "$vocab" "$ref_dir/$name.ref_$reference.logits" "$ref_dir/$name.ref_$other.logits" \
					"$tag:$reference:$other" "$split" - "$out/compare/$kl/$name.$reference.$other.f64" >> "$out/compare/compare.jsonl"
			done
			for pair in $candidates; do
				dir=${pair%%:*}
				label=${pair#*:}
				[ -f "$out/$dir/$name.metal.logits" ] || continue
				./build/logit_compare "$prompts" "$name" "$vocab" "$ref_dir/$name.ref_$reference.logits" "$out/$dir/$name.metal.logits" \
					"$tag:$reference:$label" "$split" - "$out/compare/$kl/$name.$reference.$label.f64" >> "$out/compare/compare.jsonl"
			done
		done
	done
}
variants=""
for variant in a b h; do
	present "$variant" && variants="$variants $variant"
done
set -- $variants
if [ "$#" -lt 3 ]; then
	echo "multi-floor test: reference variants a, b and h are needed, found:$variants" >&2
	exit 0
fi
python3 -I -c 'import sys; path = sys.argv[1]; lines = [line for line in open(path) if "\"label\":\"mf:" not in line and "\"label\":\"pmf:" not in line]; open(path, "w").writelines(lines)' "$out/compare/compare.jsonl"
controls="control:control control_layers:control_layers control_rotate:control_rotate control_experts:control_experts"
floor_set mf kl "a b h" "metal:metal metal_rows64:metal_rows64 $controls"
if [ -d "$out/pinned" ] && present bp && present hp; then
	floor_set pmf kl-pinned "a bp hp" "pinned:metal pinned64:metal_rows64 $controls"
fi
