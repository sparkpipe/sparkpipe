#!/bin/bash
# validate_rocm_syntax.sh — DSV4 ROCm island sources
#
# Tier 1 (runs here, MEASURED): shimmed g++ -fsyntax-only over the shared
#   helper header — proves the bf16/reduction/GEMV helper surface parses and
#   type-checks as C++ without any HIP installation.
# Tier 2 (requires hipcc, NOT runnable on this box): the real syntax proof
#   for both island translation units, exactly as they will be compiled.
#
# Honest-classification rule (hwiface culture): anything below that has not
# RUN must be reported as unproven, never green-checked.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../../../.." && pwd)"

echo "== Tier 1: shimmed host syntax proof (common header) =="
g++ -std=c++17 -fsyntax-only -Wall -Wextra \
	-I"$REPO/include" \
	-include "$DIR/rocm_syntax_shim.h" \
	-x c++ "$DIR/spark_dsv4_rocm_kernels_common.hiph"
echo "tier1 OK"

if command -v hipcc >/dev/null 2>&1; then
	echo "== Tier 2: hipcc syntax proof (island TUs) =="
	for tu in spark_dsv4_rocm_islands_e0_embed.hip \
		spark_dsv4_rocm_islands_l1_boundary_norm_project.hip \
		spark_dsv4_rocm_islands_l2_attention_qkv.hip \
		spark_dsv4_rocm_islands_l5_moe_shared.hip \
		spark_dsv4_rocm_islands_f1_head.hip \
		spark_dsv4_rocm_islands_l3_cache_transition.hip \
		spark_dsv4_rocm_islands_l4_moe_routed.hip; do
		hipcc --offload-arch=gfx950 -fsyntax-only \
			-I"$REPO/include" -I"$DIR" "$DIR/$tu"
		echo "$tu OK"
	done
else
	echo "Tier 2 SKIPPED: hipcc not found on this box (documented limitation)."
	echo "Run on any ROCm-equipped machine:"
	echo "  hipcc --offload-arch=gfx950 -fsyntax-only -I<repo>/include -I<this-dir> <this-dir>/spark_dsv4_rocm_islands_l1_boundary_norm_project.hip"
	echo "  hipcc --offload-arch=gfx950 -fsyntax-only -I<repo>/include -I<this-dir> <this-dir>/spark_dsv4_rocm_islands_l2_attention_qkv.hip"
	echo "  hipcc --offload-arch=gfx950 -fsyntax-only -I<repo>/include -I<this-dir> <this-dir>/spark_dsv4_rocm_islands_f1_head.hip"
fi
