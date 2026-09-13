#!/bin/bash
# rocm_tu_host_proof.sh — DSV4 ROCm island FULL-TU host proof (tier 1.5).
#
# LOCAL PROOF TOOLING. Never compiled into any serving or validation image.
#
# What it proves, per translation unit:
#   - the whole TU parses as C++ with device annotations rendered inert;
#   - every __global__ body type-checks (helpers, intrinsics, shared use);
#   - every triple-angle launch's argument list type-checks, because the
#     launch is rewritten into SparkRocmHostProofLaunch(kernel, cfg..., args...)
#     whose variadic template deduces every argument expression.
# What it does NOT prove: device compilation (code generation, wave64 ISA,
# LDS budgets). That stays with
#   hipcc --offload-arch=gfx950 -fsyntax-only ...
# on a ROCm-equipped machine. Honest classification: static proof, measured
# here; hardware behavior unproven.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/dsv4_rocm_tu_proof.$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

TUS="spark_dsv4_rocm_kernels_common.hiph \
	spark_dsv4_rocm_islands_e0_embed.hip \
	spark_dsv4_rocm_islands_l1_boundary_norm_project.hip \
	spark_dsv4_rocm_islands_l2_attention_qkv.hip \
	spark_dsv4_rocm_islands_l3_cache_transition.hip \
	spark_dsv4_rocm_islands_l4_moe_routed.hip \
	spark_dsv4_rocm_islands_l5_moe_shared.hip \
	spark_dsv4_rocm_islands_f1_head.hip"

rc=0
for tu in $TUS; do
	proof_c="$TMP/${tu}.cpp"
	if command -v perl >/dev/null 2>&1; then
		perl -0777 -pe \
			's/([A-Za-z_][A-Za-z_0-9]*)\s*<<<(.*?)>>>\s*\(/SparkRocmHostProofLaunch($1, $2, /gs' \
			"$DIR/$tu" > "$proof_c"
	else
		cp "$DIR/$tu" "$proof_c"
	fi
	if g++ -std=c++17 -fsyntax-only -Wall -Wextra \
		-I"$REPO/include" -I"$DIR" \
		-include "$DIR/rocm_tu_host_proof_shim.h" \
		-x c++ "$proof_c" 2> "$TMP/$tu.err"; then
		echo "PROOF OK   $tu"
	else
		echo "PROOF FAIL $tu"
		cat "$TMP/$tu.err"
		rc=1
	fi
done
exit $rc
