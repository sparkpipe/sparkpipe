#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${GLM52_MODEL_DIR:-/home/spark2/models/hf/nvidia/GLM-5.2-NVFP4}"
OUTPUT_DIR="${GLM52_LOCAL_PIPELINE_OUTPUT_DIR:-$ROOT/build/glm52_local_pipeline_gate}"
INPUT_TOKEN_ID="${GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID:-1037}"
MAX_STAGE_US="${GLM52_LOCAL_PIPELINE_MAX_STAGE_US:-1000000}"
NVCC_BIN="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH_VALUE="${CUDA_ARCH:-sm_121a}"
ENABLE_GRAPH_REPLAY="${GLM52_ENABLE_CUDA_GRAPH_REPLAY:-0}"
AOT_OUTPUT_DIR="${B12X_AOT_OUTPUT_DIR:-build/glm52_b12x_aot}"
PACK_ROOT="${GLM52_LOCAL_PIPELINE_PACK_ROOT:-$ROOT/build}"
RESUME="${GLM52_LOCAL_PIPELINE_RESUME:-0}"

case "$MODEL_DIR" in
	/mnt/mac/*|/Volumes/*)
		echo "refusing remote GLM52 model dir for local pipeline gate: $MODEL_DIR" >&2
		exit 2
		;;
esac

if [ ! -d "$MODEL_DIR" ]; then
	echo "missing GLM52 model dir: $MODEL_DIR" >&2
	exit 3
fi
if ! command -v "$NVCC_BIN" >/dev/null 2>&1; then
	echo "missing nvcc for GLM52 local pipeline gate: $NVCC_BIN" >&2
	exit 4
fi

mkdir -p "$OUTPUT_DIR"
SUMMARY_TSV="$OUTPUT_DIR/local_pipeline_summary.tsv"
printf "stage\tmode\tfirst_layer\tlayer_count\tpack_dir\ttotal_us\tmaximum_us\ttoken\tlog\n" >"$SUMMARY_TSV"

layers_csv()
{
	local first="$1"
	local count="$2"
	local layer
	local result=""
	for ((layer=first; layer<first+count; layer++)); do
		if [ -n "$result" ]; then
			result="$result,$layer"
		else
			result="$layer"
		fi
	done
	printf "%s" "$result"
}

pack_dir_has_stage_packs()
{
	local pack_dir="$1"
	local first="$2"
	local count="$3"
	local layer
	local pack_path
	if [ ! -s "$pack_dir/resident_moe_pack_manifest.json" ]; then
		return 1
	fi
	for ((layer=first; layer<first+count; layer++)); do
		pack_path="$(printf "%s/glm52_layer_%04u_b12x_moe.spb12x" "$pack_dir" "$layer")"
		if [ ! -s "$pack_path" ]; then
			return 1
		fi
	done
	return 0
}

pack_dir_for_stage()
{
	local first="$1"
	local count="$2"
	local last
	local label
	local candidate
	last=$((first + count - 1))
	label="$(printf "%04u_%04u" "$first" "$last")"
	for candidate in \
		"$PACK_ROOT/glm52_b12x_resident_moe_${label}_v3" \
		"$PACK_ROOT/glm52_b12x_resident_moe_0003_0010_v3" \
		"$PACK_ROOT/glm52_b12x_resident_moe_0027_0050_v3" \
		"$PACK_ROOT/glm52_b12x_resident_moe_0075_0077_v3" \
		"$PACK_ROOT/glm52_b12x_resident_moe_${label}_v2" \
		"$PACK_ROOT/glm52_b12x_resident_moe_${label}" \
		"$PACK_ROOT/glm52_b12x_resident_moe"
	do
		if pack_dir_has_stage_packs "$candidate" "$first" "$count"; then
			printf "%s" "$candidate"
			return 0
		fi
	done
	echo "missing B12x pack directory for routed stage ${first}:${count}" >&2
	exit 5
}

require_stage_packs()
{
	local pack_dir="$1"
	local first="$2"
	local count="$3"
	local layer
	local pack_path
	if [ ! -s "$pack_dir/resident_moe_pack_manifest.json" ]; then
		echo "missing B12x pack manifest: $pack_dir/resident_moe_pack_manifest.json" >&2
		exit 6
	fi
	for ((layer=first; layer<first+count; layer++)); do
		pack_path="$(printf "%s/glm52_layer_%04u_b12x_moe.spb12x" "$pack_dir" "$layer")"
		if [ ! -s "$pack_path" ]; then
			echo "missing B12x resident pack: $pack_path" >&2
			exit 7
		fi
	done
}

extract_field()
{
	local name="$1"
	local text="$2"
	printf "%s" "$text" | sed -n "s/.*${name}=\([^ ]*\).*/\1/p" | tail -1
}

stage_pass_line()
{
	local log_path="$1"
	local pass_line
	pass_line="$(grep -E "^glm52_resident_decode_stage validation passed" "$log_path" | head -1 || true)"
	if [ -z "$pass_line" ]; then
		pass_line="$(grep -E "validation passed|orchestrator validation passed" "$log_path" | tail -1 || true)"
	fi
	printf "%s" "$pass_line"
}

record_stage_summary()
{
	local mode="$1"
	local first="$2"
	local count="$3"
	local pack_dir="$4"
	local log_path="$5"
	local pass_line
	local total_us
	local maximum_us
	local token
	pass_line="$(stage_pass_line "$log_path")"
	if [ -z "$pass_line" ]; then
		return 1
	fi
	total_us="$(extract_field "total_us" "$pass_line")"
	maximum_us="$(extract_field "maximum_us" "$pass_line")"
	token="$(extract_field "restricted_token" "$pass_line")"
	printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
		"${first}:${count}" "$mode" "$first" "$count" "$pack_dir" \
		"${total_us:-}" "${maximum_us:-}" "${token:-}" "$log_path" >>"$SUMMARY_TSV"
	return 0
}

run_stage()
{
	local mode="$1"
	local first="$2"
	local count="$3"
	local input_hidden="$4"
	local output_hidden="$5"
	local pack_dir="$6"
	local layer_list
	local log_path
	local pass_line
	local total_us
	local maximum_us
	local token
	local command
	layer_list="$(layers_csv "$first" "$count")"
	log_path="$OUTPUT_DIR/stage_${first}_${count}_${mode}.log"
	require_stage_packs "$pack_dir" "$first" "$count"
	command=(
		make
		glm52_resident_decode_stage_firmware_package
		"MAX_STAGE_MICROSECONDS=$MAX_STAGE_US"
		"NVCC=$NVCC_BIN"
		"CUDA_ARCH=$CUDA_ARCH_VALUE"
		"GLM52_MODEL_DIR=$MODEL_DIR"
		"B12X_AOT_OUTPUT_DIR=$AOT_OUTPUT_DIR"
		"B12X_MOE_PACK_OUTPUT_DIR=$pack_dir"
		"B12X_MOE_PACK_LAYERS=$layer_list"
		"B12X_MOE_PACK_REQUIRE_REUSE=1"
		"GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX=$first"
		"GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT=$count"
		"GLM52_ENABLE_CUDA_GRAPH_REPLAY=$ENABLE_GRAPH_REPLAY"
		"GLM52_PIPELINE_OUTPUT_HIDDEN_BF16=$output_hidden"
	)
	if [ "$mode" = "dense_prefix" ]; then
		command+=(
			"GLM52_VALIDATION_MODE=dense_to_layer3_routed"
			"GLM52_VALIDATION_INPUT_TOKEN_ID=$INPUT_TOKEN_ID"
		)
	elif [ "$mode" = "hidden" ]; then
		command+=(
			"GLM52_VALIDATION_MODE=routed_from_hidden"
			"GLM52_PIPELINE_INPUT_HIDDEN_BF16=$input_hidden"
		)
	elif [ "$mode" = "final" ]; then
		command+=(
			"GLM52_VALIDATION_MODE=routed_from_hidden_final"
			"GLM52_PIPELINE_INPUT_HIDDEN_BF16=$input_hidden"
		)
	else
		echo "unknown local pipeline stage mode: $mode" >&2
		exit 8
	fi
	if ! (cd "$ROOT" && "${command[@]}") >"$log_path" 2>&1; then
		echo "glm52_local_pipeline_stage=failed mode=$mode first_layer=$first layer_count=$count log=$log_path" >&2
		tail -80 "$log_path" >&2 || true
		exit 9
	fi
	pass_line="$(stage_pass_line "$log_path")"
	if [ -z "$pass_line" ]; then
		echo "stage did not emit validation pass line: $log_path" >&2
		exit 10
	fi
	if [ ! -s "$output_hidden" ]; then
		echo "stage did not write output hidden: $output_hidden" >&2
		exit 11
	fi
	if ! record_stage_summary "$mode" "$first" "$count" "$pack_dir" "$log_path"; then
		echo "stage did not record summary: $log_path" >&2
		exit 12
	fi
	total_us="$(extract_field "total_us" "$pass_line")"
	maximum_us="$(extract_field "maximum_us" "$pass_line")"
	token="$(extract_field "restricted_token" "$pass_line")"
	echo "glm52_local_pipeline_stage=passed mode=$mode first_layer=$first layer_count=$count total_us=${total_us:-} maximum_us=${maximum_us:-} token=${token:-} output_hidden=$output_hidden"
}

run_pipeline_once()
{
	local run_label="$1"
	local first
	local count
	local mode
	local input_hidden=""
	local output_hidden
	local pack_dir
	local stages=(
		"3:8:dense_prefix"
		"11:8:hidden"
		"19:8:hidden"
		"27:8:hidden"
		"35:8:hidden"
		"43:8:hidden"
		"51:8:hidden"
		"59:8:hidden"
		"67:8:hidden"
		"75:3:final"
	)
	for spec in "${stages[@]}"; do
		IFS=: read -r first count mode <<<"$spec"
		output_hidden="$OUTPUT_DIR/${run_label}_after_layer_$((first + count - 1)).bf16"
		pack_dir="$(pack_dir_for_stage "$first" "$count")"
		if [ "$RESUME" != "0" ] && [ -s "$output_hidden" ]; then
			if ! record_stage_summary "$mode" "$first" "$count" "$pack_dir" "$OUTPUT_DIR/stage_${first}_${count}_${mode}.log"; then
				echo "resumed stage did not have a reusable validation log: $first:$count" >&2
				exit 13
			fi
			echo "glm52_local_pipeline_stage=resume mode=$mode first_layer=$first layer_count=$count output_hidden=$output_hidden"
		else
			run_stage "$mode" "$first" "$count" "$input_hidden" "$output_hidden" "$pack_dir"
		fi
		input_hidden="$output_hidden"
	done
}

echo "glm52_local_pipeline_model_dir=$MODEL_DIR"
echo "glm52_local_pipeline_output_dir=$OUTPUT_DIR"
echo "glm52_local_pipeline_input_token=$INPUT_TOKEN_ID"
echo "glm52_local_pipeline_resume=$RESUME"
run_pipeline_once "run1"
final_line="$(tail -1 "$SUMMARY_TSV")"
final_token="$(printf "%s" "$final_line" | awk -F '\t' '{print $8}')"
if [ -z "$final_token" ]; then
	echo "GLM52 local pipeline did not emit a final restricted token" >&2
	exit 14
fi
echo "glm52_local_pipeline_token=$final_token"
echo "glm52_local_pipeline_summary=$SUMMARY_TSV"
