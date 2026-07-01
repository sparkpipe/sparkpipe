#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${GLM52_MODEL_DIR:-/home/spark2/models/hf/nvidia/GLM-5.2-NVFP4}"
OUTPUT_DIR="${GLM52_ACCURACY_OUTPUT_DIR:-$ROOT/build/glm52_accuracy_gate}"
INPUT_HIDDEN="${GLM52_ACCURACY_INPUT_HIDDEN_BF16:-/tmp/glm52_pipeline_sim/spark2local_layer0074.bf16}"
B12X_PACK_DIR="${GLM52_B12X_MOE_PACK_DIR:-$ROOT/build/glm52_b12x_resident_moe_0075_0077_v3}"
REPEAT_COUNT="${GLM52_ACCURACY_REPEAT_COUNT:-3}"
TRACE_BUFFERS="${GLM52_ACCURACY_TRACE_BUFFERS:-1}"
MAX_STAGE_US="${GLM52_ACCURACY_MAX_STAGE_US:-1000000}"
NVCC_BIN="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH_VALUE="${CUDA_ARCH:-sm_121a}"

MODULE="$ROOT/build/modules/glm52_resident_decode_stage/libglm52_resident_decode_stage.a"
DRIVER="$ROOT/build/packages/glm52_resident_decode_stage/stages/stage_000/model_driver.so"
VALIDATOR="$ROOT/modules/glm52_resident_decode_stage/validation/validate_glm52_resident_decode_stage_cuda.sh"
LINK_ARGS_FILE="$ROOT/build/glm52_b12x_aot/generated/runtime_link_args.txt"
B12X_ADAPTER="$ROOT/build/modules/glm52_sm121_flashinfer_b12x_moe/libglm52_sm121_flashinfer_b12x_moe_adapter.a"
B12X_BACKEND="$ROOT/build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_compiled_backend.a"
B12X_TABLE="$ROOT/build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_generated_kernel_table.a"

case "$MODEL_DIR" in
    /mnt/mac/*|/Volumes/*)
        echo "refusing remote GLM52 model dir for accuracy gate: $MODEL_DIR" >&2
        exit 2
        ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
    echo "missing GLM52 model dir: $MODEL_DIR" >&2
    exit 3
fi
if [ ! -f "$INPUT_HIDDEN" ]; then
    echo "missing final-stage input hidden: $INPUT_HIDDEN" >&2
    exit 4
fi
if [ ! -d "$B12X_PACK_DIR" ]; then
    echo "missing B12x pack dir: $B12X_PACK_DIR" >&2
    exit 5
fi
for required_file in "$MODULE" "$DRIVER" "$VALIDATOR" "$LINK_ARGS_FILE" "$B12X_ADAPTER" "$B12X_BACKEND" "$B12X_TABLE"; do
    if [ ! -f "$required_file" ]; then
        echo "missing required accuracy input: $required_file" >&2
        exit 6
    fi
done

mkdir -p "$OUTPUT_DIR"
LINK_ARGS="$B12X_ADAPTER $B12X_BACKEND $B12X_TABLE $(cat "$LINK_ARGS_FILE")"

echo "accuracy_gate_model_dir=$MODEL_DIR"
echo "accuracy_gate_input_hidden=$INPUT_HIDDEN"
echo "accuracy_gate_output_dir=$OUTPUT_DIR"

GLM52_REQUIRED_CUDA_LINK_ARGS="$LINK_ARGS" \
GLM52_B12X_MOE_PACK_DIR="$B12X_PACK_DIR" \
GLM52_MODEL_DIR="$MODEL_DIR" \
GLM52_INPUT_TOKEN_ID=1037 \
GLM52_LOAD_LAYER0_ATTENTION_BF16=1 \
GLM52_LOAD_LAYER0_DENSE_BF16=1 \
GLM52_PREFILL_KV_FROM_EMBEDDINGS=1 \
GLM52_CHECK_LAYER0_FULL_REFERENCE=1 \
GLM52_ENABLE_CUDA_GRAPH_REPLAY=0 \
GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1 \
NVCC="$NVCC_BIN" \
CUDA_ARCH="$CUDA_ARCH_VALUE" \
"$VALIDATOR" "$MAX_STAGE_US" "$MODULE" "$DRIVER" \
    >"$OUTPUT_DIR/dense_reference.log" 2>&1

grep -E "layer0_reference_full=1|real_lm_head=1" "$OUTPUT_DIR/dense_reference.log" >/dev/null
echo "accuracy_gate_dense_reference=passed"
grep -E "layer0_reference_full_max_error|real_lm_head_max_logit_error" "$OUTPUT_DIR/dense_reference.log" | tail -1

rm -f "$OUTPUT_DIR"/final_repeat_*.bf16 "$OUTPUT_DIR"/final_repeat_*.log "$OUTPUT_DIR"/final_repeat.sha256

for repeat_index in $(seq 1 "$REPEAT_COUNT"); do
    output_hidden="$OUTPUT_DIR/final_repeat_${repeat_index}.bf16"
    output_log="$OUTPUT_DIR/final_repeat_${repeat_index}.log"
    GLM52_REQUIRED_CUDA_LINK_ARGS="$LINK_ARGS" \
    GLM52_B12X_MOE_PACK_DIR="$B12X_PACK_DIR" \
    GLM52_MODEL_DIR="$MODEL_DIR" \
    GLM52_ROUTED_CHAIN_FIRST_LAYER_INDEX=75 \
    GLM52_ROUTED_CHAIN_LAYER_COUNT=3 \
    GLM52_ENABLE_CUDA_GRAPH_REPLAY=0 \
    GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1 \
    GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN=1 \
    GLM52_ACCURACY_TRACE_BUFFERS="$TRACE_BUFFERS" \
    GLM52_PIPELINE_INPUT_HIDDEN_BF16="$INPUT_HIDDEN" \
    GLM52_PIPELINE_OUTPUT_HIDDEN_BF16="$output_hidden" \
    NVCC="$NVCC_BIN" \
    CUDA_ARCH="$CUDA_ARCH_VALUE" \
    "$VALIDATOR" "$MAX_STAGE_US" "$MODULE" "$DRIVER" \
        >"$output_log" 2>&1
    grep -E "routed_pipeline_from_hidden_final=1|restricted_token=|real_lm_head_max_logit_error" "$output_log" | tail -1
    sha256sum "$output_hidden" >>"$OUTPUT_DIR/final_repeat.sha256"
done

unique_hash_count="$(awk '{print $1}' "$OUTPUT_DIR/final_repeat.sha256" | sort -u | wc -l | tr -d ' ')"
cat "$OUTPUT_DIR/final_repeat.sha256"
if [ "$unique_hash_count" != "1" ]; then
    if [ "$TRACE_BUFFERS" != "0" ] && [ "$REPEAT_COUNT" -ge 2 ]; then
        awk '
            /accuracy_trace_buffer/ {
                layer=""; name=""; hash="";
                for (i=1; i<=NF; i++) {
                    split($i, field, "=");
                    if (field[1] == "layer") layer=field[2];
                    if (field[1] == "name") name=field[2];
                    if (field[1] == "hash64") hash=field[2];
                }
                if (layer != "" && name != "" && hash != "")
                    print layer, name, hash;
            }
        ' "$OUTPUT_DIR/final_repeat_1.log" >"$OUTPUT_DIR/final_repeat_1.trace.tsv"
        awk '
            /accuracy_trace_buffer/ {
                layer=""; name=""; hash="";
                for (i=1; i<=NF; i++) {
                    split($i, field, "=");
                    if (field[1] == "layer") layer=field[2];
                    if (field[1] == "name") name=field[2];
                    if (field[1] == "hash64") hash=field[2];
                }
                if (layer != "" && name != "" && hash != "")
                    print layer, name, hash;
            }
        ' "$OUTPUT_DIR/final_repeat_2.log" >"$OUTPUT_DIR/final_repeat_2.trace.tsv"
        awk '
            NR == FNR {
                previous[$1 " " $2] = $3;
                next;
            }
            {
                key = $1 " " $2;
                if ((key in previous) && previous[key] != $3) {
                    printf("accuracy_gate_first_divergence=layer:%s buffer:%s repeat1:%s repeat2:%s\n", $1, $2, previous[key], $3) > "/dev/stderr";
                    exit;
                }
            }
        ' "$OUTPUT_DIR/final_repeat_1.trace.tsv" "$OUTPUT_DIR/final_repeat_2.trace.tsv"
    fi
    echo "accuracy_gate_routed_repeatability=failed unique_hash_count=$unique_hash_count" >&2
    echo "same-input routed hidden output is not bit-stable; do not call GLM52 accuracy PASS" >&2
    exit 7
fi

echo "accuracy_gate_routed_repeatability=passed"
