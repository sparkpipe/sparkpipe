#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ABI_VERSION 3u
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_MODULE_ID \
    "spark.glm52.sm121.flashinfer_b12x_fused_moe.nvfp4.bf16.v2"
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_REQUIRED_ARCH "sm_121a"
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION \
    SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_INTERMEDIATE_DIMENSION \
    SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_FUSED_W1_ROWS \
    (SPARK_GLM52_MODEL_MOE_W1_COMPONENT_COUNT * \
     SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION)
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT \
    SPARK_GLM52_MODEL_MOE_EXPERT_COUNT
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K \
    SPARK_GLM52_MODEL_MOE_TOP_K
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_NVFP4_GROUP_SIZE \
    SPARK_GLM52_MODEL_NVFP4_GROUP_SIZE

#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_GATE_UP_ORDER_UP_GATE 1u
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW 2u
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE 2u
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_QUANT_MODE_NVFP4 1u
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 1u

#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_ROUTER_LOGITS 0x00000001u
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_DETERMINISTIC_FC2_FINALIZE 0x00000002u
#define SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_KNOWN_FLAGS \
    (SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_ROUTER_LOGITS | \
     SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_DETERMINISTIC_FC2_FINALIZE)

#define SPARK_GLM52_SM121_FLASHINFER_B12X_REQUIRED_BACKEND_SYMBOL_CREATE \
    "SparkFlashInferB12xCompiledMoeCreate"
#define SPARK_GLM52_SM121_FLASHINFER_B12X_REQUIRED_BACKEND_SYMBOL_LAUNCH \
    "SparkFlashInferB12xCompiledMoeLaunch"
#define SPARK_GLM52_SM121_FLASHINFER_B12X_REQUIRED_BACKEND_SYMBOL_DESTROY \
    "SparkFlashInferB12xCompiledMoeDestroy"

typedef struct SparkGlm52Sm121FlashInferB12xMoeRecipe
{
    uint32_t abi_version;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t maximum_token_count;
    uint32_t gate_up_order;
    uint32_t weight_layout;
    uint32_t scale_layout;
    uint32_t quant_mode;
    uint32_t output_dtype;
    uint32_t cuda_architecture;
    uint64_t qualified_maximum_microseconds;
    uint64_t qualification_record_hash_low64;
    uint64_t kernel_manifest_hash_low64;
} SparkGlm52Sm121FlashInferB12xMoeRecipe;

typedef struct SparkGlm52Sm121FlashInferB12xMoeArguments
{
    uint32_t abi_version;
    uint32_t token_count;
    uint32_t maximum_token_count;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    uint32_t argument_flags;
    const void *hidden_bf16;
    int32_t *topk_ids_i32;
    float *topk_weights_fp32;
    const float *router_logits_f32;
    const float *router_score_bias_f32;
    uint32_t router_norm_topk_prob;
    float router_routed_scaling_factor;
    const void *w1_weight_fp4_static_view;
    const void *w1_scale_static_storage_ue4m3;
    const float *w1_alpha_fp32_by_expert;
    const float *fc2_input_scale_fp32_by_expert;
    const void *w2_weight_fp4_static_view;
    const void *w2_scale_static_storage_ue4m3;
    const float *w2_alpha_fp32_by_expert;
    void *output_bf16;
    void *workspace;
    uint64_t workspace_bytes;
    void *cuda_stream;
} SparkGlm52Sm121FlashInferB12xMoeArguments;

SparkStatus SparkGlm52Sm121FlashInferB12xMoeCreate(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe,
    void **state_out);

uint64_t SparkGlm52Sm121FlashInferB12xMoeActiveKernelManifestHashLow64(void);

SparkStatus SparkGlm52Sm121FlashInferB12xMoeLaunch(
    void *state,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments);

void SparkGlm52Sm121FlashInferB12xMoeDestroy(
    void *state);

#ifdef __cplusplus
}
#endif
