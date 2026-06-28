#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_glm_reference_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_GLM_STAGE_EXECUTOR_PATH_BYTES 4096u
#define SPARKPIPE_GLM_STAGE_EXECUTOR_TENSOR_NAME_BYTES 256u
#define SPARKPIPE_GLM_STAGE_EXECUTOR_DTYPE_BYTES 16u
#define SPARKPIPE_GLM_STAGE_EXECUTOR_BLOCKER_BYTES 160u
#define SPARKPIPE_GLM_STAGE_EXECUTOR_SENTINEL 0x5350474C4D535445ull
#define SPARKPIPE_GLM_STAGE_EXECUTOR_MAX_ATTENTION_PROJECTION_VALUES 32768u
#define SPARKPIPE_GLM_STAGE_EXECUTOR_MAX_MLP_INTERMEDIATE_VALUES 16384u
#define SPARKPIPE_GLM_STAGE_EXECUTOR_MAX_ROUTED_EXPERTS 512u
#define SPARKPIPE_GLM_STAGE_EXECUTOR_MAX_ROUTER_TOPK 32u
#define SPARKPIPE_GLM_STAGE_EXECUTOR_MAX_PREFIX_TOKENS 384u

typedef struct SparkGlmStageTensorBinding
{
    char path[SPARKPIPE_GLM_STAGE_EXECUTOR_PATH_BYTES];
    char tensor_name[SPARKPIPE_GLM_STAGE_EXECUTOR_TENSOR_NAME_BYTES];
    char dtype[SPARKPIPE_GLM_STAGE_EXECUTOR_DTYPE_BYTES];
    uint64_t offset_bytes;
    uint64_t size_bytes;
    bool ready;
} SparkGlmStageTensorBinding;

typedef struct SparkGlmStageExecutorConfig
{
    const char *tensor_base_dir;
    uint32_t hidden_size;
    bool check_tensor_files;
    float rms_norm_epsilon;
} SparkGlmStageExecutorConfig;

typedef struct SparkGlmStageExecutor
{
    uint32_t stage_id;
    uint32_t stage_count;
    uint32_t first_layer_group;
    uint32_t layer_group_count;
    uint32_t hidden_size;
    float rms_norm_epsilon;
    bool prepared;
    char tensor_manifest_path[SPARKPIPE_GLM_STAGE_EXECUTOR_PATH_BYTES];
    char tensor_base_dir[SPARKPIPE_GLM_STAGE_EXECUTOR_PATH_BYTES];
    SparkGlmStageTensorBinding first_input_layernorm_weight;
    SparkGlmStageTensorBinding first_q_a_proj_weight;
    SparkGlmStageTensorBinding first_q_a_layernorm_weight;
    SparkGlmStageTensorBinding first_q_b_proj_weight;
    SparkGlmStageTensorBinding first_kv_a_proj_weight;
    SparkGlmStageTensorBinding first_kv_a_layernorm_weight;
    SparkGlmStageTensorBinding first_kv_b_proj_weight;
    SparkGlmStageTensorBinding first_o_proj_weight;
    SparkGlmStageTensorBinding first_post_attention_layernorm_weight;
    SparkGlmStageTensorBinding first_dense_mlp_gate_proj_weight;
    SparkGlmStageTensorBinding first_dense_mlp_up_proj_weight;
    SparkGlmStageTensorBinding first_dense_mlp_down_proj_weight;
    SparkGlmStageTensorBinding first_moe_gate_weight;
    SparkGlmStageTensorBinding first_moe_score_correction_bias;
    uint32_t first_q_a_projection_rows;
    uint32_t first_q_b_projection_rows;
    uint32_t first_kv_a_projection_rows;
    uint32_t first_kv_latent_rows;
    uint32_t first_kv_rope_rows;
    uint32_t first_kv_b_projection_rows;
    uint32_t first_attention_value_rows;
    uint32_t first_attention_head_count;
    uint32_t first_attention_qk_nope_rows;
    uint32_t first_attention_v_head_rows;
    uint32_t first_dense_mlp_intermediate_rows;
    uint32_t first_moe_expert_count;
    uint64_t manifest_checksum;
    uint64_t tensor_binding_checksum;
} SparkGlmStageExecutor;

typedef enum SparkGlmStageFp4ExpertFormat
{
    SPARK_GLM_STAGE_FP4_EXPERT_FORMAT_NONE = 0,
    SPARK_GLM_STAGE_FP4_EXPERT_FORMAT_NVFP4_E2M1_E4M3 = 1,
    SPARK_GLM_STAGE_FP4_EXPERT_FORMAT_MXFP4_E2M1_E8M0 = 2
} SparkGlmStageFp4ExpertFormat;

typedef struct SparkGlmStageNvfp4ExpertDescriptor
{
    uint32_t expert_id;
    uint32_t hidden_size;
    uint32_t intermediate_rows;
    SparkGlmStageFp4ExpertFormat fp4_format;
    uint32_t fp4_group_size;
    bool ready;
    SparkGlmStageTensorBinding gate_weight;
    SparkGlmStageTensorBinding up_weight;
    SparkGlmStageTensorBinding down_weight;
    SparkGlmStageTensorBinding gate_weight_scale;
    SparkGlmStageTensorBinding up_weight_scale;
    SparkGlmStageTensorBinding down_weight_scale;
    SparkGlmStageTensorBinding gate_weight_scale_2;
    SparkGlmStageTensorBinding up_weight_scale_2;
    SparkGlmStageTensorBinding down_weight_scale_2;
    SparkGlmStageTensorBinding gate_input_scale;
    SparkGlmStageTensorBinding up_input_scale;
    SparkGlmStageTensorBinding down_input_scale;
    uint64_t descriptor_checksum;
} SparkGlmStageNvfp4ExpertDescriptor;

typedef struct SparkGlmStageForwardInput
{
    const uint16_t *activation_bf16;
    uint32_t activation_count;
} SparkGlmStageForwardInput;

typedef struct SparkGlmStageForwardOutput
{
    uint16_t *activation_bf16;
    uint32_t activation_capacity;
    uint32_t activation_count;
} SparkGlmStageForwardOutput;

typedef SparkStatus (*SparkGlmStageLayerKvReadFn)(const SparkGlmStageExecutor *layer_executor,
                                                  uint32_t layer_offset,
                                                  uint32_t absolute_token_position,
                                                  uint16_t *cached_kv_a_bf16,
                                                  uint32_t cached_kv_a_capacity,
                                                  uint32_t *cached_kv_latent_count,
                                                  uint32_t *cached_kv_rope_count,
                                                  void *user_data);
typedef SparkStatus (*SparkGlmStageLayerKvReadTokenFn)(const SparkGlmStageExecutor *layer_executor,
                                                       uint32_t layer_offset,
                                                       uint32_t current_absolute_token_position,
                                                       uint32_t cached_absolute_token_position,
                                                       uint16_t *cached_kv_a_bf16,
                                                       uint32_t cached_kv_a_capacity,
                                                       uint32_t *cached_kv_latent_count,
                                                       uint32_t *cached_kv_rope_count,
                                                       void *user_data);
typedef SparkStatus (*SparkGlmStageLayerKvWriteFn)(const SparkGlmStageExecutor *layer_executor,
                                                   uint32_t layer_offset,
                                                   uint32_t absolute_token_position,
                                                   const uint16_t *kv_a_bf16,
                                                   uint32_t kv_latent_count,
                                                   uint32_t kv_rope_count,
                                                   void *user_data);

typedef struct SparkGlmStageForwardKvCacheCallbacks
{
    bool require_kv_read;
    bool require_kv_write;
    uint32_t absolute_token_position;
    SparkGlmStageLayerKvReadFn read_layer;
    SparkGlmStageLayerKvReadTokenFn read_token_layer;
    SparkGlmStageLayerKvWriteFn write_layer;
    void *user_data;
} SparkGlmStageForwardKvCacheCallbacks;

typedef struct SparkGlmStageExecutorReport
{
    uint32_t stage_id;
    uint32_t stage_count;
    uint32_t first_layer_group;
    uint32_t hidden_size;
    uint32_t real_op_count;
    uint32_t real_layer_count;
    bool single_token_no_cache_body_available;
    bool full_transformer_body_available;
    uint64_t input_activation_checksum;
    uint64_t input_norm_weight_checksum;
    uint64_t input_q_a_proj_weight_checksum;
    uint64_t input_q_a_norm_weight_checksum;
    uint64_t input_q_b_proj_weight_checksum;
    uint64_t input_kv_a_proj_weight_checksum;
    uint64_t input_kv_a_norm_weight_checksum;
    uint64_t input_kv_b_proj_weight_checksum;
    uint64_t input_o_proj_weight_checksum;
    uint64_t input_post_attention_norm_weight_checksum;
    uint64_t input_dense_mlp_gate_proj_weight_checksum;
    uint64_t input_dense_mlp_up_proj_weight_checksum;
    uint64_t input_dense_mlp_down_proj_weight_checksum;
    uint64_t input_moe_gate_weight_checksum;
    uint64_t input_moe_score_correction_bias_checksum;
    uint64_t output_activation_checksum;
    uint64_t q_a_projection_checksum;
    uint64_t q_b_projection_checksum;
    uint64_t kv_a_projection_checksum;
    uint64_t kv_b_projection_checksum;
    uint64_t attention_value_checksum;
    uint64_t attention_output_checksum;
    uint64_t attention_residual_checksum;
    uint64_t post_attention_norm_checksum;
    uint64_t dense_mlp_intermediate_checksum;
    uint64_t dense_mlp_output_checksum;
    uint64_t first_layer_output_checksum;
    uint64_t moe_router_logits_checksum;
    uint64_t moe_router_topk_checksum;
    uint64_t input_moe_expert_gate_proj_weight_checksum;
    uint64_t input_moe_expert_up_proj_weight_checksum;
    uint64_t input_moe_expert_down_proj_weight_checksum;
    uint64_t moe_expert_intermediate_checksum;
    uint64_t moe_expert_output_checksum;
    uint64_t moe_nvfp4_descriptor_checksum;
    uint64_t moe_nvfp4_activation_packed_bytes;
    uint64_t moe_nvfp4_activation_scale_bytes;
    uint64_t moe_nvfp4_weight_packed_bytes;
    uint64_t moe_nvfp4_weight_scale_bytes;
    uint32_t q_a_projection_count;
    uint32_t q_b_projection_count;
    uint32_t kv_a_projection_count;
    uint32_t kv_latent_count;
    uint32_t kv_rope_count;
    uint32_t kv_b_projection_count;
    uint32_t attention_value_count;
    uint32_t attention_head_count;
    uint32_t attention_qk_nope_count;
    uint32_t attention_output_count;
    uint32_t dense_mlp_intermediate_count;
    uint32_t dense_mlp_output_count;
    uint32_t first_layer_output_count;
    uint32_t moe_expert_count;
    uint32_t moe_topk_count;
    uint32_t moe_bound_expert_count;
    uint32_t moe_first_bound_expert_id;
    uint32_t moe_quantized_expert_count;
    uint32_t moe_fp4_format;
    uint32_t moe_fp4_group_size;
    uint32_t moe_mxfp4_descriptor_count;
    uint32_t moe_nvfp4_activation_quant_descriptor_count;
    uint32_t moe_nvfp4_gemm_descriptor_count;
    uint32_t moe_expert_intermediate_count;
    uint32_t moe_expert_output_count;
    uint64_t trace_checksum;
    char first_blocker[SPARKPIPE_GLM_STAGE_EXECUTOR_BLOCKER_BYTES];
} SparkGlmStageExecutorReport;

void SparkGlmStageExecutorConfigReset(SparkGlmStageExecutorConfig *config);
void SparkGlmStageExecutorReset(SparkGlmStageExecutor *executor);
void SparkGlmStageExecutorReportReset(SparkGlmStageExecutorReport *report);
void SparkGlmStageNvfp4ExpertDescriptorReset(SparkGlmStageNvfp4ExpertDescriptor *descriptor);
SparkStatus SparkGlmCreateStageExecutor(SparkGlmStageExecutor *executor, const SparkGlmStageTensorManifest *manifest, const SparkGlmStageExecutorConfig *config);
SparkStatus SparkGlmDescribeStageFirstMoeExpertNvfp4(const SparkGlmStageExecutor *executor, uint32_t expert_id, SparkGlmStageNvfp4ExpertDescriptor *descriptor, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageFirstRmsNormHostReference(const SparkGlmStageExecutor *executor, const SparkGlmStageForwardInput *input, SparkGlmStageForwardOutput *output, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageFirstAttentionQaProjectionHostReference(const SparkGlmStageExecutor *executor, const SparkGlmStageForwardInput *normalized_input, uint16_t *q_a_output_bf16, uint32_t q_a_output_capacity, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageFirstAttentionQbProjectionHostReference(const SparkGlmStageExecutor *executor, const uint16_t *q_a_input_bf16, uint32_t q_a_input_count, uint16_t *q_b_output_bf16, uint32_t q_b_output_capacity, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageFirstAttentionKvaProjectionHostReference(const SparkGlmStageExecutor *executor, const SparkGlmStageForwardInput *normalized_input, uint16_t *kv_a_output_bf16, uint32_t kv_a_output_capacity, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageFirstAttentionKvbProjectionHostReference(const SparkGlmStageExecutor *executor, const uint16_t *kv_a_input_bf16, uint32_t kv_a_input_count, uint16_t *kv_b_output_bf16, uint32_t kv_b_output_capacity, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageFirstAttentionOutputProjectionHostReference(const SparkGlmStageExecutor *executor, const uint16_t *kv_b_input_bf16, uint32_t kv_b_input_count, uint16_t *attention_value_bf16, uint32_t attention_value_capacity, uint16_t *attention_output_bf16, uint32_t attention_output_capacity, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageFirstDenseMlpHostReference(const SparkGlmStageExecutor *executor, const uint16_t *residual_input_bf16, uint32_t residual_input_count, const uint16_t *attention_output_bf16, uint32_t attention_output_count, uint16_t *mlp_intermediate_bf16, uint32_t mlp_intermediate_capacity, uint16_t *layer_output_bf16, uint32_t layer_output_capacity, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageFirstMoeRouterHostReference(const SparkGlmStageExecutor *executor, const uint16_t *residual_input_bf16, uint32_t residual_input_count, const uint16_t *attention_output_bf16, uint32_t attention_output_count, uint32_t router_top_k, uint32_t *top_expert_ids, float *top_expert_weights, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageFirstMoeExpertsHostReference(const SparkGlmStageExecutor *executor, const uint16_t *residual_input_bf16, uint32_t residual_input_count, const uint16_t *attention_output_bf16, uint32_t attention_output_count, uint32_t router_top_k, const uint32_t *top_expert_ids, const float *top_expert_weights, uint16_t *moe_intermediate_bf16, uint32_t moe_intermediate_capacity, uint16_t *layer_output_bf16, uint32_t layer_output_capacity, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageForwardWithKvCache(const SparkGlmStageExecutor *executor, const SparkGlmStageForwardInput *input, SparkGlmStageForwardOutput *output, const SparkGlmStageForwardKvCacheCallbacks *kv_callbacks, SparkGlmStageExecutorReport *report);
SparkStatus SparkGlmExecuteStageForward(const SparkGlmStageExecutor *executor, const SparkGlmStageForwardInput *input, SparkGlmStageForwardOutput *output, SparkGlmStageExecutorReport *report);

#ifdef __cplusplus
}
#endif
