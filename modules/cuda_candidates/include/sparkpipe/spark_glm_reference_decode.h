#ifndef SPARKPIPE_SPARK_GLM_REFERENCE_DECODE_H
#define SPARKPIPE_SPARK_GLM_REFERENCE_DECODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_GLM_REFERENCE_MAX_STAGES SPARKPIPE_MAX_STAGES
#define SPARKPIPE_GLM_REFERENCE_MAX_TOKENS 64u
#define SPARKPIPE_GLM_REFERENCE_MAX_PROMPT_TOKENS 256u
#define SPARKPIPE_GLM_REFERENCE_MAX_STAGE_PATH_BYTES 512u
#define SPARKPIPE_GLM_REFERENCE_DEFAULT_STAGE_COUNT 0u

typedef enum SparkGlmReferenceBackendKind
{
    SPARK_GLM_REFERENCE_BACKEND_MANIFEST_ONLY = 0,
    SPARK_GLM_REFERENCE_BACKEND_SYNTHETIC_DETERMINISTIC = 1,
    SPARK_GLM_REFERENCE_BACKEND_CUDA_STAGE = 2
} SparkGlmReferenceBackendKind;

typedef struct SparkGlmStageTensorManifest
{
    uint32_t stage_id;
    uint32_t stage_count;
    uint32_t first_layer_group;
    uint32_t layer_group_count;
    uint32_t tensor_view_count;
    uint64_t tensor_manifest_checksum;
    uint64_t stagepack_checksum;
    bool fast_ring_ready;
    bool tensor_manifest_ready;
    bool stagepack_ready;
    char tensor_manifest_path[SPARKPIPE_GLM_REFERENCE_MAX_STAGE_PATH_BYTES];
    char tensor_base_dir[SPARKPIPE_GLM_REFERENCE_MAX_STAGE_PATH_BYTES];
} SparkGlmStageTensorManifest;

typedef SparkStatus (*SparkGlmReferenceInitialActivationCallback)(const uint32_t *prompt_tokens, uint32_t prompt_token_count, uint32_t decode_token_index, uint16_t *activation_bf16, uint32_t activation_capacity, uint32_t *activation_count, uint64_t *activation_checksum, void *user_data);
typedef SparkStatus (*SparkGlmReferenceStageExecuteCallback)(const SparkGlmStageTensorManifest *manifest, uint32_t token_index, uint64_t input_activation_checksum, uint64_t *output_activation_checksum, void *user_data);
typedef SparkStatus (*SparkGlmReferenceStageExecuteActivationCallback)(const SparkGlmStageTensorManifest *manifest, uint32_t token_index, const uint16_t *input_activation_bf16, uint32_t input_activation_count, uint16_t *output_activation_bf16, uint32_t output_activation_capacity, uint32_t *output_activation_count, uint64_t *output_activation_checksum, void *user_data);
typedef SparkStatus (*SparkGlmReferenceGreedyLogitsCallback)(uint32_t token_index, uint64_t final_activation_checksum, uint32_t *observed_token, uint64_t *logits_checksum, void *user_data);
typedef SparkStatus (*SparkGlmReferenceGreedyLogitsActivationCallback)(uint32_t token_index, const uint16_t *final_activation_bf16, uint32_t final_activation_count, uint64_t final_activation_checksum, uint32_t *observed_token, uint64_t *logits_checksum, void *user_data);
typedef SparkStatus (*SparkGlmReferenceBeginDecodeSessionCallback)(const uint32_t *prompt_tokens, uint32_t prompt_token_count, uint32_t maximum_decode_tokens, void *user_data);
typedef SparkStatus (*SparkGlmReferenceCommitGreedyTokenCallback)(uint32_t token_index, uint32_t observed_token, void *user_data);

typedef struct SparkGlmReferenceDecodeBackend
{
    SparkGlmReferenceBackendKind backend_kind;
    bool production_backend;
    bool resident_adapter_attached;
    bool supports_single_token_no_cache;
    bool supports_prompt_prefill;
    bool supports_multi_token_decode;
    bool uses_stateful_kv_cache;
    SparkGlmReferenceBeginDecodeSessionCallback begin_decode_session;
    SparkGlmReferenceCommitGreedyTokenCallback commit_greedy_token;
    SparkGlmReferenceInitialActivationCallback initialize_activation;
    SparkGlmReferenceStageExecuteCallback execute_stage;
    SparkGlmReferenceStageExecuteActivationCallback execute_stage_activation;
    SparkGlmReferenceGreedyLogitsCallback greedy_logits;
    SparkGlmReferenceGreedyLogitsActivationCallback greedy_logits_activation;
    uint32_t activation_count;
    uint16_t *activation_buffer_a;
    uint16_t *activation_buffer_b;
    uint32_t activation_buffer_capacity;
    void *user_data;
} SparkGlmReferenceDecodeBackend;

typedef struct SparkGlmReferenceDecodeConfig
{
    uint32_t stage_count;
    uint32_t prompt_token_count;
    uint32_t expected_token_count;
    uint32_t maximum_decode_tokens;
    uint32_t prompt_tokens[SPARKPIPE_GLM_REFERENCE_MAX_PROMPT_TOKENS];
    uint32_t expected_tokens[SPARKPIPE_GLM_REFERENCE_MAX_TOKENS];
    uint32_t observed_tokens[SPARKPIPE_GLM_REFERENCE_MAX_TOKENS];
    uint64_t prompt_checksum;
    uint64_t expected_logits_checksum;
    SparkGlmReferenceBackendKind backend_kind;
    bool allow_synthetic_reference_pass;
    bool allow_trusted_observed_tokens;
} SparkGlmReferenceDecodeConfig;

typedef struct SparkGlmReferenceDecodeReport
{
    uint32_t stage_count;
    uint32_t manifest_count;
    uint32_t stage_manifest_ready_count;
    uint32_t fast_ring_ready_count;
    uint32_t stagepack_ready_count;
    uint32_t stage_execution_count;
    uint32_t logits_ready_count;
    uint32_t expected_token_count;
    uint32_t observed_token_count;
    uint32_t matched_token_count;
    uint32_t first_mismatch_index;
    uint32_t first_expected_token;
    uint32_t first_observed_token;
    uint32_t observed_tokens[SPARKPIPE_GLM_REFERENCE_MAX_TOKENS];
    uint64_t token_logits_checksums[SPARKPIPE_GLM_REFERENCE_MAX_TOKENS];
    uint64_t manifest_checksum;
    uint64_t execution_checksum;
    uint64_t logits_checksum;
    uint64_t expected_logits_checksum;
    bool stage_manifests_ready;
    bool backend_ready;
    bool logits_ready;
    bool greedy_decode_ready;
    bool request_requires_kv_cache;
    bool prompt_prefill_ready;
    bool kv_cache_ready;
    bool multi_token_decode_ready;
    bool single_token_no_cache_ready;
    bool rolling_prompt_used;
    uint32_t rolling_prompt_token_count;
    bool exact_token_match;
    bool logits_checksum_match;
    bool reference_artifact_match;
    bool glm_reference_decode_ready;
    bool production_backend_used;
    bool resident_adapter_attached;
    char first_blocker[192];
} SparkGlmReferenceDecodeReport;

const char *SparkGlmReferenceBackendKindToString(SparkGlmReferenceBackendKind backend_kind);
void SparkGlmReferenceDecodeConfigReset(SparkGlmReferenceDecodeConfig *config);
void SparkGlmReferenceDecodeReportReset(SparkGlmReferenceDecodeReport *report);
SparkStatus SparkReadGlmStageTensorManifestFromPath(const char *path, SparkGlmStageTensorManifest *manifest);
SparkStatus SparkWriteGlmStageTensorManifestSummary(const SparkGlmStageTensorManifest *manifest, char *output, size_t output_size);
void SparkGlmReferenceDecodeBackendReset(SparkGlmReferenceDecodeBackend *backend);
SparkStatus SparkDeriveGlmReferenceStageCount(const SparkGlmStageTensorManifest *manifests, uint32_t manifest_count, uint32_t *stage_count);
SparkStatus SparkRunGlmReferenceDecodeBridgeWithBackend(const SparkGlmReferenceDecodeConfig *config, const SparkGlmStageTensorManifest *manifests, uint32_t manifest_count, const SparkGlmReferenceDecodeBackend *backend, SparkGlmReferenceDecodeReport *report);
SparkStatus SparkRunGlmReferenceDecodeBridge(const SparkGlmReferenceDecodeConfig *config, const SparkGlmStageTensorManifest *manifests, uint32_t manifest_count, SparkGlmReferenceDecodeReport *report);
void SparkGlmReferenceDecodeWriteReport(const SparkGlmReferenceDecodeReport *report, char *output, size_t output_size);

#ifdef __cplusplus
}
#endif

#endif
