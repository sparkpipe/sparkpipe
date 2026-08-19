#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_glm52_dspark_draft_backend.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_sha256.h"

#define SPARK_DSPARK_BACKEND_THREADS 256u
#define SPARK_DSPARK_BACKEND_HEAD_THREADS \
    SPARK_DSPARK_DRAFT_HEAD_DIMENSION
#define SPARK_DSPARK_BACKEND_CONFIDENCE_DIMENSION \
    (SPARK_DSPARK_HIDDEN_DIMENSION + SPARK_DSPARK_MARKOV_RANK)
#define SPARK_DSPARK_MANIFEST_FORMAT \
    "sparkpipe.glm52.dspark.speculator_manifest.v2"
#define SPARK_DSPARK_MANIFEST_MODEL_ID \
    "RedHatAI/GLM-5.2-speculator.dspark"

typedef enum SparkGlm52DsparkWeightRole
{
    SPARK_DSPARK_WEIGHT_EMBED_TOKENS = 0,
    SPARK_DSPARK_WEIGHT_FUSION_FC,
    SPARK_DSPARK_WEIGHT_HIDDEN_NORM,
    SPARK_DSPARK_WEIGHT_FINAL_NORM,
    SPARK_DSPARK_WEIGHT_LM_HEAD,
    SPARK_DSPARK_WEIGHT_MARKOV_W1,
    SPARK_DSPARK_WEIGHT_MARKOV_W2,
    SPARK_DSPARK_WEIGHT_CONFIDENCE,
    SPARK_DSPARK_WEIGHT_CONFIDENCE_BIAS,
    SPARK_DSPARK_WEIGHT_LAYER_BASE
} SparkGlm52DsparkWeightRole;

typedef enum SparkGlm52DsparkLayerWeight
{
    SPARK_DSPARK_LAYER_WEIGHT_INPUT_NORM = 0,
    SPARK_DSPARK_LAYER_WEIGHT_Q,
    SPARK_DSPARK_LAYER_WEIGHT_K,
    SPARK_DSPARK_LAYER_WEIGHT_V,
    SPARK_DSPARK_LAYER_WEIGHT_Q_NORM,
    SPARK_DSPARK_LAYER_WEIGHT_K_NORM,
    SPARK_DSPARK_LAYER_WEIGHT_O,
    SPARK_DSPARK_LAYER_WEIGHT_POST_NORM,
    SPARK_DSPARK_LAYER_WEIGHT_GATE,
    SPARK_DSPARK_LAYER_WEIGHT_UP,
    SPARK_DSPARK_LAYER_WEIGHT_DOWN,
    SPARK_DSPARK_LAYER_WEIGHT_COUNT
} SparkGlm52DsparkLayerWeight;

typedef struct SparkGlm52DsparkTensorSpec
{
    const char *name;
    uint32_t role;
    uint32_t rows;
    uint32_t columns;
} SparkGlm52DsparkTensorSpec;

typedef struct SparkGlm52DsparkSafetensorsFile
{
    int file_descriptor;
    uint64_t file_bytes;
    uint64_t data_offset;
    const uint8_t *mapped_bytes;
    SparkJsonDocument header;
} SparkGlm52DsparkSafetensorsFile;

typedef struct SparkGlm52DsparkManifestU32
{
    const char *name;
    uint32_t value;
} SparkGlm52DsparkManifestU32;

static SparkStatus SparkGlm52DsparkManifestRequireString(
    const SparkJsonDocument *document,
    int32_t object_token,
    const char *name,
    const char *expected)
{
    int32_t token;

    token = SparkJsonFindObjectMember(document, object_token, name);
    if (token < 0 || !SparkJsonStringEquals(document, token, expected))
        return SPARK_STATUS_VALIDATION_FAILED;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkManifestRequireU32(
    const SparkJsonDocument *document,
    int32_t object_token,
    const SparkGlm52DsparkManifestU32 *expected)
{
    uint32_t value;
    int32_t token;

    token = SparkJsonFindObjectMember(document, object_token, expected->name);
    if (token < 0 ||
        SparkJsonGetUInt32(document, token, &value) != SPARK_STATUS_OK ||
        value != expected->value)
        return SPARK_STATUS_VALIDATION_FAILED;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkManifestValidateRevision(
    const SparkJsonDocument *document,
    int32_t root_token)
{
    char *revision;
    uint32_t index;
    int32_t token;
    SparkStatus status;

    revision = 0;
    token = SparkJsonFindObjectMember(document, root_token, "model_revision");
    status = token < 0 ? SPARK_STATUS_VALIDATION_FAILED :
        SparkJsonCopyString(document, token, &revision);
    if (status == SPARK_STATUS_OK && strlen(revision) != 40u)
        status = SPARK_STATUS_VALIDATION_FAILED;
    for (index = 0u; status == SPARK_STATUS_OK && index < 40u; ++index)
        if (!((revision[index] >= '0' && revision[index] <= '9') ||
              (revision[index] >= 'a' && revision[index] <= 'f')))
            status = SPARK_STATUS_VALIDATION_FAILED;
    free(revision);
    return status;
}

static SparkStatus SparkGlm52DsparkManifestValidateVerifier(
    const SparkJsonDocument *document,
    int32_t root_token)
{
    static const SparkGlm52DsparkManifestU32 expected[] =
    {
        {"hidden_dimension", SPARK_DSPARK_HIDDEN_DIMENSION},
        {"vocabulary_size", SPARK_DSPARK_FULL_VOCAB_SIZE}
    };
    bool quantization_independent;
    uint32_t index;
    int32_t object_token,token;
    SparkStatus status;

    object_token = SparkJsonFindObjectMember(
        document, root_token, "verifier_contract");
    if (object_token < 0 ||
        !SparkJsonTokenIsType(document, object_token, SPARK_JSON_TOKEN_OBJECT))
        return SPARK_STATUS_VALIDATION_FAILED;
    token = SparkJsonFindObjectMember(
        document, object_token, "quantization_independent");
    if (token < 0 ||
        SparkJsonGetBoolean(document, token, &quantization_independent) !=
            SPARK_STATUS_OK ||
        !quantization_independent)
        return SPARK_STATUS_VALIDATION_FAILED;
    status = SparkGlm52DsparkManifestRequireString(
        document, object_token, "hidden_dtype", "bf16");
    for (index = 0u;
         status == SPARK_STATUS_OK &&
             index < (uint32_t)(sizeof(expected) / sizeof(expected[0]));
         ++index)
        status = SparkGlm52DsparkManifestRequireU32(
            document, object_token, &expected[index]);
    return status;
}

static SparkStatus SparkGlm52DsparkManifestValidateContract(
    const SparkJsonDocument *document,
    int32_t root_token)
{
    static const SparkGlm52DsparkManifestU32 expected[] =
    {
        {"abi_version", SPARK_DSPARK_ABI_VERSION},
        {"verifier_hidden_dtype", SPARK_DSPARK_VERIFIER_HIDDEN_DTYPE_BF16},
        {"draft_dtype", SPARK_DSPARK_DRAFT_DTYPE_BF16},
        {"draft_layer_count", SPARK_DSPARK_DRAFT_LAYER_COUNT},
        {"block_size", SPARK_DSPARK_BLOCK_SIZE},
        {"hidden_dimension", SPARK_DSPARK_HIDDEN_DIMENSION},
        {"intermediate_dimension", SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION},
        {"attention_head_count", SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT},
        {"kv_head_count", SPARK_DSPARK_DRAFT_KV_HEAD_COUNT},
        {"head_dimension", SPARK_DSPARK_DRAFT_HEAD_DIMENSION},
        {"vocab_size", SPARK_DSPARK_FULL_VOCAB_SIZE},
        {"draft_vocab_size", SPARK_DSPARK_FULL_VOCAB_SIZE},
        {"markov_rank", SPARK_DSPARK_MARKOV_RANK},
        {"max_anchors", SPARK_DSPARK_MAX_ANCHORS},
        {"maximum_speculative_token_count",
            SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT},
        {"verifier_accept_k", 1u},
        {"enable_confidence_head", 1u},
        {"confidence_head_with_markov", 1u}
    };
    uint32_t index;
    int32_t object_token;
    SparkStatus status;

    object_token = SparkJsonFindObjectMember(document, root_token, "contract");
    if (object_token < 0 ||
        !SparkJsonTokenIsType(document, object_token, SPARK_JSON_TOKEN_OBJECT))
        return SPARK_STATUS_VALIDATION_FAILED;
    status = SPARK_STATUS_OK;
    for (index = 0u;
         status == SPARK_STATUS_OK &&
             index < (uint32_t)(sizeof(expected) / sizeof(expected[0]));
         ++index)
        status = SparkGlm52DsparkManifestRequireU32(
            document, object_token, &expected[index]);
    return status;
}

static SparkStatus SparkGlm52DsparkManifestValidateFile(
    const SparkJsonDocument *document,
    int32_t root_token,
    const char *record_name,
    const char *expected_name,
    const char *path)
{
    char actual_sha256[SPARK_SHA256_HEX_BYTES];
    char *manifest_sha256;
    struct stat file_stat;
    uint64_t manifest_bytes;
    int32_t object_token,token;
    SparkStatus status;

    manifest_sha256 = 0;
    object_token = SparkJsonFindObjectMember(document, root_token, record_name);
    if (object_token < 0 ||
        !SparkJsonTokenIsType(document, object_token, SPARK_JSON_TOKEN_OBJECT) ||
        SparkGlm52DsparkManifestRequireString(
            document, object_token, "path", expected_name) != SPARK_STATUS_OK ||
        stat(path, &file_stat) != 0 || file_stat.st_size < 0)
        return SPARK_STATUS_VALIDATION_FAILED;
    token = SparkJsonFindObjectMember(document, object_token, "bytes");
    if (token < 0 ||
        SparkJsonGetUInt64(document, token, &manifest_bytes) != SPARK_STATUS_OK ||
        manifest_bytes != (uint64_t)file_stat.st_size)
        return SPARK_STATUS_VALIDATION_FAILED;
    token = SparkJsonFindObjectMember(document, object_token, "sha256");
    status = token < 0 ? SPARK_STATUS_VALIDATION_FAILED :
        SparkJsonCopyString(document, token, &manifest_sha256);
    if (status == SPARK_STATUS_OK &&
        !SparkSha256HexIsValid(manifest_sha256))
        status = SPARK_STATUS_VALIDATION_FAILED;
    if (status == SPARK_STATUS_OK)
        status = SparkSha256File(path, actual_sha256);
    if (status == SPARK_STATUS_OK &&
        strcmp(actual_sha256, manifest_sha256) != 0)
        status = SPARK_STATUS_HASH_MISMATCH;
    free(manifest_sha256);
    return status;
}

static SparkStatus SparkGlm52DsparkValidateArtifactManifest(
    const SparkGlm52DsparkDraftBackendConfiguration *configuration)
{
    SparkJsonDocument document;
    int32_t root_token;
    SparkStatus status;

    memset(&document, 0, sizeof(document));
    status = SparkJsonLoadFile(configuration->manifest_path, &document);
    root_token = status == SPARK_STATUS_OK ?
        SparkJsonGetRootToken(&document) : -1;
    if (status == SPARK_STATUS_OK &&
        (root_token < 0 ||
         !SparkJsonTokenIsType(&document, root_token, SPARK_JSON_TOKEN_OBJECT)))
        status = SPARK_STATUS_VALIDATION_FAILED;
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkManifestRequireString(
            &document, root_token, "format", SPARK_DSPARK_MANIFEST_FORMAT);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkManifestRequireString(
            &document, root_token, "model_id",
            SPARK_DSPARK_MANIFEST_MODEL_ID);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkManifestValidateRevision(&document, root_token);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkManifestValidateVerifier(&document, root_token);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkManifestValidateContract(&document, root_token);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkManifestValidateFile(
            &document, root_token, "config_json", "config.json",
            configuration->config_path);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkManifestValidateFile(
            &document, root_token, "model_safetensors", "model.safetensors",
            configuration->safetensors_path);
    SparkJsonDocumentDestroy(&document);
    return status;
}

static uint32_t SparkGlm52DsparkLayerWeightIndex(
    uint32_t layer_index,
    uint32_t layer_weight)
{
    return SPARK_DSPARK_WEIGHT_LAYER_BASE +
        (layer_index * (uint32_t)SPARK_DSPARK_LAYER_WEIGHT_COUNT) +
        layer_weight;
}

static SparkStatus SparkGlm52DsparkCudaStatus(cudaError_t status)
{
    if (status == cudaSuccess)
        return SPARK_STATUS_OK;
    fprintf(stderr, "dspark cuda error: %s\n", cudaGetErrorString(status));
    return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkGlm52DsparkCublasStatus(cublasStatus_t status)
{
    if (status == CUBLAS_STATUS_SUCCESS)
        return SPARK_STATUS_OK;
    fprintf(stderr, "dspark cublas error: %d\n", (int)status);
    return SPARK_STATUS_INTERNAL_ERROR;
}

static __global__ void SparkGlm52DsparkRmsNormRowsKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint16_t *__restrict__ norm_weight_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t row_count,
    uint32_t dimension)
{
    __shared__ float partials[SPARK_DSPARK_BACKEND_THREADS];
    __shared__ float inverse_norm;
    uint32_t row_index,element_index,stride_index;
    uint64_t row_offset;
    float sum,value;

    row_index = blockIdx.x;
    if (row_index >= row_count)
        return;
    row_offset = (uint64_t)row_index * (uint64_t)dimension;
    sum = 0.0f;
    for (element_index=threadIdx.x; element_index<dimension; element_index+=blockDim.x)
    {
        value = __bfloat162float(
            ((const __nv_bfloat16 *)input_bf16)[row_offset + element_index]);
        sum += value * value;
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (stride_index=blockDim.x/2u; stride_index>0u; stride_index/=2u)
    {
        if (threadIdx.x < stride_index)
            partials[threadIdx.x] += partials[threadIdx.x + stride_index];
        __syncthreads();
    }
    if (threadIdx.x == 0u)
        inverse_norm = rsqrtf((partials[0] / (float)dimension) +
            SPARK_DSPARK_RMS_NORM_EPSILON);
    __syncthreads();
    for (element_index=threadIdx.x; element_index<dimension; element_index+=blockDim.x)
    {
        value = __bfloat162float(
            ((const __nv_bfloat16 *)input_bf16)[row_offset + element_index]);
        value = __bfloat162float(__float2bfloat16(value * inverse_norm));
        value = __bfloat162float(__float2bfloat16(value * __bfloat162float(
            ((const __nv_bfloat16 *)norm_weight_bf16)[element_index])));
        ((__nv_bfloat16 *)output_bf16)[row_offset + element_index] =
            __float2bfloat16(value);
    }
}

static __global__ void SparkGlm52DsparkAddBf16Kernel(
    uint16_t *__restrict__ destination_bf16,
    const uint16_t *__restrict__ addition_bf16,
    uint32_t element_count)
{
    uint32_t element_index;
    float value;

    element_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (element_index >= element_count)
        return;
    value = __bfloat162float(
        ((const __nv_bfloat16 *)destination_bf16)[element_index]) +
        __bfloat162float(
            ((const __nv_bfloat16 *)addition_bf16)[element_index]);
    ((__nv_bfloat16 *)destination_bf16)[element_index] = __float2bfloat16(value);
}

static __global__ void SparkGlm52DsparkSwigluRowsKernel(
    const uint16_t *__restrict__ gate_bf16,
    const uint16_t *__restrict__ up_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t element_count)
{
    uint32_t element_index;
    float gate,up;

    element_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (element_index >= element_count)
        return;
    gate = __bfloat162float(((const __nv_bfloat16 *)gate_bf16)[element_index]);
    up = __bfloat162float(((const __nv_bfloat16 *)up_bf16)[element_index]);
    gate = __bfloat162float(__float2bfloat16(
        gate / (1.0f + expf(-gate))));
    ((__nv_bfloat16 *)output_bf16)[element_index] = __float2bfloat16(gate * up);
}

static __global__ void SparkGlm52DsparkGatherStageTapsKernel(
    const uint16_t *__restrict__ tap_arena_bf16,
    const uint32_t *__restrict__ tap_row_indices,
    uint16_t *__restrict__ stage_tap_bf16,
    uint32_t stage_count)
{
    uint32_t element_index,stage_index,column_index;
    uint64_t element_count;

    element_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    element_count = (uint64_t)stage_count *
        SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION;
    if ((uint64_t)element_index >= element_count)
        return;
    stage_index =
        element_index / SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION;
    column_index =
        element_index % SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION;
    stage_tap_bf16[element_index] = tap_arena_bf16[
        ((uint64_t)tap_row_indices[stage_index] *
            SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION) +
        column_index];
}

static __global__ void SparkGlm52DsparkScatterContextBatchKernel(
    const uint16_t *__restrict__ key_bf16,
    const uint16_t *__restrict__ value_bf16,
    const uint32_t *__restrict__ backend_lane_indices,
    const uint32_t *__restrict__ context_positions,
    uint16_t *__restrict__ context_key_bf16,
    uint16_t *__restrict__ context_value_bf16,
    uint32_t stage_count,
    uint32_t layer_index,
    uint32_t maximum_context_token_count)
{
    uint32_t element_index,stage_index,column_index,backend_lane_index;
    uint64_t destination_offset,element_count;

    element_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    element_count = (uint64_t)stage_count *
        SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION;
    if ((uint64_t)element_index >= element_count)
        return;
    stage_index =
        element_index / SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION;
    column_index =
        element_index % SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION;
    backend_lane_index = backend_lane_indices[stage_index];
    destination_offset =
        ((((uint64_t)backend_lane_index *
            SPARK_DSPARK_DRAFT_LAYER_COUNT) + layer_index) *
            maximum_context_token_count + context_positions[stage_index]) *
        SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION + column_index;
    context_key_bf16[destination_offset] = key_bf16[element_index];
    context_value_bf16[destination_offset] = value_bf16[element_index];
}

static __global__ void SparkGlm52DsparkBuildQueryBlockBatchKernel(
    const uint16_t *__restrict__ embed_weight_bf16,
    const uint32_t *__restrict__ anchor_token_ids,
    uint16_t *__restrict__ block_hidden_bf16,
    uint32_t lane_count)
{
    uint32_t element_index,row_index,lane_index,column_index,token_id;
    uint64_t element_count;

    element_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    element_count = (uint64_t)lane_count * SPARK_DSPARK_BLOCK_SIZE *
        SPARK_DSPARK_HIDDEN_DIMENSION;
    if ((uint64_t)element_index >= element_count)
        return;
    row_index = element_index / SPARK_DSPARK_HIDDEN_DIMENSION;
    lane_index = row_index / SPARK_DSPARK_BLOCK_SIZE;
    column_index = element_index % SPARK_DSPARK_HIDDEN_DIMENSION;
    token_id = (row_index % SPARK_DSPARK_BLOCK_SIZE) == 0u
        ? anchor_token_ids[lane_index]
        : SPARK_DSPARK_MASK_TOKEN_ID;
    block_hidden_bf16[element_index] = embed_weight_bf16[
        ((uint64_t)token_id * SPARK_DSPARK_HIDDEN_DIMENSION) +
        column_index];
}

static __global__ void SparkGlm52DsparkHeadNormRopeBatchKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint16_t *__restrict__ head_norm_weight_bf16,
    const uint32_t *__restrict__ base_positions,
    uint16_t *__restrict__ output_bf16,
    uint32_t row_count,
    uint32_t rows_per_lane)
{
    __shared__ float values[SPARK_DSPARK_DRAFT_HEAD_DIMENSION];
    __shared__ float partials[SPARK_DSPARK_DRAFT_HEAD_DIMENSION];
    uint32_t row_index,lane_index,lane_row,head_index,element_index;
    uint32_t pair_index,paired_index,stride;
    uint64_t offset;
    float value,paired,inverse_norm,frequency,angle,cosine,sine;

    head_index = blockIdx.x;
    row_index = blockIdx.y;
    element_index = threadIdx.x;
    if (row_index >= row_count)
        return;
    lane_index = row_index / rows_per_lane;
    lane_row = row_index % rows_per_lane;
    offset = ((uint64_t)row_index *
        SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION) +
        ((uint64_t)head_index * SPARK_DSPARK_DRAFT_HEAD_DIMENSION);
    value = __bfloat162float(
        ((const __nv_bfloat16 *)input_bf16)[offset + element_index]);
    partials[element_index] = value * value;
    __syncthreads();
    for (stride=blockDim.x/2u; stride>0u; stride/=2u)
    {
        if (element_index < stride)
            partials[element_index] += partials[element_index + stride];
        __syncthreads();
    }
    inverse_norm = rsqrtf((partials[0] /
        (float)SPARK_DSPARK_DRAFT_HEAD_DIMENSION) +
        SPARK_DSPARK_RMS_NORM_EPSILON);
    value = __bfloat162float(__float2bfloat16(value * inverse_norm));
    values[element_index] = __bfloat162float(__float2bfloat16(
        value * __bfloat162float(
            ((const __nv_bfloat16 *)head_norm_weight_bf16)[element_index])));
    __syncthreads();
    pair_index = element_index %
        (SPARK_DSPARK_DRAFT_HEAD_DIMENSION / 2u);
    paired_index =
        element_index < (SPARK_DSPARK_DRAFT_HEAD_DIMENSION / 2u)
        ? element_index + (SPARK_DSPARK_DRAFT_HEAD_DIMENSION / 2u)
        : element_index - (SPARK_DSPARK_DRAFT_HEAD_DIMENSION / 2u);
    paired = values[paired_index];
    frequency = powf(SPARK_DSPARK_ROPE_THETA,
        -2.0f * (float)pair_index /
        (float)SPARK_DSPARK_DRAFT_HEAD_DIMENSION);
    angle = (float)(base_positions[lane_index] + lane_row) * frequency;
    cosine = __bfloat162float(__float2bfloat16(cosf(angle)));
    sine = __bfloat162float(__float2bfloat16(sinf(angle)));
    value = element_index <
        (SPARK_DSPARK_DRAFT_HEAD_DIMENSION / 2u)
        ? __bfloat162float(__float2bfloat16(values[element_index] * cosine)) -
            __bfloat162float(__float2bfloat16(paired * sine))
        : __bfloat162float(__float2bfloat16(values[element_index] * cosine)) +
            __bfloat162float(__float2bfloat16(paired * sine));
    ((__nv_bfloat16 *)output_bf16)[offset + element_index] =
        __float2bfloat16(value);
}

static __global__ void SparkGlm52DsparkBlockAttentionBatchKernel(
    const uint16_t *__restrict__ query_bf16,
    const uint16_t *__restrict__ context_key_bf16,
    const uint16_t *__restrict__ context_value_bf16,
    const uint16_t *__restrict__ block_key_bf16,
    const uint16_t *__restrict__ block_value_bf16,
    const uint32_t *__restrict__ backend_lane_indices,
    const uint32_t *__restrict__ context_token_counts,
    uint16_t *__restrict__ output_bf16,
    uint32_t row_count,
    uint32_t layer_index,
    uint32_t maximum_context_token_count)
{
    __shared__ float partials[SPARK_DSPARK_DRAFT_HEAD_DIMENSION];
    __shared__ float score,max_score,inverse_sum;
    uint32_t row_index,lane_index,head_index,element_index,key_index,stride;
    uint32_t context_token_count,total_key_count,backend_lane_index;
    uint64_t query_offset,key_offset,context_base;
    const uint16_t *key_base,*value_base;
    float query_value,key_value,weight,sum,output_value;

    head_index = blockIdx.x;
    row_index = blockIdx.y;
    element_index = threadIdx.x;
    if (row_index >= row_count)
        return;
    lane_index = row_index / SPARK_DSPARK_BLOCK_SIZE;
    backend_lane_index = backend_lane_indices[lane_index];
    context_token_count = context_token_counts[lane_index];
    total_key_count = context_token_count + SPARK_DSPARK_BLOCK_SIZE;
    context_base =
        (((uint64_t)backend_lane_index *
            SPARK_DSPARK_DRAFT_LAYER_COUNT) + layer_index) *
        maximum_context_token_count *
        SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION;
    query_offset = ((uint64_t)row_index *
        SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION) +
        ((uint64_t)head_index * SPARK_DSPARK_DRAFT_HEAD_DIMENSION);
    query_value = __bfloat162float(
        ((const __nv_bfloat16 *)query_bf16)[query_offset + element_index]);
    if (element_index == 0u)
        max_score = -FLT_MAX;
    __syncthreads();
    for (key_index=0u; key_index<total_key_count; ++key_index)
    {
        key_base = key_index < context_token_count
            ? context_key_bf16 + context_base : block_key_bf16;
        key_offset = key_index < context_token_count
            ? ((uint64_t)key_index *
                SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION)
            : (((uint64_t)lane_index * SPARK_DSPARK_BLOCK_SIZE +
                key_index - context_token_count) *
                SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION);
        key_offset +=
            (uint64_t)head_index * SPARK_DSPARK_DRAFT_HEAD_DIMENSION;
        key_value = __bfloat162float(
            ((const __nv_bfloat16 *)key_base)[key_offset + element_index]);
        partials[element_index] = query_value * key_value;
        __syncthreads();
        for (stride=blockDim.x/2u; stride>0u; stride/=2u)
        {
            if (element_index < stride)
                partials[element_index] += partials[element_index + stride];
            __syncthreads();
        }
        if (element_index == 0u)
        {
            score = __bfloat162float(
                __float2bfloat16(partials[0] * 0.125f));
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
    }
    sum = 0.0f;
    for (key_index=0u; key_index<total_key_count; ++key_index)
    {
        key_base = key_index < context_token_count
            ? context_key_bf16 + context_base : block_key_bf16;
        key_offset = key_index < context_token_count
            ? ((uint64_t)key_index *
                SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION)
            : (((uint64_t)lane_index * SPARK_DSPARK_BLOCK_SIZE +
                key_index - context_token_count) *
                SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION);
        key_offset +=
            (uint64_t)head_index * SPARK_DSPARK_DRAFT_HEAD_DIMENSION;
        key_value = __bfloat162float(
            ((const __nv_bfloat16 *)key_base)[key_offset + element_index]);
        partials[element_index] = query_value * key_value;
        __syncthreads();
        for (stride=blockDim.x/2u; stride>0u; stride/=2u)
        {
            if (element_index < stride)
                partials[element_index] += partials[element_index + stride];
            __syncthreads();
        }
        if (element_index == 0u)
        {
            score = __bfloat162float(
                __float2bfloat16(partials[0] * 0.125f));
            sum += expf(score - max_score);
        }
        __syncthreads();
    }
    if (element_index == 0u)
        inverse_sum = 1.0f / sum;
    __syncthreads();
    output_value = 0.0f;
    for (key_index=0u; key_index<total_key_count; ++key_index)
    {
        key_base = key_index < context_token_count
            ? context_key_bf16 + context_base : block_key_bf16;
        value_base = key_index < context_token_count
            ? context_value_bf16 + context_base : block_value_bf16;
        key_offset = key_index < context_token_count
            ? ((uint64_t)key_index *
                SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION)
            : (((uint64_t)lane_index * SPARK_DSPARK_BLOCK_SIZE +
                key_index - context_token_count) *
                SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION);
        key_offset +=
            (uint64_t)head_index * SPARK_DSPARK_DRAFT_HEAD_DIMENSION;
        key_value = __bfloat162float(
            ((const __nv_bfloat16 *)key_base)[key_offset + element_index]);
        partials[element_index] = query_value * key_value;
        __syncthreads();
        for (stride=blockDim.x/2u; stride>0u; stride/=2u)
        {
            if (element_index < stride)
                partials[element_index] += partials[element_index + stride];
            __syncthreads();
        }
        if (element_index == 0u)
            score = __bfloat162float(
                __float2bfloat16(partials[0] * 0.125f));
        __syncthreads();
        weight = __bfloat162float(__float2bfloat16(
            expf(score - max_score) * inverse_sum));
        output_value += weight * __bfloat162float(
            ((const __nv_bfloat16 *)value_base)[key_offset + element_index]);
        __syncthreads();
    }
    ((__nv_bfloat16 *)output_bf16)[query_offset + element_index] =
        __float2bfloat16(output_value);
}

static __global__ void SparkGlm52DsparkGatherMarkovBatchKernel(
    const uint16_t *__restrict__ markov_w1_bf16,
    const uint32_t *__restrict__ last_token_ids,
    const uint32_t *__restrict__ generated_token_ids,
    uint16_t *__restrict__ embedding_bf16,
    uint32_t lane_count,
    uint32_t proposal_index)
{
    uint32_t lane_index,element_index,token_id;

    lane_index = blockIdx.x;
    element_index = threadIdx.x;
    if (lane_index >= lane_count ||
        element_index >= SPARK_DSPARK_MARKOV_RANK)
        return;
    token_id = proposal_index == 0u
        ? last_token_ids[lane_index]
        : generated_token_ids[
            ((uint64_t)lane_index *
                SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT) +
            proposal_index - 1u];
    embedding_bf16[
        ((uint64_t)lane_index * SPARK_DSPARK_MARKOV_RANK) +
        element_index] = markov_w1_bf16[
            ((uint64_t)token_id * SPARK_DSPARK_MARKOV_RANK) +
            element_index];
}

static __global__ void SparkGlm52DsparkArgmaxBatchKernel(
    const uint16_t *__restrict__ block_logits_bf16,
    const uint16_t *__restrict__ markov_logits_bf16,
    const uint32_t *__restrict__ restricted_token_ids,
    uint32_t restricted_token_count,
    uint32_t proposal_index,
    uint32_t lane_count,
    uint32_t *token_ids_out)
{
    __shared__ float values[SPARK_DSPARK_BACKEND_THREADS];
    __shared__ uint32_t token_ids[SPARK_DSPARK_BACKEND_THREADS];
    uint32_t lane_index,candidate_count,candidate_index,token_id,stride;
    uint64_t logits_offset,markov_offset,output_offset;
    float value,best_value;
    uint32_t best_token;

    lane_index = blockIdx.x;
    if (lane_index >= lane_count)
        return;
    logits_offset =
        (((uint64_t)lane_index * SPARK_DSPARK_BLOCK_SIZE) +
            proposal_index + 1u) *
        SPARK_DSPARK_FULL_VOCAB_SIZE;
    markov_offset =
        (uint64_t)lane_index * SPARK_DSPARK_FULL_VOCAB_SIZE;
    candidate_count = restricted_token_count == 0u
        ? SPARK_DSPARK_FULL_VOCAB_SIZE : restricted_token_count;
    best_value = -FLT_MAX;
    best_token = UINT32_MAX;
    for (candidate_index=threadIdx.x; candidate_index<candidate_count;
         candidate_index+=blockDim.x)
    {
        token_id = restricted_token_count == 0u
            ? candidate_index : restricted_token_ids[candidate_index];
        value = __bfloat162float(
            ((const __nv_bfloat16 *)block_logits_bf16)[
                logits_offset + token_id]) +
            __bfloat162float(
                ((const __nv_bfloat16 *)markov_logits_bf16)[
                    markov_offset + token_id]);
        value = __bfloat162float(__float2bfloat16(value));
        if (value > best_value || (value == best_value && token_id < best_token))
        {
            best_value = value;
            best_token = token_id;
        }
    }
    values[threadIdx.x] = best_value;
    token_ids[threadIdx.x] = best_token;
    __syncthreads();
    for (stride=blockDim.x/2u; stride>0u; stride/=2u)
    {
        if (threadIdx.x < stride &&
            (values[threadIdx.x + stride] > values[threadIdx.x] ||
             (values[threadIdx.x + stride] == values[threadIdx.x] &&
              token_ids[threadIdx.x + stride] < token_ids[threadIdx.x])))
        {
            values[threadIdx.x] = values[threadIdx.x + stride];
            token_ids[threadIdx.x] = token_ids[threadIdx.x + stride];
        }
        __syncthreads();
    }
    output_offset =
        ((uint64_t)lane_index *
            SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT) + proposal_index;
    if (threadIdx.x == 0u)
        token_ids_out[output_offset] = token_ids[0u];
}

static __global__ void SparkGlm52DsparkConfidenceBatchKernel(
    const uint16_t *__restrict__ block_final_bf16,
    const uint16_t *__restrict__ markov_embedding_bf16,
    const uint16_t *__restrict__ weight_bf16,
    const uint16_t *__restrict__ bias_bf16,
    uint32_t proposal_index,
    uint32_t lane_count,
    float *confidence_out)
{
    __shared__ float partials[SPARK_DSPARK_BACKEND_THREADS];
    uint32_t lane_index,element_index,stride;
    uint64_t hidden_offset,markov_offset,output_offset;
    float sum,value;

    lane_index = blockIdx.x;
    if (lane_index >= lane_count)
        return;
    hidden_offset =
        (((uint64_t)lane_index * SPARK_DSPARK_BLOCK_SIZE) +
            proposal_index + 1u) *
        SPARK_DSPARK_HIDDEN_DIMENSION;
    markov_offset =
        (uint64_t)lane_index * SPARK_DSPARK_MARKOV_RANK;
    sum = 0.0f;
    for (element_index=threadIdx.x;
         element_index<SPARK_DSPARK_BACKEND_CONFIDENCE_DIMENSION;
         element_index+=blockDim.x)
    {
        value = element_index < SPARK_DSPARK_HIDDEN_DIMENSION
            ? __bfloat162float(
                ((const __nv_bfloat16 *)block_final_bf16)[
                    hidden_offset + element_index])
            : __bfloat162float(
                ((const __nv_bfloat16 *)markov_embedding_bf16)[
                    markov_offset + element_index -
                        SPARK_DSPARK_HIDDEN_DIMENSION]);
        sum += value * __bfloat162float(
            ((const __nv_bfloat16 *)weight_bf16)[element_index]);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (stride=blockDim.x/2u; stride>0u; stride/=2u)
    {
        if (threadIdx.x < stride)
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    output_offset =
        ((uint64_t)lane_index *
            SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT) + proposal_index;
    if (threadIdx.x == 0u)
    {
        value = partials[0] + __bfloat162float(
            ((const __nv_bfloat16 *)bias_bf16)[0]);
        confidence_out[output_offset] = 1.0f / (1.0f + expf(-value));
    }
}

static SparkStatus SparkGlm52DsparkSafetensorsOpen(
    const char *path,
    SparkGlm52DsparkSafetensorsFile *file)
{
    struct stat file_stat;
    uint64_t header_bytes;
    SparkStatus status;

    memset(file, 0, sizeof(*file));
    file->file_descriptor = -1;
    file->file_descriptor = open(path, O_RDONLY);
    if (file->file_descriptor < 0)
        return SPARK_STATUS_NOT_FOUND;
    if (fstat(file->file_descriptor, &file_stat) != 0 ||
        (uint64_t)file_stat.st_size < sizeof(uint64_t))
    {
        close(file->file_descriptor);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    file->file_bytes = (uint64_t)file_stat.st_size;
    file->mapped_bytes = (const uint8_t *)mmap(
        0, file->file_bytes, PROT_READ, MAP_PRIVATE, file->file_descriptor, 0);
    if (file->mapped_bytes == MAP_FAILED)
    {
        close(file->file_descriptor);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    memcpy(&header_bytes, file->mapped_bytes, sizeof(header_bytes));
    if (header_bytes == 0u ||
        (sizeof(uint64_t) + header_bytes) > file->file_bytes)
    {
        munmap((void *)file->mapped_bytes, file->file_bytes);
        close(file->file_descriptor);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    file->data_offset = sizeof(uint64_t) + header_bytes;
    status = SparkJsonParseText(
        (const char *)(file->mapped_bytes + sizeof(uint64_t)),
        (size_t)header_bytes,
        &file->header);
    if (status != SPARK_STATUS_OK)
    {
        munmap((void *)file->mapped_bytes, file->file_bytes);
        close(file->file_descriptor);
    }
    return status;
}

static void SparkGlm52DsparkSafetensorsClose(
    SparkGlm52DsparkSafetensorsFile *file)
{
    SparkJsonDocumentDestroy(&file->header);
    if (file->mapped_bytes != 0 && file->mapped_bytes != MAP_FAILED)
        munmap((void *)file->mapped_bytes, file->file_bytes);
    if (file->file_descriptor >= 0)
        close(file->file_descriptor);
    memset(file, 0, sizeof(*file));
    file->file_descriptor = -1;
}

static void SparkGlm52DsparkSafetensorsPrintInventory(
    const SparkGlm52DsparkSafetensorsFile *file)
{
    int32_t root_token;
    uint32_t token_index;
    const SparkJsonToken *token;

    root_token = SparkJsonGetRootToken(&file->header);
    for (token_index=0u; token_index<file->header.token_count; ++token_index)
    {
        token = &file->header.tokens[token_index];
        if (token->parent == root_token &&
            token->type == SPARK_JSON_TOKEN_STRING)
        {
            fprintf(stderr, "dspark safetensors tensor: %.*s\n",
                (int)(token->end - token->start),
                file->header.text + token->start);
        }
    }
}

static SparkStatus SparkGlm52DsparkSafetensorsFindTensor(
    const SparkGlm52DsparkSafetensorsFile *file,
    const SparkGlm52DsparkTensorSpec *spec,
    uint64_t *byte_offset_out,
    uint64_t *byte_count_out)
{
    char prefixed_name[192];
    int32_t root_token,tensor_token,dtype_token,shape_token,offsets_token;
    uint64_t begin_offset,end_offset,expected_bytes;
    uint32_t shape_count,shape_rows,shape_columns;

    root_token = SparkJsonGetRootToken(&file->header);
    tensor_token = SparkJsonFindObjectMember(&file->header, root_token, spec->name);
    if (tensor_token < 0)
    {
        snprintf(prefixed_name, sizeof(prefixed_name), "model.%s", spec->name);
        tensor_token = SparkJsonFindObjectMember(
            &file->header, root_token, prefixed_name);
    }
    if (tensor_token < 0)
        return SPARK_STATUS_NOT_FOUND;
    dtype_token = SparkJsonFindObjectMember(&file->header, tensor_token, "dtype");
    shape_token = SparkJsonFindObjectMember(&file->header, tensor_token, "shape");
    offsets_token = SparkJsonFindObjectMember(
        &file->header, tensor_token, "data_offsets");
    if (dtype_token < 0 || shape_token < 0 || offsets_token < 0 ||
        !SparkJsonStringEquals(&file->header, dtype_token, "BF16"))
        return SPARK_STATUS_INVALID_ARGUMENT;
    shape_count = SparkJsonGetArrayElementCount(&file->header, shape_token);
    shape_rows = 0u;
    shape_columns = 1u;
    if (shape_count == 0u || shape_count > 2u ||
        SparkJsonGetUInt32(&file->header,
            SparkJsonGetArrayElement(&file->header, shape_token, 0u),
            &shape_rows) != SPARK_STATUS_OK ||
        (shape_count == 2u && SparkJsonGetUInt32(&file->header,
            SparkJsonGetArrayElement(&file->header, shape_token, 1u),
            &shape_columns) != SPARK_STATUS_OK) ||
        shape_rows != spec->rows || shape_columns != spec->columns)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (SparkJsonGetUInt64(&file->header,
            SparkJsonGetArrayElement(&file->header, offsets_token, 0u),
            &begin_offset) != SPARK_STATUS_OK ||
        SparkJsonGetUInt64(&file->header,
            SparkJsonGetArrayElement(&file->header, offsets_token, 1u),
            &end_offset) != SPARK_STATUS_OK || end_offset <= begin_offset)
        return SPARK_STATUS_INVALID_ARGUMENT;
    expected_bytes = (uint64_t)spec->rows * (uint64_t)spec->columns *
        sizeof(uint16_t);
    if ((end_offset - begin_offset) != expected_bytes ||
        (file->data_offset + end_offset) > file->file_bytes)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *byte_offset_out = file->data_offset + begin_offset;
    *byte_count_out = expected_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkUploadTensor(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkSafetensorsFile *file,
    const SparkGlm52DsparkTensorSpec *spec)
{
    uint64_t byte_offset,byte_count;
    SparkStatus status;

    status = SparkGlm52DsparkSafetensorsFindTensor(
        file, spec, &byte_offset, &byte_count);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,
            "dspark tensor mismatch: name=%s rows=%u columns=%u status=%d\n",
            spec->name, spec->rows, spec->columns, (int)status);
        SparkGlm52DsparkSafetensorsPrintInventory(file);
        return status;
    }
    status = SparkGlm52DsparkCudaStatus(cudaMalloc(
        &backend->device_weights[spec->role], (size_t)byte_count));
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52DsparkCudaStatus(cudaMemcpy(
        backend->device_weights[spec->role],
        file->mapped_bytes + byte_offset,
        (size_t)byte_count,
        cudaMemcpyHostToDevice));
    return status;
}

static SparkStatus SparkGlm52DsparkLoadFixedTensors(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkSafetensorsFile *file)
{
    static const SparkGlm52DsparkTensorSpec specs[] =
    {
        {"embed_tokens.weight", SPARK_DSPARK_WEIGHT_EMBED_TOKENS,
            SPARK_DSPARK_FULL_VOCAB_SIZE,
            SPARK_DSPARK_HIDDEN_DIMENSION},
        {"fc.weight", SPARK_DSPARK_WEIGHT_FUSION_FC,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION},
        {"hidden_norm.weight", SPARK_DSPARK_WEIGHT_HIDDEN_NORM,
            SPARK_DSPARK_HIDDEN_DIMENSION, 1u},
        {"norm.weight", SPARK_DSPARK_WEIGHT_FINAL_NORM,
            SPARK_DSPARK_HIDDEN_DIMENSION, 1u},
        {"lm_head.weight", SPARK_DSPARK_WEIGHT_LM_HEAD,
            SPARK_DSPARK_FULL_VOCAB_SIZE,
            SPARK_DSPARK_HIDDEN_DIMENSION},
        {"markov_head.markov_w1.weight", SPARK_DSPARK_WEIGHT_MARKOV_W1,
            SPARK_DSPARK_FULL_VOCAB_SIZE,
            SPARK_DSPARK_MARKOV_RANK},
        {"markov_head.markov_w2.weight", SPARK_DSPARK_WEIGHT_MARKOV_W2,
            SPARK_DSPARK_FULL_VOCAB_SIZE,
            SPARK_DSPARK_MARKOV_RANK},
        {"confidence_head.proj.weight", SPARK_DSPARK_WEIGHT_CONFIDENCE,
            1u, SPARK_DSPARK_BACKEND_CONFIDENCE_DIMENSION},
        {"confidence_head.proj.bias", SPARK_DSPARK_WEIGHT_CONFIDENCE_BIAS,
            1u, 1u}
    };
    uint32_t spec_index;
    SparkStatus status;

    for (spec_index=0u;
         spec_index<(uint32_t)(sizeof(specs)/sizeof(specs[0]));
         ++spec_index)
    {
        status = SparkGlm52DsparkUploadTensor(backend, file, &specs[spec_index]);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkLoadLayerTensor(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkSafetensorsFile *file,
    uint32_t layer_index,
    const char *suffix,
    uint32_t layer_weight,
    uint32_t rows,
    uint32_t columns)
{
    char tensor_name[192];
    SparkGlm52DsparkTensorSpec spec;

    snprintf(tensor_name, sizeof(tensor_name), "layers.%u.%s", layer_index, suffix);
    spec.name = tensor_name;
    spec.role = SparkGlm52DsparkLayerWeightIndex(layer_index, layer_weight);
    spec.rows = rows;
    spec.columns = columns;
    return SparkGlm52DsparkUploadTensor(backend, file, &spec);
}

static SparkStatus SparkGlm52DsparkLoadLayerTensors(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkSafetensorsFile *file)
{
    static const struct
    {
        const char *suffix;
        uint32_t layer_weight;
        uint32_t rows;
        uint32_t columns;
    } specs[] =
    {
        {"input_layernorm.weight", SPARK_DSPARK_LAYER_WEIGHT_INPUT_NORM,
            SPARK_DSPARK_HIDDEN_DIMENSION, 1u},
        {"self_attn.q_proj.weight", SPARK_DSPARK_LAYER_WEIGHT_Q,
            SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            SPARK_DSPARK_HIDDEN_DIMENSION},
        {"self_attn.k_proj.weight", SPARK_DSPARK_LAYER_WEIGHT_K,
            SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            SPARK_DSPARK_HIDDEN_DIMENSION},
        {"self_attn.v_proj.weight", SPARK_DSPARK_LAYER_WEIGHT_V,
            SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            SPARK_DSPARK_HIDDEN_DIMENSION},
        {"self_attn.q_norm.weight", SPARK_DSPARK_LAYER_WEIGHT_Q_NORM,
            SPARK_DSPARK_DRAFT_HEAD_DIMENSION, 1u},
        {"self_attn.k_norm.weight", SPARK_DSPARK_LAYER_WEIGHT_K_NORM,
            SPARK_DSPARK_DRAFT_HEAD_DIMENSION, 1u},
        {"self_attn.o_proj.weight", SPARK_DSPARK_LAYER_WEIGHT_O,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION},
        {"post_attention_layernorm.weight", SPARK_DSPARK_LAYER_WEIGHT_POST_NORM,
            SPARK_DSPARK_HIDDEN_DIMENSION, 1u},
        {"mlp.gate_proj.weight", SPARK_DSPARK_LAYER_WEIGHT_GATE,
            SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION,
            SPARK_DSPARK_HIDDEN_DIMENSION},
        {"mlp.up_proj.weight", SPARK_DSPARK_LAYER_WEIGHT_UP,
            SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION,
            SPARK_DSPARK_HIDDEN_DIMENSION},
        {"mlp.down_proj.weight", SPARK_DSPARK_LAYER_WEIGHT_DOWN,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION}
    };
    uint32_t layer_index,spec_index;
    SparkStatus status;

    for (layer_index=0u; layer_index<SPARK_DSPARK_DRAFT_LAYER_COUNT;
         ++layer_index)
    {
        for (spec_index=0u;
             spec_index<(uint32_t)(sizeof(specs)/sizeof(specs[0]));
             ++spec_index)
        {
            status = SparkGlm52DsparkLoadLayerTensor(
                backend, file, layer_index, specs[spec_index].suffix,
                specs[spec_index].layer_weight, specs[spec_index].rows,
                specs[spec_index].columns);
            if (status != SPARK_STATUS_OK)
                return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkAllocate(void **pointer_out,uint64_t bytes)
{
    if (pointer_out == 0 || bytes == 0u || bytes > (uint64_t)SIZE_MAX)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SparkGlm52DsparkCudaStatus(cudaMalloc(pointer_out, (size_t)bytes));
}

static SparkStatus SparkGlm52DsparkAllocateWorkspaces(
    SparkGlm52DsparkDraftBackend *backend)
{
    uint64_t lane_count,execution_row_count,context_elements,tap_elements;
    uint64_t hidden_block_elements,attention_block_elements,mlp_block_elements;
    SparkStatus status;

    lane_count = backend->maximum_lane_count;
    execution_row_count = lane_count * SPARK_DSPARK_BLOCK_SIZE;
    tap_elements = execution_row_count *
        SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION;
    context_elements = lane_count * SPARK_DSPARK_DRAFT_LAYER_COUNT *
        backend->maximum_context_token_count *
        SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION;
    hidden_block_elements = execution_row_count *
        SPARK_DSPARK_HIDDEN_DIMENSION;
    attention_block_elements = execution_row_count *
        SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION;
    mlp_block_elements = execution_row_count *
        SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION;
    status = SparkGlm52DsparkAllocate(
        (void **)&backend->device_tap_arena_bf16,
        tap_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_stage_tap_bf16,
            tap_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_target_hidden_bf16,
            hidden_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_context_key_bf16,
            context_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_context_value_bf16,
            context_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_hidden_bf16,
            hidden_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_normed_bf16,
            hidden_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_attention_bf16,
            attention_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_query_bf16,
            attention_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_key_bf16,
            attention_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_value_bf16,
            attention_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_gate_bf16,
            mlp_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_up_bf16,
            mlp_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_mlp_bf16,
            mlp_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_final_bf16,
            hidden_block_elements * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_block_logits_bf16,
            execution_row_count *
                SPARK_DSPARK_FULL_VOCAB_SIZE * sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_markov_logits_bf16,
            lane_count * SPARK_DSPARK_FULL_VOCAB_SIZE *
                sizeof(uint16_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_argmax_u32,
            lane_count * SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT *
                sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_confidence_f32,
            lane_count * SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT *
                sizeof(float));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_tap_row_indices,
            execution_row_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_backend_lane_indices,
            execution_row_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_sequence_positions,
            execution_row_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_context_token_counts,
            lane_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocate(
            (void **)&backend->device_last_token_ids,
            lane_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMallocHost(
            (void **)&backend->host_argmax_u32,
            lane_count * SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT *
                sizeof(uint32_t)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMallocHost(
            (void **)&backend->host_confidence_f32,
            lane_count * SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT *
                sizeof(float)));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaEventCreateWithFlags(
            (cudaEvent_t *)&backend->completion_event,
            cudaEventDisableTiming));
    return status;
}

static SparkStatus SparkGlm52DsparkUploadRestrictedTokenIds(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftBackendConfiguration *configuration)
{
    uint32_t token_index;
    SparkStatus status;

    if (configuration->restricted_vocabulary_count == 0u)
        return SPARK_STATUS_OK;
    if (configuration->restricted_token_ids == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (token_index=0u;
         token_index<configuration->restricted_vocabulary_count;
         ++token_index)
    {
        if (configuration->restricted_token_ids[token_index] >=
            SPARK_DSPARK_FULL_VOCAB_SIZE)
            return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52DsparkAllocate(
        (void **)&backend->device_restricted_token_ids,
        (uint64_t)configuration->restricted_vocabulary_count * sizeof(uint32_t));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMemcpy(
            backend->device_restricted_token_ids,
            configuration->restricted_token_ids,
            (size_t)configuration->restricted_vocabulary_count * sizeof(uint32_t),
            cudaMemcpyHostToDevice));
    return status;
}

static SparkStatus SparkGlm52DsparkLaunchGemm(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t weight_index,
    const uint16_t *input_bf16,
    uint16_t *output_bf16,
    uint32_t row_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t accumulate)
{
    float alpha,beta;
    cublasStatus_t status;

    alpha = 1.0f;
    beta = accumulate == 0u ? 0.0f : 1.0f;
    status = cublasGemmEx(
        (cublasHandle_t)backend->cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        (int)output_dimension,
        (int)row_count,
        (int)input_dimension,
        &alpha,
        backend->device_weights[weight_index],
        CUDA_R_16BF,
        (int)input_dimension,
        input_bf16,
        CUDA_R_16BF,
        (int)input_dimension,
        &beta,
        output_bf16,
        CUDA_R_16BF,
        (int)output_dimension,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return SparkGlm52DsparkCublasStatus(status);
}

static void SparkGlm52DsparkFillModelContract(
    SparkGlm52DsparkModelContract *contract)
{
    static const uint32_t aux_layer_ids[SPARK_DSPARK_AUX_LAYER_COUNT] =
        SPARK_DSPARK_AUX_LAYER_IDS_INITIALIZER;
    uint32_t layer_index;

    memset(contract, 0, sizeof(*contract));
    contract->abi_version = SPARK_DSPARK_ABI_VERSION;
    contract->descriptor_bytes = SPARK_DSPARK_MODEL_CONTRACT_DESCRIPTOR_BYTES;
    contract->verifier_hidden_dtype =
        SPARK_DSPARK_VERIFIER_HIDDEN_DTYPE_BF16;
    contract->draft_dtype = SPARK_DSPARK_DRAFT_DTYPE_BF16;
    contract->draft_layer_count = SPARK_DSPARK_DRAFT_LAYER_COUNT;
    contract->block_size = SPARK_DSPARK_BLOCK_SIZE;
    contract->hidden_dimension = SPARK_DSPARK_HIDDEN_DIMENSION;
    contract->intermediate_dimension =
        SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION;
    contract->attention_head_count =
        SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT;
    contract->kv_head_count = SPARK_DSPARK_DRAFT_KV_HEAD_COUNT;
    contract->head_dimension = SPARK_DSPARK_DRAFT_HEAD_DIMENSION;
    contract->vocab_size = SPARK_DSPARK_FULL_VOCAB_SIZE;
    contract->draft_vocab_size = SPARK_DSPARK_FULL_VOCAB_SIZE;
    contract->markov_rank = SPARK_DSPARK_MARKOV_RANK;
    contract->max_anchors = SPARK_DSPARK_MAX_ANCHORS;
    contract->maximum_speculative_token_count =
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
    contract->verifier_accept_k = 1u;
    contract->aux_layer_count = SPARK_DSPARK_AUX_LAYER_COUNT;
    contract->enable_confidence_head = 1u;
    contract->confidence_head_with_markov = 1u;
    for (layer_index=0u; layer_index<SPARK_DSPARK_AUX_LAYER_COUNT;
         ++layer_index)
        contract->aux_layer_ids[layer_index] = aux_layer_ids[layer_index];
}

SparkStatus SparkGlm52DsparkDraftBackendInitialize(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftBackendConfiguration *configuration)
{
    SparkGlm52DsparkSafetensorsFile safetensors;
    SparkStatus status;

    if (backend == 0 || configuration == 0 ||
        configuration->abi_version != SPARK_DSPARK_DRAFT_BACKEND_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_DSPARK_DRAFT_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->maximum_lane_count == 0u ||
        configuration->maximum_lane_count >
            SPARK_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT ||
        configuration->maximum_context_token_count == 0u ||
        configuration->maximum_context_token_count >
            SPARK_DSPARK_MAXIMUM_CONTEXT_TOKENS ||
        configuration->restricted_vocabulary_count >
            SPARK_DSPARK_FULL_VOCAB_SIZE ||
        configuration->manifest_path == 0 ||
        configuration->config_path == 0 ||
        configuration->safetensors_path == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(&safetensors, 0, sizeof(safetensors));
    safetensors.file_descriptor = -1;
    memset(backend, 0, sizeof(*backend));
    backend->abi_version = SPARK_DSPARK_DRAFT_BACKEND_ABI_VERSION;
    backend->descriptor_bytes = SPARK_DSPARK_DRAFT_BACKEND_DESCRIPTOR_BYTES;
    backend->maximum_lane_count = configuration->maximum_lane_count;
    backend->maximum_context_token_count =
        configuration->maximum_context_token_count;
    backend->restricted_vocabulary_count =
        configuration->restricted_vocabulary_count;
    backend->weight_count = SPARK_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT;
    backend->maximum_tap_row_count =
        configuration->maximum_lane_count * SPARK_DSPARK_BLOCK_SIZE;
    backend->tap_arena_lane_stride_bytes =
        (uint64_t)SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION *
        sizeof(uint16_t);
    SparkGlm52DsparkFillModelContract(&backend->contract);
    status = SparkGlm52DsparkValidateArtifactManifest(configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52DsparkDraftBackendTeardown(backend);
        return status;
    }
    if (configuration->cuda_stream != 0)
        backend->cuda_stream = configuration->cuda_stream;
    else
    {
        status = SparkGlm52DsparkCudaStatus(cudaStreamCreate(
            (cudaStream_t *)&backend->cuda_stream));
        if (status != SPARK_STATUS_OK)
            return status;
        backend->owns_cuda_stream = 1u;
    }
    status = SparkGlm52DsparkCublasStatus(cublasCreate(
        (cublasHandle_t *)&backend->cublas_handle));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCublasStatus(cublasSetStream(
            (cublasHandle_t)backend->cublas_handle,
            (cudaStream_t)backend->cuda_stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkSafetensorsOpen(
            configuration->safetensors_path, &safetensors);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLoadFixedTensors(backend, &safetensors);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLoadLayerTensors(backend, &safetensors);
    if (safetensors.mapped_bytes != 0)
        SparkGlm52DsparkSafetensorsClose(&safetensors);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkUploadRestrictedTokenIds(backend, configuration);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkAllocateWorkspaces(backend);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52DsparkDraftBackendTeardown(backend);
        return status;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52DsparkFreeDeviceWorkspaces(
    SparkGlm52DsparkDraftBackend *backend)
{
    void *workspaces[] =
    {
        backend->device_restricted_token_ids,
        backend->device_tap_arena_bf16,
        backend->device_stage_tap_bf16,
        backend->device_target_hidden_bf16,
        backend->device_context_key_bf16,
        backend->device_context_value_bf16,
        backend->device_block_hidden_bf16,
        backend->device_block_normed_bf16,
        backend->device_block_attention_bf16,
        backend->device_block_query_bf16,
        backend->device_block_key_bf16,
        backend->device_block_value_bf16,
        backend->device_block_gate_bf16,
        backend->device_block_up_bf16,
        backend->device_block_mlp_bf16,
        backend->device_block_final_bf16,
        backend->device_block_logits_bf16,
        backend->device_markov_logits_bf16,
        backend->device_argmax_u32,
        backend->device_confidence_f32,
        backend->device_tap_row_indices,
        backend->device_backend_lane_indices,
        backend->device_sequence_positions,
        backend->device_context_token_counts,
        backend->device_last_token_ids
    };
    uint32_t workspace_index;

    for (workspace_index=0u;
         workspace_index<(uint32_t)(sizeof(workspaces)/sizeof(workspaces[0]));
         ++workspace_index)
    {
        if (workspaces[workspace_index] != 0)
            cudaFree(workspaces[workspace_index]);
    }
}

SparkStatus SparkGlm52DsparkDraftBackendTeardown(
    SparkGlm52DsparkDraftBackend *backend)
{
    uint32_t weight_index;

    if (backend == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (backend->cuda_stream != 0)
        cudaStreamSynchronize((cudaStream_t)backend->cuda_stream);
    if (backend->cublas_handle != 0)
        cublasDestroy((cublasHandle_t)backend->cublas_handle);
    for (weight_index=0u;
         weight_index<SPARK_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT;
         ++weight_index)
    {
        if (backend->device_weights[weight_index] != 0)
            cudaFree(backend->device_weights[weight_index]);
    }
    SparkGlm52DsparkFreeDeviceWorkspaces(backend);
    if (backend->host_argmax_u32 != 0)
        cudaFreeHost(backend->host_argmax_u32);
    if (backend->host_confidence_f32 != 0)
        cudaFreeHost(backend->host_confidence_f32);
    if (backend->completion_event != 0)
        cudaEventDestroy((cudaEvent_t)backend->completion_event);
    if (backend->owns_cuda_stream != 0u && backend->cuda_stream != 0)
        cudaStreamDestroy((cudaStream_t)backend->cuda_stream);
    memset(backend, 0, sizeof(*backend));
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkDraftBackendModelContract(
    const SparkGlm52DsparkDraftBackend *backend,
    SparkGlm52DsparkModelContract *contract_out)
{
    if (backend == 0 || contract_out == 0 ||
        backend->abi_version != SPARK_DSPARK_DRAFT_BACKEND_ABI_VERSION)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *contract_out = backend->contract;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkDraftBackendTapOutputPointers(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_index,
    void *tap_output_bf16[SPARK_DSPARK_AUX_LAYER_COUNT],
    uint64_t *lane_stride_bytes_out)
{
    uint8_t *lane_base;
    uint32_t tap_index;

    if (backend == 0 || tap_output_bf16 == 0 || lane_stride_bytes_out == 0 ||
        backend->abi_version != SPARK_DSPARK_DRAFT_BACKEND_ABI_VERSION ||
        lane_index >= backend->maximum_lane_count)
        return SPARK_STATUS_INVALID_ARGUMENT;
    lane_base = (uint8_t *)backend->device_tap_arena_bf16 +
        ((uint64_t)lane_index * backend->tap_arena_lane_stride_bytes);
    for (tap_index=0u; tap_index<SPARK_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        tap_output_bf16[tap_index] = lane_base +
            ((uint64_t)tap_index * SPARK_DSPARK_HIDDEN_DIMENSION *
                sizeof(uint16_t));
    }
    *lane_stride_bytes_out = backend->tap_arena_lane_stride_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkValidateStageBatch(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftBackendStage *stages,
    uint32_t stage_count)
{
    SparkGlm52DsparkDraftBackendLaneState *lane_state;
    const SparkGlm52DsparkDraftBackendStage *stage;
    uint32_t stage_index;

    memcpy(
        backend->validation_lane_states,
        backend->lane_states,
        (size_t)backend->maximum_lane_count * sizeof(backend->lane_states[0u]));
    for (stage_index=0u; stage_index<stage_count; ++stage_index)
    {
        stage = &stages[stage_index];
        if (stage->tap_row_index >= backend->maximum_tap_row_count ||
            stage->backend_lane_index >= backend->maximum_lane_count ||
            stage->sequence_position == 0u ||
            stage->sequence_position > backend->maximum_context_token_count ||
            stage->sequence_position > UINT32_MAX ||
            stage->token_id >= SPARK_DSPARK_FULL_VOCAB_SIZE ||
            stage->reserved != 0u)
            return SPARK_STATUS_INVALID_ARGUMENT;
        lane_state =
            &backend->validation_lane_states[stage->backend_lane_index];
        if ((lane_state->staged == 0u ||
             lane_state->sequence_id != stage->sequence_id) &&
            stage->sequence_position != 1u)
            return SPARK_STATUS_INVALID_ARGUMENT;
        if (lane_state->staged != 0u &&
            lane_state->sequence_id == stage->sequence_id &&
            stage->sequence_position !=
                (uint64_t)lane_state->context_token_count + 1u)
            return SPARK_STATUS_INVALID_ARGUMENT;
        lane_state->sequence_id = stage->sequence_id;
        lane_state->sequence_position = stage->sequence_position;
        lane_state->tap_generation = stage->tap_generation;
        lane_state->last_token_id = stage->token_id;
        lane_state->context_token_count = (uint32_t)stage->sequence_position;
        lane_state->staged = 1u;
        backend->host_tap_row_indices[stage_index] = stage->tap_row_index;
        backend->host_backend_lane_indices[stage_index] =
            stage->backend_lane_index;
        backend->host_sequence_positions[stage_index] =
            (uint32_t)stage->sequence_position - 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkUploadStageBatch(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t stage_count)
{
    uint64_t element_count;
    uint32_t block_count;
    SparkStatus status;

    status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
        backend->device_tap_row_indices,
        backend->host_tap_row_indices,
        (size_t)stage_count * sizeof(uint32_t),
        cudaMemcpyHostToDevice,
        (cudaStream_t)backend->cuda_stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
            backend->device_backend_lane_indices,
            backend->host_backend_lane_indices,
            (size_t)stage_count * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            (cudaStream_t)backend->cuda_stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
            backend->device_sequence_positions,
            backend->host_sequence_positions,
            (size_t)stage_count * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            (cudaStream_t)backend->cuda_stream));
    if (status != SPARK_STATUS_OK)
        return status;
    element_count = (uint64_t)stage_count *
        SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION;
    block_count = (uint32_t)((element_count +
        SPARK_DSPARK_BACKEND_THREADS - 1u) /
        SPARK_DSPARK_BACKEND_THREADS);
    SparkGlm52DsparkGatherStageTapsKernel<<<block_count,
        SPARK_DSPARK_BACKEND_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_tap_arena_bf16,
        backend->device_tap_row_indices,
        backend->device_stage_tap_bf16,
        stage_count);
    return SparkGlm52DsparkCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52DsparkAppendContextBatchLayer(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t stage_count,
    uint32_t layer_index)
{
    uint64_t element_count;
    uint32_t block_count;
    dim3 rope_grid;
    SparkStatus status;

    status = SparkGlm52DsparkLaunchGemm(
        backend,
        SparkGlm52DsparkLayerWeightIndex(
            layer_index,SPARK_DSPARK_LAYER_WEIGHT_K),
        backend->device_target_hidden_bf16,
        backend->device_block_key_bf16,
        stage_count,
        SPARK_DSPARK_HIDDEN_DIMENSION,
        SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
        0u);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_V),
            backend->device_target_hidden_bf16,
            backend->device_block_value_bf16,
            stage_count,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            0u);
    if (status != SPARK_STATUS_OK)
        return status;
    rope_grid = dim3(
        SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT,stage_count,1u);
    SparkGlm52DsparkHeadNormRopeBatchKernel<<<rope_grid,
        SPARK_DSPARK_BACKEND_HEAD_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_key_bf16,
        (const uint16_t *)backend->device_weights[
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_K_NORM)],
        backend->device_sequence_positions,
        backend->device_block_key_bf16,
        stage_count,
        1u);
    status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    if (status != SPARK_STATUS_OK)
        return status;
    element_count = (uint64_t)stage_count *
        SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION;
    block_count = (uint32_t)((element_count +
        SPARK_DSPARK_BACKEND_THREADS - 1u) /
        SPARK_DSPARK_BACKEND_THREADS);
    SparkGlm52DsparkScatterContextBatchKernel<<<block_count,
        SPARK_DSPARK_BACKEND_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_key_bf16,
        backend->device_block_value_bf16,
        backend->device_backend_lane_indices,
        backend->device_sequence_positions,
        backend->device_context_key_bf16,
        backend->device_context_value_bf16,
        stage_count,
        layer_index,
        backend->maximum_context_token_count);
    return SparkGlm52DsparkCudaStatus(cudaGetLastError());
}

SparkStatus SparkGlm52DsparkDraftBackendStageBatch(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftBackendStage *stages,
    uint32_t stage_count)
{
    uint32_t layer_index;
    SparkStatus status;

    if (backend == 0 || stages == 0 || stage_count == 0u ||
        backend->abi_version != SPARK_DSPARK_DRAFT_BACKEND_ABI_VERSION ||
        stage_count > backend->maximum_tap_row_count ||
        backend->pending_operation_kind !=
            SPARK_DSPARK_DRAFT_BACKEND_PENDING_NONE)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkGlm52DsparkValidateStageBatch(backend,stages,stage_count);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkUploadStageBatch(backend,stage_count);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SPARK_DSPARK_WEIGHT_FUSION_FC,
            backend->device_stage_tap_bf16,
            backend->device_block_normed_bf16,
            stage_count,
            SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            0u);
    if (status == SPARK_STATUS_OK)
    {
        SparkGlm52DsparkRmsNormRowsKernel<<<stage_count,
            SPARK_DSPARK_BACKEND_THREADS,0u,
            (cudaStream_t)backend->cuda_stream>>>(
            backend->device_block_normed_bf16,
            (const uint16_t *)backend->device_weights[
                SPARK_DSPARK_WEIGHT_HIDDEN_NORM],
            backend->device_target_hidden_bf16,
            stage_count,
            SPARK_DSPARK_HIDDEN_DIMENSION);
        status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    }
    for (layer_index=0u;
         status == SPARK_STATUS_OK &&
             layer_index<SPARK_DSPARK_DRAFT_LAYER_COUNT;
         ++layer_index)
        status = SparkGlm52DsparkAppendContextBatchLayer(
            backend,stage_count,layer_index);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaEventRecord(
            (cudaEvent_t)backend->completion_event,
            (cudaStream_t)backend->cuda_stream));
    if (status != SPARK_STATUS_OK)
        return status;
    memcpy(
        backend->lane_states,
        backend->validation_lane_states,
        (size_t)backend->maximum_lane_count * sizeof(backend->lane_states[0u]));
    backend->pending_operation_kind =
        SPARK_DSPARK_DRAFT_BACKEND_PENDING_STAGE;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkUploadDraftBatchMetadata(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftRequest *requests,
    uint32_t lane_count,
    uint32_t *maximum_requested_token_count)
{
    uint8_t seen[SPARK_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT];
    SparkGlm52DsparkDraftBackendLaneState *lane_state;
    const SparkGlm52DsparkDraftRequest *request;
    uint32_t lane_index,backend_lane_index;
    SparkStatus status;

    memset(seen,0,backend->maximum_lane_count);
    *maximum_requested_token_count = 0u;
    for (lane_index=0u; lane_index<lane_count; ++lane_index)
    {
        request = &requests[lane_index];
        backend_lane_index = request->active_sequence_index;
        if (request->abi_version != SPARK_DSPARK_ABI_VERSION ||
            request->descriptor_bytes !=
                SPARK_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES ||
            request->requested_token_count == 0u ||
            request->requested_token_count >
                SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ||
            backend_lane_index >= backend->maximum_lane_count ||
            seen[backend_lane_index] != 0u)
            return SPARK_STATUS_INVALID_ARGUMENT;
        lane_state = &backend->lane_states[backend_lane_index];
        if (lane_state->staged == 0u ||
            lane_state->sequence_id != request->sequence_id ||
            lane_state->sequence_position != request->sequence_position ||
            lane_state->tap_generation != request->tap_generation ||
            lane_state->sequence_position > UINT32_MAX)
            return SPARK_STATUS_INVALID_ARGUMENT;
        seen[backend_lane_index] = 1u;
        backend->host_backend_lane_indices[lane_index] = backend_lane_index;
        backend->host_sequence_positions[lane_index] =
            (uint32_t)lane_state->sequence_position;
        backend->host_context_token_counts[lane_index] =
            lane_state->context_token_count;
        backend->host_last_token_ids[lane_index] =
            lane_state->last_token_id;
        backend->host_requested_token_counts[lane_index] =
            request->requested_token_count;
        if (request->requested_token_count > *maximum_requested_token_count)
            *maximum_requested_token_count = request->requested_token_count;
    }
    status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
        backend->device_backend_lane_indices,
        backend->host_backend_lane_indices,
        (size_t)lane_count * sizeof(uint32_t),
        cudaMemcpyHostToDevice,
        (cudaStream_t)backend->cuda_stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
            backend->device_sequence_positions,
            backend->host_sequence_positions,
            (size_t)lane_count * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            (cudaStream_t)backend->cuda_stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
            backend->device_context_token_counts,
            backend->host_context_token_counts,
            (size_t)lane_count * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            (cudaStream_t)backend->cuda_stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
            backend->device_last_token_ids,
            backend->host_last_token_ids,
            (size_t)lane_count * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            (cudaStream_t)backend->cuda_stream));
    return status;
}

static SparkStatus SparkGlm52DsparkBlockAttentionBatch(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_count,
    uint32_t layer_index)
{
    uint32_t row_count;
    dim3 rope_grid,attention_grid;
    SparkStatus status;

    row_count = lane_count * SPARK_DSPARK_BLOCK_SIZE;
    SparkGlm52DsparkRmsNormRowsKernel<<<row_count,
        SPARK_DSPARK_BACKEND_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_hidden_bf16,
        (const uint16_t *)backend->device_weights[
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_INPUT_NORM)],
        backend->device_block_normed_bf16,
        row_count,
        SPARK_DSPARK_HIDDEN_DIMENSION);
    status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_Q),
            backend->device_block_normed_bf16,
            backend->device_block_query_bf16,
            row_count,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            0u);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_K),
            backend->device_block_normed_bf16,
            backend->device_block_key_bf16,
            row_count,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            0u);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_V),
            backend->device_block_normed_bf16,
            backend->device_block_value_bf16,
            row_count,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            0u);
    if (status != SPARK_STATUS_OK)
        return status;
    rope_grid = dim3(
        SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT,row_count,1u);
    SparkGlm52DsparkHeadNormRopeBatchKernel<<<rope_grid,
        SPARK_DSPARK_BACKEND_HEAD_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_query_bf16,
        (const uint16_t *)backend->device_weights[
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_Q_NORM)],
        backend->device_sequence_positions,
        backend->device_block_query_bf16,
        row_count,
        SPARK_DSPARK_BLOCK_SIZE);
    SparkGlm52DsparkHeadNormRopeBatchKernel<<<rope_grid,
        SPARK_DSPARK_BACKEND_HEAD_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_key_bf16,
        (const uint16_t *)backend->device_weights[
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_K_NORM)],
        backend->device_sequence_positions,
        backend->device_block_key_bf16,
        row_count,
        SPARK_DSPARK_BLOCK_SIZE);
    status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    if (status != SPARK_STATUS_OK)
        return status;
    attention_grid = dim3(
        SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT,row_count,1u);
    SparkGlm52DsparkBlockAttentionBatchKernel<<<attention_grid,
        SPARK_DSPARK_BACKEND_HEAD_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_query_bf16,
        backend->device_context_key_bf16,
        backend->device_context_value_bf16,
        backend->device_block_key_bf16,
        backend->device_block_value_bf16,
        backend->device_backend_lane_indices,
        backend->device_context_token_counts,
        backend->device_block_attention_bf16,
        row_count,
        layer_index,
        backend->maximum_context_token_count);
    return SparkGlm52DsparkCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52DsparkBlockMlpBatch(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_count,
    uint32_t layer_index)
{
    uint64_t element_count;
    uint32_t row_count,block_count;
    SparkStatus status;

    row_count = lane_count * SPARK_DSPARK_BLOCK_SIZE;
    SparkGlm52DsparkRmsNormRowsKernel<<<row_count,
        SPARK_DSPARK_BACKEND_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_hidden_bf16,
        (const uint16_t *)backend->device_weights[
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_POST_NORM)],
        backend->device_block_normed_bf16,
        row_count,
        SPARK_DSPARK_HIDDEN_DIMENSION);
    status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_GATE),
            backend->device_block_normed_bf16,
            backend->device_block_gate_bf16,
            row_count,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION,
            0u);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_UP),
            backend->device_block_normed_bf16,
            backend->device_block_up_bf16,
            row_count,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION,
            0u);
    if (status != SPARK_STATUS_OK)
        return status;
    element_count = (uint64_t)row_count *
        SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION;
    block_count = (uint32_t)((element_count +
        SPARK_DSPARK_BACKEND_THREADS - 1u) /
        SPARK_DSPARK_BACKEND_THREADS);
    SparkGlm52DsparkSwigluRowsKernel<<<block_count,
        SPARK_DSPARK_BACKEND_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_gate_bf16,
        backend->device_block_up_bf16,
        backend->device_block_mlp_bf16,
        (uint32_t)element_count);
    status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_DOWN),
            backend->device_block_mlp_bf16,
            backend->device_block_final_bf16,
            row_count,
            SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            0u);
    element_count =
        (uint64_t)row_count * SPARK_DSPARK_HIDDEN_DIMENSION;
    block_count = (uint32_t)((element_count +
        SPARK_DSPARK_BACKEND_THREADS - 1u) /
        SPARK_DSPARK_BACKEND_THREADS);
    if (status == SPARK_STATUS_OK)
        SparkGlm52DsparkAddBf16Kernel<<<block_count,
            SPARK_DSPARK_BACKEND_THREADS,0u,
            (cudaStream_t)backend->cuda_stream>>>(
            backend->device_block_hidden_bf16,
            backend->device_block_final_bf16,
            (uint32_t)element_count);
    return status == SPARK_STATUS_OK
        ? SparkGlm52DsparkCudaStatus(cudaGetLastError()) : status;
}

static SparkStatus SparkGlm52DsparkLayerForwardBatch(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_count,
    uint32_t layer_index)
{
    uint64_t element_count;
    uint32_t row_count,block_count;
    SparkStatus status;

    row_count = lane_count * SPARK_DSPARK_BLOCK_SIZE;
    status = SparkGlm52DsparkBlockAttentionBatch(
        backend,lane_count,layer_index);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SparkGlm52DsparkLayerWeightIndex(
                layer_index,SPARK_DSPARK_LAYER_WEIGHT_O),
            backend->device_block_attention_bf16,
            backend->device_block_final_bf16,
            row_count,
            SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            0u);
    element_count =
        (uint64_t)row_count * SPARK_DSPARK_HIDDEN_DIMENSION;
    block_count = (uint32_t)((element_count +
        SPARK_DSPARK_BACKEND_THREADS - 1u) /
        SPARK_DSPARK_BACKEND_THREADS);
    if (status == SPARK_STATUS_OK)
        SparkGlm52DsparkAddBf16Kernel<<<block_count,
            SPARK_DSPARK_BACKEND_THREADS,0u,
            (cudaStream_t)backend->cuda_stream>>>(
            backend->device_block_hidden_bf16,
            backend->device_block_final_bf16,
            (uint32_t)element_count);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkBlockMlpBatch(
            backend,lane_count,layer_index);
    return status;
}

static SparkStatus SparkGlm52DsparkRunHeadStepBatch(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_count,
    uint32_t proposal_index)
{
    SparkStatus status;

    SparkGlm52DsparkGatherMarkovBatchKernel<<<lane_count,
        SPARK_DSPARK_BACKEND_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        (const uint16_t *)backend->device_weights[
            SPARK_DSPARK_WEIGHT_MARKOV_W1],
        backend->device_last_token_ids,
        backend->device_argmax_u32,
        backend->device_block_query_bf16,
        lane_count,
        proposal_index);
    status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SPARK_DSPARK_WEIGHT_MARKOV_W2,
            backend->device_block_query_bf16,
            backend->device_markov_logits_bf16,
            lane_count,
            SPARK_DSPARK_MARKOV_RANK,
            SPARK_DSPARK_FULL_VOCAB_SIZE,
            0u);
    if (status != SPARK_STATUS_OK)
        return status;
    SparkGlm52DsparkArgmaxBatchKernel<<<lane_count,
        SPARK_DSPARK_BACKEND_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_logits_bf16,
        backend->device_markov_logits_bf16,
        backend->device_restricted_token_ids,
        backend->restricted_vocabulary_count,
        proposal_index,
        lane_count,
        backend->device_argmax_u32);
    SparkGlm52DsparkConfidenceBatchKernel<<<lane_count,
        SPARK_DSPARK_BACKEND_THREADS,0u,
        (cudaStream_t)backend->cuda_stream>>>(
        backend->device_block_final_bf16,
        backend->device_block_query_bf16,
        (const uint16_t *)backend->device_weights[
            SPARK_DSPARK_WEIGHT_CONFIDENCE],
        (const uint16_t *)backend->device_weights[
            SPARK_DSPARK_WEIGHT_CONFIDENCE_BIAS],
        proposal_index,
        lane_count,
        backend->device_confidence_f32);
    return SparkGlm52DsparkCudaStatus(cudaGetLastError());
}

SparkStatus SparkGlm52DsparkDraftBackendLaunchDraftBatch(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftRequest *requests,
    uint32_t lane_count)
{
    uint64_t element_count;
    uint32_t row_count,block_count,layer_index,proposal_index;
    uint32_t maximum_requested_token_count;
    SparkStatus status;

    if (backend == 0 || requests == 0 || lane_count == 0u ||
        backend->abi_version != SPARK_DSPARK_DRAFT_BACKEND_ABI_VERSION ||
        lane_count > backend->maximum_lane_count ||
        backend->pending_operation_kind ==
            SPARK_DSPARK_DRAFT_BACKEND_PENDING_DRAFT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkGlm52DsparkUploadDraftBatchMetadata(
        backend,requests,lane_count,&maximum_requested_token_count);
    row_count = lane_count * SPARK_DSPARK_BLOCK_SIZE;
    element_count =
        (uint64_t)row_count * SPARK_DSPARK_HIDDEN_DIMENSION;
    block_count = (uint32_t)((element_count +
        SPARK_DSPARK_BACKEND_THREADS - 1u) /
        SPARK_DSPARK_BACKEND_THREADS);
    if (status == SPARK_STATUS_OK)
    {
        SparkGlm52DsparkBuildQueryBlockBatchKernel<<<block_count,
            SPARK_DSPARK_BACKEND_THREADS,0u,
            (cudaStream_t)backend->cuda_stream>>>(
            (const uint16_t *)backend->device_weights[
                SPARK_DSPARK_WEIGHT_EMBED_TOKENS],
            backend->device_last_token_ids,
            backend->device_block_hidden_bf16,
            lane_count);
        status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    }
    for (layer_index=0u;
         status == SPARK_STATUS_OK &&
             layer_index<SPARK_DSPARK_DRAFT_LAYER_COUNT;
         ++layer_index)
        status = SparkGlm52DsparkLayerForwardBatch(
            backend,lane_count,layer_index);
    if (status == SPARK_STATUS_OK)
    {
        SparkGlm52DsparkRmsNormRowsKernel<<<row_count,
            SPARK_DSPARK_BACKEND_THREADS,0u,
            (cudaStream_t)backend->cuda_stream>>>(
            backend->device_block_hidden_bf16,
            (const uint16_t *)backend->device_weights[
                SPARK_DSPARK_WEIGHT_FINAL_NORM],
            backend->device_block_final_bf16,
            row_count,
            SPARK_DSPARK_HIDDEN_DIMENSION);
        status = SparkGlm52DsparkCudaStatus(cudaGetLastError());
    }
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkLaunchGemm(
            backend,
            SPARK_DSPARK_WEIGHT_LM_HEAD,
            backend->device_block_final_bf16,
            backend->device_block_logits_bf16,
            row_count,
            SPARK_DSPARK_HIDDEN_DIMENSION,
            SPARK_DSPARK_FULL_VOCAB_SIZE,
            0u);
    for (proposal_index=0u;
         status == SPARK_STATUS_OK &&
             proposal_index<maximum_requested_token_count;
         ++proposal_index)
        status = SparkGlm52DsparkRunHeadStepBatch(
            backend,lane_count,proposal_index);
    element_count = (uint64_t)lane_count *
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
            backend->host_argmax_u32,
            backend->device_argmax_u32,
            (size_t)element_count * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            (cudaStream_t)backend->cuda_stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaMemcpyAsync(
            backend->host_confidence_f32,
            backend->device_confidence_f32,
            (size_t)element_count * sizeof(float),
            cudaMemcpyDeviceToHost,
            (cudaStream_t)backend->cuda_stream));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52DsparkCudaStatus(cudaEventRecord(
            (cudaEvent_t)backend->completion_event,
            (cudaStream_t)backend->cuda_stream));
    if (status != SPARK_STATUS_OK)
        return status;
    backend->pending_operation_kind =
        SPARK_DSPARK_DRAFT_BACKEND_PENDING_DRAFT;
    backend->pending_draft_lane_count = lane_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkDraftBackendTakeBatchResults(
    SparkGlm52DsparkDraftBackend *backend,
    SparkGlm52DsparkDraftResult *results,
    uint32_t result_capacity,
    uint32_t *result_count)
{
    uint32_t lane_index,proposal_index,confidence_milli;
    uint64_t result_offset;
    cudaError_t event_status;

    if (backend == 0 || result_count == 0 ||
        backend->abi_version != SPARK_DSPARK_DRAFT_BACKEND_ABI_VERSION ||
        backend->pending_operation_kind ==
            SPARK_DSPARK_DRAFT_BACKEND_PENDING_NONE)
        return SPARK_STATUS_INVALID_ARGUMENT;
    event_status = cudaEventQuery((cudaEvent_t)backend->completion_event);
    if (event_status == cudaErrorNotReady)
        return SPARK_STATUS_BUSY;
    if (event_status != cudaSuccess)
        return SparkGlm52DsparkCudaStatus(event_status);
    if (backend->pending_operation_kind ==
        SPARK_DSPARK_DRAFT_BACKEND_PENDING_STAGE)
    {
        *result_count = 0u;
        backend->pending_operation_kind =
            SPARK_DSPARK_DRAFT_BACKEND_PENDING_NONE;
        return SPARK_STATUS_OK;
    }
    if (results == 0 ||
        result_capacity < backend->pending_draft_lane_count)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    for (lane_index=0u;
         lane_index<backend->pending_draft_lane_count;
         ++lane_index)
    {
        memset(&results[lane_index],0,sizeof(results[lane_index]));
        results[lane_index].abi_version = SPARK_DSPARK_ABI_VERSION;
        results[lane_index].descriptor_bytes =
            SPARK_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES;
        results[lane_index].token_count =
            backend->host_requested_token_counts[lane_index];
        result_offset = (uint64_t)lane_index *
            SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
        for (proposal_index=0u;
             proposal_index<results[lane_index].token_count;
             ++proposal_index)
        {
            results[lane_index].token_ids[proposal_index] =
                backend->host_argmax_u32[result_offset + proposal_index];
            confidence_milli = (uint32_t)(
                backend->host_confidence_f32[result_offset + proposal_index] *
                (float)SPARK_DSPARK_CONFIDENCE_MILLI_ONE);
            if (confidence_milli > SPARK_DSPARK_CONFIDENCE_MILLI_ONE)
                confidence_milli =
                    SPARK_DSPARK_CONFIDENCE_MILLI_ONE;
            results[lane_index].confidence_milli[proposal_index] =
                confidence_milli;
        }
    }
    *result_count = backend->pending_draft_lane_count;
    backend->pending_draft_lane_count = 0u;
    backend->pending_operation_kind =
        SPARK_DSPARK_DRAFT_BACKEND_PENDING_NONE;
    return SPARK_STATUS_OK;
}
