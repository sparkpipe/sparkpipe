#ifndef SPARKPIPE_SPARK_MODEL_RUNTIME_H
#define SPARKPIPE_SPARK_MODEL_RUNTIME_H

#include <stdint.h>

#include "sparkpipe/spark_runtime_completion.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_RUNTIME_ABI_VERSION 1u
#define SPARK_MODEL_RUNTIME_ARTIFACT_SHA256_LENGTH 64u

#define SPARK_MODEL_RUNTIME_PRECISION_NONE 0u
#define SPARK_MODEL_RUNTIME_PRECISION_MXFP4 1u
#define SPARK_MODEL_RUNTIME_PRECISION_FP4 2u
#define SPARK_MODEL_RUNTIME_PRECISION_FP8_E4M3 3u
#define SPARK_MODEL_RUNTIME_PRECISION_BF16 4u
#define SPARK_MODEL_RUNTIME_PRECISION_FP32 5u
#define SPARK_MODEL_RUNTIME_PRECISION_KNOWN_MAX \
    SPARK_MODEL_RUNTIME_PRECISION_FP32

#define SPARK_MODEL_RUNTIME_OPERATION_LINEAR_ATTENTION \
    UINT64_C(0x0000000000000001)
#define SPARK_MODEL_RUNTIME_OPERATION_GLOBAL_ATTENTION \
    UINT64_C(0x0000000000000002)
#define SPARK_MODEL_RUNTIME_OPERATION_ROUTED_MOE \
    UINT64_C(0x0000000000000004)
#define SPARK_MODEL_RUNTIME_OPERATION_RECURRENT_REPLAY \
    UINT64_C(0x0000000000000008)
#define SPARK_MODEL_RUNTIME_OPERATION_DEPTH_STATE_DELTA \
    UINT64_C(0x0000000000000010)
#define SPARK_MODEL_RUNTIME_OPERATION_DRAFT \
    UINT64_C(0x0000000000000020)
#define SPARK_MODEL_RUNTIME_OPERATION_POSITION_ENCODING \
    UINT64_C(0x0000000000000040)
#define SPARK_MODEL_RUNTIME_OPERATION_GATE_PROJECTION \
    UINT64_C(0x0000000000000080)
#define SPARK_MODEL_RUNTIME_OPERATION_RECURRENT_UPDATE \
    UINT64_C(0x0000000000000100)
#define SPARK_MODEL_RUNTIME_OPERATION_ATTENTION_OUTPUT_GATE \
    UINT64_C(0x0000000000000200)
#define SPARK_MODEL_RUNTIME_OPERATION_SLIDING_ATTENTION \
    UINT64_C(0x0000000000000400)
#define SPARK_MODEL_RUNTIME_OPERATION_COMPRESSED_SPARSE_ATTENTION \
    UINT64_C(0x0000000000000800)
#define SPARK_MODEL_RUNTIME_OPERATION_HIERARCHICAL_COMPRESSED_ATTENTION \
    UINT64_C(0x0000000000001000)
#define SPARK_MODEL_RUNTIME_OPERATION_HYPER_CONNECTION \
    UINT64_C(0x0000000000002000)
#define SPARK_MODEL_RUNTIME_OPERATION_HASH_ROUTED_MOE \
    UINT64_C(0x0000000000004000)
#define SPARK_MODEL_RUNTIME_OPERATION_GROUPED_MOE \
    UINT64_C(0x0000000000008000)
#define SPARK_MODEL_RUNTIME_OPERATION_MULTI_TOKEN_PREDICTION \
    UINT64_C(0x0000000000010000)
#define SPARK_MODEL_RUNTIME_OPERATION_DENSE_MLP \
    UINT64_C(0x0000000000020000)
#define SPARK_MODEL_RUNTIME_OPERATION_ROUTER \
    UINT64_C(0x0000000000040000)
#define SPARK_MODEL_RUNTIME_OPERATION_SPARSE_INDEX \
    UINT64_C(0x0000000000080000)
#define SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD \
    UINT64_C(0x0000000000100000)
#define SPARK_MODEL_RUNTIME_OPERATION_SHARED_MOE \
    UINT64_C(0x0000000000200000)
#define SPARK_MODEL_RUNTIME_OPERATION_KNOWN_MASK \
    (SPARK_MODEL_RUNTIME_OPERATION_LINEAR_ATTENTION | \
     SPARK_MODEL_RUNTIME_OPERATION_GLOBAL_ATTENTION | \
     SPARK_MODEL_RUNTIME_OPERATION_ROUTED_MOE | \
     SPARK_MODEL_RUNTIME_OPERATION_RECURRENT_REPLAY | \
     SPARK_MODEL_RUNTIME_OPERATION_DEPTH_STATE_DELTA | \
     SPARK_MODEL_RUNTIME_OPERATION_DRAFT | \
     SPARK_MODEL_RUNTIME_OPERATION_POSITION_ENCODING | \
     SPARK_MODEL_RUNTIME_OPERATION_GATE_PROJECTION | \
     SPARK_MODEL_RUNTIME_OPERATION_RECURRENT_UPDATE | \
     SPARK_MODEL_RUNTIME_OPERATION_ATTENTION_OUTPUT_GATE | \
     SPARK_MODEL_RUNTIME_OPERATION_SLIDING_ATTENTION | \
     SPARK_MODEL_RUNTIME_OPERATION_COMPRESSED_SPARSE_ATTENTION | \
     SPARK_MODEL_RUNTIME_OPERATION_HIERARCHICAL_COMPRESSED_ATTENTION | \
     SPARK_MODEL_RUNTIME_OPERATION_HYPER_CONNECTION | \
     SPARK_MODEL_RUNTIME_OPERATION_HASH_ROUTED_MOE | \
     SPARK_MODEL_RUNTIME_OPERATION_GROUPED_MOE | \
     SPARK_MODEL_RUNTIME_OPERATION_MULTI_TOKEN_PREDICTION | \
     SPARK_MODEL_RUNTIME_OPERATION_DENSE_MLP | \
     SPARK_MODEL_RUNTIME_OPERATION_ROUTER | \
     SPARK_MODEL_RUNTIME_OPERATION_SPARSE_INDEX | \
     SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD | \
     SPARK_MODEL_RUNTIME_OPERATION_SHARED_MOE)

#define SPARK_MODEL_RUNTIME_PROVIDER_FLAG_EXACT_ARTIFACT \
    UINT32_C(0x00000001)
#define SPARK_MODEL_RUNTIME_PROVIDER_FLAG_TRANSACTIONAL_STAGING \
    UINT32_C(0x00000002)
#define SPARK_MODEL_RUNTIME_PROVIDER_FLAG_COMMIT_INFALLIBLE \
    UINT32_C(0x00000004)
#define SPARK_MODEL_RUNTIME_PROVIDER_FLAG_CANCEL_IDEMPOTENT \
    UINT32_C(0x00000008)
#define SPARK_MODEL_RUNTIME_PROVIDER_REQUIRED_FLAGS \
    (SPARK_MODEL_RUNTIME_PROVIDER_FLAG_EXACT_ARTIFACT | \
     SPARK_MODEL_RUNTIME_PROVIDER_FLAG_TRANSACTIONAL_STAGING | \
     SPARK_MODEL_RUNTIME_PROVIDER_FLAG_COMMIT_INFALLIBLE | \
     SPARK_MODEL_RUNTIME_PROVIDER_FLAG_CANCEL_IDEMPOTENT)
#define SPARK_MODEL_RUNTIME_PROVIDER_KNOWN_FLAGS \
    SPARK_MODEL_RUNTIME_PROVIDER_REQUIRED_FLAGS

typedef struct SparkModelRuntimePrecisionPolicy
{
    uint32_t expert_weight_precision;
    uint32_t expert_activation_precision;
    uint32_t nonexpert_weight_precision;
    uint32_t nonexpert_activation_precision;
    uint32_t accumulator_precision;
    uint32_t reserved[3];
} SparkModelRuntimePrecisionPolicy;

typedef struct SparkModelRuntimeContract
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *contract_id;
    SparkModelRuntimePrecisionPolicy precision_policy;
    uint64_t required_operation_mask;
    uint64_t reserved[2];
} SparkModelRuntimeContract;

typedef SparkStatus (*SparkModelRuntimeContractValidationFunction)(
    const SparkModelRuntimeContract *contract);

typedef SparkStatus (*SparkModelRuntimeBeginFunction)(
    void *provider_context,
    const SparkRuntimeTransactionRequest *request);
typedef SparkStatus (*SparkModelRuntimeInvokeFunction)(
    void *provider_context,
    const SparkRuntimeTransactionRequest *request,
    uint64_t operation,
    uint32_t operation_ordinal);
typedef void (*SparkModelRuntimeCommitFunction)(
    void *provider_context,
    const SparkRuntimeTransactionRequest *request);
typedef void (*SparkModelRuntimeCancelFunction)(
    void *provider_context,
    const SparkRuntimeTransactionRequest *request,
    SparkStatus reason);

typedef struct SparkModelRuntimeProvider
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t reserved0;
    const char *model_id;
    const char *model_revision;
    const char *artifact_sha256;
    const SparkModelRuntimeContract *contract;
    uint64_t supported_operation_mask;
    const uint64_t *operation_sequence;
    uint32_t operation_count;
    uint32_t reserved1;
    SparkModelRuntimeContractValidationFunction validate_contract;
    void *provider_context;
    SparkModelRuntimeBeginFunction begin;
    SparkModelRuntimeInvokeFunction invoke;
    SparkModelRuntimeCommitFunction commit;
    SparkModelRuntimeCancelFunction cancel;
} SparkModelRuntimeProvider;

#define SPARK_MODEL_RUNTIME_PRECISION_POLICY_BYTES \
    ((uint32_t)sizeof(SparkModelRuntimePrecisionPolicy))
#define SPARK_MODEL_RUNTIME_CONTRACT_BYTES \
    ((uint32_t)sizeof(SparkModelRuntimeContract))
#define SPARK_MODEL_RUNTIME_PROVIDER_BYTES \
    ((uint32_t)sizeof(SparkModelRuntimeProvider))

uint32_t SparkModelRuntimeArtifactSha256IsValid(
    const char *artifact_sha256);
SparkStatus SparkModelRuntimeValidateContract(
    const SparkModelRuntimeContract *contract);
uint32_t SparkModelRuntimeContractsMatch(
    const SparkModelRuntimeContract *left,
    const SparkModelRuntimeContract *right);
SparkStatus SparkModelRuntimeValidateProvider(
    const SparkModelRuntimeProvider *provider);
SparkStatus SparkModelRuntimeExecute(
    const SparkModelRuntimeProvider *provider,
    const SparkRuntimeTransactionRequest *request);

#ifdef __cplusplus
}
#endif

#endif
