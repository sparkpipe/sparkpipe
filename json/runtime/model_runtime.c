#include "sparkpipe/spark_model_runtime.h"

#include <stddef.h>
#include <string.h>

static uint32_t SparkModelRuntimePrecisionIsKnown(
    uint32_t precision)
{
    return precision >= SPARK_MODEL_RUNTIME_PRECISION_MXFP4 &&
        precision <= SPARK_MODEL_RUNTIME_PRECISION_KNOWN_MAX;
}

static uint32_t SparkModelRuntimeOperationIsSingleBit(
    uint64_t operation)
{
    return operation != 0u && (operation & (operation - 1u)) == 0u;
}

uint32_t SparkModelRuntimeArtifactSha256IsValid(
    const char *artifact_sha256)
{
    uint32_t character_index;

    if (artifact_sha256 == 0)
    {
        return 0u;
    }
    for (character_index = 0u;
         character_index < SPARK_MODEL_RUNTIME_ARTIFACT_SHA256_LENGTH;
         ++character_index)
    {
        char character;

        character = artifact_sha256[character_index];
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f')))
        {
            return 0u;
        }
    }
    return artifact_sha256[SPARK_MODEL_RUNTIME_ARTIFACT_SHA256_LENGTH] ==
        '\0';
}

SparkStatus SparkModelRuntimeValidateContract(
    const SparkModelRuntimeContract *contract)
{
    const SparkModelRuntimePrecisionPolicy *precision_policy;
    uint32_t reserved_index;

    if (contract == 0 ||
        contract->abi_version != SPARK_MODEL_RUNTIME_ABI_VERSION ||
        contract->descriptor_bytes != SPARK_MODEL_RUNTIME_CONTRACT_BYTES ||
        contract->contract_id == 0 || contract->contract_id[0] == '\0' ||
        contract->required_operation_mask == 0u ||
        (contract->required_operation_mask &
            ~SPARK_MODEL_RUNTIME_OPERATION_KNOWN_MASK) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    precision_policy = &contract->precision_policy;
    if (!((precision_policy->expert_weight_precision ==
                SPARK_MODEL_RUNTIME_PRECISION_NONE &&
            precision_policy->expert_activation_precision ==
                SPARK_MODEL_RUNTIME_PRECISION_NONE) ||
           (SparkModelRuntimePrecisionIsKnown(
                precision_policy->expert_weight_precision) != 0u &&
            SparkModelRuntimePrecisionIsKnown(
                precision_policy->expert_activation_precision) != 0u)) ||
        SparkModelRuntimePrecisionIsKnown(
            precision_policy->nonexpert_weight_precision) == 0u ||
        SparkModelRuntimePrecisionIsKnown(
            precision_policy->nonexpert_activation_precision) == 0u ||
        precision_policy->accumulator_precision !=
            SPARK_MODEL_RUNTIME_PRECISION_FP32)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    for (reserved_index = 0u; reserved_index < 3u; ++reserved_index)
    {
        if (precision_policy->reserved[reserved_index] != 0u)
        {
            return SPARK_STATUS_VALIDATION_FAILED;
        }
    }
    for (reserved_index = 0u; reserved_index < 2u; ++reserved_index)
    {
        if (contract->reserved[reserved_index] != 0u)
        {
            return SPARK_STATUS_VALIDATION_FAILED;
        }
    }
    return SPARK_STATUS_OK;
}

uint32_t SparkModelRuntimeContractsMatch(
    const SparkModelRuntimeContract *left,
    const SparkModelRuntimeContract *right)
{
    uint32_t reserved_index;

    if (left == 0 || right == 0 ||
        left->abi_version != right->abi_version ||
        left->descriptor_bytes != right->descriptor_bytes ||
        left->contract_id == 0 || right->contract_id == 0 ||
        strcmp(left->contract_id,right->contract_id) != 0 ||
        left->precision_policy.expert_weight_precision !=
            right->precision_policy.expert_weight_precision ||
        left->precision_policy.expert_activation_precision !=
            right->precision_policy.expert_activation_precision ||
        left->precision_policy.nonexpert_weight_precision !=
            right->precision_policy.nonexpert_weight_precision ||
        left->precision_policy.nonexpert_activation_precision !=
            right->precision_policy.nonexpert_activation_precision ||
        left->precision_policy.accumulator_precision !=
            right->precision_policy.accumulator_precision ||
        left->required_operation_mask != right->required_operation_mask)
    {
        return 0u;
    }
    for (reserved_index = 0u; reserved_index < 3u; ++reserved_index)
    {
        if (left->precision_policy.reserved[reserved_index] !=
            right->precision_policy.reserved[reserved_index])
        {
            return 0u;
        }
    }
    for (reserved_index = 0u; reserved_index < 2u; ++reserved_index)
    {
        if (left->reserved[reserved_index] != right->reserved[reserved_index])
        {
            return 0u;
        }
    }
    return 1u;
}

SparkStatus SparkModelRuntimeValidateProvider(
    const SparkModelRuntimeProvider *provider)
{
    uint64_t observed_operation_mask;
    uint32_t operation_index;
    SparkStatus status;

    if (provider == 0 ||
        provider->abi_version != SPARK_MODEL_RUNTIME_ABI_VERSION ||
        provider->descriptor_bytes != SPARK_MODEL_RUNTIME_PROVIDER_BYTES ||
        (provider->flags & SPARK_MODEL_RUNTIME_PROVIDER_REQUIRED_FLAGS) !=
            SPARK_MODEL_RUNTIME_PROVIDER_REQUIRED_FLAGS ||
        (provider->flags & ~SPARK_MODEL_RUNTIME_PROVIDER_KNOWN_FLAGS) != 0u ||
        provider->reserved0 != 0u ||
        provider->model_id == 0 || provider->model_id[0] == '\0' ||
        provider->model_revision == 0 ||
        provider->model_revision[0] == '\0' ||
        SparkModelRuntimeArtifactSha256IsValid(
            provider->artifact_sha256) == 0u ||
        provider->contract == 0 ||
        provider->supported_operation_mask == 0u ||
        (provider->supported_operation_mask &
            ~SPARK_MODEL_RUNTIME_OPERATION_KNOWN_MASK) != 0u ||
        provider->operation_sequence == 0 ||
        provider->operation_count == 0u ||
        provider->reserved1 != 0u ||
        provider->validate_contract == 0 ||
        provider->begin == 0 ||
        provider->invoke == 0 ||
        provider->commit == 0 ||
        provider->cancel == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkModelRuntimeValidateContract(provider->contract);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = provider->validate_contract(provider->contract);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    observed_operation_mask = 0u;
    for (operation_index = 0u;
         operation_index < provider->operation_count;
         ++operation_index)
    {
        uint64_t operation;

        operation = provider->operation_sequence[operation_index];
        if (SparkModelRuntimeOperationIsSingleBit(operation) == 0u ||
            (operation & ~SPARK_MODEL_RUNTIME_OPERATION_KNOWN_MASK) != 0u ||
            (provider->supported_operation_mask & operation) == 0u)
        {
            return SPARK_STATUS_VALIDATION_FAILED;
        }
        observed_operation_mask |= operation;
    }
    if ((provider->supported_operation_mask &
            provider->contract->required_operation_mask) !=
            provider->contract->required_operation_mask ||
        (observed_operation_mask &
            provider->contract->required_operation_mask) !=
            provider->contract->required_operation_mask ||
        provider->operation_sequence[provider->operation_count - 1u] !=
            SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkModelRuntimeExecute(
    const SparkModelRuntimeProvider *provider,
    const SparkRuntimeTransactionRequest *request)
{
    SparkStatus status;
    uint32_t operation_index;

    status = SparkModelRuntimeValidateProvider(provider);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (request == 0 ||
        request->descriptor_bytes != SPARK_RUNTIME_TRANSACTION_REQUEST_BYTES ||
        SparkWorkTransactionValidateIdentity(&request->identity) !=
            SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = provider->begin(provider->provider_context,request);
    if (status != SPARK_STATUS_OK)
    {
        provider->cancel(provider->provider_context,request,status);
        return status;
    }
    for (operation_index = 0u;
         operation_index < provider->operation_count;
         ++operation_index)
    {
        status = provider->invoke(
            provider->provider_context,
            request,
            provider->operation_sequence[operation_index],
            operation_index);
        if (status != SPARK_STATUS_OK)
        {
            provider->cancel(provider->provider_context,request,status);
            return status;
        }
    }
    provider->commit(provider->provider_context,request);
    return SPARK_STATUS_OK;
}
