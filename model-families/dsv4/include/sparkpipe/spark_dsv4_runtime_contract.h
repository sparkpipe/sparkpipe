#pragma once

#include <string.h>

#include "sparkpipe/spark_model_runtime.h"

static inline SparkModelRuntimeContract SparkDsv4RuntimeContractBase(
    const char *contract_id,
    uint64_t attention_operation_mask,
    uint64_t auxiliary_operation_mask,
    uint32_t nonexpert_activation_precision)
{
    SparkModelRuntimeContract contract;

    memset(&contract,0,sizeof(contract));
    contract.abi_version = SPARK_MODEL_RUNTIME_ABI_VERSION;
    contract.descriptor_bytes = SPARK_MODEL_RUNTIME_CONTRACT_BYTES;
    contract.contract_id = contract_id;
    contract.precision_policy.expert_weight_precision =
        SPARK_MODEL_RUNTIME_PRECISION_FP4;
    contract.precision_policy.expert_activation_precision =
        SPARK_MODEL_RUNTIME_PRECISION_FP8_E4M3;
    contract.precision_policy.nonexpert_weight_precision =
        SPARK_MODEL_RUNTIME_PRECISION_FP8_E4M3;
    contract.precision_policy.nonexpert_activation_precision =
        nonexpert_activation_precision;
    contract.precision_policy.accumulator_precision =
        SPARK_MODEL_RUNTIME_PRECISION_FP32;
    contract.required_operation_mask = attention_operation_mask |
        SPARK_MODEL_RUNTIME_OPERATION_HYPER_CONNECTION |
        SPARK_MODEL_RUNTIME_OPERATION_HASH_ROUTED_MOE |
        SPARK_MODEL_RUNTIME_OPERATION_GROUPED_MOE |
        auxiliary_operation_mask |
        SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD;
    return contract;
}

static inline SparkModelRuntimeContract SparkDsv4FlashRuntimeContract(void)
{
    return SparkDsv4RuntimeContractBase(
        "deepseek-v4-flash-0731-ga-baseline-mixed-v3",
        SPARK_MODEL_RUNTIME_OPERATION_SLIDING_ATTENTION |
        SPARK_MODEL_RUNTIME_OPERATION_COMPRESSED_SPARSE_ATTENTION |
        SPARK_MODEL_RUNTIME_OPERATION_HIERARCHICAL_COMPRESSED_ATTENTION,
        0u,
        SPARK_MODEL_RUNTIME_PRECISION_BF16);
}

static inline SparkModelRuntimeContract SparkDsv4ProRuntimeContract(void)
{
    return SparkDsv4RuntimeContractBase(
        "deepseek-v4-pro-checkpoint-mixed-v1",
        SPARK_MODEL_RUNTIME_OPERATION_COMPRESSED_SPARSE_ATTENTION |
        SPARK_MODEL_RUNTIME_OPERATION_HIERARCHICAL_COMPRESSED_ATTENTION,
        SPARK_MODEL_RUNTIME_OPERATION_MULTI_TOKEN_PREDICTION,
        SPARK_MODEL_RUNTIME_PRECISION_FP8_E4M3);
}

static inline SparkStatus SparkDsv4FlashRuntimeValidateContract(
    const SparkModelRuntimeContract *contract)
{
    SparkModelRuntimeContract expected_contract;

    expected_contract = SparkDsv4FlashRuntimeContract();
    return SparkModelRuntimeContractsMatch(contract,&expected_contract) != 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED;
}

static inline SparkStatus SparkDsv4ProRuntimeValidateContract(
    const SparkModelRuntimeContract *contract)
{
    SparkModelRuntimeContract expected_contract;

    expected_contract = SparkDsv4ProRuntimeContract();
    return SparkModelRuntimeContractsMatch(contract,&expected_contract) != 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED;
}
