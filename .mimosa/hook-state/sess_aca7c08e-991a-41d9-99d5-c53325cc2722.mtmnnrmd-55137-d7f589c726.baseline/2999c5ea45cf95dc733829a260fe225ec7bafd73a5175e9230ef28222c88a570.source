#ifndef SPARKPIPE_SPARK_QWEN38_27B_RUNTIME_CONTRACT_H
#define SPARKPIPE_SPARK_QWEN38_27B_RUNTIME_CONTRACT_H

#include <string.h>

#include "sparkpipe/spark_model_runtime.h"

static inline SparkModelRuntimeContract SparkQwen38_27bRuntimeContract(void)
{
    SparkModelRuntimeContract contract;

    memset(&contract,0,sizeof(contract));
    contract.abi_version = SPARK_MODEL_RUNTIME_ABI_VERSION;
    contract.descriptor_bytes = SPARK_MODEL_RUNTIME_CONTRACT_BYTES;
    contract.contract_id = "qwen-3.6-27b-bf16-v1";
    contract.precision_policy.expert_weight_precision =
        SPARK_MODEL_RUNTIME_PRECISION_NONE;
    contract.precision_policy.expert_activation_precision =
        SPARK_MODEL_RUNTIME_PRECISION_NONE;
    contract.precision_policy.nonexpert_weight_precision =
        SPARK_MODEL_RUNTIME_PRECISION_BF16;
    contract.precision_policy.nonexpert_activation_precision =
        SPARK_MODEL_RUNTIME_PRECISION_BF16;
    contract.precision_policy.accumulator_precision =
        SPARK_MODEL_RUNTIME_PRECISION_FP32;
    contract.required_operation_mask =
        SPARK_MODEL_RUNTIME_OPERATION_POSITION_ENCODING |
        SPARK_MODEL_RUNTIME_OPERATION_GATE_PROJECTION |
        SPARK_MODEL_RUNTIME_OPERATION_RECURRENT_UPDATE |
        SPARK_MODEL_RUNTIME_OPERATION_ATTENTION_OUTPUT_GATE |
        SPARK_MODEL_RUNTIME_OPERATION_DENSE_MLP |
        SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD;
    return contract;
}

static inline SparkStatus SparkQwen38_27bRuntimeValidateContract(
    const SparkModelRuntimeContract *contract)
{
    SparkModelRuntimeContract expected_contract;

    expected_contract = SparkQwen38_27bRuntimeContract();
    return SparkModelRuntimeContractsMatch(contract,&expected_contract) != 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED;
}

#endif
