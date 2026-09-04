#ifndef SPARKPIPE_SPARK_K3_RUNTIME_CONTRACT_H
#define SPARKPIPE_SPARK_K3_RUNTIME_CONTRACT_H

#include <string.h>

#include "sparkpipe/spark_model_runtime.h"

static inline SparkModelRuntimeContract SparkK3RuntimeContract(void)
{
    SparkModelRuntimeContract contract;

    memset(&contract,0,sizeof(contract));
    contract.abi_version = SPARK_MODEL_RUNTIME_ABI_VERSION;
    contract.descriptor_bytes = SPARK_MODEL_RUNTIME_CONTRACT_BYTES;
    contract.contract_id = "kimi-k3-mxfp4-expert-bf16-rest-v1";
    contract.precision_policy.expert_weight_precision =
        SPARK_MODEL_RUNTIME_PRECISION_MXFP4;
    contract.precision_policy.expert_activation_precision =
        SPARK_MODEL_RUNTIME_PRECISION_BF16;
    contract.precision_policy.nonexpert_weight_precision =
        SPARK_MODEL_RUNTIME_PRECISION_BF16;
    contract.precision_policy.nonexpert_activation_precision =
        SPARK_MODEL_RUNTIME_PRECISION_BF16;
    contract.precision_policy.accumulator_precision =
        SPARK_MODEL_RUNTIME_PRECISION_FP32;
    contract.required_operation_mask =
        SPARK_MODEL_RUNTIME_OPERATION_LINEAR_ATTENTION |
        SPARK_MODEL_RUNTIME_OPERATION_GLOBAL_ATTENTION |
        SPARK_MODEL_RUNTIME_OPERATION_ROUTED_MOE |
        SPARK_MODEL_RUNTIME_OPERATION_SHARED_MOE |
        SPARK_MODEL_RUNTIME_OPERATION_RECURRENT_REPLAY |
        SPARK_MODEL_RUNTIME_OPERATION_DEPTH_STATE_DELTA |
        SPARK_MODEL_RUNTIME_OPERATION_DRAFT |
        SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD;
    return contract;
}

static inline SparkStatus SparkK3RuntimeValidateContract(
    const SparkModelRuntimeContract *contract)
{
    SparkModelRuntimeContract expected_contract;

    expected_contract = SparkK3RuntimeContract();
    return SparkModelRuntimeContractsMatch(contract,&expected_contract) != 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED;
}

#endif
