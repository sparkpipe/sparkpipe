#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_DEFAULT_WORKSPACE_BYTES \
    (32ull * 1024ull * 1024ull)

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE 0x00000001u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP 0x00000002u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN 0x00000004u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_ROUTER_LOGITS 0x00000008u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_REQUIRED_GLM52_PREFIX \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_ROUTER_LOGITS)

typedef struct SparkGlm52ResidentDecodeStageLinearPlanResidentBinding
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding;

typedef struct SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo
{
    uint32_t abi_version;
    uint32_t maximum_active_sequence_count;
    uint32_t dense_intermediate_dimension;
    uint32_t expert_count;
    uint32_t required_plan_mask;
    uint32_t autotune_warmup_iterations;
    uint32_t autotune_measurement_iterations;
    uint32_t reserved0;
    uint64_t workspace_limit_bytes;
    void *cuda_stream;
    const void *dense_input_bf16;
    const void *dense_gate_weight_bf16;
    const void *dense_up_weight_bf16;
    const void *dense_down_weight_bf16;
    void *dense_gate_output_bf16;
    void *dense_up_output_bf16;
    void *dense_intermediate_bf16;
    void *dense_down_output_bf16;
    const void *router_input_bf16;
    const void *router_weight_bf16;
    void *router_logits_f32;
} SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo;

SparkStatus SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreate(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding **binding_out,
    const SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo *create_info);

void SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding);

const SparkGlm52ResidentDecodeStageLinearPlan *
SparkGlm52ResidentDecodeStageLinearPlanResidentBindingPlans(
    const SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    uint32_t *plan_count_out);

#ifdef __cplusplus
}
#endif
