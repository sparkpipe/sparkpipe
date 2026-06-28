#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_CUBLASLT_PLANS_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_CUBLASLT_PLANS_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkGlm52ResidentDecodeStageCublasLtLinearPlanBuildRequest
{
    uint32_t plan_kind;
    uint32_t input_dimension;
    uint32_t output_dimension;
    uint32_t maximum_active_sequence_count;
    uint32_t output_is_f32;
    void *cublaslt_handle;
    void *workspace;
    uint64_t workspace_bytes;
} SparkGlm52ResidentDecodeStageCublasLtLinearPlanBuildRequest;

SparkStatus SparkGlm52ResidentDecodeStageBuildCublasLtLinearPlan(
    const SparkGlm52ResidentDecodeStageCublasLtLinearPlanBuildRequest *request,
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plan);

void SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlan(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plan);

#ifdef __cplusplus
}
#endif

#endif
