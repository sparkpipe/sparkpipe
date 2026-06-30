#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PLAN_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PLAN_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_MAGIC_BYTES 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_MAGIC "SPARKGLM52B12X\0\0"
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_HEADER_BYTES 512u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_ALIGNMENT 4096u

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W1_WEIGHT 0u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W1_SCALE 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W1_ALPHA 2u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_FC2_INPUT_SCALE 3u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W2_WEIGHT 4u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W2_SCALE 5u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W2_ALPHA 6u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_COUNT 7u

typedef struct SparkGlm52ResidentDecodeStageB12xMoePackRegion
{
    uint64_t offset;
    uint64_t bytes;
} SparkGlm52ResidentDecodeStageB12xMoePackRegion;

typedef struct SparkGlm52ResidentDecodeStageB12xMoePackHeader
{
    uint8_t magic[SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_MAGIC_BYTES];
    uint32_t abi_version;
    uint32_t header_bytes;
    uint32_t layer_index;
    uint32_t maximum_token_count;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t gate_up_order;
    uint32_t weight_layout;
    uint32_t scale_layout;
    uint32_t quant_mode;
    uint32_t output_dtype;
    uint32_t cuda_architecture;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t qualified_maximum_microseconds;
    uint64_t qualification_record_hash_low64;
    uint64_t kernel_manifest_hash_low64;
    uint64_t pack_hash_low64;
    SparkGlm52ResidentDecodeStageB12xMoePackRegion regions[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_COUNT];
    uint8_t reserved_bytes[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_HEADER_BYTES -
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_MAGIC_BYTES -
        (16u * sizeof(uint32_t)) -
        (4u * sizeof(uint64_t)) -
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_COUNT *
         sizeof(SparkGlm52ResidentDecodeStageB12xMoePackRegion))];
} SparkGlm52ResidentDecodeStageB12xMoePackHeader;

typedef struct SparkGlm52ResidentDecodeStageB12xMoeResidentBinding
{
    uint32_t abi_version;
    uint32_t layer_index;
    SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan dispatch_plan;
    SparkGlm52ResidentDecodeStageB12xMoePlan plan;
    void *state_cell;
    void *w1_weight_fp4_static_view;
    void *w1_scale_static_storage_ue4m3;
    void *w1_alpha_fp32_by_expert;
    void *fc2_input_scale_fp32_by_expert;
    void *w2_weight_fp4_static_view;
    void *w2_scale_static_storage_ue4m3;
    void *w2_alpha_fp32_by_expert;
} SparkGlm52ResidentDecodeStageB12xMoeResidentBinding;

typedef struct SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateInfo
{
    uint32_t abi_version;
    uint32_t layer_index;
    uint32_t maximum_active_sequence_count;
    uint32_t reserved;
    const char *pack_path;
} SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateInfo;

SparkStatus SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateFromPackFile(
    SparkGlm52ResidentDecodeStageB12xMoeResidentBinding *binding,
    const SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateInfo *create_info);

void SparkGlm52ResidentDecodeStageB12xMoeResidentBindingDestroy(
    SparkGlm52ResidentDecodeStageB12xMoeResidentBinding *binding);

#ifdef __cplusplus
}
#endif

#endif
