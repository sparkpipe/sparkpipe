#pragma once

/*
 * Compatibility home for the flat-dialect stage-pack mechanics' historical
 * names. Everything mechanical moved to the ONE parameterizable reader
 * (runtime/spark_stagepack_reader.h): payload density classes, scale-plane
 * geometry, the scale-group rule, period arithmetic behind the inventory
 * counts, the layer-class resolution tail with its exact refusal codes, and
 * u32-prefix header comparison. This header keeps every historical name and
 * its exact signature so family loaders and module translation units keep
 * compiling unchanged; the six weight-class ids are bit-identical in both
 * homes (the nibble class was named mxfp4 here - same elements/2 density).
 * New code includes the reader directly.
 */

#include "spark_stagepack_reader.h"

#define SPARK_HYBRID_STAGEPACK_CLASS_GLOBAL SPARK_STAGE_PACK_LAYER_CLASS_GLOBAL
#define SPARK_HYBRID_STAGEPACK_CLASS_EVERY_LAYER SPARK_STAGE_PACK_LAYER_CLASS_EVERY_LAYER
#define SPARK_HYBRID_STAGEPACK_CLASS_GDN_LAYER SPARK_STAGE_PACK_LAYER_CLASS_GDN_LAYER
#define SPARK_HYBRID_STAGEPACK_CLASS_ATTN_LAYER SPARK_STAGE_PACK_LAYER_CLASS_ATTN_LAYER

#define SPARK_HYBRID_STAGEPACK_WEIGHT_BF16 SPARK_STAGE_PACK_WEIGHT_BF16
#define SPARK_HYBRID_STAGEPACK_WEIGHT_F32 SPARK_STAGE_PACK_WEIGHT_F32
#define SPARK_HYBRID_STAGEPACK_WEIGHT_U32 SPARK_STAGE_PACK_WEIGHT_U32
#define SPARK_HYBRID_STAGEPACK_WEIGHT_MXFP4_E2M1 SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1
#define SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_F32B128 SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128
#define SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_E8M0B128 SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_E8M0B128

#define SparkHybridStagePackFullAttentionLayersBelow SparkStagePackFullAttentionLayersBelow
#define SparkHybridStagePackPayloadBytes SparkStagePackWeightPayloadBytes
#define SparkHybridStagePackHeaderFieldsMatch SparkStagePackHeaderFieldsMatch
#define SparkHybridStagePackResolveLayerClass SparkStagePackResolveLayerClass
#define SparkHybridStagePackScaleBytes SparkStagePackWeightClassScaleBytes
#define SparkHybridStagePackScaleGroupSizeOk SparkStagePackWeightClassScaleGroupSizeOk
