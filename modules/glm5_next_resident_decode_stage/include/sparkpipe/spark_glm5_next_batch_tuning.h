#ifndef SPARKPIPE_SPARK_GLM5_NEXT_BATCH_TUNING_H
#define SPARKPIPE_SPARK_GLM5_NEXT_BATCH_TUNING_H

// THE BATCH-VARIANT TUNING HEADER, glm5_next resident decode stage.
//
// Same contract as the glm52 variant header: one source tree, N compiled
// modules, -DSPARK_BATCH_BUCKET=<n> the ONLY difference between them, a
// bucket a capacity ceiling rather than a fixed batch, the eleven-bucket
// B1..B1024 ladder, and the shared ladder core
// (include/sparkpipe/spark_batch_variant_tuning_common.h) supplying the
// machinery. This file carries only what is glm5_next's.

#include <stdint.h>

#include "sparkpipe/spark_glm5_next_model.h"

#ifndef SPARK_BATCH_BUCKET
// The unflagged build IS the b1024 module.
#define SPARK_BATCH_BUCKET 1024u
#endif

#if SPARK_BATCH_BUCKET != 1u && SPARK_BATCH_BUCKET != 2u && \
	SPARK_BATCH_BUCKET != 4u && SPARK_BATCH_BUCKET != 8u && \
	SPARK_BATCH_BUCKET != 16u && SPARK_BATCH_BUCKET != 32u && \
	SPARK_BATCH_BUCKET != 64u && SPARK_BATCH_BUCKET != 128u && \
	SPARK_BATCH_BUCKET != 256u && SPARK_BATCH_BUCKET != 512u && \
	SPARK_BATCH_BUCKET != 1024u
#error SPARK_BATCH_BUCKET must name a built variant bucket: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
#endif

#ifndef GLM5_NEXT_EXPERT_CODEC_NAME
// The module identifier names the expert codec, and the module Makefile is
// the only allowed source of that name; a host probe must pass
// -DGLM5_NEXT_EXPERT_CODEC_NAME explicitly rather than guess one.
#error GLM5_NEXT_EXPERT_CODEC_NAME is required: the variant module identifier names the expert codec
#endif

// THE CANONICAL MODULE IDENTITY (glm5_next's own content), same insertion
// convention as glm52: .b<n>. ahead of the version suffix.
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_PREFIX \
	"spark.glm5_next.resident_decode_stage.bf16.expert_" GLM5_NEXT_EXPERT_CODEC_NAME \
	".h6144.l78.e256.k8"
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_SUFFIX \
	"v2"

// Family configuration for the shared ladder, then the legacy spellings.
#define SPARK_BATCH_VARIANT_MODULE_ID_PREFIX \
	SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_PREFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX \
	SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_TOP_K SPARK_GLM5_NEXT_MODEL_MOE_TOP_K
#define SPARK_BATCH_VARIANT_EXPERT_COUNT SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT
#define SPARK_BATCH_VARIANT_FN(name) SparkGlm5NextBatchVariant##name

#include "sparkpipe/spark_batch_variant_tuning_common.h"

#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B1 SPARK_BATCH_VARIANT_MODULE_ID_B1
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B2 SPARK_BATCH_VARIANT_MODULE_ID_B2
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B4 SPARK_BATCH_VARIANT_MODULE_ID_B4
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B8 SPARK_BATCH_VARIANT_MODULE_ID_B8
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B16 SPARK_BATCH_VARIANT_MODULE_ID_B16
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B32 SPARK_BATCH_VARIANT_MODULE_ID_B32
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B64 SPARK_BATCH_VARIANT_MODULE_ID_B64
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B128 SPARK_BATCH_VARIANT_MODULE_ID_B128
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B256 SPARK_BATCH_VARIANT_MODULE_ID_B256
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B512 SPARK_BATCH_VARIANT_MODULE_ID_B512
#define SPARK_GLM5_NEXT_BATCH_VARIANT_MODULE_ID_B1024 SPARK_BATCH_VARIANT_MODULE_ID_B1024
#define SPARK_GLM5_NEXT_BATCH_TUNING_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID
#define SPARK_GLM5_NEXT_BATCH_TUNING_GROUPED_PEAK_ROWS SPARK_BATCH_VARIANT_GROUPED_PEAK_ROWS
#define SPARK_GLM5_NEXT_BATCH_TUNING_GROUPED_TILE_M SPARK_BATCH_VARIANT_GROUPED_TILE_M

#endif
