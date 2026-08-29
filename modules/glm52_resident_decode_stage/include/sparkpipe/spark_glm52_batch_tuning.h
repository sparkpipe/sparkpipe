#ifndef SPARKPIPE_SPARK_GLM52_BATCH_TUNING_H
#define SPARKPIPE_SPARK_GLM52_BATCH_TUNING_H

// THE BATCH-VARIANT TUNING HEADER, glm52 resident decode stage.
//
// One source tree, N compiled modules. make variants emits
// libglm52_resident_decode_stage_<codec>_b1.a / _b2.a / ... / _b1024.a - one
// archive per power of two - from the same translation units with
// -DSPARK_BATCH_BUCKET=<n>, and this header is the ONLY thing that differs
// between them. A bucket is a CAPACITY CEILING plus tuned geometry, not a
// fixed batch; runtime selection picks the TIGHTEST ceiling at or above the
// microbatch; B1024 is the maximum the stage firmware knows
// (SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT unflagged),
// the unflagged build, and the last variant a footprint cut ever drops.
// The full contract and its why live in the shared ladder core
// (include/sparkpipe/spark_batch_variant_tuning_common.h) and in
// tests/test_batch_variants.py, which enforces it per bucket.
//
// The per-slot CUDA graph cache that once paired a premade graph with each
// bucket went away with the inference/stage rewrite - no graph replay exists
// in the rewritten modules yet. When graph replay lands, its cache must be
// sized against this eleven-bucket ladder again or a warm bucket graph is
// evicted the step before it would replay.

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"

#ifndef SPARK_BATCH_BUCKET
// The unflagged build IS the b1024 module: one spelling for the default, so
// every existing consumer of the firmware header sees no change.
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

#ifndef GLM52_EXPERT_CODEC_NAME
// The module identifier names the expert codec, and the module Makefile is
// the only allowed source of that name; a host probe must pass
// -DGLM52_EXPERT_CODEC_NAME explicitly rather than guess one.
#error GLM52_EXPERT_CODEC_NAME is required: the variant module identifier names the expert codec
#endif

// THE CANONICAL MODULE IDENTITY (glm52's own content). The prefix composes
// the module Makefile's -DGLM52_EXPERT_CODEC_NAME, so the six codec builds
// share this one spelling; the variant convention INSERTS .b<n>. ahead of
// the version suffix, so a variant-published glm52 module is a new module
// identity, not a rename of the old one.
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX \
	"spark.glm52.resident_decode_stage.bf16.expert_" GLM52_EXPERT_CODEC_NAME \
	".h6144.l78.e256.k8"
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX \
	"v2"

// Family configuration for the shared ladder (spark_batch_variant_tuning_
// common.h), then the legacy spellings every existing consumer keeps.
#define SPARK_BATCH_VARIANT_MODULE_ID_PREFIX \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_TOP_K SPARK_GLM52_MODEL_MOE_TOP_K
#define SPARK_BATCH_VARIANT_EXPERT_COUNT SPARK_GLM52_MODEL_MOE_EXPERT_COUNT
#define SPARK_BATCH_VARIANT_FN(name) SparkGlm52BatchVariant##name

#include "sparkpipe/spark_batch_variant_tuning_common.h"

#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1 SPARK_BATCH_VARIANT_MODULE_ID_B1
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B2 SPARK_BATCH_VARIANT_MODULE_ID_B2
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B4 SPARK_BATCH_VARIANT_MODULE_ID_B4
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B8 SPARK_BATCH_VARIANT_MODULE_ID_B8
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B16 SPARK_BATCH_VARIANT_MODULE_ID_B16
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B32 SPARK_BATCH_VARIANT_MODULE_ID_B32
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B64 SPARK_BATCH_VARIANT_MODULE_ID_B64
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B128 SPARK_BATCH_VARIANT_MODULE_ID_B128
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B256 SPARK_BATCH_VARIANT_MODULE_ID_B256
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B512 SPARK_BATCH_VARIANT_MODULE_ID_B512
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1024 SPARK_BATCH_VARIANT_MODULE_ID_B1024
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID
#define SPARK_GLM52_BATCH_TUNING_GROUPED_PEAK_ROWS SPARK_BATCH_VARIANT_GROUPED_PEAK_ROWS
#define SPARK_GLM52_BATCH_TUNING_GROUPED_TILE_M SPARK_BATCH_VARIANT_GROUPED_TILE_M

#endif
