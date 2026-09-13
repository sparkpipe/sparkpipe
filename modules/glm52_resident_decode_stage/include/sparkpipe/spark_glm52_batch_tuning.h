#ifndef SPARKPIPE_SPARK_GLM52_BATCH_TUNING_H
#define SPARKPIPE_SPARK_GLM52_BATCH_TUNING_H

// THE BATCH-VARIANT TUNING HEADER, glm52 resident decode stage.
//
// One source tree, N compiled modules. make variants emits
// libglm52_resident_decode_stage_<codec>_b1.a / _b2.a / ... / _b1024.a - one
// archive per power of two - from the same translation units with
// -DSPARK_BATCH_BUCKET=<n>, and this header is the ONLY thing that differs
// between them. A per-bucket fork of the layer behind an #if would be two
// sources wearing one name - exactly what the variant system exists to
// prevent, so the tuning constants live here or nowhere.
//
// A bucket is a CAPACITY CEILING plus tuned geometry, not a fixed batch: the
// b8 module serves 1-8 rows, b64 serves 9-64, and a runtime batch below the
// ceiling needs no recompile. What a smaller ceiling buys is compile-time
// truth the optimizer can act on - the grouped-GEMM tile height below, and
// the pools the consumers of this header scale by the bucket (the firmware
// header's active-sequence ceiling sizes the module's lane tables from it).
//
// THE SET IS EVERY POWER OF TWO FROM B1 TO B1024, eleven buckets. Runtime
// selection picks the TIGHTEST ceiling at or above the microbatch, so a live
// batch pads to at most twice itself and every pool the ceiling sizes stays
// within that factor of the truth - the memory the tight fit saves pays for
// the eleven compile units and the premade graph per size. B1024 is the
// maximum the stage firmware knows
// (SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT unflagged),
// the unflagged build, and the last variant a footprint cut ever drops.
//
// The per-slot CUDA graph cache that once paired a premade graph with each
// bucket (eleven buckets times the MTP draft variants, minus one spare
// entry, in the deleted include/sparkpipe/spark_resident_decode_stage.h)
// went away with the inference/stage rewrite - no graph replay exists in the
// rewritten modules yet. When graph replay lands, its cache must be sized
// against this eleven-bucket ladder again or a warm bucket graph is evicted
// the step before it would replay.

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

// THE CANONICAL MODULE IDENTITY. The batch bucket is the only module-ID field
// that varies per variant, and the prefix and suffix are written once so a
// rename cannot drift the eleven IDs apart. The prefix composes the module
// Makefile's -DGLM52_EXPERT_CODEC_NAME, so the six codec builds share this
// one spelling. Each variant publishes under its own ID, which is what keeps
// SPEC.md's content-addressed artifact contract untouched: eleven
// identities, eleven immutable records, each validated once, resolved by the
// same identity-key mechanism as any other module.
//
// The variant convention INSERTS .b<n>. ahead of the version suffix. That
// makes a variant-published glm52 module a new module identity, not a rename
// of the old one: the unbucketed ID stays valid for the unflagged archive -
// which IS the b1024 build - until the contract generator adopts the
// bucketed form, exactly the k3 header's stance.
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX \
	"spark.glm52.resident_decode_stage.bf16.expert_" GLM52_EXPERT_CODEC_NAME \
	".h6144.l78.e256.k8"
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX \
	"v2"
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b1." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B2 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b2." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B4 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b4." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B8 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b8." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B16 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b16." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B32 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b32." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B64 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b64." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B128 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b128." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B256 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b256." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B512 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b512." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1024 \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_PREFIX ".b1024." \
	SPARK_GLM52_BATCH_VARIANT_MODULE_ID_SUFFIX

#if SPARK_BATCH_BUCKET == 1u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1
#elif SPARK_BATCH_BUCKET == 2u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B2
#elif SPARK_BATCH_BUCKET == 4u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B4
#elif SPARK_BATCH_BUCKET == 8u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B8
#elif SPARK_BATCH_BUCKET == 16u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B16
#elif SPARK_BATCH_BUCKET == 32u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B32
#elif SPARK_BATCH_BUCKET == 64u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B64
#elif SPARK_BATCH_BUCKET == 128u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B128
#elif SPARK_BATCH_BUCKET == 256u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B256
#elif SPARK_BATCH_BUCKET == 512u
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B512
#else
#define SPARK_GLM52_BATCH_TUNING_MODULE_ID SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1024
#endif

// THE GROUPED TILE HEIGHT AT THE BUCKET CEILING. The mean group holds
// bucket*top_k/experts rows, the busiest group is priced at twice the mean,
// and the tile rounds UP through 16/32/64 - a tile shorter than the group
// splits it and doubles the weight stream, which is 96 percent of decode
// traffic, while padded mma rows are free. Derived rather than tabulated, so
// a top_k or expert-count change reprices every variant at once. For glm52
// (top-8 of 256) the ceilings land at 16 for b1 through b256, 32 at b512,
// and 64 at b1024: the peak-row count doubles with the bucket, so the tile
// height climbs the ladder monotonically and never steps down.
#define SPARK_GLM52_BATCH_TUNING_GROUPED_PEAK_ROWS \
	((((SPARK_BATCH_BUCKET) * SPARK_GLM52_MODEL_MOE_TOP_K + \
	SPARK_GLM52_MODEL_MOE_EXPERT_COUNT - 1u) / \
	SPARK_GLM52_MODEL_MOE_EXPERT_COUNT) * 2u)
#define SPARK_GLM52_BATCH_TUNING_GROUPED_TILE_M \
	(SPARK_GLM52_BATCH_TUNING_GROUPED_PEAK_ROWS <= 16u ? 16u : \
	SPARK_GLM52_BATCH_TUNING_GROUPED_PEAK_ROWS <= 32u ? 32u : 64u)

// RUNTIME VARIANT SELECTION. The stage loader resolves the smallest built
// bucket >= the requested maximum active-sequence count - the ceiling rule
// the bucket semantics above promise. A request above b1024 gets 0, which
// the caller must treat as no-variant: there is nothing larger to fall back
// to, and silently serving it under b1024 would oversubscribe the pools the
// ceiling sizes.
static inline uint32_t SparkGlm52BatchVariantBucketCeiling(
	uint32_t max_active_sequence_count)
{
	if (max_active_sequence_count == 0u ||
		max_active_sequence_count > 1024u)
		return(0u);
	if (max_active_sequence_count <= 1u)
		return(1u);
	if (max_active_sequence_count <= 2u)
		return(2u);
	if (max_active_sequence_count <= 4u)
		return(4u);
	if (max_active_sequence_count <= 8u)
		return(8u);
	if (max_active_sequence_count <= 16u)
		return(16u);
	if (max_active_sequence_count <= 32u)
		return(32u);
	if (max_active_sequence_count <= 64u)
		return(64u);
	if (max_active_sequence_count <= 128u)
		return(128u);
	if (max_active_sequence_count <= 256u)
		return(256u);
	if (max_active_sequence_count <= 512u)
		return(512u);
	return(1024u);
}

static inline const char *SparkGlm52BatchVariantModuleId(
	uint32_t batch_bucket)
{
	switch (batch_bucket)
	{
	case 1u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1);
	case 2u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B2);
	case 4u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B4);
	case 8u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B8);
	case 16u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B16);
	case 32u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B32);
	case 64u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B64);
	case 128u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B128);
	case 256u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B256);
	case 512u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B512);
	case 1024u:
		return(SPARK_GLM52_BATCH_VARIANT_MODULE_ID_B1024);
	default:
		return(0);
	}
}

#endif
