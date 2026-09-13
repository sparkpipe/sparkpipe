#pragma once

// THE BATCH-VARIANT TUNING HEADER, dsv4 resident decode stage.
//
// Same contract as the glm52 variant header: one source tree, N compiled
// modules, -DSPARK_BATCH_BUCKET=<n> the ONLY difference between them, a
// bucket a capacity ceiling rather than a fixed batch. Read the why in
// modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_batch_tuning.h
// first; this file carries only what is dsv4's.
//
// THE SET IS EVERY POWER OF TWO FROM B1 TO B1024 PLUS THE THREE DSpark
// spec-step buckets (SPEC_STEP+1 for k=5/8/10 -> b6/b9/b11). One X-macro
// list below is the SINGLE source for: the #error whitelist, the fourteen
// variant module IDs, the runtime ceiling ladder, and the id lookup - so
// the selector, the builder and the validator can never drift again (the
// pre-fix header selected power-of-two ceilings only, making the built
// b6/b9/b11 archives unreachable through SparkDsv4BatchVariantBucketCeiling).
//
// The variant convention INSERTS .b<n>. ahead of the version suffix: a
// variant-published dsv4 module is a new module identity, and the
// unbucketed ID (SPARK_DSV4_MODEL_MODULE_ID, which the default archive
// carries) stays valid until the contract generator adopts the bucketed
// form - exactly the k3 header's stance.

#include <stdint.h>

// THE CANONICAL BUCKET LIST, ascending. Add a bucket here and every derived
// surface (whitelist, ids, ladder, lookup) picks it up in one edit.
#define SPARK_DSV4_BATCH_BUCKET_X \
	X(1) X(2) X(4) X(6) X(8) X(9) X(11) X(16) X(32) X(64) X(128) X(256) X(512) X(1024)

#if SPARK_BATCH_BUCKET != 1u && SPARK_BATCH_BUCKET != 2u && \
	SPARK_BATCH_BUCKET != 4u && SPARK_BATCH_BUCKET != 6u && \
	SPARK_BATCH_BUCKET != 8u && SPARK_BATCH_BUCKET != 9u && \
	SPARK_BATCH_BUCKET != 11u && SPARK_BATCH_BUCKET != 16u && \
	SPARK_BATCH_BUCKET != 32u && SPARK_BATCH_BUCKET != 64u && \
	SPARK_BATCH_BUCKET != 128u && SPARK_BATCH_BUCKET != 256u && \
	SPARK_BATCH_BUCKET != 512u && SPARK_BATCH_BUCKET != 1024u
#error SPARK_BATCH_BUCKET must name a built variant bucket: 1, 2, 4, 6, 8, 9, 11, 16, 32, 64, 128, 256, 512, 1024
#endif

// THE CANONICAL MODULE IDENTITY. Prefix and suffix written once; every
// variant ID is composed from them by token pasting, so a rename cannot
// drift them apart. Each variant publishes under its own ID and keeps
// SPEC.md's content-addressed artifact contract intact.
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_PREFIX \
	"spark.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16.h4096.l43.e256.k6.ga0731"
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_SUFFIX "v4"
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID(bucket) \
	SPARK_DSV4_BATCH_VARIANT_MODULE_ID_PREFIX ".b" #bucket "." \
	SPARK_DSV4_BATCH_VARIANT_MODULE_ID_SUFFIX

#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B1 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(1)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B2 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(2)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B4 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(4)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B6 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(6)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B8 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(8)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B9 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(9)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B11 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(11)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B16 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(16)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B32 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(32)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B64 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(64)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B128 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(128)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B256 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(256)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B512 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(512)
#define SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B1024 SPARK_DSV4_BATCH_VARIANT_MODULE_ID(1024)

#if SPARK_BATCH_BUCKET == 1u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B1
#elif SPARK_BATCH_BUCKET == 2u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B2
#elif SPARK_BATCH_BUCKET == 4u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B4
#elif SPARK_BATCH_BUCKET == 6u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B6
#elif SPARK_BATCH_BUCKET == 8u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B8
#elif SPARK_BATCH_BUCKET == 9u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B9
#elif SPARK_BATCH_BUCKET == 11u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B11
#elif SPARK_BATCH_BUCKET == 16u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B16
#elif SPARK_BATCH_BUCKET == 32u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B32
#elif SPARK_BATCH_BUCKET == 64u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B64
#elif SPARK_BATCH_BUCKET == 128u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B128
#elif SPARK_BATCH_BUCKET == 256u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B256
#elif SPARK_BATCH_BUCKET == 512u
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B512
#else
#define SPARK_DSV4_BATCH_TUNING_MODULE_ID SPARK_DSV4_BATCH_VARIANT_MODULE_ID_B1024
#endif

// THE ACTIVE-SEQUENCE CEILING THE FIRMWARE SIZES ITS COMPUTE WORKSPACE BY.
// Resident sequence ownership is independent of this batch bucket, and the
// paged KV pool is independently memory-budgeted. Every compiled bucket now
// executes its advertised width; a b1024 module is not a renamed b128.
#define SPARK_DSV4_BATCH_TUNING_SEQUENCE_CEILING \
	SPARK_BATCH_BUCKET

// THE GROUPED TILE HEIGHT AT THE BUCKET CEILING. Derived, not tabulated, the
// glm52 rule applied to dsv4's geometry: the mean group holds
// bucket*experts_per_token/routed_experts rows, the busiest group is priced
// at twice the mean, and the tile rounds UP through 16/32/64. For dsv4
// (top-6 of 256) the ceilings land at 16 for b1 through b256, 32 at b512,
// and 64 at b1024 - the same monotonic shape as glm52.
#define SPARK_DSV4_BATCH_TUNING_GROUPED_PEAK_ROWS \
	((((SPARK_BATCH_BUCKET) * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN + \
	SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT - 1u) / \
	SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT) * 2u)
#define SPARK_DSV4_BATCH_TUNING_GROUPED_TILE_M \
	(SPARK_DSV4_BATCH_TUNING_GROUPED_PEAK_ROWS <= 16u ? 16u : \
	SPARK_DSV4_BATCH_TUNING_GROUPED_PEAK_ROWS <= 32u ? 32u : 64u)

// RUNTIME VARIANT SELECTION: smallest built bucket >= the requested maximum
// active-sequence count, 0 above b1024. Generated from the same X-list as
// everything else, ascending order included, so the DSpark spec-step
// buckets b6/b9/b11 are selectable like any other rung.
static inline uint32_t SparkDsv4BatchVariantBucketCeiling(
	uint32_t max_active_sequence_count)
{
	if (max_active_sequence_count == 0u ||
		max_active_sequence_count > 1024u)
		return(0u);
#define X(bucket) if (max_active_sequence_count <= bucket##u) return(bucket##u);
	SPARK_DSV4_BATCH_BUCKET_X
#undef X
	return(1024u);
}

static inline const char *SparkDsv4BatchVariantModuleId(
	uint32_t batch_bucket)
{
	switch (batch_bucket)
	{
#define X(bucket) case bucket##u: return(SPARK_DSV4_BATCH_VARIANT_MODULE_ID(bucket));
	SPARK_DSV4_BATCH_BUCKET_X
#undef X
	default:
		return(0);
	}
}
