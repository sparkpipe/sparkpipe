#ifndef SPARKPIPE_SPARK_K3_BATCH_TUNING_H
#define SPARKPIPE_SPARK_K3_BATCH_TUNING_H

// THE BATCH-VARIANT TUNING HEADER, k3 resident decode stage.
//
// Same contract as the glm52 variant header: one source tree, N compiled
// modules, -DSPARK_BATCH_BUCKET=<n> the ONLY difference between them, a
// bucket a capacity ceiling rather than a fixed batch. Read the why in
// modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_batch_tuning.h
// first; this file carries only what is k3's.
//
// k3's module ID has no batch field today, so the variant convention INSERTS
// .b<n>. ahead of the version suffix. That makes a variant-published k3
// module a new module identity, not a rename of the old one: the unbucketed
// ID stays valid for the unbucketed archive until the k3 module Makefile and
// the contract generator adopt the bucketed form. Those two are the open
// dependency - this header is the interface they adopt against.
//
// THE SET IS EVERY POWER OF TWO FROM B1 TO B1024, the same eleven-bucket
// ladder as glm52: runtime selection takes the tightest ceiling at or above
// the microbatch, so a live batch pads to at most twice itself, and B1024
// stays the planner maximum and the unflagged build.

#include <stdint.h>

#include "inference/llms/kimi_k3/config.h"

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

// THE CANONICAL MODULE IDENTITY. Prefix and suffix written once; the eleven
// variant IDs are compositions, so a rename cannot drift them apart. Each
// variant publishes under its own ID and keeps SPEC.md's content-addressed
// artifact contract intact.
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX \
	"spark.k3.resident_decode_stage.mxfp4_routed_bf16_rest.h7168.l93.kda69.mla24"
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX \
	"v2"
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B1 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b1." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B2 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b2." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B4 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b4." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B8 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b8." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B16 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b16." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B32 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b32." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B64 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b64." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B128 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b128." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B256 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b256." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B512 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b512." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_K3_BATCH_VARIANT_MODULE_ID_B1024 \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_PREFIX ".b1024." \
	SPARK_K3_BATCH_VARIANT_MODULE_ID_SUFFIX

#if SPARK_BATCH_BUCKET == 1u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B1
#elif SPARK_BATCH_BUCKET == 2u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B2
#elif SPARK_BATCH_BUCKET == 4u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B4
#elif SPARK_BATCH_BUCKET == 8u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B8
#elif SPARK_BATCH_BUCKET == 16u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B16
#elif SPARK_BATCH_BUCKET == 32u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B32
#elif SPARK_BATCH_BUCKET == 64u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B64
#elif SPARK_BATCH_BUCKET == 128u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B128
#elif SPARK_BATCH_BUCKET == 256u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B256
#elif SPARK_BATCH_BUCKET == 512u
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B512
#else
#define SPARK_K3_BATCH_TUNING_MODULE_ID SPARK_K3_BATCH_VARIANT_MODULE_ID_B1024
#endif

// THE GROUPED TILE HEIGHT AT THE BUCKET CEILING, LmLaunchGroupedTileM
// (runtime/launch.h) with tokens fixed at the bucket. Derived, not tabulated.
// For k3 (top-16 of 896) the ceilings land at 16 for b1 through b256, 32 at
// b512, and 64 at b1024 - the same monotonic shape as glm52.
#define SPARK_K3_BATCH_TUNING_GROUPED_PEAK_ROWS \
	((((SPARK_BATCH_BUCKET) * K3_TOP_K + K3_EXPERTS - 1u) / \
	K3_EXPERTS) * 2u)
#define SPARK_K3_BATCH_TUNING_GROUPED_TILE_M \
	(SPARK_K3_BATCH_TUNING_GROUPED_PEAK_ROWS <= 16u ? 16u : \
	SPARK_K3_BATCH_TUNING_GROUPED_PEAK_ROWS <= 32u ? 32u : 64u)

// RUNTIME VARIANT SELECTION: smallest built bucket >= the requested maximum
// active-sequence count, 0 above b1024. Same contract as glm52's; duplicated
// per family because the module IDs it selects between are model content.
static inline uint32_t SparkK3BatchVariantBucketCeiling(
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

static inline const char *SparkK3BatchVariantModuleId(
	uint32_t batch_bucket)
{
	switch (batch_bucket)
	{
	case 1u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B1);
	case 2u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B2);
	case 4u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B4);
	case 8u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B8);
	case 16u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B16);
	case 32u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B32);
	case 64u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B64);
	case 128u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B128);
	case 256u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B256);
	case 512u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B512);
	case 1024u:
		return(SPARK_K3_BATCH_VARIANT_MODULE_ID_B1024);
	default:
		return(0);
	}
}

#endif
