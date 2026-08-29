/* Shared batch-variant tuning ladder (DRY wave 1).
 *
 * The per-family batch_tuning headers were one ladder pasted per family:
 * the eleven-bucket identity compositions, the compiled-bucket module-ID
 * selection, the derived grouped-tile geometry, and the runtime ceiling /
 * module-ID selection functions. This header is that ladder, parameterized
 * by family configuration macros the includer defines first. The family
 * header keeps its own identity (guard, bucket guard, ID prefix/suffix,
 * legacy macro spellings) — model content stays model-side; only the
 * machinery is shared.
 *
 * Required configuration (define before including):
 *   SPARK_BATCH_BUCKET              the compiled bucket (family owns the
 *                                   b1024 default and the #error guard)
 *   SPARK_BATCH_VARIANT_MODULE_ID_PREFIX / _SUFFIX   the family identity
 *   SPARK_BATCH_VARIANT_TOP_K / _EXPERT_COUNT        the grouped-tile inputs
 *   SPARK_BATCH_VARIANT_FN(name)    token-paste to the family function name
 *
 * The includer then aliases its legacy macro spellings onto the neutral
 * names defined here (see the family batch_tuning headers).
 */
#ifndef SPARKPIPE_SPARK_BATCH_VARIANT_TUNING_COMMON_H
#define SPARKPIPE_SPARK_BATCH_VARIANT_TUNING_COMMON_H

#include <stdint.h>

/* THE VARIANT ID COMPOSITIONS. Prefix and suffix are the family's, written
 * once there; the eleven bucket IDs are compositions, so a rename cannot
 * drift them apart. */
#define SPARK_BATCH_VARIANT_MODULE_ID_B1 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b1." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B2 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b2." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B4 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b4." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B8 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b8." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B16 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b16." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B32 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b32." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B64 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b64." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B128 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b128." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B256 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b256." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B512 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b512." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX
#define SPARK_BATCH_VARIANT_MODULE_ID_B1024 \
	SPARK_BATCH_VARIANT_MODULE_ID_PREFIX ".b1024." \
	SPARK_BATCH_VARIANT_MODULE_ID_SUFFIX

/* THE COMPILED BUCKET'S IDENTITY. */
#if SPARK_BATCH_BUCKET == 1u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B1
#elif SPARK_BATCH_BUCKET == 2u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B2
#elif SPARK_BATCH_BUCKET == 4u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B4
#elif SPARK_BATCH_BUCKET == 8u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B8
#elif SPARK_BATCH_BUCKET == 16u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B16
#elif SPARK_BATCH_BUCKET == 32u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B32
#elif SPARK_BATCH_BUCKET == 64u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B64
#elif SPARK_BATCH_BUCKET == 128u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B128
#elif SPARK_BATCH_BUCKET == 256u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B256
#elif SPARK_BATCH_BUCKET == 512u
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B512
#else
#define SPARK_BATCH_VARIANT_MODULE_ID SPARK_BATCH_VARIANT_MODULE_ID_B1024
#endif

/* THE GROUPED TILE HEIGHT AT THE BUCKET CEILING. The mean group holds
 * bucket*top_k/experts rows, the busiest group is priced at twice the mean,
 * and the tile rounds UP through 16/32/64 - a tile shorter than the group
 * splits it and doubles the weight stream, which is 96 percent of decode
 * traffic, while padded mma rows are free. Derived rather than tabulated, so
 * a top_k or expert-count change reprices every variant at once; the
 * peak-row count doubles with the bucket, so the tile height climbs the
 * ladder monotonically and never steps down. */
#define SPARK_BATCH_VARIANT_GROUPED_PEAK_ROWS \
	((((SPARK_BATCH_BUCKET) * SPARK_BATCH_VARIANT_TOP_K + \
	SPARK_BATCH_VARIANT_EXPERT_COUNT - 1u) / \
	SPARK_BATCH_VARIANT_EXPERT_COUNT) * 2u)
#define SPARK_BATCH_VARIANT_GROUPED_TILE_M \
	(SPARK_BATCH_VARIANT_GROUPED_PEAK_ROWS <= 16u ? 16u : \
	SPARK_BATCH_VARIANT_GROUPED_PEAK_ROWS <= 32u ? 32u : 64u)

/* RUNTIME VARIANT SELECTION. The stage loader resolves the smallest built
 * bucket >= the requested maximum active-sequence count - the ceiling rule
 * the bucket semantics promise. A request above b1024 gets 0, which the
 * caller must treat as no-variant: there is nothing larger to fall back to,
 * and silently serving it under b1024 would oversubscribe the pools the
 * ceiling sizes. */
static inline uint32_t SPARK_BATCH_VARIANT_FN(BucketCeiling)(
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

static inline const char *SPARK_BATCH_VARIANT_FN(ModuleId)(
	uint32_t batch_bucket)
{
	switch (batch_bucket)
	{
	case 1u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B1);
	case 2u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B2);
	case 4u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B4);
	case 8u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B8);
	case 16u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B16);
	case 32u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B32);
	case 64u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B64);
	case 128u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B128);
	case 256u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B256);
	case 512u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B512);
	case 1024u:
		return(SPARK_BATCH_VARIANT_MODULE_ID_B1024);
	default:
		return(0);
	}
}

#endif
