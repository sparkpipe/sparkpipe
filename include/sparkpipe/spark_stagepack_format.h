#ifndef SPARKPIPE_SPARK_STAGEPACK_FORMAT_H
#define SPARKPIPE_SPARK_STAGEPACK_FORMAT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stagepack format library: the ONE implementation of the shape algebra and
 * the header comparison that the family pack formats share. The v2-style
 * pack container (magic, format version, byte-exact header, byte-exact
 * directory entries, and the per-kind shard axis a TP rank narrows along)
 * is stated once here; a family declares its geometry as data and keeps
 * only the tensors that are genuinely its own.
 *
 * The shared axis is the per-layer tensor inventory the sibling families
 * carry with identical shape formulas - the norms, the routed/shared MoE
 * projections (the routed ones narrow along the expert-shard axis: rows =
 * shard_count x per-shard rows), and the GDN set. Kind numbers are the
 * values the v1 family inventories already agree on; the neutral
 * weight-format and layer-class codes are the firmware ABI values, pinned
 * per family with _Static_assert so a family cannot drift silently.
 */

#define SPARK_STAGEPACK_FORMAT_WEIGHT_BF16 0u
#define SPARK_STAGEPACK_FORMAT_WEIGHT_F32 1u
#define SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128 4u
#define SPARK_STAGEPACK_FORMAT_WEIGHT_I64 7u

#define SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL 0u
#define SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER 1u
#define SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GDN_LAYER 2u
#define SPARK_STAGEPACK_FORMAT_LAYER_CLASS_ATTN_LAYER 3u

typedef enum SparkStagePackCommonTensorKind
{
	SPARK_STAGEPACK_TENSOR_ATTENTION_NORM = 3,
	SPARK_STAGEPACK_TENSOR_MLP_NORM = 4,
	SPARK_STAGEPACK_TENSOR_MOE_GATE = 5,
	SPARK_STAGEPACK_TENSOR_MOE_W1 = 6,
	SPARK_STAGEPACK_TENSOR_MOE_W3 = 7,
	SPARK_STAGEPACK_TENSOR_MOE_DOWN = 8,
	SPARK_STAGEPACK_TENSOR_MOE_SHARED_GATE = 9,
	SPARK_STAGEPACK_TENSOR_MOE_SHARED_UP = 10,
	SPARK_STAGEPACK_TENSOR_MOE_SHARED_DOWN = 11,
	SPARK_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT = 12,
	SPARK_STAGEPACK_TENSOR_GDN_QKV = 13,
	SPARK_STAGEPACK_TENSOR_GDN_GATE = 14,
	SPARK_STAGEPACK_TENSOR_GDN_BETA = 15,
	SPARK_STAGEPACK_TENSOR_GDN_DECAY = 16,
	SPARK_STAGEPACK_TENSOR_GDN_OUTPUT = 17,
	SPARK_STAGEPACK_TENSOR_GDN_CONV_WEIGHT = 18,
	SPARK_STAGEPACK_TENSOR_GDN_A_LOG = 19,
	SPARK_STAGEPACK_TENSOR_GDN_DT_BIAS = 20,
	SPARK_STAGEPACK_TENSOR_GDN_NORM = 21
} SparkStagePackCommonTensorKind;

typedef struct SparkStagePackTensorShape
{
	uint32_t rows;
	uint32_t columns;
	uint32_t natural_format;
	uint32_t layer_class;
} SparkStagePackTensorShape;

/*
 * The family's geometry, as data. norm_width is the per-layer norm width
 * (families differ: hidden width vs hyper-connection stream width).
 */
typedef struct SparkStagePackGeometryTable
{
	uint32_t norm_width;
	uint32_t hidden_dimension;
	uint32_t routed_expert_count;
	uint32_t expert_intermediate_dimension;
	uint32_t gdn_conv_channels;
	uint32_t gdn_value_dimension;
	uint32_t gdn_value_head_count;
	uint32_t gdn_head_value_dimension;
	uint32_t gdn_conv_kernel;
} SparkStagePackGeometryTable;

void SparkStagePackShapeInit(SparkStagePackTensorShape *shape);

/* The shared EVERY_LAYER axis; -1 for kinds outside it. */
int32_t SparkStagePackShapeEveryLayerCommon(uint32_t tensor_kind,
	const SparkStagePackGeometryTable *geometry,
	SparkStagePackTensorShape *shape);

/* The shared GDN_LAYER axis; -1 for kinds outside it. */
int32_t SparkStagePackShapeGdnCommon(uint32_t tensor_kind,
	const SparkStagePackGeometryTable *geometry,
	SparkStagePackTensorShape *shape);

/*
 * The byte-exact v2-style header, stated once. Family header structs must
 * prove layout identity with SPARK_STAGEPACK_HEADER_LAYOUT_PROOF (a drift
 * is a compile error, not a silent format change); the comparison then
 * runs once, here, for every family.
 */
typedef struct SparkStagePackHeaderCommon
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	uint32_t directory_entry_bytes;
	uint32_t tensor_count;
	uint32_t hidden_dimension;
	uint32_t layer_count;
	uint32_t first_layer_index;
	uint32_t total_layer_count;
	uint32_t attention_period;
	uint32_t full_attention_phase;
	uint32_t gdn_key_head_count;
	uint32_t gdn_value_head_count;
	uint32_t gdn_head_key_dimension;
	uint32_t gdn_head_value_dimension;
	uint32_t gdn_conv_kernel;
	uint32_t attn_query_head_count;
	uint32_t attn_kv_head_count;
	uint32_t attn_head_dimension;
	uint32_t attn_rope_dimension;
	uint32_t routed_expert_count;
	uint32_t experts_per_token;
	uint32_t expert_intermediate_dimension;
	uint32_t output_vocab_count;
	uint32_t mxfp4_group_size;
	uint32_t mtp_layer_count;
	uint64_t directory_offset;
	uint64_t file_bytes;
} SparkStagePackHeaderCommon;

#define SPARK_STAGEPACK_HEADER_BYTES 120u
#define SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, field) \
	_Static_assert(offsetof(family_type,field) == \
		offsetof(SparkStagePackHeaderCommon,field), \
		"stage pack header layout drift: " #family_type "." #field);

/* A complete declaration (no trailing semicolon needed): an anonymous-proof
 * struct whose members are static assertions, so a family header admits
 * itself to the shared comparison by compiling this once. */
#define SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(family_type) \
	struct SparkStagePackHeaderLayoutProof##family_type { \
		_Static_assert(sizeof(family_type) == SPARK_STAGEPACK_HEADER_BYTES, \
			"stage pack header must be 120 wire bytes"); \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, magic) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, format_version) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, header_bytes) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, directory_entry_bytes) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, tensor_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, hidden_dimension) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, layer_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, first_layer_index) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, total_layer_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, attention_period) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, full_attention_phase) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, gdn_key_head_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, gdn_value_head_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, gdn_head_key_dimension) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, gdn_head_value_dimension) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, gdn_conv_kernel) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, attn_query_head_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, attn_kv_head_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, attn_head_dimension) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, attn_rope_dimension) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, routed_expert_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, experts_per_token) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, expert_intermediate_dimension) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, output_vocab_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, mxfp4_group_size) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, mtp_layer_count) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, directory_offset) \
		SPARK_STAGEPACK_HEADER_FIELD_PROOF(family_type, file_bytes) \
	}

/* Field-by-field comparison; 0 on match, negative group id on drift. */
int32_t SparkStagePackHeaderMatches(const SparkStagePackHeaderCommon *file_header,
	const SparkStagePackHeaderCommon *expected);

#ifdef __cplusplus
}
#endif

#endif
