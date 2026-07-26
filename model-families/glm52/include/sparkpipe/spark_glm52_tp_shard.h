#ifndef SPARKPIPE_SPARK_GLM52_TP_SHARD_H
#define SPARKPIPE_SPARK_GLM52_TP_SHARD_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_stagepack.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_glm52_model.h"

// Tensor-parallel shard geometry: the single source of truth for how each
// GLM-5.2 tensor splits across a TP group. The stagepack loader consumes the
// view to read only its rank's slice, and the kernel binding contract consumes
// the geometry hash so a shard built for a different shape fails closed rather
// than binding. Classification is by the packed tensor-name suffixes the node
// context builder actually resolves; an unrecognized name classifies as
// unknown and the caller must refuse it, never guess.
//
// Sharding follows the standard Megatron-style split for a weight stored as
// [output_features, input_features]: output-dimension shards need no collective
// (each rank produces its slice of the activation), input-dimension shards
// produce partial sums that the layer's closing all-reduce combines, and
// replicated tensors are loaded whole on every rank. The MLA latent paths
// (q_a, kv_a with mqa, their norms) are head-agnostic and replicate, which is
// what keeps the latent KV cache identical on every rank and makes TP
// affordable for this model. Attention output-dimension splits align to whole
// heads; the head-block sizes come from the caller via the model geometry so
// this module never duplicates model constants.

#define SPARK_GLM52_TP_SHARD_ABI_VERSION 1u

typedef enum SparkGlm52TpShardClass
{
	SPARK_GLM52_TP_SHARD_CLASS_UNKNOWN = 0,
	SPARK_GLM52_TP_SHARD_CLASS_REPLICATED,
	SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM_HEADS,
	SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM,
	SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM_HEADS,
	SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM
} SparkGlm52TpShardClass;

typedef struct SparkGlm52TpShapeDescriptor
{
	uint32_t abi_version;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t pp_stage_count;
	uint32_t pp_stage_index;
} SparkGlm52TpShapeDescriptor;

// Head-block sizes in elements of the output (or input) dimension for the
// attention projections, supplied by the caller from the authoritative model
// header: q_b output block = qk_nope + rope per head, kv_b output block =
// qk_nope + value per head, o_proj input block = value per head.
typedef struct SparkGlm52TpModelGeometry
{
	uint32_t abi_version;
	uint32_t head_count;
	uint32_t q_b_head_block;
	uint32_t kv_b_head_block;
	uint32_t o_proj_head_block;
} SparkGlm52TpModelGeometry;

typedef struct SparkGlm52TpShardView
{
	uint32_t abi_version;
	uint32_t shard_class;
	// Dimension index within the tensor spec that is split; undefined for
	// replicated tensors, where offset is zero and extent the full dimension.
	uint32_t split_dimension;
	uint32_t reserved0;
	uint64_t element_offset;
	uint64_t element_extent;
	uint64_t shard_bytes;
} SparkGlm52TpShardView;

// Fill the model geometry from the authoritative model constants: the q_b
// output block is one head's nope plus rope columns, the kv_b output block
// one head's nope plus value columns, and the o_proj input block one head's
// value columns. Every consumer derives from these definitions, so the
// sharder, the shape config, and the kernels cannot drift apart.
static inline void SparkGlm52TpModelGeometryFromModel(SparkGlm52TpModelGeometry *geometry)
{
	geometry->abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
	geometry->head_count = SPARK_GLM52_MODEL_HEAD_COUNT;
	geometry->q_b_head_block =
		SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION + SPARK_GLM52_MODEL_ROPE_DIMENSION;
	geometry->kv_b_head_block =
		SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION + SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION;
	geometry->o_proj_head_block = SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION;
}

// Classify by the resolved tensor-name suffix. Unknown names return the
// unknown class; callers must treat that as a hard error for any tp_degree
// above one, so a new tensor added to the pack cannot be silently mis-sharded.
SparkGlm52TpShardClass SparkGlm52TpShardClassifyTensor(const char *tensor_name);

// Compute this rank's slice of the tensor. Fails closed on: unsupported
// tp_degree (only 1, 2, 4, 8, 16 divide the model's 64 heads and both MLP
// intermediates), tp_rank out of range, unknown class at tp_degree above one,
// head counts or dimensions not divisible by the degree, and geometry
// descriptors with mismatched ABI. tp_degree one returns the full tensor as a
// replicated view for every class, so a single-rank shape needs no special
// casing anywhere downstream.
SparkStatus SparkGlm52TpShardComputeView(
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpModelGeometry *geometry,
	SparkGlm52TpShardView *view_out);

// Order-sensitive FNV-1a over the shape descriptor, tensor identity, and the
// computed view, for the pack-to-kernel binding contract: a shard image built
// under any other degree, rank, or geometry hashes differently and refuses to
// bind.
uint64_t SparkGlm52TpShardGeometryHash(
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpShardView *view);

// Read this rank's shard of the tensor from the stagepack into the caller's
// buffer, which must be exactly the shard's bytes; any mismatch fails closed
// rather than truncating. A split on the leading dimension (and every
// replicated or degree-one view) is one contiguous range read; a split on an
// inner dimension gathers one chunk per outer row at the full-tensor row
// pitch. Fails with the same closed semantics as the geometry: unknown
// tensors, indivisible dimensions, and shape mismatches never produce bytes.
SparkStatus SparkGlm52TpShardReadTensor(
	const char *stagepack_root,
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpModelGeometry *geometry,
	void *destination,
	uint64_t destination_bytes,
	SparkGlm52TpShardView *view_out);

#endif
