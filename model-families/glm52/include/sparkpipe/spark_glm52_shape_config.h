#ifndef SPARKPIPE_SPARK_GLM52_SHAPE_CONFIG_H
#define SPARKPIPE_SPARK_GLM52_SHAPE_CONFIG_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_tp_shard.h"
#include "sparkpipe/spark_status.h"

// Shape-driven node configuration: the device driver is told the inference
// shape, TP degree and rank plus PP stage count and index, and everything a
// node needs derives from it. A node holds exactly the layers of its PP stage,
// and if TP sharded, exactly its shard of those layers' tensors. This module
// computes the layer slice, the per-rank kernel dimensions, the KV bytes per
// token for the stage depth, a configuration hash that extends the fail-closed
// binding contract to the whole node shape, and writes the node's sliced
// stagepack so the standard loader reads a node pack with no knowledge of
// sharding at all.
//
// Only TP, PP, and TP times PP exist; expert parallelism was considered and
// dropped, so expert tensors shard intra-expert through the same output and
// input dimension classes as everything else. Layer partitioning across PP
// stages requires an even split and fails closed otherwise: every shape in use
// divides the 78 layers exactly (13 stages of 6, 6 of 13, 3 of 26, 1 of 78),
// and refusing remainders keeps the stage-time balance a property of the shape
// rather than a scheduling accident.

#define SPARK_GLM52_SHAPE_CONFIG_ABI_VERSION 1u

typedef struct SparkGlm52ShapeModelInputs
{
	uint32_t abi_version;
	uint32_t total_layer_count;
	uint32_t hidden_dimension;
	uint32_t moe_intermediate_dimension;
	uint32_t dense_intermediate_dimension;
	uint32_t kv_latent_plus_rope_dimension;
	uint32_t kv_bytes_per_element;
} SparkGlm52ShapeModelInputs;

typedef struct SparkGlm52ShapeNodeConfig
{
	uint32_t abi_version;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t heads_per_rank;
	uint32_t moe_intermediate_per_rank;
	uint32_t dense_intermediate_per_rank;
	uint32_t q_b_output_per_rank;
	uint32_t kv_b_output_per_rank;
	uint32_t o_proj_input_per_rank;
	uint32_t reserved0;
	uint64_t kv_bytes_per_token;
	uint64_t configuration_hash;
} SparkGlm52ShapeNodeConfig;

// Derive the node configuration from the shape. Fails closed on: unsupported
// TP degree, rank or stage index out of range, a layer count the stage count
// does not divide evenly, and head or intermediate dimensions the degree does
// not divide. The configuration hash covers the shape, the model inputs, and
// every derived value, so two nodes agreeing on the hash agree on the whole
// geometry.
SparkStatus SparkGlm52ShapeDeriveNodeConfig(
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpModelGeometry *geometry,
	const SparkGlm52ShapeModelInputs *inputs,
	SparkGlm52ShapeNodeConfig *config_out);

// Generate the node's sliced stagepack: for each supplied full-tensor spec,
// read this rank's shard from the full pack and append it to a single data
// file under the output root, then write a standard-schema index whose tensor
// shapes are the sharded shapes. The result is a self-contained pack the
// existing resolver loads whole, so nothing downstream of pack generation
// knows about sharding. The caller enumerates the specs for its stage's
// layers; scratch must hold the largest shard among them. Fails closed on the
// same conditions as the shard reader and never leaves a partially written
// index behind a successful return.
SparkStatus SparkGlm52ShapeWriteNodeStagePack(
	const char *full_stagepack_root,
	const char *node_stagepack_root,
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpModelGeometry *geometry,
	const SparkGlm52StagePackTensorSpec *specs,
	uint32_t spec_count,
	void *scratch,
	uint64_t scratch_bytes);

#endif
