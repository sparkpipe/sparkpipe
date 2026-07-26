#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "sparkpipe/spark_glm52_tp_shard.h"

#include <fcntl.h>
#include <unistd.h>

#include <string.h>

static uint32_t SparkGlm52TpShardNameEndsWith(const char *tensor_name,const char *suffix)
{
	uint32_t name_bytes,suffix_bytes;
	if (tensor_name == 0 || suffix == 0)
		return 0u;
	name_bytes = (uint32_t)strlen(tensor_name);
	suffix_bytes = (uint32_t)strlen(suffix);
	if (suffix_bytes > name_bytes)
		return 0u;
	return memcmp(tensor_name + (name_bytes - suffix_bytes),suffix,suffix_bytes) == 0 ? 1u : 0u;
}

SparkGlm52TpShardClass SparkGlm52TpShardClassifyTensor(const char *tensor_name)
{
	// Head-structured attention projections: q_b and kv_b split their output
	// dimension on whole-head boundaries; o_proj splits its input dimension the
	// same way and closes with the layer all-reduce.
	if (SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.q_b_proj.weight"))
		return SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM_HEADS;
	if (SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.kv_b_proj.weight"))
		return SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM_HEADS;
	if (SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.o_proj.weight"))
		return SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM_HEADS;
	// Dense and shared-expert MLP: gate and up split the intermediate output
	// dimension, down splits the intermediate input dimension.
	if (SparkGlm52TpShardNameEndsWith(tensor_name,"mlp.gate_proj.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"mlp.up_proj.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"mlp.shared_experts.gate_proj.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"mlp.shared_experts.up_proj.weight"))
		return SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM;
	if (SparkGlm52TpShardNameEndsWith(tensor_name,"mlp.down_proj.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"mlp.shared_experts.down_proj.weight"))
		return SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM;
	// MLA latent paths are head-agnostic and replicate on every rank, which is
	// what keeps the latent KV cache identical everywhere. The router and its
	// correction bias replicate so every rank routes identically. Norms, the
	// embedding, and the MTP projections replicate in phase one.
	if (SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.q_a_proj.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.kv_a_proj_with_mqa.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.q_a_layernorm.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.kv_a_layernorm.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.indexer.k_norm.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.indexer.k_norm.bias") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.indexer.weights_proj.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.indexer.wk.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"self_attn.indexer.wq_b.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"mlp.gate.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"mlp.gate.e_score_correction_bias") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"input_layernorm.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"post_attention_layernorm.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"model.norm.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"model.embed_tokens.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"enorm.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"hnorm.weight") ||
		SparkGlm52TpShardNameEndsWith(tensor_name,"eh_proj.weight"))
		return SPARK_GLM52_TP_SHARD_CLASS_REPLICATED;
	return SPARK_GLM52_TP_SHARD_CLASS_UNKNOWN;
}

static SparkStatus SparkGlm52TpShardValidate(const SparkGlm52StagePackTensorSpec *spec,const SparkGlm52TpShapeDescriptor *shape,const SparkGlm52TpModelGeometry *geometry)
{
	if (spec == 0 || shape == 0 || geometry == 0 ||
		shape->abi_version != SPARK_GLM52_TP_SHARD_ABI_VERSION ||
		geometry->abi_version != SPARK_GLM52_TP_SHARD_ABI_VERSION)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (shape->tp_degree != 1u && shape->tp_degree != 2u &&
		shape->tp_degree != 4u && shape->tp_degree != 8u &&
		shape->tp_degree != 16u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (shape->tp_rank >= shape->tp_degree)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (geometry->head_count == 0u ||
		geometry->head_count % shape->tp_degree != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52TpShardSplitDimension(const SparkGlm52StagePackTensorSpec *spec,const SparkGlm52TpShapeDescriptor *shape,uint32_t split_dimension,uint64_t block_elements,SparkGlm52TpShardView *view_out)
{
	uint64_t dimension_elements,block_count,blocks_per_rank,other_bytes;
	uint32_t dimension_index;
	dimension_elements = spec->shape[split_dimension];
	if (block_elements == 0u || dimension_elements % block_elements != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	block_count = dimension_elements / block_elements;
	if (block_count % shape->tp_degree != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	blocks_per_rank = block_count / shape->tp_degree;
	view_out->split_dimension = split_dimension;
	view_out->element_offset = (uint64_t)shape->tp_rank * blocks_per_rank * block_elements;
	view_out->element_extent = blocks_per_rank * block_elements;
	other_bytes = spec->bytes_per_element;
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
	{
		if (dimension_index == split_dimension)
			continue;
		other_bytes *= spec->shape[dimension_index];
	}
	view_out->shard_bytes = other_bytes * view_out->element_extent;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52TpShardComputeView(
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpModelGeometry *geometry,
	SparkGlm52TpShardView *view_out)
{
	SparkGlm52TpShardClass shard_class;
	SparkStatus status;
	uint64_t full_bytes;
	uint32_t dimension_index;
	if (view_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52TpShardValidate(spec,shape,geometry);
	if (status != SPARK_STATUS_OK)
		return status;
	shard_class = SparkGlm52TpShardClassifyTensor(spec->tensor_name);
	memset(view_out,0,sizeof(*view_out));
	view_out->abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
	view_out->shard_class = (uint32_t)shard_class;
	full_bytes = spec->bytes_per_element;
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
		full_bytes *= spec->shape[dimension_index];
	// Degree one is a whole-tensor view for every class, including unknown, so
	// existing single-shape packs keep loading without touching this module's
	// classification. At any higher degree an unknown tensor fails closed.
	if (shape->tp_degree == 1u || shard_class == SPARK_GLM52_TP_SHARD_CLASS_REPLICATED)
	{
		if (shape->tp_degree != 1u && shard_class == SPARK_GLM52_TP_SHARD_CLASS_UNKNOWN)
			return SPARK_STATUS_VALIDATION_FAILED;
		view_out->split_dimension = 0u;
		view_out->element_offset = 0u;
		view_out->element_extent = spec->shape[0];
		view_out->shard_bytes = full_bytes;
		return SPARK_STATUS_OK;
	}
	if (shard_class == SPARK_GLM52_TP_SHARD_CLASS_UNKNOWN)
		return SPARK_STATUS_VALIDATION_FAILED;
	if (spec->rank < 2u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (shard_class == SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM_HEADS)
	{
		uint64_t block = SparkGlm52TpShardNameEndsWith(spec->tensor_name,"self_attn.q_b_proj.weight") != 0u ?
			geometry->q_b_head_block : geometry->kv_b_head_block;
		return SparkGlm52TpShardSplitDimension(spec,shape,0u,block,view_out);
	}
	if (shard_class == SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM_HEADS)
		return SparkGlm52TpShardSplitDimension(spec,shape,1u,geometry->o_proj_head_block,view_out);
	if (shard_class == SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM)
		return SparkGlm52TpShardSplitDimension(spec,shape,0u,1u,view_out);
	return SparkGlm52TpShardSplitDimension(spec,shape,1u,1u,view_out);
}

static uint64_t SparkGlm52TpShardHashBytes(uint64_t hash,const void *data,uint32_t data_bytes)
{
	const uint8_t *bytes = (const uint8_t *)data;
	uint32_t byte_index;
	for (byte_index = 0u; byte_index < data_bytes; ++byte_index)
	{
		hash ^= bytes[byte_index];
		hash *= 1099511628211u;
	}
	return hash;
}

uint64_t SparkGlm52TpShardGeometryHash(
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpShardView *view)
{
	uint64_t hash = 1469598103934665603u;
	uint32_t dimension_index;
	if (spec == 0 || shape == 0 || view == 0)
		return 0u;
	hash = SparkGlm52TpShardHashBytes(hash,spec->tensor_name,(uint32_t)strlen(spec->tensor_name));
	hash = SparkGlm52TpShardHashBytes(hash,&spec->rank,sizeof(spec->rank));
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
		hash = SparkGlm52TpShardHashBytes(hash,&spec->shape[dimension_index],sizeof(spec->shape[0]));
	hash = SparkGlm52TpShardHashBytes(hash,&shape->tp_degree,sizeof(shape->tp_degree));
	hash = SparkGlm52TpShardHashBytes(hash,&shape->tp_rank,sizeof(shape->tp_rank));
	hash = SparkGlm52TpShardHashBytes(hash,&shape->pp_stage_count,sizeof(shape->pp_stage_count));
	hash = SparkGlm52TpShardHashBytes(hash,&shape->pp_stage_index,sizeof(shape->pp_stage_index));
	hash = SparkGlm52TpShardHashBytes(hash,&view->shard_class,sizeof(view->shard_class));
	hash = SparkGlm52TpShardHashBytes(hash,&view->split_dimension,sizeof(view->split_dimension));
	hash = SparkGlm52TpShardHashBytes(hash,&view->element_offset,sizeof(view->element_offset));
	hash = SparkGlm52TpShardHashBytes(hash,&view->element_extent,sizeof(view->element_extent));
	return hash;
}

// Split read geometry for a view: outer rows before the split dimension, the
// full-tensor pitch of one outer row, the shard chunk within that row, and the
// chunk's byte offset. A leading-dimension split degenerates to one outer row
// covering the contiguous shard, so the same loop serves every class.
static void SparkGlm52TpShardReadGeometry(const SparkGlm52StagePackTensorSpec *spec,const SparkGlm52TpShardView *view,uint64_t *outer_rows,uint64_t *row_pitch_bytes,uint64_t *chunk_bytes,uint64_t *chunk_offset_bytes)
{
	uint64_t inner_bytes = spec->bytes_per_element;
	uint64_t outer = 1u;
	uint32_t dimension_index;
	for (dimension_index = view->split_dimension + 1u; dimension_index < spec->rank; ++dimension_index)
		inner_bytes *= spec->shape[dimension_index];
	for (dimension_index = 0u; dimension_index < view->split_dimension; ++dimension_index)
		outer *= spec->shape[dimension_index];
	*outer_rows = outer;
	*row_pitch_bytes = spec->shape[view->split_dimension] * inner_bytes;
	*chunk_bytes = view->element_extent * inner_bytes;
	*chunk_offset_bytes = view->element_offset * inner_bytes;
}

static SparkStatus SparkGlm52TpShardReadRegion(const SparkGlm52StagePackTensorRegion *region,uint64_t outer_rows,uint64_t row_pitch_bytes,uint64_t chunk_bytes,uint64_t chunk_offset_bytes,uint8_t *destination)
{
	int file_descriptor;
	uint64_t row_index;
	file_descriptor = open(region->file_path,O_RDONLY);
	if (file_descriptor < 0)
		return SPARK_STATUS_IO_ERROR;
	for (row_index = 0u; row_index < outer_rows; ++row_index)
	{
		off_t read_offset = (off_t)(region->file_offset +
			row_index * row_pitch_bytes + chunk_offset_bytes);
		uint64_t read_bytes = 0u;
		while (read_bytes < chunk_bytes)
		{
			ssize_t got = pread(file_descriptor,
				destination + row_index * chunk_bytes + read_bytes,
				(size_t)(chunk_bytes - read_bytes),
				read_offset + (off_t)read_bytes);
			if (got <= 0)
			{
				close(file_descriptor);
				return SPARK_STATUS_IO_ERROR;
			}
			read_bytes += (uint64_t)got;
		}
	}
	close(file_descriptor);
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52TpShardReadTensor(
	const char *stagepack_root,
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpModelGeometry *geometry,
	void *destination,
	uint64_t destination_bytes,
	SparkGlm52TpShardView *view_out)
{
	SparkGlm52TpShardView view;
	SparkGlm52StagePackTensorRegion region;
	uint64_t outer_rows,row_pitch_bytes,chunk_bytes,chunk_offset_bytes;
	SparkStatus status;
	if (destination == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52TpShardComputeView(spec,shape,geometry,&view);
	if (status != SPARK_STATUS_OK)
		return status;
	if (destination_bytes != view.shard_bytes)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52StagePackResolveTensor(stagepack_root,spec,&region);
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52TpShardReadGeometry(spec,&view,&outer_rows,&row_pitch_bytes,&chunk_bytes,&chunk_offset_bytes);
	status = SparkGlm52TpShardReadRegion(&region,outer_rows,row_pitch_bytes,chunk_bytes,chunk_offset_bytes,(uint8_t *)destination);
	if (status != SPARK_STATUS_OK)
		return status;
	if (view_out != 0)
		*view_out = view;
	return SPARK_STATUS_OK;
}
