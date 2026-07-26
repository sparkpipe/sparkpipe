#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "sparkpipe/spark_glm52_shape_config.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static uint64_t SparkGlm52ShapeHashBytes(uint64_t hash,const void *data,uint32_t data_bytes)
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

SparkStatus SparkGlm52ShapeDeriveNodeConfig(
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpModelGeometry *geometry,
	const SparkGlm52ShapeModelInputs *inputs,
	SparkGlm52ShapeNodeConfig *config_out)
{
	uint64_t hash;
	if (shape == 0 || geometry == 0 || inputs == 0 || config_out == 0 ||
		shape->abi_version != SPARK_GLM52_TP_SHARD_ABI_VERSION ||
		geometry->abi_version != SPARK_GLM52_TP_SHARD_ABI_VERSION ||
		inputs->abi_version != SPARK_GLM52_SHAPE_CONFIG_ABI_VERSION)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (shape->tp_degree != 1u && shape->tp_degree != 2u &&
		shape->tp_degree != 4u && shape->tp_degree != 8u &&
		shape->tp_degree != 16u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (shape->tp_rank >= shape->tp_degree ||
		shape->pp_stage_count == 0u ||
		shape->pp_stage_index >= shape->pp_stage_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (inputs->total_layer_count == 0u ||
		inputs->total_layer_count % shape->pp_stage_count != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (geometry->head_count == 0u ||
		geometry->head_count % shape->tp_degree != 0u ||
		inputs->moe_intermediate_dimension % shape->tp_degree != 0u ||
		inputs->dense_intermediate_dimension % shape->tp_degree != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(config_out,0,sizeof(*config_out));
	config_out->abi_version = SPARK_GLM52_SHAPE_CONFIG_ABI_VERSION;
	config_out->layer_count = inputs->total_layer_count / shape->pp_stage_count;
	config_out->first_layer_index = shape->pp_stage_index * config_out->layer_count;
	config_out->heads_per_rank = geometry->head_count / shape->tp_degree;
	config_out->moe_intermediate_per_rank =
		inputs->moe_intermediate_dimension / shape->tp_degree;
	config_out->dense_intermediate_per_rank =
		inputs->dense_intermediate_dimension / shape->tp_degree;
	config_out->q_b_output_per_rank =
		config_out->heads_per_rank * geometry->q_b_head_block;
	config_out->kv_b_output_per_rank =
		config_out->heads_per_rank * geometry->kv_b_head_block;
	config_out->o_proj_input_per_rank =
		config_out->heads_per_rank * geometry->o_proj_head_block;
	// Every TP rank of a stage runs all of the stage's layers against the full
	// head-agnostic latent, so KV per token is the stage depth times the latent
	// row regardless of the TP degree; the replication is across the TP group,
	// not within a node.
	config_out->kv_bytes_per_token =
		(uint64_t)config_out->layer_count *
		inputs->kv_latent_plus_rope_dimension *
		inputs->kv_bytes_per_element;
	hash = 1469598103934665603u;
	hash = SparkGlm52ShapeHashBytes(hash,shape,sizeof(*shape));
	hash = SparkGlm52ShapeHashBytes(hash,geometry,sizeof(*geometry));
	hash = SparkGlm52ShapeHashBytes(hash,inputs,sizeof(*inputs));
	hash = SparkGlm52ShapeHashBytes(hash,config_out,
		(uint32_t)((const uint8_t *)&config_out->configuration_hash -
			(const uint8_t *)config_out));
	config_out->configuration_hash = hash;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ShapeAppendShardFile(FILE *data_stream,const void *shard,uint64_t shard_bytes,uint64_t *file_offset_inout)
{
	if (fwrite(shard,1u,(size_t)shard_bytes,data_stream) != (size_t)shard_bytes)
		return SPARK_STATUS_IO_ERROR;
	*file_offset_inout += shard_bytes;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ShapeAppendIndexEntry(FILE *index_stream,const SparkGlm52StagePackTensorSpec *spec,const SparkGlm52TpShardView *view,uint64_t file_offset,uint32_t first_entry)
{
	uint32_t dimension_index;
	if (fprintf(index_stream,"%s\n    \"%s\": {\n      \"file\": \"node_tensors.bin\",\n      \"dtype\": \"%s\",\n      \"shape\": [",
		first_entry != 0u ? "" : ",",spec->tensor_name,spec->dtype) < 0)
		return SPARK_STATUS_IO_ERROR;
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
	{
		uint64_t dimension_value = dimension_index == view->split_dimension ?
			view->element_extent : spec->shape[dimension_index];
		if (fprintf(index_stream,"%s%llu",dimension_index == 0u ? "" : ", ",
			(unsigned long long)dimension_value) < 0)
			return SPARK_STATUS_IO_ERROR;
	}
	if (fprintf(index_stream,"],\n      \"offset\": %llu,\n      \"bytes\": %llu}",
		(unsigned long long)file_offset,(unsigned long long)view->shard_bytes) < 0)
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ShapeWriteNodeStagePack(
	const char *full_stagepack_root,
	const char *node_stagepack_root,
	const SparkGlm52TpShapeDescriptor *shape,
	const SparkGlm52TpModelGeometry *geometry,
	const SparkGlm52StagePackTensorSpec *specs,
	uint32_t spec_count,
	void *scratch,
	uint64_t scratch_bytes)
{
	char data_path[SPARK_GLM52_STAGEPACK_PATH_BYTES];
	char index_path[SPARK_GLM52_STAGEPACK_PATH_BYTES];
	FILE *data_stream,*index_stream;
	uint64_t file_offset;
	uint32_t spec_index;
	SparkStatus status;
	if (node_stagepack_root == 0 || specs == 0 || spec_count == 0u ||
		scratch == 0 || scratch_bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (mkdir(node_stagepack_root,0755) != 0)
	{
		struct stat root_stat;
		if (stat(node_stagepack_root,&root_stat) != 0)
			return SPARK_STATUS_IO_ERROR;
	}
	if (snprintf(data_path,sizeof(data_path),"%s/node_tensors.bin",node_stagepack_root) <= 0 ||
		snprintf(index_path,sizeof(index_path),"%s/%s",node_stagepack_root,SPARK_GLM52_STAGEPACK_INDEX_FILE) <= 0)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	data_stream = fopen(data_path,"wb");
	if (data_stream == 0)
		return SPARK_STATUS_IO_ERROR;
	index_stream = fopen(index_path,"w");
	if (index_stream == 0)
	{
		fclose(data_stream);
		return SPARK_STATUS_IO_ERROR;
	}
	status = fprintf(index_stream,"{\n  \"format\": \"%s\",\n  \"tensor_map\": {",
		SPARK_GLM52_STAGEPACK_FORMAT) < 0 ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
	file_offset = 0u;
	for (spec_index = 0u; status == SPARK_STATUS_OK && spec_index < spec_count; ++spec_index)
	{
		SparkGlm52TpShardView view;
		uint64_t entry_offset = file_offset;
		status = SparkGlm52TpShardComputeView(&specs[spec_index],shape,geometry,&view);
		if (status != SPARK_STATUS_OK)
			break;
		if (view.shard_bytes > scratch_bytes)
		{
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
			break;
		}
		status = SparkGlm52TpShardReadTensor(full_stagepack_root,&specs[spec_index],shape,geometry,scratch,view.shard_bytes,&view);
		if (status != SPARK_STATUS_OK)
			break;
		status = SparkGlm52ShapeAppendShardFile(data_stream,scratch,view.shard_bytes,&file_offset);
		if (status != SPARK_STATUS_OK)
			break;
		status = SparkGlm52ShapeAppendIndexEntry(index_stream,&specs[spec_index],&view,entry_offset,spec_index == 0u ? 1u : 0u);
	}
	if (status == SPARK_STATUS_OK &&
		fprintf(index_stream,"\n  }\n}\n") < 0)
		status = SPARK_STATUS_IO_ERROR;
	if (fclose(data_stream) != 0 && status == SPARK_STATUS_OK)
		status = SPARK_STATUS_IO_ERROR;
	if (fclose(index_stream) != 0 && status == SPARK_STATUS_OK)
		status = SPARK_STATUS_IO_ERROR;
	if (status != SPARK_STATUS_OK)
	{
		remove(data_path);
		remove(index_path);
	}
	return status;
}
