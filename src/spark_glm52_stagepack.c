#include "sparkpipe/spark_glm52_stagepack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "sparkpipe/spark_json.h"

static SparkStatus SparkGlm52StagePackBuildPath(
	const char *root,
	const char *leaf,
	char *path,
	uint32_t path_bytes)
{
	int written;

	if (root == 0 || root[0] == '\0' || leaf == 0 || leaf[0] == '\0' ||
		path == 0 || path_bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	written = snprintf(path,path_bytes,"%s/%s",root,leaf);
	if (written < 0 || (uint32_t)written >= path_bytes)
	{
		path[0] = '\0';
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52StagePackValidateSpec(
	const SparkGlm52StagePackTensorSpec *spec)
{
	uint32_t dimension_index;

	if (spec == 0 ||
		spec->abi_version != SPARK_GLM52_STAGEPACK_ABI_VERSION ||
		spec->tensor_name == 0 || spec->tensor_name[0] == '\0' ||
		spec->dtype == 0 || spec->dtype[0] == '\0' ||
		spec->rank == 0u || spec->rank > SPARK_GLM52_STAGEPACK_MAX_RANK ||
		spec->bytes_per_element == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
	{
		if (spec->shape[dimension_index] == 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52StagePackExpectedBytes(
	const SparkGlm52StagePackTensorSpec *spec,
	uint64_t *bytes_out)
{
	uint64_t bytes;
	uint32_t dimension_index;

	if (bytes_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	bytes = spec->bytes_per_element;
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
	{
		if (bytes > UINT64_MAX / spec->shape[dimension_index])
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		bytes *= spec->shape[dimension_index];
	}
	*bytes_out = bytes;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52StagePackValidateShape(
	const SparkJsonDocument *document,
	int32_t shape_token_index,
	const SparkGlm52StagePackTensorSpec *spec)
{
	uint64_t observed_dimension;
	uint32_t dimension_index;

	if (SparkJsonGetArrayElementCount(document,shape_token_index) != spec->rank)
		return SPARK_STATUS_SCHEMA_ERROR;
	for (dimension_index = 0u; dimension_index < spec->rank; ++dimension_index)
	{
		if (SparkJsonGetUInt64(
				document,
				SparkJsonGetArrayElement(document,shape_token_index,dimension_index),
				&observed_dimension) != SPARK_STATUS_OK ||
			observed_dimension != spec->shape[dimension_index])
			return SPARK_STATUS_SCHEMA_ERROR;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52StagePackValidateFileRange(
	const char *path,
	uint64_t file_offset,
	uint64_t tensor_bytes)
{
	struct stat file_status;

	if (path == 0 || stat(path,&file_status) != 0 || file_status.st_size < 0)
		return SPARK_STATUS_NOT_FOUND;
	if (file_offset > UINT64_MAX - tensor_bytes)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ((uint64_t)file_status.st_size < (file_offset + tensor_bytes))
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52StagePackReadTensorRegion(
	const char *stagepack_root,
	const SparkGlm52StagePackTensorSpec *spec,
	const SparkJsonDocument *document,
	SparkGlm52StagePackTensorRegion *region)
{
	char *file_name;
	uint64_t observed_bytes;
	uint64_t expected_bytes;
	SparkStatus status;
	int32_t root_token_index,tensor_map_token_index,tensor_token_index;
	int32_t format_token_index;
	int32_t file_token_index,dtype_token_index,shape_token_index;
	int32_t offset_token_index,bytes_token_index;

	file_name = 0;
	root_token_index = SparkJsonGetRootToken(document);
	format_token_index = SparkJsonFindObjectMember(document,root_token_index,"format");
	tensor_map_token_index = SparkJsonFindObjectMember(document,root_token_index,"tensor_map");
	tensor_token_index = tensor_map_token_index >= 0 ?
		SparkJsonFindObjectMember(document,tensor_map_token_index,spec->tensor_name) : -1;
	file_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(document,tensor_token_index,"file") : -1;
	dtype_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(document,tensor_token_index,"dtype") : -1;
	shape_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(document,tensor_token_index,"shape") : -1;
	offset_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(document,tensor_token_index,"offset") : -1;
	bytes_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(document,tensor_token_index,"bytes") : -1;
	status = SparkGlm52StagePackExpectedBytes(spec,&expected_bytes);
	if (status == SPARK_STATUS_OK &&
		(format_token_index < 0 ||
		 !SparkJsonStringEquals(document,format_token_index,SPARK_GLM52_STAGEPACK_FORMAT) ||
		 file_token_index < 0 ||
		 SparkJsonCopyString(document,file_token_index,&file_name) != SPARK_STATUS_OK ||
		 dtype_token_index < 0 ||
		 !SparkJsonStringEquals(document,dtype_token_index,spec->dtype) ||
		 shape_token_index < 0 ||
		 offset_token_index < 0 ||
		 bytes_token_index < 0))
		status = SPARK_STATUS_SCHEMA_ERROR;
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackValidateShape(document,shape_token_index,spec);
	if (status == SPARK_STATUS_OK &&
		(SparkJsonGetUInt64(document,offset_token_index,&region->file_offset) != SPARK_STATUS_OK ||
		 SparkJsonGetUInt64(document,bytes_token_index,&observed_bytes) != SPARK_STATUS_OK ||
		 observed_bytes != expected_bytes))
		status = SPARK_STATUS_SCHEMA_ERROR;
	if (status == SPARK_STATUS_OK)
	{
		region->tensor_bytes = observed_bytes;
		status = SparkGlm52StagePackBuildPath(
			stagepack_root,
			file_name,
			region->file_path,
			sizeof(region->file_path));
	}
	free(file_name);
	return status;
}

SparkStatus SparkGlm52StagePackResolveTensor(
	const char *stagepack_root,
	const SparkGlm52StagePackTensorSpec *spec,
	SparkGlm52StagePackTensorRegion *region)
{
	SparkJsonDocument document;
	char index_path[SPARK_GLM52_STAGEPACK_PATH_BYTES];
	SparkStatus status;

	if (region == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(region,0,sizeof(*region));
	region->abi_version = SPARK_GLM52_STAGEPACK_ABI_VERSION;
	status = SparkGlm52StagePackValidateSpec(spec);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52StagePackBuildPath(
		stagepack_root,
		SPARK_GLM52_STAGEPACK_INDEX_FILE,
		index_path,
		sizeof(index_path));
	if (status != SPARK_STATUS_OK)
		return status;
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(index_path,&document);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackReadTensorRegion(
			stagepack_root,
			spec,
			&document,
			region);
	SparkJsonDocumentDestroy(&document);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackValidateFileRange(
			region->file_path,
			region->file_offset,
			region->tensor_bytes);
	if (status != SPARK_STATUS_OK)
		memset(region,0,sizeof(*region));
	return status;
}
