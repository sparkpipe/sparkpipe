#include "sparkpipe/spark_glm52_stagepack.h"

#include <ctype.h>
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

static SparkStatus SparkGlm52StagePackCopyRootString(
	const SparkJsonDocument *document,
	const char *member_name,
	char **value_out)
{
	int32_t root_token_index,member_token_index;

	if (document == 0 || member_name == 0 || value_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*value_out = 0;
	root_token_index = SparkJsonGetRootToken(document);
	member_token_index = SparkJsonFindObjectMember(
		document,root_token_index,member_name);
	if (member_token_index < 0 ||
		SparkJsonCopyString(document,member_token_index,value_out) !=
			SPARK_STATUS_OK)
		return SPARK_STATUS_SCHEMA_ERROR;
	return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52StagePackSha256IsValid(const char *text)
{
	uint32_t index;

	if (text == 0 || strlen(text) != 64u)
		return 0u;
	for (index=0u; index<64u; ++index)
	{
		if (isxdigit((unsigned char)text[index]) == 0)
			return 0u;
	}
	return 1u;
}

static SparkStatus SparkGlm52StagePackRequireRootString(
	const SparkJsonDocument *document,
	int32_t root_token_index,
	const char *member_name,
	const char *expected_value)
{
	int32_t token_index;

	if (expected_value == 0)
		return SPARK_STATUS_OK;
	token_index = SparkJsonFindObjectMember(
		document,root_token_index,member_name);
	if (token_index < 0 ||
		SparkJsonStringEquals(document,token_index,expected_value) == 0)
		return SPARK_STATUS_SCHEMA_ERROR;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52StagePackValidateQuantizedDocument(
	const SparkJsonDocument *document,
	const char *schema_member_name,
	const char *expected_format,
	const char *expected_dtype,
	const char *expected_quantization,
	const char *expected_magic,
	const char *expected_extension,
	uint64_t expected_quant_mode)
{
	uint64_t quant_mode;
	int32_t root_token_index,token_index;
	SparkStatus status;

	root_token_index = SparkJsonGetRootToken(document);
	status = SparkGlm52StagePackRequireRootString(
		document,root_token_index,schema_member_name,expected_format);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackRequireRootString(
			document,root_token_index,"non_expert_weight_dtype",expected_dtype);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackRequireRootString(
			document,root_token_index,"model_quantization",expected_quantization);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackRequireRootString(
			document,root_token_index,"pack_magic",expected_magic);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackRequireRootString(
			document,root_token_index,"pack_extension",expected_extension);
	if (status != SPARK_STATUS_OK || expected_magic == 0)
		return status;
	token_index = SparkJsonFindObjectMember(
		document,root_token_index,"quant_mode");
	if (token_index < 0 ||
		SparkJsonGetUInt64(document,token_index,&quant_mode) != SPARK_STATUS_OK ||
		quant_mode != expected_quant_mode)
		return SPARK_STATUS_SCHEMA_ERROR;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52StagePackReadQuantizedContract(
	const char *root,
	const char *index_file,
	const char *schema_member_name,
	const char *expected_format,
	const char *expected_dtype,
	const char *expected_quantization,
	const char *expected_magic,
	const char *expected_extension,
	uint64_t expected_quant_mode,
	char **source_sha256_out)
{
	SparkJsonDocument document;
	char path[SPARK_GLM52_STAGEPACK_PATH_BYTES];
	char *source_sha256;
	SparkStatus status;

	if (source_sha256_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*source_sha256_out = 0;
	source_sha256 = 0;
	status = SparkGlm52StagePackBuildPath(
		root,index_file,path,(uint32_t)sizeof(path));
	SparkJsonDocumentReset(&document);
	if (status == SPARK_STATUS_OK)
		status = SparkJsonLoadFile(path,&document);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackValidateQuantizedDocument(
			&document,schema_member_name,expected_format,expected_dtype,
			expected_quantization,expected_magic,expected_extension,
			expected_quant_mode);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackCopyRootString(
			&document,"source_model_index_sha256",&source_sha256);
	if (status == SPARK_STATUS_OK &&
		SparkGlm52StagePackSha256IsValid(source_sha256) == 0u)
		status = SPARK_STATUS_SCHEMA_ERROR;
	SparkJsonDocumentDestroy(&document);
	if (status != SPARK_STATUS_OK)
	{
		free(source_sha256);
		return status;
	}
	*source_sha256_out = source_sha256;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52StagePackValidateQuantizedContract(
	const char *stagepack_root,
	const char *pack_root,
	const char *model_quantization,
	const char *non_expert_dtype,
	const char *manifest_file,
	const char *manifest_schema_member,
	const char *manifest_schema,
	const char *pack_magic,
	const char *pack_extension,
	uint64_t quant_mode)
{
	char *stagepack_sha256,*pack_sha256;
	SparkStatus status;

	if (stagepack_root == 0 || pack_root == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	stagepack_sha256 = 0;
	pack_sha256 = 0;
	status = SparkGlm52StagePackReadQuantizedContract(
		stagepack_root,
		SPARK_GLM52_STAGEPACK_INDEX_FILE,
		"format",
		SPARK_GLM52_STAGEPACK_FORMAT,
		non_expert_dtype,
		model_quantization,
		0,
		0,
		0u,
		&stagepack_sha256);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackReadQuantizedContract(
			pack_root,
			manifest_file,
			manifest_schema_member,
			manifest_schema,
			0,
			0,
			pack_magic,
			pack_extension,
			quant_mode,
			&pack_sha256);
	if (status == SPARK_STATUS_OK &&
		strcmp(stagepack_sha256,pack_sha256) != 0)
		status = SPARK_STATUS_SCHEMA_ERROR;
	free(stagepack_sha256);
	free(pack_sha256);
	return status;
}

SparkStatus SparkGlm52StagePackValidateNvfp4Contract(
	const char *stagepack_root,
	const char *nvfp4_pack_root)
{
	return SparkGlm52StagePackValidateQuantizedContract(
		stagepack_root,
		nvfp4_pack_root,
		SPARK_GLM52_STAGEPACK_NVFP4_MODEL_QUANTIZATION,
		SPARK_GLM52_STAGEPACK_NVFP4_NON_EXPERT_DTYPE,
		SPARK_GLM52_STAGEPACK_NVFP4_MANIFEST_FILE,
		"record_schema",
		SPARK_GLM52_STAGEPACK_NVFP4_MANIFEST_SCHEMA,
		SPARK_GLM52_STAGEPACK_NVFP4_PACK_MAGIC,
		SPARK_GLM52_STAGEPACK_NVFP4_PACK_EXTENSION,
		SPARK_GLM52_STAGEPACK_NVFP4_QUANT_MODE);
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
