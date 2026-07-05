#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_STAGEPACK_ABI_VERSION 1u
#define SPARK_GLM52_STAGEPACK_PATH_BYTES 4096u
#define SPARK_GLM52_STAGEPACK_MAX_RANK 8u
#define SPARK_GLM52_STAGEPACK_INDEX_FILE "stagepack_index.json"
#define SPARK_GLM52_STAGEPACK_FORMAT "sparkpipe.glm52.pp13.stagepack.v1"

typedef struct SparkGlm52StagePackTensorSpec
{
	uint32_t abi_version;
	uint32_t rank;
	uint64_t bytes_per_element;
	uint64_t shape[SPARK_GLM52_STAGEPACK_MAX_RANK];
	const char *tensor_name;
	const char *dtype;
} SparkGlm52StagePackTensorSpec;

typedef struct SparkGlm52StagePackTensorRegion
{
	uint32_t abi_version;
	uint32_t reserved0;
	uint64_t file_offset;
	uint64_t tensor_bytes;
	char file_path[SPARK_GLM52_STAGEPACK_PATH_BYTES];
} SparkGlm52StagePackTensorRegion;

SparkStatus SparkGlm52StagePackResolveTensor(
	const char *stagepack_root,
	const SparkGlm52StagePackTensorSpec *spec,
	SparkGlm52StagePackTensorRegion *region);

#ifdef __cplusplus
}
#endif
