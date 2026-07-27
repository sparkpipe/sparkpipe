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
#define SPARK_GLM52_STAGEPACK_FORMAT "sparkpipe.glm52.ring.stagepack.v1"
#define SPARK_GLM52_STAGEPACK_NVFP4_MODEL_QUANTIZATION "nvfp4"
#define SPARK_GLM52_STAGEPACK_NVFP4_NON_EXPERT_DTYPE "BF16"
#define SPARK_GLM52_STAGEPACK_NVFP4_MANIFEST_FILE \
	"resident_moe_pack_manifest.json"
#define SPARK_GLM52_STAGEPACK_NVFP4_MANIFEST_SCHEMA \
	"sparkpipe.glm52.sm121.b12x.resident_moe_pack.v1"
#define SPARK_GLM52_STAGEPACK_NVFP4_PACK_MAGIC "SPARKGLM52B12X"
#define SPARK_GLM52_STAGEPACK_NVFP4_PACK_EXTENSION ".spb12x"
#define SPARK_GLM52_STAGEPACK_NVFP4_QUANT_MODE 1u
#define SPARK_GLM52_STAGEPACK_W8LUT_MODEL_QUANTIZATION "w8lut"
#define SPARK_GLM52_STAGEPACK_W8LUT_NON_EXPERT_DTYPE "BF16"
#define SPARK_GLM52_STAGEPACK_W8LUT_MANIFEST_FILE "w8lut_moe_pack_manifest.json"
#define SPARK_GLM52_STAGEPACK_W8LUT_MANIFEST_FORMAT \
	"sparkpipe.glm52.w8lut.resident_moe_pack.v1"
#define SPARK_GLM52_STAGEPACK_W8LUT_PACK_MAGIC "SPARKGLM52W8LUT"
#define SPARK_GLM52_STAGEPACK_W8LUT_PACK_EXTENSION ".spw8lut"
#define SPARK_GLM52_STAGEPACK_W8LUT_QUANT_MODE 3u

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

SparkStatus SparkGlm52StagePackValidateW8lutContract(
	const char *stagepack_root,
	const char *w8lut_pack_root);

SparkStatus SparkGlm52StagePackValidateNvfp4Contract(
	const char *stagepack_root,
	const char *nvfp4_pack_root);

#ifdef __cplusplus
}
#endif
