#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "sparkpipe/spark_glm52_stagepack.h"

static void SparkTestMakeDirectory(const char *path)
{
	(void)mkdir(path,0775);
}

static void SparkTestWriteTextFile(const char *path,const char *text)
{
	FILE *file;

	file = fopen(path,"wb");
	assert(file != 0);
	assert(fputs(text,file) >= 0);
	assert(fclose(file) == 0);
}

static void SparkTestWritePayloadFile(const char *path)
{
	FILE *file;
	uint8_t bytes[32];
	uint32_t index;

	for (index = 0u; index < sizeof(bytes); ++index)
		bytes[index] = (uint8_t)index;
	file = fopen(path,"wb");
	assert(file != 0);
	assert(fwrite(bytes,1,sizeof(bytes),file) == sizeof(bytes));
	assert(fclose(file) == 0);
}

static void SparkTestInitializeSpec(
	SparkGlm52StagePackTensorSpec *spec,
	const char *dtype)
{
	memset(spec,0,sizeof(*spec));
	spec->abi_version = SPARK_GLM52_STAGEPACK_ABI_VERSION;
	spec->tensor_name = "model.layers.18.self_attn.q_proj.weight";
	spec->dtype = dtype;
	spec->bytes_per_element = 2u;
	spec->rank = 2u;
	spec->shape[0] = 4u;
	spec->shape[1] = 2u;
}

static void SparkTestGlm52StagePackResolveTensor(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52StagePackTensorRegion region;
	const char *root;

	root = "build/test_glm52_stagepack_data";
	SparkTestMakeDirectory("build");
	SparkTestMakeDirectory(root);
	SparkTestWritePayloadFile("build/test_glm52_stagepack_data/stage_03_non_moe.spstage");
	SparkTestWriteTextFile(
		"build/test_glm52_stagepack_data/stagepack_index.json",
		"{\"format\":\"sparkpipe.glm52.pp13.stagepack.v1\","
		"\"tensor_map\":{"
		"\"model.layers.18.self_attn.q_proj.weight\":{"
		"\"file\":\"stage_03_non_moe.spstage\","
		"\"offset\":8,"
		"\"bytes\":16,"
		"\"dtype\":\"BF16\","
		"\"shape\":[4,2]}}}");
	SparkTestInitializeSpec(&spec,"BF16");
	assert(SparkGlm52StagePackResolveTensor(root,&spec,&region) ==
		SPARK_STATUS_OK);
	assert(region.abi_version == SPARK_GLM52_STAGEPACK_ABI_VERSION);
	assert(region.file_offset == 8u);
	assert(region.tensor_bytes == 16u);
	assert(strcmp(
		region.file_path,
		"build/test_glm52_stagepack_data/stage_03_non_moe.spstage") == 0);
	SparkTestInitializeSpec(&spec,"U8");
	assert(SparkGlm52StagePackResolveTensor(root,&spec,&region) ==
		SPARK_STATUS_SCHEMA_ERROR);
	SparkTestInitializeSpec(&spec,"BF16");
	spec.shape[1] = 3u;
	assert(SparkGlm52StagePackResolveTensor(root,&spec,&region) ==
		SPARK_STATUS_SCHEMA_ERROR);
	SparkTestInitializeSpec(&spec,"BF16");
	spec.tensor_name = "missing";
	assert(SparkGlm52StagePackResolveTensor(root,&spec,&region) ==
		SPARK_STATUS_SCHEMA_ERROR);
}

int main(void)
{
	SparkTestGlm52StagePackResolveTensor();
	return 0;
}
