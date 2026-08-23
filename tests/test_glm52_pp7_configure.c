/* PP7 split dry proof (host, no GPU): a synthetic [12,11x6] pack per stage
 * must configure through the resident module chain with every stage's
 * expected-layer inventory mask resolving.
 *
 * White-box on purpose: the harness includes the module translation unit
 * directly so SparkGlm52ModuleConfigure and the pack validation/inventory
 * statics are drivable without materializing multi-GB payloads — the proof
 * stops after inventory resolution, before any device-region load. Also
 * pins the negative space: unknown geometry (the retired PP13 grid) and
 * uniform-multiply first layers must be rejected loudly. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "spark_glm52_resident_decode_stage_module.c"

#define TEST_MODEL_REVISION "pp7-dry"
#define TEST_PACK_DIR "build/tmp/glm52_pp7_dry"

static const uint32_t TestPp7Counts[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] =
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_LAYER_COUNTS_INITIALIZER;
static const uint32_t TestPp7Firsts[SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] =
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_FIRST_LAYER_INITIALIZER;

static uint32_t TestHexDigit(char digit)
{
	if ( digit >= '0' && digit <= '9' )
		return((uint32_t)(digit - '0'));
	return((uint32_t)(digit - 'a') + 10u);
}

static void TestHexSha(const char *hex,uint8_t out[SPARK_GLM52_STAGEPACK_SHA256_BYTES])
{
	uint32_t index;
	for (index = 0u; index < SPARK_GLM52_STAGEPACK_SHA256_BYTES; index++)
		out[index] = (uint8_t)(TestHexDigit(hex[2u*index]) * 16u +
			TestHexDigit(hex[2u*index + 1u]));
}

/* One synthetic stage pack: real header + directory + shapes at tp_degree=1,
 * payload bytes claimed but left as file holes (never read here). */
static void TestBuildStagePack(const char *path,uint32_t stage_count,
	uint32_t stage_index,uint32_t first_layer,uint32_t layer_count,
	int drop_one_indexer)
{
	uint8_t sha[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
	SparkGlm52StagePackHeader header;
	SparkGlm52StagePackEntry entries[SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT];
	SparkGlm52StagePackTensorShape shape;
	FILE *file;
	uint64_t offset,file_bytes,directory_end;
	uint32_t kind,layer,count;
	uint8_t contract_sha[SPARK_GLM52_STAGEPACK_SHA256_BYTES];

	memset(&header,0,sizeof(header));
	memset(entries,0,sizeof(entries));
	count = 0u;
	directory_end = 0u;
	TestHexSha("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",contract_sha);
	TestHexSha("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",sha);
	for (kind = 0u; kind < SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT; kind++)
	{
		uint32_t is_global = SparkGlm52StagePackKindIsGlobal(kind);
		if ( is_global != 0u )
		{
			uint32_t owns_embedding = kind == SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING && stage_index == 0u && first_layer == 0u;
			uint32_t owns_head = (kind == SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM || kind == SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD) && stage_index + 1u == stage_count;
			if ( owns_embedding == 0u && owns_head == 0u )
				continue;
			assert(SparkGlm52StagePackExpectedShape(kind,SPARK_GLM52_STAGEPACK_GLOBAL_LAYER,GLM52_EXPERT_WEIGHT_CODEC,1u,&shape) == 0);
			goto add_entry;
		}
		for (layer = first_layer; layer < first_layer + layer_count; layer++)
		{
			if ( SparkGlm52StagePackKindIsIndexer(kind) != 0u && SparkGlm52StagePackLayerHasFullIndexer(layer) == 0u )
				continue;
			if ( drop_one_indexer != 0 && kind == SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q &&
				layer + 1u == first_layer + layer_count &&
				SparkGlm52StagePackLayerHasFullIndexer(layer) != 0u )
				continue;
			if ( SparkGlm52StagePackKindIsDense(kind) != 0u && layer >= SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER )
				continue;
			if ( SparkGlm52StagePackKindIsRouted(kind) != 0u && layer < SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER )
				continue;
			assert(SparkGlm52StagePackExpectedShape(kind,layer,GLM52_EXPERT_WEIGHT_CODEC,1u,&shape) == 0);
add_entry:
			assert(count < SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT);
			entries[count].tensor_kind = kind;
			entries[count].layer_index = is_global != 0u ? SPARK_GLM52_STAGEPACK_GLOBAL_LAYER : layer;
			entries[count].payload_type = shape.payload_type;
			entries[count].weight_codec = shape.weight_codec;
			entries[count].scale_encoding = shape.scale_encoding;
			entries[count].group_count = shape.group_count;
			entries[count].rows = shape.rows;
			entries[count].columns = shape.columns;
			entries[count].payload_bytes = SparkStagePackPayloadBytes(&shape);
			entries[count].scale_bytes = SparkStagePackScaleBytes(&shape);
			count++;
			if ( is_global != 0u )
				break;
		}
	}
	/* Second pass: lay payloads/scales out after the directory, 256-aligned. */
	offset = 512u + (uint64_t)count * SPARK_GLM52_STAGEPACK_ENTRY_BYTES;
	for (kind = 0u; kind < count; kind++)
	{
		offset = (offset + SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES - 1u) & ~((uint64_t)SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES - 1u);
		entries[kind].payload_offset = offset;
		offset = entries[kind].payload_offset + entries[kind].payload_bytes;
		if ( entries[kind].scale_bytes != 0u )
		{
			offset = (offset + SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES - 1u) & ~((uint64_t)SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES - 1u);
			entries[kind].scale_offset = offset;
			offset = entries[kind].scale_offset + entries[kind].scale_bytes;
		}
	}
	file_bytes = offset;

	header.magic = SPARK_GLM52_STAGEPACK_MAGIC;
	header.format_version = SPARK_GLM52_STAGEPACK_FORMAT_VERSION;
	header.header_bytes = SPARK_GLM52_STAGEPACK_HEADER_BYTES;
	header.directory_entry_bytes = SPARK_GLM52_STAGEPACK_ENTRY_BYTES;
	header.codec_abi_version = SPARK_WEIGHT_CODEC_ABI_VERSION;
	header.flags = 0u;
	header.tensor_count = count;
	header.stage_count = stage_count;
	header.stage_index = stage_index;
	header.first_layer_index = first_layer;
	header.layer_count = layer_count;
	header.total_layer_count = SPARK_GLM52_MODEL_LAYER_COUNT;
	header.hidden_dimension = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
	header.vocab_count = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
	header.routed_expert_count = SPARK_GLM52_MODEL_MOE_EXPERT_COUNT;
	header.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	header.expert_weight_codec = GLM52_EXPERT_WEIGHT_CODEC;
	header.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	header.reserved0 = 1u; /* tp_degree */
	header.reserved1 = 0u; /* tp_rank */
	header.directory_offset = 512u;
	header.file_bytes = file_bytes;
	memcpy(header.model_revision,TEST_MODEL_REVISION,sizeof(TEST_MODEL_REVISION));
	memcpy(header.contract_sha256,contract_sha,sizeof(contract_sha));
	memcpy(header.source_config_sha256,sha,sizeof(sha));
	memcpy(header.pack_recipe_sha256,sha,sizeof(sha));
	directory_end = header.directory_offset + (uint64_t)count * SPARK_GLM52_STAGEPACK_ENTRY_BYTES;
	assert(directory_end <= entries[0].payload_offset);

	file = fopen(path,"wb");
	assert(file != 0);
	assert(fwrite(&header,1,sizeof(header),file) == sizeof(header));
	assert(fseeko(file,(off_t)header.directory_offset,SEEK_SET) == 0);
	for (kind = 0u; kind < count; kind++)
		assert(fwrite(&entries[kind],1,sizeof(entries[kind]),file) == sizeof(entries[kind]));
	/* Sparse claim of the payload extent: one byte at the end makes the file
	 * size true without storing ~10 GB of zeros per stage. */
	assert(fseeko(file,(off_t)(file_bytes - 1u),SEEK_SET) == 0);
	assert(fputc(0,file) != EOF);
	assert(fclose(file) == 0);
}

static void TestFillContext(SparkGlm52ResidentDecodeStageNodeContext *context,
	uint32_t stage_count,uint32_t stage_index,uint32_t first_layer,
	uint32_t layer_count,const char *pack_path)
{
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
	context->stage_count = stage_count;
	context->stage_index = stage_index;
	context->first_layer_index = first_layer;
	context->layer_count = layer_count;
	context->expert_weight_codec = GLM52_EXPERT_WEIGHT_CODEC;
	context->resident_sequence_capacity = 16u;
	context->pipeline_slot_count = 2u;
	context->max_sequence_positions = 4096u;
	context->execution_row_capacity = 4u;
	context->tp_degree = 1u;
	context->tp_rank = 0u;
	context->stage_pack_path = pack_path;
	context->model_revision = TEST_MODEL_REVISION;
	context->kv_backing_directory = "/tmp";
}

/* Drive ModuleConfigure plus the pack's header/geometry/range/inventory
 * chain exactly as PackLoad does, minus the device-region loads. */
static void TestConfigureStage(uint32_t stage_count,uint32_t stage_index,
	const char *path,struct SparkGlm52ModuleState *state)
{
	SparkGlm52ResidentDecodeStageNodeContext context;
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices services;
	const SparkGlm52ResidentDecodeStageGeometry *geometry;
	const char *pack_path = 0;
	SparkGlm52StagePackHeader header;
	SparkGlm52StagePackEntry *entries;
	FILE *file;
	uint64_t file_bytes;
	uint32_t index,local;
	SparkStatus status;

	geometry = SparkGlm52ResidentDecodeStageGeometryFor(stage_count);
	assert(geometry != 0 && geometry->stage_count == stage_count);
	TestFillContext(&context,stage_count,stage_index,
		SparkGlm52ResidentDecodeStageFirstLayer(geometry,stage_index),
		geometry->stage_layer_counts[stage_index],path);
	memset(&configuration,0,sizeof(configuration));
	configuration.model_revision = TEST_MODEL_REVISION;
	memset(&services,0,sizeof(services));
	services.node_context = &context;
	services.execution_stream = (void *)0x1;
	memset(state,0,sizeof(*state));
	assert(SparkGlm52ModuleConfigure(state,&configuration,&services,&pack_path) == SPARK_STATUS_OK);
	assert(state->geometry == geometry);
	assert(state->owns_embedding == (stage_index == 0u));
	assert(state->owns_final_head == (stage_index + 1u == geometry->stage_count));

	file = fopen(path,"rb");
	assert(file != 0);
	assert(SparkGlm52PackFileSize(file,&file_bytes) == SPARK_STATUS_OK);
	status = SparkStageModulePackRead(SPARK_GLM52_MODULE_TAG,file,0u,&header,sizeof(header));
	assert(status == SPARK_STATUS_OK);
	assert(SparkGlm52PackValidateHeader(state,&header,file_bytes) == SPARK_STATUS_OK);
	entries = calloc(header.tensor_count,sizeof(*entries));
	assert(entries != 0);
	assert(SparkStageModulePackRead(SPARK_GLM52_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(entries[0])) == SPARK_STATUS_OK);
	for (index = 0u; index < header.tensor_count; index++)
	{
		SparkGlm52StagePackTensorShape shape;
		assert(SparkGlm52PackValidateEntryGeometry(state,&header,&entries[index],&shape) == SPARK_STATUS_OK);
		SparkGlm52PackMarkSeen(state,&entries[index]);
	}
	assert(SparkGlm52PackValidateRanges(entries,header.tensor_count) == SPARK_STATUS_OK);
	assert(SparkGlm52PackValidateInventory(state) == SPARK_STATUS_OK);
	/* Every stage's expected-layer mask resolved against what the pack marked. */
	for (local = 0u; local < state->layer_count; local++)
		assert(state->layer_seen[local] == SparkGlm52ExpectedLayerMask(state,state->first_layer_index + local));
	free(entries);
	fclose(file);
}

int main(void)
{
	static struct SparkGlm52ModuleState state;
	char path[256];
	uint32_t stage;

	mkdir(TEST_PACK_DIR,0755);
	/* All seven [12,11x6] stages configure and resolve their masks. */
	for (stage = 0u; stage < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; stage++)
	{
		snprintf(path,sizeof(path),"%s/stage%u.fp8.glm52sp",TEST_PACK_DIR,stage);
		TestBuildStagePack(path,SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,stage,
			TestPp7Firsts[stage],TestPp7Counts[stage],0);
		TestConfigureStage(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,stage,path,&state);
	}
	/* The TP8 single-stage default still configures unchanged. */
	snprintf(path,sizeof(path),"%s/tp8.fp8.glm52sp",TEST_PACK_DIR);
	TestBuildStagePack(path,SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT,0u,0u,SPARK_GLM52_MODEL_LAYER_COUNT,0);
	TestConfigureStage(SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT,0u,path,&state);

	/* Negative: an unknown stage_count — the retired uniform PP13 grid — is
	 * rejected by configure, not silently accepted. */
	{
		SparkGlm52ResidentDecodeStageNodeContext context;
		SparkFirmwareModuleConfiguration configuration;
		SparkFirmwareModuleHostServices services;
		const char *pack_path = 0;
		TestFillContext(&context,13u,3u,18u,6u,"unused");
		memset(&configuration,0,sizeof(configuration));
		configuration.model_revision = TEST_MODEL_REVISION;
		memset(&services,0,sizeof(services));
		services.node_context = &context;
		services.execution_stream = (void *)0x1;
		memset(&state,0,sizeof(state));
		assert(SparkGlm52ModuleConfigure(&state,&configuration,&services,&pack_path) == SPARK_STATUS_INVALID_ARGUMENT);
	}
	/* Negative: a uniform-multiply first layer for stage 3 (3*11=33 instead
	 * of the prefix sum 34) fails the triple cross-check. */
	{
		SparkGlm52ResidentDecodeStageNodeContext context;
		SparkFirmwareModuleConfiguration configuration;
		SparkFirmwareModuleHostServices services;
		const char *pack_path = 0;
		TestFillContext(&context,7u,3u,33u,11u,"unused");
		memset(&configuration,0,sizeof(configuration));
		configuration.model_revision = TEST_MODEL_REVISION;
		memset(&services,0,sizeof(services));
		services.node_context = &context;
		services.execution_stream = (void *)0x1;
		memset(&state,0,sizeof(state));
		assert(SparkGlm52ModuleConfigure(&state,&configuration,&services,&pack_path) == SPARK_STATUS_INVALID_ARGUMENT);
	}
	/* Negative: dropping one indexer tensor from a full-indexer layer breaks
	 * the expected-mask inventory for that stage. */
	{
		SparkGlm52ResidentDecodeStageNodeContext context;
		SparkFirmwareModuleConfiguration configuration;
		SparkFirmwareModuleHostServices services;
		const SparkGlm52ResidentDecodeStageGeometry *geometry;
		const char *pack_path = 0;
		SparkGlm52StagePackHeader header;
		SparkGlm52StagePackEntry *entries;
		FILE *file;
		uint64_t file_bytes;
		uint32_t index;
		snprintf(path,sizeof(path),"%s/stage_bad.fp8.glm52sp",TEST_PACK_DIR);
		TestBuildStagePack(path,SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,1u,
			TestPp7Firsts[1],TestPp7Counts[1],1);
		geometry = SparkGlm52ResidentDecodeStageGeometryFor(7u);
		assert(geometry != 0);
		TestFillContext(&context,7u,1u,TestPp7Firsts[1],TestPp7Counts[1],path);
		memset(&configuration,0,sizeof(configuration));
		configuration.model_revision = TEST_MODEL_REVISION;
		memset(&services,0,sizeof(services));
		services.node_context = &context;
		services.execution_stream = (void *)0x1;
		memset(&state,0,sizeof(state));
		assert(SparkGlm52ModuleConfigure(&state,&configuration,&services,&pack_path) == SPARK_STATUS_OK);
		file = fopen(path,"rb");
		assert(file != 0);
		assert(SparkGlm52PackFileSize(file,&file_bytes) == SPARK_STATUS_OK);
		assert(SparkStageModulePackRead(SPARK_GLM52_MODULE_TAG,file,0u,&header,sizeof(header)) == SPARK_STATUS_OK);
		entries = calloc(header.tensor_count,sizeof(*entries));
		assert(entries != 0);
		assert(SparkStageModulePackRead(SPARK_GLM52_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(entries[0])) == SPARK_STATUS_OK);
		for (index = 0u; index < header.tensor_count; index++)
		{
			SparkGlm52StagePackTensorShape shape;
			assert(SparkGlm52PackValidateEntryGeometry(&state,&header,&entries[index],&shape) == SPARK_STATUS_OK);
			SparkGlm52PackMarkSeen(&state,&entries[index]);
		}
		assert(SparkGlm52PackValidateInventory(&state) == SPARK_STATUS_SCHEMA_ERROR);
		free(entries);
		fclose(file);
	}
	printf("PASS glm52 pp7 dry configure: 7 synthetic stages + tp8 through the module chain, masks resolve\n");
	return(0);
}

/* Link-only stubs: the dry proof never reaches the CUDA launchers, but the
 * whole module translation unit is linked, so their symbols must resolve. */
#include <cuda_runtime.h>
cudaError_t SparkGlm52LaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	(void)stream;(void)destination_bf16;(void)source_bf16;(void)row_count;(void)width;
	return(1); /* cudaErrorNotSupported under the real runtime */
}
cudaError_t SparkGlm52LaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	(void)stream;(void)destination;(void)source;(void)element_count;
	return(1); /* cudaErrorNotSupported under the real runtime */
}
int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count)
{
	(void)multiprocessor_count;
	return(-1);
}
int32_t SparkGlm52LaunchCudaWave(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
int32_t SparkGlm52LaunchCudaWaveBegin(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
int32_t SparkGlm52LaunchCudaLayerAttention(const SparkGlm52CudaWave *wave,uint32_t local_layer) { (void)wave;(void)local_layer; return(-1); }
int32_t SparkGlm52LaunchCudaLayerMlp(const SparkGlm52CudaWave *wave,uint32_t local_layer) { (void)wave;(void)local_layer; return(-1); }
int32_t SparkGlm52LaunchCudaWaveHead(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
