#undef NDEBUG
#define GLM5_NEXT_EXPERT_WEIGHT_CODEC 5
#define GLM5_NEXT_EXPERT_CODEC_NAME "fp8"
#define GLM5_NEXT_MODEL_REVISION "test"
#define GLM5_NEXT_CONTRACT_SHA256 "test"
#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include "../modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"
#include "sparkpipe/spark_weightd_map.h"

#ifndef TEST_EXPERTS_MANIFEST_GENERATOR
#define TEST_EXPERTS_MANIFEST_GENERATOR "build/glm5_next_experts_manifest"
#endif

#define TEST_LAYER 3u
#define TEST_EXPERTS SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT
#define TEST_CHUNK (2u * 1024u * 1024u)
#define TEST_TIMEOUT UINT64_C(10000000000)

uint32_t spark_stub_cuda_outstanding_allocs(void);

typedef struct TestServer
{
	SparkWeightdServer *server;
	volatile sig_atomic_t stop;
} TestServer;

typedef struct TestPack
{
	SparkGlm5NextStagePackHeader header;
	SparkGlm5NextStagePackEntry entries[2];
	uint8_t *bytes;
	uint64_t size;
} TestPack;

static void *run_server(void *data)
{
	TestServer *state = data;
	assert(SparkWeightdServerRun(state->server,&state->stop) == SPARK_STATUS_OK);
	return(0);
}

static uint64_t align_up(uint64_t value)
{
	return((value + 255u) & ~UINT64_C(255));
}

static void build_pack(TestPack *pack,uint32_t codec,uint64_t payload,uint64_t block,uint32_t salt)
{
	uint64_t cursor,index,scale;
	uint32_t tensor,state = 0x9e3779b9u + codec + (salt * 0x85ebca6bu);
	memset(pack,0,sizeof(*pack));
	pack->header.magic = SPARK_GLM5_NEXT_STAGEPACK_MAGIC;
	pack->header.format_version = SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION;
	pack->header.header_bytes = sizeof(pack->header);
	pack->header.directory_entry_bytes = sizeof(pack->entries[0]);
	pack->header.tensor_count = 2u;
	pack->header.first_layer_index = TEST_LAYER;
	pack->header.layer_count = 1u;
	pack->header.routed_expert_count = TEST_EXPERTS;
	pack->header.expert_weight_codec = codec;
	pack->header.directory_offset = 512u;
	cursor = 4096u;
	for (tensor=0u; tensor<2u; tensor++)
	{
		SparkGlm5NextStagePackEntry *entry = &pack->entries[tensor];
		entry->tensor_kind = SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE + tensor;
		entry->layer_index = TEST_LAYER;
		entry->weight_codec = codec;
		entry->scale_encoding = SparkWeightCodecScaleEncoding(codec);
		entry->group_count = TEST_EXPERTS;
		entry->payload_offset = cursor;
		entry->payload_bytes = (uint64_t)TEST_EXPERTS * payload;
		cursor = align_up(cursor + entry->payload_bytes);
		scale = block == 0u ? 0u : (uint64_t)TEST_EXPERTS * (block + (codec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 ? 4u : 0u));
		entry->scale_offset = scale == 0u ? 0u : cursor;
		entry->scale_bytes = scale;
		cursor = align_up(cursor + scale);
	}
	pack->size = cursor + (3u * TEST_CHUNK);
	pack->header.file_bytes = pack->size;
	pack->bytes = malloc(pack->size);
	assert(pack->bytes != 0);
	for (index=0u; index<pack->size; index++)
	{
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;
		pack->bytes[index] = (uint8_t)state;
	}
	memcpy(pack->bytes,&pack->header,sizeof(pack->header));
	memcpy(pack->bytes + pack->header.directory_offset,pack->entries,sizeof(pack->entries));
}

static void write_pack(const TestPack *pack,const char *path)
{
	char command[1024];
	FILE *file = fopen(path,"wb");
	assert(file != 0 && fwrite(pack->bytes,1u,pack->size,file) == pack->size && fclose(file) == 0);
	assert(snprintf(command,sizeof(command),"%s %s >/dev/null",TEST_EXPERTS_MANIFEST_GENERATOR,path) < (int)sizeof(command));
	assert(system(command) == 0);
}

static void check_resident(const TestPack *pack,const uint8_t *base)
{
	SparkExpertPlanes planes;
	const SparkGlm5NextStagePackEntry *entry;
	const uint8_t *scale;
	uint64_t globals,block;
	uint32_t tensor,plane,expert;
	for (tensor=0u; tensor<2u; tensor++)
	{
		entry = &pack->entries[tensor];
		assert(SparkExpertPlanesDescribe(entry->tensor_kind,entry->weight_codec,entry->scale_encoding,entry->group_count,entry->payload_offset,entry->payload_bytes,entry->scale_offset,entry->scale_bytes,&planes) == SPARK_STATUS_OK);
		for (plane=0u; plane<planes.count; plane++)
			for (expert=0u; expert<TEST_EXPERTS; expert++)
			{
				uint64_t offset = SparkExpertPlaneOffset(&planes.planes[plane],expert);
				assert(memcmp(base + offset,pack->bytes + offset,planes.planes[plane].bytes) == 0);
			}
		if ( entry->weight_codec != SPARK_WEIGHT_CODEC_NVFP4_E2M1 )
			continue;
		scale = base + entry->scale_offset;
		globals = (uint64_t)TEST_EXPERTS * 4u;
		block = (entry->scale_bytes - globals) / TEST_EXPERTS;
		for (expert=0u; expert<TEST_EXPERTS; expert++)
		{
			assert(memcmp(scale + (uint64_t)expert * 4u,pack->bytes + entry->scale_offset + (uint64_t)expert * 4u,4u) == 0);
			assert(memcmp(scale + globals + (uint64_t)expert * block,pack->bytes + entry->scale_offset + globals + (uint64_t)expert * block,block) == 0);
		}
	}
}

static uint32_t count_planes(const TestPack *pack)
{
	SparkExpertPlanes planes;
	uint32_t tensor,total = 0u;
	for (tensor=0u; tensor<2u; tensor++)
	{
		const SparkGlm5NextStagePackEntry *entry = &pack->entries[tensor];
		assert(SparkExpertPlanesDescribe(entry->tensor_kind,entry->weight_codec,entry->scale_encoding,entry->group_count,entry->payload_offset,entry->payload_bytes,entry->scale_offset,entry->scale_bytes,&planes) == SPARK_STATUS_OK);
		total += planes.count;
	}
	return(total);
}

static void check_codec(const char *root,const char *socket_path,uint32_t codec,uint64_t payload,uint64_t block,uint32_t single)
{
	char pool[32];
	const uint8_t *resident;
	SparkGlm5NextModuleState *state = calloc(1u,sizeof(*state));
	SparkWeightdManifest manifest;
	TestPack pack;
	char path[512],manifest_path[600],digest[SPARK_SHA256_HEX_BYTES];
	assert(state != 0);
	build_pack(&pack,codec,payload,block,single);
	snprintf(path,sizeof(path),"%s/codec%u_%u.sp",root,codec,single);
	snprintf(manifest_path,sizeof(manifest_path),"%s.experts",path);
	write_pack(&pack,path);
	assert(SparkWeightdManifestLoad(manifest_path,pack.size,&manifest) == SPARK_STATUS_OK);
	assert(manifest.range_count == TEST_EXPERTS * count_planes(&pack));
	SparkWeightdManifestDestroy(&manifest);
	assert(SparkSha256File(path,digest) == SPARK_STATUS_OK);
	assert(setenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256,digest,1) == 0);
	assert(setenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET,socket_path,1) == 0);
	snprintf(pool,sizeof(pool),"%llu",(unsigned long long)(single != 0u ? pack.size + TEST_CHUNK : pack.size));
	assert(setenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES",pool,1) == 0);
	assert(setenv("SPARK_WEIGHTD_SPINE_BUDGET_BYTES","8388608",1) == 0);
	state->pin_experts = 1u;
	strcpy(state->model_revision,"codec-test");
	state->tp_degree = 1u;
	state->first_layer_index = TEST_LAYER;
	state->layer_count = 1u;
	state->execution_stream = 0;
	assert(SparkGlm5NextLazyOpen(state,path,pack.size,pack.entries,2u) == SPARK_STATUS_OK);
	assert(state->lazy_pack != 0 && state->lazy_pack->map != 0);
	assert(state->expert_pin_key_count == TEST_EXPERTS && state->expert_pin_lease_count != 0u);
	assert(state->decode_lease_base_saved != 0);
	resident = single != 0u ? (const uint8_t *)(uintptr_t)state->lazy_pack->attached.device_handle : state->decode_lease_base_saved;
	check_resident(&pack,resident);
	assert(SparkGlm5NextReleasePinnedAbove(state,0u) == SPARK_STATUS_OK);
	assert(SparkWeightdLazyPackDestroy(state->lazy_pack) == SPARK_STATUS_OK);
	fflush(stdout);
	printf("codec=%u pool=%s ranges_per_expert=%u pinned=%u resident planes verified\n",codec,single != 0u ? "single-alloc" : "per-chunk",count_planes(&pack),TEST_EXPERTS);
	free(pack.bytes);
	free(state);
}

static void check_layout_mismatch_refused(const char *root)
{
	SparkGlm5NextStagePackEntry claimed[2];
	SparkWeightdManifest manifest;
	SparkGlm5NextManifestContext context = {claimed,2u};
	TestPack pack;
	char path[512],manifest_path[600];
	uint32_t tensor;
	build_pack(&pack,SPARK_WEIGHT_CODEC_NVFP4_E2M1,64u,32u,7u);
	snprintf(path,sizeof(path),"%s/layout.sp",root);
	snprintf(manifest_path,sizeof(manifest_path),"%s.experts",path);
	write_pack(&pack,path);
	assert(SparkWeightdManifestLoad(manifest_path,pack.size,&manifest) == SPARK_STATUS_OK);
	memcpy(claimed,pack.entries,sizeof(claimed));
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_OK);
	for (tensor=0u; tensor<2u; tensor++)
	{
		claimed[tensor].weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
		claimed[tensor].scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_F32;
	}
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
	memcpy(claimed,pack.entries,sizeof(claimed));
	claimed[1].weight_codec = SPARK_WEIGHT_CODEC_BF16;
	claimed[1].scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
	claimed[1].scale_offset = 0u;
	claimed[1].scale_bytes = 0u;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
	SparkWeightdManifestDestroy(&manifest);
	free(pack.bytes);
}

int main(void)
{
	char root[] = "/tmp/codec-attach-XXXXXX",socket_path[256];
	TestServer server = {0};
	SparkWeightdServerConfig config = {0};
	pthread_t thread;
	assert(mkdtemp(root) != 0);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	config.socket_path = socket_path;
	config.device_bytes_max = (64u * TEST_CHUNK);
	assert(SparkWeightdServerCreate(&config,&server.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&server) == 0);
	for (uint32_t single=0u; single<2u; single++)
	{
		check_codec(root,socket_path,SPARK_WEIGHT_CODEC_FP8_E4M3,256u,8u,single);
		check_codec(root,socket_path,SPARK_WEIGHT_CODEC_BF16,512u,0u,single);
		check_codec(root,socket_path,SPARK_WEIGHT_CODEC_NVFP4_E2M1,128u,16u,single);
	}
	check_layout_mismatch_refused(root);
	__atomic_store_n(&server.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(server.server);
	assert(spark_stub_cuda_outstanding_allocs() == 0u);
	{
		char command[512];
		assert(snprintf(command,sizeof(command),"rm -rf %s",root) < (int)sizeof(command));
		assert(system(command) == 0);
	}
	puts("PASS GLM codec lazy attach: fp8/bf16/nvfp4 synthetic packs, generator manifest, weightd attach, PIN_EXPERTS, every plane resident at the kernel address");
	return(0);
}
