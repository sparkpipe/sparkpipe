#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"

#define PACK_BYTES (256u * 1024u)
#define SOCKET_PATH "/tmp/test_stage_module_weightd.sock"

static void TestStageFillPack(uint8_t *buffer, uint64_t bytes)
{
	uint64_t index;
	for (index = 0u; index < bytes; index++)
		buffer[index] = (uint8_t)(index * 31u + 7u);
}

static volatile sig_atomic_t TestStageStop = 0;

static void *TestStageServerThread(void *argument)
{
	SparkWeightdServer *server = (SparkWeightdServer *)argument;
	(void)SparkWeightdServerRun(server,&TestStageStop);
	return 0;
}

int main(void)
{
	FILE *file;
	uint8_t *pack;
	uint8_t *staging;
	SparkStageModuleLedger ledger;
	uint64_t region_offset, region_bytes;
	void *direct_pointer, *arena_pointer;
	char arena_digest[SPARK_SHA256_HEX_BYTES + 1u];
	char sha_hex[SPARK_SHA256_HEX_BYTES + 1u];
	pthread_t server_thread;
	SparkWeightdServer *server;
	SparkWeightdServerConfig server_config;

	signal(SIGPIPE,SIG_IGN);
	(void)unlink(SOCKET_PATH);

	pack = (uint8_t *)malloc(PACK_BYTES);
	assert(pack != 0);
	TestStageFillPack(pack,PACK_BYTES);
	file = fopen("/tmp/test_stage_module_weightd.pack","wb");
	assert(file != 0);
	assert(fwrite(pack,1u,PACK_BYTES,file) == PACK_BYTES);
	(void)fclose(file);

	assert(SparkSha256Bytes(pack,PACK_BYTES,sha_hex) == SPARK_STATUS_OK);

	staging = (uint8_t *)malloc(PACK_BYTES);
	assert(staging != 0);

	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test_module";
	setenv("SPARK_WEIGHTD_SOCKET",SOCKET_PATH,1);
	setenv("SPARK_WEIGHTD_ATTACH","0",1);
	file = fopen("/tmp/test_stage_module_weightd.pack","rb");
	assert(file != 0);
	region_offset = 4096u;
	region_bytes = 8192u;
	direct_pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,region_offset,
		region_bytes,&direct_pointer) == SPARK_STATUS_OK);
	assert(direct_pointer != 0);
	assert(ledger.pack_arena == 0);
	assert(cudaMemcpy(staging,direct_pointer,region_bytes,
		cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(memcmp(staging,pack + region_offset,region_bytes) == 0);
	SparkStageModuleLedgerRelease(&ledger);
	(void)fclose(file);
	printf("stage_module_weightd: no-daemon fallback byte-identical PASS\n");

	memset(&server_config,0,sizeof(server_config));
	server_config.socket_path = SOCKET_PATH;
	server_config.device_bytes_max = 8ull << 30;
	assert(SparkWeightdServerCreate(&server_config,&server) ==
		SPARK_STATUS_OK);
	assert(pthread_create(&server_thread,0,TestStageServerThread,server) == 0);

	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test_module";
	setenv("SPARK_WEIGHTD_PACK_SHA256",sha_hex,1);
	setenv("SPARK_WEIGHTD_ATTACH","1",1);
	file = fopen("/tmp/test_stage_module_weightd.pack","rb");
	assert(file != 0);
	arena_pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,region_offset,
		region_bytes,&arena_pointer) == SPARK_STATUS_OK);
	assert(arena_pointer != 0);
	assert(ledger.pack_arena != 0);
	assert(cudaMemcpy(staging,arena_pointer,region_bytes,
		cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(memcmp(staging,pack + region_offset,region_bytes) == 0);
	{
		void *second = 0;
		uint64_t second_offset = 128u * 1024u;
		assert(SparkStageModuleLoadDeviceRegion(&ledger,file,second_offset,
			4096u,&second) == SPARK_STATUS_OK);
		assert(second != 0 && second != arena_pointer);
		assert((uint8_t *)second - (uint8_t *)arena_pointer ==
			(int64_t)second_offset - (int64_t)region_offset);
	}
	{
		void *base = 0;
		assert(SparkStageModuleLoadDeviceRegion(&ledger,file,0u,PACK_BYTES,
			&base) == SPARK_STATUS_OK);
		assert(base != 0);
		assert(cudaMemcpy(staging,base,PACK_BYTES,cudaMemcpyDeviceToHost) ==
			cudaSuccess);
		assert(SparkSha256Bytes(staging,PACK_BYTES,arena_digest) ==
			SPARK_STATUS_OK);
		assert(strcmp((const char *)arena_digest,sha_hex) == 0);
	}
	SparkStageModuleLedgerRelease(&ledger);
	(void)fclose(file);
	printf("stage_module_weightd: arena slice zero-copy digest-exact PASS\n");

	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test_module";
	file = fopen("/tmp/test_stage_module_weightd.pack","rb");
	assert(file != 0);
	arena_pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,region_offset,
		region_bytes,&arena_pointer) == SPARK_STATUS_OK);
	assert(ledger.pack_arena != 0);
	SparkStageModuleLedgerRelease(&ledger);
	(void)fclose(file);
	printf("stage_module_weightd: reattach after release PASS\n");

	setenv("SPARK_WEIGHTD_ATTACH","0",1);
	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test_module";
	file = fopen("/tmp/test_stage_module_weightd.pack","rb");
	assert(file != 0);
	direct_pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,region_offset,
		region_bytes,&direct_pointer) == SPARK_STATUS_OK);
	assert(ledger.pack_arena == 0);
	assert(cudaMemcpy(staging,direct_pointer,region_bytes,
		cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(memcmp(staging,pack + region_offset,region_bytes) == 0);
	SparkStageModuleLedgerRelease(&ledger);
	(void)fclose(file);
	printf("stage_module_weightd: kill-switch direct path PASS\n");

	__atomic_store_n(&TestStageStop,1,__ATOMIC_SEQ_CST);
	pthread_join(server_thread,0);
	SparkWeightdServerDestroy(server);
	(void)unlink(SOCKET_PATH);
	free(staging);
	free(pack);
	printf("stage_module_weightd: ALL PASS\n");
	return 0;
}
