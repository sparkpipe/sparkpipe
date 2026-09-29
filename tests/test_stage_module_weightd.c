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

void spark_stub_cuda_fail_next_alloc(void);

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

static void TestStageAttachFailureAllocatesNothing(const char *digest)
{
	SparkStageModuleLedger ledger;
	FILE *file;
	void *pointer;
	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test_module";
	setenv("SPARK_WEIGHTD_ATTACH","1",1);
	if (digest != 0)
		setenv("SPARK_WEIGHTD_PACK_SHA256",digest,1);
	else
		unsetenv("SPARK_WEIGHTD_PACK_SHA256");
	file = fopen("/tmp/test_stage_module_weightd.pack","rb");
	assert(file != 0);
	pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,4096u,8192u,
		&pointer) != SPARK_STATUS_OK);
	assert(pointer == 0);
	assert(ledger.device_allocation_count == 0u);
	assert(ledger.device_bytes_resident == 0u);
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,4096u,8192u,
		&pointer) != SPARK_STATUS_OK);
	assert(pointer == 0 && ledger.device_allocation_count == 0u);
	SparkStageModuleLedgerRelease(&ledger);
	(void)fclose(file);
}

#define SECOND_PACK_PATH "/tmp/test_stage_module_weightd.second.pack"
#define SECOND_PACK_BYTES (64u * 1024u)

static void TestStageWriteSecondPack(uint8_t *second,char second_hex[SPARK_SHA256_HEX_BYTES])
{
	FILE *file;
	uint64_t index;
	for (index = 0u; index < SECOND_PACK_BYTES; index++)
		second[index] = (uint8_t)(index * 17u + 3u);
	file = fopen(SECOND_PACK_PATH,"wb");
	assert(file != 0);
	assert(fwrite(second,1u,SECOND_PACK_BYTES,file) == SECOND_PACK_BYTES);
	(void)fclose(file);
	assert(SparkSha256Bytes(second,SECOND_PACK_BYTES,second_hex) == SPARK_STATUS_OK);
	(void)unlink(SECOND_PACK_PATH ".sha256");
}

static void TestStageWriteSidecar(const char *digest)
{
	FILE *file = fopen(SECOND_PACK_PATH ".sha256","wb");
	assert(file != 0);
	assert(fprintf(file,"%s  test_stage_module_weightd.second.pack\n",digest) > 0);
	(void)fclose(file);
}

static void TestStageSecondPackAttach(const uint8_t *pack,const char *second_digest,SparkStatus expected)
{
	SparkStageModuleLedger ledger;
	FILE *first,*second;
	void *first_pointer = 0,*second_pointer = 0;
	uint8_t staging[4096];
	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test_module";
	first = fopen("/tmp/test_stage_module_weightd.pack","rb");
	second = fopen(SECOND_PACK_PATH,"rb");
	assert(first != 0 && second != 0);
	assert(SparkStageModuleLoadDeviceRegion(&ledger,first,4096u,4096u,&first_pointer) == SPARK_STATUS_OK);
	if (second_digest != 0)
		TestStageWriteSidecar(second_digest);
	assert(SparkStageModuleLoadDeviceRegion(&ledger,second,4096u,4096u,&second_pointer) == expected);
	assert((expected == SPARK_STATUS_OK) == (second_pointer != 0));
	assert(SparkStageModuleLoadDeviceRegion(&ledger,second,4096u,4096u,&second_pointer) == expected);
	first_pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,first,4096u,4096u,&first_pointer) == SPARK_STATUS_OK);
	assert(cudaMemcpy(staging,first_pointer,4096u,cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(memcmp(staging,pack + 4096u,4096u) == 0);
	SparkStageModuleLedgerRelease(&ledger);
	assert(ledger.pack_arena == 0);
	(void)fclose(first);
	(void)fclose(second);
	(void)unlink(SECOND_PACK_PATH ".sha256");
}

static void TestStageSecondPackMaps(const uint8_t *pack,const uint8_t *second,const char *second_hex)
{
	uint8_t staging[4096];
	SparkStageModuleLedger ledger;
	FILE *first_file,*second_file;
	void *first_pointer = 0,*second_pointer = 0;
	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test_module";
	first_file = fopen("/tmp/test_stage_module_weightd.pack","rb");
	second_file = fopen(SECOND_PACK_PATH,"rb");
	assert(first_file != 0 && second_file != 0);
	assert(SparkStageModuleLoadDeviceRegion(&ledger,first_file,4096u,4096u,&first_pointer) == SPARK_STATUS_OK);
	TestStageWriteSidecar(second_hex);
	assert(SparkStageModuleLoadDeviceRegion(&ledger,second_file,8192u,4096u,&second_pointer) == SPARK_STATUS_OK);
	assert(second_pointer != 0 && second_pointer != first_pointer);
	assert(cudaMemcpy(staging,second_pointer,4096u,cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(memcmp(staging,second + 8192u,4096u) == 0);
	assert(cudaMemcpy(staging,first_pointer,4096u,cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(memcmp(staging,pack + 4096u,4096u) == 0);
	second_pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,second_file,SECOND_PACK_BYTES - 4096u,8192u,&second_pointer) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(second_pointer == 0);
	SparkStageModuleLedgerRelease(&ledger);
	assert(ledger.pack_arena == 0);
	(void)fclose(first_file);
	(void)fclose(second_file);
}

static void TestStageSecondPack(const uint8_t *pack)
{
	uint8_t *second;
	char second_hex[SPARK_SHA256_HEX_BYTES];
	second = (uint8_t *)malloc(SECOND_PACK_BYTES);
	assert(second != 0);
	TestStageWriteSecondPack(second,second_hex);
	TestStageSecondPackAttach(pack,0,SPARK_STATUS_NOT_FOUND);
	printf("stage_module_weightd: second pack without digest sidecar refused, first arena intact PASS\n");
	TestStageSecondPackAttach(pack,"2222222222222222222222222222222222222222222222222222222222222222",SPARK_STATUS_HASH_MISMATCH);
	TestStageSecondPackAttach(pack,"not-a-digest",SPARK_STATUS_VALIDATION_FAILED);
	printf("stage_module_weightd: second pack with a wrong or malformed sidecar refused PASS\n");
	TestStageSecondPackMaps(pack,second,second_hex);
	(void)unlink(SECOND_PACK_PATH ".sha256");
	(void)unlink(SECOND_PACK_PATH);
	free(second);
	printf("stage_module_weightd: second pack maps its own arena by its sidecar digest PASS\n");
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
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,region_offset,region_bytes,&direct_pointer) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(direct_pointer == 0 && ledger.device_allocation_count == 0u && ledger.pack_arena == 0);
	unsetenv("SPARK_WEIGHTD_SOCKET");
	direct_pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,region_offset,
		region_bytes,&direct_pointer) == SPARK_STATUS_UNSUPPORTED);
	assert(direct_pointer == 0 && ledger.device_allocation_count == 0u && ledger.pack_arena == 0);
	SparkStageModuleLedgerRelease(&ledger);
	(void)fclose(file);
	printf("stage_module_weightd: conflicting attach rejected; unconfigured load refused PASS\n");
	unsetenv("SPARK_WEIGHTD_SOCKET");
	TestStageAttachFailureAllocatesNothing(sha_hex);
	setenv("SPARK_WEIGHTD_SOCKET",SOCKET_PATH,1);
	TestStageAttachFailureAllocatesNothing(0);
	TestStageAttachFailureAllocatesNothing(sha_hex);

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
		void *unaligned = 0;
		assert(SparkStageModuleLoadDeviceRegion(&ledger,file,region_offset + 1u,
			4096u,&unaligned) == SPARK_STATUS_INVALID_ARGUMENT);
		assert(unaligned == 0);
	}
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
	TestStageSecondPack(pack);
	TestStageAttachFailureAllocatesNothing("1111111111111111111111111111111111111111111111111111111111111111");
	spark_stub_cuda_fail_next_alloc();
	TestStageAttachFailureAllocatesNothing(sha_hex);

	setenv("SPARK_WEIGHTD_ATTACH","0",1);
	unsetenv("SPARK_WEIGHTD_SOCKET");
	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test_module";
	file = fopen("/tmp/test_stage_module_weightd.pack","rb");
	assert(file != 0);
	direct_pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,region_offset,
		region_bytes,&direct_pointer) == SPARK_STATUS_UNSUPPORTED);
	assert(direct_pointer == 0 && ledger.pack_arena == 0 &&
		ledger.device_allocation_count == 0u);
	SparkStageModuleLedgerRelease(&ledger);
	(void)fclose(file);
	printf("stage_module_weightd: detach refusal PASS\n");

	__atomic_store_n(&TestStageStop,1,__ATOMIC_SEQ_CST);
	pthread_join(server_thread,0);
	SparkWeightdServerDestroy(server);
	(void)unlink(SOCKET_PATH);
	free(staging);
	free(pack);
	printf("stage_module_weightd: ALL PASS\n");
	return 0;
}
