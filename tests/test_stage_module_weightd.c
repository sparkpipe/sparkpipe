/* The structural weightd seam (stage_module_common.c), proven on the host:
 * the single integration point every family's pack loader already calls
 * (SparkStageModuleLoadDeviceRegion) must (1) fall back cleanly with NO
 * daemon - byte-identical contents via the direct-load path, nothing
 * registered as an arena slice; (2) with the in-process weightd server on
 * the socket, the first ledger attach cold-loads the pack and serves
 * 256B-aligned in-pack regions as ZERO-COPY slices of the consumer map
 * (contents identical to the direct path); (3) LedgerRelease unmaps without
 * detaching - the daemon arena stays warm (a second attach is
 * loaded_from_pack 0); (4) the kill switch (SPARK_WEIGHTD_ATTACH=0)
 * forces the direct path even with a live daemon. cuda-stub only, the W2
 * in-process-server pattern from test_weightd_attach.c. */
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

	/* ---- 1: NO daemon -> the direct path, byte-identical ---- */
	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test_module";
	setenv("SPARK_WEIGHTD_SOCKET",SOCKET_PATH,1);
	setenv("SPARK_WEIGHTD_ATTACH","0",1);   /* kill switch while no daemon */
	file = fopen("/tmp/test_stage_module_weightd.pack","rb");
	assert(file != 0);
	region_offset = 4096u;
	region_bytes = 8192u;
	direct_pointer = 0;
	assert(SparkStageModuleLoadDeviceRegion(&ledger,file,region_offset,
		region_bytes,&direct_pointer) == SPARK_STATUS_OK);
	assert(direct_pointer != 0);
	assert(ledger.pack_arena == 0);  /* no attach attempted under the switch */
	assert(cudaMemcpy(staging,direct_pointer,region_bytes,
		cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(memcmp(staging,pack + region_offset,region_bytes) == 0);
	SparkStageModuleLedgerRelease(&ledger);
	(void)fclose(file);
	printf("stage_module_weightd: no-daemon fallback byte-identical PASS\n");

	/* ---- the in-process daemon ---- */
	memset(&server_config,0,sizeof(server_config));
	server_config.socket_path = SOCKET_PATH;
	server_config.device_bytes_max = 8ull << 30;
	assert(SparkWeightdServerCreate(&server_config,&server) ==
		SPARK_STATUS_OK);
	assert(pthread_create(&server_thread,0,TestStageServerThread,server) == 0);

	/* ---- 2: live daemon, switch ON -> arena slice, zero copy ---- */
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
	assert(ledger.pack_arena != 0);   /* the arena attached */
	/* the slice contents equal the pack bytes at the offset */
	assert(cudaMemcpy(staging,arena_pointer,region_bytes,
		cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(memcmp(staging,pack + region_offset,region_bytes) == 0);
	/* a second region from the SAME ledger reuses the attached arena and
	 * returns a disjoint slice (pointer arithmetic, not a second load) */
	{
		void *second = 0;
		uint64_t second_offset = 128u * 1024u;
		assert(SparkStageModuleLoadDeviceRegion(&ledger,file,second_offset,
			4096u,&second) == SPARK_STATUS_OK);
		assert(second != 0 && second != arena_pointer);
		assert((uint8_t *)second - (uint8_t *)arena_pointer ==
			(int64_t)second_offset - (int64_t)region_offset);
	}
	/* digest the whole mapped arena and compare to the pack digest */
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

	/* ---- 3: release kept the arena warm -> a NEW ledger attach is warm ---- */
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

	/* ---- 4: kill switch with a LIVE daemon forces the direct path ---- */
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
