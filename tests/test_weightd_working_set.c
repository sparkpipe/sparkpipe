#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"

#define CHUNK (2u * 1024u * 1024u)
#define TIMEOUT UINT64_C(10000000000)

void spark_stub_cuda_fail_next_alloc(void);
void spark_stub_cuda_fail_alloc_after(uint32_t calls);
uint32_t spark_stub_cuda_outstanding_allocs(void);

typedef struct TestServer
{
	SparkWeightdServer *server;
	volatile sig_atomic_t stop;
} TestServer;

static void *run_server(void *data)
{
	TestServer *state = data;
	assert(SparkWeightdServerRun(state->server,&state->stop) == SPARK_STATUS_OK);
	return(0);
}

static uint64_t range_offset(uint32_t expert,uint32_t kind)
{
	if ( expert < 2u )
		return(((kind / 2u) * CHUNK) + (expert * 256u) + ((kind % 2u) * 64u));
	return((2u * CHUNK) + ((expert - 2u) * 256u) + (kind * 64u));
}

static void write_range(FILE *pack,FILE *manifest,uint32_t expert,uint32_t kind)
{
	uint8_t record[48] = {0},data[64],digest[16];
	uint64_t offset = range_offset(expert,kind),bytes = sizeof(data);
	SparkCk128Context ck;
	memset(data,(int32_t)(1u + (expert * 4u) + kind),sizeof(data));
	assert(fseek(pack,(long)offset,SEEK_SET) == 0);
	assert(fwrite(data,1u,sizeof(data),pack) == sizeof(data));
	SparkCk128Initialize(&ck);
	SparkCk128Update(&ck,data,sizeof(data));
	SparkCk128Finalize(&ck,digest);
	if ( expert == 3u && kind == 3u )
		digest[0] ^= 1u;
	memcpy(record + 4u,&expert,4u);
	memcpy(record + 8u,&kind,4u);
	memcpy(record + 16u,&offset,8u);
	memcpy(record + 24u,&bytes,8u);
	memcpy(record + 32u,digest,16u);
	assert(fwrite(record,1u,sizeof(record),manifest) == sizeof(record));
}

static void write_fixture(const char *path,const char *manifest_path)
{
	FILE *pack = fopen(path,"wb"),*manifest = fopen(manifest_path,"wb");
	uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,16u,0u};
	uint32_t expert,kind;
	assert(pack != 0 && manifest != 0);
	assert(ftruncate(fileno(pack),(3u * CHUNK)) == 0);
	assert(fwrite(header,1u,sizeof(header),manifest) == sizeof(header));
	for (expert=0u; expert<4u; expert++)
		for (kind=0u; kind<4u; kind++)
			write_range(pack,manifest,expert,kind);
	assert(fclose(pack) == 0);
	assert(fclose(manifest) == 0);
}

static uint64_t attach(SparkWeightdClient *client,const char *path,uint64_t *base)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyAttachResult result;
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (3u * CHUNK);
	memcpy(request.identity.model,"working-set-test",17u);
	memset(request.identity.pack_sha256,'a',64u);
	assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	request.expert_pool_bytes = (2u * CHUNK);
	assert(SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT) == SPARK_STATUS_OK);
	assert(result.expert_count == 4u);
	*base = result.device_handle;
	return(result.arena_generation);
}

static SparkWeightdWorkingSetResult acquire(SparkWeightdClient *client,uint64_t generation,uint32_t expert,SparkStatus expected)
{
	SparkWeightdExpertKey key = {0u,expert};
	SparkWeightdWorkingSetResult result;
	assert(SparkWeightdClientAcquire(client,generation,&key,1u,&result,TIMEOUT) == expected);
	assert(result.status == expected);
	assert((result.lease_identifier != 0u) == (expected == SPARK_STATUS_OK));
	return(result);
}

static void release(SparkWeightdClient *client,uint64_t generation,uint64_t lease)
{
	SparkWeightdWorkingSetResult result;
	assert(SparkWeightdClientRelease(client,generation,lease,&result,TIMEOUT) == SPARK_STATUS_OK);
}

static void check_ranges(uint64_t base,uint32_t expert)
{
	const uint8_t *data;
	uint32_t kind,i;
	for (kind=0u; kind<4u; kind++)
	{
		data = (const uint8_t *)(uintptr_t)(base + range_offset(expert,kind));
		for (i=0u; i<64u; i++)
			assert(data[i] == (1u + (expert * 4u) + kind));
	}
}

static void check_two_clients(SparkWeightdClient *a,SparkWeightdClient *b,uint64_t generation,uint64_t base)
{
	SparkWeightdWorkingSetResult first,second,result;
	SparkWeightdDetachResult detached;
	uint32_t allocations = spark_stub_cuda_outstanding_allocs();
	spark_stub_cuda_fail_next_alloc();
	result = acquire(a,generation,0u,SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(result.resident_bytes == 0u);
	spark_stub_cuda_fail_alloc_after(2u);
	result = acquire(a,generation,0u,SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(result.resident_bytes == 0u);
	assert(spark_stub_cuda_outstanding_allocs() == allocations);
	first = acquire(a,generation,0u,SPARK_STATUS_OK);
	assert(SparkWeightdClientDetach(a,generation,&detached,TIMEOUT) == SPARK_STATUS_OK);
	assert(detached.status == SPARK_STATUS_BUSY);
	assert(first.resident_bytes == (2u * CHUNK));
	check_ranges(base,0u);
	second = acquire(b,generation,1u,SPARK_STATUS_OK);
	assert(second.resident_bytes == first.resident_bytes);
	check_ranges(base,1u);
	assert(SparkWeightdClientRelease(b,generation,first.lease_identifier,&result,TIMEOUT) == SPARK_STATUS_NOT_FOUND);
	(void)acquire(b,generation,2u,SPARK_STATUS_CAPACITY_EXCEEDED);
	check_ranges(base,0u);
	check_ranges(base,1u);
	release(a,generation,first.lease_identifier);
	(void)acquire(a,generation,2u,SPARK_STATUS_CAPACITY_EXCEEDED);
	release(b,generation,second.lease_identifier);
	result = acquire(a,generation,2u,SPARK_STATUS_OK);
	check_ranges(base,2u);
	release(a,generation,result.lease_identifier);
}

static void check_transaction(SparkWeightdClient *client,uint64_t generation,uint64_t base)
{
	SparkWeightdExpertKey keys[3] = {{0u,0u},{0u,1u},{0u,0u}};
	SparkWeightdWorkingSetResult result;
	assert(SparkWeightdClientAcquire(client,generation,keys,3u,&result,TIMEOUT) == SPARK_STATUS_OK);
	assert(result.resident_bytes == (2u * CHUNK));
	check_ranges(base,0u);
	check_ranges(base,1u);
	release(client,generation,result.lease_identifier);
	(void)acquire(client,generation,3u,SPARK_STATUS_HASH_MISMATCH);
	result = acquire(client,generation,2u,SPARK_STATUS_OK);
	assert(result.resident_bytes <= (2u * CHUNK));
	check_ranges(base,2u);
	release(client,generation,result.lease_identifier);
}

static void check_orphan(SparkWeightdClient *a,SparkWeightdClient *b,uint64_t generation,uint64_t base,const char *socket_path,const char *path)
{
	SparkWeightdWorkingSetResult pinned,result;
	SparkWeightdClient *replacement;
	uint64_t other_base;
	pinned = acquire(a,generation,0u,SPARK_STATUS_OK);
	SparkWeightdClientClose(a);
	assert(SparkWeightdClientConnect(socket_path,&replacement,0) == SPARK_STATUS_OK);
	assert(attach(replacement,path,&other_base) == generation);
	assert(SparkWeightdClientRelease(replacement,generation,pinned.lease_identifier,&result,TIMEOUT) == SPARK_STATUS_NOT_FOUND);
	(void)acquire(b,generation,2u,SPARK_STATUS_CAPACITY_EXCEEDED);
	check_ranges(base,0u);
	SparkWeightdClientClose(replacement);
}

int main(void)
{
	char root[] = "/tmp/weightd-set-XXXXXX",path[256],manifest[272],socket_path[256];
	TestServer state = {0};
	SparkWeightdServerConfig config = {0};
	SparkWeightdClient *a,*b;
	pthread_t thread;
	uint64_t generation,base,other_base;
	assert(mkdtemp(root) != 0);
	snprintf(path,sizeof(path),"%s/pack",root);
	snprintf(manifest,sizeof(manifest),"%s.experts",path);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	write_fixture(path,manifest);
	config.socket_path = socket_path;
	config.device_bytes_max = (4u * CHUNK);
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	assert(SparkWeightdClientConnect(socket_path,&a,0) == SPARK_STATUS_OK);
	assert(SparkWeightdClientConnect(socket_path,&b,0) == SPARK_STATUS_OK);
	generation = attach(a,path,&base);
	assert(attach(b,path,&other_base) == generation && other_base == base);
	check_two_clients(a,b,generation,base);
	check_transaction(a,generation,base);
	check_orphan(a,b,generation,base,socket_path,path);
	SparkWeightdClientClose(b);
	__atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	assert(spark_stub_cuda_outstanding_allocs() == 0u);
	assert(unlink(manifest) == 0 && unlink(path) == 0 && rmdir(root) == 0);
	puts("PASS working-set IPC: all ranges, shared chunks, owner leases, capacity and corruption");
	return(0);
}
