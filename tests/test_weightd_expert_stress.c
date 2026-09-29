#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "cuda.h"
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_sha256.h"

#define CHUNK (2u * 1024u * 1024u)
#define EXPERTS 4u
#define POOL_CHUNKS 2u
#define TIMEOUT UINT64_C(10000000000)

void spark_stub_cuda_set_create_delay(uint32_t delay);
uint32_t spark_stub_cuda_outstanding_allocs(void);

static uint32_t test_failures;

#define CHECK(cond, name) do { \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)

typedef struct TestServer
{
	SparkWeightdServer *server;
	volatile sig_atomic_t stop;
	pthread_t thread;
	char root[64];
	char path[256];
	char manifest[272];
	char recording[272];
	char socket_path[256];
} TestServer;

typedef struct ConcurrentAcquire
{
	SparkWeightdClient *client;
	uint64_t generation;
	uint32_t *arrived;
	SparkStatus status;
	SparkWeightdWorkingSetResult result;
} ConcurrentAcquire;

static void *run_server(void *data)
{
	TestServer *state = data;
	assert(SparkWeightdServerRun(state->server,&state->stop) == SPARK_STATUS_OK);
	return(0);
}

static void write_fixture(const char *path,const char *manifest_path)
{
	FILE *pack = fopen(path,"wb"),*manifest = fopen(manifest_path,"wb");
	uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,EXPERTS,0u},expert;
	uint8_t *data = malloc(CHUNK);
	assert(pack != 0 && manifest != 0 && data != 0);
	assert(fwrite(header,1u,sizeof(header),manifest) == sizeof(header));
	for (expert=0u; expert<EXPERTS; expert++)
	{
		uint8_t record[48] = {0},digest[16];
		uint64_t offset = (uint64_t)expert * CHUNK,bytes = CHUNK;
		SparkCk128Context ck;
		memset(data,(int)(expert + 1u),CHUNK);
		SparkCk128Initialize(&ck);
		SparkCk128Update(&ck,data,CHUNK);
		SparkCk128Finalize(&ck,digest);
		memcpy(record + 4u,&expert,4u);
		memcpy(record + 16u,&offset,8u);
		memcpy(record + 24u,&bytes,8u);
		memcpy(record + 32u,digest,16u);
		assert(fwrite(record,1u,sizeof(record),manifest) == sizeof(record));
		assert(fwrite(data,1u,CHUNK,pack) == CHUNK);
	}
	free(data);
	assert(fclose(pack) == 0 && fclose(manifest) == 0);
}

static void server_start(TestServer *state)
{
	SparkWeightdServerConfig config = {0};
	memset(state,0,sizeof(*state));
	snprintf(state->root,sizeof(state->root),"/tmp/weightd-expert-stress-XXXXXX");
	assert(mkdtemp(state->root) != 0);
	snprintf(state->path,sizeof(state->path),"%s/pack",state->root);
	snprintf(state->manifest,sizeof(state->manifest),"%s.experts",state->path);
	snprintf(state->recording,sizeof(state->recording),"%s.wset",state->path);
	snprintf(state->socket_path,sizeof(state->socket_path),"%s/socket",state->root);
	write_fixture(state->path,state->manifest);
	config.socket_path = state->socket_path;
	config.device_bytes_max = (POOL_CHUNKS + 1u) * CHUNK;
	assert(SparkWeightdServerCreate(&config,&state->server) == SPARK_STATUS_OK);
	assert(pthread_create(&state->thread,0,run_server,state) == 0);
}

static void server_stop(TestServer *state,const char *name)
{
	__atomic_store_n(&state->stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(state->thread,0) == 0);
	SparkWeightdServerDestroy(state->server);
	CHECK( spark_stub_cuda_outstanding_allocs() == 0u,name);
	(void)unlink(state->recording);
	assert(unlink(state->manifest) == 0 && unlink(state->path) == 0 && rmdir(state->root) == 0);
}

static SparkStatus attach(SparkWeightdClient *client,const char *path,uint64_t *generation)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyAttachResult result;
	SparkStatus status;
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (uint64_t)EXPERTS * CHUNK;
	memcpy(request.identity.model,"expert-stress",14u);
	assert(SparkSha256File(path,request.identity.pack_sha256) == SPARK_STATUS_OK);
	assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	request.expert_pool_bytes = (uint64_t)POOL_CHUNKS * CHUNK;
	status = SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT);
	if ( status == SPARK_STATUS_OK && (result.status != SPARK_STATUS_OK || result.expert_count != EXPERTS) )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && generation != 0 )
		*generation = result.arena_generation;
	return(status);
}

static SparkStatus acquire(SparkWeightdClient *client,uint64_t generation,const uint32_t *experts,uint32_t count,SparkWeightdWorkingSetResult *result)
{
	SparkWeightdExpertKey keys[EXPERTS];
	uint32_t index;
	assert(count <= EXPERTS);
	for (index=0u; index<count; index++)
	{
		keys[index].layer = 0u;
		keys[index].expert = experts[index];
	}
	memset(result,0,sizeof(*result));
	return(SparkWeightdClientAcquire(client,generation,keys,count,result,TIMEOUT));
}

static void release(SparkWeightdClient *client,uint64_t generation,uint64_t lease)
{
	SparkWeightdWorkingSetResult result;
	assert(SparkWeightdClientRelease(client,generation,lease,&result,TIMEOUT) == SPARK_STATUS_OK);
	assert(result.status == SPARK_STATUS_OK);
}

static int lease_serves(SparkWeightdClient *client,uint64_t generation,uint64_t lease,const uint32_t *experts,uint32_t count)
{
	SparkWeightdExportBatch batch;
	CUmemGenericAllocationHandle handles[EXPERTS];
	CUmemAccessDesc access = {0};
	CUdeviceptr base;
	uint8_t *expected = malloc(CHUNK);
	uint32_t index,found,exact = 1u;
	assert(expected != 0);
	memset(&batch,0,sizeof(batch));
	assert(SparkWeightdClientExportLeaseBatch(client,generation,lease,0u,&batch,TIMEOUT) == SPARK_STATUS_OK);
	if ( batch.status != SPARK_STATUS_OK || batch.batch_count != count || batch.lease_chunk_count != count || batch.chunk_bytes != CHUNK )
	{
		for (index=0u; index<batch.batch_count; index++)
			(void)close(batch.fds[index]);
		free(expected);
		return(0);
	}
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.location.id = 0;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
	assert(cuMemAddressReserve(&base,(size_t)EXPERTS * CHUNK,0u,0u,0u) == CUDA_SUCCESS);
	for (index=0u; index<count; index++)
	{
		assert(batch.chunk_indices[index] < EXPERTS);
		assert(cuMemImportFromShareableHandle(&handles[index],(void *)(intptr_t)batch.fds[index],CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS);
		assert(close(batch.fds[index]) == 0);
		assert(cuMemMap(base + ((CUdeviceptr)batch.chunk_indices[index] * CHUNK),CHUNK,0u,handles[index],0u) == CUDA_SUCCESS);
		assert(cuMemSetAccess(base + ((CUdeviceptr)batch.chunk_indices[index] * CHUNK),CHUNK,&access,1u) == CUDA_SUCCESS);
	}
	for (index=0u; index<count; index++)
	{
		for (found=0u; found<count && batch.chunk_indices[found] != experts[index]; found++)
			;
		memset(expected,(int)(experts[index] + 1u),CHUNK);
		if ( found == count || memcmp((const void *)(uintptr_t)(base + ((CUdeviceptr)experts[index] * CHUNK)),expected,CHUNK) != 0 )
			exact = 0u;
	}
	for (index=0u; index<count; index++)
	{
		assert(cuMemUnmap(base + ((CUdeviceptr)batch.chunk_indices[index] * CHUNK),CHUNK) == CUDA_SUCCESS);
		assert(cuMemRelease(handles[index]) == CUDA_SUCCESS);
	}
	assert(cuMemAddressFree(base,(size_t)EXPERTS * CHUNK) == CUDA_SUCCESS);
	free(expected);
	return(exact != 0u);
}

static void *acquire_together(void *data)
{
	ConcurrentAcquire *acquirer = data;
	uint32_t expert = 2u;
	__atomic_add_fetch(acquirer->arrived,1u,__ATOMIC_SEQ_CST);
	while ( __atomic_load_n(acquirer->arrived,__ATOMIC_SEQ_CST) < 2u )
		;
	acquirer->status = acquire(acquirer->client,acquirer->generation,&expert,1u,&acquirer->result);
	return(0);
}

static void check_concurrent_same_expert(void)
{
	TestServer state;
	ConcurrentAcquire acquirers[2];
	pthread_t threads[2];
	uint32_t arrived = 0u,baseline,index,expert = 2u;
	uint64_t generations[2] = {0u,0u};
	server_start(&state);
	memset(acquirers,0,sizeof(acquirers));
	for (index=0u; index<2u; index++)
	{
		assert(SparkWeightdClientConnect(state.socket_path,&acquirers[index].client,0) == SPARK_STATUS_OK);
		assert(attach(acquirers[index].client,state.path,&generations[index]) == SPARK_STATUS_OK);
		acquirers[index].generation = generations[index];
		acquirers[index].arrived = &arrived;
	}
	CHECK( generations[0] == generations[1],"concurrent: both clients attach the same arena");
	baseline = spark_stub_cuda_outstanding_allocs();
	spark_stub_cuda_set_create_delay(50000u);
	for (index=0u; index<2u; index++)
		assert(pthread_create(&threads[index],0,acquire_together,&acquirers[index]) == 0);
	for (index=0u; index<2u; index++)
		assert(pthread_join(threads[index],0) == 0);
	spark_stub_cuda_set_create_delay(0u);
	for (index=0u; index<2u; index++)
		CHECK( acquirers[index].status == SPARK_STATUS_OK && acquirers[index].result.status == SPARK_STATUS_OK,
			"concurrent: both acquires of one expert succeed");
	CHECK( spark_stub_cuda_outstanding_allocs() == baseline + 1u,
		"concurrent: two simultaneous acquirers of one expert create its chunk once");
	for (index=0u; index<2u; index++)
	{
		if ( acquirers[index].status != SPARK_STATUS_OK )
			continue;
		CHECK( lease_serves(acquirers[index].client,generations[0],acquirers[index].result.lease_identifier,&expert,1u),
			"concurrent: each lease maps the expert byte-exact");
		release(acquirers[index].client,generations[0],acquirers[index].result.lease_identifier);
	}
	for (index=0u; index<2u; index++)
		SparkWeightdClientClose(acquirers[index].client);
	server_stop(&state,"concurrent: the daemon releases every chunk at shutdown");
}

static void check_mid_acquire_death(void)
{
	TestServer state;
	SparkWeightdClient *a = 0;
	SparkWeightdWorkingSetResult result;
	const uint32_t dying[2] = {1u,2u},other[2] = {0u,3u};
	uint64_t generation = 0u;
	uint32_t baseline,waited,attempt;
	SparkStatus status = SPARK_STATUS_IO_ERROR;
	pid_t pid;
	int wstatus = 0;
	server_start(&state);
	assert(SparkWeightdClientConnect(state.socket_path,&a,0) == SPARK_STATUS_OK);
	assert(attach(a,state.path,&generation) == SPARK_STATUS_OK);
	baseline = spark_stub_cuda_outstanding_allocs();
	spark_stub_cuda_set_create_delay(200000u);
	pid = fork();
	assert(pid >= 0);
	if ( pid == 0 )
	{
		SparkWeightdClient *child = 0;
		SparkWeightdWorkingSetResult held;
		uint64_t child_generation = 0u;
		if ( SparkWeightdClientConnect(state.socket_path,&child,0) != SPARK_STATUS_OK ||
			attach(child,state.path,&child_generation) != SPARK_STATUS_OK ||
			child_generation != generation )
			_exit(3);
		(void)acquire(child,generation,dying,2u,&held);
		_exit(4);
	}
	for (waited=0u; waited<5000u && spark_stub_cuda_outstanding_allocs() == baseline; waited++)
		usleep(1000);
	CHECK( waited < 5000u,"mid-acquire death: the child's expert load is in flight");
	assert(kill(pid,SIGKILL) == 0);
	assert(waitpid(pid,&wstatus,0) == pid);
	CHECK( WIFSIGNALED(wstatus) && WTERMSIG(wstatus) == SIGKILL,
		"mid-acquire death: the child dies by SIGKILL inside its acquire");
	spark_stub_cuda_set_create_delay(0u);
	for (attempt=0u; attempt<2000u; attempt++)
	{
		status = acquire(a,generation,other,2u,&result);
		if ( status != SPARK_STATUS_CAPACITY_EXCEEDED )
			break;
		usleep(1000);
	}
	CHECK( status == SPARK_STATUS_OK && result.status == SPARK_STATUS_OK,
		"mid-acquire death: the dead client's lease stops pinning pool chunks");
	if ( status == SPARK_STATUS_OK )
	{
		CHECK( lease_serves(a,generation,result.lease_identifier,other,2u),
			"mid-acquire death: experts evicting the dead client's chunks map byte-exact");
		release(a,generation,result.lease_identifier);
	}
	CHECK( spark_stub_cuda_outstanding_allocs() == baseline + POOL_CHUNKS,
		"mid-acquire death: the pool holds exactly its budget of chunks");
	status = acquire(a,generation,dying,2u,&result);
	CHECK( status == SPARK_STATUS_OK && result.status == SPARK_STATUS_OK,
		"mid-acquire death: the dead client's experts reload");
	if ( status == SPARK_STATUS_OK )
	{
		CHECK( lease_serves(a,generation,result.lease_identifier,dying,2u),
			"mid-acquire death: reloaded experts map byte-exact");
		release(a,generation,result.lease_identifier);
	}
	SparkWeightdClientClose(a);
	server_stop(&state,"mid-acquire death: the daemon releases every chunk at shutdown");
}

static void check_eviction_reload(void)
{
	TestServer state;
	SparkWeightdClient *a = 0;
	SparkWeightdWorkingSetResult result;
	const uint32_t order[6] = {0u,1u,2u,3u,0u,1u};
	uint64_t generation = 0u;
	uint32_t baseline,step,resident;
	server_start(&state);
	assert(SparkWeightdClientConnect(state.socket_path,&a,0) == SPARK_STATUS_OK);
	assert(attach(a,state.path,&generation) == SPARK_STATUS_OK);
	baseline = spark_stub_cuda_outstanding_allocs();
	for (step=0u; step<6u; step++)
	{
		SparkStatus status = acquire(a,generation,&order[step],1u,&result);
		CHECK( status == SPARK_STATUS_OK && result.status == SPARK_STATUS_OK,
			"eviction: every expert loads through a two-chunk pool");
		if ( status != SPARK_STATUS_OK )
			continue;
		CHECK( lease_serves(a,generation,result.lease_identifier,&order[step],1u),
			"eviction: loaded and reloaded experts map byte-exact");
		release(a,generation,result.lease_identifier);
		resident = step + 1u < POOL_CHUNKS ? step + 1u : POOL_CHUNKS;
		CHECK( spark_stub_cuda_outstanding_allocs() == baseline + resident,
			"eviction: the pool never holds more than its budget of chunks");
	}
	SparkWeightdClientClose(a);
	server_stop(&state,"eviction: the daemon releases every chunk at shutdown");
}

int main(void)
{
	check_concurrent_same_expert();
	check_mid_acquire_death();
	check_eviction_reload();
	fprintf(stderr,"test_weightd_expert_stress: %s\n",
		test_failures == 0u ? "PASS" : "FAILED");
	return( test_failures != 0u ? 1 : 0 );
}
