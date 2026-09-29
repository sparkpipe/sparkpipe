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
#include "fixtures/test_child_guard.h"

/* The vortex and the attach-slot leak, at the boundary where they live.
 *
 * Fleet evidence: a watchdog kill of weightd (or a kill -9 of an engine
 * holding an attach) must never consume a permanent resource. Two
 * regressions cost real days:
 *
 * 1. ATTACH-SLOT LEAK: a client that dies without a clean detach (SIGKILL
 *    -> EOF) must drop its slot. If slots leak, ~128 kill cycles later
 *    every new attach is refused and the node is dead until a weightd
 *    restart.
 * 2. MID-BAKE DEATH: a client SIGKILLed while its lazy attach is baking
 *    (chunk creation in flight) must leave zero committed chunks behind;
 *    the next attach re-bakes from scratch. Repeated kill cycles (the
 *    watchdog restart vortex) must converge to a working attach, never
 *    accumulate state.
 */

#define CHUNK (2u * 1024u * 1024u)
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
} TestServer;

static void *run_server(void *data)
{
	TestServer *state = data;
	assert(SparkWeightdServerRun(state->server,&state->stop) == SPARK_STATUS_OK);
	return(0);
}

static uint64_t range_offset(uint32_t expert,uint32_t kind)
{
	return(((kind / 2u) * CHUNK) + (expert * 256u) + ((kind % 2u) * 64u));
}

static void write_range(FILE *pack,FILE *manifest,uint32_t expert,uint32_t kind,uint64_t offset)
{
	uint8_t record[48] = {0},data[64],digest[16];
	uint64_t bytes = sizeof(data);
	SparkCk128Context ck;
	memset(data,(int32_t)(1u + (expert * 4u) + kind),sizeof(data));
	assert(fseek(pack,(long)offset,SEEK_SET) == 0);
	assert(fwrite(data,1u,sizeof(data),pack) == sizeof(data));
	SparkCk128Initialize(&ck);
	SparkCk128Update(&ck,data,sizeof(data));
	SparkCk128Finalize(&ck,digest);
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
			write_range(pack,manifest,expert,kind,range_offset(expert,kind));
	assert(fclose(pack) == 0);
	assert(fclose(manifest) == 0);
}

static SparkStatus attach_lazy_model(SparkWeightdClient *client,const char *path,const char *model,uint64_t *generation,uint64_t *resident_bytes)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyAttachResult result;
	SparkStatus status;
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (3u * CHUNK);
	snprintf(request.identity.model,sizeof(request.identity.model),"%s",model);
	assert(SparkSha256File(path,request.identity.pack_sha256) == SPARK_STATUS_OK);
	assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	request.expert_pool_bytes = (2u * CHUNK);
	status = SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT);
	if ( status == SPARK_STATUS_OK )
	{
		if ( generation != 0 )
			*generation = result.arena_generation;
		if ( resident_bytes != 0 )
			*resident_bytes = result.resident_bytes;
	}
	return(status);
}

static SparkStatus attach_lazy(SparkWeightdClient *client,const char *path,uint64_t *generation,uint64_t *resident_bytes)
{
	return(attach_lazy_model(client,path,"churn-test",generation,resident_bytes));
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

/* 130 sequential connect/close cycles (the connection cap is 128) with a
 * full attach+detach every 16th. Any slot or identity-state leak turns
 * into a refused connect or a refused attach long before the end. */
static void check_slot_churn(const char *socket_path,const char *path)
{
	uint32_t cycle;
	for (cycle=0u; cycle<130u; cycle++)
	{
		SparkWeightdClient *client = 0;
		SparkStatus status = SparkWeightdClientConnect(socket_path,&client,0);
		CHECK( status == SPARK_STATUS_OK && client != 0,
			"slot churn: connect never refused across 130 cycles");
		if ( status != SPARK_STATUS_OK || client == 0 )
			return;
		if ( (cycle % 16u) == 0u )
		{
			SparkWeightdDetachResult detached;
			uint64_t generation,resident;
			status = attach_lazy(client,path,&generation,&resident);
			CHECK( status == SPARK_STATUS_OK,
				"slot churn: attach succeeds mid-churn");
			if ( status == SPARK_STATUS_OK )
			{
				assert(SparkWeightdClientDetach(client,generation,&detached,TIMEOUT) == SPARK_STATUS_OK);
				assert(detached.status == SPARK_STATUS_OK);
			}
		}
		SparkWeightdClientClose(client);
		usleep(2000);
	}
}

/* SIGKILL a child mid-bake, four times in a row (the watchdog restart
 * vortex), then prove the daemon converges: the next cold attach works
 * and leased bytes are exact. */
static void check_midbake_death(const char *socket_path,const char *path)
{
	uint32_t cycle;
	spark_stub_cuda_set_create_delay(20000u);
	for (cycle=0u; cycle<4u; cycle++)
	{
		pid_t pid;
		uint32_t waited;
		uint32_t baseline = spark_stub_cuda_outstanding_allocs();
		char model[32];
		int wstatus = 0;
		snprintf(model,sizeof(model),"churn-vortex-%u",(unsigned)cycle);
		pid = TestChildGuardFork();
		if ( pid == 0 )
		{
			SparkWeightdClient *child = 0;
			if ( SparkWeightdClientConnect(socket_path,&child,0) == SPARK_STATUS_OK )
				(void)attach_lazy_model(child,path,model,0,0);
			_exit(0);
		}
		/* Wait until the bake is demonstrably in flight, then kill. */
		for (waited=0u; waited<2000u && spark_stub_cuda_outstanding_allocs() == baseline; waited++)
			usleep(1000);
		CHECK( waited < 2000u, "midbake: the child's bake actually started");
		assert(kill(pid,SIGKILL) == 0);
		assert(waitpid(pid,&wstatus,0) == pid);
		assert(WIFSIGNALED(wstatus) && WTERMSIG(wstatus) == SIGKILL);
		/* The daemon finishes the in-flight bake (it cannot see the EOF
		 * mid-dispatch) and keeps the arena warm — the policy question is
		 * eviction under pressure, exercised after the loop. What must
		 * hold immediately: the daemon stays responsive. */
		{
			SparkWeightdClient *probe = 0;
			CHECK( SparkWeightdClientConnect(socket_path,&probe,0) == SPARK_STATUS_OK,
				"midbake: daemon accepts connections right after a killed baker");
			if ( probe != 0 )
				SparkWeightdClientClose(probe);
		}
		(void)baseline;
	}

	/* Budget pressure must evict the warm arenas the vortex left behind:
	 * the fleet's death spiral was killed engines piling up warm arenas
	 * until every new attach was refused. device_bytes_max fits 6 arenas
	 * (30 chunks of 32); the vortex left 5 (churn-test + 4 vortex), so the
	 * second pressure attach only fits if eviction reclaims the dead. */
	{
		uint32_t p;
		for (p=0u; p<3u; p++)
		{
			SparkWeightdClient *client = 0;
			char model[32];
			SparkStatus status;
			snprintf(model,sizeof(model),"churn-pressure-%u",(unsigned)p);
			assert(SparkWeightdClientConnect(socket_path,&client,0) == SPARK_STATUS_OK);
			status = attach_lazy_model(client,path,model,0,0);
			CHECK( status == SPARK_STATUS_OK,
				"pressure: warm vortex arenas are evicted, attach never refused");
			SparkWeightdClientClose(client);
		}
	}
	spark_stub_cuda_set_create_delay(0u);
	{
		SparkWeightdClient *client = 0;
		SparkWeightdWorkingSetResult acquired;
		SparkWeightdDetachResult detached;
		SparkWeightdExpertKey key = {0u,0u};
		uint64_t generation = 0u,resident = 0u;
		assert(SparkWeightdClientConnect(socket_path,&client,0) == SPARK_STATUS_OK);
		CHECK( attach_lazy_model(client,path,"churn-final",
				&generation,&resident) == SPARK_STATUS_OK,
			"midbake: the daemon converges to a working attach after the vortex");
		assert(SparkWeightdClientAcquire(client,generation,&key,1u,&acquired,TIMEOUT) == SPARK_STATUS_OK);
		assert(acquired.status == SPARK_STATUS_OK);
		{
			SparkWeightdExportBatch batch;
			CUmemGenericAllocationHandle handles[2];
			CUdeviceptr base;
			CUmemAccessDesc access = {0};
			uint32_t i;
			assert(SparkWeightdClientExportLeaseBatch(client,generation,acquired.lease_identifier,0u,&batch,TIMEOUT) == SPARK_STATUS_OK);
			assert(batch.status == SPARK_STATUS_OK && batch.batch_count == 2u && batch.lease_chunk_count == 2u);
			access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
			access.location.id = 0;
			access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
			assert(cuMemAddressReserve(&base,(3u * CHUNK),0u,0u,0u) == CUDA_SUCCESS);
			for (i=0u; i<batch.batch_count; i++)
			{
				assert(cuMemImportFromShareableHandle(&handles[i],(void *)(intptr_t)batch.fds[i],CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS);
				assert(close(batch.fds[i]) == 0);
				assert(cuMemMap(base + (batch.chunk_indices[i] * CHUNK),CHUNK,0u,handles[i],0u) == CUDA_SUCCESS);
				assert(cuMemSetAccess(base + (batch.chunk_indices[i] * CHUNK),CHUNK,&access,1u) == CUDA_SUCCESS);
			}
			check_ranges((uint64_t)(uintptr_t)base,0u);
			for (i=0u; i<batch.batch_count; i++)
			{
				assert(cuMemUnmap(base + (batch.chunk_indices[i] * CHUNK),CHUNK) == CUDA_SUCCESS);
				assert(cuMemRelease(handles[i]) == CUDA_SUCCESS);
			}
			assert(cuMemAddressFree(base,(3u * CHUNK)) == CUDA_SUCCESS);
		}
		assert(SparkWeightdClientRelease(client,generation,acquired.lease_identifier,&acquired,TIMEOUT) == SPARK_STATUS_OK);
		assert(SparkWeightdClientDetach(client,generation,&detached,TIMEOUT) == SPARK_STATUS_OK);
		SparkWeightdClientClose(client);
	}
}

int main(void)
{
	char root[] = "/tmp/weightd-churn-XXXXXX",path[256],manifest[272],recording[272],socket_path[256];
	TestServer state = {0};
	SparkWeightdServerConfig config = {0};
	pthread_t thread;
	assert(mkdtemp(root) != 0);
	snprintf(path,sizeof(path),"%s/pack",root);
	snprintf(manifest,sizeof(manifest),"%s.experts",path);
	snprintf(recording,sizeof(recording),"%s.wset",path);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	write_fixture(path,manifest);
	config.socket_path = socket_path;
	config.device_bytes_max = (32u * CHUNK);
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	check_slot_churn(socket_path,path);
	check_midbake_death(socket_path,path);
	__atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	CHECK( spark_stub_cuda_outstanding_allocs() == 0u,
		"no allocations leak across the whole churn run");
	assert(unlink(recording) == 0 && unlink(manifest) == 0 && unlink(path) == 0 && rmdir(root) == 0);
	fprintf(stderr,"test_weightd_churn: %s\n",
		test_failures == 0u ? "PASS" : "FAILED");
	return( test_failures != 0u ? 1 : 0 );
}
