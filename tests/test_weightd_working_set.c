#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cuda.h"
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_weightd_spine.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_sha256.h"

#define CHUNK (2u * 1024u * 1024u)
#define TIMEOUT UINT64_C(10000000000)

void spark_stub_cuda_event_pending(uint32_t pending);
void spark_stub_cuda_event_record_failure(uint32_t failure);
void spark_stub_cuda_fail_import_after(uint32_t calls);
void spark_stub_cuda_fail_next_unmap(void);
void spark_stub_cuda_fail_event_destroy_after(uint32_t calls);
void spark_stub_cuda_set_import_delay(uint32_t delay);
void spark_stub_cuda_fail_next_alloc(void);
void spark_stub_cuda_fail_alloc_after(uint32_t calls);
void spark_stub_cuda_fail_export_after(uint32_t calls);
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
			write_range(pack,manifest,expert,kind,range_offset(expert,kind));
	assert(fclose(pack) == 0);
	assert(fclose(manifest) == 0);
}

static void check_spine_load(const char *path,const char *manifest_path)
{
	SparkWeightdManifest manifest;
	char digest[SPARK_SHA256_HEX_BYTES];
	uint8_t *destination,value = 255u;
	uint64_t i,j,cursor = 0u;
	int32_t fd = open(path,O_RDWR);
	assert(fd >= 0);
	assert(pwrite(fd,&value,1u,512) == 1);
	assert(SparkWeightdManifestLoad(manifest_path,3u * CHUNK,&manifest) == SPARK_STATUS_OK);
	assert(SparkSha256File(path,digest) == SPARK_STATUS_OK);
	assert(posix_memalign((void **)&destination,256u,(size_t)manifest.spine_allocation_bytes) == 0);
	memset(destination,0xa5,(size_t)manifest.spine_allocation_bytes);
	assert(SparkWeightdSpineLoad(fd,&manifest,3u * CHUNK,digest,destination,manifest.spine_allocation_bytes - 1u) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(destination[0] == 0xa5);
	assert(lseek(fd,123,SEEK_SET) == 123);
	assert(SparkWeightdSpineLoad(fd,&manifest,3u * CHUNK,digest,destination,manifest.spine_allocation_bytes) == SPARK_STATUS_OK);
	assert(lseek(fd,0,SEEK_CUR) == 123);
	for (i=0u; i<manifest.spine_count; i++)
	{
		for (j=cursor; j<manifest.spine[i].compact_offset; j++)
			assert(destination[j] == 0xa5);
		cursor = (manifest.spine[i].compact_offset + manifest.spine[i].bytes);
		for (j=manifest.spine[i].compact_offset; j<cursor; j++)
			assert(destination[j] == ((manifest.spine[i].offset + j - manifest.spine[i].compact_offset) == 512u ? 255u : 0u));
	}
	// Changing an expert byte must invalidate the whole-pack identity too.
	assert(pwrite(fd,&value,1u,0) == 1);
	assert(SparkWeightdSpineLoad(fd,&manifest,3u * CHUNK,digest,destination,manifest.spine_allocation_bytes) == SPARK_STATUS_HASH_MISMATCH);
	value = 1u;
	assert(pwrite(fd,&value,1u,0) == 1);
	free(destination);
	SparkWeightdManifestDestroy(&manifest);
	assert(close(fd) == 0);
}

static uint64_t attach_config(SparkWeightdClient *client,const char *path,uint32_t chunks,uint32_t pool,uint32_t experts,uint64_t *base)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyAttachResult result;
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (chunks * CHUNK);
	memcpy(request.identity.model,"working-set-test",17u);
	memset(request.identity.pack_sha256,'a',64u);
	assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	request.expert_pool_bytes = (pool * CHUNK);
	assert(SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT) == SPARK_STATUS_OK);
	assert(result.expert_count == experts);
	*base = result.device_handle;
	return(result.arena_generation);
}

static uint64_t attach(SparkWeightdClient *client,const char *path,uint64_t *base)
{
	return(attach_config(client,path,3u,2u,4u,base));
}

static void write_many(const char *path,const char *manifest_path)
{
	FILE *pack = fopen(path,"wb"),*manifest = fopen(manifest_path,"wb");
	uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,65u,0u},i;
	assert(pack != 0 && manifest != 0);
	assert(ftruncate(fileno(pack),(65u * CHUNK)) == 0);
	assert(fwrite(header,1u,sizeof(header),manifest) == sizeof(header));
	for (i=0u; i<65u; i++)
		write_range(pack,manifest,i,0u,((uint64_t)i * CHUNK));
	assert(fclose(pack) == 0 && fclose(manifest) == 0);
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

static uint32_t count_fds(void)
{
	uint32_t count = 0u,i;
	for (i=0u; i<1024u; i++)
		count += fcntl((int32_t)i,F_GETFD) >= 0;
	return(count);
}

static void check_export(SparkWeightdClient *client,uint64_t generation,uint64_t lease,uint64_t daemon_base)
{
	SparkWeightdExportBatch batch;
	CUmemGenericAllocationHandle handles[2];
	CUdeviceptr base;
	CUmemAccessDesc access = {0};
	uint32_t i,descriptors = count_fds();
	spark_stub_cuda_fail_export_after(2u);
	assert(SparkWeightdClientExportLeaseBatch(client,generation,lease,0u,&batch,TIMEOUT) == SPARK_STATUS_OK);
	assert(batch.status == SPARK_STATUS_IO_ERROR && batch.batch_count == 0u);
	assert(count_fds() == descriptors);
	assert(SparkWeightdClientExportLeaseBatch(client,generation,lease,0u,&batch,TIMEOUT) == SPARK_STATUS_OK);
	assert(batch.status == SPARK_STATUS_OK && batch.batch_count == 2u && batch.lease_chunk_count == 2u && batch.chunk_count == 3u);
	assert(batch.chunk_bytes == CHUNK && batch.chunk_indices[0] == 0u && batch.chunk_indices[1] == 1u);
	assert(cuMemAddressReserve(&base,(3u * CHUNK),0u,0u,0u) == CUDA_SUCCESS);
	assert(base != daemon_base);
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.location.id = 0;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
	for (i=0u; i<2u; i++)
	{
		assert((fcntl(batch.fds[i],F_GETFD) & FD_CLOEXEC) != 0);
		assert(cuMemImportFromShareableHandle(&handles[i],(void *)(intptr_t)batch.fds[i],CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS);
		assert(close(batch.fds[i]) == 0);
		assert(cuMemMap(base + (batch.chunk_indices[i] * CHUNK),CHUNK,0u,handles[i],0u) == CUDA_SUCCESS);
		assert(cuMemSetAccess(base + (batch.chunk_indices[i] * CHUNK),CHUNK,&access,1u) == CUDA_SUCCESS);
	}
	check_ranges(base,0u);
	for (i=0u; i<2u; i++)
	{
		assert(cuMemUnmap(base + (batch.chunk_indices[i] * CHUNK),CHUNK) == CUDA_SUCCESS);
		assert(cuMemRelease(handles[i]) == CUDA_SUCCESS);
	}
	assert(cuMemAddressFree(base,(3u * CHUNK)) == CUDA_SUCCESS);
	assert(count_fds() == descriptors);
}

static void check_export_refused(SparkWeightdClient *client,uint64_t generation,uint64_t lease)
{
	SparkWeightdExportBatch batch;
	assert(SparkWeightdClientExportLeaseBatch(client,generation,lease,0u,&batch,TIMEOUT) == SPARK_STATUS_OK);
	assert(batch.status == SPARK_STATUS_NOT_FOUND && batch.batch_count == 0u);
}

static void check_sparse_export(SparkWeightdClient *client,uint64_t generation,uint64_t lease)
{
	SparkWeightdExportBatch batch;
	assert(SparkWeightdClientExportLeaseBatch(client,generation,lease,0u,&batch,TIMEOUT) == SPARK_STATUS_OK);
	assert(batch.status == SPARK_STATUS_OK && batch.batch_count == 1u && batch.chunk_indices[0] == 2u);
	assert(close(batch.fds[0]) == 0);
	assert(SparkWeightdClientExportLeaseBatch(client,generation,lease,1u,&batch,TIMEOUT) == SPARK_STATUS_OK);
	assert(batch.status == SPARK_STATUS_INVALID_ARGUMENT && batch.batch_count == 0u);
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
	check_export(a,generation,first.lease_identifier,base);
	check_export_refused(b,generation,first.lease_identifier);
	second = acquire(b,generation,1u,SPARK_STATUS_OK);
	assert(second.resident_bytes == first.resident_bytes);
	check_ranges(base,1u);
	assert(SparkWeightdClientRelease(b,generation,first.lease_identifier,&result,TIMEOUT) == SPARK_STATUS_NOT_FOUND);
	(void)acquire(b,generation,2u,SPARK_STATUS_CAPACITY_EXCEEDED);
	check_ranges(base,0u);
	check_ranges(base,1u);
	release(a,generation,first.lease_identifier);
	check_export_refused(a,generation,first.lease_identifier);
	(void)acquire(a,generation,2u,SPARK_STATUS_CAPACITY_EXCEEDED);
	release(b,generation,second.lease_identifier);
	result = acquire(a,generation,2u,SPARK_STATUS_OK);
	check_sparse_export(a,generation,result.lease_identifier);
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

static void check_map_lifetime(SparkWeightdClient *client,uint64_t generation,uint64_t daemon_base)
{
	SparkWeightdLazyAttachResult attached = {0};
	SparkWeightdMap *map;
	SparkWeightdExpertKey key = {0u,0u};
	uint64_t first,second;
	void *address,*other;
	attached.status = SPARK_STATUS_OK;
	attached.arena_generation = generation;
	attached.arena_bytes = (3u * CHUNK);
	attached.chunk_bytes = CHUNK;
	attached.chunk_count = 3u;
	assert(SparkWeightdMapCreate(client,&attached,&map) == SPARK_STATUS_OK);
	assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_OK);
	key.expert = 1u;
	assert(SparkWeightdMapAcquire(map,&key,1u,&second,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdMapDestroy(map) == SPARK_STATUS_BUSY);
	assert(SparkWeightdMapBeginUse(map,first,&address) == SPARK_STATUS_OK);
	assert((uint64_t)(uintptr_t)address != daemon_base);
	check_ranges((uint64_t)(uintptr_t)address,0u);
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_BUSY);
	spark_stub_cuda_event_record_failure(1u);
	assert(SparkWeightdMapRecordCompletion(map,first,0) == SPARK_STATUS_IO_ERROR);
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_BUSY);
	spark_stub_cuda_event_record_failure(0u);
	assert(SparkWeightdMapRecordCompletion(map,first,0) == SPARK_STATUS_OK);
	spark_stub_cuda_event_pending(1u);
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_BUSY);
	spark_stub_cuda_event_pending(0u);
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_NOT_FOUND);
	assert(SparkWeightdMapBeginUse(map,second,&other) == SPARK_STATUS_OK);
	assert(other == address);
	check_ranges((uint64_t)(uintptr_t)other,1u);
	assert(SparkWeightdMapRecordCompletion(map,second,0) == SPARK_STATUS_OK);
	assert(SparkWeightdMapRelease(map,second,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_OK);
	spark_stub_cuda_fail_export_after(2u);
	assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_IO_ERROR);
	assert(first == 0u);
	assert(SparkWeightdMapDestroy(map) == SPARK_STATUS_OK);
	assert(SparkWeightdMapCreate(client,&attached,&map) == SPARK_STATUS_OK);
	spark_stub_cuda_fail_import_after(2u);
	assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_IO_ERROR);
	assert(first == 0u);
	assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_OK);
	spark_stub_cuda_fail_next_unmap();
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_IO_ERROR);
	assert(SparkWeightdMapDestroy(map) == SPARK_STATUS_BUSY);
	(void)acquire(client,generation,2u,SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdMapDestroy(map) == SPARK_STATUS_OK);
	assert(SparkWeightdMapCreate(client,&attached,&map) == SPARK_STATUS_OK);
	spark_stub_cuda_set_import_delay(200000u);
	assert(SparkWeightdMapAcquire(map,&key,1u,&first,UINT64_C(100000000)) == SPARK_STATUS_BUSY);
	spark_stub_cuda_set_import_delay(0u);
	assert(first != 0u);
	assert(SparkWeightdMapDestroy(map) == SPARK_STATUS_BUSY);
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_OK);
	spark_stub_cuda_fail_event_destroy_after(2u);
	assert(SparkWeightdMapDestroy(map) == SPARK_STATUS_IO_ERROR);
	assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_IO_ERROR);
	assert(first == 0u);
	assert(SparkWeightdMapDestroy(map) == SPARK_STATUS_OK);
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

static void check_many_exports(void)
{
	char root[] = "/tmp/weightd-many-XXXXXX",path[256],manifest[272],socket_path[256];
	TestServer state = {0};
	SparkWeightdServerConfig config = {0};
	SparkWeightdClient *client;
	SparkWeightdExpertKey keys[65];
	SparkWeightdWorkingSetResult result;
	SparkWeightdExportBatch batch;
	pthread_t thread;
	uint64_t generation,base;
	uint32_t i,offset = 0u;
	assert(mkdtemp(root) != 0);
	snprintf(path,sizeof(path),"%s/pack",root);
	snprintf(manifest,sizeof(manifest),"%s.experts",path);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	write_many(path,manifest);
	config.socket_path = socket_path;
	config.device_bytes_max = (66u * CHUNK);
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	assert(SparkWeightdClientConnect(socket_path,&client,0) == SPARK_STATUS_OK);
	generation = attach_config(client,path,65u,65u,65u,&base);
	for (i=0u; i<65u; i++)
		keys[i] = (SparkWeightdExpertKey){0u,i};
	assert(SparkWeightdClientAcquire(client,generation,keys,65u,&result,TIMEOUT) == SPARK_STATUS_OK);
	while ( offset < 65u )
	{
		assert(SparkWeightdClientExportLeaseBatch(client,generation,result.lease_identifier,offset,&batch,TIMEOUT) == SPARK_STATUS_OK);
		assert(batch.status == SPARK_STATUS_OK && batch.lease_chunk_count == 65u && batch.chunk_count == 65u);
		assert(batch.batch_count == (offset == 0u ? 64u : 1u));
		for (i=0u; i<batch.batch_count; i++)
		{
			assert(batch.chunk_indices[i] == (offset + i));
			assert(close(batch.fds[i]) == 0);
		}
		offset += batch.batch_count;
	}
	release(client,generation,result.lease_identifier);
	SparkWeightdClientClose(client);
	__atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	assert(spark_stub_cuda_outstanding_allocs() == 0u);
	assert(unlink(manifest) == 0 && unlink(path) == 0 && rmdir(root) == 0);
}

static SparkStatus reject_manifest(const SparkWeightdManifest *manifest,void *context)
{
	uint32_t *calls = (uint32_t *)context;
	assert(manifest->range_count != 0u);
	*calls += 1u;
	return(SPARK_STATUS_SCHEMA_ERROR);
}

static void check_lazy_pack(const char *socket_path,const char *path,const char *manifest_path)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyPack *pack;
	SparkWeightdExpertKey key = {0u,0u};
	uint64_t lease;
	uint32_t checks = 0u;
	const void *pointer;
	char saved[300];
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (3u * CHUNK);
	memcpy(request.identity.model,"lazy-pack-test",15u);
	assert(SparkSha256File(path,request.identity.pack_sha256) == SPARK_STATUS_OK);
	assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	request.expert_pool_bytes = (2u * CHUNK);
	assert(SparkWeightdLazyPackCreate(socket_path,&request,1u,TIMEOUT,&pack) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(pack == 0);
	snprintf(saved,sizeof(saved),"%s.saved",manifest_path);
	assert(rename(manifest_path,saved) == 0);
	assert(SparkWeightdLazyPackCreate(socket_path,&request,4u * CHUNK,TIMEOUT,&pack) == SPARK_STATUS_NOT_FOUND);
	assert(pack == 0);
	assert(rename(saved,manifest_path) == 0);
	spark_stub_cuda_fail_next_alloc();
	assert(SparkWeightdLazyPackCreateChecked(socket_path,&request,4u * CHUNK,TIMEOUT,reject_manifest,&checks,&pack) == SPARK_STATUS_SCHEMA_ERROR);
	assert(pack == 0 && checks == 1u);
	// Rejection must precede allocation: the injected failure remains pending.
	assert(SparkWeightdLazyPackCreate(socket_path,&request,4u * CHUNK,TIMEOUT,&pack) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(pack == 0);
	memset(request.identity.pack_sha256,'b',64u);
	assert(SparkWeightdLazyPackCreate(socket_path,&request,4u * CHUNK,TIMEOUT,&pack) == SPARK_STATUS_HASH_MISMATCH);
	assert(pack == 0);
	assert(SparkSha256File(path,request.identity.pack_sha256) == SPARK_STATUS_OK);
	assert(SparkWeightdLazyPackCreate(socket_path,&request,4u * CHUNK,TIMEOUT,&pack) == SPARK_STATUS_OK);
	assert(pack != 0 && pack->attached.resident_bytes == 0u);
	assert(SparkWeightdLazyPackSlice(pack,512u,1u,&pointer) == SPARK_STATUS_OK);
	assert(*(const uint8_t *)pointer == 255u);
	assert(SparkWeightdLazyPackSlice(pack,0u,64u,&pointer) == SPARK_STATUS_NOT_FOUND && pointer == 0);
	assert(SparkWeightdMapAcquire(pack->map,&key,1u,&lease,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_BUSY);
	assert(SparkWeightdLazyPackSlice(pack,512u,1u,&pointer) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkWeightdMapRelease(pack->map,lease,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_OK);
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
	check_spine_load(path,manifest);
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
	check_map_lifetime(a,generation,base);
	check_orphan(a,b,generation,base,socket_path,path);
	SparkWeightdClientClose(b);
	__atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	assert(spark_stub_cuda_outstanding_allocs() == 0u);
	state.stop = 0;
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	check_lazy_pack(socket_path,path,manifest);
	__atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	assert(spark_stub_cuda_outstanding_allocs() == 0u);
	assert(unlink(manifest) == 0 && unlink(path) == 0 && rmdir(root) == 0);
	check_many_exports();
	puts("PASS working-set IPC: all ranges, leases, rollback, scoped imports and 65-chunk exports");
	return(0);
}
