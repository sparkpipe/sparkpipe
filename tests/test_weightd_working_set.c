#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
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

#define PACK_BYTES ((3u * CHUNK) + 512u)

static uint64_t range_offset(uint32_t expert,uint32_t kind)
{
	if ( expert < 2u )
		return(((kind / 2u) * CHUNK) + (expert * 256u) + ((kind % 2u) * 64u));
	return((3u * CHUNK) + ((expert - 2u) * 256u) + (kind * 64u));
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
	assert(ftruncate(fileno(pack),PACK_BYTES) == 0);
	assert(fwrite(header,1u,sizeof(header),manifest) == sizeof(header));
	for (expert=0u; expert<4u; expert++)
		for (kind=0u; kind<4u; kind++)
			write_range(pack,manifest,expert,kind,range_offset(expert,kind));
	assert(fclose(pack) == 0);
	assert(fclose(manifest) == 0);
}

/* Scan /tmp/spark-weightd-spine for the receipt bound to this pack's
 * (size,mtime,ctime) - the same binding the loader validates - and hand
 * back its path and proof basis. Used to assert the prong-1 trust chain:
 * client full hash writes proof 0, the daemon recorder writes proof 1,
 * tampered bases are re-proven. */
static int find_spine_receipt(const char *pack_path,char *out,size_t out_bytes,
	uint64_t *proof)
{
	static const char directory_path[] = "/tmp/spark-weightd-spine";
	DIR *directory = opendir(directory_path);
	struct dirent *entry;
	struct stat st;
	int found = 0;
	if ( directory == 0 || stat(pack_path,&st) != 0 )
	{
		if ( directory != 0 )
			closedir(directory);
		return(0);
	}
	while ( found == 0 && (entry = readdir(directory)) != 0 )
	{
		char path[512];
		uint8_t raw[88];
		FILE *file;
		uint64_t magic,size,mtime_ns,ctime_ns,recorded;
		if ( entry->d_name[0] == '.' )
			continue;
		snprintf(path,sizeof(path),"%s/%s",directory_path,entry->d_name);
		file = fopen(path,"rb");
		if ( file == 0 || fread(raw,1u,sizeof(raw),file) != sizeof(raw) )
		{
			if ( file != 0 )
				fclose(file);
			continue;
		}
		fclose(file);
		memcpy(&magic,raw,8u);
		memcpy(&size,raw + 8u,8u);
		memcpy(&mtime_ns,raw + 16u,8u);
		memcpy(&ctime_ns,raw + 24u,8u);
		memcpy(&recorded,raw + 80u,8u);
		if ( magic != UINT64_C(0x5350494e45524531) ||
			size != (uint64_t)st.st_size )
			continue;
#if defined(__APPLE__)
		if ( mtime_ns != ((uint64_t)st.st_mtimespec.tv_sec * UINT64_C(1000000000) + (uint64_t)st.st_mtimespec.tv_nsec) ||
			ctime_ns != ((uint64_t)st.st_ctimespec.tv_sec * UINT64_C(1000000000) + (uint64_t)st.st_ctimespec.tv_nsec) )
			continue;
#else
		if ( mtime_ns != ((uint64_t)st.st_mtim.tv_sec * UINT64_C(1000000000) + (uint64_t)st.st_mtim.tv_nsec) ||
			ctime_ns != ((uint64_t)st.st_ctim.tv_sec * UINT64_C(1000000000) + (uint64_t)st.st_ctim.tv_nsec) )
			continue;
#endif
		snprintf(out,out_bytes,"%s",path);
		*proof = recorded;
		found = 1;
	}
	closedir(directory);
	return(found);
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
	assert(SparkWeightdManifestLoad(manifest_path,PACK_BYTES,&manifest) == SPARK_STATUS_OK);
	assert(SparkSha256File(path,digest) == SPARK_STATUS_OK);
	assert(posix_memalign((void **)&destination,256u,(size_t)manifest.spine_allocation_bytes) == 0);
	memset(destination,0xa5,(size_t)manifest.spine_allocation_bytes);
	assert(SparkWeightdSpineLoad(fd,&manifest,PACK_BYTES,digest,destination,manifest.spine_allocation_bytes - 1u) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(destination[0] == 0xa5);
	assert(lseek(fd,123,SEEK_SET) == 123);
	assert(SparkWeightdSpineLoad(fd,&manifest,PACK_BYTES,digest,destination,manifest.spine_allocation_bytes) == SPARK_STATUS_OK);
	assert(lseek(fd,0,SEEK_CUR) == 123);
	for (i=0u; i<manifest.spine_count; i++)
	{
		for (j=cursor; j<manifest.spine[i].compact_offset; j++)
			assert(destination[j] == 0xa5);
		cursor = (manifest.spine[i].compact_offset + manifest.spine[i].bytes);
		for (j=manifest.spine[i].compact_offset; j<cursor; j++)
			assert(destination[j] == ((manifest.spine[i].offset + j - manifest.spine[i].compact_offset) == 512u ? 255u : 0u));
	}
	/* Prong-1 trust chain: the load above (full client hash) recorded a
	 * proof-0 receipt; the daemon recorder writes the same binding with
	 * the proof-1 basis and the loader accepts it; a tampered basis is
	 * treated as absent and re-proven by a full hash (proof back to 0). */
	{
		char receipt[512];
		uint8_t sha_bytes[32];
		uint64_t proof = 99u;
		int nibble,index;
		assert(find_spine_receipt(path,receipt,sizeof(receipt),&proof) == 1);
		assert(proof == UINT64_C(0));
		for (index=0; index<64; index++)
		{
			char letter = digest[index];
			nibble = letter >= 'a' ? letter - 'a' + 10 : letter - '0';
			if ( (index % 2) == 0 )
				sha_bytes[index / 2] = (uint8_t)(nibble << 4);
			else
				sha_bytes[index / 2] |= (uint8_t)nibble;
		}
		assert(SparkWeightdSpineReceiptRecordDaemon(fd,digest,sha_bytes) == SPARK_STATUS_OK);
		assert(find_spine_receipt(path,receipt,sizeof(receipt),&proof) == 1);
		assert(proof == UINT64_C(1));
		memset(destination,0xa5,(size_t)manifest.spine_allocation_bytes);
		assert(SparkWeightdSpineLoad(fd,&manifest,PACK_BYTES,digest,destination,manifest.spine_allocation_bytes) == SPARK_STATUS_OK);
		cursor = 0u;
		for (i=0u; i<manifest.spine_count; i++)
		{
			for (j=cursor; j<manifest.spine[i].compact_offset; j++)
				assert(destination[j] == 0xa5);
			cursor = (manifest.spine[i].compact_offset + manifest.spine[i].bytes);
			for (j=manifest.spine[i].compact_offset; j<cursor; j++)
				assert(destination[j] == ((manifest.spine[i].offset + j - manifest.spine[i].compact_offset) == 512u ? 255u : 0u));
		}
		{
			FILE *file = fopen(receipt,"r+b");
			uint64_t bad = UINT64_C(2);
			assert(file != 0);
			assert(fseek(file,80L,SEEK_SET) == 0);
			assert(fwrite(&bad,8u,1u,file) == 1u);
			assert(fclose(file) == 0);
		}
		assert(SparkWeightdSpineLoad(fd,&manifest,PACK_BYTES,digest,destination,manifest.spine_allocation_bytes) == SPARK_STATUS_OK);
		assert(find_spine_receipt(path,receipt,sizeof(receipt),&proof) == 1);
		assert(proof == UINT64_C(0));
	}
	// Changing an expert byte must invalidate the whole-pack identity too.
	assert(pwrite(fd,&value,1u,0) == 1);
	assert(SparkWeightdSpineLoad(fd,&manifest,PACK_BYTES,digest,destination,manifest.spine_allocation_bytes) == SPARK_STATUS_HASH_MISMATCH);
	value = 1u;
	assert(pwrite(fd,&value,1u,0) == 1);
	free(destination);
	SparkWeightdManifestDestroy(&manifest);
	assert(close(fd) == 0);
}

static uint64_t attach_config(SparkWeightdClient *client,const char *path,uint64_t arena_bytes,uint32_t pool,uint32_t experts,uint64_t *base)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyAttachResult result;
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = arena_bytes;
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
	return(attach_config(client,path,PACK_BYTES,2u,4u,base));
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
	assert(batch.status == SPARK_STATUS_OK && batch.batch_count == 2u && batch.lease_chunk_count == 2u && batch.chunk_count == 4u);
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
	assert(batch.status == SPARK_STATUS_OK && batch.batch_count == 1u && batch.chunk_indices[0] == 3u);
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
	result = acquire(a,generation,2u,SPARK_STATUS_CAPACITY_EXCEEDED);
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
	assert(result.resident_bytes == CHUNK);
	check_ranges(base,2u);
	release(client,generation,result.lease_identifier);
}

static uint64_t working_set_seed = 7u;
static uint32_t working_set_rounds = 300u;

static void check_seeded_leases(SparkWeightdClient *a,SparkWeightdClient *b,uint64_t generation,uint64_t base)
{
    struct { uint64_t id; uint32_t owner,expert; } leases[16] = {{0}};
    SparkWeightdClient *clients[2] = {a,b};
    SparkWeightdWorkingSetResult result;
    SparkWeightdExpertKey keys[3];
    uint64_t random = working_set_seed,stale = UINT64_MAX;
    uint32_t step,slot,i,owner,expert,mask,operation,coverage[7] = {0};
    SparkStatus expected,actual;
    for (step=0u; step<working_set_rounds + 7u; step++)
    {
        random = random * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        slot = (uint32_t)((random >> 17u) % 16u);
        owner = (uint32_t)((random >> 27u) & 1u);
        expert = (uint32_t)((random >> 32u) % 3u);
        operation = step < 7u ? step : (uint32_t)((random >> 39u) % 7u);
        if ( step < 3u )
        {
            slot = step == 1u ? 1u : 0u;
            owner = 0u;
            expert = step == 1u ? 1u : 0u;
        }
        if ( operation == 2u && leases[slot].id == 0u ) operation = 6u;
        coverage[operation]++;
        mask = 0u;
        for (i=0u; i<16u; i++)
            if ( leases[i].id != 0u ) mask |= leases[i].expert < 2u ? 3u : 4u;
        if ( operation == 0u || operation == 1u )
        {
            if ( leases[slot].id != 0u )
            {
                release(clients[leases[slot].owner],generation,leases[slot].id);
                stale = leases[slot].id;
                leases[slot].id = 0u;
                mask = 0u;
                for (i=0u; i<16u; i++)
                    if ( leases[i].id != 0u ) mask |= leases[i].expert < 2u ? 3u : 4u;
            }
            keys[0] = (SparkWeightdExpertKey){0u,expert};
            keys[1] = keys[0];
            keys[2] = keys[0];
            expected = (mask | (expert < 2u ? 3u : 4u)) == 7u ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_OK;
            actual = SparkWeightdClientAcquire(clients[owner],generation,keys,operation == 0u ? 1u : 3u,&result,TIMEOUT);
            if ( actual != expected )
                fprintf(stderr,"working-set seed=%llu step=%u op=%u pinned=%u expert=%u expected=%u actual=%u\n",(unsigned long long)working_set_seed,step,operation,mask,expert,(unsigned)expected,(unsigned)actual);
            assert(actual == expected && result.status == expected);
            assert((result.lease_identifier != 0u) == (expected == SPARK_STATUS_OK));
            if ( actual == SPARK_STATUS_OK )
            {
                leases[slot].id = result.lease_identifier;
                leases[slot].owner = owner;
                leases[slot].expert = expert;
            }
        }
        else if ( operation == 2u && leases[slot].id != 0u )
        {
            assert(SparkWeightdClientRelease(clients[leases[slot].owner ^ 1u],generation,leases[slot].id,&result,TIMEOUT) == SPARK_STATUS_NOT_FOUND);
            check_export_refused(clients[leases[slot].owner ^ 1u],generation,leases[slot].id);
        }
        else if ( operation == 3u )
        {
            keys[0] = (SparkWeightdExpertKey){0u,expert};
            assert(SparkWeightdClientAcquire(clients[owner],generation + 1u,keys,1u,&result,TIMEOUT) == SPARK_STATUS_NOT_FOUND);
            assert(result.lease_identifier == 0u);
        }
        else if ( operation == 4u )
        {
            keys[0] = (SparkWeightdExpertKey){0u,0u};
            keys[1] = (SparkWeightdExpertKey){0u,2u};
            assert(SparkWeightdClientAcquire(clients[owner],generation,keys,2u,&result,TIMEOUT) == SPARK_STATUS_CAPACITY_EXCEEDED);
            assert(result.lease_identifier == 0u);
        }
        else if ( operation == 5u )
        {
            keys[0] = (SparkWeightdExpertKey){0u,3u};
            expected = (mask & 3u) != 0u ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_HASH_MISMATCH;
            assert(SparkWeightdClientAcquire(clients[owner],generation,keys,1u,&result,TIMEOUT) == expected);
            assert(result.lease_identifier == 0u);
        }
        else
        {
            assert(SparkWeightdClientRelease(clients[owner],generation,stale,&result,TIMEOUT) == SPARK_STATUS_NOT_FOUND);
            check_export_refused(clients[owner],generation,stale);
        }
        for (i=0u; i<16u; i++)
            if ( leases[i].id != 0u ) check_ranges(base,leases[i].expert);
        assert(result.resident_bytes <= 4u * CHUNK);
    }
    for (i=0u; i<16u; i++)
        if ( leases[i].id != 0u ) release(clients[leases[i].owner],generation,leases[i].id);
    for (i=0u; i<7u; i++) assert(coverage[i] != 0u);
    (void)acquire(a,generation,3u,SPARK_STATUS_HASH_MISMATCH);
    result = acquire(a,generation,2u,SPARK_STATUS_OK);
    check_ranges(base,2u);
    release(a,generation,result.lease_identifier);
    result = acquire(b,generation,0u,SPARK_STATUS_OK);
    check_ranges(base,0u);
    release(b,generation,result.lease_identifier);
    printf("PASS working-set lifecycle seed=%llu rounds=%u cases=7\n",(unsigned long long)working_set_seed,working_set_rounds);
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
	attached.arena_bytes = PACK_BYTES;
	attached.chunk_bytes = CHUNK;
	attached.chunk_count = 4u;
	assert(SparkWeightdMapCreate(client,&attached,-1,-1,&map) == SPARK_STATUS_OK);
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
	assert(SparkWeightdMapCreate(client,&attached,-1,-1,&map) == SPARK_STATUS_OK);
	spark_stub_cuda_fail_import_after(2u);
	assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_IO_ERROR);
	assert(first == 0u);
	assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdMapDestroy(map) == SPARK_STATUS_OK);
	assert(SparkWeightdMapCreate(client,&attached,-1,-1,&map) == SPARK_STATUS_OK);
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

static void write_full_chunk_fixture(const char *path,const char *manifest_path)
{
    FILE *pack = fopen(path,"wb"),*manifest = fopen(manifest_path,"wb");
    uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,3u,0u},i;
    uint8_t *data = malloc(CHUNK);
    assert(pack != 0 && manifest != 0 && data != 0);
    assert(fwrite(header,1u,sizeof(header),manifest) == sizeof(header));
    for (i=0u; i<3u; i++)
    {
        uint8_t record[48] = {0},digest[16];
        uint64_t offset = (uint64_t)i * CHUNK,bytes = CHUNK;
        SparkCk128Context ck;
        memset(data,(int)i + 1,CHUNK);
        SparkCk128Initialize(&ck);
        SparkCk128Update(&ck,data,CHUNK);
        SparkCk128Finalize(&ck,digest);
        memcpy(record + 4u,&i,4u);
        memcpy(record + 16u,&offset,8u);
        memcpy(record + 24u,&bytes,8u);
        memcpy(record + 32u,digest,16u);
        assert(fwrite(record,1u,sizeof(record),manifest) == sizeof(record));
        assert(fwrite(data,1u,CHUNK,pack) == CHUNK);
    }
    free(data);
    assert(fclose(pack) == 0 && fclose(manifest) == 0);
}

static void check_full_chunk(const void *address,uint32_t expert)
{
    const uint8_t *data = (const uint8_t *)address + (uint64_t)expert * CHUNK;
    uint32_t i;
    for (i=0u; i<CHUNK; i++)
        assert(data[i] == (uint8_t)(expert + 1u));
}

static void check_map_eviction(void)
{
    char root[] = "/tmp/weightd-evict-XXXXXX",path[256],manifest[272],socket_path[256],wset[272];
    TestServer state = {0};
    SparkWeightdServerConfig config = {0};
    SparkWeightdLazyAttachResult attached = {0};
    SparkWeightdWorkingSetResult result;
    SparkWeightdClient *a,*b;
    SparkWeightdMap *map;
    SparkWeightdExpertKey key = {0u,0u},pair[2] = {{0u,1u},{0u,2u}};
    pthread_t thread;
    uint64_t generation,base,other,first,second,leases[64],epoch;
    uint32_t i,allocations;
    int epoch_fd = -1;
    void *address,*second_address;
    assert(mkdtemp(root) != 0);
    snprintf(path,sizeof(path),"%s/pack",root);
    snprintf(manifest,sizeof(manifest),"%s.experts",path);
    snprintf(wset,sizeof(wset),"%s.wset",path);
    snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
    write_full_chunk_fixture(path,manifest);
    config.socket_path = socket_path;
    config.device_bytes_max = 3u * CHUNK;
    assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
    assert(pthread_create(&thread,0,run_server,&state) == 0);
    assert(SparkWeightdClientConnect(socket_path,&a,0) == SPARK_STATUS_OK);
    assert(SparkWeightdClientConnect(socket_path,&b,0) == SPARK_STATUS_OK);
    generation = attach_config(a,path,3u * CHUNK,2u,3u,&base);
    assert(attach_config(b,path,3u * CHUNK,2u,3u,&other) == generation);
    for (i=0u; i<3u; i++)
    {
        result = acquire(b,generation,i,SPARK_STATUS_OK);
        release(b,generation,result.lease_identifier);
    }
    assert(SparkWeightdClientEpochExport(a,generation,&epoch_fd,TIMEOUT) == SPARK_STATUS_OK);
    attached.status = SPARK_STATUS_OK;
    attached.arena_generation = generation;
    attached.arena_bytes = 3u * CHUNK;
    attached.chunk_bytes = CHUNK;
    attached.chunk_count = 3u;
    assert(SparkWeightdMapCreate(a,&attached,epoch_fd,-1,&map) == SPARK_STATUS_OK);
    assert(cudaMemcpy(&epoch,SparkWeightdMapEpochDevice(map),sizeof(epoch),cudaMemcpyDeviceToHost) == cudaSuccess && epoch > 0u);
    assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_OK);
    assert(SparkWeightdMapBeginUse(map,first,&address) == SPARK_STATUS_OK);
    check_full_chunk(address,0u);
    assert(SparkWeightdMapAcquire(map,&key,1u,&second,TIMEOUT) == SPARK_STATUS_OK);
    for (i=1u; i<3u; i++)
    {
        result = acquire(b,generation,i,SPARK_STATUS_OK);
        release(b,generation,result.lease_identifier);
    }
    assert(SparkWeightdMapBeginUse(map,second,&second_address) == SPARK_STATUS_OK && second_address == address);
    assert(SparkWeightdMapRecordCompletion(map,first,0) == SPARK_STATUS_OK);
    allocations = spark_stub_cuda_outstanding_allocs();
    assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_OK);
    assert(spark_stub_cuda_outstanding_allocs() == allocations);
    check_full_chunk(second_address,0u);
    assert(SparkWeightdMapRecordCompletion(map,second,0) == SPARK_STATUS_OK);
    spark_stub_cuda_fail_next_unmap();
    assert(SparkWeightdMapRelease(map,second,TIMEOUT) == SPARK_STATUS_IO_ERROR);
    assert(SparkWeightdClientAcquire(b,generation,pair,2u,&result,TIMEOUT) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_BUSY && first == 0u);
    assert(SparkWeightdMapRelease(map,second,TIMEOUT) == SPARK_STATUS_OK);
    assert(SparkWeightdClientAcquire(b,generation,pair,2u,&result,TIMEOUT) == SPARK_STATUS_OK);
    release(b,generation,result.lease_identifier);
    allocations = spark_stub_cuda_outstanding_allocs();
    for (i=0u; i<24u; i++)
    {
        key.expert = i % 3u;
        assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_OK);
        assert(SparkWeightdMapBeginUse(map,first,&address) == SPARK_STATUS_OK);
        check_full_chunk(address,key.expert);
        assert(SparkWeightdMapRecordCompletion(map,first,0) == SPARK_STATUS_OK);
        assert(SparkWeightdMapRelease(map,first,TIMEOUT) == SPARK_STATUS_OK);
        assert(spark_stub_cuda_outstanding_allocs() == allocations);
    }
    key.expert = 0u;
    for (i=0u; i<64u; i++)
        assert(SparkWeightdMapAcquire(map,&key,1u,&leases[i],TIMEOUT) == SPARK_STATUS_OK);
    assert(SparkWeightdMapAcquire(map,&key,1u,&first,TIMEOUT) == SPARK_STATUS_BUSY && first == 0u);
    assert(SparkWeightdMapBeginUse(map,leases[63],&address) == SPARK_STATUS_OK);
    allocations = spark_stub_cuda_outstanding_allocs();
    for (i=0u; i<63u; i++)
        assert(SparkWeightdMapRelease(map,leases[i],TIMEOUT) == SPARK_STATUS_OK);
    assert(spark_stub_cuda_outstanding_allocs() == allocations);
    check_full_chunk(address,0u);
    assert(SparkWeightdMapRecordCompletion(map,leases[63],0) == SPARK_STATUS_OK);
    assert(SparkWeightdMapRelease(map,leases[63],TIMEOUT) == SPARK_STATUS_OK);
    assert(SparkWeightdMapDestroy(map) == SPARK_STATUS_OK);
    SparkWeightdClientClose(a);
    SparkWeightdClientClose(b);
    __atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
    assert(pthread_join(thread,0) == 0);
    SparkWeightdServerDestroy(state.server);
    assert(spark_stub_cuda_outstanding_allocs() == 0u);
    assert(unlink(manifest) == 0 && unlink(path) == 0 && unlink(wset) == 0 && rmdir(root) == 0);
    puts("PASS partial map: actual eviction epoch, overlapping leases, failed unmap pin retention, 24 bounded reloads, 64-owner limit");
}

static void check_orphan(SparkWeightdClient *a,uint64_t generation,uint64_t base,const char *socket_path,const char *path)
{
	SparkWeightdWorkingSetResult pinned,result;
	SparkWeightdClient *replacement;
	uint64_t other_base;
	pinned = acquire(a,generation,0u,SPARK_STATUS_OK);
	SparkWeightdClientClose(a);
	assert(SparkWeightdClientConnect(socket_path,&replacement,0) == SPARK_STATUS_OK);
	assert(attach(replacement,path,&other_base) == generation);
	assert(SparkWeightdClientRelease(replacement,generation,pinned.lease_identifier,&result,TIMEOUT) == SPARK_STATUS_NOT_FOUND);
	result = acquire(replacement,generation,2u,SPARK_STATUS_OK);
	check_ranges(base,2u);
	release(replacement,generation,result.lease_identifier);
	SparkWeightdClientClose(replacement);
}

static void check_many_exports(void)
{
	char root[] = "/tmp/weightd-many-XXXXXX",path[256],manifest[272],socket_path[256],wset[272];
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
	snprintf(wset,sizeof(wset),"%s.wset",path);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	write_many(path,manifest);
	config.socket_path = socket_path;
	config.device_bytes_max = (66u * CHUNK);
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	assert(SparkWeightdClientConnect(socket_path,&client,0) == SPARK_STATUS_OK);
	generation = attach_config(client,path,(65u * CHUNK),65u,65u,&base);
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
	assert(unlink(manifest) == 0 && unlink(path) == 0 && unlink(wset) == 0 && rmdir(root) == 0);
}

static void check_pooled_attach(void)
{
	char root[] = "/tmp/weightd-pool-XXXXXX",path[256],manifest[272],socket_path[256],wset[272];
	SparkWeightdServerConfig config = {0};
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyAttachResult result;
	SparkWeightdExpertKey keys[65];
	SparkWeightdWorkingSetResult working;
	SparkWeightdExportBatch batch;
	TestServer state = {0};
	pthread_t thread;
	SparkWeightdClient *a,*b;
	uint64_t generation;
	uint32_t i;
	assert(mkdtemp(root) != 0);
	snprintf(path,sizeof(path),"%s/pack",root);
	snprintf(manifest,sizeof(manifest),"%s.experts",path);
	snprintf(wset,sizeof(wset),"%s.wset",path);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	write_many(path,manifest);
	config.socket_path = socket_path;
	config.device_bytes_max = (66u * CHUNK);
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	assert(SparkWeightdClientConnect(socket_path,&a,0) == SPARK_STATUS_OK);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (65u * CHUNK);
	memcpy(request.identity.model,"pooled-attach-test",19u);
	memset(request.identity.pack_sha256,'a',64u);
	assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	request.expert_pool_bytes = (66u * CHUNK);
	assert(SparkWeightdClientAttachLazy(a,&request,&result,TIMEOUT) == SPARK_STATUS_OK);
	assert(result.status == SPARK_STATUS_OK && result.expert_count == 65u);
	assert(result.pool_fd >= 0);
	assert(close(result.pool_fd) == 0);
	generation = result.arena_generation;
	for (i=0u; i<65u; i++)
		keys[i] = (SparkWeightdExpertKey){0u,i};
	assert(SparkWeightdClientAcquire(a,generation,keys,65u,&working,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdClientExportLeaseBatch(a,generation,working.lease_identifier,0u,&batch,TIMEOUT) == SPARK_STATUS_OK);
	assert(batch.status == SPARK_STATUS_OK && batch.lease_chunk_count == 0u && batch.batch_count == 0u);
	assert(SparkWeightdClientRelease(a,generation,working.lease_identifier,&working,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdClientConnect(socket_path,&b,0) == SPARK_STATUS_OK);
	memset(&result,0,sizeof(result));
	assert(SparkWeightdClientAttachLazy(b,&request,&result,TIMEOUT) == SPARK_STATUS_OK);
	assert(result.status == SPARK_STATUS_OK && result.arena_generation == generation);
	assert(result.pool_fd >= 0);
	assert(close(result.pool_fd) == 0);
	SparkWeightdClientClose(b);
	SparkWeightdClientClose(a);
	__atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	assert(spark_stub_cuda_outstanding_allocs() == 0u);
	assert(unlink(manifest) == 0 && unlink(path) == 0 && unlink(wset) == 0 && rmdir(root) == 0);
}

static SparkStatus reject_manifest(const SparkWeightdManifest *manifest,void *context)
{
	uint32_t *calls = (uint32_t *)context;
	assert(manifest->range_count != 0u);
	*calls += 1u;
	return(SPARK_STATUS_SCHEMA_ERROR);
}

static SparkStatus change_manifest_after_parse(const SparkWeightdManifest *manifest,void *context)
{
	FILE *file;
	uint8_t byte;
	(void)manifest;
	file = fopen((const char *)context,"r+b");
	assert(file != 0);
	assert(fseek(file,48,SEEK_SET) == 0 && fread(&byte,1u,1u,file) == 1u);
	byte ^= 1u;
	assert(fseek(file,48,SEEK_SET) == 0 && fwrite(&byte,1u,1u,file) == 1u);
	assert(fclose(file) == 0);
	return(SPARK_STATUS_OK);
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
	request.identity.arena_bytes = PACK_BYTES;
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
	snprintf(request.identity.model,sizeof(request.identity.model),"manifest-race-test");
	assert(SparkWeightdLazyPackCreateChecked(socket_path,&request,4u * CHUNK,TIMEOUT,change_manifest_after_parse,(void *)manifest_path,&pack) == SPARK_STATUS_HASH_MISMATCH);
	assert(pack == 0);
	assert(change_manifest_after_parse(0,(void *)manifest_path) == SPARK_STATUS_OK);
	snprintf(request.identity.model,sizeof(request.identity.model),"lazy-pack-test");
	assert(SparkWeightdLazyPackCreate(socket_path,&request,4u * CHUNK,TIMEOUT,&pack) == SPARK_STATUS_OK);
	assert(pack != 0 && pack->attached.resident_bytes == 0u);
	assert(SparkWeightdLazyPackSlice(pack,512u,1u,&pointer) == SPARK_STATUS_OK);
	assert(*(const uint8_t *)pointer == 255u);
	assert(SparkWeightdLazyPackSlice(pack,0u,64u,&pointer) == SPARK_STATUS_NOT_FOUND && pointer == 0);
	/* Prong 2 (hill-climb): the D2D arena spine copy must assemble exactly
	 * the bytes the proven file path produces for the same manifest - the
	 * lazy pack above already built its spine through MapSpineCopy (the
	 * pool is mapped in this flow), and this cross-check pins the two
	 * sources byte-identical before any later file mutation. */
	{
		SparkWeightdManifest verify;
		uint8_t *from_arena,*from_file;
		uint64_t capacity;
		int32_t verify_fd;
		assert(SparkWeightdManifestLoad(manifest_path,PACK_BYTES,&verify) == SPARK_STATUS_OK);
		capacity = verify.spine_allocation_bytes;
		if ( capacity != 0u )
		{
			SparkStatus copied;
			assert(posix_memalign((void **)&from_arena,256u,(size_t)capacity) == 0);
			assert(posix_memalign((void **)&from_file,256u,(size_t)capacity) == 0);
			memset(from_arena,0xa5,(size_t)capacity);
			memset(from_file,0xa5,(size_t)capacity);
			copied = SparkWeightdMapSpineCopy(pack->map,&verify,from_arena,capacity);
			/* UNSUPPORTED = the pool is not mapped in this configuration and
			 * the lazy pack took the proven file fallback; where the pool IS
			 * mapped, the D2D result must be byte-identical to the file. */
			if ( copied == SPARK_STATUS_OK )
			{
				verify_fd = open(path,O_RDONLY);
				assert(verify_fd >= 0);
				assert(SparkWeightdSpineLoad(verify_fd,&verify,PACK_BYTES,request.identity.pack_sha256,from_file,capacity) == SPARK_STATUS_OK);
				assert(memcmp(from_arena,from_file,(size_t)capacity) == 0);
				(void)close(verify_fd);
			}
			else
				assert(copied == SPARK_STATUS_UNSUPPORTED);
			free(from_arena);
			free(from_file);
		}
		SparkWeightdManifestDestroy(&verify);
	}
	assert(SparkWeightdMapAcquire(pack->map,&key,1u,&lease,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_BUSY);
	assert(SparkWeightdLazyPackSlice(pack,512u,1u,&pointer) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkWeightdMapRelease(pack->map,lease,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_OK);
}

static void check_budget_contract(SparkWeightdClient *client,const char *path)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyAttachResult result;
	SparkWeightdDetachResult detached;
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = PACK_BYTES;
	memcpy(request.identity.model,"working-set-test",17u);
	memset(request.identity.pack_sha256,'a',64u);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	request.expert_pool_bytes = UINT64_MAX;
	assert(SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(result.arena_count == 0u);
	request.expert_pool_bytes = 5u * CHUNK;
	assert(SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(result.arena_count == 0u);
	request.expert_pool_bytes = 2u * CHUNK;
	assert(SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdClientDetach(client,result.arena_generation,&detached,TIMEOUT) == SPARK_STATUS_OK);
	assert(detached.status == SPARK_STATUS_OK);
	request.expert_pool_bytes = 3u * CHUNK;
	assert(SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT) == SPARK_STATUS_INVALID_ARGUMENT);
	request.expert_pool_bytes = CHUNK;
	assert(SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT) == SPARK_STATUS_INVALID_ARGUMENT);
	request.expert_pool_bytes = 2u * CHUNK;
	assert(SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdClientDetach(client,result.arena_generation,&detached,TIMEOUT) == SPARK_STATUS_OK);
	assert(detached.status == SPARK_STATUS_OK);
	assert(result.expert_pool_bytes == 2u * CHUNK);
}


static void check_spine_residency_modes(const char *path,const char *manifest_path,const char *socket_path)
{
	uint32_t full;
	for (full=0u; full<2u; full++)
	{
		TestServer state = {0};
		SparkWeightdServerConfig config = {0};
		SparkWeightdLazyAttachRequest request = {0};
		SparkWeightdLazyPack *pack = 0;
		SparkWeightdManifest manifest;
		SparkWeightdExpertKey key = {0u,0u};
		const void *pointer;
		pthread_t thread;
		uint64_t lease;
		config.socket_path = socket_path;
		config.device_bytes_max = (full ? 5u : 2u) * CHUNK;
		request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
		request.identity.arena_bytes = PACK_BYTES;
		snprintf(request.identity.model,sizeof(request.identity.model),"spine-mode-%u",full);
		assert(SparkSha256File(path,request.identity.pack_sha256) == SPARK_STATUS_OK);
		assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
		snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
		request.expert_pool_bytes = config.device_bytes_max;
		assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
		assert(pthread_create(&thread,0,run_server,&state) == 0);
		assert(SparkWeightdLazyPackCreate(socket_path,&request,4u * CHUNK,TIMEOUT,&pack) == SPARK_STATUS_OK);
		assert(pack->attached.resident_bytes == (full ? 4u * CHUNK : 0u));
		assert(SparkWeightdManifestLoad(manifest_path,PACK_BYTES,&manifest) == SPARK_STATUS_OK);
		for (uint32_t i=0u; i<manifest.spine_count; i++)
		{
			const SparkWeightdSpan *span = &manifest.spine[i];
			uint8_t *expected = malloc((size_t)span->bytes);
			int fd = open(path,O_RDONLY);
			assert(expected != 0 && fd >= 0);
			assert(pread(fd,expected,(size_t)span->bytes,(off_t)span->offset) == (ssize_t)span->bytes);
			assert(SparkWeightdLazyPackSlice(pack,span->offset,span->bytes,&pointer) == SPARK_STATUS_OK);
			assert(memcmp(pointer,expected,(size_t)span->bytes) == 0);
			free(expected);
			assert(close(fd) == 0);
		}
		SparkWeightdManifestDestroy(&manifest);
		assert(SparkWeightdMapAcquire(pack->map,&key,1u,&lease,TIMEOUT) == SPARK_STATUS_OK);
		assert(SparkWeightdMapRelease(pack->map,lease,TIMEOUT) == SPARK_STATUS_OK);
		assert(SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_OK);
		__atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
		assert(pthread_join(thread,0) == 0);
		SparkWeightdServerDestroy(state.server);
		assert(spark_stub_cuda_outstanding_allocs() == 0u);
	}
}

int main(int argc,char **argv)
{
	char root[] = "/tmp/weightd-set-XXXXXX",path[256],manifest[272],socket_path[256],wset[272];
	TestServer state = {0};
	SparkWeightdServerConfig config = {0};
	SparkWeightdClient *a,*b;
	pthread_t thread;
	uint64_t generation,base,other_base;
	if ( argc > 1 ) working_set_seed = strtoull(argv[1],0,0);
	if ( argc > 2 ) working_set_rounds = (uint32_t)strtoul(argv[2],0,0);
	assert(mkdtemp(root) != 0);
	snprintf(path,sizeof(path),"%s/pack",root);
	snprintf(manifest,sizeof(manifest),"%s.experts",path);
	snprintf(wset,sizeof(wset),"%s.wset",path);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	write_fixture(path,manifest);
	check_spine_load(path,manifest);
	check_spine_residency_modes(path,manifest,socket_path);
	config.socket_path = socket_path;
	config.device_bytes_max = (4u * CHUNK);
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	assert(SparkWeightdClientConnect(socket_path,&a,0) == SPARK_STATUS_OK);
	assert(SparkWeightdClientConnect(socket_path,&b,0) == SPARK_STATUS_OK);
	check_budget_contract(a,path);
	generation = attach(a,path,&base);
	assert(attach(b,path,&other_base) == generation && other_base == base);
	check_two_clients(a,b,generation,base);
	check_transaction(a,generation,base);
	check_seeded_leases(a,b,generation,base);
	check_map_lifetime(a,generation,base);
	check_orphan(a,generation,base,socket_path,path);
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
	assert(unlink(manifest) == 0 && unlink(path) == 0 && unlink(wset) == 0 && rmdir(root) == 0);
	check_map_eviction();
	check_many_exports();
	check_pooled_attach();
	puts("PASS working-set IPC: all ranges, leases, rollback, scoped imports, 65-chunk exports and pooled single-alloc attach");
	return(0);
}
