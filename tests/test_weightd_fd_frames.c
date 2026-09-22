#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#include <sys/mman.h>
#include <stdint.h>
#include <stdlib.h>
#include "sparkpipe/spark_weightd.h"

static uint32_t mesh_map_forced,mesh_map_offset,mesh_map_failure,mesh_map_calls;
static uint8_t *mesh_map_raw;
static void *test_mmap(void *address,size_t bytes,int protection,int flags,int fd,off_t offset)
{
    if (mesh_map_forced == 0u) return mmap(address,bytes,protection,flags,fd,offset);
    mesh_map_calls++;
    if (mesh_map_calls == mesh_map_failure) { errno = ENOMEM;return MAP_FAILED; }
    if (mesh_map_calls == 1u)
    {
        size_t alignment = SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES;
        uint8_t *reservation = mmap(0,bytes + 2u * alignment,PROT_NONE,MAP_PRIVATE | MAP_ANONYMOUS,-1,0);
        assert(reservation != MAP_FAILED);
        mesh_map_raw = (uint8_t *)(((uintptr_t)reservation + alignment - 1u) & ~(uintptr_t)(alignment - 1u)) + mesh_map_offset;
        size_t prefix = (size_t)(mesh_map_raw - reservation);
        size_t suffix = bytes + 2u * alignment - prefix - bytes;
        if (prefix != 0u) assert(munmap(reservation,prefix) == 0);
        if (suffix != 0u) assert(munmap(mesh_map_raw + bytes,suffix) == 0);
        return mmap(mesh_map_raw,bytes,protection,flags | MAP_FIXED,fd,offset);
    }
    return mmap(address,bytes,protection,flags,fd,offset);
}

static pthread_mutex_t read_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t read_changed = PTHREAD_COND_INITIALIZER;
static unsigned read_blocked,read_entered;
static size_t read_limit;
static unsigned fail_record_rename;

static ssize_t test_pread(int fd,void *buffer,size_t bytes,off_t offset)
{
	pthread_mutex_lock(&read_mutex);
	if (read_blocked != 0u)
	{
		read_entered = 1u;
		pthread_cond_broadcast(&read_changed);
		while (read_blocked != 0u)
			pthread_cond_wait(&read_changed,&read_mutex);
	}
	pthread_mutex_unlock(&read_mutex);
	if (read_limit != 0u && bytes > read_limit)
		bytes = read_limit;
	return pread(fd,buffer,bytes,offset);
}

static int test_rename(const char *source,const char *destination)
{
	if (__atomic_load_n(&fail_record_rename,__ATOMIC_SEQ_CST) != 0u)
	{
		errno = EIO;
		return -1;
	}
	return rename(source,destination);
}

#define mmap test_mmap
#define rename test_rename
#define pread test_pread
#include "../runtime/spark_weightd.c"
#undef pread
#undef rename
#undef mmap

void spark_stub_cuda_fail_alloc_after(uint32_t calls);

static uint32_t fd_count(void)
{
	uint32_t i,count = 0u;
	for (i=0u; i<1024u; i++)
		count += fcntl((int32_t)i,F_GETFD) >= 0;
	return(count);
}

static void check_mesh_mapping(uint32_t raw_offset,uint32_t failure,uint64_t advertised,uint64_t file_bytes)
{
    SparkWeightdClient client = {0};
    SparkWeightdLazyAttachRequest request = {0};
    SparkWeightdLazyAttachResult result;
    SparkWeightdIpcAttachLazyResult wire = {0};
    union { struct cmsghdr align; unsigned char bytes[CMSG_SPACE(2u * sizeof(int))]; } control = {0};
    struct msghdr message = {0};
    struct iovec vector = {&wire,sizeof(wire)};
    struct cmsghdr *header;
    char path[] = "/tmp/spark-mesh-mapping-XXXXXX";
    int sockets[2],source = mkstemp(path),pool = open("/dev/null",O_RDONLY),fds[2];
    uint32_t before;
    unsigned char first = 0x31u,last = 0xe7u,observed;
    assert(source >= 0 && pool >= 0 && unlink(path) == 0);
    assert(ftruncate(source,(off_t)file_bytes) == 0);
    assert(pwrite(source,&first,1u,0) == 1 && pwrite(source,&last,1u,(off_t)file_bytes - 1) == 1);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    client.fd = sockets[0];
    request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    request.identity.arena_bytes = 4096u;
    memcpy(request.identity.model,"mesh-map",9u);
    memset(request.identity.pack_sha256,'a',64u);
    memcpy(request.pack_path,"/unused",8u);
    request.expert_pool_bytes = 4096u;
    SparkWeightdBuildHeader((uint8_t *)&wire,SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY_RESULT,1u);
    wire.mesh_ready = 1u;
    wire.pool_fd_staged = 1u;
    wire.mesh_send_buffer_bytes = advertised;
    message.msg_iov = &vector;
    message.msg_iovlen = 1u;
    message.msg_control = control.bytes;
    message.msg_controllen = sizeof(control.bytes);
    header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(fds));
    fds[0] = source;fds[1] = pool;
    memcpy(CMSG_DATA(header),fds,sizeof(fds));
    before = fd_count();
    assert(sendmsg(sockets[1],&message,0) == (ssize_t)sizeof(wire));
    mesh_map_offset = raw_offset;
    mesh_map_failure = failure;
    mesh_map_calls = 0u;
    mesh_map_raw = 0;
    mesh_map_forced = 1u;
    SparkStatus status = SparkWeightdClientAttachLazy(&client,&request,&result,UINT64_C(1000000000));
    mesh_map_forced = 0u;
    SparkStatus expected = advertised != SPARK_WEIGHTD_MESH_REGION_BYTES || file_bytes < advertised ?
        SPARK_STATUS_SCHEMA_ERROR : failure != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
    assert(status == expected);
    if (status == SPARK_STATUS_OK)
    {
        unsigned char *mapped = result.mesh_mapping;
        assert(mapped != 0 && (uintptr_t)mapped % SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES == 0u);
        assert(mapped[0] == first && mapped[SPARK_WEIGHTD_MESH_REGION_BYTES - 1u] == last);
        mapped[0] = 0x79u;
        assert(pread(source,&observed,1u,0) == 1 && observed == 0x79u);
        assert(result.mesh_send_buffer_addr == (uint64_t)(uintptr_t)mapped);
        if (mesh_map_raw != mapped)
        { errno = 0;assert(msync(mesh_map_raw,(size_t)sysconf(_SC_PAGESIZE),MS_SYNC) == -1 && errno == ENOMEM); }
        errno = 0;
        assert(msync(mapped + SPARK_WEIGHTD_MESH_REGION_BYTES,(size_t)sysconf(_SC_PAGESIZE),MS_SYNC) == -1 && errno == ENOMEM);
        assert(munmap(mapped,SPARK_WEIGHTD_MESH_REGION_BYTES) == 0 && close(result.pool_fd) == 0);
        errno = 0;
        assert(msync(mapped,(size_t)sysconf(_SC_PAGESIZE),MS_SYNC) == -1 && errno == ENOMEM);
    }
    else
    {
        assert(result.mesh_mapping == 0);
        if (mesh_map_raw != 0)
        { errno = 0;assert(msync(mesh_map_raw,(size_t)sysconf(_SC_PAGESIZE),MS_SYNC) == -1 && errno == ENOMEM); }
    }
    assert(fd_count() == before);
    assert(close(source) == 0 && close(pool) == 0 && close(sockets[0]) == 0 && close(sockets[1]) == 0);
}

static void check_mesh_mapping_offsets(void)
{
    uint32_t page = (uint32_t)sysconf(_SC_PAGESIZE),cases = 0u;
    assert(page > 0u && page <= SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES && SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES % page == 0u);
    for (uint32_t offset=page; offset<SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES; offset+=page)
    { check_mesh_mapping(offset,0u,SPARK_WEIGHTD_MESH_REGION_BYTES,SPARK_WEIGHTD_MESH_REGION_BYTES); cases++; }
    check_mesh_mapping(0u,0u,SPARK_WEIGHTD_MESH_REGION_BYTES,SPARK_WEIGHTD_MESH_REGION_BYTES); cases++;
    for (uint32_t failure=1u; failure<=2u; failure++)
    { check_mesh_mapping(page,failure,SPARK_WEIGHTD_MESH_REGION_BYTES,SPARK_WEIGHTD_MESH_REGION_BYTES); cases++; }
    check_mesh_mapping(page,0u,SPARK_WEIGHTD_MESH_REGION_BYTES - 1u,SPARK_WEIGHTD_MESH_REGION_BYTES);
    check_mesh_mapping(page,0u,UINT64_MAX,SPARK_WEIGHTD_MESH_REGION_BYTES);
    check_mesh_mapping(page,0u,SPARK_WEIGHTD_MESH_REGION_BYTES,SPARK_WEIGHTD_MESH_REGION_BYTES - 1u);
    printf("PASS mesh mapping: %u actual mmap offsets/faults/schema cases; offset zero, final byte, shared writes, trimmed reservation, FD conservation\n",cases + 3u);
}

static void send_fds(int32_t socket_fd,int32_t source,uint32_t count)
{
	union { struct cmsghdr align; uint8_t bytes[CMSG_SPACE(65u * sizeof(int))]; } control;
	struct msghdr message = {0};
	struct iovec vector;
	struct cmsghdr *header;
	uint8_t byte = 1u;
	uint32_t i;
	vector.iov_base = &byte;
	vector.iov_len = 1u;
	message.msg_iov = &vector;
	message.msg_iovlen = 1u;
	message.msg_control = control.bytes;
	message.msg_controllen = CMSG_SPACE(count * sizeof(int));
	memset(&control,0,sizeof(control));
	header = CMSG_FIRSTHDR(&message);
	header->cmsg_level = SOL_SOCKET;
	header->cmsg_type = SCM_RIGHTS;
	header->cmsg_len = CMSG_LEN(count * sizeof(int));
	for (i=0u; i<count; i++)
		memcpy((uint8_t *)CMSG_DATA(header) + (i * sizeof(int)),&source,sizeof(source));
	assert(sendmsg(socket_fd,&message,0) == 1);
}

static void check_frame(uint32_t sent,uint32_t capacity,SparkStatus expected)
{
	int32_t sockets[2],source,fds[64];
	SparkWeightdClient client = {0};
	uint32_t before,received = 0u,i;
	uint8_t byte;
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
	source = open("/dev/null",O_RDONLY);
	assert(source >= 0);
	client.fd = sockets[0];
	before = fd_count();
	send_fds(sockets[1],source,sent);
	assert(SparkWeightdClientReadFrameWithFds(&client,&byte,1u,SparkWeightdMonotonicTimeNs() + UINT64_C(1000000000),fds,capacity,&received) == expected);
	assert(received == (expected == SPARK_STATUS_OK ? sent : 0u));
	for (i=0u; i<received; i++)
	{
		assert((fcntl(fds[i],F_GETFD) & FD_CLOEXEC) != 0);
		assert(close(fds[i]) == 0);
	}
	if ( fd_count() != before )
		fprintf(stderr,"FD leak: sent=%u capacity=%u received=%u before=%u after=%u\n",sent,capacity,received,before,fd_count());
	assert(fd_count() == before);
	assert(close(source) == 0 && close(sockets[0]) == 0 && close(sockets[1]) == 0);
}

static void check_reported_truncation(void)
{
	union { struct cmsghdr align; uint8_t bytes[CMSG_SPACE(sizeof(int))]; } control;
	struct msghdr message = {0};
	struct cmsghdr *header;
	int32_t fd = dup(STDERR_FILENO),fds[1];
	uint32_t count = 0u;
	assert(fd >= 0);
	message.msg_control = control.bytes;
	message.msg_controllen = sizeof(control);
	message.msg_flags = MSG_CTRUNC;
	header = CMSG_FIRSTHDR(&message);
	header->cmsg_level = SOL_SOCKET;
	header->cmsg_type = SCM_RIGHTS;
	header->cmsg_len = CMSG_LEN(sizeof(int));
	memcpy(CMSG_DATA(header),&fd,sizeof(fd));
	assert(SparkWeightdReceiveFds(&message,fds,1u,&count) == SPARK_STATUS_IO_ERROR);
	assert(count == 1u && fds[0] == fd);
	assert(close(fds[0]) == 0);
}

static void check_lease_frame_shape(void)
{
	SparkWeightdIpcExportLease request = {0};
	SparkWeightdIpcExportLeaseResult response = {0},bad;
	SparkWeightdBuildHeader((uint8_t *)&request,SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE,1u);
	SparkWeightdBuildHeader((uint8_t *)&response,SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE_RESULT,1u);
	request.arena_generation = response.base.arena_generation = 2u;
	request.lease_identifier = response.lease_identifier = 3u;
	response.base.chunk_bytes = 2097152u;
	response.base.chunk_count = 3u;
	response.base.batch_count = response.lease_chunk_count = 2u;
	response.chunk_indices[1] = 2u;
	assert(SparkWeightdValidateLeaseExport(&request,&response,2u) == SPARK_STATUS_OK);
	bad = response;
	bad.chunk_indices[1] = 0u;
	assert(SparkWeightdValidateLeaseExport(&request,&bad,2u) == SPARK_STATUS_SCHEMA_ERROR);
	bad = response;
	bad.lease_identifier++;
	assert(SparkWeightdValidateLeaseExport(&request,&bad,2u) == SPARK_STATUS_SCHEMA_ERROR);
	bad = response;
	bad.base.arena_generation++;
	assert(SparkWeightdValidateLeaseExport(&request,&bad,2u) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkWeightdValidateLeaseExport(&request,&response,1u) == SPARK_STATUS_SCHEMA_ERROR);
	bad = response;
	bad.base.batch_offset = 2u;
	assert(SparkWeightdValidateLeaseExport(&request,&bad,2u) == SPARK_STATUS_SCHEMA_ERROR);
}

static void check_short_range_reads(void)
{
	char path[] = "/tmp/weightd-short-read-XXXXXX";
	uint8_t source[512],destination[512];
	SparkWeightdArena arena = {0};
	SparkWeightdRange ranges[2] = {0};
	int fd = mkstemp(path);
	assert(fd >= 0);
	for (uint32_t i=0u; i<sizeof(source); i++)
		source[i] = (uint8_t)(i * 17u + 3u);
	assert(write(fd,source,sizeof(source)) == sizeof(source));
	for (uint32_t i=0u; i<2u; i++)
	{
		SparkCk128Context digest;
		ranges[i].offset = i * 256u;
		ranges[i].bytes = 256u;
		SparkCk128Initialize(&digest);
		SparkCk128Update(&digest,source + ranges[i].offset,256u);
		SparkCk128Finalize(&digest,ranges[i].digest);
	}
	arena.staging = malloc(SPARK_WEIGHTD_ARENA_STAGING_BYTES);
	arena.device_base = destination;
	assert(arena.staging != 0);
	memset(destination,0,sizeof(destination));
	read_limit = 19u;
	assert(SparkWeightdLoadRangeGroup(&arena,fd,ranges,2u) == SPARK_STATUS_OK);
	assert(memcmp(source,destination,sizeof(source)) == 0);
	ranges[1].digest[0] ^= 1u;
	assert(SparkWeightdLoadRangeGroup(&arena,fd,ranges,2u) == SPARK_STATUS_HASH_MISMATCH);
	read_limit = 0u;
	free(arena.staging);
	assert(close(fd) == 0 && unlink(path) == 0);
}

static void check_exchange_timeout_poison(void)
{
	int sockets[2];
	SparkWeightdClient client = {0};
	SparkWeightdIpcHello request = {0};
	SparkWeightdIpcHelloAck response;
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
	client.fd = sockets[0];
	SparkWeightdBuildHeader((uint8_t *)&request,SPARK_WEIGHTD_IPC_KIND_HELLO,1u);
	assert(SparkWeightdClientExchange(&client,&request,sizeof(request),&response,
		sizeof(response),UINT64_C(1000000)) == SPARK_STATUS_BUSY);
	assert(client.fd == -1 && SparkWeightdClientAlive(&client) == 0u);
	assert(close(sockets[1]) == 0);
}

typedef struct ProgressServer
{
	SparkWeightdServer *server;
	volatile sig_atomic_t stop;
} ProgressServer;

typedef struct ProgressAcquire
{
	SparkWeightdClient *client;
	uint64_t generation,timeout;
	SparkStatus status;
	SparkWeightdWorkingSetResult result;
	SparkWeightdExpertKey key;
} ProgressAcquire;

static void *progress_server(void *context)
{
	ProgressServer *state = context;
	assert(SparkWeightdServerRun(state->server,&state->stop) == SPARK_STATUS_OK);
	return 0;
}

static void *progress_acquire(void *context)
{
	ProgressAcquire *state = context;
	state->status = SparkWeightdClientAcquire(state->client,state->generation,
		&state->key,1u,&state->result,state->timeout);
	return 0;
}

static void block_reads(void)
{
	pthread_mutex_lock(&read_mutex);
	read_blocked = 1u;
	read_entered = 0u;
	pthread_mutex_unlock(&read_mutex);
}

static void wait_for_read(void)
{
	struct timespec deadline;
	assert(clock_gettime(CLOCK_REALTIME,&deadline) == 0);
	deadline.tv_sec += 5;
	pthread_mutex_lock(&read_mutex);
	while (read_entered == 0u)
		assert(pthread_cond_timedwait(&read_changed,&read_mutex,&deadline) == 0);
	pthread_mutex_unlock(&read_mutex);
}

static void unblock_reads(void)
{
	pthread_mutex_lock(&read_mutex);
	read_blocked = 0u;
	pthread_cond_broadcast(&read_changed);
	pthread_mutex_unlock(&read_mutex);
}

static void check_cold_control_progress(void)
{
	char root[] = "/tmp/weightd-progress-XXXXXX",path[256],manifest_path[272],socket_path[256],wset_path[272];
	SparkWeightdServerConfig config = {0};
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyAttachResult attached;
	SparkWeightdHelloResult first_hello,next_hello;
	SparkWeightdClient *client,*probe;
	ProgressServer server = {0};
	ProgressAcquire acquire = {0};
	pthread_t server_thread,acquire_thread;
	uint8_t source[768];
	uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,3u,0u};
	FILE *pack,*manifest,*recording;
	uint32_t recorded[4] = {0u,1u,0u,0u};
	assert(mkdtemp(root) != 0);
	snprintf(path,sizeof(path),"%s/pack",root);
	snprintf(manifest_path,sizeof(manifest_path),"%s.experts",path);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	snprintf(wset_path,sizeof(wset_path),"%s.wset",path);
	recording = fopen(wset_path,"wb");
	assert(recording != 0 && fwrite(recorded,sizeof(uint32_t),2u,recording) == 2u);
	assert(fclose(recording) == 0);
	pack = fopen(path,"wb");
	manifest = fopen(manifest_path,"wb");
	assert(pack != 0 && manifest != 0);
	memset(source,0x5a,sizeof(source));
	assert(fwrite(source,1u,sizeof(source),pack) == sizeof(source));
	assert(fwrite(header,1u,sizeof(header),manifest) == sizeof(header));
	for (uint32_t i=0u; i<3u; i++)
	{
		uint8_t record[48] = {0},digest[16];
		uint64_t offset = i * 256u,bytes = 256u;
		SparkCk128Context ck;
		SparkCk128Initialize(&ck);
		SparkCk128Update(&ck,source + offset,(size_t)bytes);
		SparkCk128Finalize(&ck,digest);
		memcpy(record + 4u,&i,sizeof(i));
		memcpy(record + 16u,&offset,sizeof(offset));
		memcpy(record + 24u,&bytes,sizeof(bytes));
		memcpy(record + 32u,digest,sizeof(digest));
		assert(fwrite(record,1u,sizeof(record),manifest) == sizeof(record));
	}
	assert(fclose(pack) == 0 && fclose(manifest) == 0);
	config.socket_path = socket_path;
	config.device_bytes_max = UINT64_C(8388608);
	assert(SparkWeightdServerCreate(&config,&server.server) == SPARK_STATUS_OK);
	assert(pthread_create(&server_thread,0,progress_server,&server) == 0);
	assert(SparkWeightdClientConnect(socket_path,&client,&first_hello) == SPARK_STATUS_OK);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = sizeof(source);
	memcpy(request.identity.model,"control-progress",17u);
	memset(request.identity.pack_sha256,'a',64u);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	request.expert_pool_bytes = UINT64_C(2097152);
	for (uint32_t bytes=0u; bytes<=8u; bytes+=4u)
	{
		uint32_t invalid[2] = {0u,99u};
		recording = fopen(wset_path,"wb");
		assert(recording != 0 && fwrite(invalid,1u,bytes,recording) == bytes);
		assert(fclose(recording) == 0);
		assert(SparkWeightdClientAttachLazy(client,&request,&attached,UINT64_C(1000000000)) == SPARK_STATUS_SCHEMA_ERROR);
		assert(attached.arena_count == 0u && attached.resident_bytes == 0u);
	}
	recording = fopen(wset_path,"wb");
	assert(recording != 0 && fwrite(recorded,sizeof(uint32_t),2u,recording) == 2u);
	assert(fclose(recording) == 0);
	spark_stub_cuda_fail_alloc_after(3u);
	assert(SparkWeightdClientAttachLazy(client,&request,&attached,UINT64_C(1000000000)) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(attached.arena_count == 0u && attached.resident_bytes == 0u);
	assert(SparkWeightdClientAttachLazy(client,&request,&attached,UINT64_C(1000000000)) == SPARK_STATUS_OK);
	assert(attached.pool_fd >= 0 && close(attached.pool_fd) == 0);
	assert(server.server->arenas[0].recorded_count == 1u);
	assert(server.server->arenas[0].recorded_keys[0] == UINT64_C(1));
	acquire.client = client;
	acquire.generation = attached.arena_generation;
	acquire.timeout = UINT64_C(5000000000);
	block_reads();
	assert(pthread_create(&acquire_thread,0,progress_acquire,&acquire) == 0);
	wait_for_read();
	assert(SparkWeightdClientConnect(socket_path,&probe,&next_hello) == SPARK_STATUS_OK);
	assert(next_hello.daemon_generation == first_hello.daemon_generation);
	assert(SparkWeightdClientMeshWrite(probe,0u,0u,0u,8u,UINT64_C(100000000)) == SPARK_STATUS_UNSUPPORTED);
	assert(SparkWeightdClientMeshBroadcast(probe,1u,0u,0u,8u,1u,0u,UINT64_C(100000000)) == SPARK_STATUS_BUSY);
	assert(SparkWeightdClientAlive(probe) != 0u);
	assert(read_blocked != 0u);
	SparkWeightdClientClose(probe);
	for (uint32_t i=0u; i<SPARK_WEIGHTD_CONNECTION_COUNT_MAX + 16u; i++)
	{
		assert(SparkWeightdClientConnect(socket_path,&probe,0) == SPARK_STATUS_OK);
		SparkWeightdClientClose(probe);
	}
	assert(read_blocked != 0u);
	unblock_reads();
	assert(pthread_join(acquire_thread,0) == 0 && acquire.status == SPARK_STATUS_OK);
	assert(memcmp((void *)(uintptr_t)attached.device_handle,source,256u) == 0);
	assert(SparkWeightdClientRelease(client,attached.arena_generation,
		acquire.result.lease_identifier,&acquire.result,UINT64_C(1000000000)) == SPARK_STATUS_OK);
	recording = fopen(wset_path,"rb");
	assert(recording != 0 && fread(recorded,sizeof(uint32_t),4u,recording) == 4u);
	assert(fgetc(recording) == EOF && !ferror(recording) && fclose(recording) == 0);
	assert(recorded[0] == 0u && recorded[1] == 1u && recorded[2] == 0u && recorded[3] == 0u);
	acquire.key.expert = 1u;
	acquire.timeout = UINT64_C(50000000);
	block_reads();
	assert(pthread_create(&acquire_thread,0,progress_acquire,&acquire) == 0);
	wait_for_read();
	assert(pthread_join(acquire_thread,0) == 0 && acquire.status == SPARK_STATUS_BUSY);
	assert(SparkWeightdClientAlive(client) == 0u);
	unblock_reads();
	SparkWeightdClientClose(client);
	assert(SparkWeightdClientConnect(socket_path,&probe,0) == SPARK_STATUS_OK);
	assert(SparkWeightdClientAttachLazy(probe,&request,&attached,UINT64_C(1000000000)) == SPARK_STATUS_OK);
	assert(attached.pool_fd >= 0 && close(attached.pool_fd) == 0);
	assert(SparkWeightdClientAcquire(probe,attached.arena_generation,&acquire.key,1u,
		&acquire.result,UINT64_C(1000000000)) == SPARK_STATUS_OK);
	assert(SparkWeightdClientRelease(probe,attached.arena_generation,
		acquire.result.lease_identifier,&acquire.result,UINT64_C(1000000000)) == SPARK_STATUS_OK);
	acquire.key.expert = 2u;
	__atomic_store_n(&fail_record_rename,1u,__ATOMIC_SEQ_CST);
	assert(SparkWeightdClientAcquire(probe,attached.arena_generation,&acquire.key,1u,
		&acquire.result,UINT64_C(1000000000)) == SPARK_STATUS_IO_ERROR);
	assert(acquire.result.status == SPARK_STATUS_IO_ERROR && acquire.result.lease_identifier == 0u);
	assert(server.server->arenas[0].recorded_count == 2u);
	for (uint32_t i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		assert(server.server->arenas[0].leases->leases[i].count == 0u);
	for (uint32_t i=0u; i<server.server->arenas[0].manifest.group_count; i++)
		assert(server.server->arenas[0].leases->pins[i] == 0u);
	recording = fopen(wset_path,"rb");
	assert(recording != 0 && fread(recorded,sizeof(uint32_t),4u,recording) == 4u);
	assert(fgetc(recording) == EOF && !ferror(recording) && fclose(recording) == 0);
	assert(recorded[0] == 0u && recorded[1] == 1u && recorded[2] == 0u && recorded[3] == 0u);
	__atomic_store_n(&fail_record_rename,0u,__ATOMIC_SEQ_CST);
	assert(SparkWeightdClientAcquire(probe,attached.arena_generation,&acquire.key,1u,
		&acquire.result,UINT64_C(1000000000)) == SPARK_STATUS_OK);
	assert(server.server->arenas[0].recorded_count == 3u);
	assert(SparkWeightdClientRelease(probe,attached.arena_generation,
		acquire.result.lease_identifier,&acquire.result,UINT64_C(1000000000)) == SPARK_STATUS_OK);
	SparkWeightdClientClose(probe);
	__atomic_store_n(&server.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(server_thread,0) == 0);
	SparkWeightdServerDestroy(server.server);
	assert(unlink(path) == 0 && unlink(manifest_path) == 0 && unlink(wset_path) == 0 && rmdir(root) == 0);
}

int main(void)
{
	check_mesh_mapping_offsets();
	check_frame(2u,2u,SPARK_STATUS_OK);
	check_frame(64u,64u,SPARK_STATUS_OK);
	check_frame(2u,1u,SPARK_STATUS_IO_ERROR);
	check_frame(65u,64u,SPARK_STATUS_IO_ERROR);
	check_reported_truncation();
	check_lease_frame_shape();
	check_short_range_reads();
	check_exchange_timeout_poison();
	check_cold_control_progress();
	puts("PASS FD frames and cold progress: short reads, failed exchange isolation, concurrent HELLO and orphan cleanup");
	return(0);
}
