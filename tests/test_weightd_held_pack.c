#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "tests/test_stderr_capture.h"

#define PACK_BYTES ((3u * 1024u * 1024u) + 512u)
#define TIMEOUT UINT64_C(10000000000)

void spark_stub_cuda_discrete_ordering(uint32_t enabled);

typedef struct TestServer
{
	SparkWeightdServer *server;
	volatile sig_atomic_t stop;
} TestServer;

typedef struct OldServer
{
	int listen_fd;
	uint32_t closed_on_kind;
} OldServer;

static void *run_server(void *data)
{
	TestServer *state = data;
	assert(SparkWeightdServerRun(state->server,&state->stop) == SPARK_STATUS_OK);
	return(0);
}

static void fill(uint8_t *bytes,uint64_t count,uint32_t seed)
{
	uint32_t value = seed;
	for ( uint64_t index = 0u; index < count; index++ )
	{
		value ^= value << 13;
		value ^= value >> 17;
		value ^= value << 5;
		bytes[index] = (uint8_t)value;
	}
}

static void write_file(const char *path,const void *data,size_t bytes)
{
	FILE *file = fopen(path,"wb");
	assert(file != 0 && fwrite(data,1u,bytes,file) == bytes && fclose(file) == 0);
}

static void write_pack(const char *path,const uint8_t *bytes,uint32_t ranges)
{
	char manifest_path[512];
	uint8_t manifest[16u + 48u];
	const uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,SPARK_WEIGHTD_RANGE_MANIFEST_VERSION,ranges,0u};
	const uint64_t span[2] = {0u,4096u};
	write_file(path,bytes,PACK_BYTES);
	memset(manifest,0,sizeof(manifest));
	memcpy(manifest,header,sizeof(header));
	memcpy(manifest + 32u,span,sizeof(span));
	snprintf(manifest_path,sizeof(manifest_path),"%s.experts",path);
	write_file(manifest_path,manifest,16u + (ranges != 0u ? 48u : 0u));
}

static SparkWeightdLazyAttachRequest request_for(const char *path,const char *model)
{
	SparkWeightdLazyAttachRequest request;
	memset(&request,0,sizeof(request));
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = PACK_BYTES;
	snprintf(request.identity.model,sizeof(request.identity.model),"%s",model);
	snprintf(request.identity.revision,sizeof(request.identity.revision),"r1");
	assert(SparkSha256File(path,request.identity.pack_sha256) == SPARK_STATUS_OK);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	return(request);
}

static SparkStatus bind_pack(const char *socket_path,const SparkWeightdLazyAttachRequest *request,SparkWeightdLazyPack **pack,char **log)
{
	SparkWeightdLazyPackOptions options;
	TestStderrCapture capture;
	SparkStatus status;
	memset(&options,0,sizeof(options));
	options.timeout_nanoseconds = TIMEOUT;
	TestStderrCaptureBegin(&capture);
	status = SparkWeightdLazyPackCreate(socket_path,request,&options,pack);
	*log = TestStderrCaptureEnd(&capture);
	return(status);
}

static void check_spine(const SparkWeightdLazyPack *pack,const uint8_t *expected)
{
	const void *slice = 0;
	assert(SparkWeightdLazyPackSlice(pack,0u,PACK_BYTES,&slice) == SPARK_STATUS_OK && slice != 0);
	assert(pack->spine_allocation != 0 && pack->map == 0);
	assert(memcmp(slice,expected,PACK_BYTES) == 0);
}

static void check_bind_and_restart(const char *root,const char *socket_path)
{
	char path[512];
	uint8_t *bytes = malloc(PACK_BYTES),*other = malloc(PACK_BYTES);
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPack *pack = 0;
	char *log;
	assert(bytes != 0 && other != 0);
	fill(bytes,PACK_BYTES,7u);
	snprintf(path,sizeof(path),"%s/pack",root);
	write_pack(path,bytes,0u);
	request = request_for(path,"held-test");
	assert(bind_pack(socket_path,&request,&pack,&log) == SPARK_STATUS_OK);
	assert(strstr(log,"WD-SPINE-HELD spans=1 bytes=3146240 first_hold=1 held=1 ") != 0);
	assert(strstr(log,"WD-SPINE-SHARED") == 0 && strstr(log,"WD-SPINE-PRIVATE") == 0);
	free(log);
	check_spine(pack,bytes);
	assert(SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_OK);
	fill(other,PACK_BYTES,11u);
	write_file(path,other,PACK_BYTES);
	assert(bind_pack(socket_path,&request,&pack,&log) == SPARK_STATUS_OK);
	assert(strstr(log,"WD-SPINE-HELD spans=1 bytes=3146240 first_hold=0 held=1 ") != 0);
	free(log);
	check_spine(pack,bytes);
	assert(SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_OK);
	free(bytes);
	free(other);
}

static void check_sealed(const char *root,const char *socket_path)
{
	char path[512];
	uint8_t *bytes = malloc(PACK_BYTES);
	SparkWeightdLazyAttachRequest request;
	SparkWeightdHoldResult held;
	SparkWeightdClient *client;
	const int wanted = F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL;
	struct stat info;
	void *mapping;
	assert(bytes != 0);
	fill(bytes,PACK_BYTES,23u);
	snprintf(path,sizeof(path),"%s/sealed",root);
	write_pack(path,bytes,0u);
	request = request_for(path,"held-sealed");
	assert(SparkWeightdClientConnect(socket_path,&client,0) == SPARK_STATUS_OK);
	assert(SparkWeightdClientHold(client,&request.identity,path,&held,TIMEOUT) == SPARK_STATUS_OK);
	assert(held.first_hold == 1u && held.held_count == 2u && held.held_bytes == PACK_BYTES && held.fd >= 0);
	assert((fcntl(held.fd,F_GET_SEALS) & wanted) == wanted);
	assert(fstat(held.fd,&info) == 0 && (uint64_t)info.st_size == PACK_BYTES);
	assert(mmap(0,PACK_BYTES,PROT_READ | PROT_WRITE,MAP_SHARED,held.fd,0) == MAP_FAILED);
	assert(ftruncate(held.fd,0) != 0);
	mapping = mmap(0,PACK_BYTES,PROT_READ,MAP_SHARED,held.fd,0);
	assert(mapping != MAP_FAILED && memcmp(mapping,bytes,PACK_BYTES) == 0 && munmap(mapping,PACK_BYTES) == 0);
	assert(close(held.fd) == 0);
	assert(SparkWeightdClientHold(client,&request.identity,path,&held,TIMEOUT) == SPARK_STATUS_OK);
	assert(held.first_hold == 0u && held.held_count == 2u && close(held.fd) == 0);
	request.identity.pack_sha256[0] = request.identity.pack_sha256[0] == '0' ? '1' : '0';
	assert(SparkWeightdClientHold(client,&request.identity,path,&held,TIMEOUT) == SPARK_STATUS_HASH_MISMATCH);
	assert(held.fd < 0);
	assert(SparkWeightdClientHold(client,&request.identity,path,&held,TIMEOUT) == SPARK_STATUS_HASH_MISMATCH);
	SparkWeightdClientClose(client);
	free(bytes);
}

static void check_expert_ranges_refused(const char *root,const char *socket_path)
{
	char path[512];
	uint8_t *bytes = malloc(PACK_BYTES);
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPack *pack = 0;
	char *log;
	assert(bytes != 0);
	fill(bytes,PACK_BYTES,31u);
	snprintf(path,sizeof(path),"%s/experts",root);
	write_pack(path,bytes,1u);
	request = request_for(path,"held-experts");
	assert(bind_pack(socket_path,&request,&pack,&log) == SPARK_STATUS_UNSUPPORTED && pack == 0);
	assert(strstr(log,"names 1 expert ranges; on a device without unified memory a pack binds whole, as its spine") != 0);
	free(log);
	free(bytes);
}

static void *run_old_server(void *data)
{
	OldServer *old = data;
	SparkWeightdIpcHello hello;
	SparkWeightdIpcHelloAck ack;
	SparkWeightdIpcHeader header;
	const int fd = accept(old->listen_fd,0,0);
	assert(fd >= 0);
	assert(recv(fd,&hello,sizeof(hello),MSG_WAITALL) == (ssize_t)sizeof(hello));
	memset(&ack,0,sizeof(ack));
	ack.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
	ack.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	ack.header.kind = SPARK_WEIGHTD_IPC_KIND_HELLO_ACK;
	ack.header.body_bytes = (uint32_t)(sizeof(ack) - sizeof(ack.header));
	ack.header.request_id = hello.header.request_id;
	assert(send(fd,&ack,sizeof(ack),0) == (ssize_t)sizeof(ack));
	assert(recv(fd,&header,sizeof(header),MSG_WAITALL) == (ssize_t)sizeof(header));
	old->closed_on_kind = header.kind;
	assert(close(fd) == 0);
	return(0);
}

static void check_weightd_without_hold(const char *root)
{
	char path[512];
	uint8_t *bytes = malloc(PACK_BYTES);
	struct sockaddr_un address;
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPack *pack = 0;
	OldServer old = {-1,0u};
	pthread_t thread;
	char *log;
	assert(bytes != 0);
	fill(bytes,PACK_BYTES,43u);
	snprintf(path,sizeof(path),"%s/old",root);
	write_pack(path,bytes,0u);
	memset(&address,0,sizeof(address));
	address.sun_family = AF_UNIX;
	snprintf(address.sun_path,sizeof(address.sun_path),"%s/old-socket",root);
	old.listen_fd = socket(AF_UNIX,SOCK_STREAM,0);
	assert(old.listen_fd >= 0 && bind(old.listen_fd,(const struct sockaddr *)&address,sizeof(address)) == 0 && listen(old.listen_fd,1) == 0);
	assert(pthread_create(&thread,0,run_old_server,&old) == 0);
	request = request_for(path,"held-old");
	assert(bind_pack(address.sun_path,&request,&pack,&log) != SPARK_STATUS_OK && pack == 0);
	assert(pthread_join(thread,0) == 0 && old.closed_on_kind == SPARK_WEIGHTD_IPC_KIND_HOLD);
	assert(strstr(log,"weightd hold: no reply to kind 61 (HOLD)") != 0);
	assert(strstr(log,"WD-SPINE-") == 0);
	free(log);
	assert(close(old.listen_fd) == 0);
	free(bytes);
}

int main(void)
{
	char root[] = "/tmp/weightd-held-XXXXXX",socket_path[256];
	TestServer state = {0};
	SparkWeightdServerConfig config = {0};
	pthread_t thread;
	spark_stub_cuda_discrete_ordering(1u);
	assert(mkdtemp(root) != 0);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	config.socket_path = socket_path;
	config.device_bytes_max = UINT64_C(1) << 30;
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	check_bind_and_restart(root,socket_path);
	check_sealed(root,socket_path);
	check_expert_ranges_refused(root,socket_path);
	check_weightd_without_hold(root);
	state.stop = 1;
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	printf("PASS weightd held packs: HOLD verifies a pack once into a sealed memfd held until weightd exits; a device without unified memory copies the spine into its own memory, a rebind reuses the held copy without reading the pack, a wrong digest holds nothing, expert ranges are refused, and a weightd without HOLD fails loudly naming kind 61\n");
	return(0);
}
