#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"

#define ATTACH_ARENA_BYTES (8ull << 20)
#define ATTACH_TIMEOUT_NS 120000000000ull
#define ATTACH_MODEL "nvrm-attach"
#define ATTACH_REVISION "gpu"
#define ATTACH_SAMPLES 64u
#define ATTACH_SAMPLE_BYTES (1ull << 20)

typedef struct AttachServer
{
	SparkWeightdServer *server;
	volatile sig_atomic_t stop;
} AttachServer;

static uint32_t attach_failures;

static void Check(int condition, const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	attach_failures++;
}

static void *ServerMain(void *raw)
{
	AttachServer *context = (AttachServer *)raw;
	(void)SparkWeightdServerRun(context->server,&context->stop);
	return(0);
}

static void Slice(SparkWeightdPackSlice *slice)
{
	memset(slice,0,sizeof(*slice));
	slice->model = ATTACH_MODEL;
	slice->revision = ATTACH_REVISION;
	slice->topology = 1u;
	slice->geometry_fingerprint = 1ull;
	slice->pack_bytes = ATTACH_ARENA_BYTES;
}

static int ReadBack(uint64_t address, uint8_t *host, uint64_t bytes)
{
	SparkDeviceBuffer view,back;
	SparkDeviceRange source,target;
	uint32_t adopted = 0u;
	int copied;
	if ( SparkDeviceResolve((SparkDeviceAddress)address,bytes,&source) != SPARK_STATUS_OK )
	{
		if ( SparkDeviceAdopt(&view,(SparkDeviceAddress)address,0,bytes,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"attach readback") != SPARK_STATUS_OK )
			return(0);
		adopted = 1u;
		source.buffer = &view;
		source.offset = 0u;
	}
	if ( SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PINNED,bytes,"attach readback host") != SPARK_STATUS_OK )
	{
		if ( adopted != 0u )
			(void)SparkDeviceFree(&view);
		return(0);
	}
	target.buffer = &back;
	target.offset = 0u;
	copied = SparkDeviceCopySync(target,source,bytes) == SPARK_STATUS_OK;
	if ( copied )
		memcpy(host,back.host,(size_t)bytes);
	(void)SparkDeviceFree(&back);
	if ( adopted != 0u )
		(void)SparkDeviceFree(&view);
	return(copied);
}

static int Consumer(const char *pack_path)
{
	SparkWeightdPackSlice slice;
	SparkWeightdAttachOutcome outcome;
	char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
	uint8_t *expected = (uint8_t *)malloc(ATTACH_ARENA_BYTES),*seen = (uint8_t *)malloc(ATTACH_ARENA_BYTES);
	FILE *file = fopen(pack_path,"rb");
	(void)signal(SIGPIPE,SIG_IGN);
	if ( expected == 0 || seen == 0 || file == 0 || fread(expected,1u,ATTACH_ARENA_BYTES,file) != ATTACH_ARENA_BYTES )
	{
		printf("FAIL the consumer cannot read the pack\n");
		return(1);
	}
	(void)fclose(file);
	Slice(&slice);
	memset(&outcome,0,sizeof(outcome));
	if ( SparkWeightdAttachPack(&slice,pack_path,ATTACH_TIMEOUT_NS,&outcome,reason) != SPARK_STATUS_OK || outcome.client == 0 )
	{
		printf("FAIL the consumer process did not attach: %s\n",reason);
		return(1);
	}
	Check(outcome.loaded_from_pack == 0u,"a second process attaches to the arena already loaded");
	if ( SparkWeightdAttachImportMap(&outcome,ATTACH_ARENA_BYTES,ATTACH_TIMEOUT_NS,reason) != SPARK_STATUS_OK || outcome.map_base == 0 )
	{
		printf("FAIL the consumer process did not import and map the arena: %s\n",reason);
		SparkWeightdAttachRelease(&outcome);
		return(1);
	}
	Check(ReadBack((uint64_t)(uintptr_t)outcome.map_base,seen,ATTACH_ARENA_BYTES) && memcmp(seen,expected,ATTACH_ARENA_BYTES) == 0,
		"the consumer process reads the pack's bytes through its own mapping");
	SparkWeightdAttachRelease(&outcome);
	free(expected);
	free(seen);
	return(attach_failures != 0u ? 1 : 0);
}

static int SampleMatches(int fd, uint64_t base, uint64_t offset, uint64_t bytes, uint8_t *expected, uint8_t *seen)
{
	uint64_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t got = pread(fd,expected + done,(size_t)(bytes - done),(off_t)(offset + done));
		if ( got <= 0 )
			return(0);
		done += (uint64_t)got;
	}
	return(ReadBack(base + offset,seen,bytes) && memcmp(seen,expected,(size_t)bytes) == 0);
}

static int External(const char *pack_path, const char *model, const char *revision)
{
	SparkWeightdPackSlice slice;
	SparkWeightdAttachOutcome outcome;
	char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
	uint8_t *expected = (uint8_t *)malloc(ATTACH_SAMPLE_BYTES),*seen = (uint8_t *)malloc(ATTACH_SAMPLE_BYTES);
	const int fd = open(pack_path,O_RDONLY | O_CLOEXEC);
	const off_t size = fd >= 0 ? lseek(fd,0,SEEK_END) : -1;
	uint32_t sample,matched = 0u;
	(void)signal(SIGPIPE,SIG_IGN);
	if ( expected == 0 || seen == 0 || size < (off_t)ATTACH_SAMPLE_BYTES )
	{
		printf("FAIL cannot read %s\n",pack_path);
		return(1);
	}
	memset(&slice,0,sizeof(slice));
	slice.model = model;
	slice.revision = revision;
	slice.pack_bytes = (uint64_t)size;
	memset(&outcome,0,sizeof(outcome));
	if ( SparkWeightdAttachPack(&slice,pack_path,ATTACH_TIMEOUT_NS,&outcome,reason) != SPARK_STATUS_OK || outcome.client == 0 )
	{
		printf("FAIL this process did not attach to %s: %s\n",pack_path,reason);
		return(1);
	}
	if ( SparkWeightdAttachImportMap(&outcome,outcome.arena_bytes,ATTACH_TIMEOUT_NS,reason) != SPARK_STATUS_OK || outcome.map_base == 0 )
	{
		printf("FAIL this process did not import and map the arena: %s\n",reason);
		SparkWeightdAttachRelease(&outcome);
		return(1);
	}
	for (sample = 0u; sample < ATTACH_SAMPLES; sample++)
	{
		const uint64_t span = (uint64_t)size - ATTACH_SAMPLE_BYTES;
		const uint64_t offset = sample + 1u == ATTACH_SAMPLES ? span : span / ATTACH_SAMPLES * sample / 4096u * 4096u;
		matched += (uint32_t)SampleMatches(fd,(uint64_t)(uintptr_t)outcome.map_base,offset,ATTACH_SAMPLE_BYTES,expected,seen);
	}
	printf("%s %u of %u samples of %llu bytes match the pack, arena %llu bytes, generation %llu, %s attach, %u holders, %u chunks of %llu bytes\n",
		matched == ATTACH_SAMPLES ? "PASS" : "FAIL",matched,ATTACH_SAMPLES,(unsigned long long)ATTACH_SAMPLE_BYTES,(unsigned long long)outcome.arena_bytes,
		(unsigned long long)outcome.arena_generation,outcome.loaded_from_pack != 0u ? "cold" : "warm",outcome.refcount,outcome.map_chunk_count,
		(unsigned long long)outcome.map_chunk_bytes);
	SparkWeightdAttachRelease(&outcome);
	(void)close(fd);
	free(expected);
	free(seen);
	return(matched == ATTACH_SAMPLES ? 0 : 1);
}

static int WritePack(const char *pack_path, uint8_t *bytes, char hex[SPARK_SHA256_HEX_BYTES])
{
	SparkSha256Context digest;
	uint8_t raw[SPARK_SHA256_DIGEST_BYTES];
	uint64_t offset;
	FILE *file;
	for (offset = 0u; offset < ATTACH_ARENA_BYTES; offset++)
		bytes[offset] = (uint8_t)(offset * 131ull + (offset >> 9));
	file = fopen(pack_path,"wb");
	if ( file == 0 || fwrite(bytes,1u,ATTACH_ARENA_BYTES,file) != ATTACH_ARENA_BYTES )
		return(0);
	(void)fclose(file);
	SparkSha256Initialize(&digest);
	SparkSha256Update(&digest,bytes,ATTACH_ARENA_BYTES);
	SparkSha256Finalize(&digest,raw);
	SparkSha256DigestToHex(raw,hex);
	return(1);
}

static int Attach(const char *socket_path, const SparkWeightdAttachRequest *request, SparkWeightdClient **client, SparkWeightdAttachResult *result)
{
	memset(result,0,sizeof(*result));
	return(SparkWeightdClientConnect(socket_path,client,0) == SPARK_STATUS_OK &&
		SparkWeightdClientAttach(*client,request,result,ATTACH_TIMEOUT_NS) == SPARK_STATUS_OK && result->status == SPARK_STATUS_OK);
}

static void InProcessImport(const char *pack_path, const uint8_t *expected, uint8_t *seen)
{
	SparkWeightdPackSlice slice;
	SparkWeightdAttachOutcome helper;
	char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
	Slice(&slice);
	memset(&helper,0,sizeof(helper));
	if ( SparkWeightdAttachPack(&slice,pack_path,ATTACH_TIMEOUT_NS,&helper,reason) != SPARK_STATUS_OK || helper.client == 0 )
	{
		printf("FAIL the attach helper did not attach: %s\n",reason);
		attach_failures++;
		return;
	}
	Check(helper.loaded_from_pack == 0u && helper.refcount == 3u,"the attach helper takes the warm arena as its third holder");
	if ( SparkWeightdAttachImportMap(&helper,ATTACH_ARENA_BYTES,ATTACH_TIMEOUT_NS,reason) != SPARK_STATUS_OK || helper.map_base == 0 )
	{
		printf("FAIL the attach helper did not import and map the arena: %s\n",reason);
		attach_failures++;
	}
	else
		Check(ReadBack((uint64_t)(uintptr_t)helper.map_base,seen,ATTACH_ARENA_BYTES) && memcmp(seen,expected,ATTACH_ARENA_BYTES) == 0,
			"an import mapped at the consumer's own address reads the pack's bytes");
	SparkWeightdAttachRelease(&helper);
}

static void OtherProcess(const char *pack_path)
{
	int status = 0;
	const pid_t child = fork();
	if ( child == 0 )
	{
		execl("/proc/self/exe","test_nvrm_weightd_attach","--consumer",pack_path,(char *)0);
		_exit(127);
	}
	Check(child > 0 && waitpid(child,&status,0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
		"a second process attaches, imports, maps and reads the arena, and exits cleanly");
}

int main(int argc, char **argv)
{
	AttachServer context;
	SparkWeightdServerConfig config;
	SparkWeightdAttachRequest request;
	SparkWeightdAttachResult result;
	SparkWeightdClient *first = 0,*second = 0,*fresh = 0;
	pthread_t thread;
	char socket_path[128],pack_path[128],hex[SPARK_SHA256_HEX_BYTES];
	uint8_t *bytes = (uint8_t *)malloc(ATTACH_ARENA_BYTES),*seen = (uint8_t *)malloc(ATTACH_ARENA_BYTES);
	uint64_t generation;
	(void)signal(SIGPIPE,SIG_IGN);
	if ( argc == 3 && strcmp(argv[1],"--consumer") == 0 )
		return(Consumer(argv[2]));
	if ( argc == 5 && strcmp(argv[1],"--pack") == 0 )
		return(External(argv[2],argv[3],argv[4]));
	snprintf(socket_path,sizeof(socket_path),"/tmp/spark_nvrm_weightd_attach_%ld.sock",(long)getpid());
	snprintf(pack_path,sizeof(pack_path),"/tmp/spark_nvrm_weightd_attach_%ld.spack",(long)getpid());
	if ( bytes == 0 || seen == 0 || !WritePack(pack_path,bytes,hex) )
	{
		printf("FAIL cannot write the pack\n");
		return(1);
	}
	memset(&request,0,sizeof(request));
	memcpy(request.identity.model,ATTACH_MODEL,sizeof(ATTACH_MODEL));
	memcpy(request.identity.revision,ATTACH_REVISION,sizeof(ATTACH_REVISION));
	memcpy(request.identity.pack_sha256,hex,SPARK_SHA256_HEX_BYTES);
	request.identity.topology = 1u;
	request.identity.geometry_fingerprint = 1ull;
	request.identity.arena_bytes = ATTACH_ARENA_BYTES;
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	memcpy(request.pack_path,pack_path,strlen(pack_path) + 1u);
	memset(&config,0,sizeof(config));
	config.socket_path = socket_path;
	config.device_bytes_max = 4ull * ATTACH_ARENA_BYTES;
	(void)unlink(socket_path);
	if ( SparkWeightdIdentityPrepare(&request.identity) != SPARK_STATUS_OK || SparkWeightdServerCreate(&config,&context.server) != SPARK_STATUS_OK )
	{
		printf("FAIL cannot start weightd\n");
		return(1);
	}
	context.stop = 0;
	if ( pthread_create(&thread,0,ServerMain,&context) != 0 )
		return(1);
	if ( Attach(socket_path,&request,&first,&result) && result.loaded_from_pack == 1u && result.device_handle != 0ull )
	{
		generation = result.arena_generation;
		Check(ReadBack(result.device_handle,seen,ATTACH_ARENA_BYTES) && memcmp(seen,bytes,ATTACH_ARENA_BYTES) == 0,
			"weightd loads the pack into its arena byte for byte");
		Check(Attach(socket_path,&request,&second,&result) && result.loaded_from_pack == 0u && result.refcount == 2u,
			"a second attach shares the loaded arena");
		Check(setenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET,socket_path,1) == 0 && setenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256,hex,1) == 0,"set the attach environment");
		InProcessImport(pack_path,bytes,seen);
		OtherProcess(pack_path);
		Check(Attach(socket_path,&request,&fresh,&result) && result.loaded_from_pack == 0u && result.arena_generation == generation,
			"the arena stays loaded, with its generation, after the second process exits");
	}
	else
	{
		printf("FAIL the first attach did not load the pack: %s\n",SparkStatusToString(result.status));
		attach_failures++;
	}
	if ( fresh != 0 )
		SparkWeightdClientClose(fresh);
	if ( second != 0 )
		SparkWeightdClientClose(second);
	if ( first != 0 )
		SparkWeightdClientClose(first);
	__atomic_store_n(&context.stop,1,__ATOMIC_SEQ_CST);
	(void)pthread_join(thread,0);
	SparkWeightdServerDestroy(context.server);
	(void)remove(pack_path);
	(void)remove(socket_path);
	free(bytes);
	free(seen);
	if ( attach_failures != 0u )
	{
		printf("FAIL (%u)\n",attach_failures);
		return(1);
	}
	printf("PASS nvrm weightd loads a pack into an arena of shared device memory; a second attach shares it, an import maps it at the "
		"consumer's own address, another process attaches, maps and reads every byte, and the arena outlives that process\n");
	return(0);
}
