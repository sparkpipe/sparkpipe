#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_draft_bridge.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_tap_stream.h"
#include "sparkpipe/spark_weightd.h"

#define PACK_BYTES ((3u * 1024u * 1024u) + 512u)
#define HUGE_CONTEXT UINT64_C(1099511627776)
#define TAP_HIDDEN 4u
#define TAP_ENTRIES 2u
#define WAIT_MS 1000u
#define OUTPUT_BYTES 65536u

static char draftd_output[OUTPUT_BYTES];
static size_t draftd_output_used;

typedef struct Ask
{
	SparkDraftBridge *bridge;
	uint64_t started;
	uint32_t done;
	uint32_t server_status;
	uint32_t count;
	double seconds;
} Ask;

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

static void write_file(const char *path,const void *data,size_t bytes)
{
	FILE *file = fopen(path,"wb");
	assert(file != 0 && fwrite(data,1u,bytes,file) == bytes && fclose(file) == 0);
}

static void write_pack(const char *path)
{
	char side[600],digest[SPARK_SHA256_HEX_BYTES],text[700];
	const uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,0u,0u};
	uint8_t *bytes = malloc(PACK_BYTES);
	uint32_t value = 97u;
	assert(bytes != 0);
	for ( uint32_t index = 0u; index < PACK_BYTES; index++ )
	{
		value = value * 1664525u + 1013904223u;
		bytes[index] = (uint8_t)(value >> 24);
	}
	write_file(path,bytes,PACK_BYTES);
	snprintf(side,sizeof(side),"%s.experts",path);
	write_file(side,header,sizeof(header));
	assert(SparkSha256File(path,digest) == SPARK_STATUS_OK);
	snprintf(side,sizeof(side),"%s.sha256",path);
	snprintf(text,sizeof(text),"%s  pack\n",digest);
	write_file(side,text,strlen(text));
	free(bytes);
}

static uint32_t free_port(void)
{
	struct sockaddr_in address;
	socklen_t length = sizeof(address);
	const int fd = socket(AF_INET,SOCK_STREAM,0);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	assert(fd >= 0 && bind(fd,(struct sockaddr *)&address,sizeof(address)) == 0 && getsockname(fd,(struct sockaddr *)&address,&length) == 0);
	assert(close(fd) == 0);
	return(ntohs(address.sin_port));
}

static void lane_json(char *text,size_t bytes,const char *name,uint32_t port,uint32_t tap_port,const char *module,const char *pack,uint64_t context,
	int last)
{
	char taps[256] = "";
	if ( tap_port != 0u )
		snprintf(taps,sizeof(taps),"\"taps\":{\"host\":\"127.0.0.1\",\"port\":%u,\"model_tag\":\"probe-taps\",\"hidden\":%u,\"entries\":\"o3,o7\"},",
			tap_port,TAP_HIDDEN);
	snprintf(text,bytes,"{\"name\":\"%s\",\"host\":\"127.0.0.1\",\"draft_port\":%u,\"target_model\":\"%s-target\",\"io_timeout_ms\":3000,%s"
		"\"drafter\":{\"kind\":\"engine\",\"module\":\"%s\",\"pack\":\"%s\",\"model\":\"probe-pack\",\"revision\":\"r1\",\"context_bytes\":%llu,"
		"\"wait_ms\":%u}}%s",name,port,name,taps,module,pack,(unsigned long long)context,WAIT_MS,last != 0 ? "" : ",");
}

static int tap_connect(uint32_t port,SparkTapStreamHello *hello)
{
	struct sockaddr_in address;
	const uint64_t deadline = SparkMonotonicNs() + UINT64_C(5000000000);
	int fd = -1;
	memset(hello,0,sizeof(*hello));
	hello->magic = SPARK_TAP_STREAM_MAGIC;
	hello->version = SPARK_TAP_STREAM_VERSION;
	hello->kind = SPARK_TAP_STREAM_KIND_HELLO;
	hello->header_bytes = sizeof(*hello);
	snprintf(hello->model_tag,sizeof(hello->model_tag),"%s","probe-taps");
	hello->hidden = TAP_HIDDEN;
	hello->total_layers = 12u;
	hello->dtype = SPARK_TAP_STREAM_DTYPE_BF16;
	hello->entry_count = TAP_ENTRIES;
	hello->tap_rank = 0u;
	hello->tp_degree = 1u;
	hello->entries[0] = (SparkTapStreamEntry){1u,3u};
	hello->entries[1] = (SparkTapStreamEntry){1u,7u};
	hello->fingerprint = SparkTapStreamFingerprint(hello);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((uint16_t)port);
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	while ( SparkMonotonicNs() < deadline )
	{
		fd = socket(AF_INET,SOCK_STREAM,0);
		assert(fd >= 0);
		if ( connect(fd,(const struct sockaddr *)&address,(socklen_t)sizeof(address)) == 0 )
			break;
		(void)close(fd);
		fd = -1;
		(void)poll(0,0u,20);
	}
	assert(fd >= 0 && send(fd,hello,sizeof(*hello),0) == (ssize_t)sizeof(*hello));
	return(fd);
}

static void tap_send(int fd,const SparkTapStreamHello *hello,uint64_t sequence,uint64_t first,const uint32_t *tokens,uint32_t rows)
{
	uint8_t bytes[1024];
	SparkTapStreamFrame frame;
	size_t used = 0u;
	const uint16_t zero = 0u;
	memset(&frame,0,sizeof(frame));
	frame.magic = SPARK_TAP_STREAM_MAGIC;
	frame.version = SPARK_TAP_STREAM_VERSION;
	frame.kind = SPARK_TAP_STREAM_KIND_FRAME;
	frame.header_bytes = sizeof(frame);
	frame.fingerprint = hello->fingerprint;
	frame.control_generation = 1u;
	frame.request_id = sequence;
	frame.request_generation = 1u;
	frame.sequence_id = sequence;
	frame.first_position = first;
	frame.row_count = rows;
	frame.entry_count = TAP_ENTRIES;
	frame.payload_bytes = (uint64_t)rows * (sizeof(uint32_t) + TAP_ENTRIES * TAP_HIDDEN * sizeof(uint16_t));
	memcpy(bytes,&frame,sizeof(frame));
	used = sizeof(frame);
	memcpy(bytes + used,tokens,(size_t)rows * sizeof(uint32_t));
	used += (size_t)rows * sizeof(uint32_t);
	for ( uint32_t element = 0u; element < rows * TAP_ENTRIES * TAP_HIDDEN; ++element )
	{
		memcpy(bytes + used,&zero,sizeof(zero));
		used += sizeof(zero);
	}
	assert(used <= sizeof(bytes) && send(fd,bytes,used,0) == (ssize_t)used);
}

static pid_t start_draftd(const char *config,int *output)
{
	int pipes[2];
	pid_t child;
	assert(pipe(pipes) == 0);
	child = fork();
	assert(child >= 0);
	if ( child == 0 )
	{
		(void)dup2(pipes[1],1);
		(void)close(pipes[0]);
		(void)close(pipes[1]);
		execl(SPARK_TEST_DRAFTD,SPARK_TEST_DRAFTD,"--config",config,(char *)0);
		_exit(127);
	}
	(void)close(pipes[1]);
	*output = pipes[0];
	return(child);
}

static void drain(int fd,int milliseconds)
{
	struct pollfd poller = {fd,POLLIN,0};
	while ( draftd_output_used + 1u < sizeof(draftd_output) && poll(&poller,1u,milliseconds) > 0 )
	{
		const ssize_t got = read(fd,draftd_output + draftd_output_used,sizeof(draftd_output) - 1u - draftd_output_used);
		if ( got <= 0 )
			break;
		draftd_output_used += (size_t)got;
		draftd_output[draftd_output_used] = '\0';
		milliseconds = 0;
	}
}

static void wait_for(int fd,const char *marker)
{
	const uint64_t deadline = SparkMonotonicNs() + UINT64_C(30000000000);
	while ( SparkMonotonicNs() < deadline && strstr(draftd_output,marker) == 0 )
		drain(fd,500);
	if ( strstr(draftd_output,marker) != 0 )
		return;
	fprintf(stderr,"draftd output without %s:\n%s\n",marker,draftd_output);
	assert(!"draftd marker");
}

static void expect(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"%s; draftd output so far:\n%s\n",what,draftd_output);
	assert(!"draftd engine expectation");
}

static uint32_t count_lines(const char *text,const char *pattern)
{
	uint32_t found = 0u;
	for ( const char *cursor = strstr(text,pattern); cursor != 0; cursor = strstr(cursor + 1,pattern) )
		found++;
	return(found);
}

static int control(const char *socket_path,const char *command,const char *lane,char *reply,size_t bytes)
{
	char line[1024];
	FILE *pipe_in;
	size_t used = 0u;
	int status;
	snprintf(line,sizeof(line),"%s --control %s %s %s 2>&1",SPARK_TEST_DRAFTD,socket_path,command,lane != 0 ? lane : "");
	pipe_in = popen(line,"r");
	assert(pipe_in != 0);
	used = fread(reply,1u,bytes - 1u,pipe_in);
	reply[used] = '\0';
	status = pclose(pipe_in);
	return(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
}

static Ask ask_start(uint32_t port,const char *target,uint64_t sequence,const uint32_t *committed,uint32_t committed_count)
{
	SparkDraftBridgeConfig config;
	SparkRowSampling sampling;
	Ask ask;
	memset(&config,0,sizeof(config));
	memset(&sampling,0,sizeof(sampling));
	memset(&ask,0,sizeof(ask));
	config.abi_version = SPARK_DRAFT_BRIDGE_ABI_VERSION;
	config.descriptor_bytes = SPARK_DRAFT_BRIDGE_CONFIG_BYTES;
	config.host = "127.0.0.1";
	config.port = port;
	snprintf(config.target_model,sizeof(config.target_model),"%s",target);
	config.max_committed_tokens = 1024u;
	config.max_nodes = 16u;
	config.connect_timeout_ms = 3000u;
	config.io_timeout_ms = 3000u;
	assert(SparkDraftBridgeInitialize(&config,&ask.bridge) == SPARK_STATUS_OK);
	ask.started = SparkMonotonicNs();
	assert(SparkDraftBridgeRequestTree(ask.bridge,6u,sequence,3u,committed_count - 1u,committed[committed_count - 1u],committed,committed_count,0,0u,4u,
		1000u,4u,&sampling,9u) == SPARK_STATUS_OK);
	return(ask);
}

static void ask_pump(Ask *ask,int milliseconds)
{
	SparkDraftBridgeProposalInfo info;
	SparkDraftBridgeNode nodes[16];
	struct pollfd poller = {-1,0,0};
	uint32_t events = 0u;
	uint64_t deadline = 0u;
	SparkStatus reply;
	if ( ask->done != 0u )
	{
		(void)poll(0,0u,milliseconds);
		return;
	}
	memset(&info,0,sizeof(info));
	reply = SparkDraftBridgeReply(ask->bridge,nodes,&ask->count,&info);
	if ( reply != SPARK_STATUS_PENDING )
	{
		assert(reply == SPARK_STATUS_OK || (reply == SPARK_STATUS_VALIDATION_FAILED && info.server_status != SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK));
		ask->done = 1u;
		ask->server_status = info.server_status;
		ask->seconds = (double)(SparkMonotonicNs() - ask->started) / 1e9;
		return;
	}
	if ( SparkDraftBridgePoll(ask->bridge,&poller.fd,&events,&deadline) != 0u )
	{
		poller.events = (short)((events & SPARK_DRAFT_BRIDGE_POLL_READ) != 0u ? POLLIN : POLLOUT);
		(void)poll(&poller,1u,milliseconds);
	}
}

static uint32_t ask_finish(Ask *ask,uint32_t *count,double *seconds)
{
	while ( ask->done == 0u && SparkMonotonicNs() - ask->started < UINT64_C(6000000000) )
		ask_pump(ask,20);
	assert(ask->done != 0u);
	*count = ask->count;
	*seconds = ask->seconds;
	SparkDraftBridgeDestroy(ask->bridge);
	ask->bridge = 0;
	return(ask->server_status);
}

static uint32_t ask(uint32_t port,const char *target,uint32_t *count,double *seconds)
{
	const uint32_t committed[4] = {11u,12u,13u,14u};
	Ask pending = ask_start(port,target,77u,committed,4u);
	return(ask_finish(&pending,count,seconds));
}

static void idle(Ask *ask,int milliseconds)
{
	const uint64_t until = SparkMonotonicNs() + (uint64_t)milliseconds * UINT64_C(1000000);
	while ( SparkMonotonicNs() < until )
		ask_pump(ask,10);
}

int main(void)
{
	char root[] = "/tmp/draftd-engine-XXXXXX",weightd_socket[256],control_socket[256],pack[256],config[256],reply[8192];
	char lanes[3][1600],text[6000];
	TestServer state = {0};
	SparkWeightdServerConfig server_config = {0};
	SparkTapStreamHello hello;
	const uint32_t probe_port = free_port(),huge_port = free_port(),abi_port = free_port(),tap_port = free_port();
	const uint32_t first_two[2] = {11u,12u},third[1] = {13u},mismatch[4] = {11u,99u,13u,14u},other[3] = {21u,22u,23u},absent[4] = {31u,32u,33u,34u};
	uint32_t count = 0u;
	size_t mark;
	double seconds = 0.0;
	Ask held,replacement;
	int taps = -1;
	pthread_t thread;
	pid_t draftd;
	int output = -1,status = 0;
	assert(setenv("SPARK_STUB_CUDA_DISCRETE","1",1) == 0);
	assert(mkdtemp(root) != 0);
	snprintf(weightd_socket,sizeof(weightd_socket),"%s/weightd.sock",root);
	snprintf(control_socket,sizeof(control_socket),"%s/control.sock",root);
	snprintf(pack,sizeof(pack),"%s/probe.rank0.sp",root);
	snprintf(config,sizeof(config),"%s/draftd.json",root);
	write_pack(pack);
	server_config.socket_path = weightd_socket;
	server_config.device_bytes_max = UINT64_C(1) << 30;
	assert(SparkWeightdServerCreate(&server_config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	lane_json(lanes[0],sizeof(lanes[0]),"probe",probe_port,tap_port,SPARK_TEST_PROBE_MODULE,pack,UINT64_C(1) << 20,0);
	lane_json(lanes[1],sizeof(lanes[1]),"huge",huge_port,0u,SPARK_TEST_PROBE_MODULE,pack,HUGE_CONTEXT,0);
	lane_json(lanes[2],sizeof(lanes[2]),"abi",abi_port,0u,SPARK_TEST_PROBE_ABI_MODULE,pack,UINT64_C(1) << 20,1);
	snprintf(text,sizeof(text),"{\"weightd_socket\":\"%s\",\"control_socket\":\"%s\",\"lanes\":[%s%s%s]}",weightd_socket,control_socket,lanes[0],lanes[1],
		lanes[2]);
	write_file(config,text,strlen(text));
	draftd = start_draftd(config,&output);
	wait_for(output,"DRAFTD-READY lanes=3 engines=3");
	assert(ask(probe_port,"probe-target",&count,&seconds) == SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE && count == 0u && seconds < 1.0);
	assert(control(control_socket,"status",0,reply,sizeof(reply)) == 0 && strstr(reply,"OK lanes=3") != 0 &&
		strstr(reply,"lane=probe drafter=engine active=0") != 0 && strstr(reply,"unavailable=1") != 0);
	assert(control(control_socket,"activate","probe",reply,sizeof(reply)) == 0 && strstr(reply,"OK lane=probe device_bytes=") != 0 &&
		strstr(reply,"first_hold=1") != 0);
	taps = tap_connect(tap_port,&hello);
	mark = draftd_output_used;
	held = ask_start(probe_port,"probe-target",77u,(const uint32_t[]){11u,12u,13u,14u},4u);
	idle(&held,150);
	tap_send(taps,&hello,77u,0u,first_two,2u);
	idle(&held,150);
	drain(output,50);
	expect(count_lines(draftd_output + mark,"answer=await sequence=77 position=3 forwards=0") >= 2u &&
		count_lines(draftd_output + mark,"answer=drafts") == 0u,"a request waits and its proposals do no work before the covering frame");
	tap_send(taps,&hello,77u,2u,third,1u);
	assert(ask_finish(&held,&count,&seconds) == SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK && count == 2u && seconds < 0.9);
	wait_for(output,"answer=drafts sequence=77 position=3 forwards=1");
	assert(count_lines(draftd_output + mark,"answer=drafts") == 1u);
	mark = draftd_output_used;
	assert(ask(probe_port,"probe-target",&count,&seconds) == SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK && count == 2u && seconds < 0.5);
	held = ask_start(probe_port,"probe-target",78u,absent,4u);
	assert(ask_finish(&held,&count,&seconds) == SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE && count == 0u &&
		seconds >= (double)WAIT_MS / 1000.0 && seconds < (double)WAIT_MS / 1000.0 + 1.5);
	drain(output,50);
	assert(count_lines(draftd_output + mark,"answer=await sequence=78 position=3 forwards=2") == 1u);
	held = ask_start(probe_port,"probe-target",77u,mismatch,4u);
	assert(ask_finish(&held,&count,&seconds) == SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE && count == 0u && seconds < 0.5);
	wait_for(output,"answer=no-context sequence=77 position=3 forwards=2");
	held = ask_start(probe_port,"probe-target",79u,other,3u);
	idle(&held,150);
	replacement = ask_start(probe_port,"probe-target",79u,other,3u);
	assert(ask_finish(&held,&count,&seconds) == SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE && count == 0u && seconds < 0.8);
	idle(&replacement,150);
	tap_send(taps,&hello,79u,0u,other,2u);
	assert(ask_finish(&replacement,&count,&seconds) == SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK && count == 2u && seconds < 0.9);
	assert(close(taps) == 0);
	assert(control(control_socket,"status",0,reply,sizeof(reply)) == 0 && strstr(reply,"lane=probe drafter=engine active=1") != 0);
	assert(control(control_socket,"activate","probe",reply,sizeof(reply)) == 0 && strstr(reply,"already active") != 0);
	assert(control(control_socket,"deactivate","probe",reply,sizeof(reply)) == 0 && strstr(reply,"OK lane=probe released") != 0);
	assert(ask(probe_port,"probe-target",&count,&seconds) == SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE && count == 0u && seconds < 1.0);
	assert(control(control_socket,"activate","probe",reply,sizeof(reply)) == 0 && strstr(reply,"first_hold=0") != 0);
	assert(control(control_socket,"deactivate","probe",reply,sizeof(reply)) == 0);
	assert(control(control_socket,"activate","huge",reply,sizeof(reply)) == 1 && strstr(reply,"REFUSED lane=huge lane huge needs ") != 0 &&
		strstr(reply,"(pack 3146240 + context cap 1099511627776)") != 0 && strstr(reply," free (") != 0);
	assert(control(control_socket,"activate","abi",reply,sizeof(reply)) == 1 && strstr(reply,"speaks drafter ABI 99; sparkpipe_draftd speaks 1") != 0);
	assert(control(control_socket,"activate","nowhere",reply,sizeof(reply)) == 1 && strstr(reply,"REFUSED lane=nowhere is not an engine lane") != 0);
	assert(control(control_socket,"status",0,reply,sizeof(reply)) == 0 && strstr(reply,"lane=huge drafter=engine active=0") != 0 &&
		strstr(reply,"lane=abi drafter=engine active=0") != 0);
	assert(kill(draftd,SIGTERM) == 0 && waitpid(draftd,&status,0) == draftd && WIFEXITED(status) && WEXITSTATUS(status) == 0);
	assert(close(output) == 0);
	state.stop = 1;
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	puts("PASS draftd engine lanes: an inactive lane answers unavailable at once, activate binds the pack through weightd's HOLD into the "
		"lane's own device memory, a request waits for the taps that cover its anchor and drafts after the covering frame while its waiting "
		"proposals do no device work, a request without taps is answered unavailable at wait_ms, a token mismatch is answered unavailable at "
		"once, a new connection replaces a waiting request, deactivate releases the device memory, a second activate reuses the held copy, a "
		"lane whose pack and context cap exceed free device memory is refused with the numbers, a module of another drafter ABI is refused, and "
		"status reports every lane");
	return(0);
}
