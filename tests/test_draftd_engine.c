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
#include "sparkpipe/spark_weightd.h"

#define PACK_BYTES ((3u * 1024u * 1024u) + 512u)
#define HUGE_CONTEXT UINT64_C(1099511627776)

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

static void lane_json(char *text,size_t bytes,const char *name,uint32_t port,const char *module,const char *pack,uint64_t context,int last)
{
	snprintf(text,bytes,"{\"name\":\"%s\",\"host\":\"127.0.0.1\",\"draft_port\":%u,\"target_model\":\"%s-target\",\"io_timeout_ms\":2000,"
		"\"drafter\":{\"kind\":\"engine\",\"module\":\"%s\",\"pack\":\"%s\",\"model\":\"probe-pack\",\"revision\":\"r1\",\"context_bytes\":%llu}}%s",
		name,port,name,module,pack,(unsigned long long)context,last != 0 ? "" : ",");
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

static void wait_for(int fd,const char *marker)
{
	char text[8192];
	size_t used = 0u;
	const uint64_t deadline = SparkMonotonicNs() + UINT64_C(30000000000);
	struct pollfd poller = {fd,POLLIN,0};
	while ( SparkMonotonicNs() < deadline && used + 1u < sizeof(text) )
	{
		ssize_t got;
		if ( poll(&poller,1u,500) <= 0 )
			continue;
		got = read(fd,text + used,sizeof(text) - 1u - used);
		if ( got <= 0 )
			break;
		used += (size_t)got;
		text[used] = '\0';
		if ( strstr(text,marker) != 0 )
			return;
	}
	text[used] = '\0';
	fprintf(stderr,"draftd output without %s:\n%s\n",marker,text);
	assert(!"draftd marker");
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

static uint32_t ask(uint32_t port,const char *target,uint32_t *count,double *seconds)
{
	SparkDraftBridgeConfig config;
	SparkDraftBridgeProposalInfo info;
	SparkDraftBridgeNode nodes[16];
	SparkDraftBridge *bridge = 0;
	SparkRowSampling sampling;
	const uint32_t committed[4] = {11u,12u,13u,14u};
	SparkStatus reply;
	uint64_t started;
	memset(&config,0,sizeof(config));
	memset(&sampling,0,sizeof(sampling));
	memset(&info,0,sizeof(info));
	config.abi_version = SPARK_DRAFT_BRIDGE_ABI_VERSION;
	config.descriptor_bytes = SPARK_DRAFT_BRIDGE_CONFIG_BYTES;
	config.host = "127.0.0.1";
	config.port = port;
	snprintf(config.target_model,sizeof(config.target_model),"%s",target);
	config.max_committed_tokens = 1024u;
	config.max_nodes = 16u;
	config.connect_timeout_ms = 2000u;
	config.io_timeout_ms = 2000u;
	assert(SparkDraftBridgeInitialize(&config,&bridge) == SPARK_STATUS_OK);
	started = SparkMonotonicNs();
	assert(SparkDraftBridgeRequestTree(bridge,6u,77u,3u,3u,14u,committed,4u,0,0u,4u,1000u,4u,&sampling,9u) == SPARK_STATUS_OK);
	*count = 0u;
	reply = SparkDraftBridgeReply(bridge,nodes,count,&info);
	while ( reply == SPARK_STATUS_PENDING && SparkMonotonicNs() - started < UINT64_C(5000000000) )
	{
		struct pollfd poller = {-1,0,0};
		uint32_t events = 0u;
		uint64_t deadline = 0u;
		if ( SparkDraftBridgePoll(bridge,&poller.fd,&events,&deadline) != 0u )
		{
			poller.events = (short)((events & SPARK_DRAFT_BRIDGE_POLL_READ) != 0u ? POLLIN : POLLOUT);
			(void)poll(&poller,1u,50);
		}
		reply = SparkDraftBridgeReply(bridge,nodes,count,&info);
	}
	assert(reply == SPARK_STATUS_OK || (reply == SPARK_STATUS_VALIDATION_FAILED && info.server_status != SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK));
	*seconds = (double)(SparkMonotonicNs() - started) / 1e9;
	SparkDraftBridgeDestroy(bridge);
	return(info.server_status);
}

int main(void)
{
	char root[] = "/tmp/draftd-engine-XXXXXX",weightd_socket[256],control_socket[256],pack[256],config[256],reply[8192];
	char lanes[3][1600],text[6000];
	TestServer state = {0};
	SparkWeightdServerConfig server_config = {0};
	const uint32_t probe_port = free_port(),huge_port = free_port(),abi_port = free_port();
	uint32_t count = 0u;
	double seconds = 0.0;
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
	lane_json(lanes[0],sizeof(lanes[0]),"probe",probe_port,SPARK_TEST_PROBE_MODULE,pack,UINT64_C(1) << 20,0);
	lane_json(lanes[1],sizeof(lanes[1]),"huge",huge_port,SPARK_TEST_PROBE_MODULE,pack,HUGE_CONTEXT,0);
	lane_json(lanes[2],sizeof(lanes[2]),"abi",abi_port,SPARK_TEST_PROBE_ABI_MODULE,pack,UINT64_C(1) << 20,1);
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
	assert(ask(probe_port,"probe-target",&count,&seconds) == SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK && count == 1u);
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
		"lane's own device memory, deactivate releases it, a second activate reuses the held copy, a lane whose pack and context cap exceed "
		"free device memory is refused with the numbers, a module of another drafter ABI is refused, and status reports every lane");
	return(0);
}
