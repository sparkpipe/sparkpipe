#define _POSIX_C_SOURCE 200809L

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#endif

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "fixtures/model_resident_deployment_fixture.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#ifndef TEST_MODEL_API_PATH
#define TEST_MODEL_API_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENTD_PATH
#define TEST_MODEL_RESIDENTD_PATH ""
#endif
#ifndef TEST_MODEL_SERVING_ADAPTER_PATH
#define TEST_MODEL_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENT_TRANSPORT_PATH
#define TEST_MODEL_RESIDENT_TRANSPORT_PATH ""
#endif

#define TEST_LOOPBACK_RANK_COUNT 3u
#define TEST_LOOPBACK_HTTP_DEADLINE_MS 10000u
#define TEST_LOOPBACK_OVERLOAD_REQUESTS 96u
#define TEST_LOOPBACK_FUZZ_DEADLINE_MS 5000u
#define TEST_LOOPBACK_TOKEN_FLOOR 4200u
#define TEST_LOOPBACK_TOKEN_CEILING 4210u
#define TEST_LOOPBACK_MAX_ACTIVE 16u
#define TEST_LOOPBACK_PREFILL_TOKEN_OFFSET 3u
#define TEST_LOOPBACK_TOKEN_LANE_CEILING (TEST_LOOPBACK_TOKEN_FLOOR + TEST_LOOPBACK_PREFILL_TOKEN_OFFSET + TEST_LOOPBACK_MAX_ACTIVE)

static const char *const TestLoopbackTransportHosts[TEST_LOOPBACK_RANK_COUNT] =
{
	"test-stage-a","test-stage-b","test-stage-c"
};

typedef struct TestLoopbackStack
{
	char root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	char deployment_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	char socket_paths[TEST_LOOPBACK_RANK_COUNT][108];
	pid_t residents[TEST_LOOPBACK_RANK_COUNT];
	pid_t api_child;
	uint32_t control_tcp_port;
	uint32_t api_port;
} TestLoopbackStack;

static TestLoopbackStack TestLoopbackTracked;
static uint32_t TestLoopbackTrackArmed;
static uint32_t TestLoopbackSendChunk;
static uint32_t TestLoopbackDisconnectAt;

static uint64_t TestLoopbackNowMs(void)
{
	struct timespec timestamp;
	assert(clock_gettime(CLOCK_MONOTONIC,&timestamp) == 0);
	return((uint64_t)timestamp.tv_sec * 1000u +
		(uint64_t)timestamp.tv_nsec / 1000000u);
}

static void TestLoopbackReapOrphans(void)
{
	uint32_t rank;
	if ( TestLoopbackTrackArmed == 0u )
		return;
	for (rank=0u; rank<TEST_LOOPBACK_RANK_COUNT; rank++)
		if ( TestLoopbackTracked.residents[rank] > 0 )
			(void)kill(TestLoopbackTracked.residents[rank],SIGKILL);
	if ( TestLoopbackTracked.api_child > 0 )
		(void)kill(TestLoopbackTracked.api_child,SIGKILL);
}

static void TestLoopbackFatalSignal(int32_t signal_number)
{
	TestLoopbackReapOrphans();
	(void)signal(signal_number,SIG_DFL);
	(void)raise(signal_number);
	_exit(128 + signal_number);
}

static void TestLoopbackTrackReset(void)
{
	memset(&TestLoopbackTracked,0,sizeof(TestLoopbackTracked));
	TestLoopbackTrackArmed = 1u;
}

static uint32_t TestLoopbackProbeFreeTcpPort(void)
{
	struct sockaddr_in address;
	socklen_t address_length;
	int32_t fd;
	uint32_t port;
	port = 0u;
	fd = socket(AF_INET,SOCK_STREAM,0);
	if ( fd >= 0 )
	{
		memset(&address,0,sizeof(address));
		address.sin_family = AF_INET;
		address.sin_addr.s_addr = htonl(0x7f000001u);
		address.sin_port = htons(0u);
		address_length = (socklen_t)sizeof(address);
		if ( bind(fd,(const struct sockaddr *)&address,(socklen_t)sizeof(address)) == 0 && getsockname(fd,(struct sockaddr *)&address,&address_length) == 0 )
			port = (uint32_t)ntohs(address.sin_port);
		close(fd);
	}
	return(port);
}

static int32_t TestLoopbackProbeTcpPort(uint32_t port)
{
	struct sockaddr_in address;
	int32_t fd;
	fd = socket(AF_INET,SOCK_STREAM,0);
	if ( fd < 0 )
		return(0);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(0x7f000001u);
	address.sin_port = htons((uint16_t)port);
	if ( connect(fd,(struct sockaddr *)&address,sizeof(address)) != 0 )
	{
		close(fd);
		return(0);
	}
	close(fd);
	return(1);
}

static void TestLoopbackCopyFile(const char *source,const char *target)
{
	FILE *input,*output;
	uint8_t buffer[8192];
	size_t chunk;
	input = fopen(source,"rb");
	assert(input != 0);
	output = fopen(target,"wb");
	assert(output != 0);
	while ( (chunk = fread(buffer,1u,sizeof(buffer),input)) != 0u )
		assert(fwrite(buffer,1u,chunk,output) == chunk);
	assert(ferror(input) == 0);
	assert(fclose(input) == 0);
	assert(fclose(output) == 0);
}

static void TestLoopbackSetupRoot(TestLoopbackStack *stack)
{
	char scratch[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	char path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	assert(snprintf(scratch,sizeof(scratch),"/tmp/sparkpipe-loopback-%ld-XXXXXX",(long)getpid()) > 0);
	assert(mkdtemp(scratch) != 0);
	assert(realpath(scratch,stack->root) != 0);
	assert(snprintf(path,sizeof(path),"%s/build",stack->root) > 0);
	assert(mkdir(path,0755) == 0);
	assert(snprintf(path,sizeof(path),"%s/build/test_modules",stack->root) > 0);
	assert(mkdir(path,0755) == 0);
	assert(snprintf(path,sizeof(path),"%s/%s",stack->root,TEST_MODEL_SERVING_ADAPTER_PATH) > 0);
	TestLoopbackCopyFile(TEST_MODEL_SERVING_ADAPTER_PATH,path);
	assert(snprintf(path,sizeof(path),"%s/%s",stack->root,TEST_MODEL_RESIDENT_TRANSPORT_PATH) > 0);
	TestLoopbackCopyFile(TEST_MODEL_RESIDENT_TRANSPORT_PATH,path);
	assert(snprintf(path,sizeof(path),"%s/tests",stack->root) > 0);
	assert(mkdir(path,0755) == 0);
	assert(snprintf(path,sizeof(path),"%s/tests/fixtures",stack->root) > 0);
	assert(mkdir(path,0755) == 0);
	assert(snprintf(path,sizeof(path),"%s/tests/fixtures/model_serving_adapter_config.json",stack->root) > 0);
	TestLoopbackCopyFile("tests/fixtures/model_serving_adapter_config.json",path);
}

static void TestLoopbackWriteDeployment(TestLoopbackStack *stack)
{
	TestModelResidentDeploymentFixture fixture;
	SparkModelResidentEndpoint endpoints[TEST_LOOPBACK_RANK_COUNT];
	const char *runtime_roots[TEST_LOOPBACK_RANK_COUNT];
	uint32_t stage_indices[TEST_LOOPBACK_RANK_COUNT];
	uint32_t rank;
	memset(endpoints,0,sizeof(endpoints));
	for (rank=0u; rank<TEST_LOOPBACK_RANK_COUNT; rank++)
	{
		runtime_roots[rank] = stack->root;
		stage_indices[rank] = rank;
		endpoints[rank].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[rank].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		if ( rank == 0u )
		{
			endpoints[rank].kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP;
			endpoints[rank].tcp_host = "127.0.0.1";
			endpoints[rank].tcp_port = stack->control_tcp_port;
		}
		else
		{
			assert(snprintf(stack->socket_paths[rank],sizeof(stack->socket_paths[rank]),"%s/rank%u.sock",stack->root,rank) > 0);
			unlink(stack->socket_paths[rank]);
			endpoints[rank].kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX;
			endpoints[rank].unix_socket_path = stack->socket_paths[rank];
		}
	}
	stage_indices[1] = 2u;
	stage_indices[2] = 1u;
	memset(&fixture,0,sizeof(fixture));
	fixture.adapter_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_program_name = "resident_decode";
	fixture.transport_shared_object_path = TEST_MODEL_RESIDENT_TRANSPORT_PATH;
	fixture.transport_mode = "host-rdma";
	fixture.node_target = "test.model.serving.target";
	fixture.adapter_configuration_path = "tests/fixtures/model_serving_adapter_config.json";
	fixture.runtime_roots = runtime_roots;
	fixture.transport_hosts = TestLoopbackTransportHosts;
	fixture.stage_indices = stage_indices;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	fixture.runtime_limits.max_inflight_submission_count = 2u;
	fixture.runtime_limits.max_active_sequence_count = TEST_LOOPBACK_MAX_ACTIVE;
	fixture.runtime_limits.max_input_row_count = 32u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 128u;
	fixture.runtime_limits.kv_physical_page_capacity = 64u;
	fixture.control_port_base = TestLoopbackProbeFreeTcpPort();
	if ( fixture.control_port_base == 0u || fixture.control_port_base > UINT16_MAX - (TEST_LOOPBACK_RANK_COUNT - 1u) )
		fixture.control_port_base = 59000u;
	fixture.node_count = TEST_LOOPBACK_RANK_COUNT;
	fixture.coordinator_rank_index = 0u;
	assert(snprintf(stack->deployment_path,sizeof(stack->deployment_path),"%s/deployment.json",stack->root) > 0);
	unlink(stack->deployment_path);
	assert(TestModelResidentDeploymentWrite(stack->deployment_path,&fixture) == 0);
}

static void TestLoopbackRedirectStderr(const char *path)
{
	int32_t fd;
	fd = open(path,O_WRONLY | O_CREAT | O_TRUNC,0644);
	if ( fd >= 0 )
	{
		(void)dup2(fd,2);
		(void)close(fd);
	}
}

static pid_t TestLoopbackStartResident(
	const TestLoopbackStack *stack,
	uint32_t rank_index)
{
	pid_t child;
	char rank[16];
	char log_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	assert(snprintf(rank,sizeof(rank),"%u",rank_index) > 0);
	assert(snprintf(log_path,sizeof(log_path),"%s/rank%u.log",stack->root,rank_index) > 0);
	child = fork();
	assert(child >= 0);
	if ( child == 0 )
	{
		TestLoopbackRedirectStderr(log_path);
		execl(TEST_MODEL_RESIDENTD_PATH,TEST_MODEL_RESIDENTD_PATH,
			"--deployment",stack->deployment_path,
			"--rank-index",rank,
			(char *)0);
		_exit(127);
	}
	return(child);
}

static void TestLoopbackWaitRankReady(
	const TestLoopbackStack *stack,
	uint32_t rank_index)
{
	struct stat status;
	struct timespec delay;
	uint32_t attempt;
	delay.tv_sec = 0;
	delay.tv_nsec = 2000000;
	for (attempt=0u; attempt<2500u; attempt++)
	{
		pid_t probe;
		int32_t child_status;
		probe = waitpid(stack->residents[rank_index],&child_status,WNOHANG);
		assert(probe == 0 || probe == -1);
		if ( rank_index == 0u )
		{
			if ( TestLoopbackProbeTcpPort(stack->control_tcp_port) != 0 )
				return;
		}
		else if ( lstat(stack->socket_paths[rank_index],&status) == 0 && S_ISSOCK(status.st_mode) )
			return;
		nanosleep(&delay,0);
	}
	assert(0 && "system loopback rank did not become ready");
}

static void TestLoopbackStartResidents(TestLoopbackStack *stack)
{
	uint32_t rank;
	for (rank=0u; rank<TEST_LOOPBACK_RANK_COUNT; rank++)
		stack->residents[rank] = TestLoopbackStartResident(stack,rank);
	for (rank=0u; rank<TEST_LOOPBACK_RANK_COUNT; rank++)
		TestLoopbackWaitRankReady(stack,rank);
}

static void TestLoopbackKillRank(TestLoopbackStack *stack,uint32_t rank_index)
{
	int32_t child_status;
	assert(stack->residents[rank_index] > 0);
	assert(kill(stack->residents[rank_index],SIGKILL) == 0);
	assert(waitpid(stack->residents[rank_index],&child_status,0) == stack->residents[rank_index]);
	assert(WIFSIGNALED(child_status) && WTERMSIG(child_status) == SIGKILL);
	stack->residents[rank_index] = 0;
	TestLoopbackTracked.residents[rank_index] = 0;
	if ( rank_index != 0u )
		unlink(stack->socket_paths[rank_index]);
}

static void TestLoopbackRestartRank(TestLoopbackStack *stack,uint32_t rank_index)
{
	assert(stack->residents[rank_index] == 0);
	stack->residents[rank_index] = TestLoopbackStartResident(stack,rank_index);
	TestLoopbackTracked.residents[rank_index] = stack->residents[rank_index];
	TestLoopbackWaitRankReady(stack,rank_index);
}

static void TestLoopbackStartApi(TestLoopbackStack *stack)
{
	struct timespec delay;
	uint32_t attempt;
	stack->api_child = fork();
	assert(stack->api_child >= 0);
	if ( stack->api_child == 0 )
	{
		char port[16];
		char log_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
		char *argv[10];
		int argument;
		(void)snprintf(port,sizeof(port),"%u",stack->api_port);
		assert(snprintf(log_path,sizeof(log_path),"%s/api.log",stack->root) > 0);
		TestLoopbackRedirectStderr(log_path);
		argument = 0;
		argv[argument++] = (char *)TEST_MODEL_API_PATH;
		argv[argument++] = "--deployment";
		argv[argument++] = stack->deployment_path;
		argv[argument++] = "--runtime-root";
		argv[argument++] = stack->root;
		argv[argument++] = "--port";
		argv[argument++] = port;
		argv[argument] = 0;
		execv(TEST_MODEL_API_PATH,argv);
		_exit(127);
	}
	TestLoopbackTracked.api_child = stack->api_child;
	delay.tv_sec = 0;
	delay.tv_nsec = 20000000;
	for (attempt=0u; attempt<750u; attempt++)
	{
		pid_t probe;
		int32_t child_status;
		probe = waitpid(stack->api_child,&child_status,WNOHANG);
		assert(probe == 0 || probe == -1);
		if ( TestLoopbackProbeTcpPort(stack->api_port) != 0 )
			return;
		nanosleep(&delay,0);
	}
	assert(0 && "system loopback model_api did not become ready");
}

static void TestLoopbackKillApi(TestLoopbackStack *stack,int32_t signal_number)
{
	int32_t child_status;
	assert(stack->api_child > 0);
	assert(kill(stack->api_child,signal_number) == 0);
	assert(waitpid(stack->api_child,&child_status,0) == stack->api_child);
	assert(WIFEXITED(child_status) || WIFSIGNALED(child_status));
	stack->api_child = 0;
	TestLoopbackTracked.api_child = 0;
}

static int32_t TestLoopbackHttpPost(
	uint32_t port,
	const char *body,
	char *response,
	size_t response_capacity,
	uint64_t deadline_ms,
	uint64_t *elapsed_ms_out)
{
	struct sockaddr_in address;
	struct timeval timeout;
	uint64_t started,now;
	ssize_t chunk,sent;
	size_t body_length,received,sent_total;
	char request[4096];
	int32_t fd;
	int32_t status;
	started = TestLoopbackNowMs();
	response[0] = '\0';
	fd = socket(AF_INET,SOCK_STREAM,0);
	assert(fd >= 0);
	timeout.tv_sec = 0;
	timeout.tv_usec = 250000;
	assert(setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)) == 0);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(0x7f000001u);
	address.sin_port = htons((uint16_t)port);
	if ( connect(fd,(struct sockaddr *)&address,sizeof(address)) != 0 )
	{
		close(fd);
		if ( elapsed_ms_out != 0 )
			*elapsed_ms_out = TestLoopbackNowMs() - started;
		return(0);
	}
	body_length = strlen(body);
	assert(body_length < sizeof(request) - 256u);
	assert(snprintf(request,sizeof(request),
		"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
		"Content-Type: application/json\r\nContent-Length: %zu\r\n"
		"Connection: close\r\n\r\n%s",
		body_length,body) > 0);
	sent_total = 0u;
	while ( sent_total < strlen(request) )
	{
		size_t remaining = strlen(request) - sent_total;
		if ( TestLoopbackSendChunk != 0u && remaining > TestLoopbackSendChunk )
			remaining = TestLoopbackSendChunk;
		if ( TestLoopbackDisconnectAt != 0u && sent_total >= TestLoopbackDisconnectAt )
		{
			close(fd);
			return(0);
		}
		sent = send(fd,request + sent_total,remaining,0);
		if ( sent <= 0 )
		{
			close(fd);
			if ( elapsed_ms_out != 0 )
				*elapsed_ms_out = TestLoopbackNowMs() - started;
			return(0);
		}
		sent_total += (size_t)sent;
	}
	if ( TestLoopbackDisconnectAt == UINT32_MAX )
	{
		close(fd);
		return(0);
	}
	received = 0u;
	for (;;)
	{
		now = TestLoopbackNowMs();
		if ( now - started >= deadline_ms )
			break;
		if ( received + 1u >= response_capacity )
			break;
		chunk = recv(fd,response + received,response_capacity - 1u - received,0);
		if ( chunk == 0 )
			break;
		if ( chunk < 0 )
		{
			if ( errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR )
				continue;
			break;
		}
		received += (size_t)chunk;
	}
	close(fd);
	response[received] = '\0';
	if ( elapsed_ms_out != 0 )
		*elapsed_ms_out = TestLoopbackNowMs() - started;
	if ( received == 0u )
		return(0);
	{
		const char *space;
		space = strchr(response,' ');
		if ( space == 0 )
			return(0);
		status = atoi(space + 1);
	}
	return(status);
}

static const char *TestLoopbackResponseBody(const char *response)
{
	const char *split;
	split = strstr(response,"\r\n\r\n");
	assert(split != 0);
	return(split + 4);
}

static uint32_t TestLoopbackScanTokenArray(
	const char *body,
	uint32_t *tokens,
	uint32_t capacity)
{
	const char *cursor;
	uint32_t count;
	cursor = strstr(body,"\"tokens\":[");
	assert(cursor != 0);
	cursor += strlen("\"tokens\":[");
	count = 0u;
	while ( *cursor != ']' )
	{
		char *end;
		long value;
		value = strtol(cursor,&end,10);
		assert(end != cursor);
		assert(count < capacity);
		tokens[count++] = (uint32_t)value;
		cursor = end;
		while ( *cursor == ',' || *cursor == ' ' )
			cursor++;
	}
	return(count);
}

static uint64_t TestLoopbackExpectServed(
	const TestLoopbackStack *stack,
	uint32_t max_tokens,
	uint64_t deadline_ms,
	uint64_t elapsed_limit_ms)
{
	char body[64];
	char response[65536];
	uint32_t tokens[512];
	uint32_t token_count,index;
	uint64_t elapsed,signature = 0u;
	int32_t status;
	assert(snprintf(body,sizeof(body),
		"{\"prompt_token_ids\":[11,12],\"max_tokens\":%u}",max_tokens) > 0);
	status = TestLoopbackHttpPost(stack->api_port,body,response,sizeof(response),
		deadline_ms,&elapsed);
	if ( status != 200 )
		fprintf(stderr,"test_system_loopback: expected 200, got %d after %llums; response: %.256s\n",
			(int)status,(unsigned long long)elapsed,response);
	assert(status == 200);
	assert(elapsed <= elapsed_limit_ms);
	token_count = TestLoopbackScanTokenArray(TestLoopbackResponseBody(response),
		tokens,512u);
	assert(token_count == max_tokens);
	for (index=0u; index<token_count; index++)
	{
		assert(tokens[index] >= TEST_LOOPBACK_TOKEN_FLOOR && tokens[index] < TEST_LOOPBACK_TOKEN_CEILING);
		signature = signature * 16u + tokens[index] - TEST_LOOPBACK_TOKEN_FLOOR;
	}
	return(signature);
}

static void TestLoopbackExpectStatus(const TestLoopbackStack *stack,const char *body,int32_t expected,char *response,size_t capacity)
{
	int32_t status = TestLoopbackHttpPost(stack->api_port,body,response,capacity,TEST_LOOPBACK_HTTP_DEADLINE_MS,0);
	if ( status != expected )
		fprintf(stderr,"test_system_loopback: %s expected %d, got %d; response: %.512s\n",body,(int)expected,(int)status,response);
	assert(status == expected);
}

static void TestLoopbackTokens(const TestLoopbackStack *stack,const char *body,uint32_t *tokens)
{
	char response[65536];
	TestLoopbackExpectStatus(stack,body,200,response,sizeof(response));
	assert(TestLoopbackScanTokenArray(TestLoopbackResponseBody(response),tokens,16u) == 4u);
}

static void TestLoopbackSampling(const TestLoopbackStack *stack)
{
	char response[65536];
	uint32_t first[16],second[16],other[16],greedy[16],index,differ = 0u;
	TestLoopbackTokens(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"temperature\":0.8,\"seed\":42}",first);
	TestLoopbackTokens(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"temperature\":0.8,\"seed\":42}",second);
	TestLoopbackTokens(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"temperature\":0.8,\"seed\":43,\"top_p\":1}",other);
	TestLoopbackTokens(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"temperature\":0,\"seed\":42}",greedy);
	for (index=0u; index<4u; index++)
	{
		assert(first[index] == second[index]);
		differ += first[index] != other[index] ? 1u : 0u;
	}
	assert(differ == 4u && memcmp(first,greedy,4u * sizeof(uint32_t)) != 0);
	TestLoopbackTokens(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"temperature\":0.8}",other);
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"temperature\":3}",400,response,sizeof(response));
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"temperature\":\"0.5\"}",400,response,sizeof(response));
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"temperature\":0.5,\"top_p\":0.9}",400,response,sizeof(response));
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"seed\":-1}",400,response,sizeof(response));
}

static void TestLoopbackServingOptions(const TestLoopbackStack *stack)
{
	char response[65536];
	uint32_t tokens[64],count,events,index;
	const char *cursor;
	count = events = 0u;
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"stream\":true}",200,response,sizeof(response));
	assert(strstr(response,"Content-Type: text/event-stream") != 0 && strstr(response,"data: [DONE]") != 0);
	for (cursor = strstr(response,"data: {"); cursor != 0; cursor = strstr(cursor + 1,"data: {"))
	{
		events++;
		count += TestLoopbackScanTokenArray(cursor,tokens + count,64u - count);
	}
	assert(count == 4u && events >= 2u && strstr(response,"\"finish_reason\":\"length\"") != 0 && strstr(response,"\"completion_tokens\":4") != 0);
	for (index=0u; index<count; index++)
		assert(tokens[index] >= TEST_LOOPBACK_TOKEN_FLOOR && tokens[index] < TEST_LOOPBACK_TOKEN_CEILING);
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":200,\"deadline_ms\":1}",504,response,sizeof(response));
	assert(strstr(response,"deadline_exceeded") != 0);
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"priority\":7}",200,response,sizeof(response));
	assert(strstr(response,"\"finish_reason\":\"length\"") != 0 && strstr(response,"\"total_tokens\":6") != 0);
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":512}",400,response,sizeof(response));
	assert(strstr(response,"context_length_exceeded") != 0);
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":512,\"stream\":true}",400,response,sizeof(response));
	assert(strstr(response,"context_length_exceeded") != 0 && strstr(response,"text/event-stream") == 0);
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"priority\":-1}",400,response,sizeof(response));
	TestLoopbackExpectStatus(stack,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4,\"stream\":3}",400,response,sizeof(response));
	TestLoopbackExpectServed(stack,4u,(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS,(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS);
}

static pid_t TestLoopbackForkRequest(
	uint32_t port,
	const char *body,
	uint64_t deadline_ms)
{
	pid_t child;
	child = fork();
	assert(child >= 0);
	if ( child == 0 )
	{
		char response[65536];
		uint64_t elapsed;
		int32_t status;
		elapsed = 0u;
		status = TestLoopbackHttpPost(port,body,response,sizeof(response),
			deadline_ms,&elapsed);
		if ( status == 200 )
		{
			uint32_t tokens[16],count,index;
			count = TestLoopbackScanTokenArray(TestLoopbackResponseBody(response),tokens,16u);
			assert(count == 4u);
			for (index=0u; index<count; index++)
				assert(tokens[index] >= TEST_LOOPBACK_TOKEN_FLOOR && tokens[index] < TEST_LOOPBACK_TOKEN_LANE_CEILING);
			_exit(0);
		}
		if ( status >= 500 && status <= 599 && strstr(response,"error") != 0 )
		{
			fprintf(stderr,"test_system_loopback: in-flight request answered "
				"status=%d after %llums\n",(int)status,(unsigned long long)elapsed);
			_exit(10);
		}
		fprintf(stderr,"test_system_loopback: in-flight request produced no "
			"response within %llums\n",(unsigned long long)deadline_ms);
		_exit(20);
	}
	return(child);
}

static int32_t TestLoopbackJoinRequest(pid_t child,uint64_t deadline_ms)
{
	struct timespec delay;
	uint64_t started;
	int32_t child_status;
	delay.tv_sec = 0;
	delay.tv_nsec = 10000000;
	started = TestLoopbackNowMs();
	for (;;)
	{
		pid_t probe;
		probe = waitpid(child,&child_status,WNOHANG);
		if ( probe == child )
		{
			if ( WIFEXITED(child_status) )
				return(WEXITSTATUS(child_status));
			return(-1);
		}
		assert(probe == 0 || probe == -1);
		if ( TestLoopbackNowMs() - started >= deadline_ms )
		{
			(void)kill(child,SIGKILL);
			assert(waitpid(child,&child_status,0) == child);
			return(-1);
		}
		nanosleep(&delay,0);
	}
}

static void TestLoopbackOverload(const TestLoopbackStack *stack)
{
	pid_t requests[TEST_LOOPBACK_OVERLOAD_REQUESTS];
	uint32_t index;
	for (index=0u; index<TEST_LOOPBACK_OVERLOAD_REQUESTS; index++)
		requests[index] = TestLoopbackForkRequest(stack->api_port,"{\"prompt_token_ids\":[11,12],\"max_tokens\":4}",TEST_LOOPBACK_HTTP_DEADLINE_MS);
	for (index=0u; index<TEST_LOOPBACK_OVERLOAD_REQUESTS; index++)
		assert(TestLoopbackJoinRequest(requests[index],TEST_LOOPBACK_HTTP_DEADLINE_MS) == 0);
}

static void TestLoopbackSleepMs(uint64_t milliseconds)
{
	struct timespec delay;
	uint64_t until;
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	until = TestLoopbackNowMs() + milliseconds;
	while ( TestLoopbackNowMs() < until )
		nanosleep(&delay,0);
}

static void TestLoopbackResurrectIdleRank(TestLoopbackStack *stack)
{
	pid_t request_child;
	int32_t outcome;
	TestLoopbackSleepMs(750u);
	TestLoopbackKillRank(stack,1u);
	request_child = TestLoopbackForkRequest(stack->api_port,
		"{\"prompt_token_ids\":[11,12],\"max_tokens\":4}",
		(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS);
	{
		struct timespec delay;
		delay.tv_sec = 0;
		delay.tv_nsec = 300000000;
		nanosleep(&delay,0);
	}
	TestLoopbackRestartRank(stack,1u);
	outcome = TestLoopbackJoinRequest(request_child,
		(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS + 5000u);
	if ( outcome != 0 && outcome != 10 )
		fprintf(stderr,"test_system_loopback: idle-kill request outcome=%d\n",
			(int)outcome);
	assert(outcome == 0 || outcome == 10);
	TestLoopbackExpectServed(stack,4u,(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS,
		(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS);
	printf("test_system_loopback: idle rank kill -> recover -> serve OK\n");
}

static void TestLoopbackResurrectApi(TestLoopbackStack *stack)
{
	TestLoopbackKillApi(stack,SIGKILL);
	TestLoopbackStartApi(stack);
	TestLoopbackExpectServed(stack,4u,(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS,
		(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS);
	printf("test_system_loopback: api kill -> restart -> serve OK\n");
}

static void TestLoopbackResurrectMidFlight(TestLoopbackStack *stack)
{
	pid_t request_child;
	int32_t outcome;
	struct timespec delay;
	request_child = TestLoopbackForkRequest(stack->api_port,
		"{\"prompt_token_ids\":[11,12],\"max_tokens\":4}",
		(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS);
	delay.tv_sec = 0;
	delay.tv_nsec = 100000000;
	{
		uint64_t kill_at = TestLoopbackNowMs() + 120u;
		while ( TestLoopbackNowMs() < kill_at )
			nanosleep(&delay,0);
	}
	TestLoopbackKillRank(stack,1u);
	nanosleep(&delay,0);
	TestLoopbackRestartRank(stack,1u);
	outcome = TestLoopbackJoinRequest(request_child,
		(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS + 5000u);
	if ( outcome != 0 && outcome != 10 )
		fprintf(stderr,"test_system_loopback: mid-flight request outcome=%d\n",
			(int)outcome);
	assert(outcome == 0 || outcome == 10);
	TestLoopbackExpectServed(stack,4u,(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS,
		(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS);
	printf("test_system_loopback: mid-flight rank kill -> recover -> serve OK\n");
}

static void TestLoopbackFuzz(TestLoopbackStack *stack,uint32_t rounds,uint32_t seed)
{
	uint32_t round,rank,kind,initial_seed = seed;
	uint32_t seen[8][TEST_LOOPBACK_RANK_COUNT] = {{0}};
	uint64_t started = TestLoopbackNowMs();
	for (round=0u; round<rounds; round++)
	{
		char response[65536];
		pid_t requests[3];
		uint32_t index;
		int32_t outcome;
		seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
		kind = round < 24u ? round / TEST_LOOPBACK_RANK_COUNT : (seed >> 16) % 8u;
		rank = round < 24u ? round % TEST_LOOPBACK_RANK_COUNT : (seed >> 8) % TEST_LOOPBACK_RANK_COUNT;
		seen[kind][rank]++;
		fprintf(stderr,"loopback seed=%u round=%u kind=%u rank=%u root=%s\n",
			initial_seed,round,kind,rank,stack->root);
		TestLoopbackSendChunk = 1u + seed % 31u;
		switch (kind)
		{
		case 0u:
			break;
		case 1u:
			TestLoopbackKillRank(stack,rank);
			TestLoopbackRestartRank(stack,rank);
			break;
		case 2u:
			TestLoopbackKillApi(stack,SIGKILL);
			TestLoopbackStartApi(stack);
			break;
		case 3u:
			for (index=0u; index<TEST_LOOPBACK_RANK_COUNT; index++)
				TestLoopbackKillRank(stack,index);
			for (index=0u; index<TEST_LOOPBACK_RANK_COUNT; index++)
				TestLoopbackRestartRank(stack,index);
			break;
		case 4u:
			for (index=0u; index<3u; index++)
				requests[index] = TestLoopbackForkRequest(stack->api_port,
					"{\"prompt_token_ids\":[11,12],\"max_tokens\":4}",TEST_LOOPBACK_HTTP_DEADLINE_MS);
			for (index=0u; index<3u; index++)
				assert(TestLoopbackJoinRequest(requests[index],TEST_LOOPBACK_HTTP_DEADLINE_MS) == 0);
			break;
		case 5u:
			assert(TestLoopbackHttpPost(stack->api_port,"{",response,sizeof(response),
				TEST_LOOPBACK_HTTP_DEADLINE_MS,0) == 400);
			break;
		case 6u:
			TestLoopbackDisconnectAt = 1u + seed % 80u;
			assert(TestLoopbackHttpPost(stack->api_port,
				"{\"prompt_token_ids\":[11,12],\"max_tokens\":4}",response,sizeof(response),
				TEST_LOOPBACK_HTTP_DEADLINE_MS,0) == 0);
			TestLoopbackDisconnectAt = UINT32_MAX;
			assert(TestLoopbackHttpPost(stack->api_port,
				"{\"prompt_token_ids\":[11,12],\"max_tokens\":4}",response,sizeof(response),
				TEST_LOOPBACK_HTTP_DEADLINE_MS,0) == 0);
			TestLoopbackDisconnectAt = 0u;
			break;
		case 7u:
			assert(kill(stack->residents[rank],SIGSTOP) == 0);
			requests[0] = TestLoopbackForkRequest(stack->api_port,
				"{\"prompt_token_ids\":[11,12],\"max_tokens\":4}",TEST_LOOPBACK_HTTP_DEADLINE_MS);
			TestLoopbackSleepMs(100u);
			TestLoopbackKillRank(stack,rank);
			TestLoopbackRestartRank(stack,rank);
			outcome = TestLoopbackJoinRequest(requests[0],TEST_LOOPBACK_HTTP_DEADLINE_MS);
			assert(outcome == 0 || outcome == 10);
			break;
		}
		TestLoopbackExpectServed(stack,1u + seed % 8u,TEST_LOOPBACK_HTTP_DEADLINE_MS,
			TEST_LOOPBACK_HTTP_DEADLINE_MS);
		assert(waitpid(stack->api_child,0,WNOHANG) == 0);
		for (index=0u; index<TEST_LOOPBACK_RANK_COUNT; index++)
			assert(waitpid(stack->residents[index],0,WNOHANG) == 0);
	}
	for (kind=0u; kind<8u; kind++)
		for (rank=0u; rank<TEST_LOOPBACK_RANK_COUNT; rank++)
			assert(seen[kind][rank] != 0u);
	printf("test_system_loopback: seed=%u rounds=%u scenarios=24 elapsed_ms=%llu\n",
		initial_seed,rounds,(unsigned long long)(TestLoopbackNowMs() - started));
}

static void TestLoopbackRemoveTree(const TestLoopbackStack *stack)
{
	char path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	uint32_t rank;
	for (rank=1u; rank<TEST_LOOPBACK_RANK_COUNT; rank++)
		unlink(stack->socket_paths[rank]);
	for (rank=0u; rank<TEST_LOOPBACK_RANK_COUNT; rank++)
	{
		assert(snprintf(path,sizeof(path),"%s/rank%u.log",stack->root,rank) > 0);
		unlink(path);
	}
	assert(snprintf(path,sizeof(path),"%s/api.log",stack->root) > 0);
	unlink(path);
	unlink(stack->deployment_path);
	assert(snprintf(path,sizeof(path),"%s/api_submission.seq",stack->root) > 0);
	unlink(path);
	assert(snprintf(path,sizeof(path),"%s/%s",stack->root,TEST_MODEL_SERVING_ADAPTER_PATH) > 0);
	unlink(path);
	assert(snprintf(path,sizeof(path),"%s/%s",stack->root,TEST_MODEL_RESIDENT_TRANSPORT_PATH) > 0);
	unlink(path);
	assert(snprintf(path,sizeof(path),"%s/build/test_modules",stack->root) > 0);
	assert(rmdir(path) == 0);
	assert(snprintf(path,sizeof(path),"%s/build",stack->root) > 0);
	assert(rmdir(path) == 0);
	assert(snprintf(path,sizeof(path),"%s/tests/fixtures/model_serving_adapter_config.json",stack->root) > 0);
	unlink(path);
	assert(snprintf(path,sizeof(path),"%s/tests/fixtures",stack->root) > 0);
	assert(rmdir(path) == 0);
	assert(snprintf(path,sizeof(path),"%s/tests",stack->root) > 0);
	assert(rmdir(path) == 0);
	assert(rmdir(stack->root) == 0);
}

static void TestLoopbackStopStack(TestLoopbackStack *stack)
{
	uint32_t rank;
	if ( stack->api_child > 0 )
		TestLoopbackKillApi(stack,SIGTERM);
	for (rank=0u; rank<TEST_LOOPBACK_RANK_COUNT; rank++)
		if ( stack->residents[rank] > 0 )
			TestLoopbackKillRank(stack,rank);
	TestLoopbackTrackReset();
	TestLoopbackTrackArmed = 0u;
	TestLoopbackRemoveTree(stack);
}

static void TestLoopbackBoot(TestLoopbackStack *stack)
{
	memset(stack,0,sizeof(*stack));
	TestLoopbackTrackReset();
	assert(setenv("SPARK_BATCH_INFLIGHT_BUDGET_NS","5000000000",1) == 0);
	stack->control_tcp_port = TestLoopbackProbeFreeTcpPort();
	assert(stack->control_tcp_port != 0u);
	stack->api_port = TestLoopbackProbeFreeTcpPort();
	assert(stack->api_port != 0u && stack->api_port != stack->control_tcp_port);
	TestLoopbackSetupRoot(stack);
	TestLoopbackWriteDeployment(stack);
	TestLoopbackStartResidents(stack);
	memcpy(TestLoopbackTracked.residents,stack->residents,sizeof(stack->residents));
	TestLoopbackStartApi(stack);
}

int main(int argc,char **argv)
{
	TestLoopbackStack stack;
	assert(atexit(TestLoopbackReapOrphans) == 0);
	assert(signal(SIGABRT,TestLoopbackFatalSignal) != SIG_ERR);
	assert(signal(SIGSEGV,TestLoopbackFatalSignal) != SIG_ERR);
	assert(signal(SIGPIPE,SIG_IGN) != SIG_ERR);
	if ( (argc == 3 || argc == 4) && strcmp(argv[1],"--fuzz") == 0 )
	{
		unsigned long rounds,seed = 1u;
		char *end;
		errno = 0;
		rounds = strtoul(argv[2],&end,10);
		if ( errno != 0 || end == argv[2] || *end != '\0' || rounds < 24u || rounds > 100000u )
			return(2);
		if ( argc == 4 )
		{
			errno = 0;
			seed = strtoul(argv[3],&end,10);
			if ( errno != 0 || end == argv[3] || *end != '\0' || seed > UINT32_MAX )
				return(2);
		}
		TestLoopbackBoot(&stack);
		assert(TestLoopbackExpectServed(&stack,4u,(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS,
			(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS) == UINT64_C(0x3010));
		TestLoopbackFuzz(&stack,(uint32_t)rounds,(uint32_t)seed);
		TestLoopbackStopStack(&stack);
		return(0);
	}
	assert(argc == 1);
	TestLoopbackBoot(&stack);
	TestLoopbackExpectServed(&stack,4u,(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS,
		(uint64_t)TEST_LOOPBACK_HTTP_DEADLINE_MS);
	printf("test_system_loopback: warm serve OK\n");
	TestLoopbackServingOptions(&stack);
	printf("test_system_loopback: streaming, deadline, priority and option validation OK\n");
	TestLoopbackSampling(&stack);
	printf("test_system_loopback: seeded sampling reproducible end to end OK\n");
	TestLoopbackOverload(&stack);
	printf("test_system_loopback: %u concurrent requests beyond engine capacity OK\n",TEST_LOOPBACK_OVERLOAD_REQUESTS);
	TestLoopbackResurrectIdleRank(&stack);
	TestLoopbackResurrectApi(&stack);
	TestLoopbackResurrectMidFlight(&stack);
	TestLoopbackStopStack(&stack);
	printf("test_system_loopback: ALL OK\n");
	return(0);
}
