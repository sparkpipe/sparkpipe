#include <assert.h>
#include <errno.h>
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
#include "sparkpipe/spark_tokenizer_sidecar.h"

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

#define TEST_RANK_COUNT 3u

static const char *const TestApiTransportHosts[TEST_RANK_COUNT] =
{
	"test-stage-a","test-stage-b","test-stage-c"
};

static void TestApiAppendUtf8(FILE *file, uint32_t code_point)
{
	if (code_point < 0x80u)
		fputc((int)code_point,file);
	else if (code_point < 0x800u)
	{
		fputc((int)(0xc0u | (code_point >> 6)),file);
		fputc((int)(0x80u | (code_point & 0x3fu)),file);
	}
	else
	{
		fputc((int)(0xe0u | (code_point >> 12)),file);
		fputc((int)(0x80u | ((code_point >> 6) & 0x3fu)),file);
		fputc((int)(0x80u | (code_point & 0x3fu)),file);
	}
}

static uint32_t TestApiByteGlyphCodePoint(uint8_t byte)
{
	uint32_t shifted;
	uint8_t candidate;
	if ((byte >= '!' && byte <= '~') ||
		(byte >= 0xa1 && byte <= 0xac) ||
		(byte >= 0xae))
		return (uint32_t)byte;
	shifted = 0u;
	for (candidate = 0u; ; candidate++)
	{
		if (!((candidate >= '!' && candidate <= '~') ||
			(candidate >= 0xa1 && candidate <= 0xac) ||
			(candidate >= 0xae)))
		{
			if (candidate == byte)
				return 256u + shifted;
			shifted++;
		}
		if (candidate == 255u)
			break;
	}
	return 0u;
}

static void TestApiWriteTokenizerFixture(const char *path)
{
	FILE *file;
	uint32_t byte_value;
	file = fopen(path,"wb");
	assert(file != 0);
	fprintf(file,
		"{\n"
		"  \"model\": {\n"
		"    \"type\": \"BPE\",\n"
		"    \"ignore_merges\": false,\n"
		"    \"byte_fallback\": false,\n"
		"    \"vocab\": {\n");
	for (byte_value = 0u; byte_value < 256u; byte_value++)
	{
		uint32_t code_point = TestApiByteGlyphCodePoint((uint8_t)byte_value);
		fprintf(file,"      \"");
		if (code_point == (uint32_t)'"')
			fputs("\\\"",file);
		else if (code_point == (uint32_t)'\\')
			fputs("\\\\",file);
		else
			TestApiAppendUtf8(file,code_point);
		fprintf(file,"\": %u,\n",byte_value);
	}
	for (byte_value = 0u; byte_value < 199u; byte_value++)
		fprintf(file,"      \"W%u\": %u,\n",4200u + byte_value,4200u + byte_value);
	fprintf(file,"      \"W4399\": 4399\n");
	fprintf(file,
		"    },\n"
		"    \"merges\": []\n"
		"  },\n"
		"  \"pre_tokenizer\": {\n"
		"    \"type\": \"Sequence\",\n"
		"    \"pretokenizers\": [\n"
		"      {\"type\": \"Split\", \"pattern\": {\"Regex\": \"%s\"}, \"behavior\": \"Isolated\", \"invert\": false},\n"
		"      {\"type\": \"ByteLevel\", \"add_prefix_space\": false}\n"
		"    ]\n"
		"  },\n"
		"  \"added_tokens\": []\n"
		"}\n",
		"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+");
	assert(fclose(file) == 0);
}

static uint32_t TestApiProbeFreeTcpPort(void)
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
		if ( bind(fd,(const struct sockaddr *)&address,(socklen_t)sizeof(address)) == 0 &&
			getsockname(fd,(struct sockaddr *)&address,&address_length) == 0 )
			port = (uint32_t)ntohs(address.sin_port);
		close(fd);
	}
	return(port);
}

static void TestApiWriteDeployment(const char *path,
	const SparkModelResidentEndpoint *endpoints,
	uint32_t control_port_base,
	const char *tokenizer_asset_path)
{
	TestModelResidentDeploymentFixture fixture;
	const char *runtime_roots[TEST_RANK_COUNT];
	uint32_t stage_indices[TEST_RANK_COUNT];
	char runtime_root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	uint32_t rank;
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	for (rank=0u; rank<TEST_RANK_COUNT; rank++)
	{
		runtime_roots[rank] = runtime_root;
		stage_indices[rank] = rank;
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
	fixture.tokenizer_asset_path = tokenizer_asset_path;
	fixture.runtime_roots = runtime_roots;
	fixture.transport_hosts = TestApiTransportHosts;
	fixture.stage_indices = stage_indices;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	fixture.runtime_limits.max_inflight_submission_count = 2u;
	fixture.runtime_limits.max_active_sequence_count = 16u;
	fixture.runtime_limits.max_input_row_count = 32u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 128u;
	fixture.runtime_limits.kv_physical_page_capacity = 32u;
	fixture.control_port_base = control_port_base;
	fixture.node_count = TEST_RANK_COUNT;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
}

static pid_t TestApiStartResident(const char *deployment_path,uint32_t rank_index)
{
	pid_t child;
	char rank[16];
	assert(snprintf(rank,sizeof(rank),"%u",rank_index) > 0);
	child = fork();
	assert(child >= 0);
	if ( child == 0 )
	{
		execl(TEST_MODEL_RESIDENTD_PATH,TEST_MODEL_RESIDENTD_PATH,
			"--deployment",deployment_path,
			"--rank-index",rank,
			(char *)0);
		_exit(127);
	}
	return(child);
}

static void TestApiWaitForSockets(char paths[][108])
{
	struct stat status;
	struct timespec delay;
	uint32_t attempt,rank,ready;
	delay.tv_sec = 0;
	delay.tv_nsec = 10000000;
	for (attempt=0u; attempt<500u; attempt++)
	{
		ready = 1u;
		for (rank=1u; rank<TEST_RANK_COUNT; rank++)
			if ( lstat(paths[rank],&status) != 0 || !S_ISSOCK(status.st_mode) )
				ready = 0u;
		if ( ready != 0u )
			return;
		nanosleep(&delay,0);
	}
	assert(0 && "model api test sockets did not become ready");
}

static void TestApiStopResidents(pid_t children[TEST_RANK_COUNT],char paths[][108])
{
	uint32_t rank;
	int32_t child_status;
	for (rank=0u; rank<TEST_RANK_COUNT; rank++)
	{
		assert(kill(children[rank],SIGTERM) == 0 || errno == ESRCH);
		assert(waitpid(children[rank],&child_status,0) == children[rank]);
		assert(WIFEXITED(child_status));
		assert((uint32_t)WEXITSTATUS(child_status) == 0u ||
			(uint32_t)WEXITSTATUS(child_status) == 1u);
		unlink(paths[rank]);
	}
}


static int TestApiHttpProbePort(uint32_t port)
{
	struct sockaddr_in address;
	int fd;
	fd = socket(AF_INET,SOCK_STREAM,0);
	if ( fd < 0 )
		return 0;
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(0x7f000001u);
	address.sin_port = htons((uint16_t)port);
	if ( connect(fd,(struct sockaddr *)&address,sizeof(address)) != 0 )
	{
		close(fd);
		return 0;
	}
	close(fd);
	return 1;
}

static void TestApiHttpCallRetry(uint32_t port,const char *method,const char *path,
	const char *body,char *response,size_t response_capacity,uint32_t attempt)
{
	struct sockaddr_in address;
	char request[4096];
	int fd;
	ssize_t sent;
	size_t body_len = body != 0 ? strlen(body) : 0u;
	size_t received = 0u;
	struct timeval timeout;
	struct timespec delay;
	delay.tv_sec = 0;
	delay.tv_nsec = 100000000;
	assert(body_len < sizeof(request) - 256u);
	fd = socket(AF_INET,SOCK_STREAM,0);
	assert(fd >= 0);
	timeout.tv_sec = 20;
	timeout.tv_usec = 0;
	assert(setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)) == 0);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(0x7f000001u);
	address.sin_port = htons((uint16_t)port);
	assert(connect(fd,(struct sockaddr *)&address,sizeof(address)) == 0);
	if ( body != 0 )
		(void)snprintf(request,sizeof(request),
			"%s %s HTTP/1.1\r\nHost: localhost\r\n"
			"Content-Type: application/json\r\nContent-Length: %zu\r\n"
			"Connection: close\r\n\r\n%s",
			method,path,body_len,body);
	else
		(void)snprintf(request,sizeof(request),
			"%s %s HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
			method,path);
	sent = send(fd,request,strlen(request),0);
	assert(sent == (ssize_t)strlen(request));
	while ( received + 1u < response_capacity )
	{
		ssize_t chunk = recv(fd,response + received,response_capacity - 1u - received,0);
		if ( chunk <= 0 )
		{
			if ( received == 0u && attempt < 2u )
			{
				close(fd);
				nanosleep(&delay,0);
				return(TestApiHttpCallRetry(port,method,path,body,response,
					response_capacity,attempt + 1u));
			}
			break;
		}
		received += (size_t)chunk;
	}
	close(fd);
	response[received] = '\0';
	assert(received != 0u);
}

static void TestApiHttpCall(uint32_t port,const char *method,const char *path,
	const char *body,char *response,size_t response_capacity)
{
	TestApiHttpCallRetry(port,method,path,body,response,response_capacity,0u);
}

static int TestApiResponseStatus(const char *response)
{
	const char *space = strchr(response,' ');
	int code;
	if ( space == 0 )
		return 0;
	code = atoi(space + 1);
	return code;
}

static const char *TestApiResponseJsonBody(const char *response)
{
	const char *split = strstr(response,"\r\n\r\n");
	assert(split != 0);
	return split + 4;
}


static uint32_t TestApiScanTokenArray(const char *body,uint32_t *tokens,uint32_t capacity)
{
	const char *cursor = strstr(body,"\"tokens\":[");
	uint32_t count = 0u;
	assert(cursor != 0);
	cursor += strlen("\"tokens\":[");
	while ( *cursor != ']' )
	{
		char *end;
		long value = strtol(cursor,&end,10);
		assert(end != cursor);
		assert(count < capacity);
		tokens[count++] = (uint32_t)value;
		cursor = end;
		while ( *cursor == ',' || *cursor == ' ' )
			cursor++;
	}
	return count;
}

static void TestApiScanTextField(const char *body,char *text,size_t text_capacity)
{
	const char *cursor = strstr(body,"\"text\":\"");
	size_t out = 0u;
	assert(cursor != 0);
	cursor += strlen("\"text\":\"");
	while ( *cursor != '"' )
	{
		assert(out + 1u < text_capacity);
		if ( *cursor == '\\' )
		{
			cursor++;
			switch (*cursor)
			{
				case 'n': text[out++] = '\n'; break;
				case 't': text[out++] = '\t'; break;
				case 'r': text[out++] = '\r'; break;
				case '"': text[out++] = '"'; break;
				case '\\': text[out++] = '\\'; break;
				default: text[out++] = *cursor; break;
			}
			cursor++;
		}
		else
			text[out++] = *cursor++;
	}
	text[out] = '\0';
}

static int TestApiBodyContains(const char *body,const char *needle)
{
	return strstr(body,needle) != 0;
}


typedef struct TestApiStack
{
	pid_t residents[TEST_RANK_COUNT];
	char paths[TEST_RANK_COUNT][108];
	char deployment_path[108];
	pid_t api_child;
	uint32_t api_port;
} TestApiStack;

static void TestApiStartStack(TestApiStack *stack,const char *tokenizer_asset_path)
{
	SparkModelResidentEndpoint endpoints[TEST_RANK_COUNT];
	uint32_t tcp_port,rank;
	tcp_port = TestApiProbeFreeTcpPort();
	if ( tcp_port == 0u )
		tcp_port = 30000u + ((uint32_t)getpid() % 20000u);
	memset(endpoints,0,sizeof(endpoints));
	for (rank=0u; rank<TEST_RANK_COUNT; rank++)
	{
		assert(snprintf(stack->paths[rank],sizeof(stack->paths[rank]),
			"/tmp/sparkpipe-api-text-%ld-%u.sock",(long)getpid(),rank) > 0);
		unlink(stack->paths[rank]);
		endpoints[rank].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[rank].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		endpoints[rank].kind = rank == 0u ?
			SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP : SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX;
		endpoints[rank].tcp_port = rank == 0u ? tcp_port : 0u;
		endpoints[rank].tcp_host = rank == 0u ? "127.0.0.1" : 0;
		endpoints[rank].unix_socket_path = rank == 0u ? 0 : stack->paths[rank];
	}
	assert(snprintf(stack->deployment_path,sizeof(stack->deployment_path),
		"/tmp/sparkpipe-api-text-%ld.json",(long)getpid()) > 0);
	unlink(stack->deployment_path);
	TestApiWriteDeployment(stack->deployment_path,endpoints,tcp_port,tokenizer_asset_path);
	for (rank=0u; rank<TEST_RANK_COUNT; rank++)
		stack->residents[rank] = TestApiStartResident(stack->deployment_path,rank);
	TestApiWaitForSockets(stack->paths);
}

static void TestApiStartApi(TestApiStack *stack)
{
	struct timespec delay;
	uint32_t attempt;
	stack->api_child = fork();
	assert(stack->api_child >= 0);
	if ( stack->api_child == 0 )
	{
		char port[16];
		char *argv[10];
		char runtime_root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
		int argument = 0;
		(void)snprintf(port,sizeof(port),"%u",stack->api_port);
		assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
		argv[argument++] = (char *)TEST_MODEL_API_PATH;
		argv[argument++] = "--deployment";
		argv[argument++] = stack->deployment_path;
		argv[argument++] = "--runtime-root";
		argv[argument++] = runtime_root;
		argv[argument++] = "--port";
		argv[argument++] = port;
		argv[argument] = 0;
		execv(TEST_MODEL_API_PATH,argv);
		_exit(127);
	}
	delay.tv_sec = 0;
	delay.tv_nsec = 20000000;
	for (attempt=0u; attempt<250u; attempt++)
	{
		if ( TestApiHttpProbePort(stack->api_port) != 0 )
			return;
		nanosleep(&delay,0);
	}
	assert(0 && "model_api did not become ready");
}

static void TestApiStopApi(TestApiStack *stack)
{
	int32_t child_status;
	assert(kill(stack->api_child,SIGTERM) == 0 || errno == ESRCH);
	assert(waitpid(stack->api_child,&child_status,0) == stack->api_child);
	assert(WIFEXITED(child_status) || WIFSIGNALED(child_status));
	if ( WIFSIGNALED(child_status) )
		assert(WTERMSIG(child_status) == SIGTERM);
	stack->api_child = 0;
}

static void TestApiStopStack(TestApiStack *stack)
{
	if ( stack->api_child != 0 )
		TestApiStopApi(stack);
	TestApiStopResidents(stack->residents,stack->paths);
	unlink(stack->deployment_path);
}


static void TestApiTokenIdServingWithoutTokenizer(TestApiStack *stack)
{
	char response[65536];
	uint32_t tokens[64];
	uint32_t token_count;
	const char *body;
	TestApiHttpCall(stack->api_port,"POST","/v1/completions",
		"{\"prompt\":\"hello\"}",response,sizeof(response));
	assert(TestApiResponseStatus(response) == 400);
	body = TestApiResponseJsonBody(response);
	assert(TestApiBodyContains(body,"tokenizer_unavailable"));

	TestApiHttpCall(stack->api_port,"POST","/v1/completions",
		"{\"prompt\":\"hello\",\"prompt_token_ids\":[11,12]}",
		response,sizeof(response));
	assert(TestApiResponseStatus(response) == 400);
	body = TestApiResponseJsonBody(response);
	assert(TestApiBodyContains(body,"ambiguous_prompt"));

	TestApiHttpCall(stack->api_port,"POST","/v1/completions",
		"{\"prompt_token_ids\":[11,12],\"max_tokens\":2}",
		response,sizeof(response));
	assert(TestApiResponseStatus(response) == 200);
	body = TestApiResponseJsonBody(response);
	assert(TestApiBodyContains(body,"\"object\":\"text_completion\""));
	assert(TestApiBodyContains(body,"\"status\":0"));
	assert(!TestApiBodyContains(body,"\"text\":"));
	token_count = TestApiScanTokenArray(body,tokens,64u);
	assert(token_count >= 1u);

	TestApiHttpCall(stack->api_port,"POST","/v1/completions",
		"{\"prompt_token_ids\":[11],\"stop_token_ids\":[4200],\"max_tokens\":16}",
		response,sizeof(response));
	assert(TestApiResponseStatus(response) == 200);
	body = TestApiResponseJsonBody(response);
	token_count = TestApiScanTokenArray(body,tokens,64u);
	for ( uint32_t index = 0u; index < token_count; index++ )
		assert(tokens[index] != 4200u);

	TestApiHttpCall(stack->api_port,"GET","/health",0,response,sizeof(response));
	assert(TestApiResponseStatus(response) == 200);
	assert(TestApiBodyContains(response,"\"tokenizer\":false"));
	printf("test_model_api_text: no-tokenizer contract OK (400 naming the "
		"sidecar; token-id form intact)\n");
}

static void TestApiTextServingWithTokenizer(TestApiStack *stack)
{
	SparkTokenizerSidecar sidecar;
	SparkTokenizerSidecarConfiguration configuration;
	char response[65536];
	char text[16384];
	char *decoded;
	uint32_t decoded_bytes = 0;
	uint32_t tokens[64];
	uint32_t token_count;
	const char *body;
	SparkTokenizerSidecarReset(&sidecar);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TOKENIZER_SIDECAR_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.asset_path = "build/test_tokenizer_sidecar_api_hf.json";
	configuration.format = SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO;
	assert(SparkTokenizerSidecarLoad(&sidecar,&configuration) == SPARK_STATUS_OK);
	decoded = malloc(65536u);
	assert(decoded != 0);

	TestApiHttpCall(stack->api_port,"POST","/v1/completions",
		"{\"prompt\":\"hello\",\"max_tokens\":3}",response,sizeof(response));
	assert(TestApiResponseStatus(response) == 200);
	body = TestApiResponseJsonBody(response);
	assert(TestApiBodyContains(body,"\"object\":\"text_completion\""));
	assert(TestApiBodyContains(body,"\"status\":0"));
	token_count = TestApiScanTokenArray(body,tokens,64u);
	assert(token_count >= 1u);
	TestApiScanTextField(body,text,sizeof(text));
	assert(SparkTokenizerSidecarDecodeText(&sidecar,tokens,token_count,0,0,0u,
		decoded,65536u,&decoded_bytes) == SPARK_STATUS_OK);
	assert(decoded_bytes == strlen(text));
	assert(memcmp(decoded,text,decoded_bytes) == 0);

	TestApiHttpCall(stack->api_port,"POST","/v1/completions",
		"{\"prompt\":\"hello\",\"stop_token_ids\":[4200],\"max_tokens\":16}",
		response,sizeof(response));
	assert(TestApiResponseStatus(response) == 200);
	body = TestApiResponseJsonBody(response);
	token_count = TestApiScanTokenArray(body,tokens,64u);
	for ( uint32_t index = 0u; index < token_count; index++ )
		assert(tokens[index] != 4200u);
	TestApiScanTextField(body,text,sizeof(text));
	assert(SparkTokenizerSidecarDecodeText(&sidecar,tokens,token_count,0,0,0u,
		decoded,65536u,&decoded_bytes) == SPARK_STATUS_OK);
	assert(decoded_bytes == strlen(text));
	assert(memcmp(decoded,text,decoded_bytes) == 0);

	TestApiHttpCall(stack->api_port,"POST","/v1/completions",
		"{\"prompt_token_ids\":[11,12],\"max_tokens\":2}",
		response,sizeof(response));
	assert(TestApiResponseStatus(response) == 200);
	body = TestApiResponseJsonBody(response);
	assert(TestApiBodyContains(body,"\"text\":"));

	TestApiHttpCall(stack->api_port,"GET","/health",0,response,sizeof(response));
	assert(TestApiResponseStatus(response) == 200);
	assert(TestApiBodyContains(response,"\"tokenizer\":true"));

	free(decoded);
	SparkTokenizerSidecarUnload(&sidecar);
	printf("test_model_api_text: text-in/text-out contract OK (text == "
		"decode(tokens); stops bound the stream)\n");
}

static void TestApiMissingAssetIsFatal(void)
{
	TestApiStack stack;
	int32_t child_status;
	memset(&stack,0,sizeof(stack));
	stack.api_port = TestApiProbeFreeTcpPort();
	if ( stack.api_port == 0u )
		stack.api_port = 40000u + ((uint32_t)getpid() % 20000u);
	TestApiStartStack(&stack,"build/definitely_missing_tokenizer_asset.json");
	stack.api_child = fork();
	assert(stack.api_child >= 0);
	if ( stack.api_child == 0 )
	{
		char port[16];
		char *argv[10];
		char runtime_root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
		int argument = 0;
		(void)snprintf(port,sizeof(port),"%u",stack.api_port);
		assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
		argv[argument++] = (char *)TEST_MODEL_API_PATH;
		argv[argument++] = "--deployment";
		argv[argument++] = stack.deployment_path;
		argv[argument++] = "--runtime-root";
		argv[argument++] = runtime_root;
		argv[argument++] = "--port";
		argv[argument++] = port;
		argv[argument] = 0;
		execv(TEST_MODEL_API_PATH,argv);
		_exit(127);
	}
	assert(waitpid(stack.api_child,&child_status,0) == stack.api_child);
	assert(WIFEXITED(child_status));
	assert(WEXITSTATUS(child_status) != 0);
	{
		struct timespec delay;
		delay.tv_sec = 0;
		delay.tv_nsec = 20000000;
		nanosleep(&delay,0);
		assert(TestApiHttpProbePort(stack.api_port) == 0);
	}
	stack.api_child = 0;
	TestApiStopStack(&stack);
	printf("test_model_api_text: missing tokenizer asset refuses startup OK\n");
}

int main(void)
{
	TestApiStack stack;
	TestApiWriteTokenizerFixture("build/test_tokenizer_sidecar_api_hf.json");

	memset(&stack,0,sizeof(stack));
	stack.api_port = TestApiProbeFreeTcpPort();
	if ( stack.api_port == 0u )
		stack.api_port = 40000u + ((uint32_t)getpid() % 20000u);
	TestApiStartStack(&stack,0);
	TestApiStartApi(&stack);
	TestApiTokenIdServingWithoutTokenizer(&stack);
	TestApiStopApi(&stack);
	TestApiStopResidents(stack.residents,stack.paths);
	unlink(stack.deployment_path);
	memset(&stack,0,sizeof(stack));
	stack.api_port = TestApiProbeFreeTcpPort();
	if ( stack.api_port == 0u )
		stack.api_port = 40000u + ((uint32_t)getpid() % 20000u);
	TestApiStartStack(&stack,"build/test_tokenizer_sidecar_api_hf.json");
	TestApiStartApi(&stack);
	TestApiTextServingWithTokenizer(&stack);
	TestApiStopStack(&stack);

	TestApiMissingAssetIsFatal();
	printf("test_model_api_text: ALL OK\n");
	return 0;
}
