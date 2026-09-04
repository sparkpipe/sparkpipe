#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_draft_bridge.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_speculation_seam.h"
#include "sparkpipe/spark_qwen38_27b_model.h"
#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"

#ifndef QWEN38_27B_MODEL_REVISION
#error "QWEN38_27B_MODEL_REVISION must match the adapter build"
#endif
#ifndef TEST_QWEN38_27B_REMOTE_SPEC_ADAPTER_PATH
#define TEST_QWEN38_27B_REMOTE_SPEC_ADAPTER_PATH ""
#endif
#ifndef TEST_QWEN38_27B_REMOTE_SPEC_DRIVER_PATH
#define TEST_QWEN38_27B_REMOTE_SPEC_DRIVER_PATH ""
#endif
#ifndef TEST_QWEN38_27B_REMOTE_SPEC_CONFIG_PATH
#define TEST_QWEN38_27B_REMOTE_SPEC_CONFIG_PATH ""
#endif

#define TEST_REMOTE_EXCHANGE_COUNT 4u
#define TEST_REMOTE_COMMITTED_CAP 16u
#define TEST_REMOTE_NODE_COUNT 3u
#define TEST_REMOTE_LISTEN_BACKLOG 4
#define TEST_REMOTE_CAPTURE_BYTES 256u

#define TEST_REMOTE_CHECK(condition) \
	do { \
		if ( !(condition) ) \
		{ \
			fprintf(stderr,"CHECK-FAIL %s:%u: %s\n",__FILE__,(unsigned)__LINE__,#condition); \
			return(1u); \
		} \
	} while (0)

typedef struct TestRemoteStub
{
	uint32_t port;
	int listen_fd;
	pthread_t thread;
	uint32_t failed;
	uint32_t exchanges_completed;
	uint32_t last_mask[TEST_REMOTE_EXCHANGE_COUNT];
	uint64_t last_sequence_id[TEST_REMOTE_EXCHANGE_COUNT];
	uint64_t last_generation[TEST_REMOTE_EXCHANGE_COUNT];
	uint32_t last_committed_count[TEST_REMOTE_EXCHANGE_COUNT];
	uint32_t last_committed[TEST_REMOTE_EXCHANGE_COUNT][TEST_REMOTE_COMMITTED_CAP];
} TestRemoteStub;

static const uint32_t TestRemoteChains[TEST_REMOTE_EXCHANGE_COUNT][2] =
{
	{ 6001u, 6002u },
	{ 6011u, 6012u },
	{ 6021u, 6022u },
	{ 6031u, 6032u }
};

static void TestRemotePutU32Le(uint8_t *destination, uint32_t value)
{
	destination[0] = (uint8_t)(value & 0xFFu);
	destination[1] = (uint8_t)((value >> 8u) & 0xFFu);
	destination[2] = (uint8_t)((value >> 16u) & 0xFFu);
	destination[3] = (uint8_t)((value >> 24u) & 0xFFu);
}

static void TestRemotePutU64Le(uint8_t *destination, uint64_t value)
{
	uint32_t shift;
	for (shift=0u; shift<64u; shift+=8u)
		destination[shift / 8u] = (uint8_t)(value >> shift);
}

static void TestRemotePutF32Le(uint8_t *destination, float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	TestRemotePutU32Le(destination,bits);
}

static uint32_t TestRemoteGetU32Le(const uint8_t *source)
{
	return((uint32_t)source[0] | ((uint32_t)source[1] << 8u) | ((uint32_t)source[2] << 16u) | ((uint32_t)source[3] << 24u));
}

static uint64_t TestRemoteGetU64Le(const uint8_t *source)
{
	uint64_t value;
	uint32_t shift;
	value = 0u;
	for (shift=0u; shift<64u; shift+=8u)
		value |= (uint64_t)source[shift / 8u] << shift;
	return(value);
}

static int TestRemoteReadExact(int connection, uint8_t *data, uint32_t data_bytes)
{
	uint32_t offset;
	ssize_t received;
	offset = 0u;
	while ( offset < data_bytes )
	{
		received = recv(connection,data + offset,data_bytes - offset,0);
		if ( received < 0 && errno == EINTR )
			continue;
		if ( received <= 0 )
			return(-1);
		offset += (uint32_t)received;
	}
	return(0);
}

static int TestRemoteWriteAll(int connection, const uint8_t *data, uint32_t data_bytes)
{
	uint32_t offset;
	ssize_t written;
	offset = 0u;
	while ( offset < data_bytes )
	{
		written = send(connection,data + offset,data_bytes - offset,0);
		if ( written < 0 && errno == EINTR )
			continue;
		if ( written <= 0 )
			return(-1);
		offset += (uint32_t)written;
	}
	return(0);
}

static void *TestRemoteStubMain(void *argument)
{
	TestRemoteStub *stub;
	uint8_t header[SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES];
	uint8_t response[SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES + TEST_REMOTE_NODE_COUNT * SPARK_DRAFT_BRIDGE_NODE_RECORD_BYTES + SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES];
	uint32_t exchange_index;
	uint32_t committed_count,tap_count,node_index;
	uint32_t committed[TEST_REMOTE_COMMITTED_CAP];
	int connection;
	stub = (TestRemoteStub *)argument;
	connection = -1;
	for (exchange_index=0u; exchange_index<TEST_REMOTE_EXCHANGE_COUNT && stub->failed == 0u; exchange_index++)
	{
		uint8_t *cursor;
		if ( connection < 0 )
		{
			do {
				connection = accept(stub->listen_fd,0,0);
			} while ( connection < 0 && errno == EINTR );
			if ( connection < 0 )
			{
				stub->failed = 1u;
				break;
			}
		}
		if ( TestRemoteReadExact(connection,header,SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES) != 0 )
		{
			close(connection);
			connection = -1;
			exchange_index--;
			continue;
		}
		committed_count = TestRemoteGetU32Le(header + 72u);
		tap_count = TestRemoteGetU32Le(header + 76u);
		if ( committed_count == 0u || committed_count > TEST_REMOTE_COMMITTED_CAP || tap_count != 0u )
		{
			stub->failed = 1u;
			break;
		}
		if ( TestRemoteReadExact(connection,(uint8_t *)committed,committed_count * (uint32_t)sizeof(uint32_t)) != 0 )
		{
			stub->failed = 1u;
			break;
		}
		stub->last_mask[exchange_index] = TestRemoteGetU32Le(header + 8u);
		stub->last_sequence_id[exchange_index] = TestRemoteGetU64Le(header + 44u);
		stub->last_generation[exchange_index] = TestRemoteGetU64Le(header + 52u);
		stub->last_committed_count[exchange_index] = committed_count;
		memcpy(stub->last_committed[exchange_index],committed,committed_count * sizeof(uint32_t));
		cursor = response;
		memcpy(cursor,"DFT3",4u);
		cursor += 4u;
		TestRemotePutU32Le(cursor,SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK);
		cursor += sizeof(uint32_t);
		TestRemotePutU32Le(cursor,TEST_REMOTE_NODE_COUNT);
		cursor += sizeof(uint32_t);
		TestRemotePutU64Le(cursor,stub->last_sequence_id[exchange_index]);
		cursor += sizeof(uint64_t);
		TestRemotePutU64Le(cursor,stub->last_generation[exchange_index]);
		cursor += sizeof(uint64_t);
		for (node_index=0u; node_index<TEST_REMOTE_NODE_COUNT; node_index++)
		{
			TestRemotePutU32Le(cursor,node_index == 0u ? 0u : TestRemoteChains[exchange_index][node_index - 1u]);
			cursor += sizeof(uint32_t);
			TestRemotePutU32Le(cursor,node_index == 0u ? SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX : node_index - 1u);
			cursor += sizeof(uint32_t);
			TestRemotePutU32Le(cursor,node_index);
			cursor += sizeof(uint32_t);
			TestRemotePutU32Le(cursor,SPARK_SPECULATION_SEAM_SOURCE_NGRAM);
			cursor += sizeof(uint32_t);
			TestRemotePutF32Le(cursor,node_index == 0u ? 1.0f : 0.9f);
			cursor += sizeof(float);
		}
		memset(cursor,0,SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES);
		cursor += SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES;
		if ( TestRemoteWriteAll(connection,response,(uint32_t)(cursor - response)) != 0 )
		{
			close(connection);
			connection = -1;
		}
		stub->exchanges_completed++;
	}
	if ( connection >= 0 )
		close(connection);
	return(0);
}

static uint32_t TestRemoteStubStart(TestRemoteStub *stub)
{
	struct sockaddr_in address;
	socklen_t address_bytes;
	int reuse;
	memset(stub,0,sizeof(*stub));
	stub->listen_fd = socket(AF_INET,SOCK_STREAM,0);
	if ( stub->listen_fd < 0 )
		return(1u);
	reuse = 1;
	(void)setsockopt(stub->listen_fd,SOL_SOCKET,SO_REUSEADDR,&reuse,(socklen_t)sizeof(reuse));
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	address.sin_port = htons(0u);
	if ( bind(stub->listen_fd,(const struct sockaddr *)&address,(socklen_t)sizeof(address)) != 0 )
		return(1u);
	if ( listen(stub->listen_fd,TEST_REMOTE_LISTEN_BACKLOG) != 0 )
		return(1u);
	address_bytes = (socklen_t)sizeof(address);
	if ( getsockname(stub->listen_fd,(struct sockaddr *)&address,&address_bytes) != 0 )
		return(1u);
	stub->port = (uint32_t)ntohs(address.sin_port);
	if ( pthread_create(&stub->thread,0,TestRemoteStubMain,stub) != 0 )
		return(1u);
	return(0u);
}

static void TestRemoteStubStop(TestRemoteStub *stub)
{
	(void)pthread_join(stub->thread,0);
	(void)close(stub->listen_fd);
}

static uint32_t TestRemoteClosedPort(void)
{
	struct sockaddr_in address;
	socklen_t address_bytes;
	uint32_t port;
	int fd;
	fd = socket(AF_INET,SOCK_STREAM,0);
	if ( fd < 0 )
		return(0u);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	address.sin_port = htons(0u);
	if ( bind(fd,(const struct sockaddr *)&address,(socklen_t)sizeof(address)) != 0 )
	{
		close(fd);
		return(0u);
	}
	address_bytes = (socklen_t)sizeof(address);
	if ( getsockname(fd,(struct sockaddr *)&address,&address_bytes) != 0 )
	{
		close(fd);
		return(0u);
	}
	port = (uint32_t)ntohs(address.sin_port);
	close(fd);
	return(port);
}

static uint32_t TestRemoteWriteConfiguration(const char *path, uint32_t bridge_port)
{
	FILE *file;
	file = fopen(path,"w");
	if ( file == 0 )
		return(1u);
	fprintf(file,"{\n  \"schema_version\": 3,\n  \"model_revision\": \"%s\",\n  \"stage_pack_path\": \"tests/fixtures/qwen38_27b-stage.qwen38_27bsp\",\n  \"max_sequence_positions\": 4096,\n  \"draft_bridge_host\": \"127.0.0.1\",\n  \"draft_bridge_port\": %u\n}\n",QWEN38_27B_MODEL_REVISION,bridge_port);
	fclose(file);
	return(0u);
}

typedef struct TestRemoteState
{
	uint32_t completion_count;
	void *execution_stream;
	SparkModelServingCompletion completion;
} TestRemoteState;

static void TestRemoteCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	TestRemoteState *state;
	state = (TestRemoteState *)completion_context;
	state->completion = *completion;
	state->completion_count++;
}

static void TestRemoteConfiguration(
	SparkModelServingAdapterConfiguration *configuration,
	const char *config_path,
	const char *runtime_root,
	TestRemoteState *test_state)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration->rank_index = 0u;
	configuration->stage_index = 0u;
	configuration->runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	configuration->runtime_limits.max_inflight_submission_count = 2u;
	configuration->runtime_limits.max_active_sequence_count = 2u;
	configuration->runtime_limits.max_input_row_count = 4u;
	configuration->runtime_limits.resident_sequence_capacity = 2u;
	configuration->runtime_limits.kv_logical_page_capacity = 64u;
	configuration->runtime_limits.kv_physical_page_capacity = 64u;
	configuration->runtime_root = runtime_root;
	configuration->node_id = "spark-test";
	configuration->node_target = "cuda.sm121.qwen38_27b.resident_decode_stage.bf16";
	configuration->adapter_configuration_path = config_path;
	configuration->driver_shared_object_path = TEST_QWEN38_27B_REMOTE_SPEC_DRIVER_PATH;
	configuration->driver_program_name = "resident_decode";
	configuration->execution_stream = test_state->execution_stream;
	configuration->completion_function = TestRemoteCompletion;
	configuration->completion_context = test_state;
}

static uint32_t TestRemoteCheckCompletion(
	const TestRemoteState *test_state,
	const uint32_t *expected_ids,
	uint32_t expected_count,
	uint32_t expected_accepted)
{
	uint32_t step;
	TEST_REMOTE_CHECK((test_state->completion.completion_flags & SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS) != 0u);
	TEST_REMOTE_CHECK((test_state->completion.completion_flags & SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION) != 0u);
	TEST_REMOTE_CHECK(test_state->completion.tokens_per_sequence == expected_count);
	TEST_REMOTE_CHECK(test_state->completion.token_count == expected_count);
	for (step=0u; step<expected_count; step++)
		TEST_REMOTE_CHECK(test_state->completion.token_ids[step] == expected_ids[step]);
	TEST_REMOTE_CHECK(test_state->completion.accepted_token_count == expected_accepted);
	TEST_REMOTE_CHECK(test_state->completion.model_extension_kind == 0x5136u);
	TEST_REMOTE_CHECK(test_state->completion.model_extension_bytes == 8u);
	TEST_REMOTE_CHECK(((const uint32_t *)test_state->completion.model_extension)[0] == 0u);
	TEST_REMOTE_CHECK(((const uint32_t *)test_state->completion.model_extension)[1] == 0u);
	return(0u);
}

static uint32_t TestRemoteCheckCommitted(
	const TestRemoteStub *stub,
	uint32_t exchange,
	uint64_t sequence_id,
	uint64_t generation,
	const uint32_t *expected_ids,
	uint32_t expected_count)
{
	uint32_t index;
	TEST_REMOTE_CHECK(stub->last_mask[exchange] == SPARK_SPECULATION_SEAM_SOURCE_NGRAM);
	TEST_REMOTE_CHECK(stub->last_sequence_id[exchange] == sequence_id);
	TEST_REMOTE_CHECK(stub->last_generation[exchange] == generation);
	TEST_REMOTE_CHECK(stub->last_committed_count[exchange] == expected_count);
	for (index=0u; index<expected_count; index++)
		TEST_REMOTE_CHECK(stub->last_committed[exchange][index] == expected_ids[index]);
	return(0u);
}

static uint32_t TestRemoteCheckVerifyCapture(const char *capture_path)
{
	static const char *const expected_lines[TEST_REMOTE_EXCHANGE_COUNT] =
	{
		"3 5000 6001 6002",
		"3 5000 6011 6012",
		"3 5000 6021 6022",
		"3 5000 6031 6032"
	};
	char line[TEST_REMOTE_CAPTURE_BYTES];
	uint32_t line_index;
	FILE *file;
	file = fopen(capture_path,"r");
	TEST_REMOTE_CHECK(file != 0);
	line_index = 0u;
	while ( fgets(line,sizeof(line),file) != 0 && line_index < TEST_REMOTE_EXCHANGE_COUNT )
	{
		size_t length = strlen(line);
		if ( length != 0u && line[length - 1u] == '\n' )
			line[length - 1u] = '\0';
		TEST_REMOTE_CHECK(strcmp(line,expected_lines[line_index]) == 0);
		line_index++;
	}
	fclose(file);
	TEST_REMOTE_CHECK(line_index == TEST_REMOTE_EXCHANGE_COUNT);
	return(0u);
}

int main(void)
{
	SparkModelServingAdapterDynamicLibrary library;
	SparkModelServingAdapterConfiguration configuration;
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	TestRemoteState test_state;
	TestRemoteStub stub;
	void *adapter_state;
	uint32_t token_ids[4],row_lane_indices[4];
	uint64_t row_positions[4],row_sequence_ids[4];
	char runtime_root[4096];
	char config_path[64];
	char dead_config_path[64];
	char capture_path[64];
	uint32_t closed_port;
	snprintf(config_path,sizeof(config_path),"/tmp/qwen38_27b_remote_spec_%d.json",(int)getpid());
	snprintf(dead_config_path,sizeof(dead_config_path),"/tmp/qwen38_27b_remote_spec_dead_%d.json",(int)getpid());
	snprintf(capture_path,sizeof(capture_path),"/tmp/qwen38_27b_remote_verify_%d.txt",(int)getpid());
	setenv("SPARK_QWEN38_27B_SPECULATORS","0x8",1);
	setenv("SPARK_QWEN38_27B_TEST_VERIFY_CAPTURE",capture_path,1);
	unlink(capture_path);
	if ( TestRemoteStubStart(&stub) != 0u )
	{
		fprintf(stderr,"CHECK-FAIL stub start\n");
		return(1);
	}
	if ( TestRemoteWriteConfiguration(config_path,stub.port) != 0u )
	{
		fprintf(stderr,"CHECK-FAIL config write\n");
		return(1);
	}
	memset(&test_state,0,sizeof(test_state));
	if ( cudaStreamCreate((cudaStream_t *)&test_state.execution_stream) != cudaSuccess )
		return(1);
	if ( SparkModelServingAdapterLoadInterfaceFromSharedObject(TEST_QWEN38_27B_REMOTE_SPEC_ADAPTER_PATH,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE,&library) != SPARK_STATUS_OK )
		return(1);
	if ( getcwd(runtime_root,sizeof(runtime_root)) == 0 )
		return(1);
	TestRemoteConfiguration(&configuration,config_path,runtime_root,&test_state);
	adapter_state = 0;
	if ( library.adapter_interface.initialize(&configuration,&adapter_state) != SPARK_STATUS_OK || adapter_state == 0 )
	{
		fprintf(stderr,"CHECK-FAIL adapter init with bridge\n");
		return(1);
	}

	memset(&lane,0,sizeof(lane));
	lane.request_id = 950u;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.sequence_id = 200u;
	lane.resident_sequence_slot = 0u;
	lane.context_token_count = 0u;
	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	token_ids[0] = 31u;
	token_ids[1] = 32u;
	token_ids[2] = 33u;
	token_ids[3] = 34u;
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 0u;
	row_lane_indices[2] = 0u;
	row_lane_indices[3] = 0u;
	row_positions[0] = 0u;
	row_positions[1] = 1u;
	row_positions[2] = 2u;
	row_positions[3] = 3u;
	row_sequence_ids[0] = 200u;
	row_sequence_ids[1] = 200u;
	row_sequence_ids[2] = 200u;
	row_sequence_ids[3] = 200u;
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.tokens_per_sequence = 1u;
	submission.submission_id = 80u;
	submission.request_id = 95u;
	submission.sequence_id = 200u;
	submission.control_generation = 1u;
	submission.transaction_id = 1080u;
	submission.dispatch_generation = 2080u;
	submission.request_generation = 1u;
	submission.step_generation = 3080u;
	submission.residency.word0 = 80u;
	submission.residency.word1 = 180u;
	submission.residency.generation = 280u;
	submission.residency.owner = 13u;
	submission.active_sequence_count = 1u;
	submission.new_token_count = 4u;
	submission.lane_count = 1u;
	submission.row_count = 4u;
	submission.token_count = 4u;
	submission.lanes = &lane;
	submission.token_ids = token_ids;
	submission.row_lane_indices = row_lane_indices;
	submission.row_positions = row_positions;
	submission.row_sequence_ids = row_sequence_ids;
	if ( library.adapter_interface.submit(adapter_state,&submission) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"CHECK-FAIL prefill\n");
		return(1);
	}
	if ( test_state.completion_count != 1u || test_state.completion.token_ids[0] != 4242u )
	{
		fprintf(stderr,"CHECK-FAIL prefill completion\n");
		return(1);
	}

	lane.context_token_count = 4u;
	token_ids[0] = 4242u;
	row_positions[0] = 4u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.submission_id = 81u;
	submission.new_token_count = 1u;
	submission.row_count = 1u;
	submission.token_count = 1u;
	if ( library.adapter_interface.submit(adapter_state,&submission) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"CHECK-FAIL decode step 1\n");
		return(1);
	}
	{
		const uint32_t expected[4] = { 5000u, 6001u, 9998u, 8888u };
		if ( TestRemoteCheckCompletion(&test_state,expected,4u,1u) != 0u )
			return(1);
	}
	{
		const uint32_t committed[5] = { 31u, 32u, 33u, 34u, 4242u };
		if ( TestRemoteCheckCommitted(&stub,0u,200u,0u,committed,5u) != 0u )
			return(1);
	}

	lane.context_token_count = 8u;
	token_ids[0] = 8888u;
	row_positions[0] = 8u;
	submission.submission_id = 82u;
	if ( library.adapter_interface.submit(adapter_state,&submission) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"CHECK-FAIL decode step 2\n");
		return(1);
	}
	{
		const uint32_t expected[5] = { 5000u, 6011u, 6012u, 9999u, 8888u };
		if ( TestRemoteCheckCompletion(&test_state,expected,5u,2u) != 0u )
			return(1);
	}
	{
		const uint32_t committed[9] = { 31u, 32u, 33u, 34u, 4242u, 5000u, 6001u, 9998u, 8888u };
		if ( TestRemoteCheckCommitted(&stub,1u,200u,1u,committed,9u) != 0u )
			return(1);
	}

	lane.flags = 0u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	submission.tokens_per_sequence = 0u;
	submission.new_token_count = 0u;
	submission.row_count = 0u;
	submission.token_count = 0u;
	submission.token_ids = 0;
	submission.row_lane_indices = 0;
	submission.row_positions = 0;
	submission.row_sequence_ids = 0;
	submission.submission_id = 83u;
	if ( library.adapter_interface.submit(adapter_state,&submission) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"CHECK-FAIL release 200\n");
		return(1);
	}

	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	lane.sequence_id = 300u;
	lane.context_token_count = 1u;
	token_ids[0] = 11u;
	row_positions[0] = 0u;
	row_sequence_ids[0] = 300u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.tokens_per_sequence = 1u;
	submission.new_token_count = 1u;
	submission.row_count = 1u;
	submission.token_count = 1u;
	submission.token_ids = token_ids;
	submission.row_lane_indices = row_lane_indices;
	submission.row_positions = row_positions;
	submission.row_sequence_ids = row_sequence_ids;
	submission.sequence_id = 300u;
	submission.submission_id = 84u;
	if ( library.adapter_interface.submit(adapter_state,&submission) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"CHECK-FAIL decode 300\n");
		return(1);
	}
	{
		const uint32_t expected[5] = { 5000u, 6021u, 6022u, 9999u, 8888u };
		if ( TestRemoteCheckCompletion(&test_state,expected,5u,2u) != 0u )
			return(1);
	}
	{
		const uint32_t committed[1] = { 11u };
		if ( TestRemoteCheckCommitted(&stub,2u,300u,2u,committed,1u) != 0u )
			return(1);
	}

	lane.flags = 0u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	submission.tokens_per_sequence = 0u;
	submission.new_token_count = 0u;
	submission.row_count = 0u;
	submission.token_count = 0u;
	submission.token_ids = 0;
	submission.row_lane_indices = 0;
	submission.row_positions = 0;
	submission.row_sequence_ids = 0;
	submission.submission_id = 85u;
	if ( library.adapter_interface.submit(adapter_state,&submission) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"CHECK-FAIL release 300\n");
		return(1);
	}

	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	lane.sequence_id = 400u;
	token_ids[0] = 12u;
	row_sequence_ids[0] = 400u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.tokens_per_sequence = 1u;
	submission.new_token_count = 1u;
	submission.row_count = 1u;
	submission.token_count = 1u;
	submission.token_ids = token_ids;
	submission.row_lane_indices = row_lane_indices;
	submission.row_positions = row_positions;
	submission.row_sequence_ids = row_sequence_ids;
	submission.sequence_id = 400u;
	submission.submission_id = 86u;
	if ( library.adapter_interface.submit(adapter_state,&submission) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"CHECK-FAIL decode 400 (engine lane reuse after release)\n");
		return(1);
	}
	{
		const uint32_t expected[5] = { 5000u, 6031u, 6032u, 9999u, 8888u };
		if ( TestRemoteCheckCompletion(&test_state,expected,5u,2u) != 0u )
			return(1);
	}
	{
		const uint32_t committed[1] = { 12u };
		if ( TestRemoteCheckCommitted(&stub,3u,400u,3u,committed,1u) != 0u )
			return(1);
	}

	if ( library.adapter_interface.quiesce(adapter_state,UINT64_MAX) != SPARK_STATUS_OK )
		return(1);
	library.adapter_interface.destroy(adapter_state);
	TestRemoteStubStop(&stub);
	if ( stub.failed != 0u || stub.exchanges_completed != TEST_REMOTE_EXCHANGE_COUNT )
	{
		fprintf(stderr,"CHECK-FAIL stub failed=%u exchanges=%u\n",stub.failed,stub.exchanges_completed);
		return(1);
	}
	if ( TestRemoteCheckVerifyCapture(capture_path) != 0u )
		return(1);

	adapter_state = 0;
	TestRemoteConfiguration(&configuration,TEST_QWEN38_27B_REMOTE_SPEC_CONFIG_PATH,runtime_root,&test_state);
	if ( library.adapter_interface.initialize(&configuration,&adapter_state) != SPARK_STATUS_SCHEMA_ERROR || adapter_state != 0 )
	{
		fprintf(stderr,"CHECK-FAIL remote without bridge config must fail SCHEMA_ERROR\n");
		return(1);
	}

	closed_port = TestRemoteClosedPort();
	if ( closed_port == 0u || TestRemoteWriteConfiguration(dead_config_path,closed_port) != 0u )
		return(1);
	adapter_state = 0;
	TestRemoteConfiguration(&configuration,dead_config_path,runtime_root,&test_state);
	if ( library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK || adapter_state != 0 )
	{
		fprintf(stderr,"CHECK-FAIL init must fail loudly when the bridge is unreachable\n");
		return(1);
	}

	SparkModelServingAdapterUnloadInterface(&library);
	if ( cudaStreamDestroy((cudaStream_t)test_state.execution_stream) != cudaSuccess )
		return(1);
	unlink(config_path);
	unlink(dead_config_path);
	unlink(capture_path);
	printf("PASS qwen38_27b remote speculation\n");
	return(0);
}
