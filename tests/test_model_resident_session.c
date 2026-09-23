#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime_api.h>

static uint32_t TestStreamCalls;
static uint32_t TestStreamFailure;
static cudaError_t TestCreateStream(cudaStream_t *stream,unsigned int flags)
{
	TestStreamCalls++;
	if ( TestStreamCalls == TestStreamFailure )
		return(cudaErrorMemoryAllocation);
	return(cudaStreamCreateWithFlags(stream,flags));
}

#define main SparkModelResidentdProgramMain
#define cudaStreamCreateWithFlags TestCreateStream
#include "../node/model_residentd.c"
#undef cudaStreamCreateWithFlags
#undef main

typedef struct TestSession
{
	SparkModelResidentdRuntime runtime;
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelResidentdSequenceSlot slot;
	SparkModelResidentdRoute route;
	SparkModelServingLane lane;
	SparkModelResidentdOutput outputs[4];
	uint8_t messages[4][65536];
	uint32_t reset_calls;
	SparkStatus reset_status;
	SparkStatus progress_status;
} TestSession;

static SparkStatus TestReset(void *context,uint64_t generation)
{
	TestSession *test = context;
	assert(generation == test->runtime.client.pending_client_reset);
	test->reset_calls++;
	return(test->reset_status);
}

static SparkStatus TestProgress(void *context,uint32_t maximum_steps)
{
	TestSession *test = context;
	assert(maximum_steps != 0u);
	SparkModelResidentdStop = 1;
	return(test->progress_status);
}

static void TestInitialize(TestSession *test)
{
	memset(test,0,sizeof(*test));
	SparkModelResidentdRuntime *runtime = &test->runtime;
	SparkModelServingAdapterDescriptor *descriptor = &test->descriptor;
	assert(pthread_mutex_init(&runtime->mutex,0) == 0);
	atomic_init(&runtime->failed_status,SPARK_STATUS_OK);
	runtime->client.fd = runtime->candidate_fd = runtime->listen_fd = -1;
	runtime->wake_read_fd = runtime->wake_write_fd = -1;
	runtime->client.generation = 7u;
	runtime->client.session_epoch = 100u;
	runtime->client.hello_complete = 1u;
	runtime->client.last_message_id = 42u;
	runtime->client.output = test->outputs;
	runtime->client.output_capacity = 4u;
	runtime->client.output_message_capacity = sizeof(test->messages[0]);
	for (uint32_t index=0u; index<4u; index++)
		test->outputs[index].message = test->messages[index];
	runtime->sequence_slots = &test->slot;
	runtime->runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	runtime->runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	runtime->runtime_limits.max_inflight_submission_count = 1u;
	runtime->runtime_limits.max_active_sequence_count = 1u;
	runtime->runtime_limits.max_input_row_count = 1u;
	runtime->runtime_limits.resident_sequence_capacity = 1u;
	runtime->runtime_limits.kv_logical_page_capacity = 1u;
	runtime->runtime_limits.kv_physical_page_capacity = 1u;
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->cache_block_token_count = 4u;
	descriptor->stage_count = 1u;
	descriptor->layer_count = 1u;
	descriptor->stage_layer_counts[0] = 1u;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = 32u;
	descriptor->boundary_element_bytes = 2u;
	descriptor->linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->expert_weight_codec = SPARK_WEIGHT_CODEC_INT8;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->max_inflight_submission_count = 1u;
	descriptor->max_active_sequence_count = 1u;
	descriptor->max_input_row_count = 1u;
	descriptor->max_resident_sequence_count = 1u;
	descriptor->max_output_token_count = 1u;
	descriptor->adapter_id = "test.adapter";
	descriptor->model_id = "test/model";
	descriptor->model_revision = "revision-1";
	descriptor->driver_program_name = "decode";
	descriptor->artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
	runtime->adapter_library.adapter_interface.descriptor = descriptor;
	runtime->adapter_library.adapter_interface.reset = TestReset;
	runtime->adapter_library.adapter_interface.progress = TestProgress;
	runtime->adapter_state = test;
}

static int TestAdopt(TestSession *test)
{
	int sockets[2];
	SparkModelResidentIpcHello hello;
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
	assert(SparkModelResidentdSetNonblocking(sockets[0]) == 0);
	test->runtime.candidate_fd = sockets[0];
	assert(SparkModelResidentIpcInitializeHello(&hello,1u,0u,0u,101u,
		&test->descriptor) == SPARK_STATUS_OK);
	assert(SparkModelResidentdAdoptCandidate(&test->runtime,&hello) == SPARK_STATUS_OK);
	return(sockets[1]);
}

static void TestReadHello(TestSession *test,int peer)
{
	SparkModelResidentIpcHelloAck ack;
	assert(SparkModelResidentdWriteClient(&test->runtime) == SPARK_STATUS_OK);
	assert(read(peer,&ack,sizeof(ack)) == sizeof(ack));
	assert(SparkModelResidentIpcValidateHelloAck(&ack,sizeof(ack),1u,0u,0u,
		101u,&test->descriptor,&test->runtime.runtime_limits) == SPARK_STATUS_OK);
	assert(ack.status == SPARK_STATUS_OK && ack.client_generation == 8u);
}

static void TestPartialReplyIsNotReplayed(void)
{
	TestSession test;
	int old[2],size=1024;
	TestInitialize(&test);
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,old) == 0);
	assert(SparkModelResidentdSetNonblocking(old[0]) == 0);
	assert(setsockopt(old[0],SOL_SOCKET,SO_SNDBUF,&size,sizeof(size)) == 0);
	test.runtime.client.fd = old[0];
	memset(test.messages[1],0x5a,sizeof(test.messages[1]));
	assert(SparkModelResidentdQueueRawLocked(&test.runtime,test.messages[1],
		sizeof(test.messages[1])) == SPARK_STATUS_OK);
	assert(SparkModelResidentdWriteClient(&test.runtime) == SPARK_STATUS_OK);
	assert(test.outputs[0].sent_bytes != 0u);
	assert(test.outputs[0].sent_bytes < test.outputs[0].message_bytes);
	SparkModelResidentdDetachClient(&test.runtime);
	assert(test.runtime.client.output_count == 0u);
	int peer = TestAdopt(&test);
	assert(test.runtime.client.output_count == 0u);
	assert(test.runtime.client.hello_complete == 0u);
	assert(SparkModelResidentdProgressReset(&test.runtime) == SPARK_STATUS_OK);
	assert(test.reset_calls == 1u);
	TestReadHello(&test,peer);
	assert(test.runtime.client.last_message_id == 1u);
	close(old[1]);
	close(peer);
	SparkModelResidentdCloseClient(&test.runtime);
	assert(pthread_mutex_destroy(&test.runtime.mutex) == 0);
}

static void TestResetWaitsForOwnedRoute(void)
{
	TestSession test;
	TestInitialize(&test);
	test.runtime.routes = &test.route;
	test.runtime.route_capacity = 1u;
	test.route.active = 1u;
	test.route.active_since_ns = 1u;
	test.route.client_generation = 7u;
	test.route.state = SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER;
	test.route.resident_slots_claimed = 1u;
	test.route.submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	test.route.submission.active_sequence_count = 1u;
	test.route.submission.lanes = &test.lane;
	test.lane.request_id = 20u;
	test.lane.request_generation = 1u;
	test.lane.sequence_id = 30u;
	test.slot.active_owner = 1u;
	int peer = TestAdopt(&test);
	assert(test.route.active != 0u && test.route.abandoned != 0u);
	assert(test.slot.active_owner == 1u && test.route.resident_slots_claimed == 1u);
	SparkModelResidentdReportStuckRoutes(&test.runtime);
	assert(test.route.active != 0u && test.slot.active_owner == 1u);
	assert(SparkModelResidentdProgressReset(&test.runtime) == SPARK_STATUS_OK);
	assert(test.reset_calls == 0u && test.runtime.client.output_count == 0u);
	SparkModelResidentIpcHeader submit = {0};
	submit.kind = SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT;
	submit.message_id = 2u;
	assert(SparkModelResidentdProcessMessage(&test.runtime,&submit,sizeof(submit)) == SPARK_STATUS_SCHEMA_ERROR);
	test.route.state = SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION;
	assert(SparkModelResidentdFinishRoute(&test.runtime,&test.route) == SPARK_STATUS_OK);
	assert(test.route.active == 0u && test.slot.active_owner == 0u);
	test.reset_status = SPARK_STATUS_BUSY;
	assert(SparkModelResidentdProgressReset(&test.runtime) == SPARK_STATUS_OK);
	assert(test.runtime.client.hello_complete == 0u && test.runtime.client.output_count == 0u);
	test.reset_status = SPARK_STATUS_IO_ERROR;
	assert(SparkModelResidentdProgressReset(&test.runtime) == SPARK_STATUS_IO_ERROR);
	assert(test.runtime.client.pending_client_reset != 0u);
	assert(test.runtime.client.hello_complete == 0u && test.runtime.client.output_count == 0u);
	assert(atomic_load(&test.runtime.failed_status) == SPARK_STATUS_IO_ERROR);
	close(peer);
	SparkModelResidentdCloseClient(&test.runtime);
	assert(pthread_mutex_destroy(&test.runtime.mutex) == 0);
}

static void TestFatalProgressExits(void)
{
	TestSession test;
	TestInitialize(&test);
	test.progress_status = SPARK_STATUS_IO_ERROR;
	SparkModelResidentdStop = 0;
	assert(SparkModelResidentdRun(&test.runtime) == SPARK_STATUS_IO_ERROR);
	assert(pthread_mutex_destroy(&test.runtime.mutex) == 0);
	TestInitialize(&test);
	test.progress_status = SPARK_STATUS_BUSY;
	SparkModelResidentdStop = 0;
	assert(SparkModelResidentdOpenWakePipe(&test.runtime) == SPARK_STATUS_OK);
	SparkModelResidentdWake(&test.runtime);
	assert(SparkModelResidentdRun(&test.runtime) == SPARK_STATUS_OK);
	close(test.runtime.wake_read_fd);
	close(test.runtime.wake_write_fd);
	assert(pthread_mutex_destroy(&test.runtime.mutex) == 0);
	SparkModelResidentdStop = 0;
}

static void TestDisconnectDoesNotRaiseSigpipe(void)
{
	TestSession test;
	int sockets[2];
	uint8_t message = 1u;
	TestInitialize(&test);
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
	assert(SparkModelResidentConfigureSocket(sockets[0]) == SPARK_STATUS_OK);
	test.runtime.client.fd = sockets[0];
	close(sockets[1]);
	assert(SparkModelResidentdQueueRawLocked(&test.runtime,&message,sizeof(message)) == SPARK_STATUS_OK);
	assert(SparkModelResidentdWriteClient(&test.runtime) == SPARK_STATUS_IO_ERROR);
	SparkModelResidentdDetachClient(&test.runtime);
	assert(pthread_mutex_destroy(&test.runtime.mutex) == 0);
}

static void TestTcpOptions(void)
{
	int fd = socket(AF_INET,SOCK_STREAM,0);
	assert(fd >= 0);
	assert(SparkModelResidentConfigureTcp(fd) == SPARK_STATUS_OK);
	assert(SparkModelResidentConfigureTcp(-1) == SPARK_STATUS_IO_ERROR);
	close(fd);
}

static void TestCudaStartupFailureNamesStream(void)
{
	for (uint32_t failing=1u; failing<=2u; failing++)
	{
		SparkModelResidentdRuntime runtime;
		FILE *log = tmpfile();
		int saved = dup(STDERR_FILENO);
		char text[1024] = {0};
		assert(log != 0 && saved >= 0);
		memset(&runtime,0,sizeof(runtime));
		runtime.rank_plan.rank_index = 4u;
		runtime.rank_plan.stage_index = 4u;
		TestStreamCalls = 0u;
		TestStreamFailure = failing;
		assert(dup2(fileno(log),STDERR_FILENO) >= 0);
		assert(SparkModelResidentdAllocateCuda(&runtime,SPARK_MODEL_RESIDENTD_MEMORY_MAPPED_HOST) == SPARK_STATUS_INTERNAL_ERROR);
		fflush(stderr);
		assert(dup2(saved,STDERR_FILENO) >= 0);
		close(saved);
		rewind(log);
		assert(fread(text,1,sizeof(text)-1u,log) > 0u);
		fclose(log);
		assert(strstr(text,failing == 1u ? "stream=execution" : "stream=transport") != 0);
		assert(strstr(text,"rank=4 stage=4 cuda_error=2 detail=") != 0);
		assert(TestStreamCalls == failing);
		assert(runtime.transport_stream == 0);
		if ( runtime.execution_stream != 0 )
			assert(cudaStreamDestroy(runtime.execution_stream) == cudaSuccess);
	}
	TestStreamFailure = 0u;
}

int main(void)
{
	TestCudaStartupFailureNamesStream();
	TestPartialReplyIsNotReplayed();
	TestResetWaitsForOwnedRoute();
	TestFatalProgressExits();
	TestTcpOptions();
	TestDisconnectDoesNotRaiseSigpipe();
	return(0);
}
