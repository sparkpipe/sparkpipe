#include <assert.h>
#include <errno.h>
#include <sys/mman.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_tp_mesh_round_control.h"

#define _GNU_SOURCE 1
#include <cuda_runtime.h>

static uint32_t test_failures;
static uint32_t test_checks;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)



extern uint32_t cuda_stub_mesh_publish_calls;
extern uint32_t cuda_stub_mesh_publish_null_seq_cell;
extern uint32_t cuda_stub_mesh_publish_null_epoch_cell;

extern uint32_t cuda_stub_mesh_hardware_calls;
extern int cuda_stub_mesh_hardware_prepare_result;
extern int cuda_stub_mesh_hardware_launch_result;
extern void *cuda_stub_mesh_hardware_alias;
extern void *cuda_stub_mesh_hardware_band;
extern void *cuda_stub_mesh_hardware_gate;
extern void *cuda_stub_mesh_hardware_control;
extern cudaError_t cuda_stub_stream_query_result;
extern uint64_t cuda_stub_mesh_hardware_elements;
extern uint32_t cuda_stub_mesh_hardware_operation;
extern uint32_t cuda_stub_mesh_hardware_logical_rows;
extern uint32_t cuda_stub_mesh_hardware_slice_routes;

static uint64_t mock_client_alive = 1u;
static uint32_t mock_server;
static uint64_t mock_lane_mask[16];
static uint32_t mock_clients_live;
static SparkStatus mock_lane_status = SPARK_STATUS_OK;

SparkStatus SparkWeightdClientConnect(const char *socket_path, SparkWeightdClient **client, SparkWeightdHelloResult *hello_out)
{
	(void)socket_path; (void)hello_out;
	*client = (SparkWeightdClient *)calloc(1u, 64u+sizeof(SparkWeightdMeshTopology));
	((uint64_t *)*client)[2] = SPARK_WEIGHTD_LANE_NONE;
	((uint64_t *)*client)[4] = mock_server;
	mock_clients_live++;
	return(SPARK_STATUS_OK);
}

void SparkWeightdClientClose(SparkWeightdClient *client)
{
    uint64_t *state = (uint64_t *)client;
    assert(state[3] == 0u);
    if (state[2] < SPARK_WEIGHTD_MESH_MAX_LANES)
        mock_lane_mask[state[4]] &= ~(UINT64_C(1) << state[2]);
    mock_clients_live--;
	free(client);
}

static SparkStatus mock_mesh_map_status = SPARK_STATUS_OK;
static void *mock_owned_mapping;

SparkStatus SparkWeightdClientMeshMap(SparkWeightdClient *client,void **mapping,uint64_t timeout)
{
    (void)client; (void)timeout;
    *mapping = 0;
    if (mock_mesh_map_status != SPARK_STATUS_OK) return mock_mesh_map_status;
    *mapping = mmap(0,SPARK_WEIGHTD_MESH_REGION_BYTES,PROT_READ | PROT_WRITE,MAP_PRIVATE | MAP_ANONYMOUS,-1,0);
    assert(*mapping != MAP_FAILED);
    mock_owned_mapping = *mapping;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientLaneAcquire(SparkWeightdClient *client,
    uint32_t requested,const SparkWeightdMeshTopology *topology,uint32_t *out,uint64_t timeout)
{
    uint64_t *state = (uint64_t *)client;
    (void)timeout;
    if (mock_lane_status != SPARK_STATUS_OK) return mock_lane_status;
    if (state[2] != SPARK_WEIGHTD_LANE_NONE) return SPARK_STATUS_DUPLICATE;
    for (uint32_t lane=0u; lane<SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
        if ((requested == SPARK_WEIGHTD_LANE_NONE || requested == lane) &&
            (mock_lane_mask[state[4]] & (UINT64_C(1) << lane)) == 0u)
        {
            mock_lane_mask[state[4]] |= UINT64_C(1) << lane;
            state[2] = lane;
            if (topology != 0) memcpy(state+8,topology,sizeof(*topology));
            *out = lane;
            return SPARK_STATUS_OK;
        }
    return SPARK_STATUS_NO_LANE;
}

SparkStatus SparkWeightdClientLaneBind(SparkWeightdClient *owner,
    SparkWeightdClient *peer,uint32_t band,const SparkWeightdMeshTopology *topology,uint32_t *out)
{
    uint64_t *state = (uint64_t *)owner;
    if (topology == 0 || memcmp(state+8,topology,sizeof(*topology)) != 0 ||
        state[4] != ((const uint64_t *)peer)[4] || state[2] >= SPARK_WEIGHTD_MESH_MAX_LANES || band >= 2u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ((state[3] & (1u << band)) != 0u) return SPARK_STATUS_DUPLICATE;
    state[3] |= 1u << band;
    *out = (uint32_t)state[2];
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientLaneUnbind(SparkWeightdClient *owner,uint32_t band)
{
    uint64_t *state = (uint64_t *)owner;
    assert(band < 2u && (state[3] & (1u << band)) != 0u);
    state[3] &= ~(1u << band);
    return SPARK_STATUS_OK;
}

uint32_t SparkWeightdClientAlive(const SparkWeightdClient *client)
{
	(void)client;
	return((uint32_t)mock_client_alive);
}

SparkStatus SparkWeightdClientMeshActivity(SparkWeightdClient *client,
    uint64_t generation,uint32_t active,uint64_t timeout_nanoseconds)
{
    uint64_t *state = (uint64_t *)client;
    (void)timeout_nanoseconds;
    if ( active != 0u )
    {
        if ( state[1] != 0u || generation <= state[0] )
            return SPARK_STATUS_INVALID_ARGUMENT;
        state[0] = generation;
        state[1] = 1u;
    }
    else
    {
        if ( state[1] == 0u || generation != state[0] )
            return SPARK_STATUS_INVALID_ARGUMENT;
        state[1] = 0u;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientMeshBroadcast(SparkWeightdClient *client, uint32_t peer_mask, uint64_t source_offset, uint64_t remote_offset, uint32_t length, uint64_t seq_value, uint64_t seq_remote_offset, uint64_t timeout_nanoseconds)
{
	(void)client; (void)peer_mask; (void)source_offset; (void)remote_offset;
	(void)length; (void)seq_value; (void)seq_remote_offset; (void)timeout_nanoseconds;
	return(SPARK_STATUS_OK);
}

static uint32_t mock_combine_calls;

static SparkStatus TestCombineFusedBf16(void *combine_context, void *destination_device, const void *const *source_devices, uint32_t source_count, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context; (void)destination_device; (void)source_devices;
	(void)source_count; (void)active_sequence_count; (void)hidden_dimension; (void)cuda_stream;
	mock_combine_calls++;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)source_device;
	return(TestCombineFusedBf16(combine_context,destination_device,0,0,active_sequence_count,hidden_dimension,cuda_stream));
}

static SparkStatus TestCombineU64Max(void *combine_context, uint64_t *destination_device, const uint64_t *source_device, uint32_t element_count, void *cuda_stream)
{
	(void)combine_context; (void)destination_device; (void)source_device; (void)element_count; (void)cuda_stream;
	mock_combine_calls++;
	return(SPARK_STATUS_OK);
}

static void TestComplete(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	(void)context; (void)completion;
}

typedef struct PeerWaitArgs
{
	SparkTpDeviceCollective *collective;
	uint64_t request_id;
	SparkStatus status;
	uint64_t waited_ns;
} PeerWaitArgs;

static uint64_t TestNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0ull);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static void *PeerWaitMain(void *data)
{
	PeerWaitArgs *args = (PeerWaitArgs *)data;
	uint64_t start = TestNowNs();
	args->status = SparkTpDeviceCollectiveChainKey(args->collective,args->request_id);
	args->waited_ns = TestNowNs() - start;
	return(0);
}

typedef struct TestEpochPublish
{
    volatile uint64_t *cell;
    uint64_t epoch;
} TestEpochPublish;

static void *TestPublishNewEpoch(void *context)
{
    TestEpochPublish *publish = context;
    usleep(100000);
    __atomic_store_n(publish->cell,publish->epoch,__ATOMIC_RELEASE);
    return 0;
}

static void TestRestartEpoch(SparkTpDeviceCollectiveConfig config,void *mesh)
{
    setenv("SPARK_TP_WAIT_MODE","hardware",1);
    config.tp_rank = 1u;
    config.operation_timeout_milli = 1000u;
    mock_server = 1u;
    for (uint32_t delayed = 0u; delayed < 2u; delayed++)
    {
        SparkTpDeviceCollective collective = {0};
        SparkTpDeviceCollectiveSubmission submission = {0};
        uint16_t values[128] = {0};
        pthread_t thread;
        memset(mesh,0,SPARK_WEIGHTD_MESH_REGION_BYTES);
        volatile uint64_t *cell = (volatile uint64_t *)((uint8_t *)mesh + SPARK_WEIGHTD_MESH_DOORBELL_OFFSET + SPARK_WEIGHTD_MESH_DOORBELL_CELL_BASE * SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES);
        cell[0] = 1024u;
        cell[1] = 0x10u;
        cell[2] = 2u;
        *(uint64_t *)((uint8_t *)mesh + SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u,1u)) = 1268u;
        CHECK(SparkTpDeviceCollectiveCreate(&config,&collective) == SPARK_STATUS_OK,"restart peer create");
        CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,mesh,1u,64u,0u,0) == SPARK_STATUS_OK,"restart maps old epoch before all-rank ready");
        TestEpochPublish publish = {cell,2048u};
        if (delayed) pthread_create(&thread,0,TestPublishNewEpoch,&publish);
        else cell[0] = 2048u;
        submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        submission.descriptor_bytes = sizeof(submission);
        submission.active_sequence_count = submission.logical_sequence_count = 1u;
        submission.local_device = submission.full_device = values;
        submission.cuda_stream = (void *)1;
        submission.completion_function = TestComplete;
        cuda_stub_mesh_hardware_launch_result = cudaErrorUnknown;
        uint64_t start = TestNowNs();
        CHECK(SparkTpDeviceCollectiveEnqueue(&collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) == SPARK_STATUS_IO_ERROR,"restart reaches test hardware launch");
        uint64_t elapsed = TestNowNs() - start;
        CHECK(!delayed || elapsed >= 50000000u,"restart waits for the new root epoch rather than using the stale nonzero cell");
        if (delayed) pthread_join(thread,0);
        SparkTpMeshRoundControl *control = cuda_stub_mesh_hardware_control;
        CHECK(control != 0 && control->seq >= 2048u,"restart dispatch uses current epoch for either rank order");
        SparkTpDeviceCollectiveDestroy(&collective);
    }
    mock_server = 0u;
    cuda_stub_mesh_hardware_launch_result = 0;
    unsetenv("SPARK_TP_WAIT_MODE");
}

static void TestTopologySlice(void)
{
	SparkTpDeviceCollectiveTopology source,sliced,before;
	uint32_t rank,peer,rail,first;
	char expected[64];
	memset(&source,0,sizeof(source));
	source.abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
	source.descriptor_bytes = sizeof(source);
	source.rank_count = 16u;
	source.rail_count = 2u;
	source.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS;
	source.direct_all_to_all_max_payload_bytes = 81920u;
	source.split_ring_min_payload_bytes = 655360u;
	source.step_rail_indices[1] = 1u;
	for (rank=0u; rank<16u; rank++)
	{
		(void)snprintf(source.rank_hosts[rank],64u,"host%u",rank);
		for (rail=0u; rail<2u; rail++)
			(void)snprintf(source.rail_rank_hosts[rail][rank],64u,
			    "rail%u-host%u",rail,rank);
		for (peer=0u; peer<16u; peer++)
			source.session_ports[rank][peer] = rank == peer ? 0u :
			    (uint16_t)(60000u + rank * 16u + peer);
	}
	before = source;
	for (first=0u; first<16u; first+=4u)
	{
		CHECK(SparkTpDeviceCollectiveSliceTopology(&source,first,4u,&sliced) ==
		    SPARK_STATUS_OK,"each PP stage slices its TP rank group");
		CHECK(sliced.rank_count == 4u && sliced.rail_count == 2u &&
		    sliced.algorithm_mask == source.algorithm_mask &&
		    sliced.direct_all_to_all_max_payload_bytes == 81920u &&
		    sliced.split_ring_min_payload_bytes == 655360u &&
		    sliced.step_rail_indices[1] == 1u,"slice preserves algorithm policy");
		for (rank=0u; rank<4u; rank++)
		{
			(void)snprintf(expected,sizeof(expected),"host%u",first + rank);
			CHECK(strcmp(sliced.rank_hosts[rank],expected) == 0,"slice selects rank host");
			for (rail=0u; rail<2u; rail++)
			{
				(void)snprintf(expected,sizeof(expected),"rail%u-host%u",rail,first + rank);
				CHECK(strcmp(sliced.rail_rank_hosts[rail][rank],expected) == 0,
				    "slice selects each rail host");
			}
			for (peer=0u; peer<4u; peer++)
				CHECK(sliced.session_ports[rank][peer] == (rank == peer ? 0u :
				    60000u + (first + rank) * 16u + first + peer),
				    "slice selects source and destination session port");
		}
		CHECK(sliced.rank_hosts[4][0] == 0 && sliced.rail_rank_hosts[0][4][0] == 0 &&
		    sliced.session_ports[4][0] == 0 && sliced.session_ports[0][4] == 0,
		    "slice clears ranks outside selected group");
	}
	CHECK(memcmp(&source,&before,sizeof(source)) == 0,"slice preserves source");
	CHECK(SparkTpDeviceCollectiveSliceTopology(&source,12u,4u,&source) == SPARK_STATUS_OK &&
	    memcmp(&source,&sliced,sizeof(source)) == 0,"slice supports in-place destination");
	before = sliced;
	CHECK(SparkTpDeviceCollectiveSliceTopology(&source,3u,2u,&sliced) == SPARK_STATUS_INVALID_ARGUMENT &&
	    memcmp(&sliced,&before,sizeof(sliced)) == 0,"invalid slice preserves destination");
	CHECK(SparkTpDeviceCollectiveSliceTopology(&source,UINT32_MAX,1u,&sliced) == SPARK_STATUS_INVALID_ARGUMENT,
	    "slice rejects overflowed first rank");
}

static void TestHardwareDispatch(SparkTpDeviceCollectiveConfig config,void *mesh)
{
    SparkTpDeviceCollective collective = {0};
    SparkTpDeviceCollectiveSubmission submission = {0};
    uint16_t local[256] = {0},output[512] = {0};
    uint32_t logical,operation;
    uint32_t old_publish = cuda_stub_mesh_publish_calls;
    SparkWeightdMeshWaitRequest *request = (SparkWeightdMeshWaitRequest *)
        ((uint8_t *)mesh + SPARK_WEIGHTD_MESH_WAIT_ENTRY(0u,0u));
    config.combine_bf16_function = 0;
    config.combine_u64_max_function = 0;
    config.combine_gather_bf16_function = 0;
    setenv("SPARK_TP_WAIT_MODE","automatic",1);
    CHECK(SparkTpDeviceCollectiveCreate(&config,&collective) == SPARK_STATUS_INVALID_ARGUMENT,
        "unknown wait mode fails instead of silently selecting spin");
    setenv("SPARK_TP_WAIT_MODE","hardware",1);
    CHECK(SparkTpDeviceCollectiveCreate(&config,&collective) == SPARK_STATUS_OK,"hardware create");
    cuda_stub_mesh_hardware_prepare_result = cudaErrorUnknown;
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,mesh,2u,64u,0u,0) == SPARK_STATUS_IO_ERROR,
        "unsupported hardware wait preparation fails explicitly");
    cuda_stub_mesh_hardware_prepare_result = 0;
    request->version = SPARK_WEIGHTD_MESH_WAIT_VERSION + 1u;
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,mesh,2u,64u,0u,0) == SPARK_STATUS_ABI_MISMATCH,
        "hardware gate rejects unsupported record version");
    request->version = 0u;
    request->request_id = UINT64_MAX;
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,mesh,2u,64u,0u,0) == SPARK_STATUS_CAPACITY_EXCEEDED,
        "hardware gate rejects exhausted request IDs");
    request->request_id = 1u;
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,mesh,2u,64u,0u,0) == SPARK_STATUS_BUSY,
        "hardware gate refuses a previous owner's pending request");
    memset(request,0,sizeof(*request));
    cuda_stub_mesh_hardware_alias = (uint8_t *)mesh + 64u;
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,mesh,2u,64u,0u,0) == SPARK_STATUS_OK,
        "failed hardware preparation retains ownership for retry");
    CHECK(SparkTpDeviceCollectiveChainKey(&collective,777u) == SPARK_STATUS_OK,"hardware chain key");
    CHECK(SparkTpDeviceCollectiveArmCapture(&collective) == SPARK_STATUS_OK,"hardware arm capture");
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.active_sequence_count = 2u;
    submission.local_device = local;
    submission.full_device = output;
    submission.cuda_stream = (void *)1;
    submission.completion_function = TestComplete;
    cuda_stub_mesh_hardware_launch_result = 0;
    cuda_stub_mesh_hardware_calls = 0u;
    for ( logical = 1u; logical <= 2u; logical++ )
    {
        submission.logical_sequence_count = logical;
        for ( operation = 0u; operation < 3u; operation++ )
        {
            uint64_t expected_elements = operation == 2u ? 2u : operation == 0u ? 256u : 128u;
            CHECK(SparkTpDeviceCollectiveEnqueue(&collective,&submission,operation) == SPARK_STATUS_OK,
                "captured B1 and B2 operations select hardware launcher");
            CHECK(cuda_stub_mesh_hardware_logical_rows == 1u &&
                cuda_stub_mesh_hardware_operation == operation &&
                cuda_stub_mesh_hardware_elements == expected_elements,
                "a batch that fits one slot runs as one direct hardware round with exact payload geometry");
            CHECK(cuda_stub_mesh_hardware_band == cuda_stub_mesh_hardware_alias &&
                cuda_stub_mesh_hardware_gate == (uint8_t *)cuda_stub_mesh_hardware_alias +
                    SPARK_WEIGHTD_MESH_WAIT_ENTRY(0u,0u),
                "hardware kernels use actual mapped device alias");
        }
    }
    submission.logical_sequence_count = 1u;
    submission.active_sequence_count = 4096u;
    CHECK(SparkTpDeviceCollectiveEnqueue(&collective,&submission,1u) == SPARK_STATUS_OK &&
        cuda_stub_mesh_hardware_logical_rows == 1u && cuda_stub_mesh_hardware_elements == 262144u,
        "a single-sequence wave larger than one slot runs as chunked direct hardware rounds");
    CHECK(SparkTpMeshDirectChunks(262144u,2u,1u,SPARK_WEIGHTD_MESH_SLOT_BYTES) == 2u &&
        SparkTpMeshDirectChunks(131072u,16u,1u,SPARK_WEIGHTD_MESH_SLOT_BYTES) == 1u &&
        SparkTpMeshDirectChunks(1048576u,16u,1u,SPARK_WEIGHTD_MESH_SLOT_BYTES) == 8u &&
        SparkTpMeshDirectChunks(4096u,16u,1u,SPARK_WEIGHTD_MESH_SLOT_BYTES) == 1u &&
        SparkTpMeshDirectChunks(65536u,16u,0u,SPARK_WEIGHTD_MESH_SLOT_BYTES) == 1u &&
        SparkTpMeshDirectChunks(65536u,16u,2u,SPARK_WEIGHTD_MESH_SLOT_BYTES) == 2u,
        "direct chunk geometry: eight 16384-wide BF16 rows fit one slot, larger payloads split into slot-sized chunks");
    request->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES;
    CHECK(SparkTpDeviceCollectiveEnqueue(&collective,&submission,1u) == SPARK_STATUS_OK &&
        cuda_stub_mesh_hardware_slice_routes == 1u,
        "a weightd that advertises slice routes lets the launcher choose reduce-scatter + all-gather");
    request->capabilities = 0u;
    CHECK(SparkTpDeviceCollectiveEnqueue(&collective,&submission,1u) == SPARK_STATUS_OK &&
        cuda_stub_mesh_hardware_slice_routes == 0u,
        "an older weightd keeps every collective on direct rounds");
    CHECK(SparkTpMeshDirectPhasesPerChunk(262144u,16u,1u,1u) == 2u &&
        SparkTpMeshDirectPhasesPerChunk(262144u,16u,1u,0u) == 1u &&
        SparkTpMeshDirectPhasesPerChunk(262144u,2u,1u,1u) == 1u &&
        SparkTpMeshDirectPhasesPerChunk(32768u,16u,1u,1u) == 1u &&
        SparkTpMeshDirectPhasesPerChunk(262144u,16u,2u,1u) == 1u &&
        SparkTpMeshDirectPhasesPerChunk(262144u,16u,0u,1u) == 1u,
        "reduce-scatter + all-gather only for BF16 sums of at least 12 rows over 4+ ranks");
    CHECK(SparkTpMeshRsagSlice(131068u,16u) == 8192u && SparkTpMeshRsagSlice(513u,16u) == 36u &&
        SparkTpMeshRsagSlice(49152u,3u) == 16384u && SparkTpMeshRsagSlice(49153u,4u) % 4u == 0u,
        "slices are whole 8-byte words and cover the chunk");
    submission.active_sequence_count = 2u;
    CHECK(cuda_stub_mesh_hardware_calls == 9u && cuda_stub_mesh_publish_calls == old_publish,
        "hardware capture never dispatches spinning publish or wait path");
    CHECK(SparkTpDeviceCollectiveDisarmCapture(&collective) == SPARK_STATUS_OK,"hardware disarm capture");
    cuda_stub_mesh_hardware_launch_result = cudaErrorUnknown;
    for (logical = 1u; logical <= 2u; logical++)
    {
        submission.logical_sequence_count = logical;
        for (operation = 0u; operation < 3u; operation++)
        {
            uint32_t calls = cuda_stub_mesh_hardware_calls;
            CHECK(SparkTpDeviceCollectiveEnqueue(&collective,&submission,operation) == SPARK_STATUS_IO_ERROR &&
                cuda_stub_mesh_hardware_calls == calls + 1u &&
                cuda_stub_mesh_hardware_operation == operation,
                "eager native operation reaches hardware and propagates launch error without legacy callbacks");
        }
        uint32_t calls = cuda_stub_mesh_hardware_calls;
        CHECK(SparkTpDeviceCollectiveEnqueueRounds(&collective,&submission,3u) == SPARK_STATUS_IO_ERROR &&
            cuda_stub_mesh_hardware_calls == calls + 1u,
            "native round loop reaches hardware without a legacy BF16 callback");
    }
    cuda_stub_mesh_hardware_launch_result = 0;
    CHECK(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) == SPARK_STATUS_IO_ERROR,
        "completed launch with missing rounds is terminal, not retryable pressure");
    ((SparkTpMeshRoundControl *)cuda_stub_mesh_hardware_control)->error_word = 123u;
    CHECK(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) == SPARK_STATUS_IO_ERROR,
        "failed device round is terminal, not retryable pressure");
    ((SparkTpMeshRoundControl *)cuda_stub_mesh_hardware_control)->error_word = 0u;
    CHECK(cuda_stub_mesh_publish_calls == old_publish,"failed hardware launch never selects spin path");
    CHECK(cuda_stub_mesh_hardware_control != 0,"native dispatch provides a control record");
    if (cuda_stub_mesh_hardware_control != 0)
    {
        SparkTpMeshRoundControl *control = cuda_stub_mesh_hardware_control;
        SparkTpDeviceCollectiveHardwareTiming timing = {0};
        control->source_wait_ns = 100u;
        control->peer_wait_ns = 200u;
        control->copy_ns = 300u;
        control->combine_ns = 400u;
        cuda_stub_stream_query_result = cudaErrorNotReady;
        CHECK(SparkTpDeviceCollectiveHardwareStats(&collective,&timing) == SPARK_STATUS_BUSY,
            "hardware metrics cannot synchronize or read an active stream");
        cuda_stub_stream_query_result = cudaSuccess;
        CHECK(SparkTpDeviceCollectiveHardwareStats(&collective,&timing) == SPARK_STATUS_OK &&
            timing.source_wait_ns == 100u && timing.peer_wait_ns == 200u &&
            timing.copy_ns == 300u && timing.combine_ns == 400u,
            "terminal hardware metric readback preserves each measured phase");
        request->request_id = UINT64_MAX;
        CHECK(SparkTpDeviceCollectiveChainKey(&collective,778u) == SPARK_STATUS_CAPACITY_EXCEEDED,
            "cached mapped alias still validates request IDs before new chain");
        request->request_id = 0u;
        CHECK(SparkTpDeviceCollectiveChainKey(&collective,778u) == SPARK_STATUS_OK &&
            SparkTpDeviceCollectiveHardwareStats(&collective,&timing) == SPARK_STATUS_OK &&
            timing.source_wait_ns == 0u && timing.peer_wait_ns == 0u &&
            timing.copy_ns == 0u && timing.combine_ns == 0u,
            "new chain resets timing without erasing gate generation");
    }
    SparkTpDeviceCollectiveDestroy(&collective);
    CHECK(collective.implementation == 0,"hardware mapping owner destroys after terminal stream");
    cuda_stub_mesh_hardware_alias = 0;
    unsetenv("SPARK_TP_WAIT_MODE");
    CHECK(SparkTpDeviceCollectiveCreate(&config,&collective) == SPARK_STATUS_OK &&
        SparkTpDeviceCollectivePrepareReceiveBf16(&collective,mesh,2u,64u,0u,0) == SPARK_STATUS_OK,
        "legacy route fixture has no registered callbacks");
    submission.logical_sequence_count = 1u;
    for (operation = 0u; operation < 3u; operation++)
        CHECK(SparkTpDeviceCollectiveEnqueue(&collective,&submission,operation) == SPARK_STATUS_UNSUPPORTED &&
            SparkTpDeviceCollectiveRoundIndex(&collective) == 0u,
            "legacy B1 still rejects its missing operation callback before consuming a round");
    CHECK(SparkTpDeviceCollectiveEnqueueRounds(&collective,&submission,3u) == SPARK_STATUS_UNSUPPORTED,
        "legacy B1 round loop retains its missing callback rejection");
    CHECK(cuda_stub_mesh_publish_calls == old_publish,
        "missing legacy callbacks never publish data");
    SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestOwnedMesh(SparkTpDeviceCollectiveConfig config)
{
    SparkTpDeviceCollective collective = {0};
    uint32_t before = mock_clients_live;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&collective) == SPARK_STATUS_OK,"dense collective create");
    mock_mesh_map_status = SPARK_STATUS_BUSY;
    CHECK(SparkTpDeviceCollectiveAttachMesh(&collective) == SPARK_STATUS_BUSY,"unwired mesh fails explicitly");
    mock_mesh_map_status = SPARK_STATUS_OK;
    CHECK(SparkTpDeviceCollectiveAttachMesh(&collective) == SPARK_STATUS_OK,"dense collective maps mesh without a weight pack");
    CHECK(SparkTpDeviceCollectiveAttachMesh(&collective) == SPARK_STATUS_DUPLICATE,"duplicate mapping is rejected");
    CHECK(SparkTpDeviceCollectiveChainKey(&collective,42u) == SPARK_STATUS_OK,"fresh dense mesh can start a chain");
    CHECK(SparkTpDeviceCollectiveEndChain(&collective,0) == SPARK_STATUS_OK,"dense mesh chain ends");
    SparkTpDeviceCollectiveDestroy(&collective);
    CHECK(collective.implementation == 0 && mock_clients_live == before,"mapping owner and lane released");
    errno = 0;
    CHECK(msync(mock_owned_mapping,(size_t)sysconf(_SC_PAGESIZE),MS_SYNC) == -1 && errno == ENOMEM,"owned mesh mapping unmapped after destroy");
}

static void TestSharedLanes(SparkTpDeviceCollectiveConfig config,void *mesh)
{
    SparkTpDeviceCollective first = {0},second = {0},rejected = {0};
    SparkWeightdClient *owner = 0;
    uint32_t lane,live = mock_clients_live;
    SparkWeightdMeshTopology topology;
    CHECK(SparkTpDeviceCollectiveMeshTopology(config.tp_rank,config.tp_degree,&topology) == SPARK_STATUS_OK,"paired topology");
    uint64_t bands = mock_lane_mask[0];
    config.collective_identifier = 1u;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&first) == SPARK_STATUS_OK,
        "first model reserves a common lane");
    config.collective_identifier = 17u;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&second) == SPARK_STATUS_OK,
        "second model with modulo-colliding identifier reserves separate lane");
    CHECK(__builtin_popcountll(mock_lane_mask[0] ^ bands) == 2,
        "logical identifiers do not choose or alias physical bands");
    SparkTpDeviceCollectiveDestroy(&first);
    SparkTpDeviceCollectiveDestroy(&second);
    CHECK(mock_clients_live == live && mock_lane_mask[0] == bands,
        "owned collective teardown releases exactly its reservations");
    mock_lane_status = SPARK_STATUS_NO_LANE;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&rejected) == SPARK_STATUS_NO_LANE &&
        rejected.implementation == 0 && mock_clients_live == live,
        "reservation rejection rolls back common create client");
    mock_lane_status = SPARK_STATUS_OK;
    char invalid_lane[16];
    snprintf(invalid_lane,sizeof(invalid_lane),"%u",SPARK_WEIGHTD_MESH_MAX_LANES);
    CHECK(setenv("SPARK_WEIGHTD_LANE",invalid_lane,1) == 0 &&
        SparkTpDeviceCollectiveCreate(&config,&rejected) == SPARK_STATUS_INVALID_ARGUMENT &&
        rejected.implementation == 0 && mock_clients_live == live,
        "malformed common lane config fails without leaked client");
    CHECK(unsetenv("SPARK_WEIGHTD_LANE") == 0,"clear invalid test lane");
    CHECK(SparkWeightdClientConnect("fixture",&owner,0) == SPARK_STATUS_OK &&
        SparkWeightdClientLaneAcquire(owner,7u,&topology,&lane,1u) == SPARK_STATUS_OK,
        "paired owner reserves explicit lane");
    config.mesh_lane_client = owner;
    config.mesh_band_index = 0u;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&first) == SPARK_STATUS_OK,
        "borrowed main band binds owner reservation");
    CHECK(SparkTpDeviceCollectiveCreate(&config,&rejected) == SPARK_STATUS_DUPLICATE &&
        rejected.implementation == 0 && ((uint64_t *)owner)[3] == 1u,
        "borrowed duplicate band cannot overwrite main ownership");
    config.mesh_band_index = 1u;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&second) == SPARK_STATUS_OK &&
        ((uint64_t *)owner)[3] == 3u,"borrowed HC uses the other band");
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&first,mesh,0u,0u,0u,0) == SPARK_STATUS_OK &&
        SparkTpDeviceCollectiveChainKey(&first,912u) == SPARK_STATUS_OK,
        "borrowed owner begins a stream interval");
    CHECK(SparkTpDeviceCollectiveArmCapture(&first) == SPARK_STATUS_OK &&
        SparkTpDeviceCollectiveGraphPreLaunch(&first,(void *)1) == SPARK_STATUS_OK,
        "borrowed stream ownership is recorded");
    cuda_stub_stream_query_result = cudaErrorNotReady;
    SparkTpDeviceCollectiveDestroy(&first);
    CHECK(first.implementation != 0 && ((uint64_t *)owner)[3] == 3u,
        "busy destroy retains both borrowed band claims");
    cuda_stub_stream_query_result = cudaSuccess;
    SparkTpDeviceCollectiveDestroy(&first);
    CHECK(first.implementation == 0 && ((uint64_t *)owner)[3] == 2u &&
        (mock_lane_mask[0] & (UINT64_C(1) << 7u)) != 0u,
        "main destroy leaves HC and parent lane reservation alive");
    SparkTpDeviceCollectiveDestroy(&second);
    CHECK(((uint64_t *)owner)[3] == 0u && (mock_lane_mask[0] & (UINT64_C(1) << 7u)) != 0u,
        "last borrowed destroy leaves reservation with its explicit owner");
    mock_server = 1u;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&rejected) == SPARK_STATUS_INVALID_ARGUMENT &&
        rejected.implementation == 0,"borrowed lane from another daemon is rejected");
    mock_server = 0u;
    SparkWeightdClientClose(owner);
    CHECK(mock_clients_live == live && mock_lane_mask[0] == bands,
        "paired owner closes after both borrowed collectives release");
}

int main(void)
{
	SparkTpDeviceCollectiveConfig config;
	TestTopologySlice();
	(void)setenv("SPARK_WEIGHTD_SOCKET","/tmp/tp_collective_mock.sock",1);
	SparkTpDeviceCollective collective;
	SparkStatus status;
	void *mesh_buffer;
	void *device_scratch;
	uint8_t *peer_tails;

	cuda_stub_mesh_publish_calls = 0u;
	cuda_stub_mesh_publish_null_seq_cell = 0u;
	cuda_stub_mesh_publish_null_epoch_cell = 0u;
	mock_combine_calls = 0u;

	mesh_buffer = calloc(1u, (size_t)SPARK_WEIGHTD_MESH_REGION_BYTES);
	if ( mesh_buffer == 0 )
	{
		fprintf(stderr,"mesh buffer alloc failed\n");
		return(1);
	}

	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	config.tp_degree = 2u;
	config.tp_rank = 0u;
	config.local_hidden_dimension = 64u;
	config.max_active_sequence_count = 4096u;
	config.connect_timeout_milli = 1000u;
	config.operation_timeout_milli = 2000u;
	config.collective_identifier = 0u;
	config.combine_bf16_function = TestCombineBf16;
	config.combine_fused_bf16_function = TestCombineFusedBf16;
	config.combine_u64_max_function = TestCombineU64Max;

	memset(&collective,0,sizeof(collective));
	status = SparkTpDeviceCollectiveCreate(&config, &collective);
	CHECK(status == SPARK_STATUS_OK, "create");

	cudaMalloc(&device_scratch, 4096u);
	status = SparkTpDeviceCollectivePrepareReceiveBf16(&collective, mesh_buffer, 2u, 64u, 0u, 0);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"prepare status=%d\n",(int)status);
	CHECK(status == SPARK_STATUS_OK, "prepare receive");

	peer_tails = (uint8_t *)mesh_buffer + (1u * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK) * SPARK_WEIGHTD_MESH_SLOT_BYTES;

	{
		SparkTpDeviceCollectiveSubmission submission;
		memset(&submission,0,sizeof(submission));
		submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
		submission.descriptor_bytes = sizeof(submission);
		submission.slot_index = 0u;
		submission.active_sequence_count = 2u;
		submission.logical_sequence_count = 1u;
		submission.ordinal = 1u;
		submission.local_device = device_scratch;
		submission.full_device = device_scratch;
		submission.cuda_stream = (void *)0x1;
		submission.completion_function = TestComplete;

		status = SparkTpDeviceCollectiveEnqueue(&collective, &submission,
			SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
		CHECK( cuda_stub_mesh_publish_calls >= 1u, "eager round publishes (kernel publish ran)");
		CHECK( cuda_stub_mesh_publish_null_seq_cell == 0u, "eager publish had a live seq_cell (no NULL atomic)");
		CHECK( cuda_stub_mesh_publish_null_epoch_cell == 0u, "eager publish had a live epoch_cell");
	}

	(void)peer_tails;

	/* The reset cascade: rank 1 waits on rank 0's chain cell; rank 0
	 * cancels (its engine's session reset kills the chain locally).
	 * Rank 1 must fail fast on the cancel cell, not spin out the whole
	 * round timeout — the fleet symptom was peers wedging 30s per chain
	 * behind a reset rank. */
	{
		SparkTpDeviceCollective peer;
		SparkTpDeviceCollectiveConfig peer_config = config;
		peer_config.tp_rank = 1u;
		peer_config.operation_timeout_milli = 10000u;
		memset(&peer,0,sizeof(peer));
		mock_server = 1u;
		status = SparkTpDeviceCollectiveCreate(&peer_config, &peer);
		mock_server = 0u;
		CHECK(status == SPARK_STATUS_OK, "peer create");
		if ( status == SPARK_STATUS_OK )
		{
			pthread_t peer_thread;
			PeerWaitArgs wait_args;
			peer_config.operation_timeout_milli = 10000u;
			status = SparkTpDeviceCollectivePrepareReceiveBf16(&peer, mesh_buffer, 2u, 64u, 0u, 0);
			CHECK(status == SPARK_STATUS_OK, "peer prepare receive");
			/* negative control: without a cancel the wait burns the full
			 * timeout */
			{
				uint64_t t0 = TestNowNs();
				SparkStatus wait_status = SparkTpDeviceCollectiveChainKey(&peer, 4242u);
				uint64_t waited_ms = (TestNowNs() - t0) / 1000000ull;
				CHECK( wait_status == SPARK_STATUS_BUSY, "uncancelled wait ends BUSY at the timeout");
				CHECK( waited_ms >= 8000u, "uncancelled wait really spans the timeout");
			}
			/* the fix: rank 0's cancel must cut the wait to milliseconds */
			wait_args.collective = &peer;
			wait_args.request_id = 4243u;
			wait_args.status = SPARK_STATUS_OK;
			wait_args.waited_ns = 0u;
			pthread_create(&peer_thread,0,PeerWaitMain,&wait_args);
			usleep(200000);
			SparkTpDeviceCollectiveBroadcastCancel(&collective);
			pthread_join(peer_thread,0);
			CHECK( wait_args.status == SPARK_STATUS_BUSY, "cancelled wait ends BUSY");
			CHECK( wait_args.waited_ns < 3000000000ull, "cancelled wait fails fast (<3s, not the 10s timeout)");
			SparkTpDeviceCollectiveDestroy(&peer);
		}
	}

	SparkTpDeviceCollectiveDestroy(&collective);
	TestHardwareDispatch(config,mesh_buffer);
	TestSharedLanes(config,mesh_buffer);
	TestOwnedMesh(config);
	TestRestartEpoch(config,mesh_buffer);
	free(mesh_buffer);

	fprintf(stderr,"%s: %u checks, %u failures (publish=%u combine=%u)\n",
		"test_tp_device_collective_mock", test_checks, test_failures,
		cuda_stub_mesh_publish_calls, mock_combine_calls);
	return( test_failures != 0u ? 1 : 0 );
}
