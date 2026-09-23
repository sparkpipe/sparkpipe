#include <assert.h>
#include <sys/mman.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd.h"

#define FUZZ_MAX_RANKS SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE
#define FUZZ_HIDDEN 64u
#define FUZZ_MAX_ROWS 4u
#define FUZZ_ELEMENTS (FUZZ_HIDDEN * FUZZ_MAX_ROWS)
#define FUZZ_OUTPUT_ELEMENTS (FUZZ_ELEMENTS * FUZZ_MAX_RANKS)
#define FUZZ_MAX_SUBMISSIONS 8192u
#define FUZZ_WATCHDOG_NS (8ull * 1000000000ull)
#define FUZZ_ROUND_TIMEOUT_MS 400u

static uint32_t test_failures;
static uint32_t test_checks;
static uint32_t g_seed;
static uint32_t g_schedule;
static const char *g_phase = "setup";
static uint64_t g_round;
static uint32_t g_rows = 1u;
static uint32_t g_test_hidden = FUZZ_HIDDEN;
static uint32_t g_test_timeout_ms = FUZZ_ROUND_TIMEOUT_MS;
static uint32_t g_logical_rows = 1u;
static uint32_t g_operation = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
static uint32_t g_capture_rank = UINT32_MAX;
static volatile uint32_t g_dead_rank_mask;
static volatile uint32_t g_broadcast_fail;
static volatile uint32_t g_combine_fail;
static uint32_t g_cases;
static volatile uint64_t g_payload_deliveries;
static uint32_t g_fail_collective_alloc;
static uint32_t g_activity_fail;
static uint32_t g_activity_begins;
static uint32_t g_activity_ends;
static uint32_t g_activity_calls;

typedef struct FuzzMeshClient
{
    uint32_t rank;
    uint32_t active;
    uint64_t generation;
    uint32_t lane;
    uint32_t bands;
} FuzzMeshClient;
static uint32_t g_lane_masks[FUZZ_MAX_RANKS];

static uint32_t FuzzRandom(void)
{
    uint32_t x = g_schedule;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    g_schedule = x;
    return x;
}

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s seed=%u phase=%s round=%llu ranks=%u rows=%u logical=%u op=%u\n",__FILE__,__LINE__,name,g_seed,g_phase,(unsigned long long)g_round,g_rank_count,g_rows,g_logical_rows,g_operation); \
		} \
	} while (0)

static uint8_t *g_regions[FUZZ_MAX_RANKS];
static uint32_t g_rank_count;
static uint32_t g_connect_rank_hint;
static volatile uint64_t g_broadcast_count;
static volatile uint32_t g_shipper_stop;
static volatile uint32_t g_shipper_hold;
static volatile uint32_t g_shipper_paused;

extern uint32_t cuda_stub_roundloop_launches;
extern uint64_t cuda_stub_roundloop_rounds;
extern uint32_t cuda_stub_mesh_publish_calls;
extern uint32_t cuda_stub_mesh_seq_pad_calls;
extern int cuda_stub_stream_query_result;
extern uint32_t cuda_stub_host_register_calls;
extern uint32_t cuda_stub_host_register_flags;
extern uint32_t cuda_stub_host_unregister_calls;
extern int cuda_stub_host_register_result;
extern int cuda_stub_host_unregister_result;
extern uint32_t spark_stub_cuda_host_registered(void *address);
extern int cudaHostRegister(void *address,size_t bytes,unsigned int flags);
extern int cudaHostUnregister(void *address);

SparkStatus SparkWeightdClientConnect(const char *socket_path, SparkWeightdClient **client, SparkWeightdHelloResult *hello_out)
{
	(void)socket_path; (void)hello_out;
	*client = (SparkWeightdClient *)calloc(1u, sizeof(FuzzMeshClient));
	if ( *client == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	*(uint32_t *)*client = g_connect_rank_hint;
    ((FuzzMeshClient *)*client)->lane = SPARK_WEIGHTD_LANE_NONE;
	return(SPARK_STATUS_OK);
}

void SparkWeightdClientClose(SparkWeightdClient *client)
{
    FuzzMeshClient *mesh = (FuzzMeshClient *)client;
    assert(mesh->bands == 0u);
    if (mesh->lane < SPARK_WEIGHTD_MESH_MAX_LANES)
        g_lane_masks[mesh->rank] &= ~(1u << mesh->lane);
    free(client);
}

SparkStatus SparkWeightdClientLaneAcquire(SparkWeightdClient *client,
    uint32_t requested,const SparkWeightdMeshTopology *topology,uint32_t *out,uint64_t timeout)
{
    FuzzMeshClient *mesh = (FuzzMeshClient *)client;
    (void)timeout; (void)topology;
    if (mesh->lane != SPARK_WEIGHTD_LANE_NONE) return SPARK_STATUS_DUPLICATE;
    for (uint32_t lane=0u; lane<SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
        if ((requested == SPARK_WEIGHTD_LANE_NONE || requested == lane) &&
            (g_lane_masks[mesh->rank] & (1u << lane)) == 0u)
        {
            g_lane_masks[mesh->rank] |= 1u << lane;
            mesh->lane = lane;
            *out = lane;
            return SPARK_STATUS_OK;
        }
    return SPARK_STATUS_NO_LANE;
}

SparkStatus SparkWeightdClientLaneBind(SparkWeightdClient *owner,
    SparkWeightdClient *peer,uint32_t band,const SparkWeightdMeshTopology *topology,uint32_t *out)
{
    FuzzMeshClient *mesh = (FuzzMeshClient *)owner;
    if (topology == 0 || topology->rank_count == 0u || mesh->rank != ((const FuzzMeshClient *)peer)->rank ||
        mesh->lane >= SPARK_WEIGHTD_MESH_MAX_LANES || band >= 2u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ((mesh->bands & (1u << band)) != 0u) return SPARK_STATUS_DUPLICATE;
    mesh->bands |= 1u << band;
    *out = mesh->lane;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientLaneUnbind(SparkWeightdClient *owner,uint32_t band)
{
    FuzzMeshClient *mesh = (FuzzMeshClient *)owner;
    assert(band < 2u && (mesh->bands & (1u << band)) != 0u);
    mesh->bands &= ~(1u << band);
    return SPARK_STATUS_OK;
}

uint32_t SparkWeightdClientAlive(const SparkWeightdClient *client)
{
    return((g_dead_rank_mask & (1u << *(const uint32_t *)client)) == 0u);
}

SparkStatus SparkWeightdClientMeshActivity(SparkWeightdClient *client,
    uint64_t generation,uint32_t active,uint64_t timeout_nanoseconds)
{
    FuzzMeshClient *mesh = (FuzzMeshClient *)client;
    (void)timeout_nanoseconds;
    __sync_add_and_fetch(&g_activity_calls,1u);
    if ( g_activity_fail != 0u )
        return SPARK_STATUS_IO_ERROR;
    if ( active != 0u )
    {
        if ( mesh->active != 0u || generation <= mesh->generation )
            return SPARK_STATUS_INVALID_ARGUMENT;
        mesh->generation = generation;
        mesh->active = 1u;
        __sync_add_and_fetch(&g_activity_begins,1u);
    }
    else
    {
        if ( mesh->active == 0u || generation != mesh->generation )
            return SPARK_STATUS_INVALID_ARGUMENT;
        mesh->active = 0u;
        __sync_add_and_fetch(&g_activity_ends,1u);
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientMeshBroadcast(SparkWeightdClient *client, uint32_t peer_mask, uint64_t source_offset, uint64_t remote_offset, uint32_t length, uint64_t seq_value, uint64_t seq_remote_offset, uint64_t timeout_nanoseconds)
{
	uint32_t source_rank = *(uint32_t *)client;
	uint32_t peer;
	(void)timeout_nanoseconds;
	if ( g_broadcast_fail != 0u )
		return(SPARK_STATUS_IO_ERROR);
	__sync_add_and_fetch(&g_broadcast_count, 1u);
	if ( source_rank >= g_rank_count || g_regions[source_rank] == 0 )
		return(SPARK_STATUS_IO_ERROR);
	for ( peer = 0u; peer < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; peer++ )
	{
		if ( (peer_mask & (1u << peer)) == 0u || peer >= g_rank_count ||
		     g_regions[peer] == 0 )
			continue;
		memcpy(g_regions[peer] + remote_offset,
		    g_regions[source_rank] + source_offset, length);
        __sync_synchronize();
        if (seq_value != 0u) *(volatile uint64_t *)(g_regions[peer]+seq_remote_offset) = seq_value;
	}
	__sync_synchronize();
	return(SPARK_STATUS_OK);
}


static uint16_t FuzzBf16FromFloat(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4u);
	return((uint16_t)(bits >> 16));
}

static float FuzzBf16ToFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16;
	float out;
	memcpy(&out, &bits, 4u);
	return(out);
}

static SparkStatus FuzzCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	uint16_t *destination = (uint16_t *)destination_device;
	const uint16_t *source = (const uint16_t *)source_device;
	uint32_t count = active_sequence_count * hidden_dimension;
	uint32_t index;
	(void)combine_context; (void)cuda_stream;
	if ( g_combine_fail != 0u )
		return(SPARK_STATUS_IO_ERROR);
	for ( index = 0u; index < count; index++ )
		destination[index] = FuzzBf16FromFloat(
		    FuzzBf16ToFloat(destination[index]) +
		    FuzzBf16ToFloat(source[index]));
	return(SPARK_STATUS_OK);
}

static SparkStatus FuzzCombineU64(void *context, uint64_t *destination,
    const uint64_t *source, uint32_t count, void *stream)
{
    uint32_t i;
    (void)context; (void)stream;
    for ( i = 0u; i < count; i++ )
        if ( source[i] > destination[i] )
            destination[i] = source[i];
    return SPARK_STATUS_OK;
}

static SparkStatus FuzzGather(void *context, void *destination,
    const void *const *sources, uint32_t count, uint32_t rows,
    uint32_t hidden, void *stream)
{
    uint32_t i;
    size_t bytes = (size_t)rows * hidden * sizeof(uint16_t);
    (void)context; (void)stream;
    for ( i = 0u; i < count; i++ )
        memcpy((uint8_t *)destination + (size_t)i * bytes, sources[i], bytes);
    return SPARK_STATUS_OK;
}

typedef struct FuzzCompletionRecord
{
    uint64_t ordinal;
    uint32_t slot;
    uint32_t expect;
    volatile uint32_t count;
    volatile uint32_t wrong;
} FuzzCompletionRecord;

typedef struct FuzzRank
{
    SparkTpDeviceCollective collective;
    uint32_t rank;
    _Alignas(uint64_t) uint16_t partial[FUZZ_ELEMENTS];
    _Alignas(uint64_t) uint16_t output[FUZZ_OUTPUT_ELEMENTS + 8u];
    volatile uint64_t completion_count;
    FuzzCompletionRecord records[FUZZ_MAX_SUBMISSIONS];
    uint32_t record_count;
} FuzzRank;

static FuzzRank g_ranks[FUZZ_MAX_RANKS];
static void *g_input_override[FUZZ_MAX_RANKS];
static void *g_output_override[FUZZ_MAX_RANKS];
static volatile uint64_t g_rejected_completion_count;

static void FuzzComplete(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
    FuzzCompletionRecord *record = context;
    if ( completion->status != SPARK_STATUS_OK )
        __sync_add_and_fetch(&g_rejected_completion_count, 1u);
    if ( completion->abi_version != SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
         completion->descriptor_bytes != sizeof(*completion) ||
         completion->ordinal != record->ordinal ||
         completion->slot_index != record->slot ||
         completion->generation == 0u || completion->status != SPARK_STATUS_OK )
        __sync_add_and_fetch(&record->wrong, 1u);
    __sync_add_and_fetch(&record->count, 1u);
}

static uint64_t FuzzNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0ull);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static void *FuzzShipperMain(void *argument)
{
	uint64_t seen[FUZZ_MAX_RANKS] = {0u};
	(void)argument;
	while ( __sync_add_and_fetch(&g_shipper_stop, 0u) == 0u )
	{
		uint32_t rank;
		if ( g_shipper_hold != 0u )
		{
			__sync_lock_test_and_set(&g_shipper_paused, 1u);
			usleep(1000);
			continue;
		}
		__sync_lock_test_and_set(&g_shipper_paused, 0u);
		for ( rank = 0u; rank < g_rank_count; rank++ )
		{
			volatile uint64_t *entry = (volatile uint64_t *)
			    (g_regions[rank] +
			    SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u, rank));
			uint64_t tag;
			uint64_t slot_index;
			uint64_t bytes;
			uint64_t slot_base;
			uint64_t destinations;
			uint32_t peer;
			uint32_t stable = 0u;
			uint32_t tries;
			for ( tries = 0u; tries < 64u && stable == 0u; tries++ )
			{
				tag = entry[0];
				bytes = entry[1];
				slot_index = entry[2];
				destinations = entry[3];
				__sync_synchronize();
				if ( tag == entry[0] && bytes == entry[1] &&
				     slot_index == entry[2] && destinations == entry[3] )
					stable = 1u;
			}
			if ( stable == 0u || tag == 0ull || tag == seen[rank] )
				continue;
			if ( slot_index >= SPARK_WEIGHTD_MESH_SLOTS_PER_BAND ||
			     bytes == 0ull ||
			     bytes + 16ull > SPARK_WEIGHTD_MESH_SLOT_BYTES )
			{
				seen[rank] = tag;
				continue;
			}
			slot_base = slot_index * SPARK_WEIGHTD_MESH_SLOT_BYTES;
			for ( peer = 0u; peer < g_rank_count; peer++ )
			{
				if ( (destinations & (1u << peer)) == 0u )
					continue;
                __sync_add_and_fetch(&g_payload_deliveries,1u);
				memcpy(g_regions[peer] + slot_base,
				    g_regions[rank] + slot_base, (size_t)bytes);
			}
			__sync_synchronize();
			for ( peer = 0u; peer < g_rank_count; peer++ )
			{
				if ( (destinations & (1u << peer)) == 0u )
					continue;
				memcpy(g_regions[peer] + slot_base +
				    SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u,
				    g_regions[rank] + slot_base +
				    SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u, 8u);
			}
			__sync_synchronize();
			*(volatile uint64_t *)(g_regions[rank] +
			    SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(0u, rank)) = tag;
			__sync_synchronize();
			seen[rank] = tag;
		}
	}
	return(0);
}

static void FuzzHoldShipper(uint32_t hold)
{
    uint64_t deadline = FuzzNowNs() + UINT64_C(2000000000);
    __sync_lock_test_and_set(&g_shipper_hold,hold);
    while ( __sync_add_and_fetch(&g_shipper_paused,0u) != hold )
    {
        if ( FuzzNowNs() >= deadline )
        {
            fprintf(stderr,"shipper barrier failed seed=%u hold=%u\n",g_seed,hold);
            _Exit(2);
        }
        usleep(100);
    }
}

static SparkStatus FuzzCreateRank(uint32_t rank)
{
	SparkTpDeviceCollectiveConfig config;
	SparkStatus status;
	uint32_t index;
	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	config.tp_degree = g_rank_count;
	config.tp_rank = rank;
	config.local_hidden_dimension = g_test_hidden;
	config.max_active_sequence_count = 4u;
	config.connect_timeout_milli = 1000u;
	config.operation_timeout_milli = g_test_timeout_ms;
	config.collective_identifier = 0u;
	config.combine_bf16_function = FuzzCombineBf16;
    config.combine_u64_max_function = FuzzCombineU64;
    config.combine_gather_bf16_function = FuzzGather;
	g_connect_rank_hint = rank;
	status = SparkTpDeviceCollectiveCreate(&config, &g_ranks[rank].collective);
	if ( status != SPARK_STATUS_OK )
		return(status);
	g_ranks[rank].rank = rank;
	status = SparkTpDeviceCollectivePrepareReceiveBf16(
	    &g_ranks[rank].collective, g_regions[rank], 1u, g_test_hidden, 0u, 0);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for ( index = 0u; index < FUZZ_ELEMENTS; index++ )
		g_ranks[rank].partial[index] = FuzzBf16FromFloat((float)(rank + 1u));
	return(SPARK_STATUS_OK);
}

static void FuzzResetRank(uint32_t rank)
{
	SparkStatus status;
	SparkTpDeviceCollectiveDestroy(&g_ranks[rank].collective);
	memset(&g_ranks[rank].collective,0,sizeof(SparkTpDeviceCollective));
	status = FuzzCreateRank(rank);
	CHECK( status == SPARK_STATUS_OK, "reset recreates the rank" );
}

typedef struct FuzzTask
{
	FuzzRank *rank;
	uint64_t argument;
	uint32_t rounds;
    FuzzCompletionRecord *record;
	SparkStatus status;
	volatile uint32_t done;
	pthread_t thread;
} FuzzTask;

static FuzzTask g_tasks[FUZZ_MAX_RANKS];

static void *FuzzChainMain(void *data)
{
	FuzzTask *task = (FuzzTask *)data;
	task->status = SparkTpDeviceCollectiveChainKey(&task->rank->collective,
	    task->argument);
	__sync_synchronize();
	task->done = 1u;
	return(0);
}

static void FuzzSubmission(FuzzTask *task, SparkTpDeviceCollectiveSubmission *submission)
{
    FuzzRank *rank = task->rank;
    if ( rank->record_count >= FUZZ_MAX_SUBMISSIONS )
    {
        fprintf(stderr,"submission ledger exhausted seed=%u\n",g_seed);
        exit(2);
    }
    task->record = &rank->records[rank->record_count++];
    memset(submission,0,sizeof(*submission));
    submission->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission->descriptor_bytes = sizeof(*submission);
    submission->slot_index = (uint32_t)(task->argument % 127u);
    submission->active_sequence_count = g_rows;
    submission->logical_sequence_count = g_logical_rows;
    submission->ordinal = task->argument;
    submission->local_device = g_input_override[rank->rank] != 0 ? g_input_override[rank->rank] : rank->partial;
    submission->full_device = g_output_override[rank->rank] != 0 ? g_output_override[rank->rank] : rank->output;
    submission->cuda_stream = (void *)0x1;
    submission->completion_function = FuzzComplete;
    submission->completion_context = task->record;
    task->record->ordinal = submission->ordinal;
    task->record->slot = submission->slot_index;
}

static void *FuzzRoundMain(void *data)
{
    FuzzTask *task = data;
    SparkTpDeviceCollectiveSubmission submission;
    FuzzSubmission(task,&submission);
    task->status = SparkTpDeviceCollectiveEnqueue(&task->rank->collective,
        &submission,g_operation);
    __sync_synchronize();
    task->done = 1u;
    return 0;
}

static void *FuzzRoundsMain(void *data)
{
    FuzzTask *task = data;
    SparkTpDeviceCollectiveSubmission submission;
    FuzzSubmission(task,&submission);
    task->status = SparkTpDeviceCollectiveEnqueueRounds(&task->rank->collective,
        &submission,task->rounds);
    __sync_synchronize();
    task->done = 1u;
    return 0;
}

static void FuzzCheckCompletion(FuzzTask *task)
{
    uint64_t deadline = FuzzNowNs() + FUZZ_WATCHDOG_NS;
    FuzzCompletionRecord *record = task->record;
    if ( record == 0 )
        return;
    record->expect = task->status == SPARK_STATUS_OK &&
        task->rank->rank != g_capture_rank;
    while ( record->count < record->expect && FuzzNowNs() < deadline )
        usleep(100);
    CHECK(record->count == record->expect,
        "accepted submission completes once; rejected submission never completes");
    CHECK(record->wrong == 0u,"completion preserves submission identity and success");
    task->rank->completion_count += record->count;
}

static void FuzzCancelAll(void)
{
	uint32_t rank;
	for ( rank = 0u; rank < g_rank_count; rank++ )
		if ( g_ranks[rank].collective.implementation != 0 )
			SparkTpDeviceCollectiveBroadcastCancel(&g_ranks[rank].collective);
}

static uint32_t FuzzWedge(const char *phase, uint64_t round, uint32_t rank)
{
	test_failures++;
	fprintf(stderr,"WEDGE phase=%s round=%llu rank=%u\n", phase,
	    (unsigned long long)round, rank);
	if ( g_ranks[rank].collective.implementation != 0 )
		SparkTpDeviceCollectiveGraphStuckDump(&g_ranks[rank].collective);
	FuzzCancelAll();
    fflush(stderr);
    _Exit(1);
}

static uint32_t FuzzWaitDone(uint32_t rank, const char *phase, uint64_t round)
{
	uint64_t deadline = FuzzNowNs() + FUZZ_WATCHDOG_NS;
	while ( g_tasks[rank].done == 0u )
	{
		if ( FuzzNowNs() >= deadline )
			return(FuzzWedge(phase, round, rank));
		usleep(1000);
	}
	return(1u);
}

static uint32_t FuzzRunSet(void *(*worker)(void *), uint64_t argument,
    const uint32_t *run, uint32_t run_count, const char *phase,
    uint64_t round, int32_t reset_rank)
{
	uint32_t i;
	uint32_t ok = 1u;
    g_phase = phase;
    g_round = round;
	for ( i = 0u; i < run_count; i++ )
	{
		FuzzTask *task = &g_tasks[run[i]];
		memset(task->rank->output, 0xFF, sizeof(task->rank->output));
        task->record = 0;
		if ( worker == FuzzRoundMain && g_operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
			task->rank->partial[0] = FuzzBf16FromFloat(
			    (float)((round & 15ull) + 1u));
		task->argument = argument;
		task->status = SPARK_STATUS_INTERNAL_ERROR;
		task->done = 0u;
		__sync_synchronize();
		if ( pthread_create(&task->thread, 0, worker, task) != 0 )
		{
            fprintf(stderr,"cannot start fuzz worker seed=%u\n",g_seed);
            _Exit(2);
		}
	}
	if ( reset_rank >= 0 )
	{
		uint32_t in_set = 0u;
		for ( i = 0u; i < run_count; i++ )
			if ( run[i] == (uint32_t)reset_rank )
				in_set = 1u;
		if ( in_set == 0u )
			FuzzResetRank((uint32_t)reset_rank);
		else if ( FuzzWaitDone((uint32_t)reset_rank, phase, round) == 0u )
			ok = 0u;
		else
		{
			pthread_join(g_tasks[reset_rank].thread, 0);
            FuzzCheckCompletion(&g_tasks[reset_rank]);
			g_tasks[reset_rank].done = 2u;
			FuzzResetRank((uint32_t)reset_rank);
		}
	}
	for ( i = 0u; i < run_count; i++ )
	{
		uint32_t rank = run[i];
		if ( g_tasks[rank].done == 2u )
			continue;
		if ( g_tasks[rank].done == 0u &&
		     FuzzWaitDone(rank, phase, round) == 0u )
			ok = 0u;
		pthread_join(g_tasks[rank].thread, 0);
        FuzzCheckCompletion(&g_tasks[rank]);
	}
	return(ok);
}

static uint32_t FuzzSumOk(uint32_t rank, uint64_t round)
{
	uint16_t expected = FuzzBf16FromFloat(
	    (float)(g_rank_count * (g_rank_count + 1u) / 2u));
	uint16_t expected_zero = FuzzBf16FromFloat(
	    (float)(g_rank_count * ((uint32_t)(round & 15ull) + 1u)));
	uint32_t index;
	for ( index = 0u; index < FUZZ_HIDDEN * g_rows; index++ )
	{
		uint16_t want = index == 0u ? expected_zero : expected;
		if ( g_ranks[rank].output[index] != want )
		{
			fprintf(stderr,"WRONG-SUM rank=%u elem=%u got=%u want=%u\n",
			    rank, index, (unsigned)g_ranks[rank].output[index],
			    (unsigned)want);
			return(0u);
		}
	}
	return(1u);
}

static uint32_t FuzzAllRanks(uint32_t *run)
{
	uint32_t rank;
	for ( rank = 0u; rank < g_rank_count; rank++ )
		run[rank] = rank;
	return(g_rank_count);
}

static void FuzzLaunchRoundsAsync(uint64_t ordinal, uint32_t round_count,
    uint64_t sum_round)
{
	uint32_t run[FUZZ_MAX_RANKS];
	uint32_t run_count = FuzzAllRanks(run);
	uint32_t i;
	for ( i = 0u; i < run_count; i++ )
	{
		FuzzTask *task = &g_tasks[run[i]];
		memset(task->rank->output, 0xFF, sizeof(task->rank->output));
		task->rank->partial[0] = FuzzBf16FromFloat(
		    (float)((sum_round & 15ull) + 1u));
		task->argument = ordinal;
		task->rounds = round_count;
		task->status = SPARK_STATUS_INTERNAL_ERROR;
		task->done = 0u;
		__sync_synchronize();
		if ( pthread_create(&task->thread, 0, FuzzRoundsMain, task) != 0 )
		{
            fprintf(stderr,"cannot start fuzz worker seed=%u\n",g_seed);
            _Exit(2);
		}
	}
}

static uint32_t FuzzJoinRounds(const char *phase, uint64_t sum_round)
{
	uint32_t run[FUZZ_MAX_RANKS];
	uint32_t run_count = FuzzAllRanks(run);
	uint32_t i;
	uint32_t ok = 1u;
	for ( i = 0u; i < run_count; i++ )
	{
		uint32_t rank = run[i];
		if ( g_tasks[rank].done == 0u &&
		     FuzzWaitDone(rank, phase, sum_round) == 0u )
			ok = 0u;
		pthread_join(g_tasks[rank].thread, 0);
        FuzzCheckCompletion(&g_tasks[rank]);
	}
	return(ok);
}

static uint32_t FuzzLaunchRounds(uint64_t ordinal, uint32_t round_count,
    uint64_t sum_round, const char *phase)
{
	FuzzLaunchRoundsAsync(ordinal, round_count, sum_round);
	return(FuzzJoinRounds(phase, sum_round));
}

static void FuzzS25S3(void)
{
	uint32_t run[FUZZ_MAX_RANKS] = {0};
	uint32_t run_count = FuzzAllRanks(run);
	uint64_t request = 700000u;
	uint64_t completions_before[FUZZ_MAX_RANKS];
	uint64_t stub_rounds_before;
	uint32_t launches_before;
	uint32_t publish_before;
	const uint32_t loop_rounds = 6u;
	uint64_t loop_last = 90u;
	uint64_t ordinals = 16ull * 100ull + 1ull;
	uint32_t rank;
	request++;
	CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
	    "s3-chain", 0u, -1) != 0u, "round-loop chain key completes" );
	for ( rank = 0u; rank < run_count; rank++ )
		CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
		    "round-loop chain key status ok" );
	launches_before = cuda_stub_roundloop_launches;
	publish_before = cuda_stub_mesh_publish_calls;
	stub_rounds_before = cuda_stub_roundloop_rounds;
	for ( rank = 0u; rank < run_count; rank++ )
		completions_before[rank] = g_ranks[rank].completion_count;
	CHECK( FuzzLaunchRounds(ordinals, loop_rounds, loop_last, "s3-loop")
	    != 0u, "device round loop completes" );
	for ( rank = 0u; rank < run_count; rank++ )
	{
		CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
		    "device round loop status ok" );
		CHECK( FuzzSumOk(rank, loop_last) != 0u,
		    "device round loop sum correct" );
		CHECK( SparkTpDeviceCollectiveDeviceRoundsDone(
		        &g_ranks[rank].collective) == loop_rounds,
		    "device rounds_done advanced device-side" );
		CHECK( g_ranks[rank].completion_count ==
		    completions_before[rank] + 1ull,
		    "round loop fires one completion" );
	}
	CHECK( cuda_stub_roundloop_launches == launches_before + run_count,
	    "one host launch per rank for the whole loop" );
	CHECK( cuda_stub_mesh_publish_calls == publish_before,
	    "no per-round host publish mediation during the loop" );
	CHECK( cuda_stub_roundloop_rounds ==
	    stub_rounds_before + (uint64_t)loop_rounds * run_count,
	    "stub round loop executed every armed round" );
	{
		uint64_t cancelled_done;
		FuzzHoldShipper(1u);
		__sync_synchronize();
		request++;
		CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
		    "s3-cancel-chain", 0u, -1) != 0u,
		    "cancel-case chain key completes" );
		FuzzLaunchRoundsAsync(ordinals + 1ull, loop_rounds,
		    loop_last + 1ull);
		usleep(50000);
		FuzzCancelAll();
		CHECK( FuzzJoinRounds("s3-cancel", loop_last + 1ull) != 0u,
		    "cancel-case loop finishes bounded" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status != SPARK_STATUS_OK,
			    "cancelled device loop fails BUSY" );
			cancelled_done = SparkTpDeviceCollectiveDeviceRoundsDone(
			    &g_ranks[rank].collective);
			CHECK( cancelled_done < loop_rounds,
			    "cancelled device loop completed fewer rounds" );
			CHECK( SparkTpDeviceCollectiveGraphError(
			        &g_ranks[rank].collective) == 0ull,
			    "cancel is not a device error" );
		}
		request++;
		CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
		    "s3-deadman-chain", 0u, -1) != 0u,
		    "deadman-case chain key completes" );
		CHECK( FuzzLaunchRounds(ordinals + 2ull, loop_rounds,
		    loop_last + 2ull, "s3-deadman") != 0u,
		    "deadman-case loop finishes bounded" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status != SPARK_STATUS_OK,
			    "unshipped device loop hits the deadline" );
			CHECK( SparkTpDeviceCollectiveGraphError(
			        &g_ranks[rank].collective) != 0ull,
			    "deadline failure is a loud device error" );
		}
		FuzzHoldShipper(0u);
		__sync_synchronize();
		request++;
		CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
		    "s3-recover-chain", 0u, -1) != 0u,
		    "post-cancel chain key completes" );
		CHECK( FuzzLaunchRounds(ordinals + 3ull, loop_rounds,
		    loop_last + 3ull, "s3-recover") != 0u,
		    "post-cancel device loop completes" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "post-cancel device loop status ok" );
			CHECK( FuzzSumOk(rank, loop_last + 3ull) != 0u,
			    "post-cancel device loop sum correct" );
		}
	}
	fprintf(stderr,
	    "s25/s3: loop_rounds=%u launches=%u stub_rounds=%llu publish=%u\n",
	    loop_rounds, cuda_stub_roundloop_launches,
	    (unsigned long long)cuda_stub_roundloop_rounds,
	    cuda_stub_mesh_publish_calls);
}

static void FuzzGraphPath(void)
{
	uint32_t run[FUZZ_MAX_RANKS];
	uint32_t run_count = FuzzAllRanks(run);
	uint64_t progress = 0ull;
	uint32_t i;
	uint32_t pad_calls = cuda_stub_mesh_seq_pad_calls;
    static uint64_t request = 2000000u;
    CHECK(FuzzRunSet(FuzzChainMain,request++,run,run_count,"graph-chain",0u,-1),
        "graph path starts from a shared request generation");
    g_capture_rank = 0u;
	CHECK( SparkTpDeviceCollectiveArmCapture(&g_ranks[0].collective) == SPARK_STATUS_OK,
	    "arm capture on rank 0" );
	CHECK( SparkTpDeviceCollectiveGraphPreLaunch(&g_ranks[0].collective,
	    (void *)0x1) == SPARK_STATUS_OK, "graph pre-launch pads to capture parity" );
	for ( i = 0u; i < 4u; i++ )
	{
		uint64_t ordinal = 16ull * (uint64_t)i + 1ull;
		CHECK( FuzzRunSet(FuzzRoundMain, ordinal, run, run_count,
		    "graph-round", i, -1) != 0u, "graph-path round completes" );
        for ( uint32_t rank = 0u; rank < run_count; rank++ )
        {
            CHECK(g_tasks[rank].status == SPARK_STATUS_OK,"graph-path rank reports success");
            CHECK(FuzzSumOk(rank,i),"graph-path output includes every current contribution");
        }
	}
	CHECK( cuda_stub_mesh_seq_pad_calls - pad_calls <= 1u,
	    "pre-launch used at most one pad" );
	progress = SparkTpDeviceCollectiveGraphProgress(&g_ranks[0].collective,0);
	CHECK( progress != 0ull, "graph progress nonzero after capture-path rounds" );
	CHECK( SparkTpDeviceCollectiveGraphError(&g_ranks[0].collective) == 0ull,
	    "no graph error after capture-path rounds" );
	SparkTpDeviceCollectiveDisarmCapture(&g_ranks[0].collective);
	CHECK( SparkTpDeviceCollectiveDisarmCapture(&g_ranks[0].collective) == SPARK_STATUS_OK,
	    "disarm after capture rounds" );
    g_capture_rank = UINT32_MAX;
	for ( i = 0u; i < run_count; i++ )
	{
		SparkTpDeviceCollectiveRoundStats(&g_ranks[i].collective,0,0,1u);
		g_ranks[i].completion_count = 0u;
	}
	g_broadcast_count = 0u;
}

static void FuzzBasic(void)
{
	uint32_t run[FUZZ_MAX_RANKS];
	uint32_t run_count = FuzzAllRanks(run);
	uint32_t i, rank;
	uint64_t request = 1000u;
	for ( i = 0u; i < 8u; i++ )
	{
		uint64_t ordinal = 16ull * (uint64_t)i + 1ull;
		request++;
		CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
		    "chain", i, -1) != 0u, "chain key round completes" );
		for ( rank = 0u; rank < run_count; rank++ )
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "chain key status ok" );
		CHECK( FuzzRunSet(FuzzRoundMain, ordinal, run, run_count,
		    "round", i, -1) != 0u, "allreduce round completes" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "round status ok" );
			CHECK( FuzzSumOk(rank, i) != 0u, "round sum correct" );
		}
	}
	for ( rank = 0u; rank < run_count; rank++ )
	{
		uint64_t count = 0u;
		SparkTpDeviceCollectiveRoundStats(&g_ranks[rank].collective,
		    &count, 0, 0u);
		CHECK( count == 8u, "round stats advance (no rounds=0 wedge)" );
	}
	request++;
	CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
	    "chain", 8u, -1) != 0u, "multi-round chain key completes" );
	for ( i = 0u; i < 4u; i++ )
	{
		CHECK( FuzzRunSet(FuzzRoundMain, 16ull * 9ull + 1ull + (uint64_t)i, run,
		    run_count, "round", 8u + i, -1) != 0u,
		    "chained round completes" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "chained round status ok" );
			CHECK( FuzzSumOk(rank, 8u + i) != 0u, "chained round sum correct" );
		}
	}
	{
		uint64_t deadline = FuzzNowNs() + 2ull * 1000000000ull;
		uint32_t drained;
		do {
			drained = 1u;
			for ( rank = 0u; rank < run_count; rank++ )
				if ( g_ranks[rank].completion_count != 12u )
					drained = 0u;
			if ( drained == 0u )
				usleep(1000);
		} while ( drained == 0u && FuzzNowNs() < deadline );
		for ( rank = 0u; rank < run_count; rank++ )
			CHECK( g_ranks[rank].completion_count == 12u,
			    "every round fires its completion" );
	}
	{
		uint64_t hold_tag[FUZZ_MAX_RANKS];
		uint64_t ack_deadline;
		uint32_t acked;
		request++;
		CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
		    "chain", 12u, -1) != 0u, "hold-case chain key completes" );
		for ( rank = 0u; rank < run_count; rank++ )
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "hold-case chain key status ok" );
		CHECK( FuzzRunSet(FuzzRoundMain, 16ull * 12ull + 1ull, run,
		    run_count, "round", 12u, -1) != 0u,
		    "hold-case settle round completes" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "hold-case settle round status ok" );
			CHECK( FuzzSumOk(rank, 12u) != 0u,
			    "hold-case settle sum correct" );
		}
		ack_deadline = FuzzNowNs() + 2ull * 1000000000ull;
		do {
			acked = 1u;
			for ( rank = 0u; rank < run_count; rank++ )
			{
				volatile uint64_t *entry = (volatile uint64_t *)
				    (g_regions[rank] +
				    SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u, rank));
				volatile uint64_t *shipped = (volatile uint64_t *)
				    (g_regions[rank] +
				    SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(0u, rank));
				if ( *shipped != *entry )
					acked = 0u;
			}
			if ( acked == 0u )
				usleep(1000);
		} while ( acked == 0u && FuzzNowNs() < ack_deadline );
		CHECK( acked != 0u, "settle round ships are acked" );
		FuzzHoldShipper(1u);
		__sync_synchronize();
		CHECK( FuzzRunSet(FuzzRoundMain, 16ull * 13ull + 2ull, run,
		    run_count, "round", 13u, -1) != 0u,
		    "held round finishes (as failure)" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status != SPARK_STATUS_OK,
			    "held round fails without ships" );
			hold_tag[rank] = *(volatile uint64_t *)(g_regions[rank] +
			    SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u, rank));
			CHECK( hold_tag[rank] != 0u, "held round published" );
		}
		CHECK( FuzzRunSet(FuzzRoundMain, 16ull * 14ull + 3ull, run,
		    run_count, "round", 14u, -1) != 0u,
		    "second held round finishes (as failure)" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status != SPARK_STATUS_OK,
			    "second held round fails without ships" );
			CHECK( *(volatile uint64_t *)(g_regions[rank] +
			    SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u, rank)) ==
			    hold_tag[rank],
			    "blocked publish never overwrites the doorbell" );
		}
		FuzzHoldShipper(0u);
		__sync_synchronize();
		request++;
		CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
		    "chain", 15u, -1) != 0u, "post-hold chain key completes" );
		CHECK( FuzzRunSet(FuzzRoundMain, 16ull * 15ull + 1ull, run,
		    run_count, "round", 15u, -1) != 0u,
		    "post-hold round completes" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "post-hold round status ok" );
			CHECK( FuzzSumOk(rank, 15u) != 0u,
			    "post-hold sum correct (nothing dropped)" );
		}
	}
}

static uint32_t FuzzRun(uint32_t rounds, uint32_t seed, uint32_t kill_percent)
{
	uint32_t run[FUZZ_MAX_RANKS];
	uint32_t run_count = FuzzAllRanks(run);
	uint32_t unrecovered = 0u;
	uint32_t recovery_pending = 0u;
	uint64_t request = 500000u;
	uint32_t round;
    g_seed = seed;
    g_schedule = seed;
    fprintf(stderr,"SCHEDULE seed=%u ranks=%u rounds=%u kill_percent=%u\n",
        seed,g_rank_count,rounds,kill_percent);
	for ( round = 0u; round < rounds || recovery_pending != 0u; round++ )
	{
		uint32_t kill = 0u;
		uint32_t victim = 0u;
		uint32_t kill_before = 0u;
		uint32_t failures_before = test_failures;
		uint32_t i;
		if ( recovery_pending == 0u &&
		     FuzzRandom() % 100u < kill_percent )
		{
			kill = 1u;
			victim = FuzzRandom() % g_rank_count;
			kill_before = FuzzRandom() & 1u;
		}
        for ( i = run_count; i > 1u; i-- )
        {
            uint32_t swap = FuzzRandom() % i;
            uint32_t temporary = run[i - 1u];
            run[i - 1u] = run[swap];
            run[swap] = temporary;
        }
        fprintf(stderr,"TRACE seed=%u round=%u event=%s rank=%u order=",seed,round,
            kill == 0u ? "healthy" : kill_before != 0u ? "missing-peer" : "completed-rank-recreate",victim);
        for ( i = 0u; i < run_count; i++ ) fprintf(stderr,"%s%u",i == 0u ? "" : ",",run[i]);
        fputc('\n',stderr);
		request++;
		if ( FuzzRunSet(FuzzChainMain, request, run, run_count,
		        "chain", round, -1) == 0u )
		{
			fprintf(stderr,"%u rounds, %u unrecovered\n", rounds,
			    unrecovered + 1u);
			return(1u);
		}
		for ( i = 0u; i < run_count; i++ )
			CHECK( g_tasks[i].status == SPARK_STATUS_OK,
			    "fuzz chain key status ok" );
		if ( kill != 0u && kill_before != 0u )
		{
			uint32_t others[FUZZ_MAX_RANKS];
			uint32_t other_count = 0u;
			for ( i = 0u; i < run_count; i++ )
				if ( run[i] != victim )
					others[other_count++] = run[i];
			if ( FuzzRunSet(FuzzRoundMain, 16ull * (uint64_t)round + 1ull,
			        others, other_count, "round", round,
			        (int32_t)victim) == 0u )
			{
				fprintf(stderr,"%u rounds, %u unrecovered\n",
				    rounds, unrecovered + 1u);
				return(1u);
			}
			for ( i = 0u; i < other_count; i++ )
				CHECK( g_tasks[others[i]].status != SPARK_STATUS_OK,
				    "killed-before-publish rank fails peers fast" );
			recovery_pending = 1u;
		}
		else if ( kill != 0u )
		{
			if ( FuzzRunSet(FuzzRoundMain, 16ull * (uint64_t)round + 1ull,
			        run, run_count, "round", round,
			        (int32_t)victim) == 0u )
			{
				fprintf(stderr,"%u rounds, %u unrecovered\n",
				    rounds, unrecovered + 1u);
				return(1u);
			}
			for ( i = 0u; i < run_count; i++ )
			{
				CHECK( g_tasks[i].status == SPARK_STATUS_OK,
				    "completed-rank teardown preserves successful peers" );
				CHECK( FuzzSumOk(i, round) != 0u,
				    "completed-rank teardown preserves reduced values" );
			}
			recovery_pending = 1u;
		}
		else
		{
			if ( FuzzRunSet(FuzzRoundMain, 16ull * (uint64_t)round + 1ull,
			        run, run_count, "round", round, -1) == 0u )
			{
				fprintf(stderr,"%u rounds, %u unrecovered\n",
				    rounds, unrecovered + 1u);
				return(1u);
			}
			for ( i = 0u; i < run_count; i++ )
			{
				CHECK( g_tasks[i].status == SPARK_STATUS_OK,
				    "fuzz round status ok" );
				CHECK( FuzzSumOk(i, round) != 0u, "fuzz round sum correct" );
			}
			if ( recovery_pending != 0u )
			{
				if ( test_failures != failures_before )
					unrecovered++;
				recovery_pending = 0u;
			}
		}
	}
	fprintf(stderr,"%u rounds, %u unrecovered\n", rounds, unrecovered);
	return( unrecovered != 0u ? 1u : 0u );
}

static void FuzzCase(const char *name)
{
    g_phase = name;
    g_round = ++g_cases;
    fprintf(stderr,"CASE seed=%u ranks=%u name=%s\n",g_seed,g_rank_count,name);
}

static void FuzzConfiguration(void)
{
    SparkTpDeviceCollectiveConfig config;
    uint32_t i;
    FuzzCase("invalid-configuration");
    for ( i = 0u; i < 8u; i++ )
    {
        SparkTpDeviceCollective collective;
        SparkStatus status, expected = SPARK_STATUS_INVALID_ARGUMENT;
        memset(&config,0,sizeof(config));
        memset(&collective,0,sizeof(collective));
        config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        config.tp_degree = g_rank_count;
        config.local_hidden_dimension = FUZZ_HIDDEN;
        config.max_active_sequence_count = FUZZ_MAX_ROWS;
        config.operation_timeout_milli = FUZZ_ROUND_TIMEOUT_MS;
        switch ( i )
        {
            case 0u: config.abi_version--; expected = SPARK_STATUS_ABI_MISMATCH; break;
            case 1u: config.tp_degree = 0u; break;
            case 2u: config.tp_degree = FUZZ_MAX_RANKS + 1u; break;
            case 3u: config.tp_rank = g_rank_count; break;
            case 4u: config.local_hidden_dimension = 0u; break;
            case 5u: config.max_active_sequence_count = 0u; break;
            case 6u: config.operation_timeout_milli = 0u; break;
            case 7u: config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
                expected = SPARK_STATUS_UNSUPPORTED; break;
        }
        status = SparkTpDeviceCollectiveCreate(&config,&collective);
        CHECK(status == expected,"invalid geometry or unsupported backend fails explicitly");
        CHECK(collective.implementation == 0,"rejected configuration owns no live collective");
        if ( collective.implementation != 0 ) SparkTpDeviceCollectiveDestroy(&collective);
    }
}

static void FuzzRejections(void)
{
    SparkTpDeviceCollectiveSubmission valid, bad;
    FuzzTask *task = &g_tasks[0];
    uint32_t i;
    task->argument = 1000000u;
    FuzzSubmission(task,&valid);
    FuzzCase("invalid-submission-no-publication");
    for ( i = 0u; i < 12u; i++ )
    {
        SparkStatus expected = SPARK_STATUS_INVALID_ARGUMENT;
        SparkStatus status;
        uint64_t before = SparkTpDeviceCollectiveRoundIndex(&task->rank->collective);
        uint32_t published = cuda_stub_mesh_publish_calls;
        bad = valid;
        switch ( i )
        {
            case 0u: bad.local_device = 0; break;
            case 1u: bad.full_device = 0; break;
            case 2u: bad.cuda_stream = 0; break;
            case 3u: bad.active_sequence_count = 0u; break;
            case 4u: bad.logical_sequence_count = 0u; break;
            case 5u: bad.active_sequence_count = FUZZ_MAX_ROWS + 1u; break;
            case 6u: bad.ordinal = UINT64_MAX; break;
            case 7u: bad.completion_function = 0; break;
            case 8u: bad.flags = UINT32_MAX; break;
            case 9u: bad.reserved0 = 1u; break;
            case 10u: bad.abi_version--; expected = SPARK_STATUS_ABI_MISMATCH; break;
            case 11u: bad.descriptor_bytes--; expected = SPARK_STATUS_ABI_MISMATCH; break;
        }
        status = SparkTpDeviceCollectiveEnqueue(&task->rank->collective,&bad,g_operation);
        CHECK(status == expected,"malformed submission rejects at its validation boundary");
        CHECK(SparkTpDeviceCollectiveRoundIndex(&task->rank->collective) == before &&
            cuda_stub_mesh_publish_calls == published,"rejection preserves round and publication state");
        CHECK(task->record->count == 0u,"rejected descriptor never acquires callback ownership");
    }
    {
        uint32_t header[2] = {SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION,8u};
        const SparkTpDeviceCollectiveSubmission *truncated = (const void *)header;
        CHECK(SparkTpDeviceCollectiveEnqueue(&task->rank->collective,truncated,g_operation) ==
            SPARK_STATUS_ABI_MISMATCH,"truncated descriptor rejects before reading its missing body");
        CHECK(SparkTpDeviceCollectiveEnqueueRounds(&task->rank->collective,truncated,1u) ==
            SPARK_STATUS_ABI_MISMATCH,"round-loop rejects truncated descriptor before copying it");
    }
    CHECK(SparkTpDeviceCollectiveEnqueue(&task->rank->collective,&valid,UINT32_MAX) ==
        SPARK_STATUS_INVALID_ARGUMENT,"unknown operation is not silently interpreted as sum");
    CHECK(SparkTpDeviceCollectiveEnqueueRounds(&task->rank->collective,&valid,0u) ==
        SPARK_STATUS_INVALID_ARGUMENT,"zero device rounds reject");
    CHECK(SparkTpDeviceCollectiveEnqueueRounds(&task->rank->collective,&valid,UINT32_MAX) ==
        SPARK_STATUS_CAPACITY_EXCEEDED,"round generation exhaustion rejects before device access");
    CHECK(SparkTpDeviceCollectiveChainKey(&task->rank->collective,
        SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK + 1u) == SPARK_STATUS_INVALID_ARGUMENT,
        "unrepresentable request identity rejects");
}

static void FuzzSourceLifetime(void);
static void FuzzCellInitialization(void);
static void FuzzLaneNamespace(void);
static void FuzzMeshTopology(void);

static void FuzzCompletionCapacity(void)
{
    SparkTpDeviceCollectiveSubmission submission;
    FuzzTask *task = &g_tasks[0];
    uint32_t mode;
    FuzzCase("completion-allocation-before-acceptance");
    for ( mode = 0u; mode < 2u; mode++ )
    {
        SparkStatus status;
        uint64_t before = SparkTpDeviceCollectiveRoundIndex(&task->rank->collective);
        uint32_t published = cuda_stub_mesh_publish_calls;
        task->argument = 1040000u + mode;
        FuzzSubmission(task,&submission);
        g_fail_collective_alloc = 1u;
        status = mode == 0u ? SparkTpDeviceCollectiveEnqueue(&task->rank->collective,
            &submission,g_operation) : SparkTpDeviceCollectiveEnqueueRounds(
                &task->rank->collective,&submission,1u);
        CHECK(g_fail_collective_alloc == 0u,"completion reservation exercises the injected allocator failure");
        g_fail_collective_alloc = 0u;
        CHECK(status == SPARK_STATUS_CAPACITY_EXCEEDED,"completion OOM rejects before accepting ownership");
        CHECK(SparkTpDeviceCollectiveRoundIndex(&task->rank->collective) == before &&
            cuda_stub_mesh_publish_calls == published,"completion OOM cannot launch untracked device work");
        CHECK(task->record->count == 0u,"completion OOM cannot produce a callback");
    }
}

static void FuzzRegistrationOwnership(void)
{
    SparkTpDeviceCollectiveConfig config;
    SparkTpDeviceCollective owners[4];
    uint32_t index,calls;
    FuzzCase("mapped-region-registration-ownership");
    memset(&config,0,sizeof(config));
    memset(owners,0,sizeof(owners));
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.tp_degree = g_rank_count;
    config.local_hidden_dimension = FUZZ_HIDDEN;
    config.max_active_sequence_count = 1u;
    config.operation_timeout_milli = FUZZ_ROUND_TIMEOUT_MS;
    for ( index = 0u; index < 4u; index++ )
    {
        SparkStatus status = SparkTpDeviceCollectiveCreate(&config,&owners[index]);
        CHECK(status == SPARK_STATUS_OK,"registration owner creates");
        if ( status != SPARK_STATUS_OK ) _Exit(2);
    }
    cuda_stub_host_register_result = 2;
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&owners[0],g_regions[0],1u,
        FUZZ_HIDDEN,0u,0) == SPARK_STATUS_IO_ERROR &&
        spark_stub_cuda_host_registered(g_regions[0]) == 0u,
        "registration failure cannot accept pageable mesh memory");
    cuda_stub_host_register_result = 0;
    calls = cuda_stub_host_register_calls;
    for ( index = 0u; index < 2u; index++ )
        CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&owners[index],g_regions[0],1u,
            FUZZ_HIDDEN,0u,0) == SPARK_STATUS_OK,"main and HC share the exact registered mapping");
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&owners[0],g_regions[0],1u,
        FUZZ_HIDDEN,0u,0) == SPARK_STATUS_OK && cuda_stub_host_register_calls == calls + 1u,
        "same owner reprepare and shared owner perform one registration");
    CHECK(cuda_stub_host_register_flags == 3u,
        "shared mapping explicitly requests portable CUDA context access and device mapping");
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&owners[2],g_regions[1],1u,
        FUZZ_HIDDEN,0u,0) == SPARK_STATUS_OK && cuda_stub_host_register_calls == calls + 2u &&
        spark_stub_cuda_host_registered(g_regions[0]) != 0u &&
        spark_stub_cuda_host_registered(g_regions[1]) != 0u,
        "distinct live mappings each have an independent CUDA registration");
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&owners[3],g_regions[0] + 4096u,1u,
        FUZZ_HIDDEN,0u,0) == SPARK_STATUS_IO_ERROR,
        "partially overlapping registration is not borrowed from another owner");
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&owners[0],g_regions[1],1u,
        FUZZ_HIDDEN,0u,0) == SPARK_STATUS_UNSUPPORTED,
        "bound collective rejects mapping replacement without changing ownership");
    {
        SparkTpDeviceCollective skip_owner;
        SparkStatus skip_status;
        uint32_t skip_calls;
        memset(&skip_owner,0,sizeof(skip_owner));
        skip_status = SparkTpDeviceCollectiveCreate(&config,&skip_owner);
        CHECK(skip_status == SPARK_STATUS_OK,"skip-path owner creates");
        {
            void *fresh = mmap(0,(size_t)SPARK_WEIGHTD_MESH_REGION_BYTES,
                PROT_READ | PROT_WRITE,MAP_PRIVATE | MAP_ANONYMOUS,-1,0);
            CHECK(fresh != MAP_FAILED,"skip-path fresh mapping");
            if ( fresh != MAP_FAILED )
            {
                cuda_stub_host_register_result = 1;
                skip_calls = cuda_stub_host_register_calls;
                CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&skip_owner,fresh,1u,
                    FUZZ_HIDDEN,0u,0) == SPARK_STATUS_OK,
                    "cudaErrorInvalidValue skips registration on the coherent host path");
                CHECK(spark_stub_cuda_host_registered(fresh) == 0u &&
                    cuda_stub_host_register_calls == skip_calls + 1u,
                    "skip path attempts once, registers nothing, still owns the mapping");
                CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&skip_owner,fresh,1u,
                    FUZZ_HIDDEN,0u,0) == SPARK_STATUS_OK &&
                    cuda_stub_host_register_calls == skip_calls + 1u,
                    "skipped registration does not retry on reprepare");
                SparkTpDeviceCollectiveDestroy(&skip_owner);
                cuda_stub_host_register_result = 0;
                munmap(fresh,(size_t)SPARK_WEIGHTD_MESH_REGION_BYTES);
            }
        }
    }
    calls = cuda_stub_host_unregister_calls;
    SparkTpDeviceCollectiveDestroy(&owners[0]);
    CHECK(owners[0].implementation == 0 && cuda_stub_host_unregister_calls == calls &&
        spark_stub_cuda_host_registered(g_regions[0]) != 0u,
        "first shared owner cannot unregister the remaining owner's mapping");
    cuda_stub_host_unregister_result = 2;
    SparkTpDeviceCollectiveDestroy(&owners[1]);
    CHECK(owners[1].implementation != 0 && spark_stub_cuda_host_registered(g_regions[0]) != 0u,
        "failed final unregister retains retryable ownership");
    {
        SparkTpDeviceCollectiveSubmission submission;
        uint32_t activities = g_activity_calls,published = cuda_stub_mesh_publish_calls;
        memset(&submission,0,sizeof(submission));
        submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        submission.descriptor_bytes = sizeof(submission);
        submission.active_sequence_count = 1u;
        submission.logical_sequence_count = 1u;
        submission.local_device = g_regions[0];
        submission.full_device = g_regions[1];
        submission.cuda_stream = (void *)0x1;
        CHECK(SparkTpDeviceCollectiveSubmitBf16(&owners[1],&submission) == SPARK_STATUS_BUSY &&
            SparkTpDeviceCollectiveEnqueueRounds(&owners[1],&submission,1u) == SPARK_STATUS_BUSY &&
            SparkTpDeviceCollectiveChainKey(&owners[1],1u) == SPARK_STATUS_BUSY &&
            SparkTpDeviceCollectiveArmCapture(&owners[1]) == SPARK_STATUS_BUSY &&
            SparkTpDeviceCollectiveGraphCancelSeed(&owners[1],(void *)0x1) == SPARK_STATUS_BUSY &&
            SparkTpDeviceCollectivePrepareReceiveBf16(&owners[1],g_regions[0],1u,FUZZ_HIDDEN,0u,0) == SPARK_STATUS_BUSY,
            "retained destruction accepts only cleanup retry after stopping its completion worker");
        CHECK(g_activity_calls == activities && cuda_stub_mesh_publish_calls == published,
            "retained destruction cannot acquire a producer or launch unowned work");
    }
    cuda_stub_host_unregister_result = 0;
    SparkTpDeviceCollectiveDestroy(&owners[1]);
    CHECK(owners[1].implementation == 0 && spark_stub_cuda_host_registered(g_regions[0]) == 0u,
        "last shared owner unregisters only after successful retry");
    SparkTpDeviceCollectiveDestroy(&owners[2]);
    CHECK(owners[2].implementation == 0 && spark_stub_cuda_host_registered(g_regions[1]) == 0u,
        "distinct mapping unregisters independently");
    CHECK(cudaHostRegister(g_regions[0],(size_t)SPARK_WEIGHTD_MESH_REGION_BYTES,0u) == 0,
        "external owner registers its mapping");
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&owners[3],g_regions[0],1u,
        FUZZ_HIDDEN,0u,0) == SPARK_STATUS_IO_ERROR,
        "already registered external mapping requires explicit ownership rather than success fallback");
    calls = cuda_stub_host_unregister_calls;
    SparkTpDeviceCollectiveDestroy(&owners[3]);
    CHECK(owners[3].implementation == 0 && cuda_stub_host_unregister_calls == calls &&
        spark_stub_cuda_host_registered(g_regions[0]) != 0u,
        "failed prepare never unregisters an external owner's mapping");
    CHECK(cudaHostUnregister(g_regions[0]) == 0,"external owner can release its own registration");
}

static void FuzzPayloadCapacity(void)
{
    SparkTpDeviceCollectiveConfig config;
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveSubmission submission;
    SparkStatus status;
    FuzzTask *task = &g_tasks[0];
    uint32_t published = cuda_stub_mesh_publish_calls;
    memset(&config,0,sizeof(config));
    memset(&collective,0,sizeof(collective));
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.tp_degree = g_rank_count;
    config.local_hidden_dimension = SPARK_WEIGHTD_MESH_SLOT_BYTES / 2u;
    config.max_active_sequence_count = 1u;
    config.operation_timeout_milli = FUZZ_ROUND_TIMEOUT_MS;
    config.combine_bf16_function = FuzzCombineBf16;
    g_connect_rank_hint = 0u;
    FuzzCase("registered-payload-capacity");
    status = SparkTpDeviceCollectiveCreate(&config,&collective);
    CHECK(status == SPARK_STATUS_OK,"capacity probe creates its explicit geometry");
    if ( status != SPARK_STATUS_OK ) return;
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,g_regions[0],1u,
        config.local_hidden_dimension,0u,0) == SPARK_STATUS_OK,"capacity probe binds its real mesh region");
    task->argument = 1050000u;
    FuzzSubmission(task,&submission);
    CHECK(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_CAPACITY_EXCEEDED,"payload plus sequence trailer cannot exceed registered slot");
    CHECK(cuda_stub_mesh_publish_calls == published &&
        SparkTpDeviceCollectiveRoundIndex(&collective) == 0u,
        "oversized payload rejects before publication or generation consumption");
    SparkTpDeviceCollectiveDestroy(&collective);
    CHECK(task->record->count == 0u,"oversized payload never transfers callback ownership");
}

static void FuzzNumerics(uint64_t request, uint64_t ordinal,uint32_t logical)
{
    uint32_t run[FUZZ_MAX_RANKS], rank, i, op;
    uint32_t count = FuzzAllRanks(run);
    const uint32_t rows[] = {1u,3u,FUZZ_MAX_ROWS};
    for ( op = 0u; op <= SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64; op++ )
    {
        uint32_t row_case;
        for ( row_case = 0u; row_case < sizeof(rows)/sizeof(rows[0]); row_case++ )
        {
            size_t written;
            g_rows = rows[row_case];
            g_logical_rows = logical;
            g_operation = op;
            FuzzCase("sum-max-gather-rows-and-output-bounds");
            for ( rank = 0u; rank < count; rank++ )
            {
                for ( i = 0u; i < FUZZ_ELEMENTS; i++ )
                    g_ranks[rank].partial[i] = FuzzBf16FromFloat((float)(rank + 1u));
                if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 )
                    for ( i = 0u; i < g_rows; i++ )
                    {
                        uint64_t value = ((uint64_t)(rank + 1u) << 48u) + i;
                        memcpy((uint8_t *)g_ranks[rank].partial + i * 8u,&value,8u);
                    }
            }
            CHECK(FuzzRunSet(FuzzChainMain,request++,run,count,"numeric-chain",ordinal,-1),
                "numeric chain establishes a fresh generation");
            CHECK(FuzzRunSet(FuzzRoundMain,ordinal,run,count,"numeric-round",ordinal,-1),
                "numerical operation terminates");
            written = op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 ?
                g_rows * 8u : g_rows * FUZZ_HIDDEN * 2u *
                (op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER ? count : 1u);
            for ( rank = 0u; rank < count; rank++ )
            {
                CHECK(g_tasks[rank].status == SPARK_STATUS_OK,"numerical operation succeeds");
                if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
                    CHECK(FuzzSumOk(rank,ordinal),"all rows match independent sum oracle");
                else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 )
                    for ( i = 0u; i < g_rows; i++ )
                    {
                        uint64_t value;
                        memcpy(&value,(uint8_t *)g_ranks[rank].output + i * 8u,8u);
                        CHECK(value == ((uint64_t)count << 48u) + i,"max preserves full 64-bit candidate");
                    }
                else
                {
                    uint32_t correct = 1u;
                    for ( i = 0u; i < count * g_rows * FUZZ_HIDDEN; i++ )
                        correct &= g_ranks[rank].output[i] == FuzzBf16FromFloat(
                            (float)(1u + i / (g_rows * FUZZ_HIDDEN)));
                    CHECK(correct,"gather preserves rank and row layout");
                }
                {
                    uint32_t intact = 1u;
                    for ( i = (uint32_t)written; i < sizeof(g_ranks[rank].output); i++ )
                        intact &= ((uint8_t *)g_ranks[rank].output)[i] == 0xffu;
                    CHECK(intact,"operation never overwrites output capacity");
                }
            }
            ordinal++;
        }
    }
    g_rows = 1u;
    g_operation = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    for ( rank = 0u; rank < count; rank++ )
        for ( i = 0u; i < FUZZ_ELEMENTS; i++ )
            g_ranks[rank].partial[i] = FuzzBf16FromFloat((float)(rank + 1u));
}

static void FuzzActivityLifetime(void)
{
    uint32_t run[FUZZ_MAX_RANKS];
    uint32_t count = FuzzAllRanks(run),rank,calls,begins,ends;
    uint64_t broadcasts;
    FuzzCase("mesh-activity-terminal-stream-ownership");
    calls = g_activity_calls;
    broadcasts = g_broadcast_count;
    cuda_stub_stream_query_result = 600;
    CHECK(SparkTpDeviceCollectiveEndChain(&g_ranks[0].collective,(void *)0x1) ==
        SPARK_STATUS_BUSY,"pending GPU work cannot end mesh activity");
    CHECK(SparkTpDeviceCollectiveChainKey(&g_ranks[0].collective,2400000u) ==
        SPARK_STATUS_BUSY,"pending GPU work cannot rearm cancellation");
    CHECK(SparkTpDeviceCollectiveGraphPreLaunch(&g_ranks[0].collective,(void *)0x2) ==
        SPARK_STATUS_BUSY,"another stream cannot inherit pending collective ownership");
    {
        void *owned = g_ranks[0].collective.implementation;
        SparkTpDeviceCollectiveDestroy(&g_ranks[0].collective);
        CHECK(g_ranks[0].collective.implementation == owned,
            "destroy retains collective memory and connection while GPU work is pending");
    }
    CHECK(g_activity_calls == calls && g_broadcast_count == broadcasts,
        "rejected rearm and end perform no mesh RPC or broadcast");
    CHECK(spark_stub_cuda_host_registered(g_regions[0]) != 0u,
        "pending GPU work retains its CUDA host registration");
    cuda_stub_stream_query_result = 719;
    CHECK(SparkTpDeviceCollectiveEndChain(&g_ranks[0].collective,(void *)0x1) ==
        SPARK_STATUS_IO_ERROR,"failed GPU stream retains mesh activity");
    cuda_stub_stream_query_result = 0;
    CHECK(SparkTpDeviceCollectiveEndChain(&g_ranks[0].collective,(void *)0x2) ==
        SPARK_STATUS_INVALID_ARGUMENT,"unrelated terminal stream cannot release ownership");
    g_activity_fail = 1u;
    CHECK(SparkTpDeviceCollectiveEndChain(&g_ranks[0].collective,(void *)0x1) ==
        SPARK_STATUS_IO_ERROR,"failed end acknowledgement retains ownership");
    {
        void *owned = g_ranks[0].collective.implementation;
        SparkTpDeviceCollectiveDestroy(&g_ranks[0].collective);
        CHECK(g_ranks[0].collective.implementation == owned,
            "destroy retains collective memory and connection if activity end is unacknowledged");
    }
    g_activity_fail = 0u;
    ends = g_activity_ends;
    for ( rank = 0u; rank < count; rank++ )
        CHECK(SparkTpDeviceCollectiveEndChain(&g_ranks[rank].collective,(void *)0x1) ==
            SPARK_STATUS_OK,"terminal stream ends exact active generation");
    CHECK(g_activity_ends == ends + count,"all rank activity intervals end once");
    calls = g_activity_calls;
    CHECK(SparkTpDeviceCollectiveEndChain(&g_ranks[0].collective,(void *)0x1) ==
        SPARK_STATUS_OK && g_activity_calls == calls,"repeated local end sends no duplicate RPC");
    g_activity_fail = 1u;
    CHECK(SparkTpDeviceCollectiveChainKey(&g_ranks[0].collective,2400000u) ==
        SPARK_STATUS_IO_ERROR && g_broadcast_count == broadcasts,
        "failed start acknowledgement cannot publish a chain");
    g_activity_fail = 0u;
    begins = g_activity_begins;
    CHECK(FuzzRunSet(FuzzChainMain,2400001u,run,count,"activity-chain",0u,-1),
        "next activity generation starts on every rank");
    CHECK(g_activity_begins == begins + count,"each rank acknowledges one start");
    CHECK(FuzzRunSet(FuzzRoundMain,2400002u,run,count,"activity-round",0u,-1),
        "numerical work completes in acknowledged interval");
    for ( rank = 0u; rank < count; rank++ )
    {
        CHECK(g_tasks[rank].status == SPARK_STATUS_OK && FuzzSumOk(rank,0u),
            "activity lifecycle preserves all rank contributions");
        if ( rank != 0u )
            CHECK(SparkTpDeviceCollectiveEndChain(&g_ranks[rank].collective,(void *)0x1) ==
                SPARK_STATUS_OK,"reused generation ends only after numerical work");
    }
    CHECK(g_activity_begins == begins + count,"round submission does not duplicate acknowledged start");
    CHECK(SparkTpDeviceCollectiveGraphPreLaunch(&g_ranks[0].collective,0) ==
        SPARK_STATUS_OK,"terminal ownership may transfer explicitly to default CUDA stream");
    cuda_stub_stream_query_result = 600;
    CHECK(SparkTpDeviceCollectiveEndChain(&g_ranks[0].collective,0) ==
        SPARK_STATUS_BUSY,"default CUDA stream also requires terminal query");
    cuda_stub_stream_query_result = 0;
    ends = g_activity_ends;
    SparkTpDeviceCollectiveDestroy(&g_ranks[0].collective);
    CHECK(g_ranks[0].collective.implementation == 0 && g_activity_ends == ends + 1u,
        "terminal standalone caller destruction acknowledges end before closing connection");
    FuzzResetRank(0u);
}

static void FuzzMandatoryFaults(void)
{
    uint32_t run[FUZZ_MAX_RANKS], peers[FUZZ_MAX_RANKS];
    uint32_t count = FuzzAllRanks(run), rank, scenario;
    uint64_t request = 1100000u;
    FuzzConfiguration();
    FuzzRejections();
    FuzzCompletionCapacity();
    FuzzPayloadCapacity();
    FuzzCase("chain-broadcast-failure");
    g_broadcast_fail = 1u;
    CHECK(SparkTpDeviceCollectiveChainKey(&g_ranks[0].collective,request++) ==
        SPARK_STATUS_IO_ERROR,"failed generation broadcast cannot report ready");
    g_broadcast_fail = 0u;
    for ( scenario = 0u; scenario < 5u; scenario++ )
    {
        uint32_t victim = (g_seed + scenario) % count;
        uint32_t peer_count = 0u;
        uint64_t stale;
        uint64_t ordinal = 1200000u + 32u * scenario;
        CHECK(FuzzRunSet(FuzzChainMain,request++,run,count,"fault-chain",scenario,-1),
            "fault scenario chain completes");
        for ( rank = 0u; rank < count; rank++ )
            CHECK(g_tasks[rank].status == SPARK_STATUS_OK,"fault chain ready on all ranks");
        if ( scenario == 0u || scenario == 1u )
        {
            FuzzCase(scenario == 0u ? "missing-peer-stale-slot" : "daemon-loss-before-ack");
            for ( rank = 0u; rank < count; rank++ )
                if ( rank != victim ) peers[peer_count++] = rank;
            stale = *(uint64_t *)(g_regions[peers[0]] +
                (uint64_t)victim * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK *
                SPARK_WEIGHTD_MESH_SLOT_BYTES + SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u);
            CHECK(stale != 0u,"missing-peer scenario retains a prior generation payload");
            if ( scenario == 1u )
                g_dead_rank_mask = (1u << count) - 1u;
            CHECK(FuzzRunSet(FuzzRoundMain,ordinal,peers,peer_count,"missing-peer",scenario,-1),
                "missing peer terminates within watchdog");
            for ( rank = 0u; rank < peer_count; rank++ )
                CHECK(g_tasks[peers[rank]].status == (scenario == 0u ?
                    SPARK_STATUS_BUSY : SPARK_STATUS_IO_ERROR),
                    "stale payload never makes missing or disconnected peer successful");
            g_dead_rank_mask = 0u;
        }
        else if ( scenario == 2u )
        {
            FuzzCase("combine-failure-no-success-callback");
            g_combine_fail = 1u;
            CHECK(FuzzRunSet(FuzzRoundMain,ordinal,run,count,"combine-failure",scenario,-1),
                "combine failure terminates");
            for ( rank = 0u; rank < count; rank++ )
                CHECK(g_tasks[rank].status == SPARK_STATUS_IO_ERROR,"combine failure propagates unchanged");
            g_combine_fail = 0u;
        }
        else if ( scenario == 3u )
        {
            FuzzCase("consumed-request-generation");
            for ( rank = 1u; rank < count; rank++ ) peers[peer_count++] = rank;
            CHECK(FuzzRunSet(FuzzChainMain,request - 1u,peers,peer_count,"stale-chain",scenario,-1),
                "replayed chain identity does not wait indefinitely");
            for ( rank = 0u; rank < peer_count; rank++ )
                CHECK(g_tasks[peers[rank]].status == SPARK_STATUS_BUSY,
                    "consumed generation requires a new leader epoch");
        }
        else
        {
            FuzzCase("retire-recreate-reuse");
            FuzzResetRank(victim);
        }
        CHECK(FuzzRunSet(FuzzChainMain,request++,run,count,"recovery-chain",scenario,-1),
            "fresh coordinated generation follows every failure");
        CHECK(FuzzRunSet(FuzzRoundMain,ordinal + 1u,run,count,"recovery-round",scenario,-1),
            "recovery round completes");
        for ( rank = 0u; rank < count; rank++ )
        {
            CHECK(g_tasks[rank].status == SPARK_STATUS_OK,"recovered rank produces a successful reduction");
            CHECK(FuzzSumOk(rank,scenario),"recovered output contains every fresh contribution");
        }
    }
    FuzzNumerics(1300000u,1400000u,1u);
}

static void FuzzLegacyCallbacks(uint32_t enabled);

static void FuzzTreeCases(void)
{
    uint32_t run[FUZZ_MAX_RANKS], peers[FUZZ_MAX_RANKS];
    uint32_t count = FuzzAllRanks(run),rank,n = 0u;
    FuzzCase("tree-native-operations-without-legacy-callbacks");
    FuzzLegacyCallbacks(0u);
    g_logical_rows = 3u;
    FuzzNumerics(1600000u,1700000u,3u);
    g_rows = 1u;
    g_logical_rows = 3u;
    FuzzGraphPath();
    FuzzCase("tree-fp32-intermediate-rounding");
    for ( rank = 0u; rank < count; rank++ )
        g_ranks[rank].partial[1] = FuzzBf16FromFloat(rank == 0u ? 256.0f :
            rank == 1u ? 1.0f : rank == 2u ? -256.0f : rank == 3u ? 1.0f : 0.0f);
    CHECK(FuzzRunSet(FuzzChainMain,1800000u,run,count,"tree-rounding-chain",0u,-1),"rounding generation aligns");
    CHECK(FuzzRunSet(FuzzRoundMain,1800001u,run,count,"tree-rounding",0u,-1),"tree rounding operation terminates");
    for ( rank = 0u; rank < count; rank++ )
    {
        CHECK(g_tasks[rank].status == SPARK_STATUS_OK,"FP32 partial reduction succeeds");
        CHECK(g_ranks[rank].output[1] == FuzzBf16FromFloat(count == 2u ? 257.0f : count == 3u ? 1.0f : 2.0f),
            "intermediate cancellation keeps low bits until final BF16 conversion");
        for ( uint32_t i = 0u; i < FUZZ_ELEMENTS; i++ )
            g_ranks[rank].partial[i] = FuzzBf16FromFloat((float)(rank + 1u));
    }
    FuzzCase("tree-missing-peer-cancel-and-recovery");
    CHECK(FuzzRunSet(FuzzChainMain,1800002u,run,count,"tree-missing-chain",0u,-1),"missing-peer generation aligns");
    for ( rank = 0u; rank < count; rank++ ) if ( rank != count - 1u ) peers[n++] = rank;
    CHECK(FuzzRunSet(FuzzRoundMain,1800003u,peers,n,"tree-missing",0u,-1),"missing tree peer terminates bounded");
    for ( rank = 0u; rank < n; rank++ )
        CHECK(g_tasks[peers[rank]].status == SPARK_STATUS_BUSY,"missing tree contribution cannot complete successfully");
    CHECK(FuzzRunSet(FuzzChainMain,1800004u,run,count,"tree-loop-chain",0u,-1),"tree round-loop generation aligns");
    CHECK(FuzzLaunchRounds(1800005u,3u,1u,"tree-loop"),"tree round-loop completes");
    for ( rank = 0u; rank < count; rank++ )
    {
        CHECK(g_tasks[rank].status == SPARK_STATUS_OK,"B3 selects tree for every device round");
        CHECK(FuzzSumOk(rank,1u),"tree device rounds retain every contribution");
    }
    FuzzHoldShipper(1u);
    CHECK(FuzzRunSet(FuzzChainMain,1800006u,run,count,"tree-cancel-chain",0u,-1),"tree cancellation generation aligns");
    FuzzLaunchRoundsAsync(1800007u,3u,2u);
    usleep(50000);
    FuzzCancelAll();
    CHECK(FuzzJoinRounds("tree-cancel",2u),"cancelled tree terminates bounded");
    for ( rank = 0u; rank < count; rank++ )
        CHECK(g_tasks[rank].status == SPARK_STATUS_BUSY,"cancelled tree never reports success");
    FuzzHoldShipper(0u);
    CHECK(FuzzRunSet(FuzzChainMain,1800008u,run,count,"tree-recover-chain",0u,-1),"tree recovery generation aligns");
    CHECK(FuzzRunSet(FuzzRoundMain,1800009u,run,count,"tree-recover",3u,-1),"tree recovers after cancellation");
    for ( rank = 0u; rank < count; rank++ )
    {
        CHECK(g_tasks[rank].status == SPARK_STATUS_OK,"recovered tree succeeds");
        CHECK(FuzzSumOk(rank,3u),"recovered tree contains fresh values");
    }
    FuzzLegacyCallbacks(1u);
    g_logical_rows = 1u;
}

static void FuzzTreeLargePayloads(void)
{
    uint32_t run[FUZZ_MAX_RANKS],count = FuzzAllRanks(run),rank,operation;
    uint64_t per_rank = (uint64_t)4u * (SPARK_WEIGHTD_MESH_ROW_BYTES_MAX / 2u);
    uint64_t output_elements = per_rank * count;
    FuzzCase("tree-real-slot-boundary-sum-max-gather");
    g_test_hidden = SPARK_WEIGHTD_MESH_ROW_BYTES_MAX / 2u;
    g_test_timeout_ms = 3000u;
    g_rows = 4u;
    g_logical_rows = 3u;
    for ( rank = 0u; rank < count; rank++ )
    {
        g_input_override[rank] = malloc((size_t)per_rank * 2u);
        g_output_override[rank] = malloc((size_t)output_elements * 2u + 16u);
        if ( g_input_override[rank] == 0 || g_output_override[rank] == 0 ) _Exit(2);
        FuzzResetRank(rank);
    }
    for ( operation = 0u; operation <= 2u; operation++ )
    {
        uint64_t written = operation == 2u ? g_rows * 8u :
            (operation == 0u ? output_elements : per_rank) * 2u;
        g_operation = operation;
        for ( rank = 0u; rank < count; rank++ )
        {
            for ( uint64_t i = 0u; i < per_rank; i++ )
                ((uint16_t *)g_input_override[rank])[i] = FuzzBf16FromFloat((float)(rank + 1u + i % 17u));
            if ( operation == 2u )
                for ( uint32_t i = 0u; i < g_rows; i++ )
                    ((uint64_t *)g_input_override[rank])[i] = ((uint64_t)(rank + 1u) << 48u) + i;
            memset(g_output_override[rank],0xa5,(size_t)output_elements * 2u + 16u);
        }
        CHECK(FuzzRunSet(FuzzChainMain,2300000u + operation,run,count,"large-chain",operation,-1),
            "large payload generation aligns");
        CHECK(FuzzRunSet(FuzzRoundMain,2300010u + operation,run,count,"large-tree",operation,-1),
            "large tree payload finishes bounded");
        for ( rank = 0u; rank < count; rank++ )
        {
            uint32_t correct = 1u,intact = 1u;
            CHECK(g_tasks[rank].status == SPARK_STATUS_OK,"large logical batch uses chunked tree successfully");
            if ( operation == 2u )
                for ( uint32_t i = 0u; i < g_rows; i++ )
                    correct &= ((uint64_t *)g_output_override[rank])[i] == ((uint64_t)count << 48u) + i;
            else
                for ( uint64_t i = 0u; i < written / 2u; i++ )
                {
                    float expected = operation == 1u ?
                        (float)(count * (count + 1u) / 2u + count * (i % 17u)) :
                        (float)(1u + i / per_rank + (i % per_rank) % 17u);
                    correct &= ((uint16_t *)g_output_override[rank])[i] == FuzzBf16FromFloat(expected);
                }
            for ( uint64_t i = written; i < output_elements * 2u + 16u; i++ )
                intact &= ((uint8_t *)g_output_override[rank])[i] == 0xa5u;
            CHECK(correct,"all chunk boundaries preserve independent numerical and rank-layout oracle");
            CHECK(intact,"large tree never writes beyond the result extent");
        }
    }
    g_test_hidden = FUZZ_HIDDEN;
    g_test_timeout_ms = FUZZ_ROUND_TIMEOUT_MS;
    g_rows = 1u;
    g_logical_rows = 1u;
    g_operation = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    for ( rank = 0u; rank < count; rank++ )
    {
        free(g_input_override[rank]); free(g_output_override[rank]);
        g_input_override[rank] = 0; g_output_override[rank] = 0;
        FuzzResetRank(rank);
    }
}

static void FuzzTreeQualification(void)
{
    uint32_t run[FUZZ_MAX_RANKS], rank;
    uint32_t count = FuzzAllRanks(run);
    uint64_t before;
    FuzzCase("logical-batch-requires-tree-fanout");
    g_rows = 1u;
    g_logical_rows = 3u;
    CHECK(FuzzRunSet(FuzzChainMain,1500000u,run,count,"tree-chain",0u,-1),
        "logical batch generation establishes");
    before = __sync_add_and_fetch(&g_payload_deliveries,0u);
    CHECK(FuzzRunSet(FuzzRoundMain,1500001u,run,count,"tree-required",0u,-1),
        "split B3 reduction terminates");
    for ( rank = 0u; rank < count; rank++ )
    {
        CHECK(g_tasks[rank].status == SPARK_STATUS_OK,"split B3 is functionally supported");
        CHECK(FuzzSumOk(rank,0u),"split B3 contains every contribution");
    }
    fprintf(stderr,"POLICY logical=3 execution_rows=1 ranks=%u payload_deliveries=%llu tree_limit=%u\n",
        count,(unsigned long long)(__sync_add_and_fetch(&g_payload_deliveries,0u) - before),2u * (count - 1u));
    CHECK(__sync_add_and_fetch(&g_payload_deliveries,0u) - before <= 2u * (count - 1u),
        "I36/I50 split logical B3 must use bounded tree reduction instead of all-peer broadcast");
    g_logical_rows = 1u;
}

static uint32_t FuzzParseU32(const char *text, uint32_t *value)
{
    char *end;
    unsigned long long parsed;
    errno = 0;
    if ( text[0] == '\0' || text[0] == '-' ) return 0u;
    parsed = strtoull(text,&end,10);
    if ( errno != 0 || *end != '\0' || parsed > UINT32_MAX ) return 0u;
    *value = (uint32_t)parsed;
    return 1u;
}

static int uint64_cmp(const void *a, const void *b)
{
	uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return( x < y ? -1 : x > y ? 1 : 0 );
}

static uint32_t BenchRun(uint32_t rounds)
{
	uint32_t run[FUZZ_MAX_RANKS];
	uint32_t run_count = FuzzAllRanks(run);
	uint64_t *times;
	uint64_t start,total_begin,ordinal;
	uint32_t round,i;
	uint32_t failures_before = test_failures;
	times = (uint64_t *)calloc(rounds != 0u ? rounds : 1u,sizeof(uint64_t));
	if ( times == 0 )
		return(1u);
	ordinal = 1ull;
	uint64_t request = 900000u;
	request++;
	if ( FuzzRunSet(FuzzChainMain, request, run, run_count, "bench-chain",
	        0u, -1) == 0u )
	{
		fprintf(stderr,"bench chain key failed\n");
		free(times);
		return(1u);
	}
	for ( round = 0u; round < rounds; round++ )
	{
		ordinal = 16ull * (uint64_t)round + 1ull;
		start = FuzzNowNs();
		if ( FuzzRunSet(FuzzRoundMain, ordinal, run, run_count, "bench",
		        round, -1) == 0u )
		{
			fprintf(stderr,"bench round %u failed\n",(unsigned)round);
			free(times);
			return(1u);
		}
		times[round] = FuzzNowNs() - start;
		for ( i = 0u; i < run_count; i++ )
		{
			CHECK( g_tasks[run[i]].status == SPARK_STATUS_OK,
			    "bench round status ok" );
			CHECK( FuzzSumOk(run[i], round) != 0u,
			    "bench round sum correct" );
		}
	}
	(void)total_begin;
	qsort(times,rounds,sizeof(uint64_t),uint64_cmp);
	{
		double p50 = (double)times[rounds/2u] / 1000.0;
		double p99 = (double)times[(rounds * 99u) / 100u] / 1000.0;
		double total_s = 0.0;
		for ( i = 0u; i < rounds; i++ )
			total_s += (double)times[i];
		total_s /= 1000000000.0;
		fprintf(stderr,"bench: %u rounds, min=%.1fus p50=%.1fus p99=%.1fus max=%.1fus, %.0f rounds/s\n",
		    rounds,(double)times[0]/1000.0,p50,p99,
		    (double)times[rounds-1u]/1000.0,
		    total_s > 0.0 ? (double)rounds/total_s : 0.0);
	}
	free(times);
	return( test_failures != failures_before ? 1u : 0u );
}

int main(int argc, char **argv)
{
	uint32_t ranks = 4u;
	uint32_t fuzz_rounds = 16u;
	uint32_t bench_rounds = 0u;
	uint32_t seed = 12345u;
    uint32_t seeds = 1u;
    uint32_t qualify_tree = 0u;
	uint32_t kill_percent = 40u;
	uint32_t rank;
	uint32_t wedge;
	pthread_t shipper;
	int arg;
    for ( arg = 1; arg < argc; arg++ )
    {
        uint32_t *value = 0;
        if ( strcmp(argv[arg],"--qualify-tree") == 0 )
        {
            qualify_tree = 1u;
            continue;
        }
        if ( strcmp(argv[arg],"--ranks") == 0 ) value = &ranks;
        else if ( strcmp(argv[arg],"--fuzz") == 0 ) value = &fuzz_rounds;
        else if ( strcmp(argv[arg],"--bench") == 0 ) value = &bench_rounds;
        else if ( strcmp(argv[arg],"--seed") == 0 ) value = &seed;
        else if ( strcmp(argv[arg],"--seeds") == 0 ) value = &seeds;
        else if ( strcmp(argv[arg],"--kill-percent") == 0 ) value = &kill_percent;
        if ( value == 0 || arg + 1 >= argc || FuzzParseU32(argv[++arg],value) == 0u )
        {
            fprintf(stderr,"usage: %s [--ranks 2..16] [--fuzz 0..1024] [--bench 1..4096] [--seed 1..4294967295] [--seeds 1..32] [--kill-percent 0..100] [--qualify-tree]\n",argv[0]);
            return 2;
        }
    }
    if ( ranks < 2u || ranks > FUZZ_MAX_RANKS || seed == 0u ||
         seeds == 0u || seeds > 32u || seed > UINT32_MAX - seeds + 1u ||
         fuzz_rounds > 1024u || fuzz_rounds * seeds > 4096u ||
         bench_rounds > 4096u || kill_percent > 100u ||
         (qualify_tree != 0u && (ranks < 3u || bench_rounds != 0u)) )
    {
        fprintf(stderr,"invalid or unbounded collective fuzz schedule\n");
        return 2;
    }
    g_seed = seed;
	(void)setenv("SPARK_WEIGHTD_SOCKET","/tmp/tp_allreduce_fuzz.sock",1);
	g_rank_count = ranks;
	for ( rank = 0u; rank < ranks; rank++ )
	{
		g_regions[rank] = (uint8_t *)calloc(1u,
		    (size_t)SPARK_WEIGHTD_MESH_REGION_BYTES);
		if ( g_regions[rank] == 0 )
		{
			fprintf(stderr,"mesh region alloc failed rank=%u\n", rank);
			return(1);
		}
	}
    FuzzRegistrationOwnership();
    FuzzLaneNamespace();
    FuzzMeshTopology();
    FuzzCellInitialization();
	if ( pthread_create(&shipper, 0, FuzzShipperMain, 0) != 0 )
	{
		fprintf(stderr,"shipper start failed\n");
		return(1);
	}
	for ( rank = 0u; rank < ranks; rank++ )
    {
        SparkStatus status = FuzzCreateRank(rank);
        CHECK(status == SPARK_STATUS_OK,"rank create");
        if ( status != SPARK_STATUS_OK )
        {
            fprintf(stderr,"SETUP FAIL rank=%u status=%d\n",rank,status);
            _Exit(2);
        }
    }
	for ( rank = 0u; rank < ranks; rank++ )
		g_tasks[rank].rank = &g_ranks[rank];
	FuzzGraphPath();
	wedge = 0u;
	if ( bench_rounds != 0u )
		wedge = BenchRun(bench_rounds);
    else
    {
        uint32_t schedule;
        FuzzBasic();
        FuzzGraphPath();
        FuzzS25S3();
        FuzzSourceLifetime();
        FuzzMandatoryFaults();
        FuzzActivityLifetime();
        FuzzTreeCases();
        FuzzTreeLargePayloads();
        for ( schedule = 0u; schedule < seeds; schedule++ )
            wedge |= FuzzRun(fuzz_rounds,seed + schedule,kill_percent);
        if ( qualify_tree != 0u || bench_rounds == 0u ) FuzzTreeQualification();
    }
	g_shipper_stop = 1u;
	pthread_join(shipper, 0);
	for ( rank = 0u; rank < ranks; rank++ )
	{
		if ( g_ranks[rank].collective.implementation != 0 )
			SparkTpDeviceCollectiveDestroy(&g_ranks[rank].collective);
        {
            uint32_t record;
            for ( record = 0u; record < g_ranks[rank].record_count; record++ )
            {
                FuzzCompletionRecord *entry = &g_ranks[rank].records[record];
                CHECK(entry->count == entry->expect && entry->wrong == 0u,
                    "destroy drains each owned callback exactly once without delayed duplicates");
            }
        }
        CHECK(spark_stub_cuda_host_registered(g_regions[rank]) == 0u,
            "rank teardown leaves no stale CUDA registration before mapping release");
		free(g_regions[rank]);
	}
    fprintf(stderr,"COVERAGE mandatory_cases=%u seeds=%u numerical=host-stub tree-policy=host-qualified gpu-lifetime=unqualified\n",g_cases,seeds);
	CHECK( g_rejected_completion_count == 0u, "synchronous rejection does not queue completion" );
	fprintf(stderr,"test_tp_allreduce_fuzz: %u checks, %u failures (broadcasts=%llu)\n",
	    test_checks, test_failures, (unsigned long long)g_broadcast_count);
	return( test_failures != 0u || wedge != 0u ? 1 : 0 );
}

static void *FuzzCollectiveCalloc(size_t count, size_t size)
{
    if ( g_fail_collective_alloc != 0u )
    {
        g_fail_collective_alloc = 0u;
        return 0;
    }
    return calloc(count,size);
}

extern int cudaMalloc(void **pointer,size_t bytes);
extern int cudaFree(void *pointer);
extern int cudaMemset(void *pointer,int value,size_t bytes);
extern int cudaMemsetAsync(void *pointer,int value,size_t bytes,void *stream);
extern int cudaMemcpy(void *destination,const void *source,size_t bytes,int kind);
extern int cudaHostAlloc(void **pointer,size_t bytes,unsigned int flags);
extern uint32_t spark_stub_cuda_outstanding_allocs(void);

static uint32_t g_init_fail_at,g_init_calls,g_init_free_mask,g_init_free_calls;

static int FuzzCudaInitFailure(void)
{
    return g_init_fail_at != 0u && ++g_init_calls == g_init_fail_at;
}

static int FuzzCudaMalloc(void **pointer,size_t bytes)
{
    int result;
    if ( FuzzCudaInitFailure() ) return 2;
    result = cudaMalloc(pointer,bytes);
    if ( result == 0 && g_init_fail_at != 0u ) memset(*pointer,0xa5,bytes);
    return result;
}

static int FuzzCudaFree(void *pointer)
{
    if ( g_init_free_mask != 0u && ++g_init_free_calls <= 2u &&
        (g_init_free_mask & (1u << (g_init_free_calls - 1u))) != 0u ) return 1;
    return cudaFree(pointer);
}

static int FuzzCudaMemset(void *pointer,int value,size_t bytes)
{
    if ( FuzzCudaInitFailure() ) return 2;
    return cudaMemset(pointer,value,bytes);
}

static int FuzzCudaMemsetAsync(void *pointer,int value,size_t bytes,void *stream)
{
    if ( FuzzCudaInitFailure() ) return 2;
    return cudaMemsetAsync(pointer,value,bytes,stream);
}

static int FuzzCudaMemcpy(void *destination,const void *source,size_t bytes,int kind)
{
    if ( FuzzCudaInitFailure() ) return 2;
    return cudaMemcpy(destination,source,bytes,kind);
}

static int FuzzCudaHostAlloc(void **pointer,size_t bytes,unsigned int flags)
{
    if ( FuzzCudaInitFailure() ) return 2;
    return cudaHostAlloc(pointer,bytes,flags);
}

#define calloc FuzzCollectiveCalloc
#define cudaMalloc FuzzCudaMalloc
#define cudaFree FuzzCudaFree
#define cudaMemset FuzzCudaMemset
#define cudaMemsetAsync FuzzCudaMemsetAsync
#define cudaMemcpy FuzzCudaMemcpy
#define cudaHostAlloc FuzzCudaHostAlloc
#include "../ring/transport/tp_device_collective.c"
#undef calloc
#undef cudaMalloc
#undef cudaFree
#undef cudaMemset
#undef cudaMemsetAsync
#undef cudaMemcpy
#undef cudaHostAlloc

static void FuzzLegacyCallbacks(uint32_t enabled)
{
    for (uint32_t rank = 0u; rank < g_rank_count; rank++)
    {
        SparkTpDeviceCollectiveImplementation *implementation = g_ranks[rank].collective.implementation;
        implementation->combine_bf16 = enabled != 0u ? FuzzCombineBf16 : 0;
        implementation->combine_u64_max = enabled != 0u ? FuzzCombineU64 : 0;
        implementation->combine_gather_bf16 = enabled != 0u ? FuzzGather : 0;
    }
}

static void FuzzLaneNamespace(void)
{
    SparkTpDeviceCollectiveConfig config = {0};
    SparkTpDeviceCollective first = {0},second = {0};
    SparkTpDeviceCollectiveImplementation *a,*b;
    uint64_t stride = (uint64_t)SPARK_WEIGHTD_MESH_SLOT_BYTES * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND;
    FuzzCase("shared-lane-namespace-ignores-colliding-logical-identifiers");
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.tp_degree = g_rank_count;
    config.local_hidden_dimension = FUZZ_HIDDEN;
    config.max_active_sequence_count = 1u;
    config.operation_timeout_milli = FUZZ_ROUND_TIMEOUT_MS;
    config.collective_identifier = 1u;
    g_connect_rank_hint = 0u;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&first) == SPARK_STATUS_OK,
        "first logical model creates common namespace");
    config.collective_identifier = 17u;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&second) == SPARK_STATUS_OK,
        "modulo-colliding second model creates common namespace");
    if (first.implementation == 0 || second.implementation == 0) _Exit(2);
    a = first.implementation;
    b = second.implementation;
    CHECK(a->band_base != b->band_base &&
        a->band_base == 2u * ((FuzzMeshClient *)a->lane_client)->lane * stride &&
        b->band_base == 2u * ((FuzzMeshClient *)b->lane_client)->lane * stride,
        "actual production pointers use reserved bands instead of identifier low bits");
    SparkTpDeviceCollectiveDestroy(&first);
    SparkTpDeviceCollectiveDestroy(&second);
}

static void FuzzMeshTopology(void)
{
    SparkWeightdMeshTopology topology;
    SparkTpDeviceCollectiveConfig config = {0};
    SparkTpDeviceCollective root = {0},peer = {0};
    SparkTpDeviceCollectiveImplementation *implementation;
    const char *invalid[] = {"", "4,5,6", "4,5,6,7,8", "4,5,6,6", "4,5,6,16", "4,5,6,-1", "4,5,6, 7", "4,5,6,7x"};
    FuzzCase("explicit-physical-rank-map-and-ordered-chain-metadata");
    CHECK(unsetenv("SPARK_TP_MESH_RANKS") == 0 &&
        SparkTpDeviceCollectiveMeshTopology(3u,4u,&topology) == SPARK_STATUS_OK &&
        topology.physical_ranks[3] == 3u,"default topology is explicit identity");
    for (uint32_t i=0u; i<sizeof(invalid)/sizeof(invalid[0]); i++)
        CHECK(setenv("SPARK_TP_MESH_RANKS",invalid[i],1) == 0 &&
            SparkTpDeviceCollectiveMeshTopology(0u,4u,&topology) == SPARK_STATUS_INVALID_ARGUMENT,
            "malformed explicit rank map has no identity fallback");
    CHECK(setenv("SPARK_TP_MESH_RANKS","4,5,6,7",1) == 0 &&
        SparkTpDeviceCollectiveMeshTopology(2u,4u,&topology) == SPARK_STATUS_OK &&
        topology.rank_count == 4u && topology.local_rank == 2u &&
        topology.physical_ranks[0] == 4u && topology.physical_ranks[2] == 6u,
        "logical TP4 can explicitly use physical hosts4 through7");
    CHECK(setenv("SPARK_TP_MESH_RANKS","0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1",1) == 0 &&
        SparkTpDeviceCollectiveMeshTopology(15u,16u,&topology) == SPARK_STATUS_OK &&
        topology.physical_ranks[1] == 2u && topology.physical_ranks[15] == 1u,
        "full permuted TP16 retains the explicit rank order");
    CHECK(setenv("SPARK_TP_MESH_RANKS","1,0",1) == 0 &&
        setenv("SPARK_WEIGHTD_LANE","7",1) == 0,"select private control-test lane");
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.tp_degree = 2u;
    config.local_hidden_dimension = FUZZ_HIDDEN;
    config.max_active_sequence_count = 1u;
    config.operation_timeout_milli = FUZZ_ROUND_TIMEOUT_MS;
    g_connect_rank_hint = 1u;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&root) == SPARK_STATUS_OK &&
        SparkTpDeviceCollectivePrepareReceiveBf16(&root,g_regions[1],1u,FUZZ_HIDDEN,0u,0) == SPARK_STATUS_OK,
        "logical root binds physical peer1");
    implementation = root.implementation;
    CHECK(implementation != 0 && implementation->physical_peer_mask == 1u &&
        implementation->packed_rank_map == 1u,"CPU broadcast translates logical peers to physical mask");
    config.tp_rank = 1u;
    g_connect_rank_hint = 0u;
    CHECK(SparkTpDeviceCollectiveCreate(&config,&peer) == SPARK_STATUS_OK &&
        SparkTpDeviceCollectivePrepareReceiveBf16(&peer,g_regions[0],1u,FUZZ_HIDDEN,0u,0) == SPARK_STATUS_OK,
        "logical peer binds physical peer0");
    CHECK(SparkTpDeviceCollectiveChainKey(&root,777u) == SPARK_STATUS_OK &&
        SparkTpDeviceCollectiveChainKey(&peer,777u) == SPARK_STATUS_OK,
        "ordered map payload and chain key reach physical peer");
    SparkTpDeviceCollectiveDestroy(&peer);
    CHECK(setenv("SPARK_TP_MESH_RANKS","2,0",1) == 0 &&
        SparkTpDeviceCollectiveCreate(&config,&peer) == SPARK_STATUS_OK &&
        SparkTpDeviceCollectivePrepareReceiveBf16(&peer,g_regions[0],1u,FUZZ_HIDDEN,0u,0) == SPARK_STATUS_OK,
        "construct mismatched peer shape for cross-rank gate");
    CHECK(SparkTpDeviceCollectiveChainKey(&peer,777u) == SPARK_STATUS_INVALID_ARGUMENT,
        "matching request key cannot admit a different root or membership map");
    SparkTpDeviceCollectiveDestroy(&peer);
    SparkTpDeviceCollectiveDestroy(&root);
    CHECK(unsetenv("SPARK_TP_MESH_RANKS") == 0 && unsetenv("SPARK_WEIGHTD_LANE") == 0,
        "clear explicit topology fixture");
}

static void FuzzCellInitialization(void)
{
    SparkTpDeviceCollectiveConfig config;
    uint32_t trial;
    uint64_t empty_arrival[256] = {0u};
    memset(&config,0,sizeof(config));
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.tp_degree = g_rank_count;
    config.local_hidden_dimension = FUZZ_HIDDEN;
    config.max_active_sequence_count = 1u;
    config.operation_timeout_milli = FUZZ_ROUND_TIMEOUT_MS;
    g_connect_rank_hint = 0u;
    for ( trial = 1u; trial <= 11u; trial++ )
    {
        SparkTpDeviceCollective collective = {0};
        SparkTpDeviceCollectiveImplementation *implementation;
        SparkTpMeshRoundControl expected;
        uint32_t baseline = spark_stub_cuda_outstanding_allocs();
        FuzzCase(trial <= 8u ? "cell-initialization-rollback-and-retry" :
            trial <= 10u ? "cell-cleanup-failure-retains-retryable-allocation" :
            "capture-scratch-allocation-fails-before-arming");
        g_round = trial;
        CHECK(SparkTpDeviceCollectiveCreate(&config,&collective) == SPARK_STATUS_OK,
            "initialization fixture creates");
        if ( collective.implementation == 0 ) _Exit(2);
        implementation = collective.implementation;
        CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,g_regions[0],1u,
            FUZZ_HIDDEN,0u,0) == SPARK_STATUS_OK,"initialization fixture prepares mapping");
        implementation->round_seq = 0x1234u;
        implementation->publish_ack_prev = (UINT64_C(37) << 32u) | 19u;
        g_init_fail_at = trial <= 8u ? trial : trial <= 10u ? 5u : UINT32_MAX;
        g_init_calls = 0u;
        g_init_free_calls = 0u;
        g_init_free_mask = trial == 9u ? 1u : trial == 10u ? 2u : 0u;
        if ( trial <= 10u )
        {
            CHECK(SparkTpDeviceCollectiveEnsureCells(implementation) == SPARK_STATUS_IO_ERROR &&
                g_init_calls == g_init_fail_at,"each CUDA initialization failure rejects at its exact phase");
            CHECK(implementation->published_host_cell == 0 && implementation->seq_cell == 0 &&
                implementation->epoch_cell == 0 && implementation->round_seq_device == 0 &&
                implementation->cancel_expected == 0 && implementation->error_word == 0 &&
                implementation->diag_word == 0,"failed initialization never publishes ready aliases");
            CHECK(spark_stub_cuda_outstanding_allocs() == baseline + (trial > 8u ? 1u : 0u),
                "rollback frees all allocations except an explicitly failed free");
            if ( trial > 8u )
            {
                void *retained = trial == 9u ? implementation->round_control : implementation->arrival_ring;
                CHECK(retained != 0,"failed free retains its exact allocation for retry");
                g_init_free_mask = 1u;
                g_init_free_calls = 0u;
                g_init_calls = 0u;
                g_init_fail_at = UINT32_MAX;
                CHECK(SparkTpDeviceCollectiveEnsureCells(implementation) == SPARK_STATUS_IO_ERROR &&
                    g_init_calls == 0u && spark_stub_cuda_outstanding_allocs() == baseline + 1u,
                    "repeated cleanup failure cannot allocate replacements or report ready");
                CHECK(retained == (trial == 9u ? implementation->round_control : implementation->arrival_ring),
                    "cleanup retry cannot lose the retained allocation identity");
            }
        }
        g_init_free_mask = 0u;
        g_init_calls = 0u;
        g_init_fail_at = UINT32_MAX;
        CHECK(SparkTpDeviceCollectiveEnsureCells(implementation) == SPARK_STATUS_OK &&
            g_init_calls == 8u && spark_stub_cuda_outstanding_allocs() == baseline + 3u,
            "successful retry completes all initialization operations with exactly three allocations");
        memset(&expected,0,sizeof(expected));
        expected.seq = 0x1234u;
        expected.round_seq = (UINT64_C(37) << 32u) | 19u;
        CHECK(implementation->round_control != 0 && implementation->arrival_ring != 0 &&
            implementation->published_host_cell != 0 &&
            memcmp(implementation->round_control,&expected,sizeof(expected)) == 0 &&
            memcmp(implementation->arrival_ring,empty_arrival,sizeof(empty_arrival)) == 0 &&
            implementation->cell_mirror == expected.seq,
            "retry initializes poisoned allocations to complete control and arrival values");
        CHECK(SparkTpDeviceCollectiveEnsureCells(implementation) == SPARK_STATUS_OK &&
            g_init_calls == 8u && spark_stub_cuda_outstanding_allocs() == baseline + 3u,
            "fully initialized cells are reused without new CUDA operations");
        if ( trial == 11u )
        {
            g_init_calls = 0u;
            g_init_fail_at = 3u;
            CHECK(SparkTpDeviceCollectiveArmCapture(&collective) == SPARK_STATUS_CAPACITY_EXCEEDED &&
                g_init_calls == 3u && implementation->capture_armed == 0u &&
                implementation->f32_scratch == 0 && implementation->f32_scratch_bytes == 0u &&
                spark_stub_cuda_outstanding_allocs() == baseline + 3u,
                "capture scratch OOM leaves capture unarmed and preserves ready cells");
            g_init_fail_at = 0u;
            CHECK(SparkTpDeviceCollectiveArmCapture(&collective) == SPARK_STATUS_OK &&
                implementation->capture_armed != 0u && implementation->f32_scratch != 0 &&
                implementation->f32_scratch_bytes == 2u * implementation->slot_bytes,
                "capture retries successfully once scratch allocation is available");
            CHECK(SparkTpDeviceCollectiveDisarmCapture(&collective) == SPARK_STATUS_OK,
                "successful capture preparation can disarm");
        }
        g_init_fail_at = 0u;
        SparkTpDeviceCollectiveDestroy(&collective);
        CHECK(collective.implementation == 0 && spark_stub_cuda_outstanding_allocs() == baseline &&
            spark_stub_cuda_host_registered(g_regions[0]) == 0u,
            "initialization and capture retries release every allocation and mapping on destruction");
    }
}

static void FuzzSourceLifetime(void)
{
    SparkTpMeshRoundControl control;
    volatile uint64_t shipped = (UINT64_C(7) << 32u) | 1u;
    volatile uint64_t cancel = 0u;
    uint8_t source[7] = {1u,2u,3u,4u,5u,6u,7u};
    uint8_t destination[9];
    uint64_t entry[4] = {0u,0u,0u,0u}, tail = 0u;
    FuzzCase("source-copy-full-tag-timeout-cancel-and-byte-bounds");
    memset(&control,0,sizeof(control));
    memset(destination,0xa5,sizeof(destination));
    control.epoch = 8u;
    control.round_seq = (UINT64_C(8) << 32u) | 1u;
    CHECK(SparkGlm5NextLaunchMeshCopyDown(0,destination,source,sizeof(source),
        &shipped,&control,&cancel,1000000u) == 0,
        "source gate records asynchronous timeout in control");
    CHECK(control.error_word == control.round_seq && destination[0] == 0xa5u,
        "same low sequence from stale epoch cannot overwrite a live source");
    CHECK(SparkGlm5NextLaunchMeshPublish(0,entry,&control.seq,&control.epoch,
        &control.round_seq,sizeof(source),0u,2u,&tail,&control.error_word,2u) == 0 &&
        control.seq == 0u && entry[0] == 0u && tail == 0u,
        "failed source gate cannot publish a payload or advance sequence");
    {
        uint64_t guarded[2] = {17u,29u};
        CHECK(SparkGlm5NextLaunchMeshGuard(0,&control.error_word,guarded) == 0 &&
            guarded[0] == UINT64_MAX && guarded[1] == 29u,
            "failed mesh guard poisons only the output word");
        control.error_word = 0u;
        guarded[0] = 17u;
        CHECK(SparkGlm5NextLaunchMeshGuard(0,&control.error_word,guarded) == 0 &&
            guarded[0] == 17u && guarded[1] == 29u,
            "successful mesh guard preserves output");
    }
    shipped = control.round_seq;
    CHECK(SparkGlm5NextLaunchMeshCopyDown(0,destination,source,sizeof(source),
        &shipped,&control,&cancel,1000000u) == 0 &&
        memcmp(destination,source,sizeof(source)) == 0 &&
        destination[7] == 0xa5u && destination[8] == 0xa5u,
        "full tag release copies exact odd byte extent without changing guards");
    cancel = 1u;
    memset(destination,0xa5,sizeof(destination));
    CHECK(SparkGlm5NextLaunchMeshCopyDown(0,destination,source,sizeof(source),
        &shipped,&control,&cancel,1000000u) == 0 && destination[0] == 0xa5u &&
        control.error_word != 0u,
        "cancellation prevents source overwrite and subsequent publication after shipment");
}
