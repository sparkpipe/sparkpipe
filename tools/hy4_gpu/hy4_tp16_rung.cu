#include <atomic>
#include <cuda_runtime.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_hy4_model.h"
#include "sparkpipe/spark_tp_device_collective.h"

#define SPARK_HY4_TP16_RUNG_MESH_SLOT_BYTES (16u * 1024u * 1024u)
#define SPARK_HY4_TP16_RUNG_MESH_SLOTS_PER_BAND 32u
#define SPARK_HY4_TP16_RUNG_MESH_BANDS 4u
#define SPARK_HY4_TP16_RUNG_MESH_BUFFER_BYTES \
	(SPARK_HY4_TP16_RUNG_MESH_SLOT_BYTES * \
	SPARK_HY4_TP16_RUNG_MESH_SLOTS_PER_BAND * \
	SPARK_HY4_TP16_RUNG_MESH_BANDS)
#define SPARK_HY4_TP16_RUNG_MESH_REGION_BYTES \
	(SPARK_HY4_TP16_RUNG_MESH_BUFFER_BYTES + 4096u)
#define SPARK_HY4_TP16_RUNG_MESH_DOORBELL_ENTRY(band,rank) \
	(SPARK_HY4_TP16_RUNG_MESH_BUFFER_BYTES + \
	(((band) * 16u + (rank)) * 24u))
static_assert(4u * 16u * 24u <= 4096u,
	"doorbell entries must fit the doorbell page");

extern "C" cudaError_t SparkHy4LaunchAccumAddBf16(cudaStream_t stream,
	void *destination_bf16,const void *source_bf16,
	uint32_t row_count,uint32_t width);
extern "C" cudaError_t SparkHy4LaunchAccumU64Max(cudaStream_t stream,
	uint64_t *destination,const uint64_t *source,
	uint32_t element_count);

#define SPARK_HY4_TP16_RUNG_RANKS SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE
#define SPARK_HY4_TP16_RUNG_HIDDEN SPARK_HY4_MODEL_HIDDEN_DIMENSION
#define SPARK_HY4_TP16_RUNG_MAX_ROWS 8u
#define SPARK_HY4_TP16_RUNG_BF16_ROUNDS 12u
#define SPARK_HY4_TP16_RUNG_U64_ROUNDS 4u
#define SPARK_HY4_TP16_RUNG_ROUNDS \
	(SPARK_HY4_TP16_RUNG_BF16_ROUNDS + SPARK_HY4_TP16_RUNG_U64_ROUNDS)
#define SPARK_HY4_TP16_RUNG_REL_TOLERANCE 0.06
#define SPARK_HY4_TP16_RUNG_ABS_FLOOR 1e-3
#define SPARK_HY4_TP16_RUNG_BARRIER_TIMEOUT_NS UINT64_C(120000000000)
#define SPARK_HY4_TP16_RUNG_IDENTIFIER UINT64_C(1001)
#define SPARK_HY4_TP16_RUNG_BAND \
	(SPARK_HY4_TP16_RUNG_IDENTIFIER & (SPARK_HY4_TP16_RUNG_MESH_BANDS - 1u))

typedef struct SparkHy4Tp16RungRankContext
{
	uint32_t rank;
	uint32_t green;
	uint32_t rounds_done;
	SparkTpDeviceCollective collective;
	cudaStream_t stream;
	cudaEvent_t start_event;
	cudaEvent_t stop_event;
	void *input_device;
	void *work_device;
	uint64_t *u64_input_device;
	uint64_t *u64_work_device;
	uint16_t *host_input;
	uint16_t *host_result;
	uint64_t *host_u64_input;
	uint64_t *host_u64_result;
	std::atomic<uint32_t> completion_done;
	std::atomic<uint32_t> completion_status;
	double max_rel_delta;
	uint64_t nonzero_checked;
	uint64_t exact_matches;
} SparkHy4Tp16RungRankContext;

typedef struct SparkHy4Tp16RungBarrier
{
	pthread_mutex_t lock;
	pthread_cond_t condition;
	uint32_t generation;
	uint32_t count;
	uint32_t waiting;
	uint32_t failed;
} SparkHy4Tp16RungBarrier;

static SparkHy4Tp16RungBarrier rung_barrier;
static std::atomic<uint32_t> rung_green_ranks(0u);
static uint64_t rung_cutoff_ns;
static FILE *rung_tsv;

static uint64_t SparkHy4Tp16RungNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) +
		(uint64_t)now.tv_nsec);
}

static uint32_t SparkHy4Tp16RungLcg(uint32_t *state)
{
	*state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
	return(*state);
}

static uint16_t SparkHy4Tp16RungBf16Sample(uint32_t *state,uint32_t salt)
{
	uint32_t raw;
	raw = SparkHy4Tp16RungLcg(state);
	return((uint16_t)(((raw >> 17u) & UINT32_C(0x0000007f)) |
		((raw >> 21u) & UINT32_C(0x00008000)) |
		((126u + (salt & 1u)) << 7u)));
}

static float SparkHy4Tp16RungBf16Decode(uint16_t raw)
{
	uint32_t bits = (uint32_t)raw << 16u;
	float value;
	memcpy(&value,&bits,sizeof(value));
	return(value);
}

static uint64_t SparkHy4Tp16RungU64Sample(uint32_t *state)
{
	uint64_t high,low;
	high = (uint64_t)SparkHy4Tp16RungLcg(state) << 32u;
	low = (uint64_t)SparkHy4Tp16RungLcg(state);
	return(high | low);
}

static void *SparkHy4Tp16RungMeshBase(void)
{
	static void *mesh_base;
	if ( mesh_base == 0 )
	{
		mesh_base = malloc(SPARK_HY4_TP16_RUNG_MESH_REGION_BYTES);
		if ( mesh_base != 0 )
			memset(mesh_base,0,SPARK_HY4_TP16_RUNG_MESH_REGION_BYTES);
	}
	return(mesh_base);
}

static SparkStatus SparkHy4Tp16RungCombineBf16(void *combine_context,
	void *destination_device,const void *source_device,
	uint32_t active_sequence_count,uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkHy4LaunchAccumAddBf16((cudaStream_t)cuda_stream,
		destination_device,source_device,active_sequence_count,
		hidden_dimension);
	if ( error != cudaSuccess )
	{
		fprintf(stderr,"TP16RUNG combine_bf16 error %s\n",
			cudaGetErrorString(error));
		return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkHy4Tp16RungCombineU64Max(void *combine_context,
	uint64_t *destination_device,const uint64_t *source_device,
	uint32_t element_count,void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkHy4LaunchAccumU64Max((cudaStream_t)cuda_stream,
		destination_device,source_device,element_count);
	if ( error != cudaSuccess )
	{
		fprintf(stderr,"TP16RUNG combine_u64 error %s\n",
			cudaGetErrorString(error));
		return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static void SparkHy4Tp16RungCompletion(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkHy4Tp16RungRankContext *rank;
	rank = (SparkHy4Tp16RungRankContext *)context;
	rank->completion_status.store((uint32_t)completion->status,
		std::memory_order_release);
	rank->completion_done.store(1u,std::memory_order_release);
}

static void SparkHy4Tp16RungBarrierFail(void)
{
	pthread_mutex_lock(&rung_barrier.lock);
	rung_barrier.failed = 1u;
	pthread_cond_broadcast(&rung_barrier.condition);
	pthread_mutex_unlock(&rung_barrier.lock);
}

static int SparkHy4Tp16RungBarrierWait(void)
{
	struct timespec deadline;
	uint64_t deadline_ns;
	uint32_t generation;
	int outcome;
	deadline_ns = SparkHy4Tp16RungNowNs() +
		SPARK_HY4_TP16_RUNG_BARRIER_TIMEOUT_NS;
	deadline.tv_sec = (time_t)(deadline_ns / UINT64_C(1000000000));
	deadline.tv_nsec = (long)(deadline_ns % UINT64_C(1000000000));
	pthread_mutex_lock(&rung_barrier.lock);
	if ( rung_barrier.failed != 0u )
	{
		pthread_mutex_unlock(&rung_barrier.lock);
		return(1);
	}
	rung_barrier.waiting++;
	if ( rung_barrier.waiting == rung_barrier.count )
	{
		rung_barrier.waiting = 0u;
		rung_barrier.generation++;
		pthread_cond_broadcast(&rung_barrier.condition);
		pthread_mutex_unlock(&rung_barrier.lock);
		return(0);
	}
	generation = rung_barrier.generation;
	outcome = 0;
	while ( rung_barrier.generation == generation &&
		rung_barrier.failed == 0u )
	{
		if ( pthread_cond_timedwait(&rung_barrier.condition,
			&rung_barrier.lock,&deadline) == ETIMEDOUT )
		{
			rung_barrier.failed = 1u;
			pthread_cond_broadcast(&rung_barrier.condition);
			outcome = 1;
			break;
		}
	}
	if ( rung_barrier.failed != 0u )
		outcome = 1;
	pthread_mutex_unlock(&rung_barrier.lock);
	return(outcome);
}

static std::atomic<uint32_t> rung_monitor_stop(0u);

static void *SparkHy4Tp16RungMeshSequencer(void *argument)
{
	static uint64_t last_sequence[SPARK_HY4_TP16_RUNG_RANKS];
	uint8_t *mesh;
	uint64_t band_base;
	(void)argument;
	mesh = (uint8_t *)SparkHy4Tp16RungMeshBase();
	band_base = (uint64_t)SPARK_HY4_TP16_RUNG_BAND *
		SPARK_HY4_TP16_RUNG_MESH_SLOT_BYTES *
		SPARK_HY4_TP16_RUNG_MESH_SLOTS_PER_BAND;
	while ( rung_monitor_stop.load(std::memory_order_relaxed) == 0u )
	{
		uint32_t rank;
		for (rank=0u; rank<SPARK_HY4_TP16_RUNG_RANKS; rank++)
		{
			volatile uint64_t *entry =
				(volatile uint64_t *)(mesh +
				SPARK_HY4_TP16_RUNG_MESH_DOORBELL_ENTRY(
				SPARK_HY4_TP16_RUNG_BAND,rank));
			volatile uint64_t *sequence_word;
			uint64_t sequence,slot;
			sequence = entry[0];
			if ( sequence == 0u ||
				sequence <= last_sequence[rank] )
				continue;
			slot = entry[2];
			if ( slot >= SPARK_HY4_TP16_RUNG_MESH_SLOTS_PER_BAND )
				continue;
			sequence_word = (volatile uint64_t *)(mesh +
				band_base + slot *
				SPARK_HY4_TP16_RUNG_MESH_SLOT_BYTES +
				SPARK_HY4_TP16_RUNG_MESH_SLOT_BYTES - 8u);
			__sync_synchronize();
			*sequence_word = sequence;
			last_sequence[rank] = sequence;
		}
		{
			struct timespec pause = {0,20000};
			nanosleep(&pause,0);
		}
	}
	return(0);
}

static void SparkHy4Tp16RungFillBf16(SparkHy4Tp16RungRankContext *rank,
	uint32_t round,uint32_t rows)
{
	uint32_t state;
	uint32_t row,element;
	state = UINT32_C(0x68bc21) + rank->rank * UINT32_C(7919) +
		round * UINT32_C(104729);
	for (row=0u; row<rows; row++)
		for (element=0u; element<SPARK_HY4_TP16_RUNG_HIDDEN; element++)
			rank->host_input[
				(size_t)row * SPARK_HY4_TP16_RUNG_HIDDEN +
				element] = SparkHy4Tp16RungBf16Sample(&state,
				element);
}

static double SparkHy4Tp16RungVerifyBf16(
	SparkHy4Tp16RungRankContext *rank,uint32_t round,uint32_t rows,
	double *max_rel_out,uint64_t *checked_out,uint64_t *exact_out)
{
	double oracle[SPARK_HY4_TP16_RUNG_MAX_ROWS *
		SPARK_HY4_TP16_RUNG_HIDDEN];
	double worst_rel,absolute,difference,magnitude;
	uint32_t peer,row,element;
	uint64_t checked,exact;
	for (row=0u; row<rows; row++)
		for (element=0u; element<SPARK_HY4_TP16_RUNG_HIDDEN; element++)
			oracle[(size_t)row * SPARK_HY4_TP16_RUNG_HIDDEN +
				element] = 0.0;
	for (peer=0u; peer<SPARK_HY4_TP16_RUNG_RANKS; peer++)
	{
		uint32_t peer_state;
		peer_state = UINT32_C(0x68bc21) + peer * UINT32_C(7919) +
			round * UINT32_C(104729);
		for (row=0u; row<rows; row++)
			for (element=0u; element<SPARK_HY4_TP16_RUNG_HIDDEN;
			    element++)
				oracle[(size_t)row *
					SPARK_HY4_TP16_RUNG_HIDDEN + element] +=
					(double)SparkHy4Tp16RungBf16Decode(
					SparkHy4Tp16RungBf16Sample(
					&peer_state,element));
	}
	worst_rel = 0.0;
	checked = 0u;
	exact = 0u;
	for (row=0u; row<rows; row++)
	{
		for (element=0u; element<SPARK_HY4_TP16_RUNG_HIDDEN; element++)
		{
			size_t offset = (size_t)row *
				SPARK_HY4_TP16_RUNG_HIDDEN + element;
			if ( oracle[offset] == 0.0 )
				continue;
			difference = (double)SparkHy4Tp16RungBf16Decode(
				rank->host_result[offset]) - oracle[offset];
			absolute = difference < 0.0 ? -difference : difference;
			magnitude = oracle[offset] < 0.0 ? -oracle[offset] :
				oracle[offset];
			if ( magnitude < SPARK_HY4_TP16_RUNG_ABS_FLOOR )
				magnitude = SPARK_HY4_TP16_RUNG_ABS_FLOOR;
			if ( absolute > SPARK_HY4_TP16_RUNG_REL_TOLERANCE *
				magnitude )
				return(-1.0);
			if ( absolute / magnitude > worst_rel )
				worst_rel = absolute / magnitude;
			if ( difference == 0.0 )
				exact++;
			checked++;
		}
	}
	*max_rel_out = worst_rel;
	*checked_out = checked;
	*exact_out = exact;
	return(worst_rel);
}

static int SparkHy4Tp16RungVerifyU64(
	SparkHy4Tp16RungRankContext *rank,uint32_t round,uint32_t rows)
{
	uint64_t expected[SPARK_HY4_TP16_RUNG_MAX_ROWS];
	uint32_t peer,row;
	for (row=0u; row<rows; row++)
		expected[row] = 0u;
	for (peer=0u; peer<SPARK_HY4_TP16_RUNG_RANKS; peer++)
	{
		uint32_t peer_state;
		uint32_t element;
		peer_state = UINT32_C(0x9e3779) + peer * UINT32_C(15731) +
			round * UINT32_C(31337);
		for (element=0u; element<rows; element++)
		{
			uint64_t sample = SparkHy4Tp16RungU64Sample(&peer_state);
			if ( sample > expected[element] )
				expected[element] = sample;
		}
	}
	for (row=0u; row<rows; row++)
		if ( rank->host_u64_result[row] != expected[row] )
			return(1);
	return(0);
}

static int SparkHy4Tp16RungRankSetup(SparkHy4Tp16RungRankContext *rank)
{
	SparkTpDeviceCollectiveConfig configuration;
	SparkStatus status;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind =
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration.tp_degree = SPARK_HY4_TP16_RUNG_RANKS;
	configuration.tp_rank = rank->rank;
	configuration.operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = SPARK_HY4_TP16_RUNG_ROUNDS;
	configuration.local_hidden_dimension = SPARK_HY4_TP16_RUNG_HIDDEN;
	configuration.max_active_sequence_count =
		SPARK_HY4_TP16_RUNG_MAX_ROWS;
	configuration.connect_timeout_milli = 60000u;
	configuration.operation_timeout_milli = 60000u;
	configuration.control_port_base = 60400u;
	configuration.collective_identifier = SPARK_HY4_TP16_RUNG_IDENTIFIER;
	configuration.backend_module_path = "tp_hidden_transport";
	configuration.combine_bf16_function = SparkHy4Tp16RungCombineBf16;
	configuration.combine_u64_max_function =
		SparkHy4Tp16RungCombineU64Max;
	configuration.combine_context = rank;
	status = SparkTpDeviceCollectiveCreate(&configuration,
		&rank->collective);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"TP16RUNG create rank=%u status=%d\n",
			(unsigned)rank->rank,(int)status);
		return(1);
	}
	rank->host_input = (uint16_t *)malloc(
		(size_t)SPARK_HY4_TP16_RUNG_MAX_ROWS *
		SPARK_HY4_TP16_RUNG_HIDDEN * 2u);
	rank->host_result = (uint16_t *)malloc(
		(size_t)SPARK_HY4_TP16_RUNG_MAX_ROWS *
		SPARK_HY4_TP16_RUNG_HIDDEN * 2u);
	rank->host_u64_input = (uint64_t *)malloc(
		(size_t)SPARK_HY4_TP16_RUNG_MAX_ROWS * sizeof(uint64_t));
	rank->host_u64_result = (uint64_t *)malloc(
		(size_t)SPARK_HY4_TP16_RUNG_MAX_ROWS * sizeof(uint64_t));
	if ( rank->host_input == 0 || rank->host_result == 0 ||
		rank->host_u64_input == 0 || rank->host_u64_result == 0 )
	{
		fprintf(stderr,"TP16RUNG host_alloc rank=%u\n",
			(unsigned)rank->rank);
		return(1);
	}
	if ( cudaStreamCreateWithFlags(&rank->stream,
		cudaStreamNonBlocking) != cudaSuccess ||
		cudaEventCreate(&rank->start_event) != cudaSuccess ||
		cudaEventCreate(&rank->stop_event) != cudaSuccess ||
		cudaMalloc(&rank->input_device,
		(size_t)SPARK_HY4_TP16_RUNG_MAX_ROWS *
		SPARK_HY4_TP16_RUNG_HIDDEN * 2u) != cudaSuccess ||
		cudaMalloc(&rank->work_device,
		(size_t)SPARK_HY4_TP16_RUNG_MAX_ROWS *
		SPARK_HY4_TP16_RUNG_HIDDEN * 2u) != cudaSuccess ||
		cudaMalloc((void **)&rank->u64_input_device,
		(size_t)SPARK_HY4_TP16_RUNG_MAX_ROWS *
		sizeof(uint64_t)) != cudaSuccess ||
		cudaMalloc((void **)&rank->u64_work_device,
		(size_t)SPARK_HY4_TP16_RUNG_MAX_ROWS *
		sizeof(uint64_t)) != cudaSuccess )
	{
		fprintf(stderr,"TP16RUNG cuda_setup rank=%u\n",
			(unsigned)rank->rank);
		return(1);
	}
	status = SparkTpDeviceCollectivePrepareReceiveBf16(
		&rank->collective,SparkHy4Tp16RungMeshBase(),0u,0u,0u,0u);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"TP16RUNG prepare rank=%u status=%d\n",
			(unsigned)rank->rank,(int)status);
		return(1);
	}
	return(0);
}

static void *SparkHy4Tp16RungRankThread(void *argument)
{
	SparkHy4Tp16RungRankContext *rank;
	SparkTpDeviceCollectiveSubmission submission;
	uint32_t round,rows,operation;
	uint64_t started_ns,submit_ns,completion_ns;
	double event_milli,max_rel;
	uint64_t checked,exact;
	SparkStatus status;
	rank = (SparkHy4Tp16RungRankContext *)argument;
	if ( SparkHy4Tp16RungRankSetup(rank) != 0 )
	{
		SparkHy4Tp16RungBarrierFail();
		return(0);
	}
	for (round=0u; round<SPARK_HY4_TP16_RUNG_ROUNDS; round++)
	{
		if ( SparkHy4Tp16RungBarrierWait() != 0 )
		{
			fprintf(stderr,"TP16RUNG barrier rank=%u round=%u\n",
				(unsigned)rank->rank,(unsigned)round);
			return(0);
		}
		if ( round < SPARK_HY4_TP16_RUNG_BF16_ROUNDS )
		{
			operation =
				SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
			rows = 1u << (round & 3u);
			SparkHy4Tp16RungFillBf16(rank,round,rows);
			if ( cudaMemcpyAsync(rank->input_device,
				rank->host_input,
				(size_t)rows * SPARK_HY4_TP16_RUNG_HIDDEN *
				2u,cudaMemcpyHostToDevice,rank->stream) !=
				cudaSuccess ||
				cudaMemcpyAsync(rank->work_device,
				rank->input_device,
				(size_t)rows * SPARK_HY4_TP16_RUNG_HIDDEN *
				2u,cudaMemcpyDeviceToDevice,rank->stream) !=
				cudaSuccess )
			{
				fprintf(stderr,"TP16RUNG h2d rank=%u round=%u\n",
					(unsigned)rank->rank,(unsigned)round);
				SparkHy4Tp16RungBarrierFail();
				return(0);
			}
		}
		else
		{
			operation =
				SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64;
			rows = SPARK_HY4_TP16_RUNG_MAX_ROWS;
			{
				uint32_t state,element;
				state = UINT32_C(0x9e3779) + rank->rank *
					UINT32_C(15731) + round *
					UINT32_C(31337);
				for (element=0u; element<rows; element++)
				{
					rank->host_u64_input[element] =
						SparkHy4Tp16RungU64Sample(
						&state);
					rank->host_u64_result[element] = 0u;
				}
			}
			if ( cudaMemcpyAsync(rank->u64_input_device,
				rank->host_u64_input,
				(size_t)rows * sizeof(uint64_t),
				cudaMemcpyHostToDevice,rank->stream) !=
				cudaSuccess ||
				cudaMemcpyAsync(rank->u64_work_device,
				rank->u64_input_device,
				(size_t)rows * sizeof(uint64_t),
				cudaMemcpyDeviceToDevice,rank->stream) !=
				cudaSuccess )
			{
				fprintf(stderr,
					"TP16RUNG h2d_u64 rank=%u round=%u\n",
					(unsigned)rank->rank,(unsigned)round);
				SparkHy4Tp16RungBarrierFail();
				return(0);
			}
		}
		rank->completion_done.store(0u,std::memory_order_relaxed);
		rank->completion_status.store(
			(uint32_t)SPARK_STATUS_INTERNAL_ERROR,
			std::memory_order_relaxed);
		memset(&submission,0,sizeof(submission));
		submission.abi_version =
			SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
		submission.descriptor_bytes = sizeof(submission);
		submission.slot_index = 0u;
		submission.active_sequence_count = rows;
		submission.logical_sequence_count = rows;
		submission.flags =
			SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
		submission.ordinal = round;
		if ( operation ==
			SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 )
		{
			submission.local_device = rank->u64_work_device;
			submission.full_device = rank->u64_work_device;
		}
		else
		{
			submission.local_device = rank->work_device;
			submission.full_device = rank->work_device;
		}
		submission.cuda_stream = rank->stream;
		submission.completion_function = SparkHy4Tp16RungCompletion;
		submission.completion_context = rank;
		cudaEventRecord(rank->start_event,rank->stream);
		started_ns = SparkHy4Tp16RungNowNs();
		status = SparkTpDeviceCollectiveEnqueue(&rank->collective,
			&submission,operation);
		submit_ns = SparkHy4Tp16RungNowNs() - started_ns;
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,
				"TP16RUNG enqueue rank=%u round=%u status=%d\n",
				(unsigned)rank->rank,(unsigned)round,
				(int)status);
			SparkHy4Tp16RungBarrierFail();
			return(0);
		}
		cudaEventRecord(rank->stop_event,rank->stream);
		cudaEventSynchronize(rank->stop_event);
		{
			float milliseconds = -1.0f;
			(void)cudaEventElapsedTime(&milliseconds,
				rank->start_event,rank->stop_event);
			event_milli = (double)milliseconds;
		}
		while ( rank->completion_done.load(
			std::memory_order_acquire) == 0u )
		{
			struct timespec pause = {0,1000000};
			nanosleep(&pause,0);
			if ( SparkHy4Tp16RungNowNs() > rung_cutoff_ns )
				break;
		}
		completion_ns = SparkHy4Tp16RungNowNs() - started_ns;
		status = (SparkStatus)rank->completion_status.load(
			std::memory_order_acquire);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,
				"TP16RUNG completion rank=%u round=%u status=%d\n",
				(unsigned)rank->rank,(unsigned)round,
				(int)status);
			SparkHy4Tp16RungBarrierFail();
			return(0);
		}
		max_rel = 0.0;
		checked = 0u;
		exact = 0u;
		if ( operation ==
			SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
		{
			if ( cudaMemcpy(rank->host_result,rank->work_device,
				(size_t)rows * SPARK_HY4_TP16_RUNG_HIDDEN *
				2u,cudaMemcpyDeviceToHost) != cudaSuccess )
			{
				fprintf(stderr,"TP16RUNG d2h rank=%u round=%u\n",
					(unsigned)rank->rank,(unsigned)round);
				SparkHy4Tp16RungBarrierFail();
				return(0);
			}
			max_rel = SparkHy4Tp16RungVerifyBf16(rank,round,rows,
				&max_rel,&checked,&exact);
			if ( max_rel < 0.0 )
			{
				fprintf(stderr,
					"TP16RUNG mismatch rank=%u round=%u rows=%u\n",
					(unsigned)rank->rank,(unsigned)round,
					(unsigned)rows);
				SparkHy4Tp16RungBarrierFail();
				return(0);
			}
			rank->nonzero_checked += checked;
			rank->exact_matches += exact;
			if ( max_rel > rank->max_rel_delta )
				rank->max_rel_delta = max_rel;
		}
		else
		{
			if ( cudaMemcpy(rank->host_u64_result,
				rank->u64_work_device,
				(size_t)rows * sizeof(uint64_t),
				cudaMemcpyDeviceToHost) != cudaSuccess )
			{
				fprintf(stderr,
					"TP16RUNG d2h_u64 rank=%u round=%u\n",
					(unsigned)rank->rank,(unsigned)round);
				SparkHy4Tp16RungBarrierFail();
				return(0);
			}
			if ( SparkHy4Tp16RungVerifyU64(rank,round,rows) != 0 )
			{
				fprintf(stderr,
					"TP16RUNG u64_mismatch rank=%u round=%u\n",
					(unsigned)rank->rank,(unsigned)round);
				SparkHy4Tp16RungBarrierFail();
				return(0);
			}
			checked = rows;
		}
		rank->rounds_done++;
		if ( rung_tsv != 0 )
			fprintf(rung_tsv,
				"rank\t%u\tround\t%u\top\t%u\trows\t%u\tgpu_event_us\t%.1f\tsubmit_wall_us\t%.1f\tcompletion_wall_us\t%.1f\tmax_rel\t%.6f\tnonzero\t%llu\texact\t%llu\n",
				(unsigned)rank->rank,(unsigned)round,
				(unsigned)operation,(unsigned)rows,
				event_milli * 1000.0,
				(double)submit_ns / 1000.0,
				(double)completion_ns / 1000.0,max_rel,
				(unsigned long long)checked,
				(unsigned long long)exact);
	}
	rank->green = 1u;
	rung_green_ranks.fetch_add(1u,std::memory_order_relaxed);
	return(0);
}

int main(int argument_count,char **arguments)
{
	static SparkHy4Tp16RungRankContext ranks[SPARK_HY4_TP16_RUNG_RANKS];
	pthread_t threads[SPARK_HY4_TP16_RUNG_RANKS];
	pthread_t sequencer;
	const char *tsv_path;
	uint32_t index,green;
	uint64_t started_ns;
	(void)arguments;
	if ( argument_count != 1 )
		return(2);
	started_ns = SparkHy4Tp16RungNowNs();
	{
		const char *cutoff = getenv("HY4_TP16_CUTOFF");
		uint64_t seconds = SPARK_HY4_TP16_RUNG_ROUNDS * 60u;
		if ( cutoff != 0 && cutoff[0] != '\0' )
			seconds = (uint64_t)strtoul(cutoff,0,10);
		if ( seconds == 0u )
			seconds = 300u;
		rung_cutoff_ns = started_ns + seconds * UINT64_C(1000000000);
	}
	tsv_path = getenv("HY4_TP16_TSV");
	if ( tsv_path != 0 && tsv_path[0] != '\0' )
	{
		rung_tsv = fopen(tsv_path,"w");
		if ( rung_tsv == 0 )
		{
			fprintf(stderr,"TP16RUNG tsv_open %s errno=%d\n",
				tsv_path,errno);
			return(1);
		}
	}
	memset(&rung_barrier,0,sizeof(rung_barrier));
	{
		pthread_condattr_t condition_attributes;
		pthread_condattr_init(&condition_attributes);
		pthread_condattr_setclock(&condition_attributes,
			CLOCK_MONOTONIC);
		if ( pthread_mutex_init(&rung_barrier.lock,0) != 0 ||
			pthread_cond_init(&rung_barrier.condition,
			&condition_attributes) != 0 )
		{
			fprintf(stderr,"TP16RUNG barrier_init\n");
			return(1);
		}
		pthread_condattr_destroy(&condition_attributes);
	}
	rung_barrier.count = SPARK_HY4_TP16_RUNG_RANKS;
	if ( SparkHy4Tp16RungMeshBase() == 0 )
	{
		fprintf(stderr,"TP16RUNG mesh_alloc %zu\n",
			(size_t)SPARK_HY4_TP16_RUNG_MESH_REGION_BYTES);
		return(1);
	}
	for (index=0u; index<SPARK_HY4_TP16_RUNG_RANKS; index++)
	{
		memset(&ranks[index],0,sizeof(ranks[index]));
		ranks[index].rank = index;
	}
	if ( pthread_create(&sequencer,0,SparkHy4Tp16RungMeshSequencer,0)
		!= 0 )
	{
		fprintf(stderr,"TP16RUNG sequencer\n");
		return(1);
	}
	for (index=0u; index<SPARK_HY4_TP16_RUNG_RANKS; index++)
	{
		if ( pthread_create(&threads[index],0,
			SparkHy4Tp16RungRankThread,&ranks[index]) != 0 )
		{
			fprintf(stderr,"TP16RUNG thread rank=%u\n",
				(unsigned)index);
			SparkHy4Tp16RungBarrierFail();
			return(1);
		}
	}
	for (index=0u; index<SPARK_HY4_TP16_RUNG_RANKS; index++)
		pthread_join(threads[index],0);
	rung_monitor_stop.store(1u,std::memory_order_relaxed);
	pthread_join(sequencer,0);
	green = rung_green_ranks.load();
	for (index=0u; index<SPARK_HY4_TP16_RUNG_RANKS; index++)
	{
		printf("TP16RUNG rank=%u green=%u rounds=%u max_rel=%.6f nonzero=%llu exact=%llu\n",
			(unsigned)index,(unsigned)ranks[index].green,
			(unsigned)ranks[index].rounds_done,
			ranks[index].max_rel_delta,
			(unsigned long long)ranks[index].nonzero_checked,
			(unsigned long long)ranks[index].exact_matches);
	}
	printf("TP16RUNG green_ranks=%u/%u wall_s=%.1f\n",(unsigned)green,
		(unsigned)SPARK_HY4_TP16_RUNG_RANKS,
		(double)(SparkHy4Tp16RungNowNs() - started_ns) /
		1e9);
	if ( rung_tsv != 0 )
		fclose(rung_tsv);
	if ( green != SPARK_HY4_TP16_RUNG_RANKS )
	{
		fprintf(stderr,"TP16RUNG RED\n");
		return(1);
	}
	printf("TP16RUNG GREEN\n");
	printf("TP16_RUNG_DONE\n");
	return(0);
}
