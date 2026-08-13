#include "sparkpipe/spark_tp_device_collective.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SPARK_TP_CHARACTERIZE_DEGREE 4u
#define SPARK_TP_CHARACTERIZE_RAILS 2u
#define SPARK_TP_CHARACTERIZE_MAX_CREDITS 64u
#define SPARK_TP_CHARACTERIZE_HOST_BYTES 64u
#define SPARK_TP_CHARACTERIZE_TIMEOUT_NS UINT64_C(120000000000)
#define SPARK_TP_CHARACTERIZE_OPERATION_BF16_SUM 0u
#define SPARK_TP_CHARACTERIZE_OPERATION_U64_MAX 1u

typedef struct SparkTpCharacterizeOptions
{
	const char *module_path;
	uint32_t rank;
	uint32_t payload_bytes;
	uint32_t iterations;
	uint32_t control_port_base;
	uint32_t credit_count;
	uint32_t split_ring_min_payload_bytes;
	uint32_t inflight_count;
	uint32_t operation_kind;
	char hosts[SPARK_TP_CHARACTERIZE_RAILS][SPARK_TP_CHARACTERIZE_DEGREE]
		[SPARK_TP_CHARACTERIZE_HOST_BYTES];
} SparkTpCharacterizeOptions;

typedef struct SparkTpCharacterizeCompletion
{
	std::atomic<uint32_t> count;
	std::atomic<int32_t> status;
	std::atomic<uint64_t> ordinal[SPARK_TP_CHARACTERIZE_MAX_CREDITS];
} SparkTpCharacterizeCompletion;

static uint64_t SparkTpCharacterizeNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) +
		(uint64_t)now.tv_nsec);
}

static uint32_t SparkTpCharacterizeUint(
	const char *text,uint32_t minimum,uint32_t maximum,uint32_t *value_out)
{
	char *end;
	unsigned long value;
	if ( text == 0 || text[0] == '\0' || value_out == 0 )
		return(0u);
	end = 0;
	value = strtoul(text,&end,10);
	if ( end == text || *end != '\0' || value < minimum || value > maximum )
		return(0u);
	*value_out = (uint32_t)value;
	return(1u);
}

static uint32_t SparkTpCharacterizeHosts(
	const char *text,char hosts[SPARK_TP_CHARACTERIZE_DEGREE]
		[SPARK_TP_CHARACTERIZE_HOST_BYTES])
{
	const char *begin,*end;
	uint32_t bytes,index;
	begin = text;
	for (index=0u; index<SPARK_TP_CHARACTERIZE_DEGREE; index++)
	{
		if ( begin == 0 || begin[0] == '\0' )
			return(0u);
		end = index + 1u == SPARK_TP_CHARACTERIZE_DEGREE ?
			begin + strlen(begin) : strchr(begin,',');
		if ( end == 0 )
			return(0u);
		bytes = (uint32_t)(end - begin);
		if ( bytes == 0u || bytes >= SPARK_TP_CHARACTERIZE_HOST_BYTES )
			return(0u);
		memcpy(hosts[index],begin,bytes);
		hosts[index][bytes] = '\0';
		begin = *end == '\0' ? end : end + 1;
	}
	return(begin[0] == '\0' ? 1u : 0u);
}

static uint32_t SparkTpCharacterizeParse(
	int argc,char **argv,SparkTpCharacterizeOptions *options)
{
	if ( (argc != 11 && argc != 12) || options == 0 )
		return(0u);
	memset(options,0,sizeof(*options));
	options->module_path = argv[1];
	if ( SparkTpCharacterizeUint(argv[2],0u,3u,&options->rank) == 0u ||
		SparkTpCharacterizeHosts(argv[3],options->hosts[0]) == 0u ||
		SparkTpCharacterizeHosts(argv[4],options->hosts[1]) == 0u ||
		SparkTpCharacterizeUint(argv[5],2u,UINT32_MAX,
			&options->payload_bytes) == 0u ||
		SparkTpCharacterizeUint(argv[6],1u,1000000u,
			&options->iterations) == 0u ||
		SparkTpCharacterizeUint(argv[7],1024u,65535u,
			&options->control_port_base) == 0u ||
		SparkTpCharacterizeUint(argv[8],1u,
			SPARK_TP_CHARACTERIZE_MAX_CREDITS,&options->credit_count) == 0u ||
		SparkTpCharacterizeUint(argv[9],0u,UINT32_MAX,
			&options->split_ring_min_payload_bytes) == 0u ||
		SparkTpCharacterizeUint(argv[10],1u,
			SPARK_TP_CHARACTERIZE_MAX_CREDITS,&options->inflight_count) == 0u ||
		options->inflight_count > options->credit_count ||
		(options->payload_bytes & 1u) != 0u )
		return(0u);
	options->operation_kind = SPARK_TP_CHARACTERIZE_OPERATION_BF16_SUM;
	if ( argc == 12 )
	{
		if ( strcmp(argv[11],"u64-max") != 0 ||
			options->payload_bytes != sizeof(uint64_t) )
			return(0u);
		options->operation_kind = SPARK_TP_CHARACTERIZE_OPERATION_U64_MAX;
	}
	return(1u);
}

__global__ static void SparkTpCharacterizeFill(
	__nv_bfloat16 *values,uint32_t count,float value)
{
	uint32_t index;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		values[index] = __float2bfloat16(value);
}

__global__ static void SparkTpCharacterizeAdd(
	__nv_bfloat16 *destination,const __nv_bfloat16 *source,uint32_t count)
{
	uint32_t index;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		destination[index] = __hadd(destination[index],source[index]);
}

__global__ static void SparkTpCharacterizeFillU64(
	uint64_t *values,uint32_t count,uint64_t value)
{
	uint32_t index;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		values[index] = value;
}

__global__ static void SparkTpCharacterizeMaxU64(
	uint64_t *destination,const uint64_t *source,uint32_t count)
{
	uint32_t index;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count && source[index] > destination[index] )
		destination[index] = source[index];
}

static SparkStatus SparkTpCharacterizeCombine(
	void *context,void *destination,const void *source,uint32_t rows,
	uint32_t hidden,void *cuda_stream)
{
	uint64_t count;
	(void)context;
	count = (uint64_t)rows * hidden;
	if ( destination == 0 || source == 0 || cuda_stream == 0 ||
		count == 0u || count > UINT32_MAX )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkTpCharacterizeAdd<<<((uint32_t)count + 255u) / 256u,256u,0,
		(cudaStream_t)cuda_stream>>>((__nv_bfloat16 *)destination,
		(const __nv_bfloat16 *)source,(uint32_t)count);
	return(cudaGetLastError() == cudaSuccess ?
		SPARK_STATUS_OK : SPARK_STATUS_DRIVER_LOAD_ERROR);
}

static SparkStatus SparkTpCharacterizeCombineU64Max(
	void *context,uint64_t *destination,const uint64_t *source,
	uint32_t element_count,void *cuda_stream)
{
	(void)context;
	if ( destination == 0 || source == 0 || cuda_stream == 0 ||
		element_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkTpCharacterizeMaxU64<<<(element_count + 255u) / 256u,256u,0,
		(cudaStream_t)cuda_stream>>>(destination,source,element_count);
	return(cudaGetLastError() == cudaSuccess ?
		SPARK_STATUS_OK : SPARK_STATUS_DRIVER_LOAD_ERROR);
}

static void SparkTpCharacterizeComplete(
	void *context,const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkTpCharacterizeCompletion *state;
	state = (SparkTpCharacterizeCompletion *)context;
	if ( state == 0 || completion == 0 )
		return;
	state->status.store((int32_t)completion->status,std::memory_order_release);
	if ( completion->credit_index < SPARK_TP_CHARACTERIZE_MAX_CREDITS )
		state->ordinal[completion->credit_index].store(completion->ordinal,
			std::memory_order_release);
	state->count.fetch_add(1u,std::memory_order_release);
}

static SparkStatus SparkTpCharacterizeWait(
	SparkTpCharacterizeCompletion *completion,uint32_t expected_count)
{
	uint64_t deadline;
	deadline = SparkTpCharacterizeNowNs() + SPARK_TP_CHARACTERIZE_TIMEOUT_NS;
	while ( completion->count.load(std::memory_order_acquire) < expected_count )
	{
		if ( SparkTpCharacterizeNowNs() >= deadline )
			return(SPARK_STATUS_IO_ERROR);
		sched_yield();
	}
	return((SparkStatus)completion->status.load(std::memory_order_acquire));
}

static SparkStatus SparkTpCharacterizeWaitCredit(
	SparkTpCharacterizeCompletion *completion,uint32_t credit,
	uint64_t expected_ordinal)
{
	uint64_t deadline;
	deadline = SparkTpCharacterizeNowNs() + SPARK_TP_CHARACTERIZE_TIMEOUT_NS;
	while ( completion->ordinal[credit].load(std::memory_order_acquire) !=
		expected_ordinal )
	{
		if ( SparkTpCharacterizeNowNs() >= deadline )
			return(SPARK_STATUS_IO_ERROR);
		sched_yield();
	}
	return((SparkStatus)completion->status.load(std::memory_order_acquire));
}

static void SparkTpCharacterizeConfigure(
	const SparkTpCharacterizeOptions *options,cudaStream_t stream,
	SparkTpDeviceCollectiveCreditBinding *bindings,
	SparkTpDeviceCollectiveConfig *configuration)
{
	uint32_t rail,rank;
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration->backend_kind =
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration->tp_degree = SPARK_TP_CHARACTERIZE_DEGREE;
	configuration->tp_rank = options->rank;
	configuration->operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration->credit_count = options->credit_count;
	configuration->local_hidden_dimension = options->payload_bytes / 2u;
	configuration->max_active_sequence_count = 1u;
	configuration->connect_timeout_milli = 120000u;
	configuration->operation_timeout_milli = 120000u;
	configuration->control_port_base = options->control_port_base;
	configuration->algorithm_mask = options->split_ring_min_payload_bytes == 0u ?
		SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING :
		SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS;
	configuration->split_ring_min_payload_bytes =
		options->split_ring_min_payload_bytes;
	configuration->rail_count = SPARK_TP_CHARACTERIZE_RAILS;
	configuration->step_rail_indices[0] = 0u;
	configuration->step_rail_indices[1] = 1u;
	configuration->step_rail_indices[2] = 1u;
	configuration->collective_identifier = UINT64_C(0x54503452444d4100) |
		options->control_port_base;
	configuration->backend_module_path = options->module_path;
	configuration->local_host = options->hosts[1][options->rank];
	configuration->registration_cuda_stream = stream;
	configuration->combine_bf16_function = SparkTpCharacterizeCombine;
	configuration->combine_u64_max_function =
		SparkTpCharacterizeCombineU64Max;
	configuration->credit_bindings = bindings;
	configuration->credit_binding_count =
		options->credit_count * 2u;
	for (rank=0u; rank<SPARK_TP_CHARACTERIZE_DEGREE; rank++)
		configuration->rank_hosts[rank] = options->hosts[1][rank];
	for (rail=0u; rail<SPARK_TP_CHARACTERIZE_RAILS; rail++)
		for (rank=0u; rank<SPARK_TP_CHARACTERIZE_DEGREE; rank++)
			configuration->rail_rank_hosts[rail][rank] =
				options->hosts[rail][rank];
}

static SparkStatus SparkTpCharacterizeBindings(
	uint32_t payload_bytes,void *send_device,void *receive_device,
	void *send_transport,void *receive_transport,
	uint32_t mapped_alias,uint32_t credit_count,
	SparkTpDeviceCollectiveCreditBinding *bindings)
{
	uint64_t offset;
	uint32_t credit,step,index;
	for (step=0u; step<2u; step++)
		for (credit=0u; credit<credit_count; credit++)
		{
			index = step * credit_count + credit;
			offset = (uint64_t)index * payload_bytes;
			bindings[index].step_index = step;
			bindings[index].credit_index = credit;
			bindings[index].send_device = (uint8_t *)send_device + offset;
			bindings[index].receive_device = (uint8_t *)receive_device + offset;
			bindings[index].send_transport =
				(uint8_t *)send_transport + offset;
			bindings[index].receive_transport =
				(uint8_t *)receive_transport + offset;
			bindings[index].flags = mapped_alias != 0u ?
				SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS : 0u;
			bindings[index].reserved0 = 0u;
		}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpCharacterizeRun(
	SparkTpDeviceCollective *collective,cudaStream_t stream,void *local,
	void *output,uint64_t local_value,uint32_t operation_kind,
	uint32_t payload_bytes,uint32_t credit_count,
	uint32_t inflight_count,uint32_t iterations,uint64_t first_ordinal,
	SparkTpCharacterizeCompletion *completion)
{
	SparkTpDeviceCollectiveSubmission submission;
	SparkStatus status;
	uint64_t ordinal;
	uint32_t credit,expected_count;
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.active_sequence_count = 1u;
	submission.cuda_stream = stream;
	submission.completion_function = SparkTpCharacterizeComplete;
	submission.completion_context = completion;
	for (ordinal=first_ordinal; ordinal<first_ordinal + iterations; ordinal++)
	{
		credit = (uint32_t)(ordinal % credit_count);
		if ( ordinal >= first_ordinal + inflight_count )
		{
			expected_count = (uint32_t)(ordinal - inflight_count + 1u);
			status = SparkTpCharacterizeWait(completion,expected_count);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
		if ( ordinal >= credit_count )
		{
			status = SparkTpCharacterizeWaitCredit(completion,credit,
				ordinal - credit_count);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
		submission.local_device = (uint8_t *)local +
			(uint64_t)credit * payload_bytes;
		submission.full_device = (uint8_t *)output +
			(uint64_t)credit * payload_bytes;
		if ( operation_kind == SPARK_TP_CHARACTERIZE_OPERATION_U64_MAX )
			SparkTpCharacterizeFillU64<<<1u,1u,0,stream>>>(
				(uint64_t *)submission.local_device,1u,local_value);
		else
			SparkTpCharacterizeFill<<<
				(collective->local_hidden_dimension + 255u) / 256u,256u,0,stream>>>(
				(__nv_bfloat16 *)submission.local_device,
				collective->local_hidden_dimension,(float)local_value);
		if ( cudaGetLastError() != cudaSuccess )
			return(SPARK_STATUS_DRIVER_LOAD_ERROR);
		submission.ordinal = ordinal;
		do
		{
			status = operation_kind == SPARK_TP_CHARACTERIZE_OPERATION_U64_MAX ?
				SparkTpDeviceCollectiveSubmitU64Max(collective,&submission) :
				SparkTpDeviceCollectiveSubmitBf16(collective,&submission);
			if ( status == SPARK_STATUS_BUSY )
				sched_yield();
		} while ( status == SPARK_STATUS_BUSY );
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	status = SparkTpCharacterizeWait(completion,
		(uint32_t)(first_ordinal + iterations));
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(cudaStreamSynchronize(stream) == cudaSuccess ?
		SPARK_STATUS_OK : SPARK_STATUS_DRIVER_LOAD_ERROR);
}

int main(int argc,char **argv)
{
	SparkTpDeviceCollectiveCreditBinding bindings[128u];
	SparkTpCharacterizeCompletion completion;
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpCharacterizeOptions options;
	SparkTpDeviceCollective collective;
	cudaStream_t stream;
	__nv_bfloat16 result;
	uint64_t result_u64;
	void *local,*output,*receive_device,*receive_transport,*result_output;
	void *send_device,*send_transport;
	uint64_t start_ns,end_ns,scratch_bytes;
	uint32_t credit,memory_mode;
	SparkStatus status;
	if ( SparkTpCharacterizeParse(argc,argv,&options) == 0u )
	{
		fprintf(stderr,"usage: %s MODULE RANK DIRECT_CSV SWITCH_CSV PAYLOAD_BYTES ITERATIONS CONTROL_PORT_BASE CREDIT_COUNT SPLIT_RING_MIN_PAYLOAD_BYTES INFLIGHT_COUNT [u64-max]\n",argv[0]);
		return(2);
	}
	status = SparkTpDeviceCollectiveProbeMemoryMode(
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT,
		options.module_path,&memory_mode);
	if ( status != SPARK_STATUS_OK )
		return((int)status + 10);
	scratch_bytes = (uint64_t)options.payload_bytes *
		options.credit_count * 2u;
	receive_transport = 0;
	send_transport = 0;
	if ( cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) != cudaSuccess ||
		cudaMalloc(&local,(uint64_t)options.payload_bytes *
			options.credit_count) != cudaSuccess ||
		cudaMalloc(&output,(uint64_t)options.payload_bytes *
			options.credit_count) != cudaSuccess ||
		cudaMalloc(&send_device,scratch_bytes) != cudaSuccess ||
		cudaMalloc(&receive_device,scratch_bytes) != cudaSuccess )
		return(3);
	if ( memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
	{
		if ( cudaHostAlloc(&send_transport,scratch_bytes,
				cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
			cudaHostAlloc(&receive_transport,scratch_bytes,
				cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess )
			return(3);
		(void)cudaFree(send_device);
		(void)cudaFree(receive_device);
		if ( cudaHostGetDevicePointer(&send_device,send_transport,0u) !=
				cudaSuccess ||
			cudaHostGetDevicePointer(&receive_device,receive_transport,0u) !=
				cudaSuccess )
			return(3);
	}
	else
	{
		send_transport = send_device;
		receive_transport = receive_device;
	}
	(void)SparkTpCharacterizeBindings(options.payload_bytes,send_device,
		receive_device,send_transport,receive_transport,
		memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST,
		options.credit_count,bindings);
	SparkTpCharacterizeConfigure(&options,stream,bindings,&configuration);
	status = SparkTpDeviceCollectiveCreate(&configuration,&collective);
	if ( status != SPARK_STATUS_OK )
		return((int)status + 10);
	completion.count.store(0u,std::memory_order_relaxed);
	completion.status.store((int32_t)SPARK_STATUS_OK,
		std::memory_order_relaxed);
	for (credit=0u; credit<SPARK_TP_CHARACTERIZE_MAX_CREDITS; credit++)
		completion.ordinal[credit].store(UINT64_MAX,
			std::memory_order_relaxed);
	status = SparkTpCharacterizeRun(&collective,stream,local,output,
		(uint64_t)(options.rank + 1u),options.operation_kind,
		options.payload_bytes,options.credit_count,
		options.inflight_count,10u,0u,&completion);
	start_ns = SparkTpCharacterizeNowNs();
	if ( status == SPARK_STATUS_OK )
		status = SparkTpCharacterizeRun(&collective,stream,local,output,
			(uint64_t)(options.rank + 1u),options.operation_kind,
			options.payload_bytes,options.credit_count,
			options.inflight_count,options.iterations,10u,&completion);
	end_ns = SparkTpCharacterizeNowNs();
	result_output = (uint8_t *)output + ((10u + options.iterations - 1u) %
		options.credit_count) * (uint64_t)options.payload_bytes;
	result = __float2bfloat16(0.0f);
	result_u64 = 0u;
	if ( status == SPARK_STATUS_OK && cudaMemcpy(
			options.operation_kind == SPARK_TP_CHARACTERIZE_OPERATION_U64_MAX ?
				(void *)&result_u64 : (void *)&result,
			result_output,
			options.operation_kind == SPARK_TP_CHARACTERIZE_OPERATION_U64_MAX ?
				sizeof(result_u64) : sizeof(result),
			cudaMemcpyDeviceToHost) != cudaSuccess )
		status = SPARK_STATUS_DRIVER_LOAD_ERROR;
	printf("{\"rank\":%u,\"operation\":\"%s\",\"payload_bytes\":%u,\"iterations\":%u,\"credits\":%u,\"inflight\":%u,\"split_ring_min_payload_bytes\":%u,\"memory_mode\":%u,\"status\":%u,\"result\":%.1f,\"average_us\":%.3f}\n",
		options.rank,options.operation_kind ==
			SPARK_TP_CHARACTERIZE_OPERATION_U64_MAX ? "u64-max" : "bf16-sum",
		options.payload_bytes,options.iterations,
		options.credit_count,options.inflight_count,
		options.split_ring_min_payload_bytes,memory_mode,
		(uint32_t)status,
		options.operation_kind == SPARK_TP_CHARACTERIZE_OPERATION_U64_MAX ?
			(double)result_u64 : (double)__bfloat162float(result),
		status == SPARK_STATUS_OK ?
		(double)(end_ns - start_ns) / (double)options.iterations / 1000.0 : 0.0);
	SparkTpDeviceCollectiveDestroy(&collective);
	if ( memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
	{
		(void)cudaFreeHost(send_transport);
		(void)cudaFreeHost(receive_transport);
	}
	if ( memory_mode != SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
	{
		(void)cudaFree(receive_device);
		(void)cudaFree(send_device);
	}
	(void)cudaFree(output);
	(void)cudaFree(local);
	(void)cudaStreamDestroy(stream);
	return(status == SPARK_STATUS_OK &&
		(options.operation_kind == SPARK_TP_CHARACTERIZE_OPERATION_U64_MAX ?
			result_u64 == 4u : __bfloat162float(result) == 10.0f) ? 0 : 5);
}
