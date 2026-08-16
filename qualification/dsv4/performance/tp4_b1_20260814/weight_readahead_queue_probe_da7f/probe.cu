#include <algorithm>
#include <atomic>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <vector>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_lm_kernels.cuh"

#define POC_LAYER_COUNT 43u
#define POC_TASK_COUNT (POC_LAYER_COUNT * 3u)
#define POC_REPETITION_COUNT 31u
#define POC_WARMUP_COUNT 4u
#define POC_COLLECTIVE_DELAY_NS 60000ull
#define POC_FLUSH_BYTES (256u * 1024u * 1024u)
#define POC_MAX_BLOCK_COUNT 48u

#define POC_CUDA(call) do { cudaError_t poc_error = (call); if ( poc_error != cudaSuccess ) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(poc_error)); return(-1); } } while (0)

typedef struct PocReadAheadTask
{
	const uint4 *payload;
	uint64_t sector_count;
	const uint4 *auxiliary_payload;
	uint64_t auxiliary_vector_count;
	uint64_t payload_vector_count;
} PocReadAheadTask;

typedef struct PocBuffers
{
	uint8_t *weight_storage,*flush;
	PocReadAheadTask *tasks;
	uint32_t *sink,*block_done,*task_published,*task_done,*flush_sink;
	uint32_t *control_output,*candidate_output;
	volatile uint32_t *collective_request_host;
	volatile uint32_t *collective_done_host;
	uint32_t *collective_request_device,*collective_done_device;
	uint64_t weight_storage_bytes;
} PocBuffers;

typedef struct PocTiming
{
	float median_ms,p10_ms,p90_ms,mean_ms;
} PocTiming;

static cudaError_t PocStreamWriteValue32(cudaStream_t stream,uint32_t *address,
	uint32_t value)
{
	CUresult result = cuStreamWriteValue32((CUstream)stream,
		(CUdeviceptr)(uintptr_t)address,value,CU_STREAM_WRITE_VALUE_DEFAULT);
	return(result == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown);
}

static cudaError_t PocStreamWaitValue32(cudaStream_t stream,uint32_t *address,
	uint32_t value)
{
	CUresult result = cuStreamWaitValue32((CUstream)stream,
		(CUdeviceptr)(uintptr_t)address,value,CU_STREAM_WAIT_VALUE_GEQ);
	return(result == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown);
}

static __global__ void PocFillBytesKernel(uint8_t *values,uint64_t count,
	uint32_t seed)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	uint32_t value;
	for (; index<count; index+=stride)
	{
		value = ((uint32_t)index * UINT32_C(747796405)) + seed;
		value = ((value >> ((value >> 28u) + 4u)) ^ value) *
			UINT32_C(277803737);
		value = (value >> 22u) ^ value;
		values[index] = (uint8_t)value;
	}
}

static __global__ void PocFlushKernel(const uint4 *values,uint64_t count,
	uint32_t *sink)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	uint32_t checksum = 0u;
	for (; index<count; index+=stride)
	{
		uint4 value = values[index];
		checksum ^= value.x ^ value.y ^ value.z ^ value.w;
	}
	if ( checksum != 0u )
		atomicXor(sink,checksum);
}

static __global__ void PocConsumerKernel(const PocReadAheadTask *tasks,
	uint32_t task_index,uint32_t *output)
{
	PocReadAheadTask task = tasks[task_index];
	uint64_t thread_index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x,index;
	uint32_t checksum = UINT32_C(0x9e3779b9) ^ task_index ^
		(uint32_t)thread_index;
	uint4 value;
	for (index=thread_index; index<task.payload_vector_count; index+=stride)
	{
		value = task.payload[index];
		checksum ^= value.x ^ value.y ^ value.z ^ value.w;
	}
	for (index=thread_index; index<task.auxiliary_vector_count; index+=stride)
	{
		value = task.auxiliary_payload[index];
		checksum ^= value.x ^ value.y ^ value.z ^ value.w;
	}
	atomicXor(output + task_index,checksum);
}

static __global__ void PocPersistentReadAheadKernel(
	const PocReadAheadTask *tasks,uint32_t task_count,uint32_t *sink,
	uint32_t *block_done,uint32_t *task_published,uint32_t *task_done)
{
	uint32_t task_index,checksum;
	uint64_t thread_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x,index;
	uint4 value;
	for (task_index=0u; task_index<task_count; task_index++)
	{
		while ( atomicAdd(task_published,0u) <= task_index )
			__nanosleep(64u);
		PocReadAheadTask task = tasks[task_index];
		checksum = 0u;
		for (index=thread_index; index<task.sector_count; index+=stride)
		{
			value = task.payload[index *
				(SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES / sizeof(uint4))];
			checksum ^= value.x ^ value.y ^ value.z ^ value.w;
		}
		for (index=thread_index; index<task.auxiliary_vector_count; index+=stride)
		{
			value = task.auxiliary_payload[index];
			checksum ^= value.x ^ value.y ^ value.z ^ value.w;
		}
		sink[thread_index] = checksum;
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			uint32_t arrived = atomicAdd(block_done,1u) + 1u;
			if ( arrived == gridDim.x )
			{
				atomicExch(block_done,0u);
				__threadfence();
				atomicExch(task_done,task_index + 1u);
			}
		}
		__syncthreads();
		while ( atomicAdd(task_done,0u) <= task_index )
			__nanosleep(64u);
	}
}

static uint64_t PocAlign(uint64_t value,uint64_t alignment)
{
	return((value + alignment - 1u) & ~(alignment - 1u));
}

static int32_t PocBuildTasks(std::vector<PocReadAheadTask> *tasks,
	uint8_t *storage,uint64_t *storage_bytes)
{
	uint64_t wq_payload_bytes =
		(uint64_t)(SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION / 4u) *
		SPARK_DSV4_MODEL_QUERY_LORA_RANK;
	uint64_t wq_scale_bytes = wq_payload_bytes / 128u;
	uint64_t hc_bytes = (uint64_t)SPARK_DSV4_MODEL_HC_MIX_ROWS *
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(float);
	uint64_t head_bytes = (uint64_t)SPARK_DSV4_MODEL_HC_STREAM_COUNT *
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(float);
	uint64_t offset = 0u,payload_bytes,auxiliary_bytes;
	uint32_t layer,phase,index = 0u;
	tasks->resize(POC_TASK_COUNT);
	for (layer=0u; layer<POC_LAYER_COUNT; layer++)
		for (phase=0u; phase<3u; phase++,index++)
		{
			payload_bytes = phase == 0u ? wq_payload_bytes :
				(phase == 2u && layer + 1u == POC_LAYER_COUNT ?
				head_bytes : hc_bytes);
			auxiliary_bytes = phase == 0u ? wq_scale_bytes : 0u;
			offset = PocAlign(offset,SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES);
			(*tasks)[index].payload = storage == 0 ? 0 :
				(const uint4 *)(storage + offset);
			(*tasks)[index].sector_count = payload_bytes /
				SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES;
			(*tasks)[index].payload_vector_count = payload_bytes / sizeof(uint4);
			offset += payload_bytes;
			offset = PocAlign(offset,alignof(uint4));
			(*tasks)[index].auxiliary_payload = auxiliary_bytes == 0u ? 0 :
				(storage == 0 ? 0 : (const uint4 *)(storage + offset));
			(*tasks)[index].auxiliary_vector_count = auxiliary_bytes /
				sizeof(uint4);
			offset += auxiliary_bytes;
		}
	*storage_bytes = PocAlign(offset,256u);
	return(0);
}

static int32_t PocAllocate(PocBuffers *buffers,cudaStream_t stream,
	std::vector<PocReadAheadTask> *host_tasks)
{
	uint64_t storage_bytes;
	memset(buffers,0,sizeof(*buffers));
	PocBuildTasks(host_tasks,0,&storage_bytes);
	POC_CUDA(cudaMalloc(&buffers->weight_storage,storage_bytes));
	POC_CUDA(cudaMalloc(&buffers->tasks,
		POC_TASK_COUNT * sizeof(PocReadAheadTask)));
	POC_CUDA(cudaMalloc(&buffers->sink,
		POC_MAX_BLOCK_COUNT * SPARK_LM_CTA_THREADS * sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->block_done,sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->task_published,sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->task_done,sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->control_output,
		POC_TASK_COUNT * sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->candidate_output,
		POC_TASK_COUNT * sizeof(uint32_t)));
	POC_CUDA(cudaMalloc(&buffers->flush,POC_FLUSH_BYTES));
	POC_CUDA(cudaMalloc(&buffers->flush_sink,sizeof(uint32_t)));
	POC_CUDA(cudaHostAlloc((void **)&buffers->collective_request_host,
		2u * sizeof(uint32_t),cudaHostAllocMapped));
	buffers->collective_done_host = buffers->collective_request_host + 1u;
	POC_CUDA(cudaHostGetDevicePointer(&buffers->collective_request_device,
		(void *)buffers->collective_request_host,0u));
	POC_CUDA(cudaHostGetDevicePointer(&buffers->collective_done_device,
		(void *)buffers->collective_done_host,0u));
	buffers->weight_storage_bytes = storage_bytes;
	PocBuildTasks(host_tasks,buffers->weight_storage,&storage_bytes);
	POC_CUDA(cudaMemcpyAsync(buffers->tasks,host_tasks->data(),
		POC_TASK_COUNT * sizeof(PocReadAheadTask),cudaMemcpyHostToDevice,stream));
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(buffers->weight_storage,
		storage_bytes,UINT32_C(0x4e6917b3));
	POC_CUDA(cudaMemsetAsync(buffers->flush,0xa5,POC_FLUSH_BYTES,stream));
	POC_CUDA(cudaMemsetAsync(buffers->flush_sink,0,sizeof(uint32_t),stream));
	POC_CUDA(cudaStreamSynchronize(stream));
	return(0);
}

static void PocProgressCollectives(volatile uint32_t *request,
	volatile uint32_t *done)
{
	uint32_t task;
	for (task=1u; task<=POC_TASK_COUNT; task++)
	{
		while ( __atomic_load_n(request,__ATOMIC_ACQUIRE) < task )
			std::this_thread::yield();
		auto start = std::chrono::steady_clock::now();
		while ( (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - start).count() <
			POC_COLLECTIVE_DELAY_NS )
			std::atomic_signal_fence(std::memory_order_seq_cst);
		__atomic_store_n(done,task,__ATOMIC_RELEASE);
	}
}

static int32_t PocReset(cudaStream_t primary,const PocBuffers *buffers,
	uint32_t *output)
{
	__atomic_store_n(buffers->collective_request_host,0u,__ATOMIC_RELEASE);
	__atomic_store_n(buffers->collective_done_host,0u,__ATOMIC_RELEASE);
	if ( cudaMemsetAsync(buffers->block_done,0,sizeof(uint32_t),primary) !=
			cudaSuccess )
		return(-1);
	if ( cudaMemsetAsync(buffers->task_published,0,sizeof(uint32_t),primary) !=
			cudaSuccess )
		return(-2);
	if ( cudaMemsetAsync(buffers->task_done,0,sizeof(uint32_t),primary) !=
			cudaSuccess )
		return(-3);
	if ( cudaMemsetAsync(output,0,POC_TASK_COUNT * sizeof(uint32_t),primary) !=
			cudaSuccess )
		return(-4);
	PocFlushKernel<<<4096u,256u,0,primary>>>((const uint4 *)buffers->flush,
		POC_FLUSH_BYTES / sizeof(uint4),buffers->flush_sink);
	return(cudaStreamSynchronize(primary) == cudaSuccess ? 0 : -5);
}

static int32_t PocRunStep(cudaStream_t primary,cudaStream_t read_ahead,
	cudaEvent_t start,cudaEvent_t stop,cudaEvent_t *source_events,
	cudaEvent_t *completion_events,const PocBuffers *buffers,
	const std::vector<PocReadAheadTask> *host_tasks,uint32_t persistent,
	uint32_t persistent_block_count,float *elapsed_ms)
{
	uint32_t task;
	uint32_t *output = persistent != 0u ? buffers->candidate_output :
		buffers->control_output;
	if ( PocReset(primary,buffers,output) < 0 )
		return(-1);
	if ( persistent != 0u )
	{
		PocPersistentReadAheadKernel<<<persistent_block_count,
			SPARK_LM_CTA_THREADS,0u,read_ahead>>>(buffers->tasks,POC_TASK_COUNT,
			buffers->sink,buffers->block_done,buffers->task_published,
			buffers->task_done);
		if ( cudaGetLastError() != cudaSuccess )
			return(-2);
	}
	std::thread progress(PocProgressCollectives,
		buffers->collective_request_host,buffers->collective_done_host);
	if ( cudaEventRecord(start,primary) != cudaSuccess )
		return(-3);
	for (task=0u; task<POC_TASK_COUNT; task++)
	{
		if ( persistent != 0u )
		{
			if ( PocStreamWriteValue32(primary,buffers->task_published,
					task + 1u) != cudaSuccess )
				return(-4);
		}
		else
		{
			if ( cudaEventRecord(source_events[task],primary) != cudaSuccess ||
				cudaStreamWaitEvent(read_ahead,source_events[task],0u) != cudaSuccess )
				return(-5);
			const PocReadAheadTask *descriptor = &(*host_tasks)[task];
			uint64_t payload_bytes = descriptor->sector_count *
				SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES;
			uint64_t auxiliary_bytes = descriptor->auxiliary_vector_count *
				sizeof(uint4);
			if ( SparkLmHostLaunchWeightReadAhead(read_ahead,descriptor->payload,
					payload_bytes,descriptor->auxiliary_payload,auxiliary_bytes,
					buffers->sink,POC_MAX_BLOCK_COUNT) != cudaSuccess ||
				cudaEventRecord(completion_events[task],read_ahead) != cudaSuccess )
				return(-6);
		}
		if ( PocStreamWriteValue32(primary,buffers->collective_request_device,
				task + 1u) != cudaSuccess ||
			PocStreamWaitValue32(primary,buffers->collective_done_device,
				task + 1u) != cudaSuccess )
			return(-7);
		if ( persistent != 0u )
		{
			if ( PocStreamWaitValue32(primary,buffers->task_done,task + 1u) !=
				cudaSuccess )
				return(-8);
		}
		else if ( cudaStreamWaitEvent(primary,completion_events[task],0u) !=
			cudaSuccess )
			return(-9);
		PocConsumerKernel<<<POC_MAX_BLOCK_COUNT,SPARK_LM_CTA_THREADS,0u,
			primary>>>(buffers->tasks,task,output);
		if ( cudaGetLastError() != cudaSuccess )
			return(-10);
	}
	if ( cudaEventRecord(stop,primary) != cudaSuccess ||
		cudaEventSynchronize(stop) != cudaSuccess ||
		cudaEventElapsedTime(elapsed_ms,start,stop) != cudaSuccess )
		return(-11);
	progress.join();
	if ( cudaStreamSynchronize(read_ahead) != cudaSuccess )
		return(-12);
	return(0);
}

static PocTiming PocSummarize(std::vector<float> values)
{
	PocTiming timing = {};
	uint32_t index;
	std::sort(values.begin(),values.end());
	for (index=0u; index<values.size(); index++)
		timing.mean_ms += values[index];
	timing.mean_ms /= (float)values.size();
	timing.median_ms = values[values.size() / 2u];
	timing.p10_ms = values[values.size() / 10u];
	timing.p90_ms = values[(values.size() * 9u) / 10u];
	return(timing);
}

static int32_t PocMeasure(cudaStream_t primary,cudaStream_t read_ahead,
	cudaEvent_t start,cudaEvent_t stop,cudaEvent_t *source_events,
	cudaEvent_t *completion_events,const PocBuffers *buffers,
	const std::vector<PocReadAheadTask> *host_tasks,uint32_t block_count,
	PocTiming *control,PocTiming *candidate)
{
	std::vector<float> control_times,candidate_times;
	float elapsed;
	uint32_t repetition,warmup;
	for (warmup=0u; warmup<POC_WARMUP_COUNT; warmup++)
		if ( PocRunStep(primary,read_ahead,start,stop,source_events,
			completion_events,buffers,host_tasks,warmup & 1u,block_count,
			&elapsed) < 0 )
			return(-1);
	for (repetition=0u; repetition<POC_REPETITION_COUNT; repetition++)
	{
		if ( (repetition & 1u) == 0u )
		{
			if ( PocRunStep(primary,read_ahead,start,stop,source_events,
				completion_events,buffers,host_tasks,0u,block_count,&elapsed) < 0 )
				return(-2);
			control_times.push_back(elapsed);
			if ( PocRunStep(primary,read_ahead,start,stop,source_events,
				completion_events,buffers,host_tasks,1u,block_count,&elapsed) < 0 )
				return(-3);
			candidate_times.push_back(elapsed);
		}
		else
		{
			if ( PocRunStep(primary,read_ahead,start,stop,source_events,
				completion_events,buffers,host_tasks,1u,block_count,&elapsed) < 0 )
				return(-4);
			candidate_times.push_back(elapsed);
			if ( PocRunStep(primary,read_ahead,start,stop,source_events,
				completion_events,buffers,host_tasks,0u,block_count,&elapsed) < 0 )
				return(-5);
			control_times.push_back(elapsed);
		}
	}
	*control = PocSummarize(control_times);
	*candidate = PocSummarize(candidate_times);
	return(0);
}

static int32_t PocCheckExact(const PocBuffers *buffers)
{
	std::vector<uint32_t> control(POC_TASK_COUNT),candidate(POC_TASK_COUNT);
	if ( cudaMemcpy(control.data(),buffers->control_output,
			POC_TASK_COUNT * sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(-1);
	if ( cudaMemcpy(candidate.data(),buffers->candidate_output,
			POC_TASK_COUNT * sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(-2);
	return(memcmp(control.data(),candidate.data(),
		POC_TASK_COUNT * sizeof(uint32_t)) == 0 ? 0 : -3);
}

int main(void)
{
	PocBuffers buffers;
	PocTiming control,candidate;
	cudaDeviceProp properties;
	cudaEvent_t start,stop;
	cudaStream_t primary,read_ahead;
	cudaEvent_t source_events[POC_TASK_COUNT],completion_events[POC_TASK_COUNT];
	std::vector<PocReadAheadTask> host_tasks;
	uint32_t block_counts[] = {1u,2u,4u,8u};
	uint32_t index;
	POC_CUDA(cudaGetDeviceProperties(&properties,0));
	POC_CUDA(cudaStreamCreateWithFlags(&primary,cudaStreamNonBlocking));
	POC_CUDA(cudaStreamCreateWithFlags(&read_ahead,cudaStreamNonBlocking));
	POC_CUDA(cudaEventCreate(&start));
	POC_CUDA(cudaEventCreate(&stop));
	for (index=0u; index<POC_TASK_COUNT; index++)
	{
		POC_CUDA(cudaEventCreateWithFlags(&source_events[index],
			cudaEventDisableTiming));
		POC_CUDA(cudaEventCreateWithFlags(&completion_events[index],
			cudaEventDisableTiming));
	}
	if ( PocAllocate(&buffers,primary,&host_tasks) < 0 )
		return(2);
	printf("device=%s sm_count=%d tasks=%u layers=%u collective_delay_ns=%llu weight_storage_bytes=%llu repetitions=%u\n",
		properties.name,properties.multiProcessorCount,POC_TASK_COUNT,
		POC_LAYER_COUNT,(unsigned long long)POC_COLLECTIVE_DELAY_NS,
		(unsigned long long)buffers.weight_storage_bytes,POC_REPETITION_COUNT);
	for (index=0u; index<sizeof(block_counts) / sizeof(block_counts[0]); index++)
	{
		if ( PocMeasure(primary,read_ahead,start,stop,source_events,
				completion_events,&buffers,&host_tasks,block_counts[index],&control,
				&candidate) < 0 )
			return(3);
		if ( PocCheckExact(&buffers) < 0 )
		{
			fprintf(stderr,"consumer exactness failed blocks=%u\n",
				block_counts[index]);
			return(4);
		}
		printf("persistent_blocks=%u exact_consumer=true control_median_ms=%.6f candidate_median_ms=%.6f gain_percent=%.6f control_p10_ms=%.6f control_p90_ms=%.6f candidate_p10_ms=%.6f candidate_p90_ms=%.6f control_mean_ms=%.6f candidate_mean_ms=%.6f\n",
			block_counts[index],control.median_ms,candidate.median_ms,
			100.0f * ((control.median_ms / candidate.median_ms) - 1.0f),
			control.p10_ms,control.p90_ms,candidate.p10_ms,candidate.p90_ms,
			control.mean_ms,candidate.mean_ms);
	}
	return(0);
}
