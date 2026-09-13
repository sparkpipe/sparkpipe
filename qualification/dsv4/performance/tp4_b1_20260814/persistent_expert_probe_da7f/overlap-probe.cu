#include "sparkpipe/spark_dsv4_model.h"

#define main PocRoutedOnlyMain
#include "dsv4_persistent_expert_chain_probe.cu"
#undef main

typedef struct PocSharedBuffers
{
	uint8_t *w1_payload,*w1_scale,*w3_payload,*w3_scale;
	uint8_t *w2_payload,*w2_scale;
	__nv_bfloat16 *up,*output;
} PocSharedBuffers;

#define POC_CLUSTER_BLOCK_COUNT 8u

template<uint32_t CLUSTER_BLOCK_COUNT>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS,1)
void PocClusterExpertChainKernel(
	const uint8_t *w1_payload,const uint8_t *w1_scale,
	const uint8_t *w3_payload,const uint8_t *w3_scale,
	const uint8_t *w2_payload,const uint8_t *w2_scale,
	const void *input_bf16,const uint32_t *source_rows,
	const uint32_t *row_offsets,const uint32_t *w13_prefix,
	const uint32_t *w2_prefix,void *up_bf16,void *output_bf16,
	uint32_t group_count)
{
	extern __shared__ float shared_input[];
	cg::cluster_group cluster = cg::this_cluster();
	uint32_t cluster_block = cluster.block_rank();
	uint32_t cluster_index = blockIdx.x / CLUSTER_BLOCK_COUNT;
	uint32_t w13_tiles = POC_EXPERT_DIMENSION / POC_W13_TILE_N;
	uint32_t w2_tiles = POC_HIDDEN_DIMENSION / POC_W2_TILE_N;
	uint32_t first_task = cluster_index * w13_tiles;
	uint32_t group = SparkLmSm121GroupOfTile(w13_prefix,group_count,first_task);
	uint32_t row = __ldg(row_offsets + group),tile;
	for (tile=cluster_block; tile<w13_tiles; tile+=CLUSTER_BLOCK_COUNT)
		SparkLmSm121B1ExpertW13Task<POC_W13_TILE_N>(w1_payload,w1_scale,
			w3_payload,w3_scale,input_bf16,source_rows,up_bf16,1u,group,row,
			tile * POC_W13_TILE_N,POC_HIDDEN_DIMENSION,POC_EXPERT_DIMENSION,
			10.0f,shared_input);
	cluster.sync();
	for (tile=cluster_block; tile<w2_tiles; tile+=CLUSTER_BLOCK_COUNT)
		SparkLmSm121B1ExpertW2Task<POC_W2_TILE_N>(w2_payload,w2_scale,
			up_bf16,output_bf16,group,row,tile * POC_W2_TILE_N,
			POC_EXPERT_DIMENSION,POC_HIDDEN_DIMENSION,shared_input);
}

static int32_t PocAllocateShared(PocSharedBuffers *shared,cudaStream_t stream)
{
	uint64_t w13_payload_bytes = (uint64_t)POC_EXPERT_DIMENSION *
		POC_HIDDEN_DIMENSION;
	uint64_t w13_scale_bytes = (uint64_t)POC_EXPERT_DIMENSION *
		(POC_HIDDEN_DIMENSION / 128u);
	uint64_t w2_payload_bytes = (uint64_t)POC_HIDDEN_DIMENSION *
		POC_EXPERT_DIMENSION;
	uint64_t w2_scale_bytes = (uint64_t)POC_HIDDEN_DIMENSION *
		(POC_EXPERT_DIMENSION / 128u);
	memset(shared,0,sizeof(*shared));
	POC_CUDA(cudaMalloc(&shared->w1_payload,w13_payload_bytes));
	POC_CUDA(cudaMalloc(&shared->w1_scale,w13_scale_bytes));
	POC_CUDA(cudaMalloc(&shared->w3_payload,w13_payload_bytes));
	POC_CUDA(cudaMalloc(&shared->w3_scale,w13_scale_bytes));
	POC_CUDA(cudaMalloc(&shared->w2_payload,w2_payload_bytes));
	POC_CUDA(cudaMalloc(&shared->w2_scale,w2_scale_bytes));
	POC_CUDA(cudaMalloc(&shared->up,
		POC_EXPERT_DIMENSION * sizeof(uint16_t)));
	POC_CUDA(cudaMalloc(&shared->output,
		POC_HIDDEN_DIMENSION * sizeof(uint16_t)));
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(shared->w1_payload,
		w13_payload_bytes,UINT32_C(0x91a42d73),0u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(shared->w1_scale,
		w13_scale_bytes,UINT32_C(0xc354f809),1u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(shared->w3_payload,
		w13_payload_bytes,UINT32_C(0x2efb618d),0u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(shared->w3_scale,
		w13_scale_bytes,UINT32_C(0x78d09543),1u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(shared->w2_payload,
		w2_payload_bytes,UINT32_C(0xa52c47f1),0u);
	PocFillBytesKernel<<<4096u,256u,0,stream>>>(shared->w2_scale,
		w2_scale_bytes,UINT32_C(0x0d6e3ab7),1u);
	POC_CUDA(cudaStreamSynchronize(stream));
	return(0);
}

static cudaError_t PocLaunchShared(cudaStream_t stream,
	const PocDeviceBuffers *buffers,const PocSharedBuffers *shared)
{
	cudaError_t error;
	error = SparkLmHostLaunchSm121FusedDenseW13<
		SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(stream,shared->w1_payload,
		shared->w1_scale,shared->w3_payload,shared->w3_scale,buffers->input,
		shared->up,1u,POC_HIDDEN_DIMENSION,POC_EXPERT_DIMENSION,10.0f);
	if ( error != cudaSuccess )
		return(error);
	return(SparkLmHostLaunchSm121DecodeLinear<128u,
		SPARK_DSV4_MODEL_EXPERT_ACTIVATION_CODEC>(stream,
		SPARK_LM_WEIGHT_FORMAT_FP8_E4M3,shared->w2_payload,shared->w2_scale,
		shared->up,shared->output,1u,POC_EXPERT_DIMENSION,
		POC_HIDDEN_DIMENSION));
}

static cudaError_t PocLaunchCluster(cudaStream_t stream,
	const PocDeviceBuffers *buffers)
{
	cudaLaunchAttribute attribute = {};
	cudaLaunchConfig_t config = {};
	attribute.id = cudaLaunchAttributeClusterDimension;
	attribute.val.clusterDim.x = POC_CLUSTER_BLOCK_COUNT;
	attribute.val.clusterDim.y = 1u;
	attribute.val.clusterDim.z = 1u;
	config.gridDim = dim3(POC_ACTIVE_GROUP_COUNT * POC_CLUSTER_BLOCK_COUNT);
	config.blockDim = dim3(SPARK_LM_CTA_THREADS);
	config.dynamicSmemBytes = POC_HIDDEN_DIMENSION * sizeof(float);
	config.stream = stream;
	config.attrs = &attribute;
	config.numAttrs = 1u;
	return(cudaLaunchKernelEx(&config,
		PocClusterExpertChainKernel<POC_CLUSTER_BLOCK_COUNT>,
		buffers->w1_payload,buffers->w1_scale,buffers->w3_payload,
		buffers->w3_scale,buffers->w2_payload,buffers->w2_scale,
		(const void *)buffers->input,(const uint32_t *)buffers->source_rows,
		(const uint32_t *)buffers->row_offsets,
		(const uint32_t *)buffers->w13_prefix,
		(const uint32_t *)buffers->w2_prefix,(void *)buffers->candidate_up,
		(void *)buffers->candidate_output,(uint32_t)POC_GROUP_COUNT));
}

static int32_t PocCheckClusterExact(cudaStream_t stream,
	const cudaDeviceProp *properties,const PocDeviceBuffers *buffers,
	uint64_t *up_digest,uint64_t *output_digest)
{
	uint64_t up_elements = (uint64_t)POC_ACTIVE_GROUP_COUNT *
		POC_EXPERT_DIMENSION;
	uint64_t output_elements = (uint64_t)POC_ACTIVE_GROUP_COUNT *
		POC_HIDDEN_DIMENSION;
	std::vector<uint16_t> control_up(up_elements),candidate_up(up_elements);
	std::vector<uint16_t> control_output(output_elements);
	std::vector<uint16_t> candidate_output(output_elements);
	uint64_t index,digest;
	if ( PocLaunchControl(stream,properties,buffers) != cudaSuccess ||
		PocLaunchCluster(stream,buffers) != cudaSuccess ||
		cudaStreamSynchronize(stream) != cudaSuccess )
		return(-1);
	if ( cudaMemcpy(control_up.data(),buffers->control_up,
			up_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(candidate_up.data(),buffers->candidate_up,
			up_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(control_output.data(),buffers->control_output,
			output_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(candidate_output.data(),buffers->candidate_output,
			output_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(-2);
	if ( memcmp(control_up.data(),candidate_up.data(),
			up_elements * sizeof(uint16_t)) != 0 ||
		memcmp(control_output.data(),candidate_output.data(),
			output_elements * sizeof(uint16_t)) != 0 )
		return(-3);
	digest = UINT64_C(1469598103934665603);
	for (index=0u; index<up_elements; index++)
		digest = (digest ^ control_up[index]) * UINT64_C(1099511628211);
	*up_digest = digest;
	digest = UINT64_C(1469598103934665603);
	for (index=0u; index<output_elements; index++)
		digest = (digest ^ control_output[index]) * UINT64_C(1099511628211);
	*output_digest = digest;
	return(0);
}

static int32_t PocTimeOverlap(cudaStream_t coordinator,cudaStream_t routed,
	cudaStream_t shared_stream,cudaEvent_t start,cudaEvent_t stop,
	cudaEvent_t routed_done,cudaEvent_t shared_done,
	const cudaDeviceProp *properties,const PocDeviceBuffers *buffers,
	const PocSharedBuffers *shared,uint32_t candidate,uint32_t block_count,
	float *elapsed_ms)
{
	PocFlush(coordinator,buffers);
	if ( cudaEventRecord(start,coordinator) != cudaSuccess ||
		cudaStreamWaitEvent(routed,start,0u) != cudaSuccess ||
		cudaStreamWaitEvent(shared_stream,start,0u) != cudaSuccess )
		return(-1);
	if ( PocLaunchShared(shared_stream,buffers,shared) != cudaSuccess )
		return(-2);
	if ( (candidate == 0u ? PocLaunchControl(routed,properties,buffers) :
		(candidate == 1u ? PocLaunchPersistent(routed,buffers,block_count) :
		PocLaunchCluster(routed,buffers))) != cudaSuccess )
		return(-3);
	if ( cudaEventRecord(routed_done,routed) != cudaSuccess ||
		cudaEventRecord(shared_done,shared_stream) != cudaSuccess ||
		cudaStreamWaitEvent(coordinator,routed_done,0u) != cudaSuccess ||
		cudaStreamWaitEvent(coordinator,shared_done,0u) != cudaSuccess ||
		cudaEventRecord(stop,coordinator) != cudaSuccess ||
		cudaEventSynchronize(stop) != cudaSuccess ||
		cudaEventElapsedTime(elapsed_ms,start,stop) != cudaSuccess )
		return(-4);
	return(0);
}

static int32_t PocMeasureOverlap(cudaStream_t coordinator,
	cudaStream_t routed,cudaStream_t shared_stream,cudaEvent_t start,
	cudaEvent_t stop,cudaEvent_t routed_done,cudaEvent_t shared_done,
	const cudaDeviceProp *properties,const PocDeviceBuffers *buffers,
	const PocSharedBuffers *shared,uint32_t candidate_kind,
	uint32_t block_count,PocTiming *control,PocTiming *candidate)
{
	std::vector<float> control_times,candidate_times;
	float elapsed;
	uint32_t repetition,warmup;
	for (warmup=0u; warmup<POC_WARMUP_COUNT; warmup++)
		if ( PocTimeOverlap(coordinator,routed,shared_stream,start,stop,
			routed_done,shared_done,properties,buffers,shared,
			(warmup & 1u) != 0u ? candidate_kind : 0u,
			block_count,&elapsed) < 0 )
			return(-1);
	for (repetition=0u; repetition<POC_REPETITION_COUNT; repetition++)
	{
		if ( (repetition & 1u) == 0u )
		{
			if ( PocTimeOverlap(coordinator,routed,shared_stream,start,stop,
				routed_done,shared_done,properties,buffers,shared,0u,
				block_count,&elapsed) < 0 )
				return(-2);
			control_times.push_back(elapsed);
			if ( PocTimeOverlap(coordinator,routed,shared_stream,start,stop,
				routed_done,shared_done,properties,buffers,shared,candidate_kind,
				block_count,&elapsed) < 0 )
				return(-3);
			candidate_times.push_back(elapsed);
		}
		else
		{
			if ( PocTimeOverlap(coordinator,routed,shared_stream,start,stop,
				routed_done,shared_done,properties,buffers,shared,candidate_kind,
				block_count,&elapsed) < 0 )
				return(-4);
			candidate_times.push_back(elapsed);
			if ( PocTimeOverlap(coordinator,routed,shared_stream,start,stop,
				routed_done,shared_done,properties,buffers,shared,0u,
				block_count,&elapsed) < 0 )
				return(-5);
			control_times.push_back(elapsed);
		}
	}
	*control = PocSummarize(control_times);
	*candidate = PocSummarize(candidate_times);
	return(0);
}

int main(void)
{
	PocDeviceBuffers buffers;
	PocSharedBuffers shared;
	PocTiming control,candidate;
	cudaDeviceProp properties;
	cudaEvent_t start,stop,routed_done,shared_done;
	cudaStream_t coordinator,routed,shared_stream;
	uint64_t up_digest,output_digest;
	uint32_t blocks_per_sm,block_count;
	int32_t maximum_blocks_per_sm;
	POC_CUDA(cudaGetDeviceProperties(&properties,0));
	POC_CUDA(cudaStreamCreateWithFlags(&coordinator,cudaStreamNonBlocking));
	POC_CUDA(cudaStreamCreateWithFlags(&routed,cudaStreamNonBlocking));
	POC_CUDA(cudaStreamCreateWithFlags(&shared_stream,cudaStreamNonBlocking));
	POC_CUDA(cudaEventCreate(&start));
	POC_CUDA(cudaEventCreate(&stop));
	POC_CUDA(cudaEventCreateWithFlags(&routed_done,cudaEventDisableTiming));
	POC_CUDA(cudaEventCreateWithFlags(&shared_done,cudaEventDisableTiming));
	POC_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&maximum_blocks_per_sm,
		PocPersistentExpertChainKernel<POC_W13_TILE_N,POC_W2_TILE_N>,
		SPARK_LM_CTA_THREADS,POC_HIDDEN_DIMENSION * sizeof(float)));
	if ( PocAllocate(&buffers,coordinator) < 0 ||
		PocAllocateShared(&shared,coordinator) < 0 )
		return(2);
	printf("device=%s sm_count=%d concurrent_shared_expert=true max_resident_blocks_per_sm=%d repetitions=%u\n",
		properties.name,properties.multiProcessorCount,maximum_blocks_per_sm,
		POC_REPETITION_COUNT);
	for (blocks_per_sm=2u; blocks_per_sm<=3u; blocks_per_sm++)
	{
		if ( blocks_per_sm > (uint32_t)maximum_blocks_per_sm )
			continue;
		block_count = blocks_per_sm * properties.multiProcessorCount;
		if ( PocCheckExact(coordinator,&properties,&buffers,block_count,
				&up_digest,&output_digest) < 0 )
			return(3);
		if ( PocMeasureOverlap(coordinator,routed,shared_stream,start,stop,
				routed_done,shared_done,&properties,&buffers,&shared,1u,block_count,
				&control,&candidate) < 0 )
			return(4);
		printf("blocks_per_sm=%u exact_up=true exact_output=true up_digest=%016llx output_digest=%016llx control_median_ms=%.6f candidate_median_ms=%.6f speedup=%.6f control_p10_ms=%.6f control_p90_ms=%.6f candidate_p10_ms=%.6f candidate_p90_ms=%.6f control_mean_ms=%.6f candidate_mean_ms=%.6f\n",
			blocks_per_sm,(unsigned long long)up_digest,
			(unsigned long long)output_digest,control.median_ms,
			candidate.median_ms,control.median_ms / candidate.median_ms,
			control.p10_ms,control.p90_ms,candidate.p10_ms,candidate.p90_ms,
			control.mean_ms,candidate.mean_ms);
	}
	if ( PocCheckClusterExact(coordinator,&properties,&buffers,&up_digest,
			&output_digest) < 0 )
		return(5);
	if ( PocMeasureOverlap(coordinator,routed,shared_stream,start,stop,
			routed_done,shared_done,&properties,&buffers,&shared,2u,0u,
			&control,&candidate) < 0 )
		return(6);
	printf("cluster_blocks=8 exact_up=true exact_output=true up_digest=%016llx output_digest=%016llx control_median_ms=%.6f candidate_median_ms=%.6f speedup=%.6f control_p10_ms=%.6f control_p90_ms=%.6f candidate_p10_ms=%.6f candidate_p90_ms=%.6f control_mean_ms=%.6f candidate_mean_ms=%.6f\n",
		(unsigned long long)up_digest,(unsigned long long)output_digest,
		control.median_ms,candidate.median_ms,
		control.median_ms / candidate.median_ms,control.p10_ms,
		control.p90_ms,candidate.p10_ms,candidate.p90_ms,control.mean_ms,
		candidate.mean_ms);
	return(0);
}
