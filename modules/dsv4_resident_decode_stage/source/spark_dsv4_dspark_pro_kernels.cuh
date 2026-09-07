
#if defined(SPARK_DSV4_MODEL_BUILD) && (SPARK_DSV4_MODEL_MTP_LAYER_COUNT > 0u)

static __global__ void SparkDsv4DSparkMeanReductionKernel(
	const uint16_t *taps_bf16,
	uint16_t *mean_bf16,
	uint32_t tap_count,uint32_t stream_count,uint32_t dimension)
{
	uint32_t element_index,tap,stream_index;
	float sum;
	for (element_index = blockIdx.x * blockDim.x + threadIdx.x;
	     element_index < tap_count * dimension;
	     element_index += gridDim.x * blockDim.x)
	{
		tap = element_index / dimension;
		sum = 0.0f;
		for (stream_index = 0u; stream_index < stream_count; stream_index++)
			sum += SparkLmBf16ToFloat(taps_bf16,
				((uint64_t)tap * stream_count + stream_index) * dimension +
					(element_index - tap * dimension));
		SparkLmFloatToBf16(mean_bf16,element_index,sum / (float)stream_count);
	}
}

extern "C" cudaError_t SparkDsv4DSparkLaunchMeanReduction(cudaStream_t stream,
	const void *taps_bf16,void *mean_bf16,uint32_t tap_count,
	uint32_t stream_count,uint32_t dimension,uint32_t multiprocessor_count)
{
	uint32_t blocks;
	if ( stream == 0 || taps_bf16 == 0 || mean_bf16 == 0 ||
		tap_count == 0u || stream_count == 0u || dimension == 0u ||
		multiprocessor_count == 0u )
		return(cudaErrorInvalidValue);
	blocks = ((uint64_t)tap_count * dimension + SPARK_LM_CTA_THREADS - 1u) /
		SPARK_LM_CTA_THREADS;
	if ( blocks > multiprocessor_count )
		blocks = multiprocessor_count;
	SparkDsv4DSparkMeanReductionKernel<<<blocks,SPARK_LM_CTA_THREADS,0u,
		stream>>>((const uint16_t *)taps_bf16,(uint16_t *)mean_bf16,
		tap_count,stream_count,dimension);
	return(cudaGetLastError());
}

static __global__ void SparkDsv4DSparkMainKvWriteKernel(
	const uint16_t *kv_bf16,
	uint16_t *window_bf16,
	uint32_t dimension,uint32_t window_tokens,uint32_t seq_pos)
{
	uint32_t slot = seq_pos % window_tokens;
	uint32_t element_index;
	for (element_index = blockIdx.x * blockDim.x + threadIdx.x;
	     element_index < dimension;
	     element_index += gridDim.x * blockDim.x)
		SparkLmFloatToBf16(window_bf16,
			((uint64_t)slot * dimension) + element_index,
			SparkLmBf16ToFloat(kv_bf16,element_index));
}

extern "C" cudaError_t SparkDsv4DSparkLaunchMainKvWrite(cudaStream_t stream,
	const void *kv_bf16,void *window_bf16,uint32_t dimension,
	uint32_t window_tokens,uint32_t seq_pos)
{
	uint32_t blocks;
	if ( stream == 0 || kv_bf16 == 0 || window_bf16 == 0 ||
		dimension == 0u || window_tokens == 0u )
		return(cudaErrorInvalidValue);
	blocks = (dimension + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS;
	SparkDsv4DSparkMainKvWriteKernel<<<blocks,SPARK_LM_CTA_THREADS,0u,
		stream>>>((const uint16_t *)kv_bf16,(uint16_t *)window_bf16,
		dimension,window_tokens,seq_pos);
	return(cudaGetLastError());
}

static __global__ void SparkDsv4DSparkConfidenceKernel(
	const uint16_t *features_bf16,
	const uint16_t *weight_bf16,
	float bias,float *conf_out)
{
	__shared__ float warp_sums[SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES];
	uint32_t dimension,row = blockIdx.x;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	float local = 0.0f,reduced,logit;
	dimension = SPARK_DSV4_MODEL_HIDDEN_DIMENSION + SPARK_DSV4_MODEL_DSPARK_MARKOV_RANK;
	for (uint32_t k = lane; k < dimension; k += SPARK_LM_WARP_LANES)
		local = fmaf(SparkLmBf16ToFloat(features_bf16,
			((uint64_t)row * dimension) + k),
			SparkLmBf16ToFloat(weight_bf16,k),local);
	reduced = SparkLmWarpReduceSum(local);
	if ( lane == 0u )
		warp_sums[warp] = reduced;
	__syncthreads();
	if ( warp == 0u )
	{
		reduced = (lane < SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES)
			? warp_sums[lane] : 0.0f;
		reduced = SparkLmWarpReduceSum(reduced);
		if ( lane == 0u )
		{
			logit = reduced + bias;
			conf_out[row] = 1.0f / (1.0f + __expf(-logit));
		}
	}
}

extern "C" cudaError_t SparkDsv4DSparkLaunchConfidence(cudaStream_t stream,
	const void *features_bf16,const void *weight_bf16,float bias,
	float *conf_out,uint32_t rows)
{
	if ( stream == 0 || features_bf16 == 0 || weight_bf16 == 0 ||
		conf_out == 0 || rows == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv4DSparkConfidenceKernel<<<rows,SPARK_LM_CTA_THREADS,0u,stream>>>(
		(const uint16_t *)features_bf16,(const uint16_t *)weight_bf16,
		bias,conf_out);
	return(cudaGetLastError());
}

#endif
