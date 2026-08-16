/*
 * DSpark draft kernels - DSV4 Flash. The draft runs on ONE rank (the final
 * head rank, which owns the full lm_head and receives the full-width hidden
 * taps locally), so every kernel below is full-width and communication-free.
 *
 * Reference: inference/model.py DSparkBlock / DSparkAttention /
 * DSparkMarkovHead (DeepSeek-V4-Flash-0731 @ 7872f01b).
 *
 * Draft attention: for each of the BLOCK_SIZE positions and each head group,
 * online softmax over the sequence's sliding-window ring (all SW_TOKENS
 * slots, ring order - the reference attends the full ring, rotation-free)
 * plus the block's own BLOCK_SIZE kv vectors, with the learned attn_sink
 * added to the softmax denominator (no causal mask inside the block, no
 * sink in the numerator - exactly the reference semantics).
 */

static __global__ void SparkDsv4DsparkAttentionKernel(
	const uint16_t *q_bf16,
	const uint16_t *kv_cache_bf16,
	uint64_t lane_stride_elements,
	uint32_t lane_index,
	const uint16_t *block_kv_bf16,
	const float *sink_f32,
	float scale,
	uint16_t *out_bf16,
	uint32_t block_size,
	uint32_t head_count,
	uint32_t head_dim,
	uint32_t window_tokens)
{
	static const uint32_t heads_per_cta = SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA;
	static const uint32_t maximum_pairs_per_lane =
		SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION /
		(2u * SPARK_LM_WARP_LANES);
	extern __shared__ unsigned char grouped_attention_shared[];
	float *query_shared = reinterpret_cast<float *>(grouped_attention_shared);
	float *merge_accumulator = query_shared + (heads_per_cta * head_dim);
	float running_max[heads_per_cta];
	float running_denominator[heads_per_cta];
	float2 accumulator[heads_per_cta][maximum_pairs_per_lane];
	uint32_t position = blockIdx.x;
	uint32_t first_head = blockIdx.y * heads_per_cta;
	uint32_t active_head_count = head_count - first_head;
	uint32_t warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane_index_warp = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t pairs_per_lane =
		((head_dim >> 1u) + SPARK_LM_WARP_LANES - 1u) /
		SPARK_LM_WARP_LANES;
	uint32_t local_head,pair_index,element_index,value_pair_index,slot;
	float2 selected_values[maximum_pairs_per_lane];

	if ( position >= block_size || first_head >= head_count )
		return;
	if ( active_head_count > heads_per_cta )
		active_head_count = heads_per_cta;

	element_index = threadIdx.x;
	while ( element_index < active_head_count * head_dim )
	{
		local_head = element_index / head_dim;
		query_shared[element_index] = SparkLmBf16ToFloat(q_bf16,
			(((uint64_t)position * head_count) + first_head + local_head) *
				head_dim + (element_index - local_head * head_dim));
		element_index += blockDim.x;
	}
	element_index = threadIdx.x;
	while ( element_index < heads_per_cta * SPARK_LM_CTA_WARPS * head_dim )
	{
		merge_accumulator[element_index] = 0.0f;
		element_index += blockDim.x;
	}
	for (local_head = 0u; local_head < heads_per_cta; ++local_head)
	{
		running_max[local_head] = -3.0e38f;
		running_denominator[local_head] = 0.0f;
		for (pair_index = 0u;
		     pair_index < maximum_pairs_per_lane;
		     ++pair_index)
			accumulator[local_head][pair_index] = make_float2(0.0f, 0.0f);
	}
	__syncthreads();

	/* window ring slots + block kv vectors, one fused online-softmax pass */
	for (slot = warp_index; slot < window_tokens + block_size;
	     slot += SPARK_LM_CTA_WARPS)
	{
		float local_logit[heads_per_cta];
		float logit[heads_per_cta];
		float rescale[heads_per_cta];
		float weight[heads_per_cta];
		uint64_t cache_vector_base;

		for (local_head = 0u; local_head < heads_per_cta; ++local_head)
			local_logit[local_head] = 0.0f;
		for (pair_index = 0u; pair_index < pairs_per_lane; ++pair_index)
		{
			value_pair_index = (pair_index * SPARK_LM_WARP_LANES) + lane_index_warp;
			selected_values[pair_index] = make_float2(0.0f, 0.0f);
			if ( value_pair_index < (head_dim >> 1u) )
			{
				const uint16_t *source;
				uint64_t vector_base;
				if ( slot < window_tokens )
				{
					source = kv_cache_bf16;
					vector_base = ((uint64_t)lane_index * lane_stride_elements) +
						((uint64_t)slot * head_dim);
				}
				else
				{
					source = block_kv_bf16;
					vector_base = (uint64_t)(slot - window_tokens) * head_dim;
				}
				selected_values[pair_index] = SparkLmLoadBf16Pair(
					source, (vector_base >> 1u) + value_pair_index);
				{
					uint32_t query_element = value_pair_index << 1u;
					for (local_head = 0u; local_head < active_head_count; ++local_head)
					{
						local_logit[local_head] = fmaf(
							query_shared[(local_head * head_dim) + query_element],
							selected_values[pair_index].x,
							local_logit[local_head]);
						local_logit[local_head] = fmaf(
							query_shared[(local_head * head_dim) + query_element + 1u],
							selected_values[pair_index].y,
							local_logit[local_head]);
					}
				}
			}
		}
		for (local_head = 0u; local_head < active_head_count; ++local_head)
		{
			logit[local_head] = __shfl_sync(0xffffffffu,
				SparkLmWarpReduceSum(local_logit[local_head]), 0) * scale;
			rescale[local_head] = 0.0f;
			weight[local_head] = 0.0f;
			if ( lane_index_warp == 0u )
			{
				rescale[local_head] =
					logit[local_head] > running_max[local_head]
					? __expf(running_max[local_head] - logit[local_head])
					: 1.0f;
				weight[local_head] =
					logit[local_head] > running_max[local_head]
					? 1.0f
					: __expf(logit[local_head] - running_max[local_head]);
				running_max[local_head] =
					logit[local_head] > running_max[local_head]
					? logit[local_head] : running_max[local_head];
				running_denominator[local_head] = fmaf(
					running_denominator[local_head],
					rescale[local_head], weight[local_head]);
			}
			rescale[local_head] = __shfl_sync(0xffffffffu,rescale[local_head],0);
			weight[local_head] = __shfl_sync(0xffffffffu,weight[local_head],0);
		}
		for (pair_index = 0u; pair_index < pairs_per_lane; ++pair_index)
		{
			value_pair_index = (pair_index * SPARK_LM_WARP_LANES) + lane_index_warp;
			if ( value_pair_index < (head_dim >> 1u) )
			{
				uint32_t query_element = value_pair_index << 1u;
				for (local_head = 0u; local_head < active_head_count; ++local_head)
				{
					accumulator[local_head][pair_index].x =
						fmaf(accumulator[local_head][pair_index].x,
							rescale[local_head],
							query_shared[(local_head * head_dim) + query_element] *
								weight[local_head]);
					accumulator[local_head][pair_index].y =
						fmaf(accumulator[local_head][pair_index].y,
							rescale[local_head],
							query_shared[(local_head * head_dim) + query_element + 1u] *
								weight[local_head]);
				}
			}
		}
	}
	/* attn_sink joins the denominator only (reference semantics) */
	for (local_head = 0u; local_head < active_head_count; ++local_head)
		running_denominator[local_head] += __expf(
			__ldg(&sink_f32[first_head + local_head]) -
			running_max[local_head]);
	for (pair_index = 0u; pair_index < pairs_per_lane; ++pair_index)
	{
		value_pair_index = (pair_index * SPARK_LM_WARP_LANES) + lane_index_warp;
		if ( value_pair_index < (head_dim >> 1u) )
		{
			uint32_t query_element = value_pair_index << 1u;
			for (local_head = 0u; local_head < active_head_count; ++local_head)
			{
				float inverse = 1.0f / running_denominator[local_head];
				merge_accumulator[(local_head * head_dim) + query_element] =
					accumulator[local_head][pair_index].x * inverse;
				merge_accumulator[(local_head * head_dim) + query_element + 1u] =
					accumulator[local_head][pair_index].y * inverse;
			}
		}
	}
	__syncthreads();
	element_index = threadIdx.x;
	while ( element_index < active_head_count * head_dim )
	{
		local_head = element_index / head_dim;
		SparkLmStoreBf16(out_bf16,
			(((uint64_t)position * head_count) + first_head + local_head) *
				head_dim + (element_index - local_head * head_dim),
			merge_accumulator[element_index]);
		element_index += blockDim.x;
	}
}

extern "C" cudaError_t SparkDsv4LaunchDsparkAttention(cudaStream_t stream,
	const void *q_bf16,const void *kv_cache_bf16,uint64_t lane_stride_elements,
	uint32_t lane_index,const void *block_kv_bf16,const float *sink_f32,
	float scale,void *out_bf16,uint32_t block_size,uint32_t head_count,
	uint32_t head_dim,uint32_t window_tokens)
{
	static const uint32_t heads_per_cta = SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA;
	dim3 grid;
	size_t shared_bytes;
	if ( stream == 0 || q_bf16 == 0 || kv_cache_bf16 == 0 ||
		block_kv_bf16 == 0 || sink_f32 == 0 || out_bf16 == 0 ||
		block_size == 0u || head_count == 0u || head_dim == 0u ||
		(head_dim & 1u) != 0u || window_tokens == 0u )
		return(cudaErrorInvalidValue);
	grid = dim3(block_size,(head_count + heads_per_cta - 1u) / heads_per_cta);
	shared_bytes = heads_per_cta * head_dim * sizeof(float) +
		heads_per_cta * SPARK_LM_CTA_WARPS * head_dim * sizeof(float);
	SparkDsv4DsparkAttentionKernel<<<grid,SPARK_LM_CTA_THREADS,shared_bytes,
		stream>>>((const uint16_t *)q_bf16,(const uint16_t *)kv_cache_bf16,
		lane_stride_elements,lane_index,(const uint16_t *)block_kv_bf16,
		sink_f32,scale,(uint16_t *)out_bf16,block_size,head_count,head_dim,
		window_tokens);
	return(cudaGetLastError());
}

/*
 * Markov logits bias: bias = markov_w2 (vocab x rank) . embed (rank),
 * accumulated into the draft head logits (bf16 upcast to f32).
 */
static __global__ void SparkDsv4DsparkMarkovBiasAccumKernel(
	const uint16_t *logits_bf16,const uint16_t *markov_w2_bf16,
	const uint16_t *markov_embed_bf16,float *logits_f32,
	uint32_t vocab_offset,uint32_t shard_count,uint32_t rank,uint32_t position)
{
	uint32_t element_index;
	for (element_index = blockIdx.x * blockDim.x + threadIdx.x;
		element_index < shard_count;
		element_index += gridDim.x * blockDim.x)
	{
		float bias = 0.0f;
		uint32_t k;
		for (k = 0u; k < rank; k++)
			bias = fmaf(
				SparkLmBf16ToFloat(markov_w2_bf16,
					((uint64_t)(vocab_offset + element_index) * rank) + k),
				SparkLmBf16ToFloat(markov_embed_bf16,k),bias);
		logits_f32[((uint64_t)position * shard_count) + element_index] =
			SparkLmBf16ToFloat(logits_bf16,
				((uint64_t)position * shard_count) + element_index) + bias;
	}
}

extern "C" cudaError_t SparkDsv4LaunchDsparkMarkovBiasAccum(cudaStream_t stream,
	const void *logits_bf16,const void *markov_w2_bf16,
	const void *markov_embed_bf16,float *logits_f32,uint32_t vocab_offset,
	uint32_t shard_count,uint32_t rank,uint32_t position,
	uint32_t multiprocessor_count)
{
	if ( stream == 0 || logits_bf16 == 0 || markov_w2_bf16 == 0 ||
		markov_embed_bf16 == 0 || logits_f32 == 0 || shard_count == 0u ||
		rank == 0u || multiprocessor_count == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv4DsparkMarkovBiasAccumKernel<<<multiprocessor_count,
		SPARK_LM_CTA_THREADS,0u,stream>>>((const uint16_t *)logits_bf16,
		(const uint16_t *)markov_w2_bf16,(const uint16_t *)markov_embed_bf16,
		logits_f32,vocab_offset,shard_count,rank,position);
	return(cudaGetLastError());
}

/* Greedy argmax over a full-vocab f32 logits row. */
static __global__ void SparkDsv4DsparkArgmaxKernel(const float *logits_f32,
	uint32_t shard_count,uint32_t vocab_offset,uint32_t *output_token_id,
	float *output_score)
{
	uint32_t element_index,best = 0u;
	float best_score = -3.0e38f;
	for (element_index = threadIdx.x; element_index < shard_count;
		element_index += blockDim.x)
	{
		float score = logits_f32[element_index];
		if ( score > best_score )
		{
			best_score = score;
			best = element_index;
		}
	}
	{
		__shared__ float shared_scores[SPARK_LM_CTA_THREADS];
		__shared__ uint32_t shared_ids[SPARK_LM_CTA_THREADS];
		shared_scores[threadIdx.x] = best_score;
		shared_ids[threadIdx.x] = best;
		__syncthreads();
		for (uint32_t stride = SPARK_LM_CTA_THREADS / 2u; stride > 0u;
			stride >>= 1u)
		{
			if ( threadIdx.x < stride &&
				shared_scores[threadIdx.x + stride] > shared_scores[threadIdx.x] )
			{
				shared_scores[threadIdx.x] = shared_scores[threadIdx.x + stride];
				shared_ids[threadIdx.x] = shared_ids[threadIdx.x + stride];
			}
			__syncthreads();
		}
		if ( threadIdx.x == 0u )
		{
			*output_token_id = vocab_offset + shared_ids[0u];
			*output_score = shared_scores[0u];
		}
	}
}

extern "C" cudaError_t SparkDsv4LaunchDsparkArgmax(cudaStream_t stream,
	const float *logits_f32,uint32_t shard_count,uint32_t vocab_offset,
	uint32_t *output_token_id,float *output_score)
{
	if ( stream == 0 || logits_f32 == 0 || output_token_id == 0 ||
		output_score == 0 || shard_count == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv4DsparkArgmaxKernel<<<1u,SPARK_LM_CTA_THREADS,0u,stream>>>(
		logits_f32,shard_count,vocab_offset,output_token_id,output_score);
	return(cudaGetLastError());
}

/* Tap: mean over the hc streams of a 4-stream hidden state, per row
 * (reference: h.mean(dim=2)). */
static __global__ void SparkDsv4DsparkTapMeanKernel(
	const uint16_t *streams_bf16,uint16_t *tap_bf16,uint32_t row_count,
	uint32_t stream_count,uint32_t dimension)
{
	uint32_t element_index;
	uint32_t row,stream_index;
	float sum;
	for (element_index = blockIdx.x * blockDim.x + threadIdx.x;
		element_index < row_count * dimension;
		element_index += gridDim.x * blockDim.x)
	{
		row = element_index / dimension;
		sum = 0.0f;
		for (stream_index = 0u; stream_index < stream_count; stream_index++)
			sum += SparkLmBf16ToFloat(streams_bf16,
				((uint64_t)row * stream_count + stream_index) * dimension +
					(element_index - row * dimension));
		SparkLmStoreBf16(tap_bf16,element_index,sum / (float)stream_count);
	}
}

extern "C" cudaError_t SparkDsv4LaunchDsparkTapMean(cudaStream_t stream,
	const void *streams_bf16,void *tap_bf16,uint32_t row_count,
	uint32_t stream_count,uint32_t dimension,uint32_t multiprocessor_count)
{
	if ( stream == 0 || streams_bf16 == 0 || tap_bf16 == 0 ||
		row_count == 0u || stream_count == 0u || dimension == 0u ||
		multiprocessor_count == 0u )
		return(cudaErrorInvalidValue);
	{
		uint32_t blocks = ((uint64_t)row_count * dimension +
			SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS;
		if ( blocks > multiprocessor_count )
			blocks = multiprocessor_count;
		SparkDsv4DsparkTapMeanKernel<<<blocks,SPARK_LM_CTA_THREADS,0u,
			stream>>>((const uint16_t *)streams_bf16,(uint16_t *)tap_bf16,
			row_count,stream_count,dimension);
	}
	return(cudaGetLastError());
}

/* Expand a [rows x dim] bf16 block into [rows x streams x dim] (each stream
 * a copy), the draft block's input expansion (reference: x.unsqueeze(2)
 * .repeat(1,1,hc_mult,1)). */
static __global__ void SparkDsv4DsparkExpandStreamsKernel(
	const uint16_t *input_bf16,uint16_t *output_bf16,uint32_t row_count,
	uint32_t stream_count,uint32_t dimension)
{
	uint32_t element_index,row,stream_index;
	for (element_index = blockIdx.x * blockDim.x + threadIdx.x;
		element_index < row_count * dimension;
		element_index += gridDim.x * blockDim.x)
	{
		row = element_index / dimension;
		for (stream_index = 0u; stream_index < stream_count; stream_index++)
			SparkLmStoreBf16(output_bf16,
				((uint64_t)row * stream_count + stream_index) * dimension +
					(element_index - row * dimension),
				SparkLmBf16ToFloat(input_bf16,element_index));
	}
}

extern "C" cudaError_t SparkDsv4LaunchExpandStreams(cudaStream_t stream,
	const void *input_bf16,void *output_bf16,uint32_t row_count,
	uint32_t stream_count,uint32_t dimension,uint32_t multiprocessor_count)
{
	if ( stream == 0 || input_bf16 == 0 || output_bf16 == 0 ||
		row_count == 0u || stream_count == 0u || dimension == 0u ||
		multiprocessor_count == 0u )
		return(cudaErrorInvalidValue);
	{
		uint32_t blocks = ((uint64_t)row_count * dimension +
			SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS;
		if ( blocks > multiprocessor_count )
			blocks = multiprocessor_count;
		SparkDsv4DsparkExpandStreamsKernel<<<blocks,SPARK_LM_CTA_THREADS,0u,
			stream>>>((const uint16_t *)input_bf16,
			(uint16_t *)output_bf16,row_count,stream_count,dimension);
	}
	return(cudaGetLastError());
}
