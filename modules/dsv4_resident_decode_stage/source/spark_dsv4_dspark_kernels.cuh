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
 *
 * Each warp online-softmaxes its stride of the fused slot range and
 * accumulates VALUE vectors against QUERY logits. The CTA then merges the
 * per-warp maxima/denominators through static shared memory (global max,
 * rescaled partials, denominators summed with exactly one sink term), and
 * every warp adds its normalized partials over the FULL element range into
 * a shared accumulator staged in the query tile - one warp per phase in a
 * fixed order, so the reduction is deterministic and no partial sum is
 * dropped. The launcher raises the kernel's dynamic-shared-memory limit
 * via cudaFuncSetAttribute when a shape needs more than the 48 KiB default.
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
	/* Cross-warp softmax merge state: each warp reduces its own slot subset,
	 * then the CTA merges the partial maxima/denominators exactly once and
	 * every thread derives identical merged statistics in a fixed order. */
	__shared__ float warp_maxima[SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA][SPARK_LM_CTA_WARPS];
	__shared__ float warp_denominators[SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA][SPARK_LM_CTA_WARPS];
	__shared__ float merged_maximum[SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA];
	__shared__ float merged_denominator[SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA];
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
				for (local_head = 0u; local_head < active_head_count; ++local_head)
				{
					/* Reference semantics: the VALUE vectors weight the
					 * accumulator; the query only ever forms the logits. */
					accumulator[local_head][pair_index].x =
						fmaf(accumulator[local_head][pair_index].x,
							rescale[local_head],
							selected_values[pair_index].x * weight[local_head]);
					accumulator[local_head][pair_index].y =
						fmaf(accumulator[local_head][pair_index].y,
							rescale[local_head],
							selected_values[pair_index].y * weight[local_head]);
				}
			}
		}
	}
	/* Publish per-warp partial statistics and merge across the CTA. The
	 * attn_sink joins the MERGED denominator exactly once (reference
	 * semantics); every thread walks the warp arrays in the same fixed
	 * order, so all lanes derive identical merged maxima/denominators. */
	if ( lane_index_warp == 0u ) // only lane 0 holds live online statistics
	{
		for (local_head = 0u; local_head < heads_per_cta; ++local_head)
		{
			warp_maxima[local_head][warp_index] = running_max[local_head];
			warp_denominators[local_head][warp_index] =
				running_denominator[local_head];
		}
	}
	__syncthreads();
	for (local_head = 0u; local_head < active_head_count; ++local_head)
	{
		float maximum = warp_maxima[local_head][0];
		float denominator = 0.0f;
		uint32_t warp;
		for (warp = 1u; warp < SPARK_LM_CTA_WARPS; ++warp)
			maximum = fmaxf(maximum,warp_maxima[local_head][warp]);
		for (warp = 0u; warp < SPARK_LM_CTA_WARPS; ++warp)
			denominator += warp_denominators[local_head][warp] *
				__expf(warp_maxima[local_head][warp] - maximum);
		denominator += __expf(
			__ldg(&sink_f32[first_head + local_head]) - maximum);
		merged_maximum[local_head] = maximum;
		merged_denominator[local_head] = denominator;
	}
	__syncthreads();
	/* Cross-warp accumulation, staged through the (now dead) query tile:
	 * every warp normalizes its OWN partials against the merged statistics
	 * and adds them over the FULL element range, one warp per phase in a
	 * fixed order. Deterministic (no atomics, no race), and no partial sum
	 * is dropped - a warp-partitioned store would drop every slot whose
	 * stripe owner differs from its data owner when pairs_per_lane <
	 * SPARK_LM_CTA_WARPS. The query tile is exactly heads_per_cta x head_dim
	 * floats, so dynamic shared memory stays at the reduced footprint. */
	element_index = threadIdx.x;
	while ( element_index < active_head_count * head_dim )
	{
		query_shared[element_index] = 0.0f;
		element_index += blockDim.x;
	}
	for (uint32_t stage = 0u; stage < SPARK_LM_CTA_WARPS; ++stage)
	{
		__syncthreads();
		if ( warp_index != stage )
			continue;
		for (pair_index = 0u; pair_index < pairs_per_lane; ++pair_index)
		{
			value_pair_index =
				(pair_index * SPARK_LM_WARP_LANES) + lane_index_warp;
			if ( value_pair_index >= (head_dim >> 1u) )
				continue;
			for (local_head = 0u; local_head < active_head_count;
			     ++local_head)
			{
				/* Rescale from THIS warp's published maximum, never from
				 * the thread-private running_max: the online statistics
				 * are only maintained on lane 0, so every other lane must
				 * read the shared copy or it rescales by expf(-max) == 0
				 * and stores zeros. */
				float scale_out = __expf(
					warp_maxima[local_head][warp_index] -
					merged_maximum[local_head]) /
					merged_denominator[local_head];
				uint32_t shared_index = (local_head * head_dim) +
					(value_pair_index << 1u);
				query_shared[shared_index] = fmaf(
					accumulator[local_head][pair_index].x, scale_out,
					query_shared[shared_index]);
				query_shared[shared_index + 1u] = fmaf(
					accumulator[local_head][pair_index].y, scale_out,
					query_shared[shared_index + 1u]);
			}
		}
	}
	__syncthreads();
	element_index = threadIdx.x;
	while ( element_index < active_head_count * head_dim )
	{
		local_head = element_index / head_dim;
		SparkLmFloatToBf16(out_bf16,
			(((uint64_t)position * head_count) + first_head + local_head) *
				head_dim + (element_index - local_head * head_dim),
			query_shared[element_index]);
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
	/* Query-tile-only dynamic shared memory: the merge statistics live in
	 * static shared memory, and the same tile is reused as the staged
	 * accumulation buffer after the queries are consumed.
	 * The footprint still scales with head_dim x heads_per_cta (73,728 B
	 * at the old per-warp slab layout; 8,192 B here), so keep the opt-in
	 * path: any shape whose dynamic shared request exceeds the 48 KiB
	 * default raises the kernel's limit once via cudaFuncSetAttribute
	 * instead of failing every launch with cudaErrorInvalidValue. */
	shared_bytes = heads_per_cta * head_dim * sizeof(float);
	if ( shared_bytes > 48u * 1024u )
	{
		static size_t configured_shared_bytes = 0;
		if ( configured_shared_bytes < shared_bytes )
		{
			int device_index,optin_bytes;
			cudaError_t config_error = cudaGetDevice(&device_index);
			if ( config_error != cudaSuccess )
				return(config_error);
			config_error = cudaDeviceGetAttribute(&optin_bytes,
				cudaDevAttrMaxSharedMemoryPerBlockOptin,device_index);
			if ( config_error != cudaSuccess )
				return(config_error);
			if ( shared_bytes > (size_t)optin_bytes )
				return(cudaErrorInvalidValue);
			config_error = cudaFuncSetAttribute(
				SparkDsv4DsparkAttentionKernel,
				cudaFuncAttributeMaxDynamicSharedMemorySize,
				(int)shared_bytes);
			if ( config_error != cudaSuccess )
				return(config_error);
			/* Idempotent attribute set: a torn read of the watermark only
			 * costs one redundant configuration call. */
			configured_shared_bytes = shared_bytes;
		}
	}
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
		SparkLmFloatToBf16(tap_bf16,element_index,sum / (float)stream_count);
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
			SparkLmFloatToBf16(output_bf16,
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
