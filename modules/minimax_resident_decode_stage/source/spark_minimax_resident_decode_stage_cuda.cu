#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "sparkpipe/spark_minimax_resident_decode_stage_firmware.h"

#define SPARK_MINIMAX_KERNEL_HIDDEN SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION
#define SPARK_MINIMAX_KERNEL_HEAD_DIM SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_DIMENSION
#define SPARK_MINIMAX_KERNEL_ROPE_HALF (SPARK_MINIMAX_KERNEL_HEAD_DIM / 2u)
#define SPARK_MINIMAX_KERNEL_HEADS_PER_KV_HEAD \
	(SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_COUNT / \
	 SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_HEAD_COUNT)
#define SPARK_MINIMAX_KERNEL_THREADS 256u
#define SPARK_MINIMAX_KERNEL_ROPE_THETA 5000000.0f
#define SPARK_MINIMAX_KERNEL_NEG_INFINITY (-3.402823466e38f)

__device__ __forceinline__ float SparkMinimaxBf16ToFloat(const __nv_bfloat16 *pointer,uint64_t index)
{
	return(__bfloat162float(pointer[index]));
}

__device__ __forceinline__ __nv_bfloat16 SparkMinimaxFloatToBf16(float value)
{
	return(__float2bfloat16_rn(value));
}

__device__ __forceinline__ uint64_t SparkMinimaxSortableScore(float score,uint32_t payload)
{
	uint32_t bits = __float_as_uint(score);
	bits = (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
	return(((uint64_t)bits << 32) | (uint64_t)payload);
}

__global__ void SparkMinimaxEmbeddingGatherKernel(const uint32_t *token_ids,const __nv_bfloat16 *embedding_bf16,__nv_bfloat16 *hidden_bf16,uint32_t row_count)
{
	uint32_t column;
	uint32_t token;
	if ( blockIdx.x >= row_count )
		return;
	token = token_ids[blockIdx.x];
	for (column = threadIdx.x; column < SPARK_MINIMAX_KERNEL_HIDDEN; column += blockDim.x)
		hidden_bf16[(uint64_t)blockIdx.x * SPARK_MINIMAX_KERNEL_HIDDEN + column] =
			embedding_bf16[(uint64_t)token * SPARK_MINIMAX_KERNEL_HIDDEN + column];
}

__global__ void SparkMinimaxRmsNormKernel(const __nv_bfloat16 *input_bf16,const __nv_bfloat16 *gain_bf16,__nv_bfloat16 *output_bf16,uint32_t row_count,uint32_t dimension,float epsilon)
{
	uint32_t column;
	float summed = 0.0f;
	float scale;
	__shared__ float variance_shared;
	if ( blockIdx.x >= row_count )
		return;
	for (column = threadIdx.x; column < dimension; column += blockDim.x)
	{
		float value = SparkMinimaxBf16ToFloat(input_bf16,(uint64_t)blockIdx.x * dimension + column);
		summed += value * value;
	}
	summed += __shfl_down_sync(0xffffffffu,summed,16u);
	summed += __shfl_down_sync(0xffffffffu,summed,8u);
	summed += __shfl_down_sync(0xffffffffu,summed,4u);
	summed += __shfl_down_sync(0xffffffffu,summed,2u);
	summed += __shfl_down_sync(0xffffffffu,summed,1u);
	if ( (threadIdx.x & 31u) == 0u )
		variance_shared = 0.0f;
	__syncthreads();
	if ( (threadIdx.x & 31u) == 0u )
		atomicAdd(&variance_shared,summed);
	__syncthreads();
	scale = rsqrtf(variance_shared / (float)dimension + epsilon);
	for (column = threadIdx.x; column < dimension; column += blockDim.x)
	{
		float value = SparkMinimaxBf16ToFloat(input_bf16,(uint64_t)blockIdx.x * dimension + column);
		float gain = SparkMinimaxBf16ToFloat(gain_bf16,column);
		output_bf16[(uint64_t)blockIdx.x * dimension + column] = SparkMinimaxFloatToBf16(value * scale * gain);
	}
}

__global__ void SparkMinimaxResidualAddKernel(__nv_bfloat16 *hidden_bf16,const __nv_bfloat16 *delta_bf16,uint32_t row_count,uint32_t dimension)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t elements = (uint64_t)row_count * dimension;
	if ( index >= elements )
		return;
	hidden_bf16[index] = SparkMinimaxFloatToBf16(SparkMinimaxBf16ToFloat(hidden_bf16,index) + SparkMinimaxBf16ToFloat(delta_bf16,index));
}

__global__ void SparkMinimaxLinearKernel(const __nv_bfloat16 *weight_bf16,const __nv_bfloat16 *input_bf16,__nv_bfloat16 *output_bf16,uint32_t row_count,uint32_t output_dimension,uint32_t input_dimension)
{
	uint32_t row = blockIdx.z;
	uint32_t output_index = blockIdx.y * (blockDim.x / 32u) + (threadIdx.x / 32u);
	float accumulated = 0.0f;
	if ( row >= row_count || output_index >= output_dimension )
		return;
	for (uint32_t column = threadIdx.x % 32u; column < input_dimension; column += 32u)
		accumulated += SparkMinimaxBf16ToFloat(weight_bf16,(uint64_t)output_index * input_dimension + column) *
			SparkMinimaxBf16ToFloat(input_bf16,(uint64_t)row * input_dimension + column);
	accumulated += __shfl_down_sync(0xffffffffu,accumulated,16u);
	accumulated += __shfl_down_sync(0xffffffffu,accumulated,8u);
	accumulated += __shfl_down_sync(0xffffffffu,accumulated,4u);
	accumulated += __shfl_down_sync(0xffffffffu,accumulated,2u);
	accumulated += __shfl_down_sync(0xffffffffu,accumulated,1u);
	if ( (threadIdx.x & 31u) == 0u )
		output_bf16[(uint64_t)row * output_dimension + output_index] = SparkMinimaxFloatToBf16(accumulated);
}

__global__ void SparkMinimaxHeadNormRopeKernel(__nv_bfloat16 *query_bf16,__nv_bfloat16 *key_bf16,const __nv_bfloat16 *query_norm_bf16,const __nv_bfloat16 *key_norm_bf16,float *query_roped_f32,const uint64_t *row_positions,uint32_t row_count,uint32_t local_query_head_count,uint32_t local_kv_head_count,float epsilon,uint32_t tp_rank)
{
	uint32_t row = blockIdx.y;
	uint32_t head = blockIdx.x;
	uint32_t element = threadIdx.x;
	uint32_t is_query = head < local_query_head_count ? 1u : 0u;
	uint32_t norm_head = is_query != 0u ? head : head - local_query_head_count;
	const __nv_bfloat16 *source;
	const __nv_bfloat16 *gain;
	float value;
	float normalized;
	float roped;
	__shared__ float squared[SPARK_MINIMAX_KERNEL_HEAD_DIM];
	__shared__ float exchange[SPARK_MINIMAX_KERNEL_HEAD_DIM];
	if ( row >= row_count || head >= local_query_head_count + local_kv_head_count || element >= SPARK_MINIMAX_KERNEL_HEAD_DIM )
		return;
	if ( is_query != 0u )
	{
		source = query_bf16;
		gain = query_norm_bf16;
	}
	else
	{
		source = key_bf16;
		gain = key_norm_bf16;
	}
	value = SparkMinimaxBf16ToFloat(source,((uint64_t)row * (is_query != 0u ? local_query_head_count : local_kv_head_count) + norm_head) * SPARK_MINIMAX_KERNEL_HEAD_DIM + element);
	squared[element] = value * value;
	exchange[element] = value;
	__syncthreads();
	if ( element == 0u )
	{
		float total = 0.0f;
		for (uint32_t d = 0u; d < SPARK_MINIMAX_KERNEL_HEAD_DIM; d++)
			total += squared[d];
		squared[0] = total;
	}
	__syncthreads();
	normalized = value * rsqrtf(squared[0] / (float)SPARK_MINIMAX_KERNEL_HEAD_DIM + epsilon) * SparkMinimaxBf16ToFloat(gain,element);
	{
		float position = (float)(unsigned long long)row_positions[row];
		float angle = position * powf(SPARK_MINIMAX_KERNEL_ROPE_THETA,-2.0f * (float)element / (float)SPARK_MINIMAX_KERNEL_HEAD_DIM);
		float partner = exchange[element < SPARK_MINIMAX_KERNEL_ROPE_HALF ? element + SPARK_MINIMAX_KERNEL_ROPE_HALF : element - SPARK_MINIMAX_KERNEL_ROPE_HALF];
		float cosine = cosf(angle);
		float sine = sinf(angle);
		roped = element < SPARK_MINIMAX_KERNEL_ROPE_HALF
			? normalized * cosine - partner * sine
			: normalized * cosine + partner * sine;
	}
	__syncthreads();
	if ( is_query != 0u )
		query_roped_f32[((uint64_t)row * local_query_head_count + norm_head) * SPARK_MINIMAX_KERNEL_HEAD_DIM + element] = roped;
	else
		key_bf16[((uint64_t)row * local_kv_head_count + norm_head) * SPARK_MINIMAX_KERNEL_HEAD_DIM + element] = SparkMinimaxFloatToBf16(roped);
}

__global__ void SparkMinimaxKvCacheWriteKernel(const __nv_bfloat16 *key_bf16,const __nv_bfloat16 *value_bf16,__nv_bfloat16 *kv_cache_bf16,const uint32_t *physical_block_indices,const uint32_t *lane_block_counts,const uint32_t *row_lane_indices,const uint32_t *slot_mapping,uint32_t row_count,uint32_t local_kv_dimension,uint32_t lane_stride,uint64_t block_span,uint64_t layer_block_stride,uint32_t layer_index,uint32_t kv_head_offset)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t per_row = local_kv_dimension * 2u;
	uint32_t row = (uint32_t)(index / per_row);
	uint32_t within = (uint32_t)(index % per_row);
	uint32_t lane,block,slot;
	uint64_t base;
	if ( row >= row_count )
		return;
	lane = row_lane_indices[row];
	if ( lane_block_counts[lane] == 0u )
		return;
	block = physical_block_indices[(uint64_t)lane * lane_stride];
	slot = slot_mapping[row];
	base = (uint64_t)block * block_span + (uint64_t)layer_index * layer_block_stride + (uint64_t)slot * per_row;
	if ( within < local_kv_dimension )
		kv_cache_bf16[base + within] = key_bf16[(uint64_t)row * local_kv_dimension + within];
	else
		kv_cache_bf16[base + within] = value_bf16[(uint64_t)row * local_kv_dimension + (within - local_kv_dimension)];
	(void)kv_head_offset;
}

__global__ void SparkMinimaxAttentionDecodeKernel(const float *query_roped_f32,const __nv_bfloat16 *kv_cache_bf16,const uint32_t *physical_block_indices,const uint32_t *lane_block_counts,const uint32_t *row_lane_indices,const uint32_t *slot_mapping,const uint32_t *context_lengths,__nv_bfloat16 *attended_bf16,uint32_t row_count,uint32_t local_query_head_count,uint32_t local_kv_head_count,uint32_t local_kv_head_dimension,uint32_t lane_stride,uint64_t block_span,uint64_t layer_block_stride,float epsilon,uint32_t tp_rank,uint32_t layer_index)
{
	uint32_t row = blockIdx.y;
	uint32_t head = blockIdx.x;
	uint32_t element = threadIdx.x;
	uint32_t kv_head_global,kv_head_local,lane,context;
	float maximum,running_sum;
	float accumulator;
	__shared__ float shared_score;
	__shared__ float shared_maximum;
	__shared__ float shared_denominator;
	if ( row >= row_count || head >= local_query_head_count || element >= SPARK_MINIMAX_KERNEL_HEAD_DIM )
		return;
	(void)epsilon;
	kv_head_global = (tp_rank * local_query_head_count + head) / SPARK_MINIMAX_KERNEL_HEADS_PER_KV_HEAD;
	kv_head_local = kv_head_global - tp_rank * local_kv_head_count;
	lane = row_lane_indices[row];
	context = context_lengths[row];
	if ( lane_block_counts[lane] == 0u || context == 0u )
	{
		attended_bf16[((uint64_t)row * local_query_head_count + head) * SPARK_MINIMAX_KERNEL_HEAD_DIM + element] = SparkMinimaxFloatToBf16(0.0f);
		return;
	}
	{
		float query_value = query_roped_f32[((uint64_t)row * local_query_head_count + head) * SPARK_MINIMAX_KERNEL_HEAD_DIM + element];
		maximum = SPARK_MINIMAX_KERNEL_NEG_INFINITY;
		running_sum = 0.0f;
		accumulator = 0.0f;
		for (uint32_t position = 0u; position < context; position++)
		{
			uint32_t ordinal = position / SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
			uint32_t slot_in_block = position % SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
			uint32_t block = physical_block_indices[(uint64_t)lane * lane_stride + ordinal];
			uint64_t entry_base = (uint64_t)block * block_span +
				(uint64_t)layer_index * layer_block_stride +
				(uint64_t)slot_in_block * (2u * local_kv_head_dimension * SPARK_MINIMAX_KERNEL_HEAD_DIM) +
				(uint64_t)kv_head_local * SPARK_MINIMAX_KERNEL_HEAD_DIM;
			const __nv_bfloat16 *key_entry = kv_cache_bf16 + entry_base;
			const __nv_bfloat16 *value_entry = key_entry + (uint64_t)local_kv_head_dimension * SPARK_MINIMAX_KERNEL_HEAD_DIM;
			float score = query_value * SparkMinimaxBf16ToFloat(key_entry,element);
			for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
				score += __shfl_down_sync(0xffffffffu,score,offset);
			if ( (threadIdx.x & 31u) == 0u )
				shared_score = 0.0f;
			__syncthreads();
			if ( (threadIdx.x & 31u) == 0u )
				atomicAdd(&shared_score,score);
			__syncthreads();
			score = shared_score * rsqrtf((float)SPARK_MINIMAX_KERNEL_HEAD_DIM);
			if ( element == 0u )
				shared_maximum = maximum;
			__syncthreads();
			{
				float candidate = score > shared_maximum ? score : shared_maximum;
				float rescale = expf(shared_maximum - candidate);
				float weight = expf(score - candidate);
				accumulator = accumulator * rescale + weight * SparkMinimaxBf16ToFloat(value_entry,element);
				if ( element == 0u )
				{
					shared_denominator = running_sum * rescale + weight;
					shared_maximum = candidate;
				}
				__syncthreads();
				running_sum = shared_denominator;
			}
			maximum = shared_maximum;
			__syncthreads();
		}
		attended_bf16[((uint64_t)row * local_query_head_count + head) * SPARK_MINIMAX_KERNEL_HEAD_DIM + element] =
			SparkMinimaxFloatToBf16(accumulator / running_sum);
	}
}

__global__ void SparkMinimaxAttentionPrefillKernel(const float *query_roped_f32,const __nv_bfloat16 *kv_cache_bf16,const uint32_t *physical_block_indices,const uint32_t *lane_block_counts,const uint32_t *row_lane_indices,const uint64_t *row_positions,const __nv_bfloat16 *staged_key_bf16,const __nv_bfloat16 *staged_value_bf16,__nv_bfloat16 *attended_bf16,uint32_t row_count,uint32_t local_query_head_count,uint32_t local_kv_head_count,uint32_t local_kv_head_dimension,uint32_t lane_stride,uint64_t block_span,uint64_t layer_block_stride,float epsilon,uint32_t tp_rank,uint32_t layer_index,uint64_t base_position)
{
	uint32_t row = blockIdx.y;
	uint32_t head = blockIdx.x;
	uint32_t element = threadIdx.x;
	uint32_t kv_head_global,kv_head_local,lane;
	uint64_t position = base_position;
	float maximum,running_sum,accumulator;
	__shared__ float shared_score;
	if ( row >= row_count || head >= local_query_head_count || element >= SPARK_MINIMAX_KERNEL_HEAD_DIM )
		return;
	(void)epsilon;
	position += row;
	kv_head_global = (tp_rank * local_query_head_count + head) / SPARK_MINIMAX_KERNEL_HEADS_PER_KV_HEAD;
	kv_head_local = kv_head_global - tp_rank * local_kv_head_count;
	lane = row_lane_indices[row];
	{
		float query_value = query_roped_f32[((uint64_t)row * local_query_head_count + head) * SPARK_MINIMAX_KERNEL_HEAD_DIM + element];
		maximum = SPARK_MINIMAX_KERNEL_NEG_INFINITY;
		running_sum = 0.0f;
		accumulator = 0.0f;
		for (uint64_t past = 0u; lane_block_counts[lane] != 0u && past < position; past++)
		{
			uint32_t ordinal = (uint32_t)(past / SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
			uint32_t slot_in_block = (uint32_t)(past % SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
			uint32_t block = physical_block_indices[(uint64_t)lane * lane_stride + ordinal];
			uint64_t entry_base = (uint64_t)block * block_span +
				(uint64_t)layer_index * layer_block_stride +
				(uint64_t)slot_in_block * (2u * local_kv_head_dimension * SPARK_MINIMAX_KERNEL_HEAD_DIM) +
				(uint64_t)kv_head_local * SPARK_MINIMAX_KERNEL_HEAD_DIM;
			const __nv_bfloat16 *key_entry = kv_cache_bf16 + entry_base;
			const __nv_bfloat16 *value_entry = key_entry + (uint64_t)local_kv_head_dimension * SPARK_MINIMAX_KERNEL_HEAD_DIM;
			float score = query_value * SparkMinimaxBf16ToFloat(key_entry,element);
			for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
				score += __shfl_down_sync(0xffffffffu,score,offset);
			if ( (threadIdx.x & 31u) == 0u )
				shared_score = 0.0f;
			__syncthreads();
			if ( (threadIdx.x & 31u) == 0u )
				atomicAdd(&shared_score,score);
			__syncthreads();
			score = shared_score * rsqrtf((float)SPARK_MINIMAX_KERNEL_HEAD_DIM);
			{
				float candidate = score > maximum ? score : maximum;
				float rescale = expf(maximum - candidate);
				float weight = expf(score - candidate);
				accumulator = accumulator * rescale + weight * SparkMinimaxBf16ToFloat(value_entry,element);
				running_sum = running_sum * rescale + weight;
				maximum = candidate;
			}
			__syncthreads();
		}
		for (uint32_t within = 0u; within <= row; within++)
		{
			const __nv_bfloat16 *key_entry = staged_key_bf16 + (uint64_t)within * (local_kv_head_dimension * SPARK_MINIMAX_KERNEL_HEAD_DIM);
			const __nv_bfloat16 *value_entry = staged_value_bf16 + (uint64_t)within * (local_kv_head_dimension * SPARK_MINIMAX_KERNEL_HEAD_DIM);
			float score = query_value * SparkMinimaxBf16ToFloat(key_entry,(uint64_t)kv_head_local * SPARK_MINIMAX_KERNEL_HEAD_DIM + element);
			for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
				score += __shfl_down_sync(0xffffffffu,score,offset);
			if ( (threadIdx.x & 31u) == 0u )
				shared_score = 0.0f;
			__syncthreads();
			if ( (threadIdx.x & 31u) == 0u )
				atomicAdd(&shared_score,score);
			__syncthreads();
			score = shared_score * rsqrtf((float)SPARK_MINIMAX_KERNEL_HEAD_DIM);
			{
				float candidate = score > maximum ? score : maximum;
				float rescale = expf(maximum - candidate);
				float weight = expf(score - candidate);
				accumulator = accumulator * rescale + weight * SparkMinimaxBf16ToFloat(value_entry,(uint64_t)kv_head_local * SPARK_MINIMAX_KERNEL_HEAD_DIM + element);
				running_sum = running_sum * rescale + weight;
				maximum = candidate;
			}
			__syncthreads();
		}
		attended_bf16[((uint64_t)row * local_query_head_count + head) * SPARK_MINIMAX_KERNEL_HEAD_DIM + element] =
			SparkMinimaxFloatToBf16(accumulator / running_sum);
	}
}

__global__ void SparkMinimaxSwiGluKernel(const __nv_bfloat16 *gate_bf16,const __nv_bfloat16 *up_bf16,__nv_bfloat16 *activated_bf16,uint32_t row_count,uint32_t dimension)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t elements = (uint64_t)row_count * dimension;
	if ( index >= elements )
		return;
	{
		float gate = SparkMinimaxBf16ToFloat(gate_bf16,index);
		float up = SparkMinimaxBf16ToFloat(up_bf16,index);
		activated_bf16[index] = SparkMinimaxFloatToBf16(gate / (1.0f + expf(-gate)) * up);
	}
}

__global__ void SparkMinimaxVocabArgmaxKernel(const __nv_bfloat16 *lm_head_bf16,const __nv_bfloat16 *input_bf16,uint64_t *argmax_reduce_u64,uint32_t row_count,uint32_t local_vocab_rows,uint32_t input_dimension,uint32_t local_vocab_base)
{
	uint32_t row = blockIdx.y;
	uint32_t vocab_index = blockIdx.x * blockDim.x + threadIdx.x;
	float score = SPARK_MINIMAX_KERNEL_NEG_INFINITY;
	uint32_t token = 0u;
	if ( row >= row_count )
		return;
	if ( vocab_index < local_vocab_rows )
	{
		const __nv_bfloat16 *weight_row = lm_head_bf16 + (uint64_t)vocab_index * input_dimension;
		score = 0.0f;
		for (uint32_t column = 0u; column < input_dimension; column++)
			score += SparkMinimaxBf16ToFloat(weight_row,column) * SparkMinimaxBf16ToFloat(input_bf16,(uint64_t)row * input_dimension + column);
		token = local_vocab_base + vocab_index;
	}
	for (uint32_t stride = 16u; stride > 0u; stride >>= 1u)
	{
		float other_score = __shfl_down_sync(0xffffffffu,score,stride);
		uint32_t other_token = __shfl_down_sync(0xffffffffu,token,stride);
		if ( other_score > score )
		{
			score = other_score;
			token = other_token;
		}
	}
	if ( (threadIdx.x & 31u) == 0u )
		atomicMax(&argmax_reduce_u64[row],SparkMinimaxSortableScore(score,token));
}

__global__ void SparkMinimaxArgmaxResolveKernel(const uint64_t *argmax_reduce_u64,uint32_t *token_ids,uint32_t row_count)
{
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= row_count )
		return;
	token_ids[row] = (uint32_t)(argmax_reduce_u64[row] & 0xffffffffu);
}

__global__ void SparkMinimaxTpCombineBf16Kernel(__nv_bfloat16 *destination,const __nv_bfloat16 *source,uint32_t element_count)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= element_count )
		return;
	destination[index] = SparkMinimaxFloatToBf16(SparkMinimaxBf16ToFloat(destination,index) + SparkMinimaxBf16ToFloat(source,index));
}

__global__ void SparkMinimaxTpCombineU64MaxKernel(uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= element_count )
		return;
	destination[index] = destination[index] > source[index] ? destination[index] : source[index];
}

static cudaError_t SparkMinimaxDispatchLinear(cudaStream_t stream,const SparkMinimaxLinearView *view,const void *input_bf16,void *output_bf16,uint32_t row_count)
{
	uint32_t outputs_per_block = 8u;
	dim3 grid((view->output_dimension + outputs_per_block - 1u) / outputs_per_block,outputs_per_block,row_count);
	if ( view->weight_format != SPARK_MINIMAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 || view->weight_payload_bf16 == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxLinearKernel<<<grid,SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)view->weight_payload_bf16,(const __nv_bfloat16 *)input_bf16,
		(__nv_bfloat16 *)output_bf16,row_count,view->output_dimension,view->input_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxConfigureCudaKernels(void)
{
	return(cudaSuccess);
}

extern "C" cudaError_t SparkMinimaxLaunchEmbeddingGather(cudaStream_t stream,const uint32_t *token_ids,const void *embedding_bf16,void *hidden_bf16,uint32_t row_count)
{
	SparkMinimaxEmbeddingGatherKernel<<<row_count,SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>(token_ids,(const __nv_bfloat16 *)embedding_bf16,(__nv_bfloat16 *)hidden_bf16,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchRmsNorm(cudaStream_t stream,const void *input_bf16,const void *gain_bf16,void *output_bf16,uint32_t row_count,uint32_t dimension,float epsilon)
{
	SparkMinimaxRmsNormKernel<<<row_count,SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>((const __nv_bfloat16 *)input_bf16,(const __nv_bfloat16 *)gain_bf16,(__nv_bfloat16 *)output_bf16,row_count,dimension,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchResidualAdd(cudaStream_t stream,void *hidden_bf16,const void *delta_bf16,uint32_t row_count,uint32_t dimension)
{
	uint64_t elements = (uint64_t)row_count * dimension;
	SparkMinimaxResidualAddKernel<<<(uint32_t)((elements + SPARK_MINIMAX_KERNEL_THREADS - 1u) / SPARK_MINIMAX_KERNEL_THREADS),SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>((__nv_bfloat16 *)hidden_bf16,(const __nv_bfloat16 *)delta_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchQueryKeyProjection(cudaStream_t stream,const SparkMinimaxLinearView *query,const SparkMinimaxLinearView *key,const void *input_bf16,void *query_bf16,void *key_bf16,uint32_t row_count)
{
	cudaError_t error = SparkMinimaxDispatchLinear(stream,query,input_bf16,query_bf16,row_count);
	if ( error != cudaSuccess )
		return(error);
	return(SparkMinimaxDispatchLinear(stream,key,input_bf16,key_bf16,row_count));
}

extern "C" cudaError_t SparkMinimaxLaunchValueProjection(cudaStream_t stream,const SparkMinimaxLinearView *value,const void *input_bf16,void *value_bf16,uint32_t row_count)
{
	return(SparkMinimaxDispatchLinear(stream,value,input_bf16,value_bf16,row_count));
}

extern "C" cudaError_t SparkMinimaxLaunchHeadNormRope(cudaStream_t stream,void *query_bf16,void *key_bf16,const void *query_norm_bf16,const void *key_norm_bf16,void *query_roped_f32,const uint64_t *row_positions,uint32_t row_count,float epsilon,uint32_t local_query_head_count,uint32_t local_kv_head_count)
{
	dim3 grid(local_query_head_count + local_kv_head_count,row_count);
	SparkMinimaxHeadNormRopeKernel<<<grid,SPARK_MINIMAX_KERNEL_HEAD_DIM,0,stream>>>((__nv_bfloat16 *)query_bf16,(__nv_bfloat16 *)key_bf16,
		(const __nv_bfloat16 *)query_norm_bf16,(const __nv_bfloat16 *)key_norm_bf16,(float *)query_roped_f32,
		row_positions,row_count,local_query_head_count,local_kv_head_count,epsilon,0u);
	{
		uint32_t rank = 0u;
		(void)rank;
	}
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchKvCacheWrite(cudaStream_t stream,const void *key_bf16,const void *value_bf16,void *kv_cache_bf16,const uint32_t *physical_block_indices,const uint32_t *lane_block_counts,const uint32_t *row_lane_indices,const uint32_t *slot_mapping,uint32_t row_count,uint32_t local_kv_head_dimension,uint32_t lane_stride,uint64_t block_span,uint64_t layer_block_stride,uint32_t layer_index)
{
	uint64_t elements = (uint64_t)row_count * local_kv_head_dimension * 2u;
	SparkMinimaxKvCacheWriteKernel<<<(uint32_t)((elements + SPARK_MINIMAX_KERNEL_THREADS - 1u) / SPARK_MINIMAX_KERNEL_THREADS),SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)key_bf16,(const __nv_bfloat16 *)value_bf16,(__nv_bfloat16 *)kv_cache_bf16,
		physical_block_indices,lane_block_counts,row_lane_indices,slot_mapping,row_count,
		local_kv_head_dimension * SPARK_MINIMAX_KERNEL_HEAD_DIM,lane_stride,block_span,layer_block_stride,layer_index,0u);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchAttentionDecode(cudaStream_t stream,const void *query_roped_f32,void *kv_cache_bf16,const SparkMinimaxKvBlockTableView *table,const uint32_t *row_lane_indices,const uint32_t *slot_mapping,const uint32_t *context_lengths,void *attended_bf16,uint32_t row_count,uint32_t local_query_head_count,uint32_t local_kv_head_count,uint32_t local_kv_head_dimension,uint64_t block_span,uint64_t layer_block_stride,float epsilon,uint32_t tp_rank,uint32_t layer_index)
{
	dim3 grid(local_query_head_count,row_count);
	if ( table == 0 || table->physical_block_indices == 0 || table->lane_physical_block_counts == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxAttentionDecodeKernel<<<grid,SPARK_MINIMAX_KERNEL_HEAD_DIM,0,stream>>>(
		(const float *)query_roped_f32,(const __nv_bfloat16 *)kv_cache_bf16,
		table->physical_block_indices,table->lane_physical_block_counts,table->lane_stride,
		row_lane_indices,slot_mapping,context_lengths,(__nv_bfloat16 *)attended_bf16,
		row_count,local_query_head_count,local_kv_head_count,
		local_kv_head_dimension,block_span,layer_block_stride,epsilon,tp_rank,layer_index);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchAttentionPrefill(cudaStream_t stream,const void *query_roped_f32,void *kv_cache_bf16,const SparkMinimaxKvBlockTableView *table,const uint32_t *row_lane_indices,const uint64_t *row_positions,const void *staged_key_bf16,const void *staged_value_bf16,__nv_bfloat16 *attended_bf16,uint32_t row_count,uint32_t local_query_head_count,uint32_t local_kv_head_count,uint32_t local_kv_head_dimension,uint32_t lane_stride,uint64_t block_span,uint64_t layer_block_stride,float epsilon,uint32_t tp_rank,uint32_t layer_index,uint64_t base_position)
{
	dim3 grid(local_query_head_count,row_count);
	if ( table == 0 || table->physical_block_indices == 0 || table->lane_physical_block_counts == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxAttentionPrefillKernel<<<grid,SPARK_MINIMAX_KERNEL_HEAD_DIM,0,stream>>>(
		(const float *)query_roped_f32,(const __nv_bfloat16 *)kv_cache_bf16,
		table->physical_block_indices,table->lane_physical_block_counts,row_lane_indices,row_positions,
		(const __nv_bfloat16 *)staged_key_bf16,(const __nv_bfloat16 *)staged_value_bf16,attended_bf16,
		row_count,local_query_head_count,local_kv_head_count,local_kv_head_dimension,
		block_span,layer_block_stride,epsilon,tp_rank,layer_index,base_position);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchProjection(cudaStream_t stream,const SparkMinimaxLinearView *view,const void *input_bf16,void *output_bf16,uint32_t row_count)
{
	return(SparkMinimaxDispatchLinear(stream,view,input_bf16,output_bf16,row_count));
}

extern "C" cudaError_t SparkMinimaxLaunchSwiGlu(cudaStream_t stream,const void *gate_bf16,const void *up_bf16,void *activated_bf16,uint32_t row_count,uint32_t dimension)
{
	uint64_t elements = (uint64_t)row_count * dimension;
	SparkMinimaxSwiGluKernel<<<(uint32_t)((elements + SPARK_MINIMAX_KERNEL_THREADS - 1u) / SPARK_MINIMAX_KERNEL_THREADS),SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)gate_bf16,(const __nv_bfloat16 *)up_bf16,(__nv_bfloat16 *)activated_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchVocabArgmax(cudaStream_t stream,const void *lm_head_bf16,const void *input_bf16,uint64_t *argmax_reduce_u64,uint32_t row_count,uint32_t local_vocab_rows,uint32_t input_dimension,uint32_t local_vocab_base)
{
	dim3 grid((local_vocab_rows + SPARK_MINIMAX_KERNEL_THREADS - 1u) / SPARK_MINIMAX_KERNEL_THREADS,row_count);
	if ( cudaMemsetAsync(argmax_reduce_u64,0,(size_t)row_count * sizeof(uint64_t),stream) != cudaSuccess )
		return(cudaGetLastError());
	SparkMinimaxVocabArgmaxKernel<<<grid,SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)lm_head_bf16,(const __nv_bfloat16 *)input_bf16,argmax_reduce_u64,
		row_count,local_vocab_rows,input_dimension,local_vocab_base);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchArgmaxResolve(cudaStream_t stream,const uint64_t *argmax_reduce_u64,uint32_t *token_ids,uint32_t row_count)
{
	SparkMinimaxArgmaxResolveKernel<<<(row_count + SPARK_MINIMAX_KERNEL_THREADS - 1u) / SPARK_MINIMAX_KERNEL_THREADS,SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>(argmax_reduce_u64,token_ids,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchTpCombineBf16(cudaStream_t stream,void *destination_device,const void *source_device,uint32_t element_count)
{
	SparkMinimaxTpCombineBf16Kernel<<<(element_count + SPARK_MINIMAX_KERNEL_THREADS - 1u) / SPARK_MINIMAX_KERNEL_THREADS,SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>((__nv_bfloat16 *)destination_device,(const __nv_bfloat16 *)source_device,element_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxLaunchTpCombineU64Max(cudaStream_t stream,void *destination_device,const void *source_device,uint32_t element_count)
{
	SparkMinimaxTpCombineU64MaxKernel<<<(element_count + SPARK_MINIMAX_KERNEL_THREADS - 1u) / SPARK_MINIMAX_KERNEL_THREADS,SPARK_MINIMAX_KERNEL_THREADS,0,stream>>>((uint64_t *)destination_device,(const uint64_t *)source_device,element_count);
	return(cudaGetLastError());
}
