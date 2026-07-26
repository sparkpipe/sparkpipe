#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "sparkpipe/spark_k3_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_lm_kernels.cuh"

#define SPARK_K3_WARP_LANES 32u
#define SPARK_K3_CTA_THREADS 256u
#define SPARK_K3_CTA_WARPS (SPARK_K3_CTA_THREADS / SPARK_K3_WARP_LANES)
#define SPARK_K3_KDA_TILE 16u
#define SPARK_K3_KDA_SLAB_COLUMNS 64u
#define SPARK_K3_KDA_TILE_BLOCKS (SPARK_K3_MODEL_KDA_CHUNK_TOKENS / SPARK_K3_KDA_TILE)
#define SPARK_K3_KDA_LOWER_PAIRS ((SPARK_K3_KDA_TILE_BLOCKS * (SPARK_K3_KDA_TILE_BLOCKS + 1u)) / 2u)
#define SPARK_K3_KDA_DK SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION
#define SPARK_K3_KDA_DV SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION

namespace {

namespace wmma = nvcuda::wmma;

typedef wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> SparkK3FragARow;
typedef wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> SparkK3FragACol;
typedef wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> SparkK3FragBRow;
typedef wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> SparkK3FragBCol;
typedef wmma::fragment<wmma::accumulator,16,16,16,float> SparkK3FragAcc;

// Seeds the AttnRes candidate array from the token embedding: representation 0
// is the embedding block and representation 1 is the running partial sum, both
// the embedding vector itself. This is the entry state the published forward()
// assumes ("blocks already include token embedding", partial = hidden_states).
__global__ void SparkK3EmbeddingGatherKernel(const uint32_t *token_ids, const void *embedding_bf16, void *representations_bf16, uint32_t row_count, uint64_t representation_stride)
{
	uint32_t row = blockIdx.x;
	uint64_t source_base,destination,partial_base;
	uint32_t element;
	uint4 row_vector;
	if ( row >= row_count )
		return;
	source_base = (uint64_t)token_ids[row] * (uint64_t)SPARK_K3_MODEL_HIDDEN_DIMENSION;
	destination = (uint64_t)row * (uint64_t)SPARK_K3_MODEL_HIDDEN_DIMENSION;
	partial_base = (SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(0) * representation_stride) + destination;
	for (element = threadIdx.x; element < (SPARK_K3_MODEL_HIDDEN_DIMENSION >> 3u); element += blockDim.x)
	{
		row_vector = __ldg(((const uint4 *)embedding_bf16) + (source_base >> 3u) + element);
		((uint4 *)representations_bf16)[(destination >> 3u) + element] = row_vector;
		((uint4 *)representations_bf16)[(partial_base >> 3u) + element] = row_vector;
	}
}

/*
 * Block AttnRes mixing site. representations holds candidate vectors
 * representation-major: rep n at representations + n*representation_stride +
 * row*hidden. Candidate rep_count-1 is the running partial sum (mixed in
 * place with the completed blocks per the published pseudocode). One pass per
 * candidate accumulates both sum-of-squares and the raw pseudo-query dot so
 * logit_n = dot(w, gamma * v_n / rms_n) costs a single read of v_n; softmax
 * runs over at most SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS scalars in
 * shared memory, and a second pass forms the mixture.
 */
// Thread-0 softmax over a small shared scalar table; every thread leaves with
// the normalized weights visible.
__device__ void SparkK3SharedSoftmax(const float *logits, float *weights, uint32_t count)
{
	uint32_t candidate;
	float maximum,total;
	if ( threadIdx.x == 0u )
	{
		maximum = logits[0];
		for (candidate = 1; candidate < count; candidate++)
			if ( logits[candidate] > maximum )
				maximum = logits[candidate];
		total = 0.0f;
		for (candidate = 0; candidate < count; candidate++)
		{
			weights[candidate] = __expf(logits[candidate] - maximum);
			total += weights[candidate];
		}
		for (candidate = 0; candidate < count; candidate++)
			weights[candidate] /= total;
	}
	__syncthreads();
}

__global__ void SparkK3AttnResMixKernel(const void *representations_bf16, uint64_t representation_stride, const void *pseudo_query_bf16, const void *key_norm_weight_bf16, void *mixed_bf16, uint32_t representation_count, uint32_t row_count, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_K3_CTA_WARPS];
	__shared__ float logits[SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS];
	__shared__ float weights[SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS];
	uint32_t row = blockIdx.x;
	uint64_t row_offset,candidate_offset;
	uint32_t candidate,element;
	float sum_squares,dot,inverse_rms,mixed;
	float2 value_pair,query_pair,norm_pair;
	if ( row >= row_count )
		return;
	row_offset = (uint64_t)row * (uint64_t)SPARK_K3_MODEL_HIDDEN_DIMENSION;
	for (candidate = 0; candidate < representation_count; candidate++)
	{
		candidate_offset = ((uint64_t)candidate * representation_stride) + row_offset;
		sum_squares = 0.0f;
		dot = 0.0f;
		for (element = threadIdx.x; element < (SPARK_K3_MODEL_HIDDEN_DIMENSION >> 1u); element += blockDim.x)
		{
			value_pair = SparkLmLoadBf16Pair(representations_bf16,(candidate_offset >> 1u) + element);
			query_pair = SparkLmLoadBf16Pair(pseudo_query_bf16,element);
			norm_pair = SparkLmLoadBf16Pair(key_norm_weight_bf16,element);
			sum_squares = fmaf(value_pair.x,value_pair.x,fmaf(value_pair.y,value_pair.y,sum_squares));
			dot = fmaf(query_pair.x * norm_pair.x,value_pair.x,fmaf(query_pair.y * norm_pair.y,value_pair.y,dot));
		}
		sum_squares = SparkLmBlockReduceSum(sum_squares,reduce_scratch);
		dot = SparkLmBlockReduceSum(dot,reduce_scratch);
		inverse_rms = rsqrtf((sum_squares / (float)SPARK_K3_MODEL_HIDDEN_DIMENSION) + epsilon);
		if ( threadIdx.x == 0u )
			logits[candidate] = (dot * inverse_rms);
		__syncthreads();
	}
	SparkK3SharedSoftmax(logits,weights,representation_count);
	for (element = threadIdx.x; element < (SPARK_K3_MODEL_HIDDEN_DIMENSION >> 1u); element += blockDim.x)
	{
		mixed = 0.0f;
		sum_squares = 0.0f;
		for (candidate = 0; candidate < representation_count; candidate++)
		{
			value_pair = SparkLmLoadBf16Pair(representations_bf16,((((uint64_t)candidate * representation_stride) + row_offset) >> 1u) + element);
			mixed = fmaf(weights[candidate],value_pair.x,mixed);
			sum_squares = fmaf(weights[candidate],value_pair.y,sum_squares);
		}
		SparkLmStoreBf16Pair(mixed_bf16,(row_offset >> 1u) + element,mixed,sum_squares);
	}
}

/*
 * AttnRes partial-sum bookkeeping after a sub-layer. The running partial is
 * candidate completed_block_count of the representation array, which is where
 * the mixing site reads it from. Opening a block therefore needs no copy: the
 * partial already sits in its completed-block slot, and the sub-layer output
 * starts the next partial one candidate along. Otherwise the output simply
 * accumulates into the current partial.
 */
__global__ void SparkK3AttnResAccumulateKernel(void *representations_bf16, uint64_t representation_stride, const void *sublayer_output_bf16, uint32_t open_new_block, uint32_t completed_block_count, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	uint64_t row_offset,partial_offset;
	uint32_t element;
	float output_value;
	if ( row >= row_count )
		return;
	row_offset = (uint64_t)row * (uint64_t)SPARK_K3_MODEL_HIDDEN_DIMENSION;
	partial_offset = ((uint64_t)(completed_block_count + open_new_block) * representation_stride) + row_offset;
	for (element = threadIdx.x; element < SPARK_K3_MODEL_HIDDEN_DIMENSION; element += blockDim.x)
	{
		output_value = SparkLmBf16ToFloat(sublayer_output_bf16,row_offset + element);
		if ( open_new_block == 0u )
			output_value += SparkLmBf16ToFloat(representations_bf16,partial_offset + element);
		SparkLmFloatToBf16(representations_bf16,partial_offset + element,output_value);
	}
}

/*
 * KDA operand conditioning after the projections: per-head L2 normalization
 * of q and k (the non-expansive contract of the delta rule), per-channel
 * log decay = -softplus(pre) (GUESS pending the K3 report: guarantees decay
 * in (0,1]; swapping the nonlinearity is this one function), per-head
 * beta = sigmoid(pre), and the low-rank output gate = sigmoid(pre).
 */
__global__ void SparkK3KdaNormalizeQkKernel(void *query_bf16, void *key_bf16, uint32_t row_count)
{
	__shared__ float reduce_scratch[SPARK_K3_CTA_WARPS];
	uint32_t row = blockIdx.x,head = blockIdx.y;
	uint64_t base;
	uint32_t element;
	float query_squares,key_squares,query_scale,key_scale;
	float2 query_pair,key_pair;
	if ( row >= row_count )
		return;
	base = (((uint64_t)row * SPARK_K3_MODEL_KDA_HEAD_COUNT) + head) * SPARK_K3_KDA_DK;
	query_squares = 0.0f;
	key_squares = 0.0f;
	for (element = threadIdx.x; element < (SPARK_K3_KDA_DK >> 1u); element += blockDim.x)
	{
		query_pair = SparkLmLoadBf16Pair(query_bf16,(base >> 1u) + element);
		key_pair = SparkLmLoadBf16Pair(key_bf16,(base >> 1u) + element);
		query_squares = fmaf(query_pair.x,query_pair.x,fmaf(query_pair.y,query_pair.y,query_squares));
		key_squares = fmaf(key_pair.x,key_pair.x,fmaf(key_pair.y,key_pair.y,key_squares));
	}
	query_squares = SparkLmBlockReduceSum(query_squares,reduce_scratch);
	key_squares = SparkLmBlockReduceSum(key_squares,reduce_scratch);
	query_scale = rsqrtf(query_squares + 1e-12f);
	key_scale = rsqrtf(key_squares + 1e-12f);
	for (element = threadIdx.x; element < (SPARK_K3_KDA_DK >> 1u); element += blockDim.x)
	{
		query_pair = SparkLmLoadBf16Pair(query_bf16,(base >> 1u) + element);
		key_pair = SparkLmLoadBf16Pair(key_bf16,(base >> 1u) + element);
		SparkLmStoreBf16Pair(query_bf16,(base >> 1u) + element,query_pair.x * query_scale,query_pair.y * query_scale);
		SparkLmStoreBf16Pair(key_bf16,(base >> 1u) + element,key_pair.x * key_scale,key_pair.y * key_scale);
	}
}

// The decay and output-gate activations are elementwise over one bf16 row of
// width KDA_QK_DIMENSION and KDA_VALUE_DIMENSION respectively; the fused loop
// below is only correct while those widths are equal.
static_assert(SPARK_K3_MODEL_KDA_QK_DIMENSION == SPARK_K3_MODEL_KDA_VALUE_DIMENSION,"k3 kda gate/decay fusion requires dk == dv");

__global__ void SparkK3KdaGateBetaKernel(void *log_decay_bf16, void *output_gate_bf16, void *beta_bf16, uint32_t row_count)
{
	uint64_t wide_elements = (uint64_t)row_count * SPARK_K3_MODEL_KDA_QK_DIMENSION;
	uint64_t beta_elements = (uint64_t)row_count * SPARK_K3_MODEL_KDA_HEAD_COUNT;
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	uint64_t element;
	float2 decay_pair,gate_pair;
	for (element = index; element < (wide_elements >> 1u); element += stride)
	{
		decay_pair = SparkLmLoadBf16Pair(log_decay_bf16,element);
		gate_pair = SparkLmLoadBf16Pair(output_gate_bf16,element);
		SparkLmStoreBf16Pair(log_decay_bf16,element,-SparkLmSoftplus(decay_pair.x),-SparkLmSoftplus(decay_pair.y));
		SparkLmStoreBf16Pair(output_gate_bf16,element,SparkLmSigmoid(gate_pair.x),SparkLmSigmoid(gate_pair.y));
	}
	for (element = index; element < (beta_elements >> 1u); element += stride)
	{
		decay_pair = SparkLmLoadBf16Pair(beta_bf16,element);
		SparkLmStoreBf16Pair(beta_bf16,element,SparkLmSigmoid(decay_pair.x),SparkLmSigmoid(decay_pair.y));
	}
}

typedef struct SparkK3KdaSmemLayout
{
	uint32_t lg_last;
	uint32_t betav;
	uint32_t pseudo;
	uint32_t rowblock;
	uint32_t gram16;
	uint32_t score16;
	uint32_t kb;
	uint32_t kh;
	uint32_t qb;
	uint32_t u16;
	uint32_t total;
} SparkK3KdaSmemLayout;

__host__ __device__ static inline uint32_t SparkK3KdaSmemAlign(uint32_t offset)
{
	return((offset + 127u) & ~127u);
}

// One byte plan shared by the host launch (attribute opt-in) and the kernel
// (pointer carving) so the two can never drift. 94976 bytes at dk=128, under
// the sm_121 opt-in cap of 101376.
__host__ __device__ static inline SparkK3KdaSmemLayout SparkK3KdaSmemPlan(void)
{
	SparkK3KdaSmemLayout plan;
	uint32_t offset = 0;
	plan.lg_last = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_KDA_DK * 4u));
	plan.betav = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * 4u));
	plan.pseudo = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_KDA_SLAB_COLUMNS * 4u));
	plan.rowblock = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_KDA_TILE * SPARK_K3_KDA_SLAB_COLUMNS * 4u));
	plan.gram16 = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_MODEL_KDA_CHUNK_TOKENS * 2u));
	plan.score16 = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_MODEL_KDA_CHUNK_TOKENS * 2u));
	plan.kb = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_KDA_DK * 2u));
	plan.kh = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_KDA_DK * 2u));
	plan.qb = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_KDA_DK * 2u));
	plan.u16 = offset;
	offset = SparkK3KdaSmemAlign(offset + (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_KDA_SLAB_COLUMNS * 2u));
	plan.total = offset;
	return(plan);
}

__device__ __forceinline__ float *SparkK3KdaStateHead(const SparkK3KdaStatePool pool, uint32_t lane, uint32_t layer_ordinal, uint32_t head)
{
	return(pool.state_f32 + ((uint64_t)lane * pool.lane_stride_elements) + ((uint64_t)layer_ordinal * pool.layer_stride_elements) + ((uint64_t)head * SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_HEAD));
}

/*
 * Chunkwise Kimi Delta Attention on tensor cores. Datapath identical to the
 * CPU-validated wmma plan (kda_sm121_stage rev 2): bf16 operands kb/kh/qb
 * with the running per-channel log decay clamped at
 * SPARK_K3_MODEL_KDA_MIN_LOG_DECAY, bf16 gram after beta plus strict lower
 * mask, fp32 blocked forward substitution, bf16 pseudo-values feeding the
 * output and carry products, fp32 recurrent state resident in global memory.
 * Measured CPU-emulation error against the fp32 recurrence: out <= 1.13e-2,
 * state <= 2.0e-3, flat in chunk count.
 */
__device__ void SparkK3KdaWmmaMaterialize(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *log_decay, const __nv_bfloat16 *beta, uint32_t token_count, uint32_t token_stride, __nv_bfloat16 *kb, __nv_bfloat16 *kh, __nv_bfloat16 *qb, float *lg_last, float *betav)
{
	uint32_t channel,token,index,source;
	float running,key_value,gate;
	for (channel = threadIdx.x; channel < SPARK_K3_KDA_DK; channel += blockDim.x)
	{
		running = 0.0f;
		for (token = 0; token < token_count; token++)
		{
			source = (token * token_stride) + channel;
			running += __bfloat162float(log_decay[source]);
			if ( running < SPARK_K3_MODEL_KDA_MIN_LOG_DECAY )
				running = SPARK_K3_MODEL_KDA_MIN_LOG_DECAY;
			index = (token * SPARK_K3_KDA_DK) + channel;
			key_value = __bfloat162float(key[source]);
			gate = __expf(running);
			kb[index] = __float2bfloat16(key_value * gate);
			kh[index] = __float2bfloat16(key_value * __expf(-running));
			qb[index] = __float2bfloat16(__bfloat162float(query[source]) * gate);
		}
		lg_last[channel] = running;
		for (token = token_count; token < SPARK_K3_MODEL_KDA_CHUNK_TOKENS; token++)
		{
			index = (token * SPARK_K3_KDA_DK) + channel;
			kb[index] = __float2bfloat16(0.0f);
			kh[index] = __float2bfloat16(0.0f);
			qb[index] = __float2bfloat16(0.0f);
		}
	}
	for (token = threadIdx.x; token < SPARK_K3_MODEL_KDA_CHUNK_TOKENS; token += blockDim.x)
		betav[token] = token < token_count ? __bfloat162float(beta[token * SPARK_K3_MODEL_KDA_HEAD_COUNT]) : 0.0f;
	__syncthreads();
}

// product = mask(rows kh^T): betav != 0 selects the beta-scaled strictly
// lower gram; betav == 0 selects the inclusive score. One warp per lower
// triangular tile pair; stage reuses the pseudo buffer, free until seeding.
__device__ void SparkK3KdaWmmaMaskedProduct(const __nv_bfloat16 *rows, const __nv_bfloat16 *kh, const float *betav, float *stage, __nv_bfloat16 *product)
{
	const uint8_t pair_row[SPARK_K3_KDA_LOWER_PAIRS] = {0,1,1,2,2,2,3,3,3,3};
	const uint8_t pair_col[SPARK_K3_KDA_LOWER_PAIRS] = {0,0,1,0,1,2,0,1,2,3};
	SparkK3FragARow row_frag;
	SparkK3FragBCol column_frag;
	SparkK3FragAcc acc;
	uint32_t warp = threadIdx.x / SPARK_K3_WARP_LANES,lane = threadIdx.x % SPARK_K3_WARP_LANES;
	uint32_t index,pair,ktile,row,column;
	float *warp_stage = stage + (warp * SPARK_K3_KDA_TILE * SPARK_K3_KDA_TILE);
	float scaled;
	for (index = threadIdx.x; index < (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_MODEL_KDA_CHUNK_TOKENS); index += blockDim.x)
		product[index] = __float2bfloat16(0.0f);
	__syncthreads();
	for (pair = warp; pair < SPARK_K3_KDA_LOWER_PAIRS; pair += SPARK_K3_CTA_WARPS)
	{
		wmma::fill_fragment(acc,0.0f);
		for (ktile = 0; ktile < (SPARK_K3_KDA_DK / SPARK_K3_KDA_TILE); ktile++)
		{
			wmma::load_matrix_sync(row_frag,rows + (pair_row[pair] * SPARK_K3_KDA_TILE * SPARK_K3_KDA_DK) + (ktile * SPARK_K3_KDA_TILE),SPARK_K3_KDA_DK);
			wmma::load_matrix_sync(column_frag,kh + (pair_col[pair] * SPARK_K3_KDA_TILE * SPARK_K3_KDA_DK) + (ktile * SPARK_K3_KDA_TILE),SPARK_K3_KDA_DK);
			wmma::mma_sync(acc,row_frag,column_frag,acc);
		}
		wmma::store_matrix_sync(warp_stage,acc,SPARK_K3_KDA_TILE,wmma::mem_row_major);
		__syncwarp();
		for (index = lane; index < (SPARK_K3_KDA_TILE * SPARK_K3_KDA_TILE); index += SPARK_K3_WARP_LANES)
		{
			row = (pair_row[pair] * SPARK_K3_KDA_TILE) + (index / SPARK_K3_KDA_TILE);
			column = (pair_col[pair] * SPARK_K3_KDA_TILE) + (index % SPARK_K3_KDA_TILE);
			if ( betav != 0 ? column < row : column <= row )
			{
				scaled = betav != 0 ? (betav[row] * warp_stage[index]) : warp_stage[index];
				product[(row * SPARK_K3_MODEL_KDA_CHUNK_TOKENS) + column] = __float2bfloat16(scaled);
			}
		}
		__syncwarp();
	}
	__syncthreads();
}

__device__ void SparkK3KdaWmmaSeed(const __nv_bfloat16 *kb, const __nv_bfloat16 *value, uint32_t token_stride, const float *betav, const float *state_head, uint32_t slab, uint32_t carry_state_in, float *pseudo)
{
	uint32_t element,row,column,channel;
	float projected;
	for (element = threadIdx.x; element < (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_KDA_SLAB_COLUMNS); element += blockDim.x)
	{
		row = element / SPARK_K3_KDA_SLAB_COLUMNS;
		column = slab + (element % SPARK_K3_KDA_SLAB_COLUMNS);
		projected = 0.0f;
		if ( carry_state_in != 0u )
		{
			for (channel = 0; channel < SPARK_K3_KDA_DK; channel++)
				projected += (__bfloat162float(kb[(row * SPARK_K3_KDA_DK) + channel]) * state_head[(channel * SPARK_K3_KDA_DV) + column]);
		}
		pseudo[element] = betav[row] * (__bfloat162float(value[(row * token_stride) + column]) - projected);
	}
	__syncthreads();
}

// Blocked fp32 forward substitution for (I + tril(BA,-1)) U = seed on the
// slab-resident U; only the sixteen rows of each diagonal block are serial.
__device__ void SparkK3KdaWmmaSolve(const __nv_bfloat16 *gram16, float *pseudo)
{
	uint32_t block_base,element,row,column,prior,inner;
	float accumulator;
	for (block_base = 0; block_base < SPARK_K3_MODEL_KDA_CHUNK_TOKENS; block_base += SPARK_K3_KDA_TILE)
	{
		if ( block_base != 0u )
		{
			for (element = threadIdx.x; element < (SPARK_K3_KDA_TILE * SPARK_K3_KDA_SLAB_COLUMNS); element += blockDim.x)
			{
				row = block_base + (element / SPARK_K3_KDA_SLAB_COLUMNS);
				column = element % SPARK_K3_KDA_SLAB_COLUMNS;
				accumulator = pseudo[(row * SPARK_K3_KDA_SLAB_COLUMNS) + column];
				for (prior = 0; prior < block_base; prior++)
					accumulator -= (__bfloat162float(gram16[(row * SPARK_K3_MODEL_KDA_CHUNK_TOKENS) + prior]) * pseudo[(prior * SPARK_K3_KDA_SLAB_COLUMNS) + column]);
				pseudo[(row * SPARK_K3_KDA_SLAB_COLUMNS) + column] = accumulator;
			}
			__syncthreads();
		}
		for (inner = 1; inner < SPARK_K3_KDA_TILE; inner++)
		{
			row = block_base + inner;
			for (column = threadIdx.x; column < SPARK_K3_KDA_SLAB_COLUMNS; column += blockDim.x)
			{
				accumulator = pseudo[(row * SPARK_K3_KDA_SLAB_COLUMNS) + column];
				for (prior = 0; prior < inner; prior++)
					accumulator -= (__bfloat162float(gram16[(row * SPARK_K3_MODEL_KDA_CHUNK_TOKENS) + block_base + prior]) * pseudo[((block_base + prior) * SPARK_K3_KDA_SLAB_COLUMNS) + column]);
				pseudo[(row * SPARK_K3_KDA_SLAB_COLUMNS) + column] = accumulator;
			}
			__syncthreads();
		}
	}
}

__device__ void SparkK3KdaWmmaOutput(const __nv_bfloat16 *qb, const __nv_bfloat16 *score16, const __nv_bfloat16 *u16, const float *state_head, float *rowblock, uint32_t slab, uint32_t token_count, uint32_t token_stride, uint32_t carry_state_in, __nv_bfloat16 *output)
{
	SparkK3FragARow score_frag;
	SparkK3FragBRow value_frag;
	SparkK3FragAcc acc;
	uint32_t warp = threadIdx.x / SPARK_K3_WARP_LANES;
	uint32_t row_block,element,row,column,channel,prior_block,ntile;
	float projected;
	for (row_block = 0; row_block < SPARK_K3_KDA_TILE_BLOCKS; row_block++)
	{
		for (element = threadIdx.x; element < (SPARK_K3_KDA_TILE * SPARK_K3_KDA_SLAB_COLUMNS); element += blockDim.x)
		{
			row = (row_block * SPARK_K3_KDA_TILE) + (element / SPARK_K3_KDA_SLAB_COLUMNS);
			column = slab + (element % SPARK_K3_KDA_SLAB_COLUMNS);
			projected = 0.0f;
			if ( carry_state_in != 0u )
			{
				for (channel = 0; channel < SPARK_K3_KDA_DK; channel++)
					projected += (__bfloat162float(qb[(row * SPARK_K3_KDA_DK) + channel]) * state_head[(channel * SPARK_K3_KDA_DV) + column]);
			}
			rowblock[element] = projected;
		}
		__syncthreads();
		for (ntile = warp; ntile < (SPARK_K3_KDA_SLAB_COLUMNS / SPARK_K3_KDA_TILE); ntile += SPARK_K3_CTA_WARPS)
		{
			wmma::load_matrix_sync(acc,rowblock + (ntile * SPARK_K3_KDA_TILE),SPARK_K3_KDA_SLAB_COLUMNS,wmma::mem_row_major);
			for (prior_block = 0; prior_block <= row_block; prior_block++)
			{
				wmma::load_matrix_sync(score_frag,score16 + (row_block * SPARK_K3_KDA_TILE * SPARK_K3_MODEL_KDA_CHUNK_TOKENS) + (prior_block * SPARK_K3_KDA_TILE),SPARK_K3_MODEL_KDA_CHUNK_TOKENS);
				wmma::load_matrix_sync(value_frag,u16 + (prior_block * SPARK_K3_KDA_TILE * SPARK_K3_KDA_SLAB_COLUMNS) + (ntile * SPARK_K3_KDA_TILE),SPARK_K3_KDA_SLAB_COLUMNS);
				wmma::mma_sync(acc,score_frag,value_frag,acc);
			}
			wmma::store_matrix_sync(rowblock + (ntile * SPARK_K3_KDA_TILE),acc,SPARK_K3_KDA_SLAB_COLUMNS,wmma::mem_row_major);
		}
		__syncthreads();
		for (element = threadIdx.x; element < (SPARK_K3_KDA_TILE * SPARK_K3_KDA_SLAB_COLUMNS); element += blockDim.x)
		{
			row = (row_block * SPARK_K3_KDA_TILE) + (element / SPARK_K3_KDA_SLAB_COLUMNS);
			if ( row < token_count )
				output[(row * token_stride) + slab + (element % SPARK_K3_KDA_SLAB_COLUMNS)] = __float2bfloat16(rowblock[element]);
		}
		__syncthreads();
	}
}

__device__ void SparkK3KdaWmmaCarry(const __nv_bfloat16 *kh, const __nv_bfloat16 *u16, const float *lg_last, float *rowblock, uint32_t slab, uint32_t carry_state_in, float *state_head)
{
	SparkK3FragACol key_frag;
	SparkK3FragBRow value_frag;
	SparkK3FragAcc acc;
	uint32_t warp = threadIdx.x / SPARK_K3_WARP_LANES;
	uint32_t channel_block,element,channel,ktile,ntile;
	for (channel_block = 0; channel_block < (SPARK_K3_KDA_DK / SPARK_K3_KDA_TILE); channel_block++)
	{
		for (element = threadIdx.x; element < (SPARK_K3_KDA_TILE * SPARK_K3_KDA_SLAB_COLUMNS); element += blockDim.x)
		{
			channel = (channel_block * SPARK_K3_KDA_TILE) + (element / SPARK_K3_KDA_SLAB_COLUMNS);
			rowblock[element] = carry_state_in != 0u ? state_head[(channel * SPARK_K3_KDA_DV) + slab + (element % SPARK_K3_KDA_SLAB_COLUMNS)] : 0.0f;
		}
		__syncthreads();
		for (ntile = warp; ntile < (SPARK_K3_KDA_SLAB_COLUMNS / SPARK_K3_KDA_TILE); ntile += SPARK_K3_CTA_WARPS)
		{
			wmma::load_matrix_sync(acc,rowblock + (ntile * SPARK_K3_KDA_TILE),SPARK_K3_KDA_SLAB_COLUMNS,wmma::mem_row_major);
			for (ktile = 0; ktile < SPARK_K3_KDA_TILE_BLOCKS; ktile++)
			{
				wmma::load_matrix_sync(key_frag,kh + (ktile * SPARK_K3_KDA_TILE * SPARK_K3_KDA_DK) + (channel_block * SPARK_K3_KDA_TILE),SPARK_K3_KDA_DK);
				wmma::load_matrix_sync(value_frag,u16 + (ktile * SPARK_K3_KDA_TILE * SPARK_K3_KDA_SLAB_COLUMNS) + (ntile * SPARK_K3_KDA_TILE),SPARK_K3_KDA_SLAB_COLUMNS);
				wmma::mma_sync(acc,key_frag,value_frag,acc);
			}
			wmma::store_matrix_sync(rowblock + (ntile * SPARK_K3_KDA_TILE),acc,SPARK_K3_KDA_SLAB_COLUMNS,wmma::mem_row_major);
		}
		__syncthreads();
		for (element = threadIdx.x; element < (SPARK_K3_KDA_TILE * SPARK_K3_KDA_SLAB_COLUMNS); element += blockDim.x)
		{
			channel = (channel_block * SPARK_K3_KDA_TILE) + (element / SPARK_K3_KDA_SLAB_COLUMNS);
			state_head[(channel * SPARK_K3_KDA_DV) + slab + (element % SPARK_K3_KDA_SLAB_COLUMNS)] = rowblock[element] * __expf(lg_last[channel]);
		}
		__syncthreads();
	}
}

__global__ __launch_bounds__(SPARK_K3_CTA_THREADS) void SparkK3KdaChunkKernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value, const __nv_bfloat16 *log_decay, const __nv_bfloat16 *beta, SparkK3KdaStatePool pool, const uint32_t *state_lane_indices, uint32_t layer_ordinal, __nv_bfloat16 *output, const int32_t *sequence_token_counts, uint32_t chunk_tokens, uint32_t carry_state_in, uint32_t write_state_out)
{
	static_assert(SPARK_K3_MODEL_KDA_CHUNK_TOKENS == 64u,"pair tables assume four tile blocks");
	static_assert((SPARK_K3_KDA_DK % SPARK_K3_KDA_TILE) == 0u,"key dimension must tile");
	static_assert((SPARK_K3_KDA_DV % SPARK_K3_KDA_SLAB_COLUMNS) == 0u,"value dimension must slab");
	extern __shared__ __align__(128) uint8_t shared_storage[];
	SparkK3KdaSmemLayout plan = SparkK3KdaSmemPlan();
	float *lg_last = (float *)(shared_storage + plan.lg_last);
	float *betav = (float *)(shared_storage + plan.betav);
	float *pseudo = (float *)(shared_storage + plan.pseudo);
	float *rowblock = (float *)(shared_storage + plan.rowblock);
	__nv_bfloat16 *gram16 = (__nv_bfloat16 *)(shared_storage + plan.gram16);
	__nv_bfloat16 *score16 = (__nv_bfloat16 *)(shared_storage + plan.score16);
	__nv_bfloat16 *kb = (__nv_bfloat16 *)(shared_storage + plan.kb);
	__nv_bfloat16 *kh = (__nv_bfloat16 *)(shared_storage + plan.kh);
	__nv_bfloat16 *qb = (__nv_bfloat16 *)(shared_storage + plan.qb);
	__nv_bfloat16 *u16 = (__nv_bfloat16 *)(shared_storage + plan.u16);
	uint32_t sequence = blockIdx.y,head = blockIdx.x;
	uint32_t token_count = (uint32_t)sequence_token_counts[sequence];
	uint64_t token_base;
	uint32_t slab,element;
	float *state_head;
	if ( token_count == 0u || token_count > chunk_tokens )
		return;
	token_base = (uint64_t)sequence * chunk_tokens;
	state_head = SparkK3KdaStateHead(pool,state_lane_indices[sequence],layer_ordinal,head);
	SparkK3KdaWmmaMaterialize(query + (token_base * SPARK_K3_MODEL_KDA_QK_DIMENSION) + (head * SPARK_K3_KDA_DK),key + (token_base * SPARK_K3_MODEL_KDA_QK_DIMENSION) + (head * SPARK_K3_KDA_DK),log_decay + (token_base * SPARK_K3_MODEL_KDA_QK_DIMENSION) + (head * SPARK_K3_KDA_DK),beta + (token_base * SPARK_K3_MODEL_KDA_HEAD_COUNT) + head,token_count,SPARK_K3_MODEL_KDA_QK_DIMENSION,kb,kh,qb,lg_last,betav);
	SparkK3KdaWmmaMaskedProduct(kb,kh,betav,pseudo,gram16);
	SparkK3KdaWmmaMaskedProduct(qb,kh,0,pseudo,score16);
	for (slab = 0; slab < SPARK_K3_KDA_DV; slab += SPARK_K3_KDA_SLAB_COLUMNS)
	{
		SparkK3KdaWmmaSeed(kb,value + (token_base * SPARK_K3_MODEL_KDA_VALUE_DIMENSION) + (head * SPARK_K3_KDA_DV),SPARK_K3_MODEL_KDA_VALUE_DIMENSION,betav,state_head,slab,carry_state_in,pseudo);
		SparkK3KdaWmmaSolve(gram16,pseudo);
		for (element = threadIdx.x; element < (SPARK_K3_MODEL_KDA_CHUNK_TOKENS * SPARK_K3_KDA_SLAB_COLUMNS); element += blockDim.x)
			u16[element] = __float2bfloat16(pseudo[element]);
		__syncthreads();
		SparkK3KdaWmmaOutput(qb,score16,u16,state_head,rowblock,slab,token_count,SPARK_K3_MODEL_KDA_VALUE_DIMENSION,carry_state_in,output + (token_base * SPARK_K3_MODEL_KDA_VALUE_DIMENSION) + (head * SPARK_K3_KDA_DV));
		if ( write_state_out != 0u )
			SparkK3KdaWmmaCarry(kh,u16,lg_last,rowblock,slab,carry_state_in,state_head);
	}
}

// Single-token decode: the delta recurrence applied directly.
// S <- Diag(a) S; delta = b (v - S^T k); S += k delta^T; o = S^T q
/*
 * Delta-rule decode step with the head state RESIDENT IN SHARED for all
 * four phases: the old kernel walked the 64KB state through global four
 * times per token - decay read+write, delta read, rank-one update
 * read+write, output read: 384KB of traffic for a 64KB state. Now the
 * state loads once (decay fused into the load as float4), delta and
 * output run split-dot from shared with k and q staged, and the update
 * writes back fused - 128KB per token, one read and one write, 3x less
 * state traffic. Layout of the dynamic region, floats: state 16384,
 * gates 128, k 128, q 128, split-dot partials 256, delta 128.
 */
#define SPARK_K3_KDA_DECODE_SMEM_FLOATS (16384u + 128u + 128u + 128u + 256u + 128u)

static __device__ void SparkK3KdaDecodeSplitDot(const float *state, const float *vector, float *partial, float *combined, float combine_scale, const float *combine_bias, uint32_t negate)
{
	uint32_t column = threadIdx.x & (SPARK_K3_KDA_DV - 1u),half = threadIdx.x >> 7u,channel;
	float accumulator = 0.0f;
	for (channel = half * (SPARK_K3_KDA_DK / 2u); channel < (half + 1u) * (SPARK_K3_KDA_DK / 2u); channel++)
		accumulator = fmaf(vector[channel],state[(channel * SPARK_K3_KDA_DV) + column],accumulator);
	partial[threadIdx.x] = accumulator;
	__syncthreads();
	if ( threadIdx.x < SPARK_K3_KDA_DV )
	{
		accumulator = partial[threadIdx.x] + partial[threadIdx.x + SPARK_K3_KDA_DV];
		combined[threadIdx.x] = combine_scale * ((combine_bias != 0 ? combine_bias[threadIdx.x] : 0.0f) + (negate != 0u ? -accumulator : accumulator));
	}
	__syncthreads();
}

__global__ __launch_bounds__(SPARK_K3_CTA_THREADS) void SparkK3KdaDecodeStepKernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value, const __nv_bfloat16 *log_decay, const __nv_bfloat16 *beta, SparkK3KdaStatePool pool, const uint32_t *state_lane_indices, const uint32_t *state_cold_flags, uint32_t layer_ordinal, __nv_bfloat16 *output)
{
	extern __shared__ float kda_smem[];
	float *state = kda_smem,*gates = kda_smem + 16384u,*k_vector = gates + SPARK_K3_KDA_DK,*q_vector = k_vector + SPARK_K3_KDA_DK,*partial = q_vector + SPARK_K3_KDA_DK,*delta = partial + SPARK_K3_CTA_THREADS,*value_stage = delta;
	uint32_t row = blockIdx.y,head = blockIdx.x;
	uint64_t vector_base = (((uint64_t)row * SPARK_K3_MODEL_KDA_HEAD_COUNT) + head) * SPARK_K3_KDA_DK;
	uint64_t value_base = (((uint64_t)row * SPARK_K3_MODEL_KDA_HEAD_COUNT) + head) * SPARK_K3_KDA_DV;
	float *state_head = SparkK3KdaStateHead(pool,state_lane_indices[row],layer_ordinal,head);
	uint32_t cold = state_cold_flags[row];
	float beta_value = __bfloat162float(beta[((uint64_t)row * SPARK_K3_MODEL_KDA_HEAD_COUNT) + head]);
	uint32_t channel,group;
	float gate;
	float4 state_quad,delta_quad;
	for (channel = threadIdx.x; channel < SPARK_K3_KDA_DK; channel += blockDim.x)
	{
		gate = __bfloat162float(log_decay[vector_base + channel]);
		gates[channel] = __expf(gate < SPARK_K3_MODEL_KDA_MIN_LOG_DECAY ? SPARK_K3_MODEL_KDA_MIN_LOG_DECAY : gate);
		k_vector[channel] = __bfloat162float(key[vector_base + channel]);
		q_vector[channel] = __bfloat162float(query[vector_base + channel]);
	}
	if ( threadIdx.x < SPARK_K3_KDA_DV )
		value_stage[threadIdx.x] = __bfloat162float(value[value_base + threadIdx.x]);
	__syncthreads();
	for (group = threadIdx.x; group < (SPARK_K3_KDA_DK * SPARK_K3_KDA_DV) / 4u; group += blockDim.x)
	{
		state_quad = cold != 0u ? make_float4(0.0f,0.0f,0.0f,0.0f) : ((const float4 *)state_head)[group];
		gate = gates[group >> 5u];
		((float4 *)state)[group] = make_float4(state_quad.x * gate,state_quad.y * gate,state_quad.z * gate,state_quad.w * gate);
	}
	__syncthreads();
	SparkK3KdaDecodeSplitDot(state,k_vector,partial,delta,beta_value,value_stage,1u);
	for (group = threadIdx.x; group < (SPARK_K3_KDA_DK * SPARK_K3_KDA_DV) / 4u; group += blockDim.x)
	{
		delta_quad = ((const float4 *)delta)[group & 31u];
		gate = k_vector[group >> 5u];
		state_quad = ((const float4 *)state)[group];
		state_quad = make_float4(fmaf(gate,delta_quad.x,state_quad.x),fmaf(gate,delta_quad.y,state_quad.y),fmaf(gate,delta_quad.z,state_quad.z),fmaf(gate,delta_quad.w,state_quad.w));
		((float4 *)state)[group] = state_quad;
		((float4 *)state_head)[group] = state_quad;
	}
	__syncthreads();
	SparkK3KdaDecodeSplitDot(state,q_vector,partial,delta,1.0f,0,0u);
	if ( threadIdx.x < SPARK_K3_KDA_DV )
		output[value_base + threadIdx.x] = __float2bfloat16(delta[threadIdx.x]);
}

// Per-head RMS norm of the delta-rule output (gains shared across heads,
// GUESS pending the report) followed by the sigmoid low-rank output gate.
__global__ void SparkK3KdaFinishKernel(const void *core_output_bf16, const void *head_norm_weight_bf16, const void *gate_bf16, void *gated_bf16, uint32_t row_count, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_K3_CTA_WARPS];
	uint32_t row = blockIdx.x,head = blockIdx.y;
	uint64_t base;
	uint32_t element;
	float sum_squares,inverse_rms;
	float2 core_pair,norm_pair,gate_pair;
	if ( row >= row_count )
		return;
	base = (((uint64_t)row * SPARK_K3_MODEL_KDA_HEAD_COUNT) + head) * SPARK_K3_KDA_DV;
	sum_squares = 0.0f;
	for (element = threadIdx.x; element < (SPARK_K3_KDA_DV >> 1u); element += blockDim.x)
	{
		core_pair = SparkLmLoadBf16Pair(core_output_bf16,(base >> 1u) + element);
		sum_squares = fmaf(core_pair.x,core_pair.x,fmaf(core_pair.y,core_pair.y,sum_squares));
	}
	sum_squares = SparkLmBlockReduceSum(sum_squares,reduce_scratch);
	inverse_rms = rsqrtf((sum_squares / (float)SPARK_K3_KDA_DV) + epsilon);
	for (element = threadIdx.x; element < (SPARK_K3_KDA_DV >> 1u); element += blockDim.x)
	{
		core_pair = SparkLmLoadBf16Pair(core_output_bf16,(base >> 1u) + element);
		norm_pair = SparkLmLoadBf16Pair(head_norm_weight_bf16,element);
		gate_pair = SparkLmLoadBf16Pair(gate_bf16,(base >> 1u) + element);
		SparkLmStoreBf16Pair(gated_bf16,(base >> 1u) + element,(core_pair.x * inverse_rms) * norm_pair.x * gate_pair.x,(core_pair.y * inverse_rms) * norm_pair.y * gate_pair.y);
	}
}

/*
 * Gated MLA, NoPE absorbed-latent decode over the paged latent cache.
 * Absorb: q_lat[h] = W_kb_key[h]^T q_nope[h]; attend: online softmax of
 * q_lat . c_t over the lane's context; value: v[h] = W_kb_value[h] ctx_lat;
 * a per-head sigmoid gate scales v before the output projection. kv_b rows
 * are head-major: head h key rows occupy [h*(nope+v), h*(nope+v)+nope) and
 * value rows the following v rows, each row SPARK_K3_MODEL_MLA_LATENT_DIMENSION wide.
 */
__global__ void SparkK3MlaKvWriteKernel(const void *kv_a_bf16, void *cache_bf16, const uint32_t *slot_mapping, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	uint64_t destination;
	uint32_t element;
	if ( row >= row_count )
		return;
	destination = (uint64_t)slot_mapping[row] * SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS;
	for (element = threadIdx.x; element < (SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS >> 3u); element += blockDim.x)
		((uint4 *)cache_bf16)[(destination >> 3u) + element] = __ldg(((const uint4 *)kv_a_bf16) + ((((uint64_t)row * SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS) >> 3u)) + element);
}

__global__ void SparkK3MlaAbsorbQueryKernel(const void *query_b_bf16, uint32_t query_b_format, const void *kv_b_payload, const uint8_t *kv_b_scale_e8m0, uint32_t kv_b_format, void *query_latent_bf16, uint32_t row_count)
{
	__shared__ float query_shared[SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION];
	uint32_t row = blockIdx.x,head = blockIdx.y;
	uint64_t query_base = (((uint64_t)row * SPARK_K3_MODEL_MLA_HEAD_COUNT) + head) * SPARK_K3_MODEL_MLA_QK_HEAD_DIMENSION;
	uint64_t weight_row_base = (uint64_t)head * (SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION);
	uint64_t latent_base = (((uint64_t)row * SPARK_K3_MODEL_MLA_HEAD_COUNT) + head) * SPARK_K3_MODEL_MLA_LATENT_DIMENSION;
	uint32_t latent,channel;
	uint64_t weight_index;
	float accumulator,query_value,weight_value,scale_value;
	uint32_t packed;
	if ( row >= row_count )
		return;
	for (channel = threadIdx.x; channel < SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION; channel += blockDim.x)
		query_shared[channel] = SparkLmBf16ToFloat(query_b_bf16,query_base + channel);
	__syncthreads();
	for (latent = threadIdx.x; latent < SPARK_K3_MODEL_MLA_LATENT_DIMENSION; latent += blockDim.x)
	{
		accumulator = 0.0f;
		for (channel = 0; channel < SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION; channel++)
		{
			query_value = query_shared[channel];
			weight_index = ((weight_row_base + channel) * SPARK_K3_MODEL_MLA_LATENT_DIMENSION) + latent;
			if ( kv_b_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
				weight_value = SparkLmBf16ToFloat(kv_b_payload,weight_index);
			else
			{
				packed = ((const uint8_t *)kv_b_payload)[weight_index >> 1u];
				scale_value = SparkLmDecodeE8m0(kv_b_scale_e8m0[weight_index / SPARK_K3_MODEL_MXFP4_GROUP_SIZE]);
				weight_value = SparkLmDecodeE2m1((weight_index & 1u) != 0u ? (packed >> 4u) : (packed & 0x0fu)) * scale_value;
			}
			accumulator += (query_value * weight_value);
		}
		SparkLmFloatToBf16(query_latent_bf16,latent_base + latent,accumulator);
	}
	(void)query_b_format;
}

// Stage tile_count context tokens' latents into shared through the lane's
// block table; the whole block cooperates and leaves synchronized.
__device__ void SparkK3MlaLoadTokenTile(const void *cache_bf16, const uint32_t *physical_block_indices, uint32_t lane, uint32_t lane_stride, uint32_t block_token_count, uint32_t tile_base, uint32_t tile_count, __nv_bfloat16 tile[SPARK_K3_KDA_TILE][SPARK_K3_MODEL_MLA_LATENT_DIMENSION])
{
	uint32_t element,tile_token,logical_block,physical_block;
	uint64_t token_offset;
	__syncthreads();
	for (element = threadIdx.x; element < ((tile_count * SPARK_K3_MODEL_MLA_LATENT_DIMENSION) >> 1u); element += blockDim.x)
	{
		tile_token = element / (SPARK_K3_MODEL_MLA_LATENT_DIMENSION >> 1u);
		logical_block = (tile_base + tile_token) / block_token_count;
		physical_block = __ldg(physical_block_indices + ((uint64_t)lane * lane_stride) + logical_block);
		token_offset = (((uint64_t)physical_block * block_token_count) + ((tile_base + tile_token) % block_token_count)) * SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS;
		((uint32_t *)&tile[tile_token][0])[element % (SPARK_K3_MODEL_MLA_LATENT_DIMENSION >> 1u)] = __ldg(((const uint32_t *)cache_bf16) + (token_offset >> 1u) + (element % (SPARK_K3_MODEL_MLA_LATENT_DIMENSION >> 1u)));
	}
	__syncthreads();
}

__global__ void SparkK3MlaAttendKernel(const void *query_latent_bf16, const void *cache_bf16, const uint32_t *physical_block_indices, uint32_t lane_stride, uint32_t block_token_count, const uint32_t *lane_indices, const uint32_t *context_lengths, void *attention_latent_bf16, uint32_t row_count, float qk_scale)
{
	__shared__ __nv_bfloat16 token_tile[SPARK_K3_KDA_TILE][SPARK_K3_MODEL_MLA_LATENT_DIMENSION];
	uint32_t row = blockIdx.x,head_group = blockIdx.y;
	uint32_t warp = threadIdx.x / SPARK_K3_WARP_LANES,lane_id = threadIdx.x % SPARK_K3_WARP_LANES;
	uint32_t head = (head_group * SPARK_K3_CTA_WARPS) + warp;
	uint64_t query_base = (((uint64_t)row * SPARK_K3_MODEL_MLA_HEAD_COUNT) + head) * SPARK_K3_MODEL_MLA_LATENT_DIMENSION;
	uint32_t context_length,tile_base,tile_count,tile_token,element,latent;
	float maximum,total,score,previous_maximum,correction;
	float context_accumulator[SPARK_K3_MODEL_MLA_LATENT_DIMENSION / SPARK_K3_WARP_LANES];
	float query_register[SPARK_K3_MODEL_MLA_LATENT_DIMENSION / SPARK_K3_WARP_LANES];
	if ( row >= row_count )
		return;
	context_length = context_lengths[row];
	maximum = -3.0e38f;
	total = 0.0f;
	for (element = 0; element < (SPARK_K3_MODEL_MLA_LATENT_DIMENSION / SPARK_K3_WARP_LANES); element++)
		context_accumulator[element] = 0.0f;
	for (latent = 0; latent < (SPARK_K3_MODEL_MLA_LATENT_DIMENSION / SPARK_K3_WARP_LANES); latent++)
		query_register[latent] = SparkLmBf16ToFloat(query_latent_bf16,query_base + ((uint64_t)latent * SPARK_K3_WARP_LANES) + lane_id);
	for (tile_base = 0; tile_base < context_length; tile_base += SPARK_K3_KDA_TILE)
	{
		tile_count = (context_length - tile_base) < SPARK_K3_KDA_TILE ? (context_length - tile_base) : SPARK_K3_KDA_TILE;
		SparkK3MlaLoadTokenTile(cache_bf16,physical_block_indices,lane_indices[row],lane_stride,block_token_count,tile_base,tile_count,token_tile);
		for (tile_token = 0; tile_token < tile_count; tile_token++)
		{
			score = 0.0f;
			#pragma unroll
			for (latent = 0; latent < (SPARK_K3_MODEL_MLA_LATENT_DIMENSION / SPARK_K3_WARP_LANES); latent++)
				score = fmaf(query_register[latent],__bfloat162float(token_tile[tile_token][(latent * SPARK_K3_WARP_LANES) + lane_id]),score);
			score = SparkLmWarpReduceSum(score);
			score = __shfl_sync(0xffffffffu,score,0) * qk_scale;
			if ( lane_id == 0u )
			{
				previous_maximum = maximum;
				if ( score > maximum )
					maximum = score;
				correction = __expf(previous_maximum - maximum);
				score = __expf(score - maximum);
				total = (total * correction) + score;
			}
			maximum = __shfl_sync(0xffffffffu,maximum,0);
			correction = __shfl_sync(0xffffffffu,correction,0);
			score = __shfl_sync(0xffffffffu,score,0);
			total = __shfl_sync(0xffffffffu,total,0);
			for (element = 0; element < (SPARK_K3_MODEL_MLA_LATENT_DIMENSION / SPARK_K3_WARP_LANES); element++)
			{
				latent = (element * SPARK_K3_WARP_LANES) + lane_id;
				context_accumulator[element] = (context_accumulator[element] * correction) + (score * __bfloat162float(token_tile[tile_token][latent]));
			}
		}
	}
	for (element = 0; element < (SPARK_K3_MODEL_MLA_LATENT_DIMENSION / SPARK_K3_WARP_LANES); element++)
	{
		latent = (element * SPARK_K3_WARP_LANES) + lane_id;
		SparkLmFloatToBf16(attention_latent_bf16,query_base + latent,context_accumulator[element] / total);
	}
}

// One kv_b value row dotted against the staged latent: the thread owns
// the whole contiguous row, so weight fetches are 4-byte vector runs -
// eight E2M1 elements or two bf16 per load, one scale per in-group run.
static __device__ __forceinline__ float SparkK3ValueUpRow(const float *latent_shared, const void *payload, const uint8_t *scale_e8m0, uint32_t weight_format, uint64_t weight_row)
{
	uint64_t pair_base = (weight_row * SPARK_K3_MODEL_MLA_LATENT_DIMENSION) >> 1u,run_base = (weight_row * SPARK_K3_MODEL_MLA_LATENT_DIMENSION) >> 3u;
	uint64_t scale_row = weight_row * (SPARK_K3_MODEL_MLA_LATENT_DIMENSION / SPARK_K3_MODEL_MXFP4_GROUP_SIZE);
	uint32_t pair,run,nibble,packed;
	float accumulator = 0.0f,scale_value;
	float2 weight_pair;
	if ( weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
	{
		#pragma unroll 4
		for (pair = 0; pair < (SPARK_K3_MODEL_MLA_LATENT_DIMENSION >> 1u); pair++)
		{
			weight_pair = SparkLmLoadBf16Pair(payload,pair_base + pair);
			accumulator = fmaf(latent_shared[pair << 1u],weight_pair.x,fmaf(latent_shared[(pair << 1u) + 1u],weight_pair.y,accumulator));
		}
		return(accumulator);
	}
	#pragma unroll 2
	for (run = 0; run < (SPARK_K3_MODEL_MLA_LATENT_DIMENSION >> 3u); run++)
	{
		packed = __ldg(((const uint32_t *)payload) + run_base + run);
		scale_value = SparkLmDecodeE8m0(scale_e8m0[scale_row + ((run << 3u) / SPARK_K3_MODEL_MXFP4_GROUP_SIZE)]);
		#pragma unroll
		for (nibble = 0; nibble < 8u; nibble++)
			accumulator = fmaf(latent_shared[(run << 3u) + nibble],SparkLmDecodeE2m1((packed >> (nibble << 2u)) & 0x0fu) * scale_value,accumulator);
	}
	return(accumulator);
}

__global__ void SparkK3MlaValueUpKernel(const void *attention_latent_bf16, const void *kv_b_payload, const uint8_t *kv_b_scale_e8m0, uint32_t kv_b_format, const void *head_gate_bf16, void *head_output_bf16, uint32_t row_count)
{
	__shared__ float latent_shared[SPARK_K3_MODEL_MLA_LATENT_DIMENSION];
	uint32_t row = blockIdx.x,head = blockIdx.y;
	uint64_t latent_base = (((uint64_t)row * SPARK_K3_MODEL_MLA_HEAD_COUNT) + head) * SPARK_K3_MODEL_MLA_LATENT_DIMENSION;
	uint64_t weight_row_base = ((uint64_t)head * (SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION)) + SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION;
	uint64_t output_base = (((uint64_t)row * SPARK_K3_MODEL_MLA_HEAD_COUNT) + head) * SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION;
	uint32_t channel,latent;
	float gate_value,accumulator;
	if ( row >= row_count )
		return;
	gate_value = SparkLmSigmoid(SparkLmBf16ToFloat(head_gate_bf16,((uint64_t)row * SPARK_K3_MODEL_MLA_HEAD_COUNT) + head));
	for (latent = threadIdx.x; latent < SPARK_K3_MODEL_MLA_LATENT_DIMENSION; latent += blockDim.x)
		latent_shared[latent] = SparkLmBf16ToFloat(attention_latent_bf16,latent_base + latent);
	__syncthreads();
	for (channel = threadIdx.x; channel < SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION; channel += blockDim.x)
	{
		accumulator = SparkK3ValueUpRow(latent_shared,kv_b_payload,kv_b_scale_e8m0,kv_b_format,weight_row_base + channel);
		SparkLmFloatToBf16(head_output_bf16,output_base + channel,accumulator * gate_value);
	}
}

/*
 * Router: sigmoid scores with additive bias followed by exact block-parallel
 * top-k.  Every rank uses a CTA-wide argmax with deterministic lower-index
 * tie breaking; only the selected score table remains in shared memory.
 */
__global__ void SparkK3MoeRouteKernel(const void *router_logits_bf16, const float *score_bias_f32, uint32_t *topk_expert_ids, float *topk_weights, uint32_t row_count, float routed_scaling_factor, uint32_t norm_topk)
{
    __shared__ float scores[SPARK_K3_MODEL_MOE_EXPERT_COUNT];
    __shared__ float best_scores[SPARK_LM_CTA_WARPS];
    __shared__ uint32_t best_candidates[SPARK_LM_CTA_WARPS];
    uint32_t row = blockIdx.x;
    uint32_t expert;
    uint32_t pick;
    float total;

    if (row >= row_count)
    {
        return;
    }
    for (expert = threadIdx.x;
         expert < SPARK_K3_MODEL_MOE_EXPERT_COUNT;
         expert += blockDim.x)
    {
        float unbiased_score = SparkLmSigmoid(SparkLmBf16ToFloat(
            router_logits_bf16,
            ((uint64_t)row * SPARK_K3_MODEL_MOE_EXPERT_COUNT) + expert));
        scores[expert] = unbiased_score + score_bias_f32[expert];
    }
    __syncthreads();

    for (pick = 0u; pick < SPARK_K3_MODEL_MOE_TOP_K; ++pick)
    {
        float running_best = -3.0e38f;
        uint32_t running_candidate = UINT32_MAX;

        for (expert = threadIdx.x;
             expert < SPARK_K3_MODEL_MOE_EXPERT_COUNT;
             expert += blockDim.x)
        {
            float candidate_score = scores[expert];
            if (candidate_score > running_best ||
                (candidate_score == running_best && expert < running_candidate))
            {
                running_best = candidate_score;
                running_candidate = expert;
            }
        }
        SparkLmArgmaxReduce(
            running_best,
            running_candidate,
            best_scores,
            best_candidates);
        if (threadIdx.x == 0u)
        {
            uint32_t winner = best_candidates[0];
            float biased_score = best_scores[0];
            topk_expert_ids[((uint64_t)row * SPARK_K3_MODEL_MOE_TOP_K) + pick] = winner;
            topk_weights[((uint64_t)row * SPARK_K3_MODEL_MOE_TOP_K) + pick] =
                biased_score - score_bias_f32[winner];
            scores[winner] = -3.0e38f;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u)
    {
        total = 0.0f;
        for (pick = 0u; pick < SPARK_K3_MODEL_MOE_TOP_K; ++pick)
        {
            total += topk_weights[((uint64_t)row * SPARK_K3_MODEL_MOE_TOP_K) + pick];
        }
        if (norm_topk == 0u || total <= 0.0f)
        {
            total = 1.0f;
        }
        for (pick = 0u; pick < SPARK_K3_MODEL_MOE_TOP_K; ++pick)
        {
            uint64_t route_index = ((uint64_t)row * SPARK_K3_MODEL_MOE_TOP_K) + pick;
            topk_weights[route_index] =
                (topk_weights[route_index] / total) * routed_scaling_factor;
        }
    }
}

__device__ __forceinline__ float SparkK3ExpertWeightValue(uint32_t weight_format, const void *payload, const uint8_t *scale_e8m0, uint64_t weight_index)
{
	uint32_t packed;
	float scale_value;
	if ( weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
		return(SparkLmBf16ToFloat(payload,weight_index));
	packed = ((const uint8_t *)payload)[weight_index >> 1u];
	scale_value = SparkLmDecodeE8m0(scale_e8m0[weight_index / SPARK_K3_MODEL_MXFP4_GROUP_SIZE]);
	return(SparkLmDecodeE2m1((weight_index & 1u) != 0u ? (packed >> 4u) : (packed & 0x0fu)) * scale_value);
}

#define SPARK_K3_LINEAR_BUNDLE_MAX_ENTRY_COUNT 8u
#define SPARK_K3_LINEAR_BUNDLE_OUTPUTS_PER_CTA 64u

typedef struct SparkK3LinearBundleEntry
{
	const void *input_bf16;
	void *output_bf16;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint32_t input_dimension;
	uint32_t output_dimension;
	uint32_t weight_format;
	uint32_t output_offset;
} SparkK3LinearBundleEntry;

typedef struct SparkK3LinearBundleArguments
{
	SparkK3LinearBundleEntry entries[SPARK_K3_LINEAR_BUNDLE_MAX_ENTRY_COUNT];
	uint32_t entry_count;
	uint32_t total_output_dimension;
	uint32_t row_count;
	uint32_t maximum_input_dimension;
} SparkK3LinearBundleArguments;

static __device__ __forceinline__ float SparkK3LinearBundleDot(
	const SparkK3LinearBundleEntry *entry,
	const float *shared_input,
	uint32_t neuron,
	uint32_t lane)
{
	if ( entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
		return(SparkLmDotRowBf16(shared_input,entry->weight_payload,neuron,entry->input_dimension,lane));
	return(SparkLmDotRowMxfp4<SPARK_K3_MODEL_MXFP4_GROUP_SIZE>(
		shared_input,
		entry->weight_payload,
		entry->weight_scale_e8m0,
		neuron,
		entry->input_dimension,
		lane));
}

/*
 * One launch covers a bundle of row-major projections. Each CTA owns a
 * 64-neuron output tile from exactly one projection, stages that projection's
 * input row once, and lets every warp evaluate eight output rows in sequence.
 * The old scalar linear staged the same input once for every eight outputs;
 * this keeps the B1 weight-streaming behavior while cutting input staging and
 * CTA scheduling by eight. Bundle output boundaries are required to be
 * 64-neuron aligned, so a CTA never needs two different input rows.
 */
static __global__ void SparkK3LinearBundleKernel(SparkK3LinearBundleArguments arguments)
{
	extern __shared__ float shared_input[];
	uint32_t row = blockIdx.x;
	uint32_t output_tile_base = blockIdx.y * SPARK_K3_LINEAR_BUNDLE_OUTPUTS_PER_CTA;
	uint32_t warp = threadIdx.x / SPARK_K3_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_K3_WARP_LANES;
	uint32_t entry_index;
	uint32_t element;
	uint32_t output_in_tile;
	uint32_t neuron;
	float accumulator;
	float2 input_pair;
	const SparkK3LinearBundleEntry *entry;

	if ( row >= arguments.row_count || output_tile_base >= arguments.total_output_dimension )
		return;
	entry_index = 0u;
	while ( entry_index + 1u < arguments.entry_count &&
		output_tile_base >= arguments.entries[entry_index + 1u].output_offset )
	{
		entry_index++;
	}
	entry = &arguments.entries[entry_index];
	for (element = threadIdx.x; element < (entry->input_dimension >> 1u); element += blockDim.x)
	{
		input_pair = SparkLmLoadBf16Pair(
			entry->input_bf16,
			(((uint64_t)row * entry->input_dimension) >> 1u) + element);
		shared_input[element << 1u] = input_pair.x;
		shared_input[(element << 1u) + 1u] = input_pair.y;
	}
	for (element = ((entry->input_dimension >> 1u) << 1u) + threadIdx.x;
		element < entry->input_dimension;
		element += blockDim.x)
	{
		shared_input[element] = SparkLmBf16ToFloat(
			entry->input_bf16,
			((uint64_t)row * entry->input_dimension) + element);
	}
	__syncthreads();
	for (output_in_tile = warp;
		output_in_tile < SPARK_K3_LINEAR_BUNDLE_OUTPUTS_PER_CTA;
		output_in_tile += SPARK_K3_CTA_WARPS)
	{
		neuron = output_tile_base + output_in_tile - entry->output_offset;
		if ( neuron >= entry->output_dimension )
			continue;
		accumulator = SparkK3LinearBundleDot(entry,shared_input,neuron,lane);
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
		{
			SparkLmFloatToBf16(
				entry->output_bf16,
				((uint64_t)row * entry->output_dimension) + neuron,
				accumulator);
		}
	}
}

/*
 * SiTU expert intermediate: intermediate = sigmoid(W_gate x) * tanh(W_up x).
 * One block per (row, route); warp per neuron, activations staged in shared.
 * expert_ids == 0 selects the single expert at slot zero of the views with
 * unit weight, which serves the shared expert and dense MLP through the same
 * kernel pair (route_count 1).
 */
__global__ void SparkK3MoeExpertInterKernel(const void *input_bf16, const uint32_t *expert_ids, uint32_t weight_format, const void *gate_payload, const uint8_t *gate_scale, uint64_t gate_payload_stride, uint64_t gate_scale_stride, const void *up_payload, const uint8_t *up_scale, uint64_t up_payload_stride, uint64_t up_scale_stride, void *intermediate_bf16, uint32_t row_count, uint32_t route_count, uint32_t input_dimension, uint32_t intermediate_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t row = blockIdx.x,route = blockIdx.y;
	uint32_t warp = threadIdx.x / SPARK_K3_WARP_LANES,lane = threadIdx.x % SPARK_K3_WARP_LANES;
	uint32_t expert = expert_ids != 0 ? expert_ids[((uint64_t)row * route_count) + route] : 0u;
	const uint8_t *gate_base = ((const uint8_t *)gate_payload) + ((uint64_t)expert * gate_payload_stride);
	const uint8_t *up_base = ((const uint8_t *)up_payload) + ((uint64_t)expert * up_payload_stride);
	const uint8_t *gate_scale_base = gate_scale != 0 ? (gate_scale + ((uint64_t)expert * gate_scale_stride)) : 0;
	const uint8_t *up_scale_base = up_scale != 0 ? (up_scale + ((uint64_t)expert * up_scale_stride)) : 0;
	uint32_t neuron,element;
	float gate_accumulator,up_accumulator;
	float2 stage_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (input_dimension >> 1u); element += blockDim.x)
	{
		stage_pair = SparkLmLoadBf16Pair(input_bf16,(((uint64_t)row * input_dimension) >> 1u) + element);
		shared_input[element << 1u] = stage_pair.x;
		shared_input[(element << 1u) + 1u] = stage_pair.y;
	}
	__syncthreads();
	for (neuron = warp; neuron < intermediate_dimension; neuron += SPARK_K3_CTA_WARPS)
	{
		if ( weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
		{
			gate_accumulator = SparkLmDotRowBf16(shared_input,gate_base,neuron,input_dimension,lane);
			up_accumulator = SparkLmDotRowBf16(shared_input,up_base,neuron,input_dimension,lane);
		}
		else
		{
			gate_accumulator = SparkLmDotRowMxfp4<SPARK_K3_MODEL_MXFP4_GROUP_SIZE>(shared_input,gate_base,gate_scale_base,neuron,input_dimension,lane);
			up_accumulator = SparkLmDotRowMxfp4<SPARK_K3_MODEL_MXFP4_GROUP_SIZE>(shared_input,up_base,up_scale_base,neuron,input_dimension,lane);
		}
		gate_accumulator = SparkLmWarpReduceSum(gate_accumulator);
		up_accumulator = SparkLmWarpReduceSum(up_accumulator);
		if ( lane == 0u )
			SparkLmFloatToBf16(intermediate_bf16,((((uint64_t)row * route_count) + route) * intermediate_dimension) + neuron,SparkLmSigmoid(gate_accumulator) * tanhf(up_accumulator));
	}
}

/*
 * Routed rows are first grouped by expert on device. Launching in grouped-slot
 * order keeps consecutive CTAs on the same expert weight ranges, so L2 can
 * retain gate/up tiles instead of alternating among the row-major route list.
 * The gate and up dot products remain fused and write the activated SiTU row
 * directly; no second intermediate tensor is materialized.
 */
__global__ void SparkK3MoeGroupedExpertInterKernel(
	const void *input_bf16,
	const uint32_t *pair_expert_ids,
	const uint32_t *grouped_rows,
	const uint32_t *grouped_weight_slots,
	uint32_t weight_format,
	const void *gate_payload,
	const uint8_t *gate_scale,
	uint64_t gate_payload_stride,
	uint64_t gate_scale_stride,
	const void *up_payload,
	const uint8_t *up_scale,
	uint64_t up_payload_stride,
	uint64_t up_scale_stride,
	void *grouped_intermediate_bf16,
	uint32_t pair_count,
	uint32_t input_dimension,
	uint32_t intermediate_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t grouped_slot = blockIdx.x;
	uint32_t pair_index;
	uint32_t row;
	uint32_t expert;
	uint32_t warp = threadIdx.x / SPARK_K3_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_K3_WARP_LANES;
	uint32_t neuron;
	uint32_t element;
	const uint8_t *gate_base;
	const uint8_t *up_base;
	const uint8_t *gate_scale_base;
	const uint8_t *up_scale_base;
	float gate_accumulator;
	float up_accumulator;
	float2 input_pair;

	if ( grouped_slot >= pair_count )
		return;
	pair_index = grouped_weight_slots[grouped_slot];
	row = grouped_rows[grouped_slot];
	expert = pair_expert_ids[pair_index];
	gate_base = ((const uint8_t *)gate_payload) +
		((uint64_t)expert * gate_payload_stride);
	up_base = ((const uint8_t *)up_payload) +
		((uint64_t)expert * up_payload_stride);
	gate_scale_base = gate_scale != 0
		? gate_scale + ((uint64_t)expert * gate_scale_stride)
		: 0;
	up_scale_base = up_scale != 0
		? up_scale + ((uint64_t)expert * up_scale_stride)
		: 0;
	for (element = threadIdx.x; element < (input_dimension >> 1u); element += blockDim.x)
	{
		input_pair = SparkLmLoadBf16Pair(
			input_bf16,
			(((uint64_t)row * input_dimension) >> 1u) + element);
		shared_input[element << 1u] = input_pair.x;
		shared_input[(element << 1u) + 1u] = input_pair.y;
	}
	__syncthreads();
	for (neuron = warp; neuron < intermediate_dimension; neuron += SPARK_K3_CTA_WARPS)
	{
		if ( weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
		{
			gate_accumulator = SparkLmDotRowBf16(
				shared_input,
				gate_base,
				neuron,
				input_dimension,
				lane);
			up_accumulator = SparkLmDotRowBf16(
				shared_input,
				up_base,
				neuron,
				input_dimension,
				lane);
		}
		else
		{
			gate_accumulator = SparkLmDotRowMxfp4<SPARK_K3_MODEL_MXFP4_GROUP_SIZE>(
				shared_input,
				gate_base,
				gate_scale_base,
				neuron,
				input_dimension,
				lane);
			up_accumulator = SparkLmDotRowMxfp4<SPARK_K3_MODEL_MXFP4_GROUP_SIZE>(
				shared_input,
				up_base,
				up_scale_base,
				neuron,
				input_dimension,
				lane);
		}
		gate_accumulator = SparkLmWarpReduceSum(gate_accumulator);
		up_accumulator = SparkLmWarpReduceSum(up_accumulator);
		if ( lane == 0u )
		{
			SparkLmFloatToBf16(
				grouped_intermediate_bf16,
				((uint64_t)grouped_slot * intermediate_dimension) + neuron,
				SparkLmSigmoid(gate_accumulator) * tanhf(up_accumulator));
		}
	}
}

// out[row][o] (+)= sum over routes of route_weight * (W_down[e] intermediate).
__global__ void SparkK3MoeExpertDownKernel(const void *intermediate_bf16, const uint32_t *expert_ids, const uint32_t *inverse_map, const float *route_weights, uint32_t weight_format, const void *down_payload, const uint8_t *down_scale, uint64_t down_payload_stride, uint64_t down_scale_stride, void *output_bf16, uint32_t row_count, uint32_t route_count, uint32_t intermediate_dimension, uint32_t output_dimension, uint32_t accumulate)
{
	uint32_t row = blockIdx.x,neuron_base = blockIdx.y * SPARK_K3_CTA_WARPS;
	uint32_t warp = threadIdx.x / SPARK_K3_WARP_LANES,lane = threadIdx.x % SPARK_K3_WARP_LANES;
	uint32_t neuron = neuron_base + warp;
	uint32_t route,expert,element,pair_index,intermediate_slot;
	const uint8_t *down_base;
	const uint8_t *down_scale_base;
	float total,route_accumulator,route_weight;
	float2 stage_pair,weight_pair;
	if ( row >= row_count || neuron >= output_dimension )
		return;
	total = 0.0f;
	for (route = 0; route < route_count; route++)
	{
		pair_index = (row * route_count) + route;
		intermediate_slot = inverse_map != 0 ? inverse_map[pair_index] : pair_index;
		expert = expert_ids != 0 ? expert_ids[pair_index] : 0u;
		route_weight = route_weights != 0 ? route_weights[pair_index] : 1.0f;
		down_base = ((const uint8_t *)down_payload) + ((uint64_t)expert * down_payload_stride);
		down_scale_base = down_scale != 0 ? (down_scale + ((uint64_t)expert * down_scale_stride)) : 0;
		route_accumulator = 0.0f;
		#pragma unroll 2
		for (element = lane; element < (intermediate_dimension >> 1u); element += SPARK_K3_WARP_LANES)
		{
			stage_pair = SparkLmLoadBf16Pair(
				intermediate_bf16,
				(((uint64_t)intermediate_slot * intermediate_dimension) >> 1u) + element);
			if ( weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
			{
				weight_pair = SparkLmLoadBf16Pair(down_base,(((uint64_t)neuron * intermediate_dimension) >> 1u) + element);
				route_accumulator = fmaf(stage_pair.x,weight_pair.x,fmaf(stage_pair.y,weight_pair.y,route_accumulator));
			}
			else
			{
				route_accumulator = fmaf(stage_pair.x,SparkK3ExpertWeightValue(weight_format,down_base,down_scale_base,((uint64_t)neuron * intermediate_dimension) + (element << 1u)),route_accumulator);
				route_accumulator = fmaf(stage_pair.y,SparkK3ExpertWeightValue(weight_format,down_base,down_scale_base,((uint64_t)neuron * intermediate_dimension) + (element << 1u) + 1u),route_accumulator);
			}
		}
		total += (route_weight * route_accumulator);
	}
	total = SparkLmWarpReduceSum(total);
	if ( lane == 0u )
	{
		if ( accumulate != 0u )
			total += SparkLmBf16ToFloat(output_bf16,((uint64_t)row * output_dimension) + neuron);
		SparkLmFloatToBf16(output_bf16,((uint64_t)row * output_dimension) + neuron,total);
	}
}

// Restricted logits over the final normalized hidden plus per-row argmax.
__global__ void SparkK3RestrictedLogitsKernel(const void *normalized_hidden_bf16, const void *lm_head_bf16, float *logits_f32, uint32_t row_count, uint32_t restricted_vocab_count)
{
	uint32_t row = blockIdx.x,candidate = (blockIdx.y * SPARK_K3_CTA_WARPS) + (threadIdx.x / SPARK_K3_WARP_LANES);
	uint32_t lane = threadIdx.x % SPARK_K3_WARP_LANES;
	uint32_t element;
	float accumulator;
	float2 hidden_pair,head_pair;
	if ( row >= row_count || candidate >= restricted_vocab_count )
		return;
	accumulator = 0.0f;
	#pragma unroll 4
	for (element = lane; element < (SPARK_K3_MODEL_HIDDEN_DIMENSION >> 1u); element += SPARK_K3_WARP_LANES)
	{
		hidden_pair = SparkLmLoadBf16Pair(normalized_hidden_bf16,(((uint64_t)row * SPARK_K3_MODEL_HIDDEN_DIMENSION) >> 1u) + element);
		head_pair = SparkLmLoadBf16Pair(lm_head_bf16,(((uint64_t)candidate * SPARK_K3_MODEL_HIDDEN_DIMENSION) >> 1u) + element);
		accumulator = fmaf(hidden_pair.x,head_pair.x,fmaf(hidden_pair.y,head_pair.y,accumulator));
	}
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		logits_f32[((uint64_t)row * restricted_vocab_count) + candidate] = accumulator;
}

__global__ void SparkK3RestrictedArgmaxKernel(const float *logits_f32, const uint32_t *restricted_token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t restricted_vocab_count)
{
	uint32_t row = blockIdx.x;
	uint32_t candidate,best_candidate;
	float best_score,score;
	if ( row >= row_count || threadIdx.x != 0u )
		return;
	best_candidate = 0u;
	best_score = logits_f32[(uint64_t)row * restricted_vocab_count];
	for (candidate = 1; candidate < restricted_vocab_count; candidate++)
	{
		score = logits_f32[((uint64_t)row * restricted_vocab_count) + candidate];
		if ( score > best_score )
		{
			best_score = score;
			best_candidate = candidate;
		}
	}
	output_token_ids[row] = restricted_token_ids[best_candidate];
}

} // namespace

static SparkStatus SparkK3LaunchStatus(void)
{
	return(cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR);
}

static SparkStatus SparkK3LinearBundleAppend(
	SparkK3LinearBundleArguments *arguments,
	const SparkK3Mxfp4LinearView *view,
	const void *input_bf16,
	void *output_bf16)
{
	SparkK3LinearBundleEntry *entry;

	if ( arguments == 0 || view == 0 || input_bf16 == 0 || output_bf16 == 0 ||
		arguments->entry_count >= SPARK_K3_LINEAR_BUNDLE_MAX_ENTRY_COUNT ||
		view->abi_version != SPARK_K3_RESIDENT_DECODE_STAGE_MXFP4_LINEAR_VIEW_ABI_VERSION ||
		view->weight_payload == 0 || view->input_dimension == 0u ||
		view->output_dimension == 0u ||
		(view->output_dimension % SPARK_K3_LINEAR_BUNDLE_OUTPUTS_PER_CTA) != 0u ||
		(view->weight_format != SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 &&
		 view->weight_format != SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1) ||
		(view->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 &&
		 view->weight_scale_e8m0 == 0) ||
		arguments->total_output_dimension > UINT32_MAX - view->output_dimension)
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	entry = &arguments->entries[arguments->entry_count];
	entry->input_bf16 = input_bf16;
	entry->output_bf16 = output_bf16;
	entry->weight_payload = view->weight_payload;
	entry->weight_scale_e8m0 = view->weight_scale_e8m0;
	entry->input_dimension = view->input_dimension;
	entry->output_dimension = view->output_dimension;
	entry->weight_format = view->weight_format;
	entry->output_offset = arguments->total_output_dimension;
	arguments->entry_count++;
	arguments->total_output_dimension += view->output_dimension;
	if ( arguments->maximum_input_dimension < view->input_dimension )
		arguments->maximum_input_dimension = view->input_dimension;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3LaunchLinearBundle(
	SparkK3LinearBundleArguments *arguments,
	uint32_t row_count,
	void *stream)
{
	dim3 grid;
	uint32_t entry_index;

	if ( arguments == 0 || stream == 0 || row_count == 0u ||
		arguments->entry_count == 0u || arguments->total_output_dimension == 0u ||
		arguments->maximum_input_dimension == 0u )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	for (entry_index = 0u; entry_index < arguments->entry_count; ++entry_index)
	{
		if ( arguments->entries[entry_index].output_offset %
				SPARK_K3_LINEAR_BUNDLE_OUTPUTS_PER_CTA != 0u )
		{
			return(SPARK_STATUS_INVALID_ARGUMENT);
		}
	}
	arguments->row_count = row_count;
	grid = dim3(
		row_count,
		arguments->total_output_dimension /
			SPARK_K3_LINEAR_BUNDLE_OUTPUTS_PER_CTA,
		1u);
	SparkK3LinearBundleKernel<<<
		grid,
		SPARK_K3_CTA_THREADS,
		arguments->maximum_input_dimension * (uint32_t)sizeof(float),
		(cudaStream_t)stream>>>(*arguments);
	return(SparkK3LaunchStatus());
}

extern "C" SparkStatus SparkK3ConfigureCudaKernels(void)
{
	SparkK3KdaSmemLayout chunk_plan;
	uint32_t linear_shared_bytes;
	uint32_t expert_shared_bytes;

	chunk_plan = SparkK3KdaSmemPlan();
	linear_shared_bytes = SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION >
		SPARK_K3_MODEL_HIDDEN_DIMENSION
		? SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION * (uint32_t)sizeof(float)
		: SPARK_K3_MODEL_HIDDEN_DIMENSION * (uint32_t)sizeof(float);
	expert_shared_bytes = SPARK_K3_MODEL_HIDDEN_DIMENSION * (uint32_t)sizeof(float);
	if (cudaFuncSetAttribute(
			SparkLmLinearKernel<SPARK_K3_MODEL_MXFP4_GROUP_SIZE>,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			(int)linear_shared_bytes) != cudaSuccess ||
		cudaFuncSetAttribute(
			SparkK3KdaDecodeStepKernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			(int)(SPARK_K3_KDA_DECODE_SMEM_FLOATS * sizeof(float))) != cudaSuccess ||
		cudaFuncSetAttribute(
			SparkK3KdaChunkKernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			(int)chunk_plan.total) != cudaSuccess ||
		cudaFuncSetAttribute(
			SparkK3MoeExpertInterKernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			(int)expert_shared_bytes) != cudaSuccess)
	{
		return SPARK_STATUS_INTERNAL_ERROR;
	}
	return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkK3LaunchEmbeddingGather(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, uint32_t row_count, void *stream)
{
	uint64_t representation_stride;
	if ( node_context == 0 || slot == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( node_context->token_embedding_bf16 == 0 || node_context->owns_embedding == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	representation_stride = (uint64_t)node_context->row_capacity * SPARK_K3_MODEL_HIDDEN_DIMENSION;
	SparkK3EmbeddingGatherKernel<<<row_count,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(slot->input_token_ids,node_context->token_embedding_bf16,slot->attnres_representations_bf16,row_count,representation_stride);
	return(SparkK3LaunchStatus());
}

extern "C" SparkStatus SparkK3LaunchAttnResMix(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3AttnResSiteWeights *site, uint32_t representation_count, uint32_t row_count, void *stream)
{
	uint64_t representation_stride;
	if ( node_context == 0 || slot == 0 || site == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( representation_count == 0u || representation_count > SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	representation_stride = (uint64_t)node_context->row_capacity * SPARK_K3_MODEL_HIDDEN_DIMENSION;
	SparkK3AttnResMixKernel<<<row_count,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(slot->attnres_representations_bf16,representation_stride,site->pseudo_query_bf16,site->key_norm_weight_bf16,slot->mixed_hidden_bf16,representation_count,row_count,node_context->rms_norm_epsilon);
	return(SparkK3LaunchStatus());
}

extern "C" SparkStatus SparkK3LaunchAttnResAccumulate(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const void *sublayer_output_bf16, uint32_t open_new_block, uint32_t completed_block_count, uint32_t row_count, void *stream)
{
	uint64_t representation_stride;
	if ( node_context == 0 || slot == 0 || sublayer_output_bf16 == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( open_new_block > 1u || (completed_block_count + open_new_block) >= SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	representation_stride = (uint64_t)node_context->row_capacity * SPARK_K3_MODEL_HIDDEN_DIMENSION;
	SparkK3AttnResAccumulateKernel<<<row_count,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(slot->attnres_representations_bf16,representation_stride,sublayer_output_bf16,open_new_block,completed_block_count,row_count);
	return(SparkK3LaunchStatus());
}

extern "C" SparkStatus SparkK3LaunchRmsNorm(const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon, void *stream)
{
	if ( input_bf16 == 0 || gain_bf16 == 0 || output_bf16 == 0 || row_count == 0u || dimension == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkLmRmsNormKernel<<<row_count,SPARK_K3_CTA_THREADS,dimension * (uint32_t)sizeof(float),(cudaStream_t)stream>>>(input_bf16,gain_bf16,output_bf16,row_count,dimension,epsilon);
	return(SparkK3LaunchStatus());
}

extern "C" SparkStatus SparkK3LaunchLinear(const SparkK3Mxfp4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count, void *stream)
{
    cudaError_t cuda_status;
    uint32_t common_weight_format;

    if (view == 0 || input_bf16 == 0 || output_bf16 == 0 || row_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (view->weight_payload == 0 ||
        view->input_dimension == 0u ||
        view->output_dimension == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (view->weight_format != SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 &&
        view->weight_format != SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (view->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 &&
        view->weight_scale_e8m0 == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    common_weight_format = view->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16
        ? SPARK_LM_WEIGHT_FORMAT_BF16
        : SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1;
    cuda_status = SparkLmHostLaunchBatchedLinear<SPARK_K3_MODEL_MXFP4_GROUP_SIZE>(
        (cudaStream_t)stream,
        common_weight_format,
        view->weight_payload,
        view->weight_scale_e8m0,
        input_bf16,
        output_bf16,
        row_count,
        view->input_dimension,
        view->output_dimension);
    return cuda_status == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

extern "C" SparkStatus SparkK3LaunchKdaMaterialize(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3KdaLayerWeights *weights, uint32_t row_count, void *stream)
{
	SparkK3LinearBundleArguments input_bundle = {};
	SparkK3LinearBundleArguments low_rank_bundle = {};
	SparkStatus status;
	dim3 grid;
	uint64_t wide_pair_count;
	uint64_t beta_pair_count;
	uint64_t activation_pair_count;
	uint32_t activation_block_count;

	if ( node_context == 0 || slot == 0 || weights == 0 || row_count == 0u || stream == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkK3LinearBundleAppend(
		&input_bundle,
		&weights->query,
		slot->normalized_hidden_bf16,
		slot->kda_query_bf16);
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkK3LinearBundleAppend(
			&input_bundle,
			&weights->key,
			slot->normalized_hidden_bf16,
			slot->kda_key_bf16);
	}
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkK3LinearBundleAppend(
			&input_bundle,
			&weights->value,
			slot->normalized_hidden_bf16,
			slot->kda_value_bf16);
	}
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkK3LinearBundleAppend(
			&input_bundle,
			&weights->decay_low,
			slot->normalized_hidden_bf16,
			slot->kda_decay_low_rank_bf16);
	}
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkK3LinearBundleAppend(
			&input_bundle,
			&weights->output_gate_low,
			slot->normalized_hidden_bf16,
			slot->kda_gate_low_rank_bf16);
	}
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkK3LinearBundleAppend(
			&input_bundle,
			&weights->beta,
			slot->normalized_hidden_bf16,
			slot->kda_beta_bf16);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchLinearBundle(&input_bundle,row_count,stream);
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkK3LinearBundleAppend(
			&low_rank_bundle,
			&weights->decay_high,
			slot->kda_decay_low_rank_bf16,
			slot->kda_log_decay_bf16);
	}
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkK3LinearBundleAppend(
			&low_rank_bundle,
			&weights->output_gate_high,
			slot->kda_gate_low_rank_bf16,
			slot->kda_gate_bf16);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchLinearBundle(&low_rank_bundle,row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	grid = dim3(row_count,SPARK_K3_MODEL_KDA_HEAD_COUNT,1);
	SparkK3KdaNormalizeQkKernel<<<
		grid,
		SPARK_K3_WARP_LANES * 4u,
		0,
		(cudaStream_t)stream>>>(
			slot->kda_query_bf16,
			slot->kda_key_bf16,
			row_count);
	if ( SparkK3LaunchStatus() != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	wide_pair_count = ((uint64_t)row_count * SPARK_K3_MODEL_KDA_QK_DIMENSION) >> 1u;
	beta_pair_count = ((uint64_t)row_count * SPARK_K3_MODEL_KDA_HEAD_COUNT) >> 1u;
	activation_pair_count = wide_pair_count > beta_pair_count
		? wide_pair_count
		: beta_pair_count;
	activation_block_count = (uint32_t)((activation_pair_count +
		SPARK_K3_CTA_THREADS - 1u) / SPARK_K3_CTA_THREADS);
	SparkK3KdaGateBetaKernel<<<
		activation_block_count,
		SPARK_K3_CTA_THREADS,
		0,
		(cudaStream_t)stream>>>(
			slot->kda_log_decay_bf16,
			slot->kda_gate_bf16,
			slot->kda_beta_bf16,
			row_count);
	return(SparkK3LaunchStatus());
}

extern "C" SparkStatus SparkK3LaunchKdaDecodeStep(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, uint32_t kda_layer_ordinal, uint32_t row_count, void *stream)
{
	dim3 grid;
	if ( node_context == 0 || slot == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( node_context->kda_state_pool.state_f32 == 0 || kda_layer_ordinal >= node_context->kda_state_pool.kda_layer_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( slot->lane_indices == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	grid = dim3(SPARK_K3_MODEL_KDA_HEAD_COUNT,row_count,1);
	SparkK3KdaDecodeStepKernel<<<grid,SPARK_K3_CTA_THREADS,SPARK_K3_KDA_DECODE_SMEM_FLOATS * sizeof(float),(cudaStream_t)stream>>>((const __nv_bfloat16 *)slot->kda_query_bf16,(const __nv_bfloat16 *)slot->kda_key_bf16,(const __nv_bfloat16 *)slot->kda_value_bf16,(const __nv_bfloat16 *)slot->kda_log_decay_bf16,(const __nv_bfloat16 *)slot->kda_beta_bf16,node_context->kda_state_pool,slot->lane_indices,node_context->kda_state_pool.state_cold_by_row,kda_layer_ordinal,(__nv_bfloat16 *)slot->kda_core_output_bf16);
	return(SparkK3LaunchStatus());
}

extern "C" SparkStatus SparkK3LaunchKdaChunk(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, uint32_t kda_layer_ordinal, uint32_t sequence_count, uint32_t chunk_token_count, const int32_t *sequence_token_counts, uint32_t carry_state_in, uint32_t write_state_out, void *stream)
{
	SparkK3KdaSmemLayout plan;
	dim3 grid;
	if ( node_context == 0 || slot == 0 || sequence_count == 0u || sequence_token_counts == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( chunk_token_count != SPARK_K3_MODEL_KDA_CHUNK_TOKENS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( node_context->kda_state_pool.state_f32 == 0 || kda_layer_ordinal >= node_context->kda_state_pool.kda_layer_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( slot->lane_indices == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	plan = SparkK3KdaSmemPlan();
	grid = dim3(SPARK_K3_MODEL_KDA_HEAD_COUNT,sequence_count,1);
	SparkK3KdaChunkKernel<<<grid,SPARK_K3_CTA_THREADS,plan.total,(cudaStream_t)stream>>>((const __nv_bfloat16 *)slot->kda_query_bf16,(const __nv_bfloat16 *)slot->kda_key_bf16,(const __nv_bfloat16 *)slot->kda_value_bf16,(const __nv_bfloat16 *)slot->kda_log_decay_bf16,(const __nv_bfloat16 *)slot->kda_beta_bf16,node_context->kda_state_pool,slot->lane_indices,kda_layer_ordinal,(__nv_bfloat16 *)slot->kda_core_output_bf16,sequence_token_counts,chunk_token_count,carry_state_in,write_state_out);
	return(SparkK3LaunchStatus());
}

extern "C" SparkStatus SparkK3LaunchKdaFinish(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3KdaLayerWeights *weights, uint32_t row_count, void *stream)
{
	dim3 grid;
	if ( node_context == 0 || slot == 0 || weights == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	grid = dim3(row_count,SPARK_K3_MODEL_KDA_HEAD_COUNT,1);
	SparkK3KdaFinishKernel<<<grid,SPARK_K3_WARP_LANES * 4u,0,(cudaStream_t)stream>>>(slot->kda_core_output_bf16,weights->head_norm_weight_bf16,slot->kda_gate_bf16,slot->kda_core_output_bf16,row_count,node_context->rms_norm_epsilon);
	if ( SparkK3LaunchStatus() != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(SparkK3LaunchLinear(&weights->output,slot->kda_core_output_bf16,slot->attention_output_hidden_bf16,row_count,stream));
}

extern "C" SparkStatus SparkK3LaunchMlaDecode(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MlaLayerWeights *weights, const SparkK3MlaBlockTableView *block_table, uint32_t layer_ordinal, uint32_t row_count, void *stream)
{
	SparkStatus status;
	dim3 grid;
	uint64_t cache_layer_offset;
	void *cache_layer;
	if ( node_context == 0 || slot == 0 || weights == 0 || block_table == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( node_context->mla_cache_bf16 == 0 || block_table->physical_block_indices == 0 || block_table->lane_stride == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( slot->slot_mapping == 0 || slot->lane_indices == 0 || slot->context_lengths == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( weights->kv_b.weight_payload == 0 || (weights->kv_b.weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && weights->kv_b.weight_scale_e8m0 == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	cache_layer_offset = (uint64_t)layer_ordinal * (uint64_t)node_context->mla_cache_block_count * block_table->block_token_count * SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS;
	cache_layer = ((uint8_t *)node_context->mla_cache_bf16) + (cache_layer_offset * SPARK_K3_MODEL_BF16_ELEMENT_BYTES);
	status = SparkK3LaunchLinear(&weights->query_a,slot->normalized_hidden_bf16,slot->mla_query_a_bf16,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchRmsNorm(slot->mla_query_a_bf16,weights->query_a_norm_weight_bf16,slot->mla_query_a_bf16,row_count,SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION,node_context->rms_norm_epsilon,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchLinear(&weights->query_b,slot->mla_query_a_bf16,slot->mla_query_b_bf16,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchLinear(&weights->kv_a,slot->normalized_hidden_bf16,slot->mla_kv_a_bf16,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchRmsNorm(slot->mla_kv_a_bf16,weights->kv_a_norm_weight_bf16,slot->mla_kv_a_bf16,row_count,SPARK_K3_MODEL_MLA_LATENT_DIMENSION,node_context->rms_norm_epsilon,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchLinear(&weights->head_gate,slot->normalized_hidden_bf16,slot->mla_head_gate_bf16,row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkK3MlaKvWriteKernel<<<row_count,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(slot->mla_kv_a_bf16,cache_layer,slot->slot_mapping,row_count);
	if ( SparkK3LaunchStatus() != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	grid = dim3(row_count,SPARK_K3_MODEL_MLA_HEAD_COUNT,1);
	SparkK3MlaAbsorbQueryKernel<<<grid,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(slot->mla_query_b_bf16,weights->query_b.weight_format,weights->kv_b.weight_payload,weights->kv_b.weight_scale_e8m0,weights->kv_b.weight_format,slot->mla_query_latent_bf16,row_count);
	if ( SparkK3LaunchStatus() != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	grid = dim3(row_count,SPARK_K3_MODEL_MLA_HEAD_COUNT / SPARK_K3_CTA_WARPS,1);
	SparkK3MlaAttendKernel<<<grid,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(slot->mla_query_latent_bf16,cache_layer,block_table->physical_block_indices,block_table->lane_stride,block_table->block_token_count,slot->lane_indices,slot->context_lengths,slot->mla_attention_latent_bf16,row_count,SPARK_K3_MODEL_MLA_QK_SCALE);
	if ( SparkK3LaunchStatus() != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	grid = dim3(row_count,SPARK_K3_MODEL_MLA_HEAD_COUNT,1);
	SparkK3MlaValueUpKernel<<<grid,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(slot->mla_attention_latent_bf16,weights->kv_b.weight_payload,weights->kv_b.weight_scale_e8m0,weights->kv_b.weight_format,slot->mla_head_gate_bf16,slot->mla_head_output_bf16,row_count);
	if ( SparkK3LaunchStatus() != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(SparkK3LaunchLinear(&weights->output,slot->mla_head_output_bf16,slot->attention_output_hidden_bf16,row_count,stream));
}

extern "C" SparkStatus SparkK3LaunchMlaPrefill(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MlaLayerWeights *weights, const SparkK3MlaBlockTableView *block_table, uint32_t layer_ordinal, uint32_t row_count, void *stream)
{
	// Same absorbed path; causality comes from per-row context_lengths set to
	// the row's own position + 1 by the host before the launch.
	return(SparkK3LaunchMlaDecode(node_context,slot,weights,block_table,layer_ordinal,row_count,stream));
}

extern "C" SparkStatus SparkK3LaunchMoeRoute(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MoeLayerWeights *weights, uint32_t row_count, void *stream)
{
	SparkK3Mxfp4LinearView router_view;
	SparkStatus status;
	if ( node_context == 0 || slot == 0 || weights == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( weights->router_weight_bf16 == 0 || weights->router_score_bias_f32 == 0 || weights->expert_count != SPARK_K3_MODEL_MOE_EXPERT_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	router_view.abi_version = SPARK_K3_RESIDENT_DECODE_STAGE_MXFP4_LINEAR_VIEW_ABI_VERSION;
	router_view.weight_format = SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	router_view.input_dimension = SPARK_K3_MODEL_HIDDEN_DIMENSION;
	router_view.output_dimension = SPARK_K3_MODEL_MOE_EXPERT_COUNT;
	router_view.weight_payload = weights->router_weight_bf16;
	router_view.weight_scale_e8m0 = 0;
	router_view.weight_payload_bytes = (uint64_t)SPARK_K3_MODEL_MOE_EXPERT_COUNT * SPARK_K3_MODEL_HIDDEN_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES;
	router_view.weight_scale_bytes = 0u;
	status = SparkK3LaunchLinear(&router_view,slot->normalized_hidden_bf16,slot->moe_gate_bf16,row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkK3MoeRouteKernel<<<row_count,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(slot->moe_gate_bf16,weights->router_score_bias_f32,slot->moe_topk_expert_ids,slot->moe_topk_weights_f32,row_count,node_context->moe_routed_scaling_factor,node_context->moe_norm_topk_prob);
	return(SparkK3LaunchStatus());
}

static SparkStatus SparkK3LaunchExpertPair(const void *input_bf16, const uint32_t *expert_ids, const float *route_weights, uint32_t weight_format, const SparkK3MoeLayerWeights *weights, const void *gate_payload, const uint8_t *gate_scale, uint64_t gate_payload_stride, uint64_t gate_scale_stride, const void *up_payload, const uint8_t *up_scale, uint64_t up_payload_stride, uint64_t up_scale_stride, const void *down_payload, const uint8_t *down_scale, uint64_t down_payload_stride, uint64_t down_scale_stride, void *intermediate_bf16, void *output_bf16, uint32_t row_count, uint32_t route_count, uint32_t intermediate_dimension, uint32_t accumulate, void *stream)
{
	dim3 inter_grid,down_grid;
	uint32_t shared_bytes = SPARK_K3_MODEL_HIDDEN_DIMENSION * (uint32_t)sizeof(float);
	inter_grid = dim3(row_count,route_count,1);
	SparkK3MoeExpertInterKernel<<<inter_grid,SPARK_K3_CTA_THREADS,shared_bytes,(cudaStream_t)stream>>>(input_bf16,expert_ids,weight_format,gate_payload,gate_scale,gate_payload_stride,gate_scale_stride,up_payload,up_scale,up_payload_stride,up_scale_stride,intermediate_bf16,row_count,route_count,SPARK_K3_MODEL_HIDDEN_DIMENSION,intermediate_dimension);
	if ( SparkK3LaunchStatus() != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	down_grid = dim3(row_count,(SPARK_K3_MODEL_HIDDEN_DIMENSION + SPARK_K3_CTA_WARPS - 1u) / SPARK_K3_CTA_WARPS,1);
	SparkK3MoeExpertDownKernel<<<down_grid,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(intermediate_bf16,expert_ids,0,route_weights,weight_format,down_payload,down_scale,down_payload_stride,down_scale_stride,output_bf16,row_count,route_count,intermediate_dimension,SPARK_K3_MODEL_HIDDEN_DIMENSION,accumulate);
	(void)weights;
	return(SparkK3LaunchStatus());
}

static SparkStatus SparkK3LaunchGroupedExpertPair(
	const SparkK3PipelineSlot *slot,
	const SparkK3MoeLayerWeights *weights,
	uint32_t row_count,
	void *stream)
{
	uint32_t pair_count;
	uint32_t shared_bytes;
	dim3 down_grid;
	cudaError_t cuda_status;

	if ( slot == 0 || weights == 0 || stream == 0 || row_count == 0u ||
		slot->moe_topk_expert_ids == 0 || slot->moe_topk_weights_f32 == 0 ||
		slot->moe_expert_offsets == 0 || slot->moe_grouped_rows == 0 ||
		slot->moe_grouped_weight_slots == 0 || slot->moe_inverse_map == 0 ||
		slot->moe_intermediate_bf16 == 0 || slot->moe_output_hidden_bf16 == 0 )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( row_count > UINT32_MAX / SPARK_K3_MODEL_MOE_TOP_K )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	pair_count = row_count * SPARK_K3_MODEL_MOE_TOP_K;
	cuda_status = SparkLmHostLaunchMoeGroup(
		(cudaStream_t)stream,
		slot->moe_topk_expert_ids,
		pair_count,
		SPARK_K3_MODEL_MOE_EXPERT_COUNT,
		SPARK_K3_MODEL_MOE_TOP_K,
		slot->moe_expert_offsets,
		slot->moe_grouped_rows,
		slot->moe_grouped_weight_slots,
		slot->moe_inverse_map);
	if ( cuda_status != cudaSuccess )
		return(SPARK_STATUS_INTERNAL_ERROR);
	shared_bytes = SPARK_K3_MODEL_HIDDEN_DIMENSION * (uint32_t)sizeof(float);
	SparkK3MoeGroupedExpertInterKernel<<<
		pair_count,
		SPARK_K3_CTA_THREADS,
		shared_bytes,
		(cudaStream_t)stream>>>(
			slot->normalized_hidden_bf16,
			slot->moe_topk_expert_ids,
			slot->moe_grouped_rows,
			slot->moe_grouped_weight_slots,
			weights->weight_format,
			weights->expert_gate_payload,
			weights->expert_gate_scale_e8m0,
			weights->expert_gate_payload_stride_bytes,
			weights->expert_gate_scale_stride_bytes,
			weights->expert_up_payload,
			weights->expert_up_scale_e8m0,
			weights->expert_up_payload_stride_bytes,
			weights->expert_up_scale_stride_bytes,
			slot->moe_intermediate_bf16,
			pair_count,
			SPARK_K3_MODEL_HIDDEN_DIMENSION,
			weights->intermediate_dimension);
	if ( SparkK3LaunchStatus() != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	down_grid = dim3(
		row_count,
		(SPARK_K3_MODEL_HIDDEN_DIMENSION + SPARK_K3_CTA_WARPS - 1u) /
			SPARK_K3_CTA_WARPS,
		1u);
	SparkK3MoeExpertDownKernel<<<
		down_grid,
		SPARK_K3_CTA_THREADS,
		0,
		(cudaStream_t)stream>>>(
			slot->moe_intermediate_bf16,
			slot->moe_topk_expert_ids,
			slot->moe_inverse_map,
			slot->moe_topk_weights_f32,
			weights->weight_format,
			weights->expert_down_payload,
			weights->expert_down_scale_e8m0,
			weights->expert_down_payload_stride_bytes,
			weights->expert_down_scale_stride_bytes,
			slot->moe_output_hidden_bf16,
			row_count,
			SPARK_K3_MODEL_MOE_TOP_K,
			weights->intermediate_dimension,
			SPARK_K3_MODEL_HIDDEN_DIMENSION,
			0u);
	return(SparkK3LaunchStatus());
}

extern "C" SparkStatus SparkK3LaunchMoeExperts(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MoeLayerWeights *weights, uint32_t row_count, void *stream)
{
	SparkStatus status;
	if ( node_context == 0 || slot == 0 || weights == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( weights->expert_gate_payload == 0 || weights->expert_up_payload == 0 || weights->expert_down_payload == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkK3LaunchGroupedExpertPair(slot,weights,row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( weights->shared_gate.weight_payload == 0 || weights->shared_up.weight_payload == 0 || weights->shared_down.weight_payload == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkK3LaunchExpertPair(slot->normalized_hidden_bf16,0,0,weights->shared_gate.weight_format,weights,weights->shared_gate.weight_payload,weights->shared_gate.weight_scale_e8m0,0u,0u,weights->shared_up.weight_payload,weights->shared_up.weight_scale_e8m0,0u,0u,weights->shared_down.weight_payload,weights->shared_down.weight_scale_e8m0,0u,0u,slot->moe_intermediate_bf16,slot->moe_output_hidden_bf16,row_count,1u,weights->shared_gate.output_dimension,1u,stream));
}

extern "C" SparkStatus SparkK3LaunchDenseMlp(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MoeLayerWeights *weights, uint32_t row_count, void *stream)
{
	if ( node_context == 0 || slot == 0 || weights == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( weights->shared_gate.weight_payload == 0 || weights->shared_up.weight_payload == 0 || weights->shared_down.weight_payload == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkK3LaunchExpertPair(slot->normalized_hidden_bf16,0,0,weights->shared_gate.weight_format,weights,weights->shared_gate.weight_payload,weights->shared_gate.weight_scale_e8m0,0u,0u,weights->shared_up.weight_payload,weights->shared_up.weight_scale_e8m0,0u,0u,weights->shared_down.weight_payload,weights->shared_down.weight_scale_e8m0,0u,0u,slot->moe_intermediate_bf16,slot->moe_output_hidden_bf16,row_count,1u,weights->shared_gate.output_dimension,0u,stream));
}

extern "C" SparkStatus SparkK3LaunchRestrictedLogits(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, uint32_t row_count, void *stream)
{
	SparkStatus status;
	dim3 grid;
	if ( node_context == 0 || slot == 0 || row_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( node_context->restricted_lm_head_weight_bf16 == 0 || node_context->restricted_token_ids == 0 || node_context->restricted_vocab_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkK3LaunchRmsNorm(slot->mixed_hidden_bf16,node_context->final_norm_weight_bf16,slot->normalized_hidden_bf16,row_count,SPARK_K3_MODEL_HIDDEN_DIMENSION,node_context->rms_norm_epsilon,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	grid = dim3(row_count,(node_context->restricted_vocab_count + SPARK_K3_CTA_WARPS - 1u) / SPARK_K3_CTA_WARPS,1);
	SparkK3RestrictedLogitsKernel<<<grid,SPARK_K3_CTA_THREADS,0,(cudaStream_t)stream>>>(slot->normalized_hidden_bf16,node_context->restricted_lm_head_weight_bf16,slot->restricted_logits_f32,row_count,node_context->restricted_vocab_count);
	if ( SparkK3LaunchStatus() != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	SparkK3RestrictedArgmaxKernel<<<row_count,SPARK_K3_WARP_LANES,0,(cudaStream_t)stream>>>(slot->restricted_logits_f32,node_context->restricted_token_ids,slot->output_token_ids,row_count,node_context->restricted_vocab_count);
	return(SparkK3LaunchStatus());
}
