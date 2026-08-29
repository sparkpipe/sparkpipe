#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "modules/glm5_next_resident_decode_stage/source/cuda/unity.cu"
#include "spark_glm5_next_resident_decode_stage_internal.h"

#define SPARK_GLM5_NEXT_CUDA_THREADS 256u

/* The boundary carries ONE hidden row per token; the HC streams surface
 * initialises every stream to that row (the reference expands the
 * embedding across streams - a stage boundary is the same expansion). */
__global__ static void SparkGlm5NextBoundaryLoadKernel(
	const uint16_t *boundary,
	uint16_t *streams,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,source,stream;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= GLM5_NEXT_HIDDEN )
		return;
	source = (first_row + row) * (uint64_t)GLM5_NEXT_HIDDEN;
	for ( stream = 0u; stream < GLM5_NEXT_HC; ++stream )
		streams[((row * (uint64_t)GLM5_NEXT_HC + stream) * GLM5_NEXT_HIDDEN) + element] = boundary[source + element];
}

/* The store side of the same contract: the stream MEAN is the one hidden
 * row the boundary carries. */
__global__ static void SparkGlm5NextBoundaryStoreKernel(
	const uint16_t *streams,
	uint16_t *boundary,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,destination,stream;
	float value;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= GLM5_NEXT_HIDDEN )
		return;
	value = 0.0f;
	for ( stream = 0u; stream < GLM5_NEXT_HC; ++stream )
		value += LmBf16ToFloat(streams[((row * (uint64_t)GLM5_NEXT_HC + stream) * GLM5_NEXT_HIDDEN) + element]);
	destination = (first_row + row) * (uint64_t)GLM5_NEXT_HIDDEN;
	boundary[destination + element] = LmFloatToBf16(value / (float)GLM5_NEXT_HC);
}

/* Every HC stream initialises to the token's embedding row (the
 * reference: inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)). */
__global__ static void SparkGlm5NextEmbeddingKernel(
	const uint32_t *token_ids,
	const uint16_t *embedding,
	uint16_t *streams,
	uint32_t row_count,
	uint32_t tp_degree,
	uint32_t tp_rank)
{
	uint64_t element,row,source,stream;
	uint32_t token,vocab_per_rank,rank_offset;
	uint16_t value;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= GLM5_NEXT_HIDDEN )
		return;
	vocab_per_rank = GLM5_NEXT_VOCAB / tp_degree;
	rank_offset = tp_rank * vocab_per_rank;
	token = token_ids[row];
	source = (uint64_t)(token - rank_offset) * GLM5_NEXT_HIDDEN + element;
	value = (token >= rank_offset && token < rank_offset + vocab_per_rank) ? embedding[source] : 0u;
	for ( stream = 0u; stream < GLM5_NEXT_HC; ++stream )
		streams[((row * (uint64_t)GLM5_NEXT_HC + stream) * GLM5_NEXT_HIDDEN) + element] = value;
}

__global__ static void SparkGlm5NextWaveMetadataKernel(
	const uint32_t *resident_slots,
	const uint32_t *positions,
	uint32_t *context_lengths,
	uint32_t *dense_row_offset,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		context_lengths[resident_slots[row]] = positions[row] + 1u;
	if ( row == 0u )
	{
		dense_row_offset[0] = 0u;
		dense_row_offset[1] = row_count;
	}
}

static int32_t SparkGlm5NextCudaStatus(cudaError_t status)
{
	return(status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

/* Cross-rank head argmax: pack (score, token) into one u64 per row so the
 * TP collective can reduce-max it. The score's float bits are order-mapped
 * (negative-safe); the token is inverted so equal scores keep the LOWEST
 * token id, and the rank's vocab offset is folded in at pack time. */
static __device__ __forceinline__ uint32_t SparkGlm5NextOrderedHeadScore(float score)
{
	uint32_t bits;
	if ( isnan(score) )
		return(0u);
	bits = __float_as_uint(score);
	return(bits ^ ((bits & UINT32_C(0x80000000)) != 0u ? UINT32_MAX : UINT32_C(0x80000000)));
}

static __global__ void SparkGlm5NextHeadMaxlocPackKernel(
	const float *scores,
	const uint32_t *token_ids,
	uint64_t *maxloc,
	uint32_t row_count,
	uint32_t rank_offset)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		maxloc[row] = ((uint64_t)SparkGlm5NextOrderedHeadScore(scores[row]) << 32u) |
			(UINT32_MAX - (token_ids[row] + rank_offset));
}

static __global__ void SparkGlm5NextHeadMaxlocUnpackKernel(
	const uint64_t *maxloc,
	uint32_t *token_ids,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		token_ids[row] = UINT32_MAX - (uint32_t)maxloc[row];
}

static __device__ __forceinline__ float2 SparkGlm5NextLoadBf16Pair(const void *base,uint64_t element)
{
	/* Little-endian pair layout: x occupies the LOW 16 bits, y the HIGH.
	 * The bits are REINTERPRETED (int_as_float), never integer-converted:
	 * BF16 lives in the top 16 bits of the float's bit pattern. */
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float((int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return(pair);
}

static __device__ __forceinline__ void SparkGlm5NextStoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	uint32_t packed = (__float_as_uint(y) & UINT32_C(0xffff0000)) |
		(__float_as_uint(x) >> 16u);
	((uint32_t *)base)[element] = packed;
}

static __global__ void SparkGlm5NextAccumAddKernel(
	void *destination_bf16,
	const void *source_bf16,
	uint32_t row_count,
	uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 destination_pair,source_pair;
	if ( row >= row_count )
		return;
	for (element=threadIdx.x; element<(width >> 1u); element+=blockDim.x)
	{
		destination_pair = SparkGlm5NextLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkGlm5NextLoadBf16Pair(source_bf16,offset + element);
		SparkGlm5NextStoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkGlm5NextAccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

extern "C" cudaError_t SparkGlm5NextLaunchHeadMaxlocPack(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset)
{
	if ( scores == 0 || token_ids == 0 || maxloc == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkGlm5NextHeadMaxlocPackKernel<<<(row_count + 255u) / 256u,256u,0u,stream>>>(scores,token_ids,maxloc,row_count,rank_offset);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count)
{
	if ( maxloc == 0 || token_ids == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkGlm5NextHeadMaxlocUnpackKernel<<<(row_count + 255u) / 256u,256u,0u,stream>>>(maxloc,token_ids,row_count);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkGlm5NextAccumAddKernel<<<row_count,256u,0u,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkGlm5NextAccumU64MaxKernel<<<(element_count + 255u) / 256u,256u,0u,stream>>>(destination,source,element_count);
	return(cudaPeekAtLastError());
}

static int32_t SparkGlm5NextStageWaveMetadata(const SparkGlm5NextCudaWave *wave)
{
	SparkGlm5NextExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaMemcpyAsync(slot->resident_slots,wave->host_resident_slots,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->positions,wave->host_positions,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && wave->owns_embedding != 0u )
		error = cudaMemcpyAsync(slot->token_ids,wave->host_token_ids,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
	{
		SparkGlm5NextWaveMetadataKernel<<<(wave->row_count + SPARK_GLM5_NEXT_CUDA_THREADS - 1u) / SPARK_GLM5_NEXT_CUDA_THREADS,SPARK_GLM5_NEXT_CUDA_THREADS,0,stream>>>(slot->resident_slots,slot->positions,slot->context_lengths,slot->dense_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	return(SparkGlm5NextCudaStatus(error));
}

static int32_t SparkGlm5NextStageWaveBoundary(const SparkGlm5NextCudaWave *wave)
{
	SparkGlm5NextExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	uint64_t sideband_offset;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaSuccess;
	if ( wave->owns_embedding != 0u )
	{
		SparkGlm5NextEmbeddingKernel<<<dim3((GLM5_NEXT_HIDDEN + SPARK_GLM5_NEXT_CUDA_THREADS - 1u) / SPARK_GLM5_NEXT_CUDA_THREADS,wave->row_count),SPARK_GLM5_NEXT_CUDA_THREADS,0,stream>>>(slot->token_ids,(const uint16_t *)wave->embedding_bf16,slot->hidden_bf16,wave->row_count,wave->tp_degree,wave->tp_rank);
		error = cudaPeekAtLastError();
	}
	else
	{
		SparkGlm5NextBoundaryLoadKernel<<<dim3((GLM5_NEXT_HIDDEN + SPARK_GLM5_NEXT_CUDA_THREADS - 1u) / SPARK_GLM5_NEXT_CUDA_THREADS,wave->row_count),SPARK_GLM5_NEXT_CUDA_THREADS,0,stream>>>((const uint16_t *)wave->hidden_input_bf16,slot->hidden_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error != cudaSuccess || wave->sideband_input == 0u || wave->maximum_context <= GLM5_NEXT_DSA_SELECTED )
		return(SparkGlm5NextCudaStatus(error));
	sideband_offset = wave->sideband_row_offset * (uint64_t)GLM5_NEXT_DSA_SELECTED;
	error = cudaMemcpyAsync(slot->selected_positions,(const uint32_t *)wave->sideband_input_u32 + sideband_offset,(uint64_t)wave->row_count * GLM5_NEXT_DSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream);
	return(SparkGlm5NextCudaStatus(error));
}

static void SparkGlm5NextBuildKvView(
	LmKvView *view,
	uint8_t *pool,
	const SparkGlm5NextCudaWave *wave)
{
	view->pool = pool;
	view->page_table = wave->page_table;
	view->page_table_stride = wave->pages_per_sequence;
	view->sequence_count = wave->resident_sequence_capacity;
	view->pool_page_count = wave->resident_sequence_capacity * wave->pages_per_sequence;
	view->access_error = (LmKvAccessError *)wave->slot->kv_access_error;
}

/* The DSA ordinal of a weight layer: only the 11 sparse layers carry an
 * MLA KV pool; KDA layers never touch it. */
static uint32_t index_ordinal_of(const SparkGlm5NextCudaWave *wave,uint32_t local_layer,uint32_t layer)
{
	(void)wave;
	(void)local_layer;
	/* DSA layers are 3, 7, ..., 43: ordinal = (layer - 3) / 4 + 1 counted
	 * from 0 for layers before the first DSA layer. */
	if ( layer < 3u || layer >= SPARK_GLM5_NEXT_MODEL_LAYER_COUNT ||
	     SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer) )
		return(UINT32_MAX);
	return((layer - 3u) / SPARK_GLM5_NEXT_MODEL_ATTENTION_PERIOD);
}

static void SparkGlm5NextBindLayer(
	const SparkGlm5NextCudaWave *wave,
	uint32_t local_layer,
	Glm5NextLayerBuffers *buffers)
{
	const SparkGlm5NextLayerWeights *weight;
	SparkGlm5NextExecutionSlot *slot;
	uint32_t index_ordinal;
	uint32_t kda_ordinal;
	uint32_t layer;
	weight = &wave->layers[local_layer];
	slot = wave->slot;
	layer = wave->first_layer_index + local_layer;
	memset(buffers,0,sizeof(*buffers));
	buffers->tp_degree = wave->tp_degree;
	buffers->tp_rank = wave->tp_rank;
	buffers->layer_index = layer;
	buffers->attn_heads = SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT / wave->tp_degree;
	buffers->q_b_rows = buffers->attn_heads * (SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION);
	buffers->attn_output_columns = buffers->attn_heads * SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION;
	buffers->dense_gate_up_rows = 2u * SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->dense_intermediate = SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_w1_rows = 2u * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_intermediate = SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_gate_up_rows = 2u * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_intermediate = SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->head_vocabulary = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT / wave->tp_degree;
	buffers->kda_heads = SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT / wave->tp_degree;
	buffers->dense_row_offset = slot->dense_row_offset;
	buffers->dense_tile_prefix = slot->dense_tile_prefix;
	buffers->attn_norm_weight = weight->attn_norm_bf16;
	buffers->q_a_weight = weight->q_a_bf16;
	buffers->q_a_norm_weight = weight->q_a_norm_bf16;
	buffers->q_b_weight = weight->q_b_bf16;
	buffers->kv_a_weight = weight->kv_a_bf16;
	buffers->kv_a_norm_weight = weight->kv_a_norm_bf16;
	/* kv_b is replicated across ranks, but each rank computes only its own
	 * attn_heads local query heads, and the per-head projection kernels index
	 * the weight by LOCAL head id. Offset both per-head tensors to this
	 * rank's head slice (global head tp_rank*attn_heads). */
	{
		uint64_t head_offset = (uint64_t)wave->tp_rank * buffers->attn_heads;
		uint64_t key_head_stride = (uint64_t)GLM5_NEXT_LATENT * GLM5_NEXT_QK_NOPE_DIM;
		uint64_t value_head_stride = (uint64_t)GLM5_NEXT_VALUE_DIM * GLM5_NEXT_LATENT;
		buffers->kv_b_key_transposed_weight = (const uint16_t *)weight->kv_b_key_transposed_bf16 + head_offset * key_head_stride;
		buffers->kv_b_value_weight = (const uint16_t *)weight->kv_b_value_bf16 + head_offset * value_head_stride;
	}
	buffers->index_q_weight = weight->index_q_bf16;
	buffers->index_k_weight = weight->index_k_bf16;
	buffers->index_head_weight = weight->index_head_bf16;
	buffers->index_norm_weight = weight->index_norm_weight_bf16;
	buffers->index_norm_bias = weight->index_norm_bias_bf16;
	buffers->index_compress_ape = (const float *)weight->index_compress_ape_f32;
	buffers->index_compress_gate = weight->index_compress_gate_bf16;
	buffers->qk_scale = SPARK_GLM5_NEXT_MODEL_MLA_QK_SCALE;
	buffers->output_weight = weight->attn_output_bf16;
	buffers->mlp_norm_weight = weight->post_attn_norm_bf16;
	buffers->router_weight = weight->router_bf16;
	buffers->router_correction_bias = weight->router_correction_f32;
	buffers->dense_gate_weight = weight->dense_gate_up_bf16;
	buffers->dense_up_weight = weight->dense_gate_up_bf16 == 0 ? 0 : (const uint16_t *)weight->dense_gate_up_bf16 + ((uint64_t)GLM5_NEXT_DENSE_INTERMEDIATE * GLM5_NEXT_HIDDEN);
	buffers->dense_down_weight = weight->dense_down_bf16;
	buffers->dense_gate_up_fused = weight->dense_gate_up_bf16 != 0 ? 1u : 0u;
	buffers->expert_w1_weight = weight->expert_up_gate_payload;
	buffers->expert_w1_scale = weight->expert_up_gate_scale;
	buffers->expert_w2_weight = weight->expert_down_payload;
	buffers->expert_w2_scale = weight->expert_down_scale;
	buffers->shared_gate_up_weight = weight->shared_gate_up_bf16;
	buffers->shared_down_weight = weight->shared_down_bf16;
	/* KDA: the rank's slice of the row-sharded tensors (the pack stores
	 * per-rank shards; the per-head kernels index LOCAL head ids, so the
	 * rank's head offset is already applied by the pack). The conv and
	 * bias tensors arrive pre-sliced the same way. */
	buffers->kda_qkv_beta_weight = weight->kda_qkv_beta_bf16;
	buffers->kda_decay_gate_down_weight = weight->kda_decay_gate_down_bf16;
	buffers->kda_decay_up_weight = weight->kda_decay_up_bf16;
	buffers->kda_gate_up_weight = weight->kda_gate_up_bf16;
	buffers->kda_q_conv_weight = weight->kda_q_conv_bf16;
	buffers->kda_k_conv_weight = weight->kda_k_conv_bf16;
	buffers->kda_v_conv_weight = weight->kda_v_conv_bf16;
	buffers->kda_decay_bias = (const float *)weight->kda_decay_bias_f32;
	buffers->kda_head_log_scale = (const float *)weight->kda_head_log_scale_f32;
	buffers->kda_out_norm_weight = weight->kda_out_norm_bf16;
	buffers->kda_out_weight = weight->kda_out_bf16;
	/* Hyper-connections (F32 in the pack, replicated). */
	buffers->hc_attn_fn = (const float *)weight->hc_attn_fn_f32;
	buffers->hc_attn_base = (const float *)weight->hc_attn_base_f32;
	buffers->hc_attn_scale = (const float *)weight->hc_attn_scale_f32;
	buffers->hc_ffn_fn = (const float *)weight->hc_ffn_fn_f32;
	buffers->hc_ffn_base = (const float *)weight->hc_ffn_base_f32;
	buffers->hc_ffn_scale = (const float *)weight->hc_ffn_scale_f32;
	/* Scratch surfaces. hidden_bf16 IS the HC streams surface. */
	buffers->hidden_bf16 = slot->hidden_bf16;
	buffers->residual_bf16 = slot->residual_bf16;
	buffers->normed_bf16 = slot->normed_bf16;
	buffers->q_compressed_bf16 = slot->q_compressed_bf16;
	buffers->q_bf16 = slot->q_bf16;
	buffers->query_latent_bf16 = slot->query_latent_bf16;
	buffers->query_rope_bf16 = slot->query_rope_bf16;
	buffers->index_query_bf16 = slot->index_query_bf16;
	buffers->index_key_bf16 = slot->index_key_bf16;
	buffers->index_gate_bf16 = slot->index_gate_bf16;
	buffers->index_packed_bf16 = slot->index_packed_bf16;
	buffers->selected_pools = slot->selected_pools;
	buffers->index_head_weight_bf16 = slot->index_head_weight_bf16;
	buffers->kv_slot_bf16 = slot->kv_slot_bf16;
	buffers->attention_latent_bf16 = slot->attention_latent_bf16;
	buffers->attention_value_bf16 = slot->attention_value_bf16;
	buffers->attention_out_bf16 = slot->attention_out_bf16;
	buffers->gate_up_bf16 = slot->gate_up_bf16;
	buffers->intermediate_bf16 = slot->intermediate_bf16;
	buffers->expert_out_bf16 = slot->expert_out_bf16;
	buffers->shared_out_bf16 = slot->shared_out_bf16;
	buffers->router_logits = slot->router_logits_f32;
	buffers->selection_scores = slot->selection_scores_f32;
	buffers->route_expert = slot->route_expert;
	buffers->route_weight = slot->route_weight;
	buffers->route_source_token = slot->route_source_token;
	buffers->route_packed_row = slot->route_packed_row;
	buffers->head_candidate_score = slot->head_candidate_score;
	buffers->head_candidate_token = slot->head_candidate_token;
	buffers->output_token = slot->output_token;
	buffers->output_score = slot->output_score;
	buffers->group_row_offset = slot->group_row_offset;
	buffers->group_tile_prefix_w1 = slot->group_tile_prefix_w1;
	buffers->group_tile_prefix_w2 = slot->group_tile_prefix_w2;
	buffers->sequence_of_row = slot->resident_slots;
	buffers->context_length = slot->context_lengths;
	buffers->positions = slot->positions;
	buffers->row_positions = slot->positions;
	buffers->selected_positions = slot->selected_positions;
	/* The MLA decode kernel consumes the kpool-expanded width but its
	 * shape gate demands the SELECT width constant; long contexts only
	 * (this gate arm is context > SELECT). */
	buffers->selected_position_count = GLM5_NEXT_DSA_SELECTED;
	/* KDA scratch + state (indexed by KDA ordinal, not layer index). */
	buffers->fused_qkvb_bf16 = slot->fused_qkvb_bf16;
	buffers->fused_decay_gate_bf16 = slot->fused_decay_gate_bf16;
	buffers->kda_decay_latent_bf16 = slot->kda_decay_latent_bf16;
	buffers->kda_gate_latent_bf16 = slot->kda_gate_latent_bf16;
	buffers->kda_beta_logit = slot->kda_beta_logit;
	buffers->kda_gate_bf16 = slot->kda_gate_bf16;
	buffers->kda_decay_logit_bf16 = slot->kda_decay_logit_bf16;
	buffers->kda_output_bf16 = slot->kda_output_bf16;
	buffers->kda_retention = slot->kda_retention;
	buffers->kda_write_gate = slot->kda_write_gate;
	buffers->kda_state_index = wave->kda_state_index;
	buffers->sequence_row_begin = 0; /* decode: row i is sequence i */
	kda_ordinal = wave->kda_ordinal_by_local_layer[local_layer];
	if ( kda_ordinal != UINT32_MAX )
	{
		buffers->kda_state_pool = wave->kda_state_pools + (uint64_t)kda_ordinal * wave->kda_state_layer_stride_bytes;
		buffers->kda_state_slot_bytes = GLM5_NEXT_KDA_STATE_BYTES_PER_LAYER;
		buffers->kda_q_window = (uint16_t *)(wave->kda_q_window_pool + (uint64_t)kda_ordinal * wave->kda_window_layer_stride_bytes);
		buffers->kda_k_window = (uint16_t *)(wave->kda_k_window_pool + (uint64_t)kda_ordinal * wave->kda_window_layer_stride_bytes);
		buffers->kda_v_window = (uint16_t *)(wave->kda_v_window_pool + (uint64_t)kda_ordinal * wave->kda_window_layer_stride_bytes);
	}
	/* HC scratch. */
	buffers->hc_mixes_f32 = slot->hc_mixes_f32;
	buffers->hc_pre_f32 = slot->hc_pre_f32;
	buffers->hc_post_f32 = slot->hc_post_f32;
	buffers->hc_comb_f32 = slot->hc_comb_f32;
	buffers->hc_collapsed_bf16 = slot->hc_collapsed_bf16;
	buffers->hc_snapshot_bf16 = slot->hc_snapshot_bf16;
	buffers->hc_mean_bf16 = slot->hc_mean_bf16;
	index_ordinal = wave->index_ordinal_by_local_layer[local_layer];
	if ( index_ordinal != UINT32_MAX )
		SparkGlm5NextBuildKvView(&buffers->cache,wave->kv_cache + ((uint64_t)index_ordinal * wave->kv_layer_stride_bytes),wave);
	{
		uint32_t dsa_ordinal = index_ordinal_of(wave,local_layer,layer);
		if ( dsa_ordinal != UINT32_MAX )
			SparkGlm5NextBuildKvView(&buffers->index_cache,wave->index_cache + ((uint64_t)dsa_ordinal * wave->index_layer_stride_bytes),wave);
	}
}

/* One attention site: HC wrap + the layer-kind sublayer. */
static int32_t SparkGlm5NextValidateWaveShape(const SparkGlm5NextCudaWave *wave);

static int32_t SparkGlm5NextRunLayerAttention(const SparkGlm5NextCudaWave *wave,uint32_t local_layer)
{
	Glm5NextLayerBuffers buffers;
	uint32_t layer;
	int32_t status;
	cudaStream_t stream;
	layer = wave->first_layer_index + local_layer;
	stream = (cudaStream_t)wave->slot->stream;
	SparkGlm5NextBindLayer(wave,local_layer,&buffers);
	status = Glm5NextHcSite(&buffers,buffers.hc_attn_fn,buffers.hc_attn_base,buffers.hc_attn_scale,wave->row_count,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( wave->tp_rank == 0u && getenv("SPARK_GLM5_NEXT_PROBE") != 0 )
		fprintf(stderr,"G5N-PROBE attn L%u collapsed bf16sum %llu\n",(unsigned)layer,
			(unsigned long long)Glm5NextProbeBf16Sum(stream,(const uint16_t *)buffers.hc_collapsed_bf16,256u));
	if ( wave->tp_rank == 0u && Glm5NextKdaProbeDeep(&buffers) )
	{
		GLM5_NEXT_KDA_PROBE_RAW(stream,layer,"attnsite_collapsed",buffers.hc_collapsed_bf16);
		GLM5_NEXT_KDA_PROBE_RAW(stream,layer,"attnsite_attn_norm_weight",buffers.attn_norm_weight);
	}
	if ( SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer) )
	{
		/* Decode: one row per sequence, null run begins. The KDA out-GEMM
		 * already lands its full-width rank partial in attention_out_bf16;
		 * the HC placement runs AFTER the chain's reduce - see
		 * SparkGlm5NextLaunchCudaLayerAttentionPost. */
		return(Glm5NextLayerKda(&buffers,wave->row_count,wave->row_count,1u,wave->multiprocessor_count,stream));
	}
	status = Glm5NextLayerAttention(&buffers,wave->row_count,wave->maximum_context,layer,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( wave->tp_rank == 0u && getenv("SPARK_GLM5_NEXT_PROBE") != 0 )
		fprintf(stderr,"G5N-PROBE attn L%u dsa_partial bf16sum %llu\n",(unsigned)layer,
			(unsigned long long)Glm5NextProbeBf16Sum(stream,(const uint16_t *)buffers.attention_out_bf16,256u));
	/* MLA already writes its full-width rank partial into attention_out_bf16;
	 * the HC placement runs after the chain's reduce, not here. */
	return(LM_LAUNCH_OK);
}

static int32_t SparkGlm5NextRunLayerMlp(const SparkGlm5NextCudaWave *wave,uint32_t local_layer)
{
	Glm5NextLayerBuffers buffers;
	uint32_t layer,packed_rows;
	int32_t status;
	cudaStream_t stream;
	layer = wave->first_layer_index + local_layer;
	packed_rows = wave->row_count * GLM5_NEXT_TOP_K;
	stream = (cudaStream_t)wave->slot->stream;
	SparkGlm5NextBindLayer(wave,local_layer,&buffers);
	if ( wave->tp_rank == 0u && Glm5NextKdaProbeDeep(&buffers) )
	{
		GLM5_NEXT_KDA_PROBE_RAW(stream,layer,"mlpsite_collapsed",buffers.hc_collapsed_bf16);
		GLM5_NEXT_KDA_PROBE_RAW(stream,layer,"mlpsite_mlp_norm_weight",buffers.mlp_norm_weight);
	}
	status = Glm5NextHcSite(&buffers,buffers.hc_ffn_fn,buffers.hc_ffn_base,buffers.hc_ffn_scale,wave->row_count,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = layer < GLM5_NEXT_FIRST_ROUTED_LAYER ? Glm5NextLayerDenseMlp(&buffers,wave->row_count,wave->multiprocessor_count,stream) : Glm5NextLayerMoe<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(&buffers,wave->row_count,packed_rows,wave->multiprocessor_count,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( wave->tp_rank == 0u && Glm5NextKdaProbeDeep(&buffers) )
	{
		float probe_m[4];
		Glm5NextProbeBf16Floats(stream,buffers.attention_out_bf16,4u,probe_m);
		fprintf(stderr,"G5N-PROBE kda L%u mlp_partial_f %.6g %.6g %.6g %.6g\n",
			(unsigned)buffers.layer_index,(double)probe_m[0],(double)probe_m[1],(double)probe_m[2],(double)probe_m[3]);
	}
	/* dense/MoE wrote the full-width rank partial into attention_out_bf16;
	 * the HC placement runs after the chain's reduce - see
	 * SparkGlm5NextLaunchCudaLayerMlpPost. */
	return(LM_LAUNCH_OK);
}

/* HC placement for one sublayer, run on the REDUCED output. The placement
 * mixes the residual snapshot and the sublayer output into the streams; it
 * must see the SUMMED rank partial exactly once per sublayer. Running it
 * before the reduce made every rank add the replicated snapshot term and
 * the wide reduce then multiplied the residual streams by tp_degree every
 * layer - the layer-17 attention death (values reached 1e18, the attention
 * RMSNorm overflowed, every downstream stage emitted exact zeros). */
static int32_t SparkGlm5NextRunLayerHcPost(const SparkGlm5NextCudaWave *wave,uint32_t local_layer)
{
	Glm5NextLayerBuffers buffers;
	int32_t status;
	cudaStream_t stream;
	if ( wave == 0 || wave->slot == 0 || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	stream = (cudaStream_t)wave->slot->stream;
	SparkGlm5NextBindLayer(wave,local_layer,&buffers);
	/* G5N-PROBE (diag only): the REDUCED attention_out must be bit-identical
	 * on every rank - each rank receives the same summed partials. A
	 * cross-rank checksum divergence convicts the collective path; agreement
	 * convicts the per-rank partial math. Unlike the other probes this one
	 * prints on EVERY rank, gated on the same diag env. */
	if ( getenv("SPARK_GLM5_NEXT_PROBE") != 0 )
	{
		float probe_p[4];
		Glm5NextProbeFloats(stream,buffers.attention_out_bf16,4u,probe_p);
		fprintf(stderr,"G5N-PROBE r%u post L%u bf16sum %llu f %.6g %.6g %.6g %.6g\n",
			(unsigned)wave->tp_rank,(unsigned)(wave->first_layer_index + local_layer),
			(unsigned long long)Glm5NextProbeBf16Sum(stream,(const uint16_t *)buffers.attention_out_bf16,64u),
			(double)probe_p[0],(double)probe_p[1],(double)probe_p[2],(double)probe_p[3]);
	}
	status = Glm5NextHcPost(&buffers,buffers.attention_out_bf16,wave->row_count,stream);
	return(status);
}

extern "C" int32_t SparkGlm5NextLaunchCudaLayerAttentionPost(const SparkGlm5NextCudaWave *wave,uint32_t local_layer)
{
	if ( SparkGlm5NextValidateWaveShape(wave) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkGlm5NextRunLayerHcPost(wave,local_layer));
}

extern "C" int32_t SparkGlm5NextLaunchCudaLayerMlpPost(const SparkGlm5NextCudaWave *wave,uint32_t local_layer)
{
	if ( SparkGlm5NextValidateWaveShape(wave) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkGlm5NextRunLayerHcPost(wave,local_layer));
}

static int32_t SparkGlm5NextRunLayers(const SparkGlm5NextCudaWave *wave)
{
	uint32_t local;
	int32_t status;
	for (local=0u; local<wave->layer_count; local++)
	{
		status = SparkGlm5NextRunLayerAttention(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = SparkGlm5NextRunLayerMlp(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(LM_LAUNCH_OK);
}

static int32_t SparkGlm5NextRunHead(const SparkGlm5NextCudaWave *wave)
{
	SparkGlm5NextExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	int32_t status;
	uint64_t sideband_offset;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaSuccess;
	if ( wave->owns_final_head != 0u )
	{
		Glm5NextLayerBuffers buffers;
		uint32_t rank_offset;
		SparkGlm5NextBindLayer(wave,wave->layer_count - 1u,&buffers);
		/* The head collapse is the UNWEIGHTED stream mean. */
		Glm5NextHcHeadMeanKernel<<<wave->row_count,SPARK_GLM5_NEXT_CUDA_THREADS,0,stream>>>(slot->hidden_bf16,slot->hc_mean_bf16,wave->row_count,GLM5_NEXT_HC,GLM5_NEXT_HIDDEN);
		error = cudaPeekAtLastError();
		if ( error != cudaSuccess )
			return(SparkGlm5NextCudaStatus(error));
		if ( wave->tp_rank == 0u && getenv("SPARK_GLM5_NEXT_PROBE") != 0 )
		{
			float probe_m[4],pb_f[4];
			Glm5NextProbeBf16Floats(stream,slot->hc_mean_bf16,4u,probe_m);
			Glm5NextProbeBf16Floats(stream,slot->hidden_bf16,4u,pb_f);
			fprintf(stderr,"G5N-PROBE head mean_f %.6g %.6g %.6g %.6g stream0_f %.6g %.6g %.6g %.6g\n",
				(double)probe_m[0],(double)probe_m[1],(double)probe_m[2],(double)probe_m[3],
				(double)pb_f[0],(double)pb_f[1],(double)pb_f[2],(double)pb_f[3]);
		}
		status = Glm5NextHeadFullVocab(&buffers,wave->final_norm_bf16,wave->lm_head_bf16,wave->row_count,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		rank_offset = wave->tp_rank * buffers.head_vocabulary;
		error = SparkGlm5NextLaunchHeadMaxlocPack(stream,slot->output_score,slot->output_token,slot->head_maxloc_u64,wave->row_count,rank_offset);
	}
	else
	{
		SparkGlm5NextBoundaryStoreKernel<<<dim3((GLM5_NEXT_HIDDEN + SPARK_GLM5_NEXT_CUDA_THREADS - 1u) / SPARK_GLM5_NEXT_CUDA_THREADS,wave->row_count),SPARK_GLM5_NEXT_CUDA_THREADS,0,stream>>>(slot->hidden_bf16,(uint16_t *)wave->hidden_output_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error != cudaSuccess || wave->sideband_output == 0u )
		return(SparkGlm5NextCudaStatus(error));
	sideband_offset = wave->sideband_row_offset * (uint64_t)GLM5_NEXT_DSA_SELECTED;
	if ( wave->maximum_context > GLM5_NEXT_DSA_SELECTED )
		error = cudaMemcpyAsync((uint32_t *)wave->sideband_output_u32 + sideband_offset,slot->selected_positions,(uint64_t)wave->row_count * GLM5_NEXT_DSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream);
	else
		error = cudaMemsetAsync((uint32_t *)wave->sideband_output_u32 + sideband_offset,0,(uint64_t)wave->row_count * GLM5_NEXT_DSA_SELECTED * sizeof(uint32_t),stream);
	return(SparkGlm5NextCudaStatus(error));
}

static int32_t SparkGlm5NextValidateWaveShape(const SparkGlm5NextCudaWave *wave)
{
	if ( wave == 0 || wave->slot == 0 || wave->slot->stream == 0 || wave->layers == 0 || wave->row_count == 0u || wave->row_count > wave->resident_sequence_capacity || wave->maximum_context == 0u || wave->maximum_context > wave->max_sequence_positions || wave->multiprocessor_count == 0u || wave->tp_degree == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

extern "C" int32_t SparkGlm5NextLaunchCudaWaveBegin(const SparkGlm5NextCudaWave *wave)
{
	int32_t status;
	status = SparkGlm5NextValidateWaveShape(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkGlm5NextStageWaveMetadata(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkGlm5NextStageWaveBoundary(wave);
	return(status);
}

extern "C" int32_t SparkGlm5NextLaunchCudaLayerAttention(const SparkGlm5NextCudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkGlm5NextValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkGlm5NextRunLayerAttention(wave,local_layer));
}

extern "C" int32_t SparkGlm5NextLaunchCudaLayerMlp(const SparkGlm5NextCudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkGlm5NextValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkGlm5NextRunLayerMlp(wave,local_layer));
}

extern "C" int32_t SparkGlm5NextLaunchCudaWaveHead(const SparkGlm5NextCudaWave *wave)
{
	int32_t status;
	status = SparkGlm5NextValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(SparkGlm5NextRunHead(wave));
}

extern "C" int32_t SparkGlm5NextLaunchCudaWave(const SparkGlm5NextCudaWave *wave)
{
	int32_t status;
	status = SparkGlm5NextLaunchCudaWaveBegin(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkGlm5NextRunLayers(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkGlm5NextRunHead(wave);
	return(status);
}

extern "C" int32_t SparkGlm5NextConfigureCudaModule(uint32_t *multiprocessor_count)
{
	cudaDeviceProp properties;
	int32_t device;
	cudaError_t error;
	if ( multiprocessor_count == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	error = cudaGetDevice(&device);
	if ( error == cudaSuccess )
		error = cudaGetDeviceProperties(&properties,device);
	if ( error != cudaSuccess || properties.major != 12 || properties.minor != 1 || properties.multiProcessorCount <= 0 )
		return(LM_LAUNCH_ERR_LAUNCH);
	*multiprocessor_count = (uint32_t)properties.multiProcessorCount;
	return(LM_LAUNCH_OK);
}
