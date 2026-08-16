#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "modules/glm52_resident_decode_stage/source/cuda/unity.cu"
#include "spark_glm52_resident_decode_stage_internal.h"

#define SPARK_GLM52_CUDA_THREADS 256u

__global__ static void SparkGlm52BoundaryLoadKernel(
	const uint16_t *boundary,
	uint16_t *hidden,
	uint16_t *residual,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,source;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= GLM52_HIDDEN )
		return;
	source = (first_row + row) * (2u * (uint64_t)GLM52_HIDDEN);
	hidden[(row * (uint64_t)GLM52_HIDDEN) + element] = boundary[source + element];
	residual[(row * (uint64_t)GLM52_HIDDEN) + element] = boundary[source + GLM52_HIDDEN + element];
}

__global__ static void SparkGlm52BoundaryStoreKernel(
	const uint16_t *hidden,
	const uint16_t *residual,
	uint16_t *boundary,
	uint64_t first_row,
	uint32_t row_count)
{
	uint64_t element,row,destination;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= GLM52_HIDDEN )
		return;
	destination = (first_row + row) * (2u * (uint64_t)GLM52_HIDDEN);
	boundary[destination + element] = hidden[(row * (uint64_t)GLM52_HIDDEN) + element];
	boundary[destination + GLM52_HIDDEN + element] = residual[(row * (uint64_t)GLM52_HIDDEN) + element];
}

__global__ static void SparkGlm52EmbeddingKernel(
	const uint32_t *token_ids,
	const uint16_t *embedding,
	uint16_t *hidden,
	uint16_t *residual,
	uint32_t row_count,
	uint32_t tp_degree,
	uint32_t tp_rank)
{
	uint64_t element,row,source,destination;
	uint32_t token,vocab_per_rank,rank_offset;
	element = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y;
	if ( row >= row_count || element >= GLM52_HIDDEN )
		return;
	vocab_per_rank = GLM52_VOCAB / tp_degree;
	rank_offset = tp_rank * vocab_per_rank;
	token = token_ids[row];
	source = (uint64_t)(token - rank_offset) * GLM52_HIDDEN + element;
	destination = row * (uint64_t)GLM52_HIDDEN + element;
	hidden[destination] = (token >= rank_offset && token < rank_offset + vocab_per_rank) ? embedding[source] : 0u;
	residual[destination] = 0u;
}

__global__ static void SparkGlm52WaveMetadataKernel(
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

static int32_t SparkGlm52CudaStatus(cudaError_t status)
{
	return(status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static void SparkGlm52DbgStage(const char *name,int32_t status)
{
	const char *text;
	if ( status == LM_LAUNCH_OK )
		return;
	text = cudaGetErrorString(cudaGetLastError());
	(void)fprintf(stderr,"GLM52-DBG stage=%s status=%d cuda=%s\n",name,(int)status,text != 0 ? text : "n/a");
}

/* Cross-rank head argmax: pack (score, token) into one u64 per row so the
 * TP collective can reduce-max it. The score's float bits are order-mapped
 * (negative-safe); the token is inverted so equal scores keep the LOWEST
 * token id, and the rank's vocab offset is folded in at pack time. */
static __device__ __forceinline__ uint32_t SparkGlm52OrderedHeadScore(float score)
{
	uint32_t bits;
	if ( isnan(score) )
		return(0u);
	bits = __float_as_uint(score);
	return(bits ^ ((bits & UINT32_C(0x80000000)) != 0u ? UINT32_MAX : UINT32_C(0x80000000)));
}

static __global__ void SparkGlm52HeadMaxlocPackKernel(
	const float *scores,
	const uint32_t *token_ids,
	uint64_t *maxloc,
	uint32_t row_count,
	uint32_t rank_offset)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		maxloc[row] = ((uint64_t)SparkGlm52OrderedHeadScore(scores[row]) << 32u) |
			(UINT32_MAX - (token_ids[row] + rank_offset));
}

static __global__ void SparkGlm52HeadMaxlocUnpackKernel(
	const uint64_t *maxloc,
	uint32_t *token_ids,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		token_ids[row] = UINT32_MAX - (uint32_t)maxloc[row];
}

static __device__ __forceinline__ float2 SparkGlm52LoadBf16Pair(const void *base,uint64_t element)
{
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __uint2float_rn((packed & UINT32_C(0x0000ffff)) << 16u);
	pair.y = __uint2float_rn(packed & UINT32_C(0xffff0000));
	return(pair);
}

static __device__ __forceinline__ void SparkGlm52StoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	/* Little-endian pair layout: x occupies the LOW 16 bits, y the HIGH,
	 * matching SparkGlm52LoadBf16Pair. */
	uint32_t packed = (__float2uint_rn(y) & UINT32_C(0xffff0000)) |
		(__float2uint_rn(x) >> 16u);
	((uint32_t *)base)[element] = packed;
}

static __global__ void SparkGlm52AccumAddKernel(
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
		destination_pair = SparkGlm52LoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkGlm52LoadBf16Pair(source_bf16,offset + element);
		SparkGlm52StoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkGlm52AccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

extern "C" cudaError_t SparkGlm52LaunchHeadMaxlocPack(cudaStream_t stream,const float *scores,const uint32_t *token_ids,uint64_t *maxloc,uint32_t row_count,uint32_t rank_offset)
{
	if ( scores == 0 || token_ids == 0 || maxloc == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkGlm52HeadMaxlocPackKernel<<<(row_count + 255u) / 256u,256u,0u,stream>>>(scores,token_ids,maxloc,row_count,rank_offset);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm52LaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count)
{
	if ( maxloc == 0 || token_ids == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkGlm52HeadMaxlocUnpackKernel<<<(row_count + 255u) / 256u,256u,0u,stream>>>(maxloc,token_ids,row_count);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm52LaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkGlm52AccumAddKernel<<<row_count,256u,0u,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm52LaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkGlm52AccumU64MaxKernel<<<(element_count + 255u) / 256u,256u,0u,stream>>>(destination,source,element_count);
	return(cudaPeekAtLastError());
}

static int32_t SparkGlm52StageWaveMetadata(const SparkGlm52CudaWave *wave)
{
	SparkGlm52ExecutionSlot *slot;
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
		SparkGlm52WaveMetadataKernel<<<(wave->row_count + SPARK_GLM52_CUDA_THREADS - 1u) / SPARK_GLM52_CUDA_THREADS,SPARK_GLM52_CUDA_THREADS,0,stream>>>(slot->resident_slots,slot->positions,slot->context_lengths,slot->dense_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	return(SparkGlm52CudaStatus(error));
}

static int32_t SparkGlm52StageWaveBoundary(const SparkGlm52CudaWave *wave)
{
	SparkGlm52ExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	uint64_t sideband_offset;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaSuccess;
	if ( wave->owns_embedding != 0u )
	{
		SparkGlm52EmbeddingKernel<<<dim3((GLM52_HIDDEN + SPARK_GLM52_CUDA_THREADS - 1u) / SPARK_GLM52_CUDA_THREADS,wave->row_count),SPARK_GLM52_CUDA_THREADS,0,stream>>>(slot->token_ids,(const uint16_t *)wave->embedding_bf16,slot->hidden_bf16,slot->residual_bf16,wave->row_count,wave->tp_degree,wave->tp_rank);
		error = cudaPeekAtLastError();
	}
	else
	{
		SparkGlm52BoundaryLoadKernel<<<dim3((GLM52_HIDDEN + SPARK_GLM52_CUDA_THREADS - 1u) / SPARK_GLM52_CUDA_THREADS,wave->row_count),SPARK_GLM52_CUDA_THREADS,0,stream>>>((const uint16_t *)wave->hidden_input_bf16,slot->hidden_bf16,slot->residual_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error != cudaSuccess || wave->sideband_input == 0u || wave->maximum_context <= GLM52_DSA_SELECTED )
		return(SparkGlm52CudaStatus(error));
	sideband_offset = wave->sideband_row_offset * (uint64_t)GLM52_DSA_SELECTED;
	error = cudaMemcpyAsync(slot->selected_positions,(const uint32_t *)wave->sideband_input_u32 + sideband_offset,(uint64_t)wave->row_count * GLM52_DSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream);
	return(SparkGlm52CudaStatus(error));
}

static void SparkGlm52BuildKvView(
	LmKvView *view,
	uint8_t *pool,
	const SparkGlm52CudaWave *wave)
{
	view->pool = pool;
	view->page_table = wave->page_table;
	view->page_table_stride = wave->pages_per_sequence;
	view->sequence_count = wave->resident_sequence_capacity;
	view->pool_page_count = wave->resident_sequence_capacity * wave->pages_per_sequence;
	view->access_error = (LmKvAccessError *)wave->slot->kv_access_error;
}

static void SparkGlm52BindLayer(
	const SparkGlm52CudaWave *wave,
	uint32_t local_layer,
	Glm52LayerBuffers *buffers)
{
	const SparkGlm52LayerWeights *weight;
	SparkGlm52ExecutionSlot *slot;
	uint32_t index_ordinal;
	weight = &wave->layers[local_layer];
	slot = wave->slot;
	memset(buffers,0,sizeof(*buffers));
	buffers->tp_degree = wave->tp_degree;
	buffers->tp_rank = wave->tp_rank;
	buffers->attn_heads = SPARK_GLM52_MODEL_HEAD_COUNT / wave->tp_degree;
	buffers->q_b_rows = buffers->attn_heads * (SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION + SPARK_GLM52_MODEL_ROPE_DIMENSION);
	buffers->attn_output_columns = buffers->attn_heads * SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION;
	buffers->dense_gate_up_rows = 2u * SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->dense_intermediate = SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_w1_rows = 2u * SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_intermediate = SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_gate_up_rows = 2u * SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_intermediate = SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->head_vocabulary = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT / wave->tp_degree;
	buffers->dense_row_offset = slot->dense_row_offset;
	buffers->dense_tile_prefix = slot->dense_tile_prefix;
	buffers->attn_norm_weight = weight->attn_norm_bf16;
	buffers->q_a_weight = weight->q_a_bf16;
	buffers->q_a_norm_weight = weight->q_a_norm_bf16;
	buffers->q_b_weight = weight->q_b_bf16;
	buffers->kv_a_weight = weight->kv_a_bf16;
	buffers->kv_a_norm_weight = weight->kv_a_norm_bf16;
	buffers->kv_b_key_transposed_weight = weight->kv_b_key_transposed_bf16;
	buffers->kv_b_value_weight = weight->kv_b_value_bf16;
	buffers->index_q_weight = weight->index_q_bf16;
	buffers->index_k_weight = weight->index_k_bf16;
	buffers->index_head_weight = weight->index_head_bf16;
	buffers->index_norm_weight = weight->index_norm_weight_bf16;
	buffers->index_norm_bias = weight->index_norm_bias_bf16;
	buffers->qk_scale = SPARK_GLM52_MODEL_QK_SCALE;
	buffers->output_weight = weight->attn_output_bf16;
	buffers->mlp_norm_weight = weight->post_attn_norm_bf16;
	buffers->router_weight = weight->router_bf16;
	buffers->router_correction_bias = weight->router_correction_f32;
	buffers->dense_gate_weight = weight->dense_gate_up_bf16;
	buffers->dense_up_weight = weight->dense_gate_up_bf16 == 0 ? 0 : (const uint16_t *)weight->dense_gate_up_bf16 + ((uint64_t)GLM52_DENSE_INTERMEDIATE * GLM52_HIDDEN);
	buffers->dense_down_weight = weight->dense_down_bf16;
	buffers->dense_gate_up_fused = weight->dense_gate_up_bf16 != 0 ? 1u : 0u;
	buffers->expert_w1_weight = weight->expert_up_gate_payload;
	buffers->expert_w1_scale = weight->expert_up_gate_scale;
	buffers->expert_w2_weight = weight->expert_down_payload;
	buffers->expert_w2_scale = weight->expert_down_scale;
	buffers->shared_gate_up_weight = weight->shared_gate_up_bf16;
	buffers->shared_down_weight = weight->shared_down_bf16;
	buffers->hidden_bf16 = slot->hidden_bf16;
	buffers->residual_bf16 = slot->residual_bf16;
	buffers->normed_bf16 = slot->normed_bf16;
	buffers->q_compressed_bf16 = slot->q_compressed_bf16;
	buffers->q_bf16 = slot->q_bf16;
	buffers->query_latent_bf16 = slot->query_latent_bf16;
	buffers->query_rope_bf16 = slot->query_rope_bf16;
	buffers->index_query_bf16 = slot->index_query_bf16;
	buffers->index_key_bf16 = slot->index_key_bf16;
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
	buffers->selected_positions = slot->selected_positions;
	buffers->selected_position_count = GLM52_DSA_SELECTED;
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
	SparkGlm52BuildKvView(&buffers->cache,wave->kv_cache + ((uint64_t)local_layer * wave->kv_layer_stride_bytes),wave);
	index_ordinal = wave->index_ordinal_by_local_layer[local_layer];
	if ( index_ordinal != UINT32_MAX )
		SparkGlm52BuildKvView(&buffers->index_cache,wave->index_cache + ((uint64_t)index_ordinal * wave->index_layer_stride_bytes),wave);
}

static int32_t SparkGlm52RunLayerAttention(const SparkGlm52CudaWave *wave,uint32_t local_layer)
{
	Glm52LayerBuffers buffers;
	uint32_t layer;
	int32_t status;
	layer = wave->first_layer_index + local_layer;
	SparkGlm52BindLayer(wave,local_layer,&buffers);
	status = Glm52LayerAttention(&buffers,wave->row_count,wave->maximum_context,layer,wave->multiprocessor_count,(cudaStream_t)wave->slot->stream);
	if ( status != LM_LAUNCH_OK )
	{
		SparkGlm52DbgStage("attention",status);
		(void)fprintf(stderr,"GLM52-DBG layer=%u rows=%u ctx=%u\n",layer,wave->row_count,wave->maximum_context);
	}
	return(status);
}

static int32_t SparkGlm52RunLayerMlp(const SparkGlm52CudaWave *wave,uint32_t local_layer)
{
	Glm52LayerBuffers buffers;
	uint32_t layer,packed_rows;
	int32_t status;
	layer = wave->first_layer_index + local_layer;
	packed_rows = wave->row_count * GLM52_TOP_K;
	SparkGlm52BindLayer(wave,local_layer,&buffers);
	status = layer < GLM52_FIRST_ROUTED_LAYER ? Glm52LayerDenseMlp(&buffers,wave->row_count,wave->multiprocessor_count,(cudaStream_t)wave->slot->stream) : Glm52LayerMoe<GLM52_EXPERT_WEIGHT_CODEC>(&buffers,wave->row_count,packed_rows,wave->multiprocessor_count,(cudaStream_t)wave->slot->stream);
	if ( status != LM_LAUNCH_OK )
	{
		SparkGlm52DbgStage(layer < GLM52_FIRST_ROUTED_LAYER ? "dense_mlp" : "moe",status);
		(void)fprintf(stderr,"GLM52-DBG layer=%u rows=%u ctx=%u\n",layer,wave->row_count,wave->maximum_context);
	}
	return(status);
}

static int32_t SparkGlm52RunLayers(const SparkGlm52CudaWave *wave)
{
	uint32_t local;
	int32_t status;
	for (local=0u; local<wave->layer_count; local++)
	{
		status = SparkGlm52RunLayerAttention(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = SparkGlm52RunLayerMlp(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(LM_LAUNCH_OK);
}

static int32_t SparkGlm52RunHead(const SparkGlm52CudaWave *wave)
{
	SparkGlm52ExecutionSlot *slot;
	cudaStream_t stream;
	cudaError_t error;
	int32_t status;
	uint64_t sideband_offset;
	slot = wave->slot;
	stream = (cudaStream_t)slot->stream;
	error = cudaSuccess;
	if ( wave->owns_final_head != 0u )
	{
		Glm52LayerBuffers buffers;
		uint32_t rank_offset;
		SparkGlm52BindLayer(wave,wave->layer_count - 1u,&buffers);
		status = Glm52HeadFullVocab(&buffers,wave->final_norm_bf16,wave->lm_head_bf16,wave->row_count,stream);
		if ( status != LM_LAUNCH_OK )
		{
			SparkGlm52DbgStage("head",status);
			return(status);
		}
		rank_offset = wave->tp_rank * buffers.head_vocabulary;
		error = SparkGlm52LaunchHeadMaxlocPack(stream,slot->output_score,slot->output_token,slot->head_maxloc_u64,wave->row_count,rank_offset);
	}
	else
	{
		SparkGlm52BoundaryStoreKernel<<<dim3((GLM52_HIDDEN + SPARK_GLM52_CUDA_THREADS - 1u) / SPARK_GLM52_CUDA_THREADS,wave->row_count),SPARK_GLM52_CUDA_THREADS,0,stream>>>(slot->hidden_bf16,slot->residual_bf16,(uint16_t *)wave->hidden_output_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error != cudaSuccess || wave->sideband_output == 0u )
		return(SparkGlm52CudaStatus(error));
	sideband_offset = wave->sideband_row_offset * (uint64_t)GLM52_DSA_SELECTED;
	if ( wave->maximum_context > GLM52_DSA_SELECTED )
		error = cudaMemcpyAsync((uint32_t *)wave->sideband_output_u32 + sideband_offset,slot->selected_positions,(uint64_t)wave->row_count * GLM52_DSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream);
	else
		error = cudaMemsetAsync((uint32_t *)wave->sideband_output_u32 + sideband_offset,0,(uint64_t)wave->row_count * GLM52_DSA_SELECTED * sizeof(uint32_t),stream);
	return(SparkGlm52CudaStatus(error));
}

static int32_t SparkGlm52ValidateWaveShape(const SparkGlm52CudaWave *wave)
{
	if ( wave == 0 || wave->slot == 0 || wave->slot->stream == 0 || wave->layers == 0 || wave->row_count == 0u || wave->row_count > wave->resident_sequence_capacity || wave->maximum_context == 0u || wave->maximum_context > wave->max_sequence_positions || wave->multiprocessor_count == 0u || wave->tp_degree == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

extern "C" int32_t SparkGlm52LaunchCudaWaveBegin(const SparkGlm52CudaWave *wave)
{
	int32_t status;
	status = SparkGlm52ValidateWaveShape(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkGlm52StageWaveMetadata(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkGlm52StageWaveBoundary(wave);
	SparkGlm52DbgStage("wave_begin",status);
	return(status);
}

extern "C" int32_t SparkGlm52LaunchCudaLayerAttention(const SparkGlm52CudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkGlm52ValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkGlm52RunLayerAttention(wave,local_layer));
}

extern "C" int32_t SparkGlm52LaunchCudaLayerMlp(const SparkGlm52CudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkGlm52ValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkGlm52RunLayerMlp(wave,local_layer));
}

extern "C" int32_t SparkGlm52LaunchCudaWaveHead(const SparkGlm52CudaWave *wave)
{
	int32_t status;
	status = SparkGlm52ValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(SparkGlm52RunHead(wave));
}

extern "C" int32_t SparkGlm52LaunchCudaWave(const SparkGlm52CudaWave *wave)
{
	int32_t status;
	status = SparkGlm52LaunchCudaWaveBegin(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkGlm52RunLayers(wave);
	if ( status == LM_LAUNCH_OK )
		status = SparkGlm52RunHead(wave);
	SparkGlm52DbgStage("wave",status);
	return(status);
}

extern "C" int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count)
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
