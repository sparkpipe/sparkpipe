#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "common/common_glm_cuda_tree/spark_glm_cuda_unity.cu"
#include "sparkpipe/spark_tp_mesh_kernels.cuh"
#include "spark_glm52_resident_decode_stage_internal.h"
#define SPARK_FAMILY_CAMEL Glm52
#define SPARK_FAMILY_UPPER GLM52
#define SPARK_FAMILY_LOWER glm52

#include "sparkpipe/family/spark_family.h"

#define SPARK_GLM_CUDA_THREADS 256u

extern "C" int32_t SparkGlm52T1Enabled(void)
{
	static int32_t t1_enabled = -1;
	if ( t1_enabled < 0 )
		t1_enabled = getenv("SPARK_GLM52_T1") != 0 ? 1 : 0;
	return(t1_enabled);
}

__device__ __forceinline__ unsigned long long SparkGlm52GlobalTimerNs()
{
	unsigned long long t;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}

__global__ void SparkGlm52MeshGuardKernel(
	volatile unsigned long long *error_word,
	unsigned long long *output)
{
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	if ( *error_word != 0ull )
	{
		output[0] = 0xFFFFFFFFFFFFFFFFull;
		*error_word = 0ull;
		printf("MESH-GUARD-POISON\n");
	}
}

__global__ void SparkGlm52MeshPublishKernel(
	volatile uint64_t *entry,
	unsigned long long *seq_cell,
	unsigned long long *round_seq,
	uint64_t bytes,
	uint64_t slot_index)
{
	unsigned long long sequence;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	sequence = 1ull + atomicAdd((unsigned long long *)seq_cell,1ull);
	round_seq[0] = sequence;
	entry[2] = slot_index;
	entry[1] = bytes;
	__threadfence_system();
	entry[0] = sequence;
}

__global__ void SparkGlm52MeshWaitKernel(
	volatile uint64_t *band_base,
	uint64_t slot_bytes,
	const unsigned long long *round_seq,
	uint64_t slots_per_rank,
	uint64_t ring,
	uint32_t rank,
	uint32_t degree,
	unsigned long long *error_word,
	unsigned long long deadline_ns)
{
	uint32_t peer;
	volatile uint64_t *end_word;
	uint64_t sequence;
	unsigned long long stop_at;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	sequence = round_seq[0];
	stop_at = SparkGlm52GlobalTimerNs() + deadline_ns;
	for ( peer = 0u; peer < degree - 1u; peer++ )
	{
		uint32_t peer_rank = peer < rank ? peer : peer + 1u;
		end_word = (volatile uint64_t *)
			((uint8_t *)band_base +
			((uint64_t)peer_rank * slots_per_rank +
				(ring & (slots_per_rank - 1ull))) * slot_bytes +
			slot_bytes - 8u);
		while ( *end_word < sequence )
		{
			if ( SparkGlm52GlobalTimerNs() >= stop_at )
			{
				atomicExch((unsigned long long *)error_word,sequence);
				return;
			}
			__nanosleep(200u);
		}
	}
}

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
	if ( row >= row_count || element >= GLM_HIDDEN )
		return;
	source = (first_row + row) * (2u * (uint64_t)GLM_HIDDEN);
	hidden[(row * (uint64_t)GLM_HIDDEN) + element] = boundary[source + element];
	residual[(row * (uint64_t)GLM_HIDDEN) + element] = boundary[source + GLM_HIDDEN + element];
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
	if ( row >= row_count || element >= GLM_HIDDEN )
		return;
	destination = (first_row + row) * (2u * (uint64_t)GLM_HIDDEN);
	boundary[destination + element] = hidden[(row * (uint64_t)GLM_HIDDEN) + element];
	boundary[destination + GLM_HIDDEN + element] = residual[(row * (uint64_t)GLM_HIDDEN) + element];
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
	if ( row >= row_count || element >= GLM_HIDDEN )
		return;
	vocab_per_rank = GLM_VOCAB / tp_degree;
	rank_offset = tp_rank * vocab_per_rank;
	token = token_ids[row];
	source = (uint64_t)(token - rank_offset) * GLM_HIDDEN + element;
	destination = row * (uint64_t)GLM_HIDDEN + element;
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

static __device__ __forceinline__ float2 SparkGlm52LoadBf16Pair(const void *base,uint64_t element)
{
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float((int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return(pair);
}

static __device__ __forceinline__ void SparkGlm52StoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	uint32_t packed = (__float_as_uint(y) & UINT32_C(0xffff0000)) |
		(__float_as_uint(x) >> 16u);
	((uint32_t *)base)[element] = packed;
}

#include "sparkpipe/family/glm/spark_glm_head_maxloc.cuh"

#include "sparkpipe/family/glm/spark_glm_head_maxloc_unpack.cuh"

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
		SparkGlm52WaveMetadataKernel<<<(wave->row_count + SPARK_GLM_CUDA_THREADS - 1u) / SPARK_GLM_CUDA_THREADS,SPARK_GLM_CUDA_THREADS,0,stream>>>(slot->resident_slots,slot->positions,slot->context_lengths,slot->dense_row_offset,wave->row_count);
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
		SparkGlm52EmbeddingKernel<<<dim3((GLM_HIDDEN + SPARK_GLM_CUDA_THREADS - 1u) / SPARK_GLM_CUDA_THREADS,wave->row_count),SPARK_GLM_CUDA_THREADS,0,stream>>>(slot->token_ids,(const uint16_t *)wave->embedding_bf16,slot->hidden_bf16,slot->residual_bf16,wave->row_count,wave->tp_degree,wave->tp_rank);
		error = cudaPeekAtLastError();
	}
	else
	{
		SparkGlm52BoundaryLoadKernel<<<dim3((GLM_HIDDEN + SPARK_GLM_CUDA_THREADS - 1u) / SPARK_GLM_CUDA_THREADS,wave->row_count),SPARK_GLM_CUDA_THREADS,0,stream>>>((const uint16_t *)wave->hidden_input_bf16,slot->hidden_bf16,slot->residual_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error != cudaSuccess || wave->sideband_input == 0u || wave->maximum_context <= GLM_DSA_SELECTED )
		return(SparkGlm52CudaStatus(error));
	sideband_offset = wave->sideband_row_offset * (uint64_t)GLM_DSA_SELECTED;
	error = cudaMemcpyAsync(slot->selected_positions,(const uint32_t *)wave->sideband_input_u32 + sideband_offset,(uint64_t)wave->row_count * GLM_DSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream);
	return(SparkGlm52CudaStatus(error));
}

#include "sparkpipe/family/glm/spark_glm_kv_view.cuh"

static void SparkGlm52BindLayer(
	const SparkGlm52CudaWave *wave,
	uint32_t local_layer,
	GlmLayerBuffers *buffers)
{
	const SparkGlm52LayerWeights *weight;
	SparkGlm52ExecutionSlot *slot;
	uint32_t index_ordinal;
	weight = &wave->layers[local_layer];
	slot = wave->slot;
	memset(buffers,0,sizeof(*buffers));
	buffers->tp_degree = wave->tp_degree;
	buffers->tp_rank = wave->tp_rank;
	buffers->attn_heads = SPARK_GLM_MODEL_HEAD_COUNT / wave->tp_degree;
	buffers->q_b_rows = buffers->attn_heads * (SPARK_GLM_MODEL_QK_NOPE_HEAD_DIMENSION + SPARK_GLM_MODEL_ROPE_DIMENSION);
	buffers->attn_output_columns = buffers->attn_heads * SPARK_GLM_MODEL_VALUE_HEAD_DIMENSION;
	buffers->dense_gate_up_rows = 2u * SPARK_GLM_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->dense_intermediate = SPARK_GLM_MODEL_DENSE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_w1_rows = 2u * SPARK_GLM_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->expert_intermediate = SPARK_GLM_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_gate_up_rows = 2u * SPARK_GLM_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->shared_intermediate = SPARK_GLM_MODEL_MOE_INTERMEDIATE_DIMENSION / wave->tp_degree;
	buffers->head_vocabulary = SPARK_GLM_MODEL_OUTPUT_VOCAB_COUNT / wave->tp_degree;
	buffers->dense_row_offset = slot->dense_row_offset;
	buffers->dense_tile_prefix = slot->dense_tile_prefix;
	buffers->attn_norm_weight = weight->attn_norm_bf16;
	buffers->q_a_weight = weight->q_a_bf16;
	buffers->q_a_norm_weight = weight->q_a_norm_bf16;
	buffers->q_b_weight = weight->q_b_bf16;
	buffers->kv_a_weight = weight->kv_a_bf16;
	buffers->kv_a_norm_weight = weight->kv_a_norm_bf16;
	{
		uint64_t head_offset = (uint64_t)wave->tp_rank * buffers->attn_heads;
		uint64_t key_head_stride = (uint64_t)GLM_LATENT * GLM_QK_NOPE_DIM;
		uint64_t value_head_stride = (uint64_t)GLM_VALUE_DIM * GLM_LATENT;
		buffers->kv_b_key_transposed_weight = (const uint16_t *)weight->kv_b_key_transposed_bf16 + head_offset * key_head_stride;
		buffers->kv_b_value_weight = (const uint16_t *)weight->kv_b_value_bf16 + head_offset * value_head_stride;
	}
	buffers->index_q_weight = weight->index_q_bf16;
	buffers->index_k_weight = weight->index_k_bf16;
	buffers->index_head_weight = weight->index_head_bf16;
	buffers->index_norm_weight = weight->index_norm_weight_bf16;
	buffers->index_norm_bias = weight->index_norm_bias_bf16;
	buffers->qk_scale = SPARK_GLM_MODEL_QK_SCALE;
	buffers->output_weight = weight->attn_output_bf16;
	buffers->mlp_norm_weight = weight->post_attn_norm_bf16;
	buffers->router_weight = weight->router_bf16;
	buffers->router_correction_bias = weight->router_correction_f32;
	buffers->dense_gate_weight = weight->dense_gate_up_bf16;
	buffers->dense_up_weight = weight->dense_gate_up_bf16 == 0 ? 0 : (const uint16_t *)weight->dense_gate_up_bf16 + ((uint64_t)GLM_DENSE_INTERMEDIATE * GLM_HIDDEN);
	buffers->dense_down_weight = weight->dense_down_bf16;
	buffers->dense_gate_up_fused = weight->dense_gate_up_bf16 != 0 ? 1u : 0u;
	if ( wave->expert_lease_base != 0 &&
		wave->expert_lease_local_layer == local_layer )
	{
		buffers->expert_w1_weight = wave->expert_lease_base + weight->expert_up_gate_payload_offset;
		buffers->expert_w1_scale = weight->expert_up_gate_scale == 0 ? 0 : wave->expert_lease_base + weight->expert_up_gate_scale_offset;
		buffers->expert_w2_weight = wave->expert_lease_base + weight->expert_down_payload_offset;
		buffers->expert_w2_scale = weight->expert_down_scale == 0 ? 0 : wave->expert_lease_base + weight->expert_down_scale_offset;
	}
	else
	{
		buffers->expert_w1_weight = weight->expert_up_gate_payload;
		buffers->expert_w1_scale = weight->expert_up_gate_scale;
		buffers->expert_w2_weight = weight->expert_down_payload;
		buffers->expert_w2_scale = weight->expert_down_scale;
	}
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
	buffers->selected_position_count = GLM_DSA_SELECTED;
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
	buffers->attention_split_partials = wave->attention_split_partials_f32;
	buffers->attention_split_partial_blocks = wave->attention_split_partial_blocks;
	buffers->decode_split_context_threshold = wave->decode_split_context_threshold;
	SparkGlm52BuildKvView(&buffers->cache,wave->kv_cache + ((uint64_t)local_layer * wave->kv_layer_stride_bytes),wave);
	index_ordinal = wave->index_ordinal_by_local_layer[local_layer];
	if ( index_ordinal != UINT32_MAX )
		SparkGlm52BuildKvView(&buffers->index_cache,wave->index_cache + ((uint64_t)index_ordinal * wave->index_layer_stride_bytes),wave);
}

static int32_t SparkGlm52RunLayerAttention(const SparkGlm52CudaWave *wave,uint32_t local_layer)
{
	GlmLayerBuffers buffers;
	uint32_t layer;
	int32_t status;
	layer = wave->first_layer_index + local_layer;
	SparkGlm52BindLayer(wave,local_layer,&buffers);
	status = GlmLayerAttention(&buffers,wave->row_count,wave->maximum_context,layer,wave->multiprocessor_count,(cudaStream_t)wave->slot->stream);
	return(status);
}

static int32_t SparkGlm52RunLayerMlpRoute(const SparkGlm52CudaWave *wave,uint32_t local_layer)
{
	GlmLayerBuffers buffers;
	uint32_t layer,packed_rows;
	int32_t status;
	cudaError_t error;
	layer = wave->first_layer_index + local_layer;
	packed_rows = wave->row_count * GLM_TOP_K;
	SparkGlm52BindLayer(wave,local_layer,&buffers);
	if ( layer < GLM_FIRST_ROUTED_LAYER )
		return(GlmLayerDenseMlp(&buffers,wave->row_count,wave->multiprocessor_count,(cudaStream_t)wave->slot->stream));
	status = GlmLayerMoeRoute<GLM_EXPERT_WEIGHT_CODEC>(&buffers,wave->row_count,packed_rows,wave->multiprocessor_count,(cudaStream_t)wave->slot->stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	wave->slot->route_recorded = 0u;
	if ( wave->slot->group_row_offset_host != 0 && wave->slot->route_ready_event != 0 )
	{
		error = cudaMemcpyAsync(wave->slot->group_row_offset_host,wave->slot->group_row_offset,(GLM_EXPERTS + 1u) * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)wave->slot->stream);
		if ( error == cudaSuccess )
			error = cudaEventRecord((cudaEvent_t)wave->slot->route_ready_event,(cudaStream_t)wave->slot->stream);
		if ( error != cudaSuccess )
			return(LM_LAUNCH_ERR_LAUNCH);
		wave->slot->route_recorded = 1u;
	}
	return(LM_LAUNCH_OK);
}

static int32_t SparkGlm52RunLayerMlpExperts(const SparkGlm52CudaWave *wave,uint32_t local_layer)
{
	GlmLayerBuffers buffers;
	uint32_t layer,packed_rows;
	layer = wave->first_layer_index + local_layer;
	packed_rows = wave->row_count * GLM_TOP_K;
	if ( layer < GLM_FIRST_ROUTED_LAYER )
		return(LM_LAUNCH_OK);
	SparkGlm52BindLayer(wave,local_layer,&buffers);
	return(GlmLayerMoeExperts<GLM_EXPERT_WEIGHT_CODEC>(&buffers,wave->row_count,packed_rows,wave->multiprocessor_count,(cudaStream_t)wave->slot->stream));
}

#include "sparkpipe/family/glm/spark_glm_head_maxloc_launch.cuh"

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
		GlmLayerBuffers buffers;
		uint32_t rank_offset;
		SparkGlm52BindLayer(wave,wave->layer_count - 1u,&buffers);
		rank_offset = wave->tp_rank * buffers.head_vocabulary;
		if ( wave->row_count == 1u && wave->head_certified_fp8_payload != 0 &&
			SparkGlm52T1Enabled() == 0 )
			status = GlmHeadCertifiedB1(&buffers,wave->final_norm_bf16,wave->lm_head_bf16,wave->head_certified_fp8_payload,wave->head_certified_fp8_scale_f32,wave->head_certified_fp8_norm_f32,slot->head_certified_scratch,slot->head_certified_candidates,slot->head_screened_count,rank_offset,buffers.head_vocabulary,stream);
		else
			status = GlmHeadFullVocab(&buffers,wave->final_norm_bf16,wave->lm_head_bf16,wave->row_count,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		error = SparkGlm52LaunchHeadMaxlocPack(stream,slot->output_score,slot->output_token,slot->head_maxloc_u64,wave->row_count,rank_offset);
	}
	else
	{
		SparkGlm52BoundaryStoreKernel<<<dim3((GLM_HIDDEN + SPARK_GLM_CUDA_THREADS - 1u) / SPARK_GLM_CUDA_THREADS,wave->row_count),SPARK_GLM_CUDA_THREADS,0,stream>>>(slot->hidden_bf16,slot->residual_bf16,(uint16_t *)wave->hidden_output_bf16,wave->boundary_row_offset,wave->row_count);
		error = cudaPeekAtLastError();
	}
	if ( error != cudaSuccess || wave->sideband_output == 0u )
		return(SparkGlm52CudaStatus(error));
	sideband_offset = wave->sideband_row_offset * (uint64_t)GLM_DSA_SELECTED;
	if ( wave->maximum_context > GLM_DSA_SELECTED )
		error = cudaMemcpyAsync((uint32_t *)wave->sideband_output_u32 + sideband_offset,slot->selected_positions,(uint64_t)wave->row_count * GLM_DSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream);
	else
		error = cudaMemsetAsync((uint32_t *)wave->sideband_output_u32 + sideband_offset,0,(uint64_t)wave->row_count * GLM_DSA_SELECTED * sizeof(uint32_t),stream);
	return(SparkGlm52CudaStatus(error));
}

static int32_t SparkGlm52ValidateWaveShape(const SparkGlm52CudaWave *wave)
{
	if ( wave == 0 || wave->slot == 0 || wave->slot->stream == 0 || wave->layers == 0 || wave->row_count == 0u || wave->row_count > wave->resident_sequence_capacity || wave->maximum_context == 0u || wave->maximum_context > wave->max_sequence_positions || wave->multiprocessor_count == 0u || wave->tp_degree == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

extern "C" int32_t SparkGlm52LaunchCudaLayerMlpRoute(const SparkGlm52CudaWave *wave,uint32_t local_layer)
{
	int32_t status;
	status = SparkGlm52ValidateWaveShape(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( wave->slot->route_ready_event == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SparkGlm52RunLayerMlpRoute(wave,local_layer));
}

#include "sparkpipe/family/glm/spark_glm_layer_mlp.cuh"

#include "sparkpipe/family/glm/spark_glm_cuda_wave.cuh"

#include "sparkpipe/family/glm/spark_glm_layer_mlp_experts.cuh"
