#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_stagepack_format.h"

#define ROOF_TP 16u
#define ROOF_LAYERS SPARK_GLM5_NEXT_MODEL_LAYER_COUNT
#define ROOF_EXPERTS SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT
#define ROOF_KINDS 3u
#define ROOF_MAX_COPIES 8u
#define ROOF_MAX_BATCHES 16u
#define ROOF_PHASES 7u
#define ROOF_PHASE_EVENTS (2u + ROOF_LAYERS * 5u)
#define ROOF_EXPERT_INTERMEDIATE (SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION / ROOF_TP)
#define ROOF_EXPERT_W1_ROWS (2u * ROOF_EXPERT_INTERMEDIATE)
#define ROOF_CUDA(call) do { cudaError_t roof_error = (call); if ( roof_error != cudaSuccess ) { fprintf(stderr,"ROOFLINE-FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(roof_error),#call); exit(1); } } while (0)
#define ROOF_LAUNCH(call) do { int32_t roof_status = (call); if ( roof_status != LM_LAUNCH_OK ) { fprintf(stderr,"ROOFLINE-FAIL line=%d launch=%d call=%s\n",__LINE__,(int)roof_status,#call); exit(1); } } while (0)

typedef struct RoofConfig
{
	uint32_t batches[ROOF_MAX_BATCHES];
	uint32_t batch_count,context,iterations,copies,max_batch,index_cp;
	double round_us,nic_gbps,memory_gbps;
}
RoofConfig;

typedef struct RoofBytes
{
	uint64_t fixed,expert;
}
RoofBytes;

typedef struct RoofModel
{
	SparkGlm5NextLayerWeights templates[ROOF_MAX_COPIES][ROOF_KINDS];
	SparkGlm5NextLayerWeights layers[ROOF_LAYERS];
	RoofBytes bytes[ROOF_KINDS];
	void *final_norm,*lm_head;
	uint64_t head_bytes;
}
RoofModel;

typedef struct RoofState
{
	SparkGlm5NextExecutionSlot slot;
	SparkGlm5NextCudaWave wave;
	uint32_t index_ordinals[ROOF_LAYERS],kda_ordinals[ROOF_LAYERS];
	uint32_t *host_slots,*host_positions,*host_run_begin,*host_run_rows,*host_run_state;
	void *hidden_input;
	uint32_t *page_table;
	uint32_t pages_per_sequence,max_positions;
}
RoofState;

static __global__ void RoofFillKernel(void *data,uint64_t count,uint32_t kind,uint32_t seed,float amplitude)
{
	uint64_t index,hash;
	float value;
	uint8_t byte;
	for (index=(uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index<count; index+=(uint64_t)blockDim.x * gridDim.x)
	{
		hash = (index + 1u) * 0x9E3779B97F4A7C15ull ^ ((uint64_t)seed << 32u);
		hash ^= hash >> 29u;
		hash *= 0xBF58476D1CE4E5B9ull;
		hash ^= hash >> 32u;
		value = amplitude * (((float)(hash & 0xFFFFu) / 32768.0f) - 1.0f);
		byte = (uint8_t)(((hash >> 16u) & 0x37u) | ((hash >> 24u) & 0x80u));
		if ( kind == 3u || kind == 4u )
			value = amplitude;
		if ( kind == 0u || kind == 3u )
			((uint16_t *)data)[index] = (uint16_t)(__float_as_uint(value) >> 16u);
		else if ( kind == 1u || kind == 4u )
			((float *)data)[index] = value;
		else
			((uint8_t *)data)[index] = byte;
	}
}

static void *RoofAllocate(uint64_t bytes)
{
	void *data;
	data = 0;
	if ( bytes == 0u )
		return(0);
	ROOF_CUDA(cudaMalloc(&data,(size_t)bytes));
	ROOF_CUDA(cudaMemset(data,0,(size_t)bytes));
	return(data);
}

static void *RoofFilled(uint64_t bytes,uint32_t kind,uint32_t seed,float amplitude)
{
	void *data;
	uint64_t count;
	data = RoofAllocate(bytes);
	count = kind == 0u || kind == 3u ? bytes / 2u : kind == 1u || kind == 4u ? bytes / 4u : bytes;
	if ( data != 0 && count != 0u )
		RoofFillKernel<<<1024,256>>>(data,count,kind,seed,amplitude);
	ROOF_CUDA(cudaGetLastError());
	return(data);
}

static uint32_t RoofKindIsNorm(uint32_t kind)
{
	return(kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM || kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A_NORM || kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A_NORM || kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_POST_ATTN_NORM || kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT || kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT_NORM || kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM ? 1u : 0u);
}

static void *RoofTensor(uint32_t kind,uint32_t layer,uint32_t seed,void **scale,uint64_t *payload_bytes,uint64_t *scale_bytes)
{
	SparkGlm5NextStagePackTensorShape shape;
	uint32_t fill;
	float amplitude;
	*scale = 0;
	*payload_bytes = *scale_bytes = 0u;
	if ( SparkGlm5NextStagePackExpectedShape(kind,layer,GLM5_NEXT_EXPERT_WEIGHT_CODEC,ROOF_TP,&shape) != 0 )
		return(0);
	*payload_bytes = SparkGlm5NextStagePackExpectedPayloadBytes(&shape);
	*scale_bytes = SparkGlm5NextStagePackExpectedScaleBytes(&shape);
	fill = shape.payload_type == SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_BF16 ? 0u : shape.payload_type == SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_PACKED_WEIGHT ? 2u : 1u;
	amplitude = 1.0f / sqrtf((float)shape.columns);
	if ( *scale_bytes != 0u )
		*scale = RoofFilled(*scale_bytes,1u,seed ^ 0x5CA1Eu,0.02f);
	if ( RoofKindIsNorm(kind) != 0u )
		return(RoofFilled(*payload_bytes,fill + 3u,seed,1.0f));
	return(RoofFilled(*payload_bytes,fill,seed,amplitude));
}

static void RoofBind(SparkGlm5NextLayerWeights *weights,uint32_t kind,const void *payload,const void *scale)
{
	switch ( kind )
	{
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM: weights->attn_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A: weights->q_a_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A_NORM: weights->q_a_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_B: weights->q_b_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A: weights->kv_a_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A_NORM: weights->kv_a_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: weights->kv_b_key_transposed_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_VALUE: weights->kv_b_value_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_OUTPUT: weights->attn_output_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_POST_ATTN_NORM: weights->post_attn_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_Q: weights->index_q_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_K: weights->index_k_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_HEAD: weights->index_head_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT: weights->index_norm_weight_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_BIAS: weights->index_norm_bias_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_GATE_UP: weights->dense_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_DOWN: weights->dense_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER: weights->router_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER_CORRECTION: weights->router_correction_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE: weights->expert_up_gate_payload = payload; weights->expert_up_gate_scale = scale; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN: weights->expert_down_payload = payload; weights->expert_down_scale = scale; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_GATE_UP: weights->shared_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_DOWN: weights->shared_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_QKV_BETA: weights->kda_qkv_beta_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_GATE_DOWN: weights->kda_decay_gate_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_UP: weights->kda_decay_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_GATE_UP: weights->kda_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_Q_CONV: weights->kda_q_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_K_CONV: weights->kda_k_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_V_CONV: weights->kda_v_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_BIAS: weights->kda_decay_bias_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE: weights->kda_head_log_scale_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT_NORM: weights->kda_out_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT: weights->kda_out_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_FN: weights->hc_attn_fn_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_BASE: weights->hc_attn_base_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_SCALE: weights->hc_attn_scale_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_FN: weights->hc_ffn_fn_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_BASE: weights->hc_ffn_base_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_SCALE: weights->hc_ffn_scale_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_APE: weights->index_compress_ape_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_GATE: weights->index_compress_gate_bf16 = payload; break;
	default: break;
	}
}

static uint32_t RoofLayerKind(uint32_t layer)
{
	if ( layer < SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER )
		return(0u);
	return(SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer) ? 1u : 2u);
}

static const uint32_t RoofKindLayer[ROOF_KINDS] = {0u,4u,3u};

static void RoofBuildTemplate(SparkGlm5NextLayerWeights *weights,RoofBytes *bytes,uint32_t kind,uint32_t copy)
{
	uint64_t payload_bytes,scale_bytes;
	uint32_t tensor;
	void *payload,*scale;
	memset(weights,0,sizeof(*weights));
	memset(bytes,0,sizeof(*bytes));
	for (tensor=SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM; tensor<=SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_GATE; tensor++)
	{
		payload = RoofTensor(tensor,RoofKindLayer[kind],(copy * 131u + kind) * 64u + tensor,&scale,&payload_bytes,&scale_bytes);
		if ( payload == 0 )
			continue;
		RoofBind(weights,tensor,payload,scale);
		if ( tensor == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE || tensor == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN )
			bytes->expert += (payload_bytes + scale_bytes) / ROOF_EXPERTS;
		else
			bytes->fixed += payload_bytes + scale_bytes;
	}
}

static void RoofBuildModel(RoofModel *model,const RoofConfig *config)
{
	SparkGlm5NextStagePackTensorShape shape;
	uint32_t copy,kind,layer;
	memset(model,0,sizeof(*model));
	for (copy=0u; copy<config->copies; copy++)
		for (kind=0u; kind<ROOF_KINDS; kind++)
			RoofBuildTemplate(&model->templates[copy][kind],&model->bytes[kind],kind,copy);
	for (layer=0u; layer<ROOF_LAYERS; layer++)
		model->layers[layer] = model->templates[layer % config->copies][RoofLayerKind(layer)];
	if ( SparkGlm5NextStagePackExpectedShape(SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD,SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER,GLM5_NEXT_EXPERT_WEIGHT_CODEC,ROOF_TP,&shape) != 0 )
		exit(1);
	model->head_bytes = SparkGlm5NextStagePackExpectedPayloadBytes(&shape);
	model->lm_head = RoofFilled(model->head_bytes,0u,0x4EADu,1.0f / 64.0f);
	model->final_norm = RoofFilled((uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * 2u,3u,0u,1.0f);
}

static void *RoofRows(uint32_t rows,uint64_t width,uint64_t element)
{
	return(RoofAllocate((uint64_t)rows * width * element));
}

static void RoofAllocateSlotRows(SparkGlm5NextExecutionSlot *slot,uint32_t rows,uint32_t max_positions)
{
	uint64_t packed;
	packed = (uint64_t)rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K;
	slot->token_ids = (uint32_t *)RoofRows(rows,1u,4u);
	slot->resident_slots = (uint32_t *)RoofRows(rows,1u,4u);
	slot->positions = (uint32_t *)RoofRows(rows,1u,4u);
	slot->run_begin = (uint32_t *)RoofRows(rows + 1u,1u,4u);
	slot->run_state_index = (uint32_t *)RoofRows(rows,1u,4u);
	slot->run_row_indices = (uint32_t *)RoofRows(rows,1u,4u);
	slot->context_lengths = (uint32_t *)RoofRows(rows,1u,4u);
	slot->dense_row_offset = (uint32_t *)RoofRows(2u,1u,4u);
	slot->dense_tile_prefix = (uint32_t *)RoofRows(2u,1u,4u);
	slot->kv_access_error = RoofRows(6u,1u,4u);
	slot->router_logits_f32 = (float *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,4u);
	slot->selection_scores_f32 = (float *)RoofRows(rows,max_positions / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL,4u);
	slot->attention_split_partials_f32 = (float *)RoofAllocate(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT / ROOF_TP) * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_FLOATS * sizeof(float));
	slot->selected_positions = (uint32_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH,4u);
	slot->route_expert = (uint32_t *)RoofAllocate(packed * sizeof(uint32_t));
	slot->route_weight = (float *)RoofAllocate(packed * sizeof(float));
	slot->route_source_token = (uint32_t *)RoofAllocate(packed * sizeof(uint32_t));
	slot->route_packed_row = (uint32_t *)RoofAllocate(packed * sizeof(uint32_t));
	slot->group_row_offset = (uint32_t *)RoofRows(SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,4u);
	slot->group_tile_prefix_w1 = (uint32_t *)RoofRows(SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,4u);
	slot->group_tile_prefix_w2 = (uint32_t *)RoofRows(SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,4u);
	slot->head_candidate_score = (float *)RoofRows(rows,SparkCeilDivU64(SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,GLM5_NEXT_HEAD_TILE),4u);
	slot->head_candidate_token = (uint32_t *)RoofRows(rows,SparkCeilDivU64(SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,GLM5_NEXT_HEAD_TILE),4u);
	slot->output_token = (uint32_t *)RoofRows(rows,1u,4u);
	slot->output_score = (float *)RoofRows(rows,1u,4u);
	slot->head_maxloc_u64 = (uint64_t *)RoofRows(rows,1u,8u);
}

static void RoofAllocateSlotHidden(SparkGlm5NextExecutionSlot *slot,uint32_t rows)
{
	slot->hidden_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
	slot->residual_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
	slot->normed_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
	slot->hc_collapsed_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
	slot->hc_snapshot_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
	slot->hc_mean_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
	slot->hc_mixes_f32 = (float *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION,4u);
	slot->hc_pre_f32 = (float *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HC_MULT,4u);
	slot->hc_post_f32 = (float *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HC_MULT,4u);
	slot->hc_comb_f32 = (float *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HC_MULT,4u);
	slot->q_compressed_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_QUERY_A_DIMENSION,2u);
	slot->q_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_QUERY_B_DIMENSION,2u);
	slot->query_latent_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION,2u);
	slot->index_query_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_QUERY_DIMENSION,2u);
	slot->index_key_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_DIMENSION,2u);
	slot->index_head_weight_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_COUNT,2u);
	slot->index_gate_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_DIMENSION,2u);
	slot->index_packed_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION,2u);
	slot->selected_pools = (uint32_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL,4u);
	slot->fused_qkvb_bf16 = (uint16_t *)RoofRows(rows,3u * SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,2u);
	slot->fused_decay_gate_bf16 = (uint16_t *)RoofRows(rows,2u * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,2u);
	slot->kda_decay_latent_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,2u);
	slot->kda_gate_latent_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,2u);
	slot->kda_beta_logit = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,2u);
	slot->kda_gate_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,2u);
	slot->kda_decay_logit_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,2u);
	slot->kda_output_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
	slot->kda_retention = (float *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,4u);
	slot->kda_write_gate = (float *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,4u);
	slot->kv_slot_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_CACHE_TOKEN_ELEMENTS,2u);
	slot->attention_latent_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION,2u);
	slot->attention_value_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_VALUE_HEAD_DIMENSION,2u);
	slot->attention_out_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
	slot->gate_up_bf16 = (uint16_t *)RoofRows(rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K,SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_GATE_UP_DIMENSION,2u);
	slot->intermediate_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_MOE_TOP_K * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,2u);
	slot->expert_out_bf16 = (uint16_t *)RoofRows(rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
	slot->shared_out_bf16 = (uint16_t *)RoofRows(rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,2u);
}

static void RoofAllocateCaches(RoofState *state,const RoofConfig *config)
{
	SparkGlm5NextCudaWave *wave;
	uint64_t pages,kda_state_stride,window_stride;
	uint32_t layer,kv_count,kda_count,page;
	wave = &state->wave;
	kv_count = kda_count = 0u;
	for (layer=0u; layer<ROOF_LAYERS; layer++)
	{
		state->index_ordinals[layer] = SparkGlm5NextStagePackLayerIsDsa(layer) != 0u ? kv_count++ : UINT32_MAX;
		state->kda_ordinals[layer] = SparkGlm5NextStagePackLayerIsDsa(layer) == 0u ? kda_count++ : UINT32_MAX;
	}
	state->max_positions = ((config->context + 63u) / 64u) * 64u;
	state->pages_per_sequence = state->max_positions / 64u;
	pages = (uint64_t)config->max_batch * state->pages_per_sequence;
	ROOF_CUDA(cudaMallocHost((void **)&state->page_table,pages * sizeof(uint32_t)));
	for (page=0u; page<pages; page++)
		state->page_table[page] = page;
	wave->page_table = (const uint32_t *)RoofAllocate(pages * sizeof(uint32_t));
	ROOF_CUDA(cudaMemcpy((void *)wave->page_table,state->page_table,pages * sizeof(uint32_t),cudaMemcpyHostToDevice));
	wave->physical_page_count = (uint32_t)pages;
	wave->kv_layer_stride_bytes = pages * 64u * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES;
	wave->index_layer_stride_bytes = pages * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
	wave->kv_cache = (uint8_t *)RoofFilled(wave->kv_layer_stride_bytes * kv_count,0u,0xCAC4Eu,0.05f);
	wave->index_cache = (uint8_t *)RoofFilled(wave->index_layer_stride_bytes * kv_count,0u,0x14DE7u,0.05f);
	kda_state_stride = (uint64_t)config->max_batch * (SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER / ROOF_TP);
	window_stride = (uint64_t)config->max_batch * (SPARK_GLM5_NEXT_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER / (3u * ROOF_TP));
	wave->kda_state_pools = (uint8_t *)RoofAllocate(kda_state_stride * kda_count);
	wave->kda_state_layer_stride_bytes = kda_state_stride;
	wave->kda_q_window_pool = (uint8_t *)RoofAllocate(window_stride * kda_count * 3u);
	wave->kda_k_window_pool = wave->kda_q_window_pool + window_stride * kda_count;
	wave->kda_v_window_pool = wave->kda_k_window_pool + window_stride * kda_count;
	wave->kda_window_layer_stride_bytes = window_stride;
	wave->kda_layer_count = kda_count;
	wave->index_ordinal_by_local_layer = state->index_ordinals;
	wave->kda_ordinal_by_local_layer = state->kda_ordinals;
}

static void RoofAllocateHost(RoofState *state,const RoofConfig *config)
{
	SparkGlm5NextExecutionSlot *slot;
	uint32_t row;
	slot = &state->slot;
	ROOF_CUDA(cudaMallocHost((void **)&state->host_slots,config->max_batch * sizeof(uint32_t)));
	ROOF_CUDA(cudaMallocHost((void **)&state->host_positions,config->max_batch * sizeof(uint32_t)));
	ROOF_CUDA(cudaMallocHost((void **)&state->host_run_begin,(config->max_batch + 1u) * sizeof(uint32_t)));
	ROOF_CUDA(cudaMallocHost((void **)&state->host_run_rows,config->max_batch * sizeof(uint32_t)));
	ROOF_CUDA(cudaMallocHost((void **)&state->host_run_state,config->max_batch * sizeof(uint32_t)));
	ROOF_CUDA(cudaMallocHost((void **)&slot->host_group_row_offset,(uint64_t)ROOF_LAYERS * (ROOF_EXPERTS + 1u) * sizeof(uint32_t)));
	memset(slot->host_group_row_offset,0,(uint64_t)ROOF_LAYERS * (ROOF_EXPERTS + 1u) * sizeof(uint32_t));
	for (row=0u; row<config->max_batch; row++)
	{
		state->host_slots[row] = row;
		state->host_positions[row] = config->context - 1u;
		state->host_run_begin[row] = row;
		state->host_run_rows[row] = row;
		state->host_run_state[row] = row;
	}
	state->host_run_begin[config->max_batch] = config->max_batch;
	ROOF_CUDA(cudaEventCreateWithFlags((cudaEvent_t *)&slot->route_ready_event,cudaEventDisableTiming));
}

static void RoofBuildState(RoofState *state,const RoofModel *model,const RoofConfig *config,cudaStream_t stream)
{
	SparkGlm5NextCudaWave *wave;
	uint32_t multiprocessors;
	memset(state,0,sizeof(*state));
	wave = &state->wave;
	state->slot.stream = stream;
	RoofAllocateCaches(state,config);
	RoofAllocateSlotRows(&state->slot,config->max_batch,state->max_positions);
	if ( config->index_cp > 1u )
	{
		state->slot.index_local_scores_f32 = (float *)RoofRows(config->max_batch,SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS,sizeof(float));
		state->slot.index_gathered_scores_f32 = (float *)RoofRows(config->max_batch * config->index_cp,SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS,sizeof(float));
	}
	RoofAllocateSlotHidden(&state->slot,config->max_batch);
	RoofAllocateHost(state,config);
	state->hidden_input = RoofFilled((uint64_t)config->max_batch * SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * 2u,0u,0x41DDu,1.0f);
	ROOF_LAUNCH(SparkGlm5NextConfigureCudaModule(&multiprocessors));
	wave->layer_count = ROOF_LAYERS;
	wave->tp_degree = ROOF_TP;
	wave->index_cp_degree = config->index_cp > 1u ? config->index_cp : 1u;
	wave->commit = 1u;
	wave->resident_sequence_capacity = config->max_batch;
	wave->max_sequence_positions = state->max_positions;
	wave->execution_row_capacity = config->max_batch;
	wave->pages_per_sequence = state->pages_per_sequence;
	wave->owns_final_head = 1u;
	wave->hidden_input_bf16 = state->hidden_input;
	wave->final_norm_bf16 = model->final_norm;
	wave->lm_head_bf16 = model->lm_head;
	wave->layers = model->layers;
	wave->slot = &state->slot;
	wave->multiprocessor_count = multiprocessors;
	wave->decode_split_context_threshold = 64u;
	wave->attention_split_partials_f32 = state->slot.attention_split_partials_f32;
	wave->attention_split_partial_blocks = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(config->max_batch,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT / ROOF_TP);
	wave->host_resident_slots = state->host_slots;
	wave->host_positions = state->host_positions;
	wave->host_sequence_row_begin = state->host_run_begin;
	wave->host_sequence_row_indices = state->host_run_rows;
	wave->host_run_state_index = state->host_run_state;
	wave->sequence_row_begin = state->slot.run_begin;
	wave->sequence_row_indices = state->slot.run_row_indices;
	wave->run_state_index = state->slot.run_state_index;
	wave->kda_state_index = state->slot.run_state_index;
}

static void RoofSetRows(RoofState *state,uint32_t rows,uint32_t context)
{
	state->wave.row_count = rows;
	state->wave.run_count = rows;
	state->wave.maximum_context = context;
	state->host_run_begin[rows] = rows;
}

static void RoofAttention(const RoofState *state,uint32_t layer)
{
	const SparkGlm5NextCudaWave *wave;
	uint64_t floats;
	uint32_t sequences,peer;
	wave = &state->wave;
	sequences = SparkGlm5NextLayerIndexGatherSequences(wave,layer);
	if ( sequences == 0u )
	{
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerAttention(wave,layer));
		return;
	}
	floats = (uint64_t)sequences * SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS;
	ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerAttentionScore(wave,layer));
	for (peer=0u; peer<wave->index_cp_degree; peer++)
		ROOF_CUDA(cudaMemcpyAsync(state->slot.index_gathered_scores_f32 + peer * floats,state->slot.index_local_scores_f32,floats * sizeof(float),cudaMemcpyDeviceToDevice,(cudaStream_t)state->slot.stream));
	ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerAttentionSelect(wave,layer));
}

static void RoofStep(const RoofState *state)
{
	const SparkGlm5NextCudaWave *wave;
	uint32_t layer;
	wave = &state->wave;
	ROOF_LAUNCH(SparkGlm5NextLaunchCudaWaveBegin(wave));
	for (layer=0u; layer<ROOF_LAYERS; layer++)
	{
		RoofAttention(state,layer);
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerAttentionPost(wave,layer));
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerMlpRoute(wave,layer));
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerMlpExperts(wave,layer));
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerMlpPost(wave,layer));
	}
	ROOF_LAUNCH(SparkGlm5NextLaunchCudaWaveHead(wave));
}

static uint32_t RoofPhaseOf(uint32_t layer,uint32_t launch)
{
	if ( launch == 0u )
		return(SparkGlm5NextStagePackLayerIsDsa(layer) != 0u ? 1u : 0u);
	return(launch + 1u);
}

static void RoofRecord(cudaEvent_t *events,uint32_t *cursor,cudaStream_t stream)
{
	ROOF_CUDA(cudaEventRecord(events[*cursor],stream));
	(*cursor)++;
}

static void RoofProfileStep(const RoofState *state,cudaEvent_t *events,cudaStream_t stream)
{
	const SparkGlm5NextCudaWave *wave;
	uint32_t layer,cursor;
	wave = &state->wave;
	cursor = 0u;
	RoofRecord(events,&cursor,stream);
	ROOF_LAUNCH(SparkGlm5NextLaunchCudaWaveBegin(wave));
	for (layer=0u; layer<ROOF_LAYERS; layer++)
	{
		RoofRecord(events,&cursor,stream);
		RoofAttention(state,layer);
		RoofRecord(events,&cursor,stream);
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerAttentionPost(wave,layer));
		RoofRecord(events,&cursor,stream);
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerMlpRoute(wave,layer));
		RoofRecord(events,&cursor,stream);
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerMlpExperts(wave,layer));
		RoofRecord(events,&cursor,stream);
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerMlpPost(wave,layer));
	}
	RoofRecord(events,&cursor,stream);
	ROOF_LAUNCH(SparkGlm5NextLaunchCudaWaveHead(wave));
	ROOF_CUDA(cudaEventRecord(events[ROOF_PHASE_EVENTS],stream));
}

static void RoofProfile(const RoofState *state,uint32_t rows,cudaStream_t stream)
{
	static const char *names[ROOF_PHASES] = {"attention_kda","attention_dsa","attention_post","route","experts","mlp_post","begin_head"};
	cudaEvent_t events[ROOF_PHASE_EVENTS + 1u];
	double phases[ROOF_PHASES];
	uint32_t index,layer,launch;
	float elapsed;
	for (index=0u; index<=ROOF_PHASE_EVENTS; index++)
		ROOF_CUDA(cudaEventCreate(&events[index]));
	RoofProfileStep(state,events,stream);
	ROOF_CUDA(cudaEventSynchronize(events[ROOF_PHASE_EVENTS]));
	memset(phases,0,sizeof(phases));
	ROOF_CUDA(cudaEventElapsedTime(&elapsed,events[0],events[1]));
	phases[ROOF_PHASES - 1u] += elapsed;
	for (layer=0u; layer<ROOF_LAYERS; layer++)
		for (launch=0u; launch<5u; launch++)
		{
			index = 1u + layer * 5u + launch;
			ROOF_CUDA(cudaEventElapsedTime(&elapsed,events[index],events[index + 1u]));
			phases[RoofPhaseOf(layer,launch)] += elapsed;
		}
	ROOF_CUDA(cudaEventElapsedTime(&elapsed,events[ROOF_PHASE_EVENTS - 1u],events[ROOF_PHASE_EVENTS]));
	phases[ROOF_PHASES - 1u] += elapsed;
	printf("ROOFLINE-PHASES rows=%u",rows);
	for (index=0u; index<ROOF_PHASES; index++)
		printf(" %s_ms=%.2f",names[index],phases[index]);
	printf("\n");
	for (index=0u; index<=ROOF_PHASE_EVENTS; index++)
		ROOF_CUDA(cudaEventDestroy(events[index]));
}

static double RoofDistinctExperts(const RoofState *state)
{
	const uint32_t *offsets;
	uint32_t layer,expert,count,layers;
	count = layers = 0u;
	for (layer=SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER; layer<ROOF_LAYERS; layer++)
	{
		offsets = state->slot.host_group_row_offset + (uint64_t)layer * (ROOF_EXPERTS + 1u);
		for (expert=0u; expert<ROOF_EXPERTS; expert++)
			count += offsets[expert + 1u] > offsets[expert] ? 1u : 0u;
		layers++;
	}
	return(layers != 0u ? (double)count / (double)layers : 0.0);
}

static void RoofExpertsLayer(const RoofState *state,uint32_t layer,cudaEvent_t *events,cudaStream_t stream)
{
	const SparkGlm5NextCudaWave *wave;
	Glm5NextLayerBuffers buffers;
	uint32_t rows,packed;
	wave = &state->wave;
	rows = wave->row_count;
	packed = rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K;
	ROOF_CUDA(cudaEventRecord(events[0],stream));
	if ( wave->first_layer_index + layer >= SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER )
	{
		SparkGlm5NextBindLayer(wave,layer,&buffers);
		ROOF_LAUNCH((Glm5NextLayerMoeUp<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(&buffers,rows,packed,wave->multiprocessor_count,stream)));
		ROOF_CUDA(cudaEventRecord(events[1],stream));
		ROOF_LAUNCH((Glm5NextLayerMoeDown<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(&buffers,rows,packed,wave->multiprocessor_count,stream)));
		ROOF_CUDA(cudaEventRecord(events[2],stream));
		ROOF_LAUNCH(Glm5NextLayerMoeCombine(&buffers,rows,wave->multiprocessor_count,stream));
	}
	else
	{
		ROOF_CUDA(cudaEventRecord(events[1],stream));
		ROOF_CUDA(cudaEventRecord(events[2],stream));
	}
	ROOF_CUDA(cudaEventRecord(events[3],stream));
}

static void RoofExpertsStep(const RoofState *state,cudaEvent_t (*events)[4],cudaStream_t stream)
{
	const SparkGlm5NextCudaWave *wave;
	uint32_t layer;
	wave = &state->wave;
	ROOF_LAUNCH(SparkGlm5NextLaunchCudaWaveBegin(wave));
	for (layer=0u; layer<ROOF_LAYERS; layer++)
	{
		RoofAttention(state,layer);
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerAttentionPost(wave,layer));
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerMlpRoute(wave,layer));
		RoofExpertsLayer(state,layer,events[layer],stream);
		ROOF_LAUNCH(SparkGlm5NextLaunchCudaLayerMlpPost(wave,layer));
	}
	ROOF_LAUNCH(SparkGlm5NextLaunchCudaWaveHead(wave));
}

static void RoofExpertsProfile(const RoofState *state,uint32_t rows,cudaStream_t stream)
{
	const double w1 = (double)ROOF_EXPERT_W1_ROWS * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION + (double)(ROOF_EXPERT_W1_ROWS / 128u) * (SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION / 128u) * 4.0;
	const double w2 = (double)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * ROOF_EXPERT_INTERMEDIATE + (double)(SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION / 128u) * (ROOF_EXPERT_INTERMEDIATE / 128u) * 4.0;
	cudaEvent_t events[ROOF_LAYERS][4];
	double part[3],distinct,layers;
	uint32_t layer,index;
	float elapsed;
	for (layer=0u; layer<ROOF_LAYERS; layer++)
		for (index=0u; index<4u; index++)
			ROOF_CUDA(cudaEventCreate(&events[layer][index]));
	RoofExpertsStep(state,events,stream);
	ROOF_CUDA(cudaStreamSynchronize(stream));
	part[0] = part[1] = part[2] = 0.0;
	for (layer=0u; layer<ROOF_LAYERS; layer++)
		for (index=0u; index<3u; index++)
		{
			ROOF_CUDA(cudaEventElapsedTime(&elapsed,events[layer][index],events[layer][index + 1u]));
			part[index] += elapsed;
		}
	distinct = RoofDistinctExperts(state);
	layers = (double)(ROOF_LAYERS - SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER);
	printf("ROOFLINE-EXPERTS rows=%u up_ms=%.2f down_ms=%.2f combine_ms=%.2f up_gbps=%.0f down_gbps=%.0f\n",rows,part[0],part[1],part[2],distinct * w1 * layers / (part[0] * 1e6),distinct * w2 * layers / (part[1] * 1e6));
	for (layer=0u; layer<ROOF_LAYERS; layer++)
		for (index=0u; index<4u; index++)
			ROOF_CUDA(cudaEventDestroy(events[layer][index]));
}

static double RoofStepBytes(const RoofModel *model,uint32_t rows,uint32_t context,uint32_t index_cp,double distinct)
{
	uint32_t layer,kind;
	double bytes,attended,indexed;
	bytes = (double)model->head_bytes;
	attended = (double)(context < SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K ? context : SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K);
	indexed = context > SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K ? (double)context * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2.0 / (double)index_cp : 0.0;
	for (layer=0u; layer<ROOF_LAYERS; layer++)
	{
		kind = RoofLayerKind(layer);
		bytes += (double)model->bytes[kind].fixed + (kind != 0u ? distinct * (double)model->bytes[kind].expert : 0.0);
		if ( SparkGlm5NextStagePackLayerIsDsa(layer) != 0u )
			bytes += (double)rows * (attended * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES + indexed);
	}
	return(bytes);
}

static uint32_t RoofHiddenFinite(const RoofState *state,uint32_t rows)
{
	uint16_t values[64];
	uint32_t index,finite;
	ROOF_CUDA(cudaMemcpy(values,state->slot.hidden_bf16 + (uint64_t)(rows - 1u) * SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,sizeof(values),cudaMemcpyDeviceToHost));
	finite = 1u;
	for (index=0u; index<64u; index++)
		finite &= ((values[index] & 0x7F80u) != 0x7F80u) ? 1u : 0u;
	return(finite);
}

static void RoofReport(const RoofModel *model,const RoofState *state,const RoofConfig *config,uint32_t rows,double step_ms)
{
	double distinct,bytes,payload,direct_ms,rsag_ms,collectives;
	distinct = RoofDistinctExperts(state);
	bytes = RoofStepBytes(model,rows,config->context,state->wave.index_cp_degree,distinct);
	collectives = 2.0 * ROOF_LAYERS + 2.0;
	payload = (double)rows * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * 2.0;
	direct_ms = collectives * (15.0 * payload / (config->nic_gbps * 1e9) + config->round_us * 1e-6) * 1e3;
	rsag_ms = collectives * (1.875 * payload / (config->nic_gbps * 1e9) + 2.0 * config->round_us * 1e-6) * 1e3;
	printf("ROOFLINE rows=%u context=%u index_cp=%u step_ms=%.2f compute_tok_s=%.0f distinct_experts=%.1f uniform_expect=%.1f step_gb=%.2f achieved_gbps=%.0f memory_bound_ms=%.2f hidden_finite=%u\n",rows,config->context,state->wave.index_cp_degree,step_ms,rows * 1000.0 / step_ms,distinct,ROOF_EXPERTS * (1.0 - pow(1.0 - (double)SPARK_GLM5_NEXT_MODEL_MOE_TOP_K / ROOF_EXPERTS,(double)rows)),bytes / 1e9,bytes / (step_ms * 1e6),bytes / (config->memory_gbps * 1e6),RoofHiddenFinite(state,rows));
	printf("ROOFLINE-FLEET rows=%u collectives=%.0f direct_ms=%.1f rsag_ms=%.1f tok_s_direct=%.0f tok_s_rsag=%.0f tok_s_rsag_overlapped=%.0f\n",rows,collectives,direct_ms,rsag_ms,rows * 1000.0 / (step_ms + direct_ms),rows * 1000.0 / (step_ms + rsag_ms),rows * 1000.0 / (step_ms > rsag_ms ? step_ms : rsag_ms));
}

static double RoofMeasure(RoofState *state,const RoofConfig *config,uint32_t rows,cudaStream_t stream)
{
	cudaEvent_t start,stop;
	uint32_t iteration;
	float elapsed;
	RoofSetRows(state,rows,config->context);
	RoofStep(state);
	RoofStep(state);
	ROOF_CUDA(cudaStreamSynchronize(stream));
	ROOF_CUDA(cudaEventCreate(&start));
	ROOF_CUDA(cudaEventCreate(&stop));
	ROOF_CUDA(cudaEventRecord(start,stream));
	for (iteration=0u; iteration<config->iterations; iteration++)
		RoofStep(state);
	ROOF_CUDA(cudaEventRecord(stop,stream));
	ROOF_CUDA(cudaEventSynchronize(stop));
	ROOF_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
	ROOF_CUDA(cudaEventDestroy(start));
	ROOF_CUDA(cudaEventDestroy(stop));
	return((double)elapsed / config->iterations);
}

static uint32_t RoofParseList(const char *text,uint32_t *values,uint32_t capacity)
{
	uint32_t count;
	char *end;
	count = 0u;
	while ( *text != '\0' && count < capacity )
	{
		values[count++] = (uint32_t)strtoul(text,&end,10);
		text = *end == ',' ? end + 1 : end;
		if ( end == text && *end != '\0' )
			return(0u);
	}
	return(count);
}

static int RoofParse(int argc,char **argv,RoofConfig *config)
{
	int index;
	uint32_t batch;
	memset(config,0,sizeof(*config));
	config->batch_count = RoofParseList("1,8,32,64,128,256",config->batches,ROOF_MAX_BATCHES);
	config->context = 1024u;
	config->iterations = 5u;
	config->copies = 2u;
	config->round_us = 100.0;
	config->nic_gbps = 11.5;
	config->memory_gbps = 273.0;
	for (index=1; index + 1 < argc; index+=2)
	{
		if ( strcmp(argv[index],"--batches") == 0 )
			config->batch_count = RoofParseList(argv[index + 1],config->batches,ROOF_MAX_BATCHES);
		else if ( strcmp(argv[index],"--context") == 0 )
			config->context = (uint32_t)strtoul(argv[index + 1],0,10);
		else if ( strcmp(argv[index],"--iterations") == 0 )
			config->iterations = (uint32_t)strtoul(argv[index + 1],0,10);
		else if ( strcmp(argv[index],"--copies") == 0 )
			config->copies = (uint32_t)strtoul(argv[index + 1],0,10);
		else if ( strcmp(argv[index],"--index-cp") == 0 )
			config->index_cp = (uint32_t)strtoul(argv[index + 1],0,10);
		else if ( strcmp(argv[index],"--round-us") == 0 )
			config->round_us = strtod(argv[index + 1],0);
		else if ( strcmp(argv[index],"--nic-gbps") == 0 )
			config->nic_gbps = strtod(argv[index + 1],0);
		else
			return(-1);
	}
	for (batch=0u; batch<config->batch_count; batch++)
		config->max_batch = config->batches[batch] > config->max_batch ? config->batches[batch] : config->max_batch;
	return(config->batch_count == 0u || config->context < 2u || config->iterations == 0u || config->copies == 0u || config->copies > ROOF_MAX_COPIES || config->index_cp > ROOF_TP || (config->index_cp > 1u && SparkGlm5NextIndexCpFits(((config->context + 63u) / 64u) * 64u,config->index_cp,config->max_batch) == 0u) || config->max_batch == 0u || config->max_batch > SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS * 2u ? -1 : 0);
}

int main(int argc,char **argv)
{
	static RoofModel model;
	static RoofState state;
	RoofConfig config;
	cudaDeviceProp properties;
	cudaStream_t stream;
	uint32_t batch;
	if ( RoofParse(argc,argv,&config) != 0 )
	{
		fprintf(stderr,"usage: glm5_next_batch_roofline [--batches 1,8,32,64,128,256] [--context 1024] [--iterations 5] [--copies 2] [--index-cp 16] [--round-us 100] [--nic-gbps 11.5]\n");
		return(2);
	}
	setvbuf(stdout,0,_IOLBF,0);
	ROOF_CUDA(cudaSetDevice(0));
	ROOF_CUDA(cudaGetDeviceProperties(&properties,0));
	printf("ROOFLINE-DEVICE sm=%d.%d multiprocessors=%d l2_bytes=%d tp=%u layers=%u experts=%u copies=%u\n",properties.major,properties.minor,properties.multiProcessorCount,properties.l2CacheSize,ROOF_TP,ROOF_LAYERS,ROOF_EXPERTS,config.copies);
	ROOF_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	RoofBuildModel(&model,&config);
	RoofBuildState(&state,&model,&config,stream);
	ROOF_CUDA(cudaDeviceSynchronize());
	printf("ROOFLINE-MODEL fixed_mb_kda_dense=%.1f fixed_mb_kda_moe=%.1f fixed_mb_dsa_moe=%.1f expert_mb=%.3f head_mb=%.1f\n",model.bytes[0].fixed / 1e6,model.bytes[1].fixed / 1e6,model.bytes[2].fixed / 1e6,model.bytes[1].expert / 1e6,model.head_bytes / 1e6);
	for (batch=0u; batch<config.batch_count; batch++)
	{
		RoofReport(&model,&state,&config,config.batches[batch],RoofMeasure(&state,&config,config.batches[batch],stream));
		RoofProfile(&state,config.batches[batch],stream);
		RoofExpertsProfile(&state,config.batches[batch],stream);
	}
	ROOF_CUDA(cudaStreamDestroy(stream));
	puts("ROOFLINE-DONE");
	return(0);
}
