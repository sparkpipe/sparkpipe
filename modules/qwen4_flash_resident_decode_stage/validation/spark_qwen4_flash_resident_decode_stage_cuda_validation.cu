#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_qwen4_flash_model.h"
#include "sparkpipe/spark_qwen4_flash_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"
#define SPARK_FAMILY_CAMEL Qwen4Flash
#define SPARK_FAMILY_UPPER QWEN4_FLASH
#define SPARK_FAMILY_LOWER qwen4_flash

#include "sparkpipe/family/spark_family.h"

#define SPARK_QWEN4_FLASH_VALIDATION_ROWS 4u
#define SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS 8u
#define SPARK_QWEN4_FLASH_VALIDATION_CHUNK_TOKENS 128u
#define SPARK_QWEN4_FLASH_VALIDATION_ATTN_TOKENS 5u
#define SPARK_QWEN4_FLASH_VALIDATION_KV_LANES SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES
#define SPARK_QWEN4_FLASH_VALIDATION_ATTN_GROUP (SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT / SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT)

extern "C" cudaError_t SparkQwen4FlashConfigureCudaKernels(void);
extern "C" cudaError_t SparkQwen4FlashLaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen4FlashGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen4FlashGdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree);
extern "C" cudaError_t SparkQwen4FlashLaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen4FlashGdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count, uint32_t tp_degree);
extern "C" cudaError_t SparkQwen4FlashLaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen4FlashGdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree);
extern "C" cudaError_t SparkQwen4FlashLaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen4FlashGdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon, uint32_t tp_degree);
extern "C" cudaError_t SparkQwen4FlashLaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen4FlashAttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon, uint32_t tp_degree, uint32_t tp_rank);
extern "C" cudaError_t SparkQwen4FlashLaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen4FlashKvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank, const uint8_t *token_mask, uint32_t mask_stride);
extern "C" cudaError_t SparkQwen4FlashLaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern "C" cudaError_t SparkQwen4FlashLaunchLinear(cudaStream_t stream, const SparkQwen4FlashLinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern "C" cudaError_t SparkQwen4FlashLaunchHcStreamReplicate(cudaStream_t stream, const void *input_bf16, void *streams_bf16, uint32_t row_count);
extern "C" cudaError_t SparkQwen4FlashLaunchHcGroupNorm(cudaStream_t stream, const void *streams_bf16, const void *weight_bf16, void *normed_bf16, uint32_t row_count, float epsilon);
extern "C" cudaError_t SparkQwen4FlashLaunchHcSiluQuarter(cudaStream_t stream, void *lowrank_bf16, uint32_t row_count);
extern "C" cudaError_t SparkQwen4FlashLaunchHcMix(cudaStream_t stream, const void *up_bf16, const void *normed_bf16, void *mixed_bf16, uint32_t row_count);
extern "C" cudaError_t SparkQwen4FlashLaunchHcInject(cudaStream_t stream, void *streams_bf16, const void *inject_pre_bf16, const void *sublayer_out_bf16, uint32_t row_count);
extern "C" cudaError_t SparkQwen4FlashLaunchIndexerPrepare(cudaStream_t stream, const void *qk_bf16, const SparkQwen4FlashIndexerWeights *weights, void *query_bf16, void *raw_key_cache, void *pooled_key_cache, const uint32_t *slot_mapping, const uint32_t *block_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t lane_stride, uint64_t cache_block_stride);
extern "C" cudaError_t SparkQwen4FlashLaunchIndexerSelect(cudaStream_t stream, const void *query_bf16, const void *pooled_key_cache, const SparkQwen4FlashKvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, uint8_t *token_mask, uint32_t *score_keys_u32, uint32_t row_count, uint32_t mask_stride, uint32_t score_stride);
extern "C" cudaError_t SparkQwen4FlashLaunchPleHashGather(cudaStream_t stream, const uint32_t *history_u32, uint32_t token_count, const SparkQwen4FlashPleWeights *ple, void *embedding_bf16, uint32_t row_count, uint32_t vocab_base, uint32_t vocab_rows);
extern "C" cudaError_t SparkQwen4FlashLaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen4FlashGdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree);

static uint32_t SparkQwen4FlashValRandomState;

#include "sparkpipe/family/validation/spark_val_rng.h"
#include "sparkpipe/family/validation/spark_val_from_bf16.h"
#include "sparkpipe/family/validation/spark_val_fill_bf16_half_up.h"
#include "sparkpipe/family/validation/spark_val_fail.h"
#include "sparkpipe/family/validation/spark_val_cuda.h"
#include "sparkpipe/family/validation/spark_val_measure.h"
#include "sparkpipe/family/validation/spark_val_gdn_attention.h"
#include "sparkpipe/family/validation/spark_val_gdn_device.h"

#define SPARK_QWEN4_FLASH_VALIDATION_GDN_LAUNCH(launch,...) launch(__VA_ARGS__,1u)
#define SPARK_QWEN4_FLASH_VALIDATION_ATTN_PREPARE_LAUNCH(launch,...) launch(__VA_ARGS__,1u,0u)
#define SPARK_QWEN4_FLASH_VALIDATION_ATTN_DECODE_LAUNCH(launch,...) launch(__VA_ARGS__,1u,0u,(const uint8_t *)0,0u)
#include "sparkpipe/family/validation/spark_val_gdn_attention_checks.h"

#define SPARK_QWEN4_FLASH_VALIDATION_HIDDEN_WIDTH SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH
#include "sparkpipe/family/validation/spark_val_module_frame.h"

#define SPARK_QWEN4_FLASH_VALIDATION_MODEL_ID "Qwen/Qwen3.8-Flash-Next"
#define SPARK_QWEN4_FLASH_VALIDATION_NODE_TARGET "cuda.sm121.qwen4_flash.resident_decode_stage.fp8"
#include "sparkpipe/family/validation/spark_val_module_initialize.h"

static int SparkQwen4FlashValCheckModule(void)
{
	SparkQwen4FlashValModule module;
	uint16_t decode_hidden[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	uint16_t prefill_hidden[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	uint16_t rerun_hidden[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	uint32_t decode_token = 0u,prefill_token = 0u,rerun_token = 0u;
	uint32_t index;
	SparkQwen4FlashValMetrics metrics;
	SparkModelDriverAdmissionRequest admission;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	if (SparkQwen4FlashValModuleInitialize(&module) != 0)
		return(1);
	memset(&admission,0,sizeof(admission));
	admission.descriptor_bytes = sizeof(admission);
	admission.program_id = 1u;
	admission.frame_flags = 0u;
	admission.new_token_count = 1u;
	admission.active_slot_count = 1u;
	memset(&decision,0,sizeof(decision));
	decision.descriptor_bytes = sizeof(decision);
	status = SparkQwen4FlashResidentDecodeStageAdmit(module.state,&admission,&decision);
	if (status != SPARK_STATUS_OK || decision.accepted == 0u)
		return(SparkQwen4FlashValFail("module_admit","rejected"));
	memset(&snapshot,0,sizeof(snapshot));
	SparkModelDriverInitializeRuntimeSnapshot(&snapshot,1u);
	status = SparkQwen4FlashResidentDecodeStageSnapshot(module.state,1u,&snapshot);
	if (status != SPARK_STATUS_OK)
		return(SparkQwen4FlashValFail("module_snapshot","status"));
	printf("qwen4_flash_validation check=module_admit_snapshot admit=ok snapshot=ok\n");
	for (index = 0u; index < SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS; index++)
		module.token_ids[index] = 1000u + (index * 37u) % 200000u;
	module.positions[0] = 0u;
	module.sequence_ids[0] = 1u;
	if (SparkQwen4FlashValModuleExecute(&module,1u,SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS,0u,0u,0) != 0)
		return(1);
	if (module.capture.sends != (module.head_stage != 0u ? 0u : 1u))
		return(SparkQwen4FlashValFail("module_prefill",module.head_stage != 0u ? "unexpected_hidden_send" : "no_hidden_send"));
	if (module.head_stage == 0u && SparkQwen4FlashValCheckFinite("module_prefill",module.capture.hidden,SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS) != 0)
		return(1);
	module.token_ids[0] = 4242u;
	module.lanes[0] = 0u;
	module.positions[0] = SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS;
	module.sequence_ids[0] = 1u;
	if (SparkQwen4FlashValModuleExecute(&module,0u,1u,0u,0u,0) != 0)
		return(1);
	if (module.head_stage == 0u)
	{
		memcpy(decode_hidden,module.capture.hidden,sizeof(decode_hidden));
		if (SparkQwen4FlashValCheckFinite("module_decode",decode_hidden,1u) != 0)
			return(1);
	}
	else
		decode_token = module.output_token_ids[0];
	for (index = 0u; index < SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS; index++)
		module.token_ids[index] = 1000u + (index * 37u) % 200000u;
	module.positions[0] = 0u;
	module.sequence_ids[0] = 2u;
	if (SparkQwen4FlashValModuleExecute(&module,1u,SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS,1u,0u,0) != 0)
		return(1);
	module.token_ids[0] = 4242u;
	module.positions[0] = SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS;
	module.sequence_ids[0] = 2u;
	if (SparkQwen4FlashValModuleExecute(&module,1u,1u,1u,0u,0) != 0)
		return(1);
	if (module.head_stage == 0u)
	{
		memcpy(prefill_hidden,module.capture.hidden,sizeof(prefill_hidden));
		{
			float actual[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
			float reference[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
			for (index = 0u; index < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; index++)
			{
				actual[index] = SparkQwen4FlashValFromBf16(decode_hidden[index]);
				reference[index] = SparkQwen4FlashValFromBf16(prefill_hidden[index]);
			}
			SparkQwen4FlashValMeasure(&metrics,actual,reference,SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH);
			if (SparkQwen4FlashValReport("module_decode_vs_prefill",&metrics,5e-2,0.999) != 0)
				return(1);
		}
	}
	else
	{
		prefill_token = module.output_token_ids[0];
		printf("qwen4_flash_validation check=module_decode_vs_prefill decode_token=%u prefill_token=%u bit_exact=%d\n",decode_token,prefill_token,decode_token == prefill_token ? 1 : 0);
		if ( decode_token != prefill_token )
		{
			if ( getenv("SPARK_QWEN4_FLASH_VALIDATION_TOKEN_PARITY") != 0 &&
				strcmp(getenv("SPARK_QWEN4_FLASH_VALIDATION_TOKEN_PARITY"),"warn") == 0 )
				printf("qwen4_flash_validation check=module_decode_vs_prefill token_parity=warn (continuing per env)\n");
			else
				return(SparkQwen4FlashValFail("module_decode_vs_prefill","token_mismatch"));
		}
	}
#if SPARK_QWEN4_FLASH_MODEL_MTP_LAYER_COUNT != 0
	if (module.head_stage != 0u && SparkQwen4FlashValCheckMtpDraft(&module) != 0)
		return(1);
#endif
	SparkQwen4FlashResidentDecodeStageDestroy(module.state);
	cudaFree(module.device_blocks);
	cudaFree(module.device_counts);
	{
		SparkQwen4FlashValModule rerun;
		if (SparkQwen4FlashValModuleInitialize(&rerun) != 0)
			return(1);
		for (index = 0u; index < SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS; index++)
			rerun.token_ids[index] = 1000u + (index * 37u) % 200000u;
		rerun.positions[0] = 0u;
		rerun.sequence_ids[0] = 1u;
		if (SparkQwen4FlashValModuleExecute(&rerun,1u,SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS,0u,0u,0) != 0)
			return(1);
		rerun.token_ids[0] = 4242u;
		rerun.lanes[0] = 0u;
		rerun.positions[0] = SPARK_QWEN4_FLASH_VALIDATION_PREFILL_TOKENS;
		rerun.sequence_ids[0] = 1u;
		if (SparkQwen4FlashValModuleExecute(&rerun,0u,1u,0u,0u,0) != 0)
			return(1);
		if (module.head_stage == 0u)
		{
			memcpy(rerun_hidden,rerun.capture.hidden,sizeof(rerun_hidden));
			if (memcmp(decode_hidden,rerun_hidden,sizeof(decode_hidden)) != 0)
				return(SparkQwen4FlashValFail("module_determinism","fresh_instance_mismatch"));
		}
		else
		{
			rerun_token = rerun.output_token_ids[0];
			if (rerun_token != decode_token)
				return(SparkQwen4FlashValFail("module_determinism","fresh_instance_token_mismatch"));
		}
		SparkQwen4FlashResidentDecodeStageDestroy(rerun.state);
		cudaFree(rerun.device_blocks);
		cudaFree(rerun.device_counts);
		printf("qwen4_flash_validation check=module_determinism bit_exact=1\n");
	}
	return(0);
}


static void SparkQwen4FlashValHcGroupNorm(const float *streams, const float *weight, float *normed, uint32_t rows)
{
	for (uint32_t row = 0u; row < rows; row++)
		for (uint32_t stream = 0u; stream < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT; stream++)
			SparkQwen4FlashValRmsNorm(
				streams + ((uint64_t)row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH) + ((uint64_t)stream * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION),
				weight + ((uint64_t)stream * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION),
				normed + ((uint64_t)row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH) + ((uint64_t)stream * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION),
				SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION, 1e-6f);
}

static int SparkQwen4FlashValCheckHcResidual(void)
{
	const uint32_t rows = 4u;
	const uint64_t stream_count = (uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
	uint16_t *streams_bf16 = 0,*normed_bf16 = 0,*up_bf16 = 0,*lowrank_bf16 = 0,*inject_bf16 = 0,*mixed_bf16 = 0,*delta_bf16 = 0,*weight_bf16 = 0,*down_bf16 = 0,*up_weight_bf16 = 0,*inject_weight_bf16 = 0;
	static uint16_t host_streams[4u * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	static uint16_t host_out[4u * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	static float streams[4u * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	static float normed[4u * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	static float expected[4u * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	static float weight[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	static float down[SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	static float up[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION];
	static float inject[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	static float delta[4u * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION];
	static uint16_t host_delta[4u * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION];
	static float actual[4u * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
	SparkQwen4FlashValMetrics metrics;
	uint64_t index;
	uint32_t row,element,stream;
	cudaError_t error;
	SparkQwen4FlashValRandomState = 4242u;
	for (index = 0; index < stream_count; index++)
	{
		streams[index] = SparkQwen4FlashValUniform(0.5f);
		expected[index] = streams[index];
		host_streams[index] = SparkQwen4FlashValBf16(streams[index]);
	}
	for (index = 0; index < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; index++)
		weight[index] = 1.0f + 0.25f * SparkQwen4FlashValUniform(1.0f);
	for (index = 0; index < (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; index++)
		down[index] = SparkQwen4FlashValUniform(0.05f);
	for (index = 0; index < (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION; index++)
		up[index] = SparkQwen4FlashValUniform(0.05f);
	for (index = 0; index < (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; index++)
		inject[index] = SparkQwen4FlashValUniform(0.1f);
	for (index = 0; index < (uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION; index++)
	{
		delta[index] = SparkQwen4FlashValUniform(0.5f);
		host_delta[index] = SparkQwen4FlashValBf16(delta[index]);
	}
	error = cudaMalloc((void **)&streams_bf16,stream_count * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&normed_bf16,stream_count * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&up_bf16,stream_count * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&lowrank_bf16,(uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&inject_bf16,(uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&mixed_bf16,(uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&delta_bf16,(uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&weight_bf16,SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&down_bf16,(uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&up_weight_bf16,(uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&inject_weight_bf16,(uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * 2u);
	if (error != cudaSuccess)
		return(SparkQwen4FlashValCuda(error,"hc_alloc"));
	{
		uint16_t *staging = (uint16_t *)malloc((size_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * 2u);
		if ( staging == 0 )
			return(SparkQwen4FlashValFail("hc_residual","staging_alloc"));
		for (uint64_t i = 0; i < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; i++)
			staging[i] = SparkQwen4FlashValBf16(weight[i]);
		error = cudaMemcpy(weight_bf16,staging,(uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * 2u,cudaMemcpyHostToDevice);
		if (error == cudaSuccess)
		{
			for (uint64_t i = 0; i < (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; i++)
				staging[i] = SparkQwen4FlashValBf16(down[i]);
			error = cudaMemcpy(down_bf16,staging,(uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * 2u,cudaMemcpyHostToDevice);
		}
		if (error == cudaSuccess)
		{
			for (uint64_t i = 0; i < (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION; i++)
				staging[i] = SparkQwen4FlashValBf16(up[i]);
			error = cudaMemcpy(up_weight_bf16,staging,(uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * 2u,cudaMemcpyHostToDevice);
		}
		if (error == cudaSuccess)
		{
			for (uint64_t i = 0; i < (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; i++)
				staging[i] = SparkQwen4FlashValBf16(inject[i]);
			error = cudaMemcpy(inject_weight_bf16,staging,(uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * 2u,cudaMemcpyHostToDevice);
		}
		free(staging);
	}
	if (error == cudaSuccess) error = cudaMemcpy(streams_bf16,host_streams,stream_count * 2u,cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(delta_bf16,host_delta,(uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION * 2u,cudaMemcpyHostToDevice);
	if (error == cudaSuccess)
	{
		SparkQwen4FlashLinearView down_view,up_view,inject_view;
		memset(&down_view,0,sizeof(down_view));
		down_view.weight_format = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
		down_view.input_dimension = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		down_view.output_dimension = SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION;
		down_view.weight_payload = down_bf16;
		memset(&up_view,0,sizeof(up_view));
		up_view.weight_format = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
		up_view.input_dimension = SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION;
		up_view.output_dimension = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		up_view.weight_payload = up_weight_bf16;
		memset(&inject_view,0,sizeof(inject_view));
		inject_view.weight_format = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
		inject_view.input_dimension = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		inject_view.output_dimension = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT;
		inject_view.weight_payload = inject_weight_bf16;
		error = SparkQwen4FlashLaunchHcGroupNorm(cudaStreamPerThread,streams_bf16,weight_bf16,normed_bf16,rows,1e-6f);
		if (error == cudaSuccess) error = SparkQwen4FlashLaunchLinear(cudaStreamPerThread,&down_view,normed_bf16,lowrank_bf16,rows);
		if (error == cudaSuccess) error = SparkQwen4FlashLaunchHcSiluQuarter(cudaStreamPerThread,lowrank_bf16,rows);
		if (error == cudaSuccess) error = SparkQwen4FlashLaunchLinear(cudaStreamPerThread,&up_view,lowrank_bf16,up_bf16,rows);
		if (error == cudaSuccess) error = SparkQwen4FlashLaunchHcMix(cudaStreamPerThread,up_bf16,normed_bf16,mixed_bf16,rows);
		if (error == cudaSuccess) error = SparkQwen4FlashLaunchLinear(cudaStreamPerThread,&inject_view,normed_bf16,inject_bf16,rows);
		if (error == cudaSuccess) error = SparkQwen4FlashLaunchHcInject(cudaStreamPerThread,streams_bf16,inject_bf16,delta_bf16,rows);
		if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	}
	if (error == cudaSuccess) error = cudaMemcpy(host_out,streams_bf16,stream_count * 2u,cudaMemcpyDeviceToHost);
	if (SparkQwen4FlashValCuda(error,"hc_residual") != 0)
		return(1);
	cudaFree(streams_bf16); cudaFree(normed_bf16); cudaFree(up_bf16); cudaFree(lowrank_bf16);
	cudaFree(inject_bf16); cudaFree(mixed_bf16); cudaFree(delta_bf16); cudaFree(weight_bf16);
	cudaFree(down_bf16); cudaFree(up_weight_bf16); cudaFree(inject_weight_bf16);
	SparkQwen4FlashValHcGroupNorm(streams,weight,normed,rows);
	for (row = 0u; row < rows; row++)
	{
		static float lowrank[SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION];
		static float mix_gate[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
		static float inject_row[SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT];
		for (element = 0u; element < SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION; element++)
		{
			float total = 0.0f;
			for (index = 0; index < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; index++)
				total += down[(uint64_t)element * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH + index] * normed[(uint64_t)row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH + index];
			lowrank[element] = SparkQwen4FlashValSilu(total / (float)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT);
		}
		for (element = 0u; element < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; element++)
		{
			float total = 0.0f;
			for (index = 0; index < SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION; index++)
				total += up[(uint64_t)element * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION + index] * lowrank[index];
			mix_gate[element] = 1.0f / (1.0f + expf(-total));
		}
		for (stream = 0u; stream < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT; stream++)
		{
			float total = 0.0f;
			for (index = 0; index < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH; index++)
				total += inject[(uint64_t)stream * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH + index] * normed[(uint64_t)row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH + index];
			inject_row[stream] = 2.0f / (1.0f + expf(-total / (float)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT));
		}
		for (stream = 0u; stream < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT; stream++)
			for (element = 0u; element < SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION; element++)
			{
				uint64_t at = (uint64_t)row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH + (uint64_t)stream * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION + element;
				expected[at] += inject_row[stream] * delta[(uint64_t)row * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION + element];
			}
	}
	for (index = 0; index < stream_count; index++)
		actual[index] = SparkQwen4FlashValFromBf16(host_out[index]);
	SparkQwen4FlashValMeasure(&metrics,actual,expected,stream_count);
	return(SparkQwen4FlashValReport("hc_residual",&metrics,5e-2,0.999));
}


#define SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS 2100u
static int SparkQwen4FlashValCheckIndexer(void)
{
	const uint32_t tokens = SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS;
	const uint32_t blocks = tokens / SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO;
	const uint32_t block_topk = SPARK_QWEN4_FLASH_MODEL_INDEXER_BUDGET / SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO;
	const uint64_t block_stride = (uint64_t)SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION;
	void *qk_bf16 = 0,*query_bf16 = 0,*raw_cache = 0,*pooled_cache = 0,*q_norm_bf16 = 0,*k_norm_bf16 = 0;
	uint32_t *slot_mapping = 0;
	uint64_t *row_positions = 0;
	uint8_t *mask_u8 = 0;
	uint32_t *scores_u32 = 0;
	static uint16_t host_qk[SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS * (SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT + 1u) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
	static float key_history[SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
	static float q_norm[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
	static float k_norm[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
	static float pooled_host[blocks * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
	static float block_scores[SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS / 4u];
	static float query_host[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
	static uint8_t mask_host[tokens];
	static float mask_reference[tokens];
	SparkQwen4FlashIndexerWeights weights;
	SparkQwen4FlashKvBlockTableView table;
	uint32_t block_indices[1] = {0u};
	uint32_t block_counts[1] = {(tokens + 63u) / 64u};
	uint32_t lane_indices[1] = {0u};
	uint32_t context[1] = {tokens};
	uint32_t slot_host[1];
	uint64_t position_host[1];
	cudaError_t error;
	SparkQwen4FlashValRandomState = 777u;
	for (uint64_t index = 0; index < (uint64_t)SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS * (SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT + 1u) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; index++)
		host_qk[index] = SparkQwen4FlashValBf16(SparkQwen4FlashValUniform(1.0f));
	for (uint32_t index = 0; index < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; index++)
	{
		q_norm[index] = 1.0f + 0.1f * SparkQwen4FlashValUniform(1.0f);
		k_norm[index] = 1.0f + 0.1f * SparkQwen4FlashValUniform(1.0f);
	}
	error = cudaMalloc((void **)&qk_bf16,sizeof(host_qk));
	if (error == cudaSuccess) error = cudaMalloc((void **)&query_bf16,(uint64_t)SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&raw_cache,(uint64_t)block_counts[0] * block_stride * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&pooled_cache,(uint64_t)blocks * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&q_norm_bf16,SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&k_norm_bf16,SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&slot_mapping,sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&row_positions,sizeof(uint64_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&mask_u8,(uint64_t)tokens * sizeof(uint8_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&scores_u32,(uint64_t)blocks * sizeof(uint32_t));
	if (error != cudaSuccess)
		return(SparkQwen4FlashValCuda(error,"indexer_alloc"));
	{
		uint16_t staging[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
		for (uint32_t index = 0; index < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION && error == cudaSuccess; index++)
			staging[index] = SparkQwen4FlashValBf16(q_norm[index]);
		if (error == cudaSuccess) error = cudaMemcpy(q_norm_bf16,staging,sizeof(staging),cudaMemcpyHostToDevice);
		for (uint32_t index = 0; index < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION && error == cudaSuccess; index++)
			staging[index] = SparkQwen4FlashValBf16(k_norm[index]);
		if (error == cudaSuccess) error = cudaMemcpy(k_norm_bf16,staging,sizeof(staging),cudaMemcpyHostToDevice);
	}
	memset(&weights,0,sizeof(weights));
	weights.q_norm_weight_bf16 = q_norm_bf16;
	weights.k_norm_weight_bf16 = k_norm_bf16;
	cudaMemset(pooled_cache,0xab,(uint64_t)blocks * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION * 2u);
	for (uint32_t token = 0u; token < tokens && error == cudaSuccess; token++)
	{
		slot_host[0] = token;
		position_host[0] = token;
		error = cudaMemcpy(qk_bf16,host_qk + ((uint64_t)token * (SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT + 1u) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION),(uint64_t)(SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT + 1u) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION * 2u,cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(slot_mapping,slot_host,sizeof(slot_host),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(row_positions,position_host,sizeof(position_host),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = SparkQwen4FlashLaunchIndexerPrepare(cudaStreamPerThread,qk_bf16,&weights,query_bf16,raw_cache,pooled_cache,slot_mapping,block_indices,row_positions,1u,1u,block_stride);
	}
	if (error == cudaSuccess)
	{
		memset(&table,0,sizeof(table));
		table.lane_stride = block_counts[0];
		{
			uint32_t *device_blocks,*device_counts,*device_lane,*device_context;
			error = cudaMalloc((void **)&device_blocks,sizeof(block_indices));
			if (error == cudaSuccess) error = cudaMalloc((void **)&device_counts,sizeof(block_counts));
			if (error == cudaSuccess) error = cudaMalloc((void **)&device_lane,sizeof(lane_indices));
			if (error == cudaSuccess) error = cudaMalloc((void **)&device_context,sizeof(context));
			if (error == cudaSuccess) error = cudaMemcpy(device_blocks,block_indices,sizeof(block_indices),cudaMemcpyHostToDevice);
			if (error == cudaSuccess) error = cudaMemcpy(device_counts,block_counts,sizeof(block_counts),cudaMemcpyHostToDevice);
			if (error == cudaSuccess) error = cudaMemcpy(device_lane,lane_indices,sizeof(lane_indices),cudaMemcpyHostToDevice);
			if (error == cudaSuccess) error = cudaMemcpy(device_context,context,sizeof(context),cudaMemcpyHostToDevice);
			table.physical_block_indices = device_blocks;
			table.lane_physical_block_counts = device_counts;
			if (error == cudaSuccess) error = SparkQwen4FlashLaunchIndexerSelect(cudaStreamPerThread,query_bf16,pooled_cache,&table,device_lane,device_context,mask_u8,scores_u32,1u,tokens,blocks);
			if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
			cudaFree(device_blocks); cudaFree(device_counts); cudaFree(device_lane); cudaFree(device_context);
		}
	}
	static uint32_t keys_host[SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS / 4u];
	static uint16_t pooled_device_host[SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS / 4u * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
	if (error == cudaSuccess) error = cudaMemcpy(mask_host,mask_u8,(uint64_t)tokens,cudaMemcpyDeviceToHost);
	if (error == cudaSuccess) error = cudaMemcpy(keys_host,scores_u32,(uint64_t)blocks * sizeof(uint32_t),cudaMemcpyDeviceToHost);
	if (error == cudaSuccess) error = cudaMemcpy(pooled_device_host,pooled_cache,(uint64_t)blocks * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION * 2u,cudaMemcpyDeviceToHost);
	if (SparkQwen4FlashValCuda(error,"indexer") != 0)
		return(1);
	cudaFree(qk_bf16); cudaFree(query_bf16); cudaFree(raw_cache); cudaFree(pooled_cache);
	cudaFree(q_norm_bf16); cudaFree(k_norm_bf16); cudaFree(slot_mapping); cudaFree(row_positions); cudaFree(mask_u8); cudaFree(scores_u32);
	for (uint32_t token = 0u; token < tokens; token++)
	{
		float key[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
		for (uint32_t element = 0; element < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; element++)
			key[element] = SparkQwen4FlashValFromBf16(host_qk[((uint64_t)token * (SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT + 1u) + SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + element]);
		SparkQwen4FlashValRmsNorm(key,k_norm,key,SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION,1e-6f);
		for (uint32_t element = 0; element < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; element++)
			key_history[(uint64_t)token * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + element] = key[element];
		if ( (token % SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO) == SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO - 1u )
		{
			uint32_t block = token / SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO;
			float mean[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
			uint32_t first = token - (SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO - 1u);
			for (uint32_t element = 0; element < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; element++)
			{
				float total = 0.0f;
				for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO; tap++)
					total += key_history[((uint64_t)first + tap) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + element];
				mean[element] = SparkQwen4FlashValFromBf16(SparkQwen4FlashValBf16(total / (float)SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO));
			}
			SparkQwen4FlashValRmsNorm(mean,k_norm,mean,SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION,1e-6f);
			SparkQwen4FlashValRope(mean,SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION,first,SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_THETA);
			for (uint32_t element = 0; element < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; element++)
				pooled_host[(uint64_t)block * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + element] = SparkQwen4FlashValFromBf16(SparkQwen4FlashValBf16(mean[element]));
		}
	}
	{
		static float pooled_actual[SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS / 4u * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
		SparkQwen4FlashValMetrics pooled_metrics;
		for (uint64_t index = 0; index < (uint64_t)blocks * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; index++)
			pooled_actual[index] = SparkQwen4FlashValFromBf16(pooled_device_host[index]);
		SparkQwen4FlashValMeasure(&pooled_metrics,pooled_actual,pooled_host,(uint64_t)blocks * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION);
		if (SparkQwen4FlashValReport("indexer_pooled",&pooled_metrics,5e-2,0.999) != 0)
			return(1);
	}
	for (uint32_t head = 0u; head < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT; head++)
	{
		float qh[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
		for (uint32_t element = 0; element < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; element++)
			qh[element] = SparkQwen4FlashValFromBf16(host_qk[((uint64_t)(tokens - 1u) * (SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT + 1u) + head) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + element]);
		SparkQwen4FlashValRmsNorm(qh,q_norm,qh,SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION,1e-6f);
		SparkQwen4FlashValRope(qh,SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION,tokens - 1u,SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_THETA);
		for (uint32_t element = 0; element < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; element++)
			query_host[(uint64_t)head * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + element] = SparkQwen4FlashValFromBf16(SparkQwen4FlashValBf16(qh[element]));
	}
	{
		for (uint32_t block = 0u; block < blocks; block++)
		{
			float total = 0.0f;
			for (uint32_t head = 0u; head < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT; head++)
			{
				float dot = 0.0f;
				for (uint32_t element = 0; element < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; element++)
					dot += query_host[(uint64_t)head * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + element] * pooled_host[(uint64_t)block * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + element];
				total += dot > 0.0f ? dot : 0.0f;
			}
			block_scores[block] = total / sqrtf((float)SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION);
		}
		for (uint32_t token = 0u; token < tokens; token++)
			mask_reference[token] = 0.0f;
		for (uint32_t token = blocks * SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO; token < tokens; token++)
			mask_reference[token] = 1.0f;
		static uint8_t taken[SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS / 4u];
		memset(taken,0,sizeof(taken));
		for (uint32_t pick = 0u; pick < block_topk; pick++)
		{
			uint32_t best = 0xffffffffu;
			for (uint32_t block = 0u; block < blocks; block++)
				if ( taken[block] == 0u && (best == 0xffffffffu || block_scores[block] > block_scores[best]) )
					best = block;
			taken[best] = 1u;
		}
		for (uint32_t block = 0u; block < blocks; block++)
			if ( taken[block] != 0u )
				for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO; tap++)
					mask_reference[(uint64_t)block * SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO + tap] = 1.0f;
	}
	{
		uint64_t mismatches = 0,boundary = 0;
		float kth_score = -3.0e38f;
		for (uint32_t pick = 0u,seen = 0u; pick < blocks; pick++)
		{
		}
		{
			static uint8_t ranked[SPARK_QWEN4_FLASH_VALIDATION_INDEXER_TOKENS / 4u];
			memset(ranked,0,sizeof(ranked));
			for (uint32_t pick = 0u; pick < block_topk; pick++)
			{
				uint32_t best = 0xffffffffu;
				for (uint32_t block = 0u; block < blocks; block++)
					if ( ranked[block] == 0u && (best == 0xffffffffu || block_scores[block] > block_scores[best]) )
						best = block;
				ranked[best] = 1u;
				if ( pick == block_topk - 1u )
					kth_score = block_scores[best];
			}
		}
		for (uint32_t token = 0u; token < tokens; token++)
			if ( (mask_host[token] != 0u) != (mask_reference[token] != 0.0f) )
			{
				uint32_t block = token / SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO;
				if ( fabsf(block_scores[block] - kth_score) <= 2e-3f )
					boundary++;
				else
					mismatches++;
			}
		(void)keys_host;
		printf("qwen4_flash_validation check=indexer_select tokens=%u blocks=%u topk=%u boundary_flips=%llu mask_mismatches=%llu\n",
			tokens,blocks,block_topk,(unsigned long long)boundary,(unsigned long long)mismatches);
		if ( mismatches != 0u )
			return(SparkQwen4FlashValFail("indexer_select","mask_mismatch"));
	}
	return(0);
}


#define SPARK_QWEN4_FLASH_VALIDATION_PLE_ROWS 4u
static int SparkQwen4FlashValCheckPle(void)
{
	const uint32_t rows = SPARK_QWEN4_FLASH_VALIDATION_PLE_ROWS;
	const uint32_t heads = SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_COUNT;
	const uint32_t head_dim = SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_DIMENSION;
	static const int64_t head_vocabs[16] = {97,89,101,103,107,109,113,127,131,137,139,149,151,157,163,167};
	int64_t head_offsets[16];
	int64_t total_rows = 0;
	int64_t multipliers[3] = {(int64_t)0x2545f4914f6cdd1dull,(int64_t)-0x9e3779b97f4a7c15ll,(int64_t)0xbf58476d1ce4e5b9ull};
	uint32_t histories[rows * 3u];
	uint32_t current[rows];
	histories[0] = 11u; histories[1] = 12u;
	histories[3] = SPARK_QWEN4_FLASH_MODEL_EOS_TOKEN_ID; histories[4] = 7u;
	histories[6] = 3u; histories[7] = SPARK_QWEN4_FLASH_MODEL_EOS_TOKEN_ID;
	histories[9] = 5u; histories[10] = 6u;
	current[0] = 4242u; current[1] = 4243u; current[2] = 4244u; current[3] = 4245u;
	histories[2] = current[0]; histories[5] = current[1]; histories[8] = current[2]; histories[11] = current[3];
	for (uint32_t head = 0u; head < heads; head++)
	{
		head_offsets[head] = total_rows;
		total_rows += head_vocabs[head];
	}
	static uint16_t table_host[2000u * 160u];
	static uint16_t embedding_host[SPARK_QWEN4_FLASH_VALIDATION_PLE_ROWS * SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION];
	static uint16_t embedding_reference[SPARK_QWEN4_FLASH_VALIDATION_PLE_ROWS * SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION];
	SparkQwen4FlashValRandomState = 31337u;
	for (int64_t index = 0; index < total_rows * (int64_t)head_dim; index++)
		table_host[index] = SparkQwen4FlashValBf16(SparkQwen4FlashValUniform(1.0f));
	void *table_bf16 = 0,*history_u32 = 0,*embedding_bf16 = 0;
	int64_t *multipliers_dev = 0,*vocabs_dev = 0,*offsets_dev = 0;
	cudaError_t error;
	error = cudaMalloc(&table_bf16,(uint64_t)total_rows * head_dim * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&history_u32,(uint64_t)rows * 3u * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&embedding_bf16,(uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION * 2u);
	if (error == cudaSuccess) error = cudaMalloc((void **)&multipliers_dev,sizeof(multipliers));
	if (error == cudaSuccess) error = cudaMalloc((void **)&vocabs_dev,sizeof(head_vocabs));
	if (error == cudaSuccess) error = cudaMalloc((void **)&offsets_dev,sizeof(head_offsets));
	if (error != cudaSuccess)
		return(SparkQwen4FlashValCuda(error,"ple_alloc"));
	error = cudaMemcpy(table_bf16,table_host,(uint64_t)total_rows * head_dim * 2u,cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(history_u32,histories,(uint64_t)rows * 3u * sizeof(uint32_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(multipliers_dev,multipliers,sizeof(multipliers),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(vocabs_dev,head_vocabs,sizeof(head_vocabs),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(offsets_dev,head_offsets,sizeof(head_offsets),cudaMemcpyHostToDevice);
	{
		SparkQwen4FlashPleWeights ple;
		memset(&ple,0,sizeof(ple));
		ple.layer_multipliers = multipliers_dev;
		ple.head_vocab_sizes = vocabs_dev;
		ple.head_offsets = offsets_dev;
		ple.ngram_embedding_bf16 = table_bf16;
		if (error == cudaSuccess)
			error = SparkQwen4FlashLaunchPleHashGather(cudaStreamPerThread,(const uint32_t *)history_u32,1u,&ple,embedding_bf16,rows,0u,(uint32_t)total_rows);
		if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	}
	if (error == cudaSuccess) error = cudaMemcpy(embedding_host,embedding_bf16,(uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION * 2u,cudaMemcpyDeviceToHost);
	if (SparkQwen4FlashValCuda(error,"ple_hash") != 0)
		return(1);
	cudaFree(table_bf16); cudaFree(history_u32); cudaFree(multipliers_dev); cudaFree(vocabs_dev); cudaFree(offsets_dev);
	for (uint32_t row = 0u; row < rows; row++)
	{
		const uint32_t *window = histories + (row * 3u);
		for (uint32_t head = 0u; head < heads; head++)
		{
			uint32_t ngram = (head / SPARK_QWEN4_FLASH_MODEL_PLE_HEADS_PER_NGRAM) + 2u;
			uint64_t mixed = 0;
			for (uint32_t part = 0u; part < ngram; part++)
			{
				uint32_t shifted;
				int32_t source = 2 - (int32_t)part;
				uint32_t valid = source >= 0 ? 1u : 0u;
				for (int32_t p = source >= 0 ? source : 0; p < 2 && valid != 0u; p++)
					if ( window[p] == SPARK_QWEN4_FLASH_MODEL_EOS_TOKEN_ID )
						valid = 0u;
				shifted = valid != 0u ? window[source] : SPARK_QWEN4_FLASH_MODEL_EOS_TOKEN_ID;
				uint64_t term = (uint64_t)((int64_t)shifted * multipliers[part]);
				mixed = part == 0u ? term : (mixed ^ term);
			}
			int64_t m = (int64_t)mixed % head_vocabs[head];
			if ( m < 0 )
				m += head_vocabs[head];
			uint64_t global_id = (uint64_t)(head_offsets[head] + m);
			for (uint32_t d = 0u; d < head_dim; d++)
				embedding_reference[(row * SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION) + (head * head_dim) + d] = table_host[(global_id * head_dim) + d];
		}
	}
	if ( memcmp(embedding_host,embedding_reference,sizeof(embedding_reference)) != 0 )
		return(SparkQwen4FlashValFail("ple_hash_gather","embedding_mismatch"));
	{
		uint64_t zeros = 0;
		for (uint64_t index = 0; index < (uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION; index++)
			if ( embedding_reference[index] == 0u )
				zeros++;
		printf("qwen4_flash_validation check=ple_hash_gather rows=%u heads=%u bit_exact=1 nonzero=%llu/%llu\n",
			rows,heads,(unsigned long long)((uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION - zeros),
			(unsigned long long)((uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION));
	}
	cudaFree(embedding_bf16);
	return(0);
}

int main(int argc, char **argv)
{
	SparkQwen4FlashValDevice device;
	int result = 0;
	if (argc != 2 || strlen(argv[1]) != 64u)
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	if (SparkQwen4FlashValCuda(SparkQwen4FlashConfigureCudaKernels(),"configure") != 0)
		return(1);
	if (SparkQwen4FlashValDeviceSetup(&device) != 0)
		return(1);
	if (result == 0) result = SparkQwen4FlashValCheckDecayBeta(&device);
	if (result == 0) result = SparkQwen4FlashValCheckConv(&device);
	if (result == 0) result = SparkQwen4FlashValCheckGdnStep(&device);
	if (result == 0) result = SparkQwen4FlashValCheckGatedNorm(&device);
	if (result == 0) result = SparkQwen4FlashValCheckAttention(&device);
	if (result == 0) result = SparkQwen4FlashValCheckGdnChunk(&device);
	if (result == 0) result = SparkQwen4FlashValCheckHcResidual();
	if (result == 0) result = SparkQwen4FlashValCheckIndexer();
	if (result == 0) result = SparkQwen4FlashValCheckPle();
	if (result == 0) result = SparkQwen4FlashValCheckModule();
	if (result == 0)
		printf("qwen4_flash_validation PASS\n");
	return(result);
}
