#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_qwen38_27b_model.h"
#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"
#define SPARK_FAMILY_CAMEL Qwen38_27b
#define SPARK_FAMILY_UPPER QWEN38_27B
#define SPARK_FAMILY_LOWER qwen38_27b

#include "sparkpipe/family/spark_family.h"

#define SPARK_QWEN38_27B_VALIDATION_ROWS 4u
#define SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS 8u
#define SPARK_QWEN38_27B_VALIDATION_CHUNK_TOKENS 128u
#define SPARK_QWEN38_27B_VALIDATION_ATTN_TOKENS 5u
#define SPARK_QWEN38_27B_VALIDATION_KV_LANES SPARK_QWEN38_27B_STAGE_MAX_ACTIVE_SEQUENCES
#define SPARK_QWEN38_27B_VALIDATION_ATTN_GROUP (SPARK_QWEN38_27B_MODEL_ATTN_QUERY_HEAD_COUNT / SPARK_QWEN38_27B_MODEL_ATTN_KV_HEAD_COUNT)

extern "C" cudaError_t SparkQwen38_27bConfigureCudaKernels(void);
extern "C" cudaError_t SparkQwen38_27bLaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen38_27bGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen38_27bGdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern "C" cudaError_t SparkQwen38_27bLaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen38_27bGdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count);
extern "C" cudaError_t SparkQwen38_27bLaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen38_27bGdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern "C" cudaError_t SparkQwen38_27bLaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen38_27bGdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon);
extern "C" cudaError_t SparkQwen38_27bLaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen38_27bAttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon);
extern "C" cudaError_t SparkQwen38_27bLaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen38_27bKvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride);
extern "C" cudaError_t SparkQwen38_27bLaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen38_27bGdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal);

static uint32_t SparkQwen38_27bValRandomState;

#include "sparkpipe/family/validation/spark_val_rng.h"
#include "sparkpipe/family/validation/spark_val_from_bf16.h"
#include "sparkpipe/family/validation/spark_val_fill_bf16_half_up.h"
#include "sparkpipe/family/validation/spark_val_fail.h"
#include "sparkpipe/family/validation/spark_val_cuda.h"
#include "sparkpipe/family/validation/spark_val_measure.h"
#include "sparkpipe/family/validation/spark_val_gdn_attention.h"
#include "sparkpipe/family/validation/spark_val_gdn_device.h"

#define SPARK_QWEN38_27B_VALIDATION_GDN_LAUNCH(launch,...) launch(__VA_ARGS__)
#define SPARK_QWEN38_27B_VALIDATION_ATTN_PREPARE_LAUNCH(launch,...) launch(__VA_ARGS__)
#define SPARK_QWEN38_27B_VALIDATION_ATTN_DECODE_LAUNCH(launch,...) launch(__VA_ARGS__)
#include "sparkpipe/family/validation/spark_val_gdn_attention_checks.h"

#define SPARK_QWEN38_27B_VALIDATION_HIDDEN_WIDTH SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION
#include "sparkpipe/family/validation/spark_val_module_frame.h"

#define SPARK_QWEN38_27B_VALIDATION_MODEL_ID "Qwen/Qwen3.8-27B"
#define SPARK_QWEN38_27B_VALIDATION_NODE_TARGET "cuda.sm121.qwen38_27b.resident_decode_stage.bf16"
#include "sparkpipe/family/validation/spark_val_module_initialize.h"

static int SparkQwen38_27bValCheckModule(void)
{
	SparkQwen38_27bValModule module;
	uint16_t decode_hidden[SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION];
	uint16_t prefill_hidden[SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION];
	uint16_t rerun_hidden[SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION];
	uint32_t decode_token = 0u,prefill_token = 0u,rerun_token = 0u;
	uint32_t index;
	SparkQwen38_27bValMetrics metrics;
	SparkModelDriverAdmissionRequest admission;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	if (SparkQwen38_27bValModuleInitialize(&module) != 0)
		return(1);
	memset(&admission,0,sizeof(admission));
	admission.descriptor_bytes = sizeof(admission);
	admission.program_id = 1u;
	admission.frame_flags = 0u;
	admission.new_token_count = 1u;
	admission.active_slot_count = 1u;
	memset(&decision,0,sizeof(decision));
	decision.descriptor_bytes = sizeof(decision);
	status = SparkQwen38_27bResidentDecodeStageAdmit(module.state,&admission,&decision);
	if (status != SPARK_STATUS_OK || decision.accepted == 0u)
		return(SparkQwen38_27bValFail("module_admit","rejected"));
	memset(&snapshot,0,sizeof(snapshot));
	SparkModelDriverInitializeRuntimeSnapshot(&snapshot,1u);
	status = SparkQwen38_27bResidentDecodeStageSnapshot(module.state,1u,&snapshot);
	if (status != SPARK_STATUS_OK)
		return(SparkQwen38_27bValFail("module_snapshot","status"));
	printf("qwen38_27b_validation check=module_admit_snapshot admit=ok snapshot=ok\n");
	for (index = 0u; index < SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS; index++)
		module.token_ids[index] = 1000u + (index * 37u) % 200000u;
	module.positions[0] = 0u;
	module.sequence_ids[0] = 1u;
	if (SparkQwen38_27bValModuleExecute(&module,1u,SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS,0u,0u,0) != 0)
		return(1);
	if (module.capture.sends != (module.head_stage != 0u ? 0u : 1u))
		return(SparkQwen38_27bValFail("module_prefill",module.head_stage != 0u ? "unexpected_hidden_send" : "no_hidden_send"));
	if (module.head_stage == 0u && SparkQwen38_27bValCheckFinite("module_prefill",module.capture.hidden,SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS) != 0)
		return(1);
	module.token_ids[0] = 4242u;
	module.lanes[0] = 0u;
	module.positions[0] = SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS;
	module.sequence_ids[0] = 1u;
	if (SparkQwen38_27bValModuleExecute(&module,0u,1u,0u,0u,0) != 0)
		return(1);
	if (module.head_stage == 0u)
	{
		memcpy(decode_hidden,module.capture.hidden,sizeof(decode_hidden));
		if (SparkQwen38_27bValCheckFinite("module_decode",decode_hidden,1u) != 0)
			return(1);
	}
	else
		decode_token = module.output_token_ids[0];
	for (index = 0u; index < SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS; index++)
		module.token_ids[index] = 1000u + (index * 37u) % 200000u;
	module.positions[0] = 0u;
	module.sequence_ids[0] = 2u;
	if (SparkQwen38_27bValModuleExecute(&module,1u,SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS,1u,0u,0) != 0)
		return(1);
	module.token_ids[0] = 4242u;
	module.positions[0] = SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS;
	module.sequence_ids[0] = 2u;
	if (SparkQwen38_27bValModuleExecute(&module,1u,1u,1u,0u,0) != 0)
		return(1);
	if (module.head_stage == 0u)
	{
		memcpy(prefill_hidden,module.capture.hidden,sizeof(prefill_hidden));
		{
			float actual[SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION];
			float reference[SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION];
			for (index = 0u; index < SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION; index++)
			{
				actual[index] = SparkQwen38_27bValFromBf16(decode_hidden[index]);
				reference[index] = SparkQwen38_27bValFromBf16(prefill_hidden[index]);
			}
			SparkQwen38_27bValMeasure(&metrics,actual,reference,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION);
			if (SparkQwen38_27bValReport("module_decode_vs_prefill",&metrics,5e-2,0.999) != 0)
				return(1);
		}
	}
	else
	{
		prefill_token = module.output_token_ids[0];
		printf("qwen38_27b_validation check=module_decode_vs_prefill decode_token=%u prefill_token=%u bit_exact=%d\n",decode_token,prefill_token,decode_token == prefill_token ? 1 : 0);
		if (decode_token != prefill_token)
			return(SparkQwen38_27bValFail("module_decode_vs_prefill","token_mismatch"));
	}
#if SPARK_QWEN38_27B_MODEL_MTP_LAYER_COUNT != 0
	if (module.head_stage != 0u && SparkQwen38_27bValCheckMtpDraft(&module) != 0)
		return(1);
#endif
	SparkQwen38_27bResidentDecodeStageDestroy(module.state);
	cudaFree(module.device_blocks);
	cudaFree(module.device_counts);
	{
		SparkQwen38_27bValModule rerun;
		if (SparkQwen38_27bValModuleInitialize(&rerun) != 0)
			return(1);
		for (index = 0u; index < SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS; index++)
			rerun.token_ids[index] = 1000u + (index * 37u) % 200000u;
		rerun.positions[0] = 0u;
		rerun.sequence_ids[0] = 1u;
		if (SparkQwen38_27bValModuleExecute(&rerun,1u,SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS,0u,0u,0) != 0)
			return(1);
		rerun.token_ids[0] = 4242u;
		rerun.lanes[0] = 0u;
		rerun.positions[0] = SPARK_QWEN38_27B_VALIDATION_PREFILL_TOKENS;
		rerun.sequence_ids[0] = 1u;
		if (SparkQwen38_27bValModuleExecute(&rerun,0u,1u,0u,0u,0) != 0)
			return(1);
		if (module.head_stage == 0u)
		{
			memcpy(rerun_hidden,rerun.capture.hidden,sizeof(rerun_hidden));
			if (memcmp(decode_hidden,rerun_hidden,sizeof(decode_hidden)) != 0)
				return(SparkQwen38_27bValFail("module_determinism","fresh_instance_mismatch"));
		}
		else
		{
			rerun_token = rerun.output_token_ids[0];
			if (rerun_token != decode_token)
				return(SparkQwen38_27bValFail("module_determinism","fresh_instance_token_mismatch"));
		}
		SparkQwen38_27bResidentDecodeStageDestroy(rerun.state);
		cudaFree(rerun.device_blocks);
		cudaFree(rerun.device_counts);
		printf("qwen38_27b_validation check=module_determinism bit_exact=1\n");
	}
	return(0);
}

int main(int argc, char **argv)
{
	SparkQwen38_27bValDevice device;
	int result = 0;
	if (argc != 2 || strlen(argv[1]) != 64u)
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	if (SparkQwen38_27bValCuda(SparkQwen38_27bConfigureCudaKernels(),"configure") != 0)
		return(1);
	if (SparkQwen38_27bValDeviceSetup(&device) != 0)
		return(1);
	if (result == 0) result = SparkQwen38_27bValCheckDecayBeta(&device);
	if (result == 0) result = SparkQwen38_27bValCheckConv(&device);
	if (result == 0) result = SparkQwen38_27bValCheckGdnStep(&device);
	if (result == 0) result = SparkQwen38_27bValCheckGatedNorm(&device);
	if (result == 0) result = SparkQwen38_27bValCheckAttention(&device);
	if (result == 0) result = SparkQwen38_27bValCheckGdnChunk(&device);
	if (result == 0) result = SparkQwen38_27bValCheckModule();
	if (result == 0)
		printf("qwen38_27b_validation PASS\n");
	return(result);
}
