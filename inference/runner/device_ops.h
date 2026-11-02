#ifndef SPARKPIPE_INFERENCE_RUNNER_DEVICE_OPS_H
#define SPARKPIPE_INFERENCE_RUNNER_DEVICE_OPS_H

#include <stdint.h>

#include "sparkpipe/spark_device.h"
#include "inference/kernels/metal/kernel_params.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_OPS_GEMV_MAX_ROWS 2u
#define SPARK_OPS_GEMV_PASS_ROWS 8u
#define SPARK_OPS_FP8_GEMV_ROWS 16u
#define SPARK_OPS_DECODE_MAX_ROWS 64u
#define SPARK_OPS_SPLIT_TARGET_GROUPS 1024u
#define SPARK_OPS_PARTIAL_BYTES ((uint64_t)SPARK_OPS_DECODE_MAX_ROWS * 2u * SPARK_OPS_SPLIT_TARGET_GROUPS * SPARK_GEMM_ROWS_OUTPUTS * sizeof(float))
#define SPARK_OPS_ATTENTION_HEAD_DIM 256u
#define SPARK_OPS_ATTENTION_WIDE_HEAD_DIM 512u
#define SPARK_OPS_ATTENTION_NARROW_HEAD_DIM 128u
#define SPARK_OPS_ATTENTION_SPLIT_KEY_DIM 192u
#define SPARK_OPS_ATTENTION_SPLIT_VALUE_DIM 128u
#define SPARK_OPS_ATTENTION_LATENT_VALUE_DIM 512u
#define SPARK_OPS_ATTENTION_LATENT_ROPE_DIM 64u
#define SPARK_OPS_ATTENTION_LATENT_KEY_DIM (SPARK_OPS_ATTENTION_LATENT_VALUE_DIM + SPARK_OPS_ATTENTION_LATENT_ROPE_DIM)
#define SPARK_OPS_DELTA_KEY_DIM 128u
#define SPARK_OPS_MAX_LANES 16u

typedef enum SparkOpsKernel
{
	SPARK_OPS_GATHER = 0,
	SPARK_OPS_RMSNORM_RESIDUAL,
	SPARK_OPS_RMSNORM_GATE,
	SPARK_OPS_HEAD_RMSNORM_ROPE,
	SPARK_OPS_GEMV_R1,
	SPARK_OPS_GEMV_R2,
	SPARK_OPS_GEMV_R4,
	SPARK_OPS_GEMV_R8,
	SPARK_OPS_GEMV_F32_R1,
	SPARK_OPS_GEMV_F32_R2,
	SPARK_OPS_GEMV_F32_R4,
	SPARK_OPS_GEMV_F32_R8,
	SPARK_OPS_GEMV_FP8_R1,
	SPARK_OPS_GEMV_FP8_R2,
	SPARK_OPS_GEMV_FP8_R4,
	SPARK_OPS_GEMV_FP8_R8,
	SPARK_OPS_DEQUANT_FP8,
	SPARK_OPS_GEMM,
	SPARK_OPS_GEMM_F32,
	SPARK_OPS_GEMM_ROWS16,
	SPARK_OPS_GEMM_ROWS32,
	SPARK_OPS_GEMM_REDUCE,
	SPARK_OPS_CAUSAL_CONV,
	SPARK_OPS_DELTA_PREPARE,
	SPARK_OPS_DELTA_RULE,
	SPARK_OPS_KV_STORE,
	SPARK_OPS_ATTENTION_DECODE,
	SPARK_OPS_ATTENTION_DECODE_WIDE,
	SPARK_OPS_ATTENTION_DECODE_NARROW,
	SPARK_OPS_ATTENTION_DECODE_SPLIT,
	SPARK_OPS_ATTENTION_DECODE_LATENT,
	SPARK_OPS_ATTENTION_PREFILL,
	SPARK_OPS_ATTENTION_PREFILL_WIDE,
	SPARK_OPS_ATTENTION_PREFILL_NARROW,
	SPARK_OPS_ATTENTION_PREFILL_SPLIT,
	SPARK_OPS_ATTENTION_PREFILL_LATENT,
	SPARK_OPS_ATTENTION_COMBINE,
	SPARK_OPS_ATTENTION_COMBINE_WIDE,
	SPARK_OPS_ATTENTION_COMBINE_NARROW,
	SPARK_OPS_ATTENTION_COMBINE_SPLIT,
	SPARK_OPS_ATTENTION_COMBINE_LATENT,
	SPARK_OPS_SILU_MUL,
	SPARK_OPS_GELU_TANH_MUL,
	SPARK_OPS_SILU_MUL_ROUTED,
	SPARK_OPS_GELU_TANH_MUL_ROUTED,
	SPARK_OPS_RMSNORM_BRANCHES,
	SPARK_OPS_RMSNORM_SCALED,
	SPARK_OPS_SCALE_ROWS,
	SPARK_OPS_SOFTCAP_ROWS,
	SPARK_OPS_ROUTE_TOPK,
	SPARK_OPS_ROUTE_PINNED,
	SPARK_OPS_ROUTE_COMBINE,
	SPARK_OPS_GEMV_ROUTED,
	SPARK_OPS_GEMV_MXFP4_ROUTED,
	SPARK_OPS_ADD_ROWS,
	SPARK_OPS_LOGPROB,
	SPARK_OPS_STREAM_READ,
	SPARK_OPS_SILU_ROUNDED_MUL,
	SPARK_OPS_ROUTE_COMBINE_WEIGHTED,
	SPARK_OPS_ROUTE_SIGMOID,
	SPARK_OPS_RMSNORM_BF16,
	SPARK_OPS_HYPER_EXPAND,
	SPARK_OPS_GROUP_RMSNORM,
	SPARK_OPS_HYPER_GATE,
	SPARK_OPS_HYPER_MIX,
	SPARK_OPS_HYPER_INJECT_WEIGHTS,
	SPARK_OPS_HYPER_INJECT,
	SPARK_OPS_HYPER_KEY_GATE,
	SPARK_OPS_HYPER_ADD,
	SPARK_OPS_GEMV_NVFP4_ROUTED,
	SPARK_OPS_ROUTE_COMBINE_SHARED,
	SPARK_OPS_SIGMOID_GATE_ROWS,
	SPARK_OPS_NGRAM_HASH,
	SPARK_OPS_NGRAM_GATHER_FP8,
	SPARK_OPS_NGRAM_HISTORY,
	SPARK_OPS_ROUTE_LOCAL,
	SPARK_OPS_RANK_SUM_BF16,
	SPARK_OPS_RANK_MAX_U64,
	SPARK_OPS_RANK_ROWS,
	SPARK_OPS_HEAD_GATE,
	SPARK_OPS_CHANNEL_DELTA_PREPARE,
	SPARK_OPS_SILU_CLAMP_MUL,
	SPARK_OPS_SILU_CLAMP_MUL_ROUTED,
	SPARK_OPS_SILU_GATE_LIMIT_MUL,
	SPARK_OPS_SILU_GATE_LIMIT_MUL_ROUTED,
	SPARK_OPS_GEMV_MXFP8_R1,
	SPARK_OPS_GEMV_MXFP8_R2,
	SPARK_OPS_GEMV_MXFP8_R4,
	SPARK_OPS_GEMV_MXFP8_R8,
	SPARK_OPS_DEQUANT_MXFP8,
	SPARK_OPS_GEMV_MXFP8_ROUTED,
	SPARK_OPS_HC_MIX8,
	SPARK_OPS_HC_MIX4,
	SPARK_OPS_HC_POST,
	SPARK_OPS_HC_SITE,
	SPARK_OPS_HC_POST_COMB,
	SPARK_OPS_STREAM_MEAN,
	SPARK_OPS_GEMV_FP8_ROUTED,
	SPARK_OPS_SITU_MUL,
	SPARK_OPS_ATTNRES_SCORE,
	SPARK_OPS_ATTNRES_MIX,
	SPARK_OPS_GEMM_FP8_BLOCKS,
	SPARK_OPS_GEMM_MXFP8_BLOCKS,
	SPARK_OPS_ROUTE_SQRT_SOFTPLUS,
	SPARK_OPS_BLOCK_QDQ,
	SPARK_OPS_KV_COMPRESS,
	SPARK_OPS_KV_COMPRESS_SAVE,
	SPARK_OPS_COMPRESSED_POSITIONS,
	SPARK_OPS_HC_SITE_LAGGED,
	SPARK_OPS_HC_POST_COMB_F32,
	SPARK_OPS_HC_COLLAPSE,
	SPARK_OPS_STREAM_KEY_GATE,
	SPARK_OPS_KV_STORE_SKIP,
	SPARK_OPS_ATTENTION_DECODE_EXTRA,
	SPARK_OPS_ATTENTION_COMBINE_EXTRA,
	SPARK_OPS_NGRAM_HASH_MAPPED,
	SPARK_OPS_NGRAM_GATHER_E8M0,
	SPARK_OPS_NGRAM_HISTORY_MAPPED,
	SPARK_OPS_POLY_NORM_MUL,
	SPARK_OPS_DIFFERENTIAL_GATE,
	SPARK_OPS_HC_SITE_CLAMPED,
	SPARK_OPS_ROUTE_COMBINE_WEIGHTED_SHARED,
	SPARK_OPS_KERNEL_COUNT
} SparkOpsKernel;

typedef struct SparkTensor
{
	SparkDeviceRange range;
	uint64_t bytes;
} SparkTensor;

typedef struct SparkDeviceOps
{
	SparkDeviceLibrary *library;
	SparkDeviceKernel *kernels[SPARK_OPS_KERNEL_COUNT];
	SparkDeviceBuffer error;
	SparkDeviceBuffer partial;
	SparkDeviceQueue *partial_queue;
	uint32_t lane;
} SparkDeviceOps;

typedef struct SparkOpsSequences
{
	SparkTensor state_index;
	SparkTensor row_begin;
	uint32_t sequences;
	uint32_t commit;
} SparkOpsSequences;

typedef struct SparkOpsPaged
{
	SparkTensor pool;
	SparkTensor page_table;
	SparkTensor sequence_of_row;
	SparkTensor positions;
	uint32_t page_slots;
	uint32_t pages_per_sequence;
	uint32_t pool_pages;
	uint32_t kv_heads;
} SparkOpsPaged;

SparkStatus SparkDeviceOpsOpen(SparkDeviceOps *ops, const char *library_path);
void SparkDeviceOpsClose(SparkDeviceOps *ops);
SparkStatus SparkDeviceOpsAcquire(SparkDeviceOps **ops);
SparkStatus SparkDeviceOpsAcquireLane(SparkDeviceOps **ops, uint32_t lane);
void SparkDeviceOpsRelease(SparkDeviceOps *ops);
SparkStatus SparkDeviceOpsTakeError(SparkDeviceOps *ops, uint32_t *error);
SparkTensor SparkTensorAt(const SparkDeviceBuffer *buffer, uint64_t offset, uint64_t bytes);
SparkTensor SparkTensorSlice(SparkTensor tensor, uint64_t offset, uint64_t bytes);

SparkStatus SparkOpsGather(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor table, SparkTensor ids, SparkTensor output,
	uint32_t rows, uint32_t width, uint32_t table_rows);
SparkStatus SparkOpsGatherSlice(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor table, SparkTensor ids, SparkTensor output,
	uint32_t rows, uint32_t width, uint32_t table_rows, uint32_t first_row);
SparkStatus SparkOpsRouteLocal(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor experts, const SparkRouteParams *params);
SparkStatus SparkOpsHeadGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, SparkTensor logits, const SparkHeadGateParams *params);
SparkStatus SparkOpsRankSumBf16(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor lanes, SparkTensor full, uint32_t ranks, uint64_t elements);
SparkStatus SparkOpsRankMaxU64(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor lanes, SparkTensor full, uint32_t ranks, uint64_t elements);
SparkStatus SparkOpsRankRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor gathered, SparkTensor rows, uint32_t ranks, uint32_t row_count,
	uint32_t width);
SparkStatus SparkOpsRmsNormResidual(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, const SparkTensor *addend,
	SparkTensor weight, SparkTensor output, const SparkNormParams *params);
SparkStatus SparkOpsRmsNormGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor gate, SparkTensor weight,
	SparkTensor output, const SparkGatedNormParams *params);
SparkStatus SparkOpsHeadRmsNormRope(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor weight, SparkTensor positions,
	SparkTensor output, const SparkHeadRopeParams *params);
SparkStatus SparkOpsLinear(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor y, const SparkGemvParams *params);
SparkStatus SparkOpsLinearFp8(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor scale, SparkTensor dequant,
	SparkTensor y, const SparkGemvParams *params);
SparkStatus SparkOpsCausalConv(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor window, const SparkOpsSequences *sequences,
	SparkTensor input, SparkTensor weight, SparkTensor output, const SparkConvParams *params);
SparkStatus SparkOpsGatedDelta(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor state, const SparkOpsSequences *sequences,
	SparkTensor mix, SparkTensor gates, SparkTensor log_scale, SparkTensor bias, SparkTensor scratch, SparkTensor output,
	const SparkDeltaParams *params);
SparkStatus SparkOpsChannelDelta(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor state, const SparkOpsSequences *sequences,
	SparkTensor mix, SparkTensor decay_logits, SparkTensor beta_logits, SparkTensor log_scale, SparkTensor bias, SparkTensor scratch, SparkTensor output,
	const SparkDeltaParams *params);
SparkStatus SparkOpsKvStore(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor key, SparkTensor value,
	const SparkKvParams *params);
SparkStatus SparkOpsAttention(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor query, SparkTensor gate,
	const SparkTensor *sinks, SparkTensor partial, SparkTensor output, const SparkAttentionParams *params, const uint32_t *host_positions,
	const uint32_t *host_sequence_of_row);
SparkStatus SparkOpsAddRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, SparkTensor addend, uint32_t rows,
	uint32_t dimension, uint32_t hidden_stride, uint32_t addend_stride);
SparkStatus SparkOpsGatedMul(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor output, const SparkSiluMulParams *params,
	uint32_t activation);
SparkStatus SparkOpsGatedMulRouted(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor weights, SparkTensor output,
	const SparkSiluMulParams *params, uint32_t activation);
SparkStatus SparkOpsRmsNormBranches(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, SparkTensor first, const SparkTensor *second,
	const SparkTensor *first_weight, const SparkTensor *second_weight, SparkTensor output_weight, const SparkTensor *scalar, const SparkBranchNormParams *params);
SparkStatus SparkOpsRmsNormScaled(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor scale, SparkTensor output,
	const SparkBranchNormParams *params);
SparkStatus SparkOpsScaleRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, uint64_t count, float scale);
SparkStatus SparkOpsSoftcap(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, uint64_t count, float cap, float multiply);
SparkStatus SparkOpsRouteTopk(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor logits, SparkTensor expert_scale, const SparkTensor *pinned,
	SparkTensor experts, SparkTensor weights, const SparkRouteParams *params);
SparkStatus SparkOpsRouteCombineWeighted(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor expert_out, SparkTensor weights, SparkTensor output,
	const SparkRouteParams *params);
SparkStatus SparkOpsRouteCombineWeightedShared(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor expert_out, SparkTensor weights, SparkTensor shared,
	SparkTensor output, const SparkRouteParams *params);
SparkStatus SparkOpsRouteCombineShared(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor expert_out, SparkTensor shared, SparkTensor output,
	const SparkRouteParams *params);
SparkStatus SparkOpsSigmoidGateRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, SparkTensor logits, const SparkRouteParams *params);
SparkStatus SparkOpsRouteSigmoid(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor logits, SparkTensor bias, SparkTensor experts, SparkTensor weights,
	const SparkRouteParams *params);
SparkStatus SparkOpsAttnResScore(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor bank, SparkTensor partial, SparkTensor weight, SparkTensor scores,
	const SparkAttnResParams *params);
SparkStatus SparkOpsAttnResMix(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor bank, SparkTensor partial, SparkTensor scores, SparkTensor output,
	const SparkAttnResParams *params);
SparkStatus SparkOpsRmsNormBf16(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor weight, SparkTensor output,
	const SparkNormParams *params);
SparkStatus SparkOpsHcMix(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor fn, SparkTensor base, SparkTensor scale,
	SparkTensor post, SparkTensor mixed, const SparkHcParams *params);
SparkStatus SparkOpsHcPost(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor post, SparkTensor output, const SparkHcParams *params);
SparkStatus SparkOpsHcSite(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor fn, SparkTensor base, SparkTensor scale,
	SparkTensor post, SparkTensor comb, SparkTensor mixed, const SparkHcParams *params);
SparkStatus SparkOpsHcPostComb(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor post, SparkTensor comb, SparkTensor output,
	const SparkHcParams *params);
SparkStatus SparkOpsStreamMean(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor output, const SparkHcParams *params);
SparkStatus SparkOpsHyperExpand(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, SparkTensor streams, const SparkHyperParams *params);
SparkStatus SparkOpsGroupRmsNorm(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor weight, SparkTensor output,
	const SparkHyperParams *params);
SparkStatus SparkOpsHyperGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, const SparkHyperParams *params);
SparkStatus SparkOpsHyperMix(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor up, SparkTensor normed, SparkTensor mixed,
	const SparkHyperParams *params);
SparkStatus SparkOpsHyperInjectWeights(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor normed, SparkTensor weight, SparkTensor inject,
	const SparkHyperParams *params);
SparkStatus SparkOpsHyperInject(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor output, SparkTensor inject,
	const SparkHyperParams *params);
SparkStatus SparkOpsHyperKeyGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor key, SparkTensor query, SparkTensor value, SparkTensor gated,
	const SparkHyperParams *params);
SparkStatus SparkOpsHyperAdd(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor gated, SparkTensor conv,
	const SparkHyperParams *params);
SparkStatus SparkOpsNgramHash(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, SparkTensor positions, const SparkOpsSequences *sequences,
	SparkTensor history, SparkTensor multipliers, SparkTensor primes, SparkTensor offsets, SparkTensor ids, const SparkNgramParams *params);
SparkStatus SparkOpsNgramGatherFp8(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor ids, SparkTensor payload, SparkTensor scale, SparkTensor embed,
	const SparkNgramParams *params);
SparkStatus SparkOpsNgramHistory(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, const SparkTensor *row_map, const SparkOpsSequences *sequences,
	SparkTensor history, const SparkNgramParams *params);
SparkStatus SparkOpsRouteCombine(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor expert_out, SparkTensor output,
	const SparkRouteParams *params);
SparkStatus SparkOpsLinearRouted(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor y, SparkTensor experts,
	uint32_t expert_count, const SparkGemvParams *params);
SparkStatus SparkOpsLinearRoutedFp8(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor scale, SparkTensor y,
	SparkTensor experts, uint32_t expert_count, const SparkGemvParams *params);
SparkStatus SparkOpsLinearRoutedMxfp4(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor scale, SparkTensor y,
	SparkTensor experts, uint32_t expert_count, const SparkGemvParams *params);
SparkStatus SparkOpsLinearRoutedNvfp4(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor scale, SparkTensor global_scale,
	SparkTensor y, SparkTensor experts, uint32_t expert_count, const SparkGemvParams *params);
SparkStatus SparkOpsLogprob(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor logits, SparkTensor targets, SparkTensor result,
	SparkTensor argmax, const SparkLogprobParams *params);
SparkStatus SparkOpsStreamRead(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor words, SparkTensor sink, uint32_t groups);
SparkStatus SparkOpsBlockQdq(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor output, const SparkQdqParams *params);
SparkStatus SparkOpsHcSiteLagged(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor fn, SparkTensor base, SparkTensor scale,
	SparkTensor pre, SparkTensor post, SparkTensor comb, SparkTensor mixed, const SparkHcParams *params);
SparkStatus SparkOpsHcPostCombF32(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor post, SparkTensor comb, SparkTensor output,
	const SparkHcParams *params);
SparkStatus SparkOpsHcSiteClamped(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor fn, SparkTensor base, SparkTensor scale,
	SparkTensor post, SparkTensor comb, SparkTensor mixed, const SparkHcParams *params);
SparkStatus SparkOpsPolyNormMul(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor gate_up, SparkTensor params_f32, const SparkTensor *param_of_row,
	SparkTensor output, const SparkPolyNormParams *params);
SparkStatus SparkOpsDifferentialGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor heads, SparkTensor lambda, SparkTensor gate, SparkTensor output,
	const SparkDifferentialGateParams *params);
SparkStatus SparkOpsHcCollapse(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor pre, SparkTensor output, const SparkHcParams *params);
SparkStatus SparkOpsStreamKeyGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor kv, SparkTensor query_weight,
	SparkTensor key_weight, const SparkGateParams *params);
SparkStatus SparkOpsKvCompress(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor kv, SparkTensor score, SparkTensor state, const SparkOpsSequences *sequences,
	SparkTensor positions, SparkTensor pooled, const SparkCompressParams *params);
SparkStatus SparkOpsCompressedPositions(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor positions, SparkTensor store, SparkTensor rope, SparkTensor last,
	const SparkCompressParams *params);
SparkStatus SparkOpsAttentionExtra(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, const SparkOpsPaged *extra, SparkTensor query,
	const SparkTensor *sinks, SparkTensor partial, SparkTensor output, const SparkAttentionParams *params, const uint32_t *host_positions,
	const uint32_t *host_extra_positions);
SparkStatus SparkOpsNgramHashMapped(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, SparkTensor token_map, SparkTensor positions,
	const SparkOpsSequences *sequences, SparkTensor history, SparkTensor multipliers, SparkTensor primes, SparkTensor offsets, SparkTensor ids,
	const SparkNgramParams *params);
SparkStatus SparkOpsNgramGatherE8m0(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor ids, SparkTensor payload, SparkTensor scale, SparkTensor embed,
	const SparkNgramParams *params);
SparkStatus SparkOpsNgramHistoryMapped(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, SparkTensor token_map, const SparkOpsSequences *sequences,
	SparkTensor history, const SparkNgramParams *params);
uint64_t SparkOpsDeltaScratchBytes(uint32_t rows, uint32_t key_heads, uint32_t value_heads, uint32_t key_dimension);
uint64_t SparkOpsChannelDeltaScratchBytes(uint32_t rows, uint32_t heads, uint32_t key_dimension);
uint64_t SparkOpsAttentionScratchBytes(uint32_t rows, uint32_t heads, uint32_t head_dimension, uint32_t positions);

#ifdef __cplusplus
}
#endif

#endif
