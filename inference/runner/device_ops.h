#ifndef SPARKPIPE_INFERENCE_RUNNER_DEVICE_OPS_H
#define SPARKPIPE_INFERENCE_RUNNER_DEVICE_OPS_H

#include <stdint.h>

#include "sparkpipe/spark_device.h"
#include "inference/kernels/metal/kernel_params.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_OPS_GEMV_MAX_ROWS 8u
#define SPARK_OPS_ATTENTION_HEAD_DIM 256u
#define SPARK_OPS_DELTA_KEY_DIM 128u
#define SPARK_OPS_PARTIAL_FLOATS (SPARK_OPS_ATTENTION_HEAD_DIM + 2u)

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
	SPARK_OPS_GEMV_GATED_R1,
	SPARK_OPS_GEMV_GATED_R2,
	SPARK_OPS_GEMV_GATED_R4,
	SPARK_OPS_GEMV_GATED_R8,
	SPARK_OPS_GEMM,
	SPARK_OPS_GEMM_F32,
	SPARK_OPS_CAUSAL_CONV,
	SPARK_OPS_DELTA_PREPARE,
	SPARK_OPS_DELTA_RULE,
	SPARK_OPS_KV_STORE,
	SPARK_OPS_ATTENTION_DECODE,
	SPARK_OPS_ATTENTION_PREFILL,
	SPARK_OPS_ATTENTION_COMBINE,
	SPARK_OPS_SILU_MUL,
	SPARK_OPS_ADD_ROWS,
	SPARK_OPS_LOGPROB,
	SPARK_OPS_STREAM_READ,
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
void SparkDeviceOpsRelease(SparkDeviceOps *ops);
SparkStatus SparkDeviceOpsTakeError(SparkDeviceOps *ops, uint32_t *error);
SparkTensor SparkTensorAt(const SparkDeviceBuffer *buffer, uint64_t offset, uint64_t bytes);
SparkTensor SparkTensorSlice(SparkTensor tensor, uint64_t offset, uint64_t bytes);

SparkStatus SparkOpsGather(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor table, SparkTensor ids, SparkTensor output,
	uint32_t rows, uint32_t width, uint32_t table_rows);
SparkStatus SparkOpsRmsNormResidual(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, const SparkTensor *addend,
	SparkTensor weight, SparkTensor output, const SparkNormParams *params);
SparkStatus SparkOpsRmsNormGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor gate, SparkTensor weight,
	SparkTensor output, const SparkGatedNormParams *params);
SparkStatus SparkOpsHeadRmsNormRope(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor weight, SparkTensor positions,
	SparkTensor output, const SparkHeadRopeParams *params);
SparkStatus SparkOpsLinear(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor y, const SparkGemvParams *params);
SparkStatus SparkOpsCausalConv(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor window, const SparkOpsSequences *sequences,
	SparkTensor input, SparkTensor weight, SparkTensor output, const SparkConvParams *params);
SparkStatus SparkOpsGatedDelta(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor state, const SparkOpsSequences *sequences,
	SparkTensor mix, SparkTensor gates, SparkTensor log_scale, SparkTensor bias, SparkTensor scratch, SparkTensor output,
	const SparkDeltaParams *params);
SparkStatus SparkOpsKvStore(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor key, SparkTensor value,
	const SparkKvParams *params);
SparkStatus SparkOpsAttention(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor query, SparkTensor gate,
	SparkTensor partial, SparkTensor output, const SparkAttentionParams *params, const uint32_t *host_positions, const uint32_t *host_sequence_of_row);
SparkStatus SparkOpsAddRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, SparkTensor addend, uint32_t rows,
	uint32_t dimension, uint32_t hidden_stride, uint32_t addend_stride);
SparkStatus SparkOpsSiluMul(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor output, const SparkSiluMulParams *params);
SparkStatus SparkOpsLogprob(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor logits, SparkTensor targets, SparkTensor result,
	SparkTensor argmax, const SparkLogprobParams *params);
SparkStatus SparkOpsStreamRead(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor words, SparkTensor sink, uint32_t groups);
uint64_t SparkOpsDeltaScratchBytes(uint32_t rows, uint32_t key_heads, uint32_t value_heads, uint32_t key_dimension);
uint64_t SparkOpsAttentionScratchBytes(uint32_t rows, uint32_t heads, uint32_t positions);

#ifdef __cplusplus
}
#endif

#endif
