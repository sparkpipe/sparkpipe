#ifndef SPARKPIPE_SPARK_QWEN36_TP_H
#define SPARKPIPE_SPARK_QWEN36_TP_H

#include <stdint.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tp_device_collective.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tensor-parallel execution state for the qwen36 resident decode stage.
 *
 * v1 uses the NCCL collective backend with SYNCHRONOUS stream-ordered
 * submissions: the caller submits the reduction and synchronizes the
 * execution stream before the next sharded GEMM reads the result. The
 * credit machinery is the one-credit degenerate case; the async
 * continuation path (the dsv4 pattern) is the follow-up.
 *
 * Sharding geometry at degree D (recipe qwen36.TP4):
 *   - MLP gate/up rows and down columns: 17408 / D
 *   - attention q rows: 12288 / D, kv rows: 1024 / D, o columns: 6144 / D
 *   - GDN fused q|k|v rows: 10240 / D (stitched q|k|v per rank),
 *     GDN out columns: 6144 / D
 *   - lm_head rows: 248320 / D (embedding replicated in v1)
 *   - norms, conv, beta/decay, MTP: replicated
 */
typedef struct SparkQwen36TpState
{
	SparkTpDeviceCollective collective;
	uint32_t initialized;
	uint32_t degree;
	uint32_t rank;
	/* per-rank geometry, derived from the model constants */
	uint32_t gdn_qk_channels;
	uint32_t gdn_value_channels;
	uint32_t gdn_conv_channels;
	uint32_t attn_query_heads;
	uint32_t attn_kv_heads;
	uint32_t ffn_intermediate;
	uint32_t head_rows;
	void *credit_send_bf16;
	void *credit_receive_bf16;
	uint64_t next_ordinal;
	void *cuda_stream;
} SparkQwen36TpState;

/* degree 1 = no tensor parallelism; the struct stays zeroed and unused. */
SparkStatus SparkQwen36TpInitialize(
	SparkQwen36TpState *tp,
	uint32_t degree,
	uint32_t rank,
	uint32_t max_active_sequence_count,
	void *registration_cuda_stream);

void SparkQwen36TpDestroy(SparkQwen36TpState *tp);

/* Synchronous BF16 all-reduce of rows * hidden_dimension elements.
 * The caller must have written the local partial into buffer; on return the
 * buffer holds the reduced value on every rank. */
SparkStatus SparkQwen36TpReduceHidden(
	SparkQwen36TpState *tp,
	void *buffer,
	uint32_t rows);

/* Synchronous u64 maxloc reduce over count elements (head shard argmax). */
SparkStatus SparkQwen36TpReduceU64Max(
	SparkQwen36TpState *tp,
	uint64_t *buffer,
	uint32_t count);

static inline SparkStatus SparkQwen36TpStreamSynchronize(
	const SparkQwen36TpState *tp)
{
	return cudaStreamSynchronize((cudaStream_t)tp->cuda_stream) == cudaSuccess
		? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

#ifdef __cplusplus
}
#endif

#endif /* SPARKPIPE_SPARK_QWEN36_TP_H */
