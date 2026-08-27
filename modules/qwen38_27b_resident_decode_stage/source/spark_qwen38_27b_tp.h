#ifndef SPARKPIPE_SPARK_QWEN38_27B_TP_H
#define SPARKPIPE_SPARK_QWEN38_27B_TP_H

#include <stdatomic.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tp_device_collective.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tensor-parallel execution state for the qwen38_27b resident decode stage.
 *
 * The collective runs on the hidden-transport backend (host-RDMA verbs on
 * the 100G rail, the dsv4 pattern): submissions are stream-ordered on the
 * producing slot stream and the module waits for each operation's
 * stream-ordered completion before the consuming layer, so the reduction
 * overlaps the transport with no device synchronizations. The NCCL backend
 * remains selectable via SPARK_QWEN38_27B_TP_BACKEND=nccl.
 *
 * Sharding geometry at degree D (recipe qwen38_27b.TP4):
 *   - MLP gate/up rows and down columns: 17408 / D
 *   - attention q rows: 12288 / D, kv rows: 1024 / D, o columns: 6144 / D
 *   - GDN fused q|k|v rows: 10240 / D (stitched q|k|v per rank),
 *     GDN out columns: 6144 / D
 *   - lm_head rows: 248320 / D (embedding replicated)
 *   - norms, conv, beta/decay, MTP: replicated
 */
typedef struct SparkQwen38_27bTpState
{
	SparkTpDeviceCollective collective;
	uint32_t initialized;
	uint32_t degree;
	uint32_t rank;
	/* per-rank geometry, derived from the model constants */
	uint32_t gdn_qk_channels;
	uint32_t gdn_value_channels;
	uint32_t gdn_conv_channels;
	uint32_t gdn_key_heads;
	uint32_t gdn_value_heads;
	uint32_t attn_query_heads;
	uint32_t attn_kv_heads;
	uint32_t ffn_intermediate;
	uint32_t head_rows;
	void *credit_send_bf16;
	void *credit_receive_bf16;
	void *host_credit_send_bf16;
	void *host_credit_receive_bf16;
	/* 1 while credit_send/receive_bf16 own real cudaMalloc allocations;
	 * 0 once they alias the mapped host buffers (the originals were freed
	 * in that case). Destroy frees exactly what this flag says is owned -
	 * it must never guess aliasing from pointer nullness. */
	uint32_t credit_device_allocated;
	uint64_t next_ordinal;
	void *cuda_stream;
} SparkQwen38_27bTpState;

/* Per-operation completion wait cell: the collective's completion callback
 * stores the status and releases the spinner. */
typedef struct SparkQwen38_27bTpPending
{
	atomic_uint done;
	uint32_t status;
} SparkQwen38_27bTpPending;

/* degree 1 = no tensor parallelism; the struct stays zeroed and unused. */
SparkStatus SparkQwen38_27bTpInitialize(
	SparkQwen38_27bTpState *tp,
	uint32_t degree,
	uint32_t rank,
	uint32_t max_active_sequence_count,
	uint32_t pipeline_slot_count,
	void *registration_cuda_stream);

void SparkQwen38_27bTpDestroy(SparkQwen38_27bTpState *tp);
/* Credit-buffer allocation with explicit ownership; mapped_host selects
 * the mapped-host aliasing path. Failure paths destroy partial state. */
SparkStatus SparkQwen38_27bTpAllocateCreditMemory(
	SparkQwen38_27bTpState *tp,
	uint64_t total_bytes,
	uint32_t mapped_host);

/* Stream-ordered BF16 all-reduce of rows * hidden_dimension elements. The
 * reduction is enqueued on the caller's stream between the producing and
 * consuming kernels; on return the operation's completion has been
 * observed, so the next kernels launched on the same stream are ordered
 * after the reduction. */
SparkStatus SparkQwen38_27bTpReduceHidden(
	SparkQwen38_27bTpState *tp,
	void *buffer,
	uint32_t rows,
	void *cuda_stream);

/* Stream-ordered u64 maxloc reduce over count elements (head shard
 * argmax); same ordering contract as the hidden reduce. */
SparkStatus SparkQwen38_27bTpReduceU64Max(
	SparkQwen38_27bTpState *tp,
	uint64_t *buffer,
	uint32_t count,
	void *cuda_stream);

#ifdef __cplusplus
}
#endif

#endif /* SPARKPIPE_SPARK_QWEN38_27B_TP_H */
