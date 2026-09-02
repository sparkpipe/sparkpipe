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

typedef struct SparkQwen38_27bTpState
{
	SparkTpDeviceCollective collective;
	uint32_t initialized;
	uint32_t degree;
	uint32_t rank;
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
	uint32_t credit_device_allocated;
	uint64_t next_ordinal;
	void *cuda_stream;
} SparkQwen38_27bTpState;

typedef struct SparkQwen38_27bTpPending
{
	atomic_uint done;
	uint32_t status;
} SparkQwen38_27bTpPending;

SparkStatus SparkQwen38_27bTpInitialize(
	SparkQwen38_27bTpState *tp,
	uint32_t degree,
	uint32_t rank,
	uint32_t max_active_sequence_count,
	uint32_t pipeline_slot_count,
	void *registration_cuda_stream);

void SparkQwen38_27bTpDestroy(SparkQwen38_27bTpState *tp);
SparkStatus SparkQwen38_27bTpAllocateCreditMemory(
	SparkQwen38_27bTpState *tp,
	uint64_t total_bytes,
	uint32_t mapped_host);

SparkStatus SparkQwen38_27bTpReduceHidden(
	SparkQwen38_27bTpState *tp,
	void *buffer,
	uint32_t rows,
	void *cuda_stream);

SparkStatus SparkQwen38_27bTpReduceU64Max(
	SparkQwen38_27bTpState *tp,
	uint64_t *buffer,
	uint32_t count,
	void *cuda_stream);

#ifdef __cplusplus
}
#endif

#endif
