#pragma once

#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"

// The batch-variant tuning header controls only the per-submit compute width.
// Resident lane state has its own ceiling so the scheduler can keep more
// sequences bound than one CUDA submission can process.
#include "sparkpipe/spark_dsv4_batch_tuning.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DSV4_RESIDENT_DECODE_STAGE_CACHE_BLOCK_TOKENS 128u

/*
 * DeepSeek V4 resident decode stage. This header deliberately includes NO
 * model header: the translation unit picks Flash or Pro by including its
 * spark_dsv4[_pro]_model.h first (shared include guard) or via the build's
 * -include, and every view below is geometry-free - dimensions live in the
 * entries and the variant macros, verified at load.
 *
 * The stage boundary carries the FOUR hyper-connection streams: a boundary
 * packet row is hc_mult x hidden bf16 (BOUNDARY_STREAM_ELEMENTS), because
 * mHC never collapses between layers - only hc_pre inside a block and the
 * head reduction do. Stage 0 expands the embedding into four identical
 * streams; the last stage's head reduction is the only exit.
 *
 * Version 3 executes GA baseline DECODE batches (one token per lane per
 * frame) across
 * all three attention kinds, both router paths, and the full mHC
 * machinery. Prefill dense work executes every frame row together. Only the
 * short-window cache publication and sparse attention stay round-major: one
 * row per live lane, followed immediately by that row's attention. This keeps
 * the 128-slot ring causal while batching projections, compression, mHC, and
 * MoE across the whole frame. GA DSpark execution remains refused, its three
 * checkpoint layers are excluded from baseline packs, and serving reports
 * zero speculative-token capacity.
 * DSV4 defines the bytes and offsets within a 128-token cache page. The
 * model-neutral cache runtime owns logical pages, residency, prefix sharing,
 * prefetch, and eviction; this device contract owns only DSV4 page geometry
 * and the CUDA addresses used by its kernels.
 * Every accepted frame completes externally from one stream-ordered host
 * callback; submit never synchronizes a successful CUDA frame.
 */

#define SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 9u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 3u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_PREFILL_BATCH_VIEW_ABI_VERSION 2u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 16u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP 4096u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT \
	SPARK_DSV4_BATCH_TUNING_SEQUENCE_CEILING
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT 1024u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT 16384u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PHYSICAL_PAGE_COUNT 16384u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LOGICAL_PAGE_COUNT 1048576u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 13u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT 61u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_COUNT 64u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_TP_PEER_COUNT 16u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_TP_HOST_NAME_BYTES 64u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_TENSOR_PARALLEL UINT32_C(0x00000001)
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS \
	SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_TENSOR_PARALLEL

typedef struct SparkDsv4ResidentDecodeStageNodeContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t resident_sequence_capacity;
	uint32_t pipeline_slot_count;
	uint32_t max_sequence_positions;
	uint32_t linear_weight_codec;
	uint32_t expert_weight_codec;
	uint32_t kv_cache_codec;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint64_t tp_configuration_hash;
	uint16_t tp_listen_port;
	uint16_t tp_peer_ports[SPARK_DSV4_RESIDENT_DECODE_STAGE_TP_PEER_COUNT];
	uint32_t tp_connect_timeout_milli;
	uint32_t tp_operation_timeout_milli;
	uint64_t tp_collective_identifier;
	char tp_peer_hosts[SPARK_DSV4_RESIDENT_DECODE_STAGE_TP_PEER_COUNT][SPARK_DSV4_RESIDENT_DECODE_STAGE_TP_HOST_NAME_BYTES];
	const char *tp_local_host;
	const char *tp_transport_module_path;
	uint32_t tp_transport_control_port_base;
	/*
	 * Decode-step CUDA graph cache size, 0 disables capture and preserves the
	 * eager launch path exactly. Captured shapes are the fixed (pipeline slot,
	 * lane count) pairs actually submitted; a full cache or any capture
	 * failure falls back to eager launches, never to a wrong sequence.
	 */
	uint32_t cuda_graph_count;
	const char *stage_pack_path;
} SparkDsv4ResidentDecodeStageNodeContext;

#define SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES \
	((uint32_t)sizeof(SparkDsv4ResidentDecodeStageNodeContext))

typedef struct SparkDsv4LinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	const void *payload;
	const void *scale_data;
} SparkDsv4LinearView;

typedef struct SparkDsv4CompressorWeights
{
	const float *ape_f32;
	const void *norm_weight_bf16;
	SparkDsv4LinearView wkv;
	SparkDsv4LinearView wgate;
	uint32_t ratio;
	uint32_t overlap;
} SparkDsv4CompressorWeights;

typedef struct SparkDsv4AttnWeights
{
	const float *sink_f32;
	const void *q_norm_weight_bf16;
	const void *kv_norm_weight_bf16;
	SparkDsv4LinearView wq_a;
	SparkDsv4LinearView wq_b;
	SparkDsv4LinearView wkv;
	SparkDsv4LinearView wo_a;
	SparkDsv4LinearView wo_b;
} SparkDsv4AttnWeights;

typedef struct SparkDsv4IndexerWeights
{
	SparkDsv4LinearView wq_b;
	SparkDsv4LinearView weights_proj;
	SparkDsv4CompressorWeights compressor;
} SparkDsv4IndexerWeights;

// Routed experts are STACKED: one view spans all experts, expert e's block
// is rows_per_expert consecutive rows; the launch offsets payload and
// scale by e * rows_per_expert. Exactly one of bias / tid2eid is non-null,
// the hash pin.
typedef struct SparkDsv4MoeWeights
{
	SparkDsv4LinearView gate;
	const float *gate_bias_f32;
	const uint32_t *gate_tid2eid_u32;
	SparkDsv4LinearView experts_w1;
	SparkDsv4LinearView experts_w2;
	SparkDsv4LinearView experts_w3;
	SparkDsv4LinearView shared_w1;
	SparkDsv4LinearView shared_w2;
	SparkDsv4LinearView shared_w3;
} SparkDsv4MoeWeights;

typedef struct SparkDsv4HcWeights
{
	const float *attn_fn_f32;
	const float *attn_base_f32;
	const float *attn_scale_f32;
	const float *ffn_fn_f32;
	const float *ffn_base_f32;
	const float *ffn_scale_f32;
} SparkDsv4HcWeights;

typedef struct SparkDsv4MtpWeights
{
	SparkDsv4LinearView e_proj;
	SparkDsv4LinearView h_proj;
	const void *enorm_weight_bf16;
	const void *hnorm_weight_bf16;
	const void *final_norm_weight_bf16;
	const float *hc_head_fn_f32;
	const float *hc_head_base_f32;
	const float *hc_head_scale_f32;
} SparkDsv4MtpWeights;

typedef struct SparkDsv4DecodeBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t reserved0;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkDsv4DecodeBatchView;

typedef struct SparkDsv4PrefillBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t active_sequence_count;
	uint32_t emit_count;
	uint32_t reserved0;
	const uint32_t *token_ids;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
	const uint32_t *emit_row_indices;
	const uint32_t *emit_lane_indices;
} SparkDsv4PrefillBatchView;

#if defined(__CUDACC__)
#define SPARK_DSV4_RESIDENT_HOST_DEVICE __host__ __device__
#else
#define SPARK_DSV4_RESIDENT_HOST_DEVICE
#endif

static SPARK_DSV4_RESIDENT_HOST_DEVICE inline uint32_t SparkDsv4AttentionWindowSlot(
	uint64_t position,
	uint32_t column,
	uint32_t window_token_count)
{
	uint32_t first;
	if ( window_token_count == 0u || column >= window_token_count )
		return(UINT32_MAX);
	first = position + 1u < window_token_count ? 0u :
		(uint32_t)(((position % window_token_count) + 1u) % window_token_count);
	return((first + column) % window_token_count);
}

#undef SPARK_DSV4_RESIDENT_HOST_DEVICE

static inline uint64_t SparkDsv4PrefillRowElementOffset(
	uint32_t first_row,
	uint32_t row_element_count)
{
	return((uint64_t)first_row * row_element_count);
}

#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000001u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER 0x00000002u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER 0x00000004u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW 0x00000008u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW 0x00000010u

typedef struct SparkDsv4ResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkDsv4DecodeBatchView *decode_batch;
	const SparkDsv4PrefillBatchView *prefill_batch;
	const void *hidden_input_bf16;
	uint64_t hidden_input_bytes;
	void *hidden_output_bf16;
	uint64_t hidden_output_bytes;
} SparkDsv4ResidentDecodeStageFrameContext;

SparkStatus SparkDsv4ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
SparkStatus SparkDsv4ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
SparkStatus SparkDsv4ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkDsv4ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
void SparkDsv4ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif
