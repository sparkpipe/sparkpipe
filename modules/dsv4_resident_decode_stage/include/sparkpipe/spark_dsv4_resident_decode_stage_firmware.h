#pragma once

#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 * Version 2 executes DECODE batches (one token per lane per frame) across
 * all three attention kinds, both router paths, and the full mHC
 * machinery. Prefill frames use the same proven one-token state transition
 * in a serial loop. This is intentionally a correctness path: it writes
 * every prompt token into the resident KV state before decode, while a
 * future bulk prefill kernel can replace the loop without changing the
 * boundary contract. MTP execution remains refused until its pass lands.
 * Caches are dense per lane, bounded by SPARK_DSV4_STAGE_MAX_SEQ: the
 * window ring is 128 slots regardless, the compressed stream max_seq/ratio
 * slots, the indexer stream max_seq/4; the paged migration is scheduled
 * with the family PP pass and changes only the module, not this contract.
 */

#define SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 2u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_PREFILL_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 16u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP 4096u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 128u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT 61u

typedef struct SparkDsv4LinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	const void *payload;
	const uint8_t *scale_e8m0;
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
	uint32_t lane_count;
	const uint32_t *token_ids;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkDsv4PrefillBatchView;

#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000001u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000002u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000004u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW 0x00000008u
#define SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW 0x00000010u

typedef SparkStatus (*SparkDsv4HiddenTransportPostReceiveFunction)(SparkHiddenTransportSession *transport_session, SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkDsv4HiddenTransportSendFunction)(SparkHiddenTransportSession *transport_session, const SparkHiddenTransportPacket *packet);

typedef struct SparkDsv4ResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkDsv4DecodeBatchView *decode_batch;
	const SparkDsv4PrefillBatchView *prefill_batch;
	SparkHiddenTransportSession *hidden_input_transport_session;
	SparkHiddenTransportSession *hidden_output_transport_session;
	SparkDsv4HiddenTransportPostReceiveFunction hidden_input_post_receive_function;
	SparkDsv4HiddenTransportSendFunction hidden_output_send_function;
	SparkHiddenTransportPacket hidden_input_packet;
	SparkHiddenTransportPacket hidden_output_packet;
} SparkDsv4ResidentDecodeStageFrameContext;

SparkStatus SparkDsv4ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
SparkStatus SparkDsv4ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
SparkStatus SparkDsv4ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkDsv4ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
void SparkDsv4ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif
