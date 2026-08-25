#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"
#include "spark_qwen38_27b_dspark_format.h"

/*
 * DFlash2 selector host sequence (adoption item W7).
 *
 * ONE definition of "how the drafter turns its block hidden into draft ids",
 * shared by the module's block forward and by the host-path validator, so the
 * thing under test is the thing that ships. It replaces the DSpark host path -
 * a [block, 248320] BF16 logits D2H (3.97 MB per step) plus a full-vocabulary
 * Markov rewrite and argmax per position, over host mirrors of the two
 * [248320, 256] codebooks (254 MB of host RAM copied at initialize) - with two
 * device launches and a B-1 word D2H.
 *
 * Geometry, straight from the reference forward: the block is
 * SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE positions, position 0 carries the COMMITTED
 * token C0 and is the walk's anchor, so the selector runs on the remaining
 * B-1 mask positions (the oracle's hidden[1:]) and emits exactly B-1 drafts -
 * which is also what SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE (7)
 * asks for. The anchor id stays on the DEVICE: it is the token the target head
 * just emitted into slot->output_token_ids, so nothing round-trips to decide it.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern uint64_t SparkQwen38_27bDsparkHeadTopKChunkKeyCount(uint32_t row_count, uint32_t top_k);
extern cudaError_t SparkQwen38_27bLaunchDsparkHeadTopK(cudaStream_t stream, const SparkQwen38_27bLinearView *head, const void *hidden_bf16, uint64_t *chunk_keys, uint32_t *top_candidate_ids, float *top_scores_f32, void *top_scores_bf16, uint32_t row_count, uint32_t candidate_offset, uint32_t top_k);
extern cudaError_t SparkQwen38_27bLaunchDsparkSelector(cudaStream_t stream, const void *hidden_bf16, const void *hidden_projection_bf16, const void *predecessor_bf16, const void *successor_bf16, const uint32_t *candidate_ids, const uint32_t *anchor_token_ids, const float *unary_f32, void *context_gate_bf16, float *edges_f32, uint32_t *draft_token_ids, uint32_t *draft_candidate_slots, uint32_t batch_count, uint32_t slot_count, uint32_t top_k, uint32_t rank, uint32_t hidden_dimension);

#ifdef __cplusplus
}
#endif

/* Device scratch the sequence needs, all caller owned and all tiny: at the
 * shipped geometry (7 slots, K 16, rank 256) this is 114 KiB of chunk keys and
 * under 12 KiB of everything else, against the 3.97 MB logits buffer it
 * replaces. */
typedef struct SparkQwen38_27bDsparkSelectorWorkspace
{
	uint64_t *chunk_keys;       /* SparkQwen38_27bDsparkHeadTopKChunkKeyCount(slots, top_k) */
	uint32_t *candidate_ids;    /* slots * top_k */
	float *candidate_scores;    /* slots * top_k, the BF16-exact unary logits */
	void *context_gate_bf16;    /* slots * rank */
	float *edges_f32;           /* slots * top_k * top_k */
	uint32_t *draft_token_ids;  /* slots */
} SparkQwen38_27bDsparkSelectorWorkspace;

static inline uint64_t SparkQwen38_27bDsparkSelectorWorkspaceBytes(uint32_t slot_count, uint32_t top_k, uint32_t rank)
{
	return((SparkQwen38_27bDsparkHeadTopKChunkKeyCount(slot_count,top_k) * sizeof(uint64_t)) +
		((uint64_t)slot_count * top_k * sizeof(uint32_t)) +
		((uint64_t)slot_count * top_k * sizeof(float)) +
		((uint64_t)slot_count * rank * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES) +
		((uint64_t)slot_count * top_k * top_k * sizeof(float)) +
		((uint64_t)slot_count * sizeof(uint32_t)));
}

/*
 * mask_hidden_bf16 is the FINAL-NORMED block hidden with the anchor row already
 * skipped ([slot_count, hidden]); anchor_token_id_device is the committed token
 * the walk starts from. host_draft_token_ids receives slot_count ids and is
 * valid once the stream has been synchronized by the caller.
 */
static inline cudaError_t SparkQwen38_27bDsparkSelectorEmit(
	cudaStream_t stream,
	const SparkQwen38_27bLinearView *head,
	const void *mask_hidden_bf16,
	const void *hidden_projection_bf16,
	const void *predecessor_bf16,
	const void *successor_bf16,
	const uint32_t *anchor_token_id_device,
	const SparkQwen38_27bDsparkSelectorWorkspace *workspace,
	uint32_t slot_count,
	uint32_t top_k,
	uint32_t rank,
	uint32_t hidden_dimension,
	uint32_t *host_draft_token_ids)
{
	cudaError_t error;
	if ( head == 0 || mask_hidden_bf16 == 0 || hidden_projection_bf16 == 0 || predecessor_bf16 == 0 ||
	     successor_bf16 == 0 || anchor_token_id_device == 0 || workspace == 0 || host_draft_token_ids == 0 )
		return(cudaErrorInvalidValue);
	if ( slot_count == 0u || top_k == 0u || rank == 0u || hidden_dimension == 0u )
		return(cudaErrorInvalidValue);
	/* W4: top_k candidates per mask slot, straight off the dense BF16 target
	 * head through the module's own head view. */
	error = SparkQwen38_27bLaunchDsparkHeadTopK(stream,head,mask_hidden_bf16,workspace->chunk_keys,
		workspace->candidate_ids,workspace->candidate_scores,0,slot_count,0u,top_k);
	/* W3: context gate, the top_k x top_k lattice, and the greedy walk from the
	 * committed anchor - the walk's own kernel threads the previous pick, so the
	 * slot-to-slot dependency never leaves the device. */
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchDsparkSelector(stream,mask_hidden_bf16,hidden_projection_bf16,
			predecessor_bf16,successor_bf16,workspace->candidate_ids,anchor_token_id_device,
			workspace->candidate_scores,workspace->context_gate_bf16,workspace->edges_f32,
			workspace->draft_token_ids,0,1u,slot_count,top_k,rank,hidden_dimension);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(host_draft_token_ids,workspace->draft_token_ids,
			(size_t)slot_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	return(error);
}
