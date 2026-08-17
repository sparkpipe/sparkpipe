#pragma once
// The Kimi K3 stage slice: bind a layer's weights and state, run it, and carry
// the AttnRes stream between layers the way the reference does.
//
// A HEADER, NOT A TRANSLATION UNIT, so a host harness can execute the real
// slice loop with a recorder format. bind.cu was the only home for this code
// and it includes the 4-bit format headers, whose inline PTX assembles nowhere
// on a CPU - which is why no gate had ever executed the loop, and why every
// defect in it survived per-kernel and per-layer testing. The loop is where
// the audit's wiring findings all lived.
//
// THE DATAFLOW IS THE REFERENCE'S, STEP FOR STEP. _forward_attn_residual:
//
//     P = incoming stream
//     h = retrieval(P, bank, attn_proj)        when the bank is non-empty
//     if layer % 12 == 0: bank.append(P); P = None
//     a = attention(input_layernorm(h))
//     P = a            on a boundary           (the None branch)
//     P = P + a        otherwise
//     h = retrieval(P, bank, mlp_proj)         always
//     m = mlp(post_attention_layernorm(h))
//     P = P + m
//
// and after the last layer the model applies one more retrieval with its own
// weight before the final norm. Three things in that listing carried defects
// here: the retrieval REPLACES the module input rather than joining a residual;
// the partial RESTARTS at a block boundary rather than carrying through; and
// the attention-side retrieval sees the bank BEFORE the append, so its source
// count at a boundary is one lower than the MLP side's - a single formula
// double-counted, and the vector being appended also scored as the partial.
#include "inference/llms/kimi_k3/layer.cuh"
#include "inference/llms/kimi_k3/dspark.h"

struct K3LayerWeights
{
	const void *attn_norm_weight;
	const void *mlp_norm_weight;

	// KDA, PACK V2 (docs/K3_PACK_FORMAT_V2.md). The six projections that read
	// the normed input are TWO tensors: kda_qkv_beta_weight fuses q|k|v|beta
	// head-major (per-head widths 128/128/128/1), kda_decay_gate_down_weight
	// fuses decay_down|gate_down replicated across TP. The V1 per-projection
	// tensors no longer exist in the pack, so they no longer exist here. The
	// layer derives the section offsets from K3_KDA_*_FUSED_ROWS in
	// generated_config.h - the constants the packer validates against - and
	// asserts the tiling at compile time; bind stays pointer arithmetic and
	// no manifest JSON is parsed anywhere on this path.
	const void *kda_qkv_beta_weight;
	const void *kda_decay_down_weight;
	const float *kda_q_conv_weight;
	const float *kda_k_conv_weight;
	const float *kda_v_conv_weight;
	const void *kda_decay_up_weight;
	const float *kda_decay_bias;
	const float *kda_head_log_scale;
	const void *kda_gate_weight;
	const float *kda_out_norm_weight;
	const void *kda_out_weight;
	const void *kda_out_scale;

	const void *mla_q_down_weight;
	const void *mla_q_down_scale;
	const void *mla_q_norm_weight;
	const void *mla_q_up_weight;
	const void *mla_q_up_scale;
	const void *mla_kv_a_weight;
	const void *mla_kv_a_scale;
	const void *mla_kv_a_norm_weight;
	const void *mla_kv_b_value_weight;
	const void *mla_kv_b_scale;
	const void *mla_gate_weight;
	const void *mla_out_weight;
	const void *mla_out_scale;

	const void *router_weight;
	const void *routed_down_weight;
	const void *routed_down_scale;
	const void *routed_up_weight;
	const void *routed_up_scale;
	const void *routed_norm_weight;
	const void *expert_w1_weight;
	const void *expert_w2_weight;
	// PACK V2 INTERLEAVES THE EXPERT SCALES INTO THE WEIGHT STREAM
	// (mxfp4_ws_interleaved_v1), so the scale fields are gone and the two
	// weight pointers address the 17-row-cell grid. A V2 pack ALWAYS
	// interleaves, so the loader sets the flag; the MoE layer refuses it
	// until the grouped GEMM learns the cell - the refusal and the
	// kernels-wave contract that lifts it are in K3LayerLatentMoe. The zero
	// path exists for the host recorders, which bind no weights at all.
	uint32_t expert_interleave;
	/* the pack's interleave k-tile (128 or 32 elements); the MoE launches
	 * pick the matching INTERLEAVED_B GEMM instantiation */
	uint32_t expert_tile_k;
	const void *shared_w1_weight;
	const void *shared_w1_scale;
	const void *shared_w2_weight;
	const void *shared_w2_scale;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;
	const void *attnres_attn_weight;
	const void *attnres_mlp_weight;
};

// The per-model runtime state a slice walks: one recurrent slot per KDA layer
// per sequence, one convolution window pool per projection per KDA layer, one
// paged cache per MLA layer. These are LAYER-indexed pools and the layer's
// buffers hold CURRENT-layer pointers, so the slice must advance them - a loop
// that reuses one pointer runs every KDA layer against one state and every MLA
// layer against one cache, which decodes fluently and remembers one layer.
struct K3SliceState
{
	uint8_t *kda_state;
	uint16_t *kda_q_window;
	uint16_t *kda_k_window;
	uint16_t *kda_v_window;
	const LmKvView *mla_cache;
	// DSpark verify slabs, one per KDA layer, each sequences * verify_rows rows
	// wide. The convolution slabs preserve pre-convolution BF16 inputs. The two
	// FP32 slabs preserve the exact transformed retention and write-gate values
	// consumed during verification; accepted-prefix folding never recomputes them.
	// Null pools disable replay storage outside a verify step. verify_rows is the
	// allocator's per-sequence ceiling (draft block plus bonus row).
	uint16_t *replay_conv_q;
	uint16_t *replay_conv_k;
	uint16_t *replay_conv_v;
	float *replay_retention;
	float *replay_write_gate;
	uint32_t verify_rows;
	// DSpark's drafter reads the target's hidden stream after the aux layers.
	// One slab of aux_rows * K3_HIDDEN per aux layer, filled by the slice as
	// the stream passes; null disables the capture, which is every step that
	// is not feeding a draft.
	uint16_t *dspark_aux;
	uint32_t aux_rows;
	// Every pool above is strided by this sequence capacity, which is the
	// allocator's number and not the batch's: rows vary per step, slots do not.
	uint32_t sequences;
	// THE ALLOCATOR'S HALF OF THE BF16 STATE OPTION. Nonzero strides the
	// state pool at K3_KDA_STATE_SLOT_BYTES_BF16 and hands the flag to the
	// layer, which still refuses it - the kernel's uint16_t State variant
	// exists and is host-gated, but no launch site selects it yet, and no
	// step can launch against the half-width pool before one does. The
	// numerics contract is in config.h.
	uint32_t kda_state_bf16;
	// THE TP4 ALL-REDUCE HOOK. The slice loop runs on the HOST, so the
	// serving tier can register one host callable per layer fired AFTER the
	// layer's MLP half: an input-dimension-sharded projection (routed_up,
	// and the head-sliced KDA/MLA out-projections) leaves its partial
	// rank-local, and the next layer's attention reads the running partial,
	// so the TP sum must land before the next retrieval - a post-slice
	// all-reduce is wrong by one layer's residual. Null skips it, which is
	// every single-rank contract (the host harnesses, the numerical gates,
	// and a PP-only deployment).
	/* phase 0 fires after the attention half (before the MLP-side
	 * retrieval, which reads the POST-attention partial), phase 1 after the
	 * MLP half. The hook all-reduces the phase's sharded outputs and folds
	 * the sums into the partial. */
	void (*layer_collective)(void *context, void *stream, uint32_t layer,
		uint32_t phase);
	void *collective_context;
};

// The two kind indices below partition the backbone, and the partition is
// asserted against the counts an allocator sizes pools with: 69 KDA slots and
// 24 cache views, nothing shared and nothing skipped. The largest KDA index is
// held by layer 90 - 91 and 92 are both MLA - and the largest MLA index by the
// trailing exception layer.
static_assert(K3_KDA_LAYER_COUNT + K3_MLA_LAYER_COUNT == K3_LAYERS,
	"every layer is exactly one of the two kinds");
static_assert((90u - (90u / 4u)) == K3_KDA_LAYER_COUNT - 1u,
	"the last KDA layer must land on the last KDA pool slot");
static_assert(((K3_LAYERS - 1u) / 4u) == K3_MLA_LAYER_COUNT - 1u,
	"the trailing MLA layer must land on the last cache view");

// Each assignment is a claim that two names mean the same tensor. A KDA layer
// leaves the MLA pointers null and the reverse, and which set is read is decided
// by K3_LAYER_KIND rather than by which pointers happen to be set - a missing
// tensor should fail loudly, not silently select the other path.
//
// mla_q_norm_weight and mla_kv_b_value_weight are here because the layer reads
// them; both existed in K3LayerBuffers and neither existed here, so every
// driver-run MLA layer would have normalised the query latent and projected the
// values through a null pointer. The layer harnesses set the buffers directly,
// which is why they passed.
static void K3BindLayer(const K3LayerWeights *weights, K3LayerBuffers *buffers)
{
	buffers->attn_norm_weight = weights->attn_norm_weight;
	buffers->mlp_norm_weight = weights->mlp_norm_weight;
	buffers->kda_qkv_beta_weight = weights->kda_qkv_beta_weight;
	buffers->kda_decay_down_weight = weights->kda_decay_down_weight;
	buffers->kda_q_conv_weight = weights->kda_q_conv_weight;
	buffers->kda_k_conv_weight = weights->kda_k_conv_weight;
	buffers->kda_v_conv_weight = weights->kda_v_conv_weight;
	buffers->kda_decay_up_weight = weights->kda_decay_up_weight;
	buffers->kda_decay_bias = weights->kda_decay_bias;
	buffers->kda_head_log_scale = weights->kda_head_log_scale;
	buffers->kda_gate_weight = weights->kda_gate_weight;
	buffers->kda_out_norm_weight = weights->kda_out_norm_weight;
	buffers->kda_out_weight = weights->kda_out_weight;
	buffers->kda_out_scale = weights->kda_out_scale;
	buffers->mla_q_down_weight = weights->mla_q_down_weight;
	buffers->mla_q_down_scale = weights->mla_q_down_scale;
	buffers->mla_q_norm_weight = weights->mla_q_norm_weight;
	buffers->mla_q_up_weight = weights->mla_q_up_weight;
	buffers->mla_q_up_scale = weights->mla_q_up_scale;
	buffers->mla_kv_a_weight = weights->mla_kv_a_weight;
	buffers->mla_kv_a_scale = weights->mla_kv_a_scale;
	buffers->mla_kv_a_norm_weight = weights->mla_kv_a_norm_weight;
	buffers->mla_kv_b_value_weight = weights->mla_kv_b_value_weight;
	buffers->mla_kv_b_scale = weights->mla_kv_b_scale;
	buffers->mla_gate_weight = weights->mla_gate_weight;
	buffers->mla_out_weight = weights->mla_out_weight;
	buffers->mla_out_scale = weights->mla_out_scale;
	buffers->router_weight = weights->router_weight;
	buffers->routed_down_weight = weights->routed_down_weight;
	buffers->routed_down_scale = weights->routed_down_scale;
	buffers->routed_up_weight = weights->routed_up_weight;
	buffers->routed_up_scale = weights->routed_up_scale;
	buffers->routed_norm_weight = weights->routed_norm_weight;
	buffers->expert_w1_weight = weights->expert_w1_weight;
	buffers->expert_w2_weight = weights->expert_w2_weight;
	buffers->expert_interleave = weights->expert_interleave;
	buffers->expert_tile_k = weights->expert_tile_k;
	buffers->shared_w1_weight = weights->shared_w1_weight;
	buffers->shared_w1_scale = weights->shared_w1_scale;
	buffers->shared_w2_weight = weights->shared_w2_weight;
	buffers->shared_w2_scale = weights->shared_w2_scale;
	buffers->dense_gate_up_weight = weights->dense_gate_up_weight;
	buffers->dense_gate_up_scale = weights->dense_gate_up_scale;
	buffers->dense_down_weight = weights->dense_down_weight;
	buffers->dense_down_scale = weights->dense_down_scale;
	buffers->attnres_attn_weight = weights->attnres_attn_weight;
	buffers->attnres_mlp_weight = weights->attnres_mlp_weight;
}

// Point the current-layer state pointers at this layer's slice of the pools.
//
// The kind index arithmetic: MLA layers sit at 3, 7, ..., 91 and 92, so the
// count of MLA layers strictly before layer l is l / 4 for every l in the
// backbone - including 92, whose 23 predecessors are the periodic ones. The
// KDA index is l minus that. For a layer of the other kind the unused index is
// still well-defined and its pools are simply not read.
static void K3BindLayerState(const K3SliceState *state, uint32_t layer, K3LayerBuffers *buffers)
{
	uint32_t mla_index = layer / 4u;
	uint32_t kda_index = layer - mla_index;
	uint64_t sequences = state->sequences;
	uint64_t slot_bytes = state->kda_state_bf16 != 0u
		? (uint64_t)K3_KDA_STATE_SLOT_BYTES_BF16 : (uint64_t)K3_KDA_STATE_SLOT_BYTES;
	buffers->kda_state_bf16 = state->kda_state_bf16;
	buffers->kda_state_pool = state->kda_state
		+ ((uint64_t)kda_index * sequences * slot_bytes);
	buffers->kda_q_window = state->kda_q_window
		+ ((uint64_t)kda_index * sequences * K3_KDA_QK_DIM * K3_KDA_CONV_KERNEL);
	buffers->kda_k_window = state->kda_k_window
		+ ((uint64_t)kda_index * sequences * K3_KDA_QK_DIM * K3_KDA_CONV_KERNEL);
	buffers->kda_v_window = state->kda_v_window
		+ ((uint64_t)kda_index * sequences * K3_KDA_V_DIM * K3_KDA_CONV_KERNEL);
	buffers->replay_conv_q = state->replay_conv_q == 0 ? 0 : state->replay_conv_q
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_QK_DIM);
	buffers->replay_conv_k = state->replay_conv_k == 0 ? 0 : state->replay_conv_k
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_QK_DIM);
	buffers->replay_conv_v = state->replay_conv_v == 0 ? 0 : state->replay_conv_v
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_V_DIM);
	buffers->replay_retention = state->replay_retention == 0 ? 0 : state->replay_retention
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_QK_DIM);
	buffers->replay_write_gate = state->replay_write_gate == 0 ? 0 : state->replay_write_gate
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_HEADS);
	if ( K3_LAYER_KIND(layer) == LM_LAYER_LATENT )
		buffers->cache = state->mla_cache[mla_index];
}

// THE DEFAULT RETURNS AN ERROR. A sixth kind added to LmLayerKind without an arm
// here stops the model instead of running the wrong one, and the compiler warns
// about the unhandled enum value before that.
template<class Format, class Geometry>
static int32_t K3LaunchAttentionHalf(const K3LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint32_t sequences, uint32_t commit, uint16_t *partial_accumulate, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	enum LmLayerKind kind = (enum LmLayerKind)K3_LAYER_KIND(layer);
	switch (kind)
	{
	case LM_LAYER_RECURRENT:
		return(K3LayerKda<Format>(buffers,rows,sequences,commit,partial_accumulate,multiprocessors,stream));
	case LM_LAYER_LATENT:
		return(K3LayerMla<Format,Geometry>(buffers,rows,context,partial_accumulate,multiprocessors,stream));
	case LM_LAYER_FULL:
	case LM_LAYER_WINDOW:
	case LM_LAYER_SPARSE:
	case LM_LAYER_COMPRESSED:
	case LM_LAYER_KIND_COUNT:
	default:
		return(LM_LAUNCH_ERR_SHAPE);
	}
}

// One stage slice: the layers this rank owns, in order.
//
// The kind comes from the ABSOLUTE layer index, and for this model that matters
// twice over. K3's period is four, which no rank count in use divides, so a rank
// starting mid-period would run the wrong attention on every layer it owns. And
// the last layer is an exception the formula alone does not produce - the
// backbone always ends on global attention - so a rank holding the tail must
// know its absolute position to get that one right.
//
// ENTRY CONTRACT: hidden_bf16 holds the incoming stream. For a slice starting
// at layer 0 that is the token embedding; the loop banks it as b_0 itself, so
// there is no separate host step to forget. For a slice starting later, the
// bank and the partial arrive from the previous stage - carrying them is the
// open transport question docs/MODEL_SUPPORT.md item 7 tracks, and this loop
// is deliberately correct for the single-stage case first.
//
// CUDA-GRAPH CAPTURABILITY, AUDITED 2026-08-01 (roadmap D10/D1: ~3,300
// launches a K3 token, and this loop is the capture unit). The findings:
//
//   * NO HOST-DEVICE TRAFFIC. No cudaMemcpy, no allocation, no synchronise
//     anywhere on the layer path - the new-gate list in
//     tests/test_k3_driver_contracts.py keeps it that way. Routing is packed
//     on device (LmRouteBuild) precisely so the hot path never reads a count
//     back.
//   * HOST-RESOLVED BRANCHES BAKE IN, AND THAT IS FINE. The replay-pointer
//     tests in K3LayerKda, the boundary/source-count arithmetic below, and
//     the dspark_aux null check all resolve at capture time. They are
//     functions of the step SHAPE - verify versus committed, drafting versus
//     not - not of data, so each shape captures its own graph and the
//     planner's repeating step shapes (the property D10 was designed around)
//     are what make the set finite.
//   * LAUNCH GEOMETRY IS HOST-SCALAR OR DEVICE-BOUND, NEVER HOST-DATA. Grids
//     derive from rows/sequences/packed_rows, fixed per captured shape; the
//     grouped GEMM's persistent grid is sized by multiprocessors while its
//     tile loop bounds live in the device-side tile prefix, so route skew
//     changes work distribution inside a replay, not the launch sequence.
//   * THE SHARED-MEMORY OPT-INS ARE CACHED. K3DeltaRuleOptIn and the GEMM's
//     equivalent call cudaFuncSetAttribute once per (kernel, device) behind
//     a granted-mask; a warm step must precede capture so no attribute call
//     is ever recorded, and steady-state replay records none.
//   * cudaPeekAtLastError is thread-local host state and capture-legal.
//
// What still cannot be captured here: nothing in this file. The remaining
// launch-tax item is upstream - the per-launch tensor-map encodes (roadmap
// D2, runtime/gemm.cuh, not this area) are cache hits in steady state but
// still host work between launches.
template<class Format, class Geometry>
static int32_t K3LaunchSlice(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset,layer,boundary;
	int32_t status;
	for (offset = 0u; offset < layer_count; ++offset)
	{
		layer = first_layer + offset;
		if ( layer >= K3_LAYERS )
			return(LM_LAUNCH_ERR_SHAPE);
		K3BindLayer(&weights[offset],buffers);
		K3BindLayerState(state,layer,buffers);
		boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u ? 1u : 0u;
		// Attention-side retrieval, over the bank BEFORE any append. Its source
		// count is one behind the MLP side's on a boundary layer, which is what
		// stops the vector being appended from also scoring as the partial.
		if ( layer > 0u )
			K3AttnRes(buffers,buffers->attnres_attn_weight,
				((layer - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		if ( boundary != 0u )
		{
			// The stream at a boundary becomes a bank entry. At layer 0 the
			// stream IS the incoming hidden - the token embedding - so b_0 is
			// banked here rather than by an undocumented host step.
			if ( layer == 0u )
				K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
			K3BankStore(buffers,layer / K3_ATTNRES_BLOCK_SIZE,rows,stream);
		}
		// The partial RESTARTS from the attention output on a boundary - the
		// reference sets prefix_sum to None at the append and to `a` after
		// the attention - and ACCUMULATES it otherwise, which the module's
		// own out-projection now does in its epilogue: the half receives
		// the partial as its accumulate target except at a boundary, where
		// the restart still needs the explicit set.
		status = K3LaunchAttentionHalf<Format,Geometry>(buffers,layer,rows,sequences,commit,
			boundary != 0u ? (uint16_t *)0 : buffers->attnres_partial_bf16,context,
			multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		/* the sharded path defers the restart to the phase-0 hook, whose
		 * completion sets the SUMMED attention output into the partial */
		if ( boundary != 0u && buffers->tp_sharded == 0u )
			K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
		// The sharded attention output all-reduces and folds BEFORE the
		// MLP-side retrieval: the retrieval's partial mix must contain the
		// post-attention contribution on every rank.
		if ( state->layer_collective != 0 )
			state->layer_collective(state->collective_context,stream,layer,0u);
		// MLP-side retrieval, over the post-append bank and the post-attention
		// partial. It runs at layer 0 as well: b_0 is in the bank by then.
		K3AttnRes(buffers,buffers->attnres_mlp_weight,
			(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		// first_k_dense_replace is 1: exactly one full-width feed-forward layer
		// before the MoE stack. Running LatentMoE on layer 0 would use the
		// routed intermediate of 3072 where the model has 33792.
		if ( layer < K3_FIRST_ROUTED_LAYER )
			status = K3LayerDenseMlp<Format>(buffers,rows,multiprocessors,stream);
		else
			status = K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		if ( state->layer_collective != 0 )
			state->layer_collective(state->collective_context,stream,layer,1u);
		// THE DRAFTER READS THE STREAM, AND THE PARTIAL IS THE STREAM. What
		// flows between blocks under AttnRes is the running partial - the next
		// block's first act is a retrieval that REPLACES its input - so the
		// hidden state SpecForge captured from the reference after an aux layer
		// is exactly this value, post both module adds.
		if ( state->dspark_aux != 0 )
		{
			static const uint32_t aux_ids[K3_DSPARK_AUX_LAYER_COUNT] = K3_DSPARK_AUX_LAYER_IDS_INITIALIZER;
			uint32_t aux;
			for (aux = 0u; aux < K3_DSPARK_AUX_LAYER_COUNT; ++aux)
				if ( aux_ids[aux] == layer )
					LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
						buffers->attnres_partial_bf16,state->dspark_aux + ((uint64_t)aux * state->aux_rows * K3_HIDDEN),rows,K3_HIDDEN);
		}
	}
	if ( first_layer + layer_count == K3_LAYERS )
	{
		// The model output is one more retrieval with the output weight - the
		// reference's _apply_output_attn_res - and THAT is what the head norms.
		// Without it the head reads the last layer's MLP output alone, which is
		// neither the stream nor the retrieval.
		K3AttnRes(buffers,buffers->attnres_out_weight,
			((K3_LAYERS - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		return(LM_LAUNCH_OK);
	}
	// A non-final slice hands the STREAM to the next stage, not the last module
	// output. The bank and the partial still have to travel beside it; until
	// that transport exists this loop is correct for a single stage only.
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
		K3_LAYER_THREADS, 0, stream,
		buffers->attnres_partial_bf16,buffers->hidden_bf16,rows,K3_HIDDEN);
	return(LM_LAUNCH_OK);
}

// THE SERIAL-TP HALF STEP. Runs ONE layer's attention half (phase 0) or MLP
// half (phase 1) in isolation, for the serial-TP replay
// (docs/serial_tp_replay.md): the harness replicates the FULL hidden (phase 0)
// and the FULL AttnRes partial (both phases) into the buffers before each half,
// runs the half on every rank in turn, and host-sums the rank partials between
// halves. The AttnRes bank and the recurrent KDA state live in the buffers and
// the slice state and persist across the per-half calls exactly as the
// whole-slice loop leaves them. phase 1 reuses the weight pointers phase 0
// bound (K3BindLayer) and the scratch buffers the dispatch carved once.
template<class Format, class Geometry>
static int32_t K3LaunchSliceHalf(const K3LayerWeights *weights, const K3SliceState *state,
	K3LayerBuffers *buffers, uint32_t layer, uint32_t phase, uint32_t rows,
	uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context,
	uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	uint32_t boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u ? 1u : 0u;
	if ( phase == 0u )
	{
		K3BindLayer(weights,buffers);
		K3BindLayerState(state,layer,buffers);
		// Attention-side retrieval, over the bank BEFORE any append.
		if ( layer > 0u )
			K3AttnRes(buffers,buffers->attnres_attn_weight,
				((layer - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		if ( boundary != 0u )
		{
			if ( layer == 0u )
				K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
			K3BankStore(buffers,layer / K3_ATTNRES_BLOCK_SIZE,rows,stream);
		}
		status = K3LaunchAttentionHalf<Format,Geometry>(buffers,layer,rows,sequences,commit,
			boundary != 0u ? (uint16_t *)0 : buffers->attnres_partial_bf16,context,
			multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		if ( boundary != 0u && buffers->tp_sharded == 0u )
			K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
		return(LM_LAUNCH_OK);
	}
	K3AttnRes(buffers,buffers->attnres_mlp_weight,
		(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
	if ( layer < K3_FIRST_ROUTED_LAYER )
		return(K3LayerDenseMlp<Format>(buffers,rows,multiprocessors,stream));
	return(K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream));
}

// Fold the accepted prefix of a verify step into the real state, with the
// kernels that would have committed it. Verify ran the slice with commit off
// and left each KDA layer's raw inputs in the replay slabs; acceptance fixes
// how much of each run was real, and this replays exactly that much - conv to
// advance the windows, L2 and the gate transforms to rebuild the delta's
// inputs, and the delta itself with commit on. Bit-identical to a committed
// run by the kda gate's own equivalence, because it IS a committed run.
//
// verify_row_begin is the slab-strided prefix (sequence s begins at
// s * verify_rows); accepted[s] says how many of its rows really happened.
// The step's scratch buffers carry the intermediates; their contents on entry
// do not matter and on exit are the fold's leavings.
template<class Format>
static int32_t K3FoldAccepted(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t sequences, const uint32_t *verify_row_begin, const uint32_t *accepted, uint32_t slab_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t layer;
	uint64_t replay_capacity;
	int32_t status;
	(void)multiprocessors;
	if ( weights == 0 || state == 0 || buffers == 0
		|| verify_row_begin == 0 || accepted == 0
		|| first_layer > K3_LAYERS || layer_count > K3_LAYERS - first_layer
		|| sequences == 0u || sequences > state->sequences
		|| state->verify_rows == 0u
		|| state->replay_conv_q == 0 || state->replay_conv_k == 0
		|| state->replay_conv_v == 0 || state->replay_retention == 0
		|| state->replay_write_gate == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	replay_capacity = (uint64_t)sequences * state->verify_rows;
	if ( slab_rows == 0u || (uint64_t)slab_rows > replay_capacity )
		return(LM_LAUNCH_ERR_SHAPE);
	// The fold holds the decode path's gate on the bf16 state option: decode,
	// verify and fold must convert at the same two points or the kda gate's
	// bit-equivalence between them dies, so the refusal is one contract, not
	// three independent ones.
	if ( state->kda_state_bf16 != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	status = K3DeltaRuleOptIn((uint32_t)(K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)));
	if ( status != LM_LAUNCH_OK )
		return(status);
	for (layer = first_layer; layer < first_layer + layer_count; ++layer)
	{
		if ( K3_LAYER_KIND(layer) != LM_LAYER_RECURRENT )
			continue;
		K3BindLayer(&weights[layer - first_layer],buffers);
		K3BindLayerState(state,layer,buffers);
		LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
			buffers->kda_q_window,buffers->kda_state_index,verify_row_begin,accepted,buffers->replay_conv_q,buffers->kda_q_conv_weight,buffers->query_bf16,K3_KDA_QK_DIM,sequences,1u);
		LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
			buffers->kda_k_window,buffers->kda_state_index,verify_row_begin,accepted,buffers->replay_conv_k,buffers->kda_k_conv_weight,buffers->key_bf16,K3_KDA_QK_DIM,sequences,1u);
		LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(K3_KDA_V_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
			buffers->kda_v_window,buffers->kda_state_index,verify_row_begin,accepted,buffers->replay_conv_v,buffers->kda_v_conv_weight,buffers->value_bf16,K3_KDA_V_DIM,sequences,1u);
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(slab_rows,K3_KDA_HEADS), K3_LAYER_THREADS, 0, stream,
			buffers->key_bf16,K3_KDA_HEADS,slab_rows,K3_RMS_EPSILON);
		LM_LAUNCH((LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM>), dim3(sequences,K3_KDA_HEADS), K3_LAYER_THREADS, (uint32_t)(K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)), stream,
			buffers->kda_state_pool,state->kda_state_bf16 != 0u ? K3_KDA_STATE_SLOT_BYTES_BF16 : K3_KDA_STATE_SLOT_BYTES,buffers->kda_state_index,verify_row_begin,accepted,buffers->query_bf16,buffers->key_bf16, buffers->value_bf16,buffers->replay_retention,buffers->replay_write_gate,buffers->attention_out_bf16, K3_KDA_HEADS,1u,sequences,1u);
	}
	return(LM_LAUNCH_OK);
}
