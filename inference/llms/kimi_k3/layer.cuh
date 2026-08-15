#pragma once
// Kimi K3, one layer.
//
// Three of every four layers are Kimi Delta Attention; the fourth is gated MLA,
// and the last layer of the backbone is MLA as well so the model always ends on
// global attention. Every attention layer is followed by a Stable LatentMoE.
//
// The arithmetic here is from the technical report and from FlashKDA's
// reference, not from the architecture's name. Where the two disagree the
// reference wins, because it is what the released kernel is validated against -
// that is how the missing dt_bias was found.
#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/llms/kimi_k3/config.h"
// The pack V2 fused-row constants. config.h and this header spell every shared
// constant identically on purpose (pipeline_sideband.h co-includes them), so
// the layer can read the pack's own fused geometry instead of re-deriving it -
// bind stays pointer arithmetic against the same numbers the packer emitted.
#include "inference/llms/kimi_k3/generated_config.h"

// THE MLA LATENT IS kv_lora_rank, NOT THE KDA HEAD DIM. This was
// LmKvLatent<..., K3_KDA_KEY_DIM, 64u, ...> - 128 elements, the width of a KDA
// head, standing in for a 512-element MLA latent. Two unrelated dimensions that
// both happen to be head-shaped, so nothing looked wrong and the pool came out
// four times too small.
//
// Same defect qwen_3_6 had one commit earlier, found the same way: the constant
// the geometry needed was not in config.h, so a nearby one was used.
using K3GlobalKv = LmKvLatent<K3_KV_BITS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM, K3_KV_PAGE_SLOTS>;

// Declared here rather than in unity.cu because K3LayerMla takes the geometry
// as a template argument, so every caller of a layer needs the alias. bind.cu
// did and could not see it - the same one-line failure qwen_3_6's driver hit,
// which is a sign the alias belongs beside the layer in every model.

#include "inference/llms/kimi_k3/launch_shape.h"

static_assert(K3_KDA_QK_L2NORM == 1u, "kda qk l2norm is part of the kernel contract");
static_assert(K3_KDA_A_LOG_SOURCE_HEADS == 128u && K3_KDA_HEADS == 96u, "A_log loads 128 heads and narrows to 96");

// KDA widths. 96 heads at 128 for q, k and v alike.
#define K3_KDA_QK_DIM (K3_KDA_HEADS * K3_KDA_KEY_DIM)
#define K3_KDA_V_DIM (K3_KDA_HEADS * K3_KDA_VALUE_DIM)

// THE PACK V2 SECTION TABLE, AS ROW OFFSETS INTO THE FUSED TENSORS
// (docs/K3_PACK_FORMAT_V2.md). q|k|v|beta ship head-major in one tensor,
// per-head widths 128/128/128/1; decay_down|gate_down ship replicated in the
// other. The split kernel indexes by these and the static asserts tie the
// table to K3_KDA_*_FUSED_ROWS - the constants the packer validates against -
// so a layout change fails here at compile time, not as misread rows.
#define K3_KDA_QKVB_K_OFFSET K3_KDA_QK_DIM
#define K3_KDA_QKVB_V_OFFSET (2u * K3_KDA_QK_DIM)
#define K3_KDA_QKVB_BETA_OFFSET (2u * K3_KDA_QK_DIM + K3_KDA_V_DIM)
#define K3_KDA_GATE_DOWN_OFFSET K3_KDA_KEY_DIM
static_assert(K3_KDA_QKVB_BETA_OFFSET + K3_KDA_HEADS == K3_KDA_QKVB_FUSED_ROWS,
	"the q|k|v|beta sections must tile the fused tensor exactly");
static_assert(K3_KDA_GATE_DOWN_OFFSET + K3_KDA_KEY_DIM == K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS,
	"the decay_down|gate_down sections must tile the fused tensor exactly");

// MLA widths, IN THE ABSORBED FORM THE KERNEL IMPLEMENTS.
//
// LmAttentionDecodeKernel reads LATENT + ROPE elements per head and treats the
// cached latent row as the key directly. It does not reconstruct per-head keys
// and values. glm5_2 has used it that way from the start: its query is
// heads * (LATENT + ROPE) and its output projection reads heads * LATENT.
//
// modeling_kimi_linear.py uses the RECONSTRUCTED form instead - it caches the
// compressed 512+64 row and rebuilds k_pass and value_states through kv_b_proj
// at attention time, so its query is heads * (qk_nope + qk_rope) = 96 * 192 and
// its o_proj reads heads * v_head_dim = 96 * 128.
//
// I sized this file from the modelling file and handed it to the absorbed
// kernel: a 192-element query into a loop that reads 576. The two forms are
// mathematically equal - absorption folds W_kv_b into the query up-projection
// and the value half into the output projection - but they need DIFFERENT
// WEIGHTS, and that is a pack-time transformation, not a runtime one.
//
// THE PACKER OWES ONE FOLD:
//
//   q_up      absorb the nope half of kv_b:  q_lora -> heads * 512, then
//             concatenate the 64 unrotated rows -> heads * 576
//
// The VALUE half is NOT absorbed into o_proj: the gate is elementwise in
// v-space and does not commute with the fold, which is why kv_b_value is its
// own per-head table and mla_out_weight is o_proj exactly as shipped.
// tools/k3_pack.py performs the one fold and the split.
#define K3_MLA_Q_DIM (K3_MLA_HEADS * (K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM))
#define K3_MLA_KV_A_DIM (K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM)
#define K3_MLA_KV_B_DIM (K3_MLA_HEADS * (K3_QK_NOPE_DIM + K3_V_HEAD_DIM))
// The attention output before kv_b_value brings it back to v-space, and after.
#define K3_MLA_LATENT_OUT_DIM (K3_MLA_HEADS * K3_KV_LORA_RANK)
#define K3_MLA_OUT_DIM (K3_MLA_HEADS * K3_V_HEAD_DIM)

// The kernel's contract, asserted rather than trusted: it loads LATENT + ROPE
// per head and this file must hand it exactly that.
static_assert(K3_MLA_Q_DIM == K3_MLA_HEADS * (K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM),
	"the query must be as wide as the kernel reads");
// The gate is applied in v-space, so its width is the reference's g_proj output
// and the checkpoint tensor is used unchanged.
static_assert(K3_MLA_OUT_DIM == K3_MLA_HEADS * K3_V_HEAD_DIM,
	"the gate and output projection live in v-space, not the latent");

// The shared experts run at the full width with the routed intermediate times
// the shared count, per the report's Ns = 2.
#define K3_SHARED_INTERMEDIATE (K3_EXPERT_INTERMEDIATE * K3_SHARED_EXPERTS)

static_assert(K3_KDA_HEADS == K3_MLA_HEADS,
	"the report gives one head count for both attention kinds");
static_assert(K3_LAYERS % 4u == 1u,
	"93 layers is 23 whole blocks plus the trailing MLA layer");

// EVERY K EXTENT, ASSERTED AT COMPILE TIME.
//
// K3Project factors the quantise-and-GEMM out of five call sites, which is the
// right shape for the code and hides the widths from tests/test_gemm_k_alignment.py -
// the call site passes a variable, so the gate reported "could not resolve"
// rather than passing silently. It was right to.
//
// These assertions are the stronger replacement: LmGemmKernel computes
// k_tiles = input_dimension / TILE_K with an integer division and drops any
// remainder, so a K extent that is not a whole number of tiles loses the tail of
// every dot product. INT7 tiles at 256 and is the format this model uses, so 256
// is the number that has to divide, not 128.
static_assert(K3_HIDDEN % 256u == 0u, "KDA and MLA project from the hidden");
// THE DECAY BOTTLENECK IS 128 WIDE, WHICH IS NARROWER THAN AN INT7 TILE.
// LmInt7 tiles K at 256, so k_tiles = 128 / 256 = 0 and the up-projection would
// compute NOTHING - every decay logit zero, every retention factor
// exp(g_min * sigmoid(bias)), a model that runs at a constant decay.
//
// I wrote "|| K3_KDA_KEY_DIM == 128u" here first to make the assertion pass.
// That is the escape hatch this whole branch has been finding in other people's
// code. The projection runs in BF16 instead, which tiles at 128 - and that
// matches the checkpoint, whose MXFP4 quantisation covers only the routed
// experts while attention projections stay in higher precision.
static_assert(K3_KDA_KEY_DIM % 128u == 0u,
	"the decay bottleneck must be a whole BF16 tile");
static_assert(K3_KDA_V_DIM % 256u == 0u, "the KDA output projection");
static_assert(K3_Q_LORA_RANK % 256u == 0u, "the MLA query up-projection");
static_assert(K3_MLA_OUT_DIM % 256u == 0u, "the MLA output projection");
static_assert(K3_ROUTED_EXPERT_HIDDEN % 256u == 0u, "the routed experts' input");
static_assert(K3_EXPERT_INTERMEDIATE % 256u == 0u, "the routed down-projection");
static_assert(K3_SHARED_INTERMEDIATE % 256u == 0u, "the shared down-projection");
static_assert(K3_DENSE_INTERMEDIATE % 256u == 0u, "layer 0's dense down-projection");

struct K3LayerBuffers
{
	const void *attn_norm_weight;
	const void *mlp_norm_weight;

	// KDA, PACK V2 (docs/K3_PACK_FORMAT_V2.md). The six projections that read
	// the normed input are TWO tensors: kda_qkv_beta_weight fuses q|k|v|beta
	// head-major (per-head widths 128/128/128/1, section offsets above), and
	// The released checkpoint ships a FULL-RANK output gate (g_proj), so the
	// decay|gate fusion and the low-rank gate pair are gone: kda_decay_down
	// is the standalone 128-wide replicated bottleneck, and kda_gate_weight
	// is the checkpoint's g_proj unchanged
	// (docs/K3_GATE_RECONCILIATION.md).
	const void *kda_qkv_beta_weight;
	const void *kda_decay_down_weight;
	const float *kda_q_conv_weight;
	const float *kda_k_conv_weight;
	const float *kda_v_conv_weight;
	const void *kda_decay_up_weight;
	const float *kda_decay_bias;
	const float *kda_head_log_scale;
	// The gate is the released checkpoint's full-rank g_proj: one projection
	// from the normed hidden to heads * head_dim, applied after the delta
	// rule's head-wise RMSNorm.
	const void *kda_gate_weight;
	const float *kda_out_norm_weight;
	const void *kda_out_weight;
	const void *kda_out_scale;

	// MLA.
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
	// The gate is the released checkpoint's full-rank g_proj, unchanged, at
	// heads * v_head_dim.
	const void *mla_gate_weight;
	const void *mla_out_weight;
	const void *mla_out_scale;

	// LatentMoE. The router reads the FULL hidden; only the experts are latent.
	const void *router_weight;
	const float *router_bias;
	float *router_logits;
	const void *routed_down_weight;
	const void *routed_down_scale;
	const void *routed_up_weight;
	const void *routed_up_scale;
	const void *routed_norm_weight;
	const void *expert_w1_weight;
	const void *expert_w2_weight;
	// PACK V2 INTERLEAVES THE EXPERT SCALES INTO THE WEIGHT STREAM
	// (mxfp4_ws_interleaved_v1): there are no expert_w*_scale tensors, and the
	// two pointers above address the interleaved 17-row-cell grid. Nonzero
	// marks exactly that, and K3LayerLatentMoe refuses it: LmScaleTensor
	// models a separately-strided scale PLANE and has no encoding for a scale
	// row co-tiled with payload (inference/kernels/scale.cuh), so launching
	// would read scale bytes as payload. The kernels wave that teaches the
	// grouped GEMM the cell lifts the refusal; the contract is at the check.
	uint32_t expert_interleave;
	const void *shared_w1_weight;
	const void *shared_w1_scale;
	const void *shared_w2_weight;
	const void *shared_w2_scale;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;

	uint16_t *hidden_bf16;
	uint16_t *normed_bf16;
	// THE WIDE FUSED-GEMM SCRATCHES. The two wide GEMMs land here at the pack's
	// fused widths - rows x K3_KDA_QKVB_FUSED_ROWS and rows x
	// K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS - and the split kernel copies the
	// sections out dense, because every consumer (the convolutions, the
	// up-projections, the sigmoid) reads dense rows and a strided view would
	// make each of them carry the fused pitch. gate_latent_bf16 holds the gate
	// down-projection's half across the delta rule until gate_up reads it;
	// latent_bf16 alone cannot, because the decay half already lives there.
	uint16_t *fused_qkvb_bf16;
	uint16_t *fused_decay_gate_bf16;
	uint16_t *gate_latent_bf16;
	uint16_t *query_bf16;
	uint16_t *key_bf16;
	uint16_t *value_bf16;
	uint16_t *gate_bf16;
	uint16_t *decay_logit_bf16;
	uint16_t *latent_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_out_bf16;
	uint16_t *shared_out_bf16;
	// AttnRes. The bank holds one representation per completed block; the
	// partial is the running sum of the block in progress. b_0 is the token
	// embedding, so the bank is never empty after layer 0.
	uint16_t *attnres_bank_bf16;
	uint16_t *attnres_partial_bf16;
	const void *attnres_attn_weight;
	const void *attnres_mlp_weight;
	// Model-level, not per-layer: the reference's _apply_output_attn_res runs
	// once after the last layer, with its own fused pseudo-query, and its output
	// is what the final norm and the head consume. Per-layer weights arrive
	// through K3BindLayer; this one lives here because there is exactly one.
	const void *attnres_out_weight;
	uint16_t *kda_beta_logit;
	float *kda_write_gate_out;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	// Rows sorted by sequence, sequence_row_begin a prefix of length
	// sequences + 1: sequence s owns rows [begin[s], begin[s+1]). Null means
	// identity - row i is sequence i - which is every pure-decode step. The
	// KDA state index is per SEQUENCE under this contract, not per row.
	const uint32_t *sequence_row_begin;
	// DSpark verify keeps the exact recurrent inputs needed to commit an
	// accepted prefix: pre-convolution q/k/v rows plus the already transformed
	// retention and write-gate values. The fold must not recompute either gate;
	// approximate exponentials can drift even when two formulas look identical.
	// Null pointers disable replay storage outside verification. Storing pre-conv
	// q/k/v remains necessary because the three convolution windows are state.
	uint16_t *replay_conv_q;
	uint16_t *replay_conv_k;
	uint16_t *replay_conv_v;
	float *replay_retention;
	float *replay_write_gate;

	// The recurrent half. Fixed per sequence, never grows with context.
	uint8_t *kda_state_pool;
	// BF16 STATE, DEFAULT OFF, AND IT FAILS CLOSED HERE. Nonzero asks for the
	// half-width slot of config.h's K3_KDA_STATE_SLOT_BYTES_BF16 - the batch
	// lever the roadmap prices at half of 909 MB per sequence per token. The
	// bind strides the pool by the flag so the plumbing is honest, but the
	// layer still refuses to launch on it: the default LmDeltaRuleKernel
	// instantiation addresses the slot as float, so running against a bf16
	// pool would mis-stride every head and every sequence - a silent
	// corruption, the worst kind.
	//
	// The kernel half of the gate has lifted, in inference/kernels/
	// linear_attn.cuh and nowhere else: the uint16_t State instantiation
	// loads the slot bf16 -> fp32 into the same shared tile, runs the
	// recurrence in fp32 EXACTLY as the fp32 instantiation, and converts
	// back only on the commit store. Decode, verify and the replay fold
	// convert at the same two points or the kda gate's bit-equivalence
	// between them dies - and verify (commit == 0) never stores, so it
	// stays dtype-neutral. What remains before this flag may launch is the
	// launch-site selection (below and in slice.cuh's fold) plus on-device
	// numerics receipts; tests/test_kda_bf16_state.py gates the kernel-side
	// contract on the host shim. The numerics contract this option signs is
	// in config.h at K3_KDA_STATE_SLOT_BYTES_BF16; admission-time only,
	// never default.
	uint32_t kda_state_bf16;
	uint16_t *kda_q_window;
	uint16_t *kda_k_window;
	uint16_t *kda_v_window;
	const uint32_t *kda_state_index;
	float *kda_retention;

	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint32_t *route_expert;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
};

// One projection: quantise, GEMM, done. The KDA path does this five times and
// writing it out five times is how a scale pointer comes to describe the wrong
// buffer.
template<class Format>
static int32_t K3Project(const K3LayerBuffers *b, const uint16_t *source, const void *weight, const void *weight_scale, uint16_t *destination, uint16_t *accumulate, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	LmGemmArguments gemm;
	// EVERY PROJECTION THROUGH HERE IS UNQUANTISED. The checkpoint's ignore
	// list keeps attention, latent projections, shared experts, routers and the
	// head out of the 4-bit grid, and activations are never quantised at all -
	// input_activations is null. A Format with a scale group reaching this
	// helper is a recipe violation, caught at compile time rather than by a
	// packed buffer nothing filled.
	static_assert(Format::kScaleGroup == 0u,
		"K3Project carries the unquantised projections; experts go weight-only");
	if (weight_scale != 0)
		return(LM_LAUNCH_ERR_SHAPE);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = destination;
	gemm.accumulate_bf16 = accumulate;
	return(LmGemmLaunch<Format,K3_LAYER_TILE_N,Format::kTileK,K3_LAYER_STAGES,K3_LAYER_WARPS>(
		&gemm,source,weight,rows,rows,1u,1u,
		input_dimension,output_dimension,multiprocessors,false,stream));
}

// The common case: the result has one home. The accumulate tail exists for
// the module-output projections, whose result ALSO folds into the AttnRes
// partial in the epilogue - the separate add kernel and its full-width
// re-read never happen.
template<class Format>
static int32_t K3Project(const K3LayerBuffers *b, const uint16_t *source, const void *weight, const void *weight_scale, uint16_t *destination, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	return(K3Project<Format>(b,source,weight,weight_scale,destination,(uint16_t *)0,rows,input_dimension,output_dimension,multiprocessors,stream));
}

// EVERY PROJECTION HERE IS BF16 AND ONLY THE ROUTED EXPERTS TAKE Format.
// The checkpoint's ignore list is attention, latent projections, shared experts,
// routers and lm_head - none of them saw quantisation-aware training, so none of
// them is safe in a 4-bit grid. Format reaches exactly the two expert GEMMs.
//
// AttnRes: replace the stream with a retrieval over the bank.
//
// Report eq. 8-10. The candidates are the completed block representations plus
// the running partial sum, the score is a dot with the layer's fused
// pseudo-query, and the mix is a softmax over them.
//
// RUN TWICE PER LAYER, WITH DIFFERENT WEIGHTS. The reference applies it before
// attention with self_attention_res_proj and before the MLP with mlp_res_proj.
// One call with one weight would run and would be a different model - the two
// retrievals are asking different questions of the same bank.
static void K3AttnRes(const K3LayerBuffers *b, const void *score_weight, uint32_t sources, uint32_t rows, cudaStream_t stream)
{
	// The caller counts the candidates, because the two retrievals in a layer
	// disagree at block boundaries: the attention side runs BEFORE the append
	// and sees the old bank, the MLP side runs after and sees the new entry.
	// One formula here served both and double-counted at every boundary layer -
	// worse, the partial being appended was also the last candidate, so the
	// same vector scored twice.
	if ( sources > K3_ATTNRES_MAX_SOURCES )
		sources = K3_ATTNRES_MAX_SOURCES;
	LM_LAUNCH((LmAttnResKernel<K3_LAYER_THREADS,K3_ATTNRES_MAX_SOURCES>),
		rows, K3_LAYER_THREADS, 0, stream,
		b->attnres_bank_bf16,b->attnres_partial_bf16,
		(const uint16_t *)score_weight,b->hidden_bf16,sources,rows,K3_HIDDEN,
		K3_RMS_EPSILON);
}

// The pieces the driver sequences the partial with. Three verbs, because the
// reference uses exactly three: the partial RESTARTS from the attention output
// on a block-boundary layer (prefix_sum = None, then = a), ACCUMULATES a module
// output otherwise (prefix_sum += a, += m), and is BANKED when a block closes.
// Reusing an add as a copy banks twice the value; reusing an add as a restart
// carries the previous block's sum into the next - both run and both are a
// different model.
static void K3PartialSet(const K3LayerBuffers *b, const uint16_t *value, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
		K3_LAYER_THREADS, 0, stream,
		value,b->attnres_partial_bf16,rows,K3_HIDDEN);
}

static void K3PartialAdd(const K3LayerBuffers *b, const uint16_t *value, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
		K3_LAYER_THREADS, 0, stream,
		b->attnres_partial_bf16,value,b->attnres_partial_bf16,rows,K3_HIDDEN);
}

static void K3BankStore(const K3LayerBuffers *b, uint32_t slot, uint32_t rows, cudaStream_t stream)
{
	// [source][row][hidden], the layout LmAttnResKernel reads. Slot 0 is the
	// embedding, banked by the driver at layer 0 from the incoming hidden.
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
		K3_LAYER_THREADS, 0, stream,
		b->attnres_partial_bf16,
		b->attnres_bank_bf16 + ((uint64_t)slot * rows * K3_HIDDEN),
		rows,K3_HIDDEN);
}

// THE DELTA RULE'S 64 KiB OF DYNAMIC SHARED IS PAST THE 48 KiB DEFAULT.
// ptxas grants 48 KiB of dynamic shared without asking; the delta rule carves
// KEY_DIM * VALUE_DIM floats, which is 64 KiB, so the launch fails on device
// every time unless cudaFuncSetAttribute opts the kernel in first. The opt-in
// itself is LmKernelSharedMemoryOptIn in runtime/launch.h, shared with every
// other family that launches this kernel; this wrapper only names the
// instantiation.
static int32_t K3DeltaRuleOptIn(uint32_t shared_bytes)
{
	return(LmKernelSharedMemoryOptIn(
		(const void *)LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM>,
		shared_bytes));
}

// THE PACK-V2 SECTION SPLIT, ONE LAUNCH FOR BOTH FUSED TENSORS. The two wide
// GEMMs land at the pack's fused widths; this copies each section out dense -
// q, k, v and beta from the qkvb rows, the decay and gate bottlenecks from the
// decay|gate rows. It lives in this file, not project.cuh, because the section
// table is K3's pack contract: LmSplitQkvKernel strides its source row by
// query + key + value, so a row with a beta tail mis-strides every row after
// the first, and nothing else in the tree splits two tensors at once. The
// kernels wave can promote it the day a second model packs this way.
//
// A copy, not a view - the reason qwen_3_6's split is a copy: every consumer
// (the three convolutions, the two up-projections, the sigmoid) reads dense
// rows, and a view would make each of them carry the fused pitch. And ONE
// launch, not one per tensor: launch tax is the B1 cost (roadmap D10/D1), and
// both sources are ready the moment the second GEMM retires.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void K3SplitFusedProjectionsKernel(const uint16_t *__restrict__ qkvb_bf16, uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ key_bf16, uint16_t *__restrict__ value_bf16, uint16_t *__restrict__ beta_bf16, uint32_t rows)
{
	uint32_t row = blockIdx.x,index;
	uint64_t fused = (uint64_t)row * K3_KDA_QKVB_FUSED_ROWS;
	uint64_t dense = (uint64_t)row * K3_KDA_QK_DIM;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < K3_KDA_QK_DIM; index += THREADS)
		query_bf16[dense + index] = qkvb_bf16[fused + index];
	for (index = threadIdx.x; index < K3_KDA_QK_DIM; index += THREADS)
		key_bf16[dense + index] = qkvb_bf16[fused + K3_KDA_QKVB_K_OFFSET + index];
	for (index = threadIdx.x; index < K3_KDA_V_DIM; index += THREADS)
		value_bf16[((uint64_t)row * K3_KDA_V_DIM) + index] =
			qkvb_bf16[fused + K3_KDA_QKVB_V_OFFSET + index];
	// Beta is one element per head: 96 columns at the tail of the fused row.
	for (index = threadIdx.x; index < K3_KDA_HEADS; index += THREADS)
		beta_bf16[((uint64_t)row * K3_KDA_HEADS) + index] =
			qkvb_bf16[fused + K3_KDA_QKVB_BETA_OFFSET + index];
}

// Kimi Delta Attention, 69 of 93 layers.
//
// Report eq. 1-2 and 5-6, with FlashKDA's ordering where the report is silent:
//
//     q, k = L2Norm(Swish(ShortConv(W x)))
//     v    = Swish(ShortConv(W x))
//     z    = W_up(W_down(x))                     low rank, per key channel
//     a    = exp(g_min * sigmoid(exp(A_h) * (z + b)))
//     S    = (I - beta k k^T) Diag(a) S + beta k v^T
//     y    = W_o[ sigmoid(W_g x) * RMSNorm(S^T q) ]
template<class Format>
static int32_t K3LayerKda(const K3LayerBuffers *b, uint32_t rows, uint32_t sequences, uint32_t commit, uint16_t *partial_accumulate, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	uint32_t state_slot_bytes;
	// THE BF16 STATE OPTION IS REFUSED HERE, DELIBERATELY, FOR NOW. The kernel
	// variant exists: LmDeltaRuleKernel and LmReplayFoldKernel take the pool's
	// element type as a template parameter, and tests/test_kda_bf16_state.py
	// gates its per-step rounding and fold byte-exactness. What is left before
	// this check may lift is the launch-site selection - instantiating and
	// dispatching the uint16_t kernel from this flag, in this file and in
	// slice.cuh's fold launch - plus the on-device numerics receipts the
	// host gate only simulates. An error here stays the loud failure until
	// then: launching would alias half-width slots into a full-width reader.
	if ( b->kda_state_bf16 != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	state_slot_bytes = b->kda_state_bf16 != 0u
		? K3_KDA_STATE_SLOT_BYTES_BF16 : K3_KDA_STATE_SLOT_BYTES;
	// THE INPUT IS THE RETRIEVAL, ALONE. Under AttnRes there is no residual
	// stream to fold in: the reference computes input_layernorm(h) where h is
	// what the retrieval produced (or the raw stream at layer 0), and the
	// stream itself advances only by the module-output adds the driver makes.
	// Folding a residual here normed stream-plus-retrieval, a different model.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	// K3-PERF-003, LANDED (pack V2). What stood here: six BF16 GEMM launches
	// all reading normed_bf16, unfusable because the pack scattered the six
	// weights and the TP tables gave them two shard classes. Pack V2 ships
	// exactly the two tensors the fix wanted - q|k|v|beta fused OUTPUT_DIM_HEADS
	// and decay_down|gate_down fused REPLICATED - so the block is now TWO wide
	// GEMMs over one activation read each, plus one section split: four GEMM
	// launches and four full-width activation reads are gone per KDA layer
	// (roadmap D1 counts the launches; the roofline counts the reads). Both
	// GEMMs run back to back precisely because they share the activation.
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_qkv_beta_weight,0,
		b->fused_qkvb_bf16,rows,K3_HIDDEN,K3_KDA_QKVB_FUSED_ROWS,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_decay_down_weight,0,
		b->latent_bf16,rows,K3_HIDDEN,K3_KDA_KEY_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// The full-rank gate runs back to back with the decay because both read
	// normed_bf16: one activation read serves both GEMMs, and the gate output
	// waits in gate_bf16 until the delta rule finishes.
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_gate_weight,0,
		b->gate_bf16,rows,K3_HIDDEN,K3_KDA_V_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((K3SplitFusedProjectionsKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->fused_qkvb_bf16,
		b->query_bf16,b->key_bf16,b->value_bf16,(uint16_t *)b->kda_beta_logit,rows);
	// THE VERIFY STEP KEEPS ITS RAW INPUTS. The convolutions below overwrite
	// q, k and v in place, so this is the last moment the pre-conv rows exist -
	// and they are exactly what a fold needs to advance the windows and the
	// delta state over whatever prefix the sampler accepts.
	if ( b->replay_conv_q != 0 )
	{
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), dim3((K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
			b->query_bf16,b->replay_conv_q,rows,K3_KDA_QK_DIM);
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), dim3((K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
			b->key_bf16,b->replay_conv_k,rows,K3_KDA_QK_DIM);
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), dim3((K3_KDA_V_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
			b->value_bf16,b->replay_conv_v,rows,K3_KDA_V_DIM);
	}
	// THREE CONVOLUTIONS, NOT ONE, EACH WITH ITS OWN WINDOW. q, k and v are
	// separate ShortConvolution modules in the reference; sharing a window
	// between them would mix three token streams into one and still run.
	LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_q_window,b->kda_state_index,b->sequence_row_begin,0,b->query_bf16,b->kda_q_conv_weight,b->query_bf16,K3_KDA_QK_DIM,sequences,commit);
	LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_k_window,b->kda_state_index,b->sequence_row_begin,0,b->key_bf16,b->kda_k_conv_weight,b->key_bf16,K3_KDA_QK_DIM,sequences,commit);
	LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(K3_KDA_V_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_v_window,b->kda_state_index,b->sequence_row_begin,0,b->value_bf16,b->kda_v_conv_weight,b->value_bf16,K3_KDA_V_DIM,sequences,commit);
	// q and k only. The value is not normalised.
	LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(rows,K3_KDA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->query_bf16,K3_KDA_HEADS,rows,K3_RMS_EPSILON);
	LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(rows,K3_KDA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->key_bf16,K3_KDA_HEADS,rows,K3_RMS_EPSILON);
	// The decay logit is low rank: hidden -> head_dim -> heads * head_dim. The
	// down half already ran inside the fused decay|gate GEMM and the split
	// left its output in latent_bf16. BF16 ON BOTH HALVES OF THE DECAY
	// PROJECTION. The bottleneck is 128 wide and an INT7 tile is 256 deep, so
	// the up-projection under Format would compute zero tiles and emit
	// nothing. See the static_assert above.
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->kda_decay_up_weight,0,
		b->decay_logit_bf16,rows,K3_KDA_KEY_DIM,K3_KDA_QK_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	float *retention = b->replay_retention != 0
		? b->replay_retention : b->kda_retention;
	LM_LAUNCH((LmBoundedDecayKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(rows,K3_KDA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->decay_logit_bf16,b->kda_decay_bias,b->kda_head_log_scale,retention,K3_KDA_HEADS,K3_KDA_GATE_LOWER_BOUND,rows);
	// BETA COMES FROM THE SPLIT NOW. It was read raw from kda_write_gate,
	// which nothing filled - the comment said "still on the host" and no host
	// exists. The reference is Sigmoid(W_beta x), per head, with the sigmoid
	// inside the kernel (use_beta_sigmoid_in_kernel); the W_beta half is the
	// 1-wide-per-head section of the fused qkv|beta GEMM, BF16 for the same
	// reason as the decay: 96 outputs is narrower than an INT7 tile.
	float *write_gate = b->replay_write_gate != 0
		? b->replay_write_gate : b->kda_write_gate_out;
	LM_LAUNCH((LmSigmoidRowsKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		(const uint16_t *)b->kda_beta_logit,write_gate,K3_KDA_HEADS);
	status = K3DeltaRuleOptIn((uint32_t)(K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)));
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM>), dim3(sequences,K3_KDA_HEADS), K3_LAYER_THREADS, (uint32_t)(K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)), stream,
		b->kda_state_pool,state_slot_bytes,b->kda_state_index,b->sequence_row_begin,0,b->query_bf16,b->key_bf16, b->value_bf16,retention,write_gate,b->attention_out_bf16, K3_KDA_HEADS,1u,sequences,commit);
	// RMSNORM BEFORE THE GATE, AND ONLY HERE. Report eq. 6 normalises the
	// recurrent output head-wise before gating; eq. 7 gates the MLA output with
	// no normalisation at all. The two paths differ in exactly this step.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,float>), dim3(rows * K3_KDA_HEADS), K3_LAYER_THREADS, (K3_KDA_VALUE_DIM + 8u) * sizeof(float), stream,
		b->attention_out_bf16,0,b->kda_out_norm_weight,0,b->attention_out_bf16,K3_KDA_VALUE_DIM,K3_KDA_VALUE_DIM,K3_RMS_EPSILON);
	// The gate output has been waiting in gate_bf16 since the early
	// back-to-back projection.
	LM_LAUNCH((LmOutputGateKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,b->gate_bf16,K3_KDA_V_DIM);
	return(K3Project<LmBf16Format>(b,b->attention_out_bf16,b->kda_out_weight,b->kda_out_scale,
		b->attention_out_bf16,partial_accumulate,rows,K3_KDA_V_DIM,K3_HIDDEN,multiprocessors,stream));
}

// Gated MLA, 24 of 93 including the last layer of the backbone.
//
// NoPE: the rope slice is split out of q and kv and carried through unrotated.
// No rope kernel is called anywhere on this path, which is why unity.cu no
// longer instantiates one.
template<class Format, class Geometry>
static int32_t K3LayerMla(const K3LayerBuffers *b, uint32_t rows, uint32_t context, uint16_t *partial_accumulate, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	// THE INPUT IS THE RETRIEVAL, ALONE. Under AttnRes there is no residual
	// stream to fold in: the reference computes input_layernorm(h) where h is
	// what the retrieval produced (or the raw stream at layer 0), and the
	// stream itself advances only by the module-output adds the driver makes.
	// Folding a residual here normed stream-plus-retrieval, a different model.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_q_down_weight,b->mla_q_down_scale,
		b->latent_bf16,rows,K3_HIDDEN,K3_Q_LORA_RANK,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// q_a_layernorm. The reference is q_b_proj(q_a_layernorm(q_a_proj(x))) and
	// this went straight from down to up with nothing between. Its epsilon is
	// KimiRMSNorm's default 1e-6, not the model's 1e-5 - the lora norms take the
	// constructor default and only the layer norms are passed config.rms_norm_eps.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_Q_LORA_RANK + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->mla_q_norm_weight, 0,b->latent_bf16,K3_Q_LORA_RANK,K3_Q_LORA_RANK,K3_LORA_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->mla_q_up_weight,b->mla_q_up_scale,
		b->query_bf16,rows,K3_Q_LORA_RANK,K3_MLA_Q_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_kv_a_weight,b->mla_kv_a_scale,
		b->kv_slot_bf16,rows,K3_HIDDEN,K3_MLA_KV_A_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// The latent half is normalised before the up-projection; the unrotated
	// slice is not, and is shared across heads rather than being per head.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_KV_LORA_RANK + 8u) * sizeof(float), stream,
		b->kv_slot_bf16,0,(const uint16_t *)b->mla_kv_a_norm_weight, 0,b->kv_slot_bf16,K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_LORA_RMS_EPSILON);
	LM_LAUNCH((LmKvStoreKernel<Geometry,K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->cache,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows, Geometry::kSlotBytes / 2u);
	LM_LAUNCH((LmAttentionDecodeKernel<Geometry,K3_ATTN_THREADS,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>), dim3(rows,K3_MLA_HEADS), K3_ATTN_THREADS, 0, stream,
		b->query_bf16,b->query_bf16,b->cache,b->sequence_of_row,b->context_length, 0,0u,K3_MLA_HEADS,K3_MLA_QK_SCALE,b->attention_out_bf16,b->positions);
	(void)context;
	// PREFILL IS THAT ONE ARGUMENT. The kernel skips any cached position past
	// row_position[row], so a chunk of rows at ascending positions attends
	// causally over everything stored - including itself and the rest of the
	// chunk, which LmKvStoreKernel put there in this same stream order. Decode
	// is unchanged: every stored position is at or before the row's own, and
	// the guard admits them all. The contract the driver owes:
	// context_length[sequence] counts ALL stored rows, the chunk included.
	// BACK TO V-SPACE BEFORE THE GATE. The attention output is heads * kv_lora;
	// kv_b_value maps each head's 512 to its 128. Absorbing this into o_proj
	// would be algebraically fine and is not an option: the gate is elementwise
	// and does not commute with the fold, the checkpoint has no latent-space
	// gate tensor, and on GB10 it would cost 55 ms a token in weight reads to
	// save 46 us of arithmetic. See LmPerHeadProjectKernel.
	// A DISTINCT DESTINATION, NOT IN PLACE. Per head this projects 512 in to
	// 128 out, so an in-place call has head h's input bytes overwritten by the
	// outputs of blocks 4h..4h+3 - each block stages its own input to shared
	// first, but block scheduling across SMs is unordered, so heads 0..23 race.
	// The one-thread host shim executes blocks in ascending order and can never
	// see it, which its own header lists as exactly the class it cannot catch.
	// value_bf16 is idle on the MLA path and is sized at heads * 128.
	LM_LAUNCH((LmPerHeadProjectKernel<K3_LAYER_THREADS,K3_KV_LORA_RANK,K3_V_HEAD_DIM>), dim3(rows,K3_MLA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,(const uint16_t *)b->mla_kv_b_value_weight, b->value_bf16,K3_MLA_HEADS,rows);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_gate_weight,0,
		b->gate_bf16,rows,K3_HIDDEN,K3_MLA_OUT_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// No RMSNorm here - eq. 7 gates the raw attention output, unlike KDA's eq. 6
	// which normalises first. The gate is the checkpoint's g_proj, unchanged, at
	// heads * v_head_dim.
	LM_LAUNCH((LmOutputGateKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->value_bf16,b->gate_bf16,K3_MLA_OUT_DIM);
	return(K3Project<LmBf16Format>(b,b->value_bf16,b->mla_out_weight,b->mla_out_scale,
		b->attention_out_bf16,partial_accumulate,rows,K3_MLA_OUT_DIM,K3_HIDDEN,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerLatentMoe(const K3LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	// THE INTERLEAVED EXPERT STREAM. Pack V2 ships expert_w{1,2}_weight as
	// mxfp4_ws_interleaved_v1: payload and E8M0 scales co-tiled in 17-row
	// cells, one stream (docs/K3_PACK_FORMAT_V2.md). The kernels wave is
	// landed (INTERLEAVED_B in inference/kernels/gemm.cuh + tile.cuh +
	// runtime/gemm.cuh): a rank-3 UINT8 map [64, rows_per_expert, experts]
	// box [64, 17 * (TILE_N/16), 1] 64B swizzle, one bulk per 128-element
	// pack k-tile at (0, (t*cells + neuron_base/16)*17, expert), payload
	// rows 0..15 on the fragment path, staged row 16 feeding the scales - no
	// LmScaleTensor for these operands. The two launch sites below pick the
	// interleaved launchers (TILE_K forced to the pack grid's 128) when the
	// flag is set, so both pack forms run.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->mlp_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_f32 = b->router_logits;
	status = LmGemmLaunch<LmBf16Format,K3_LAYER_TILE_N,LmBf16Format::kTileK,K3_LAYER_STAGES,K3_LAYER_WARPS>(
		&gemm,b->normed_bf16,b->router_weight,rows,rows,1u,1u,
		K3_HIDDEN,K3_EXPERTS,multiprocessors,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmTopkSmallKernel<K3_LAYER_THREADS,K3_TOP_K,true,1u,1u,LM_TOPK_SCORE_SIGMOID>), rows, K3_LAYER_THREADS, 2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t), stream,
		b->router_logits,K3_EXPERTS,b->route_expert,b->route_weight,b->router_bias,0,K3_ROUTED_SCALE);
	// FROM CHOICES TO PACKED ORDER, ON DEVICE. The top-k lives here; a host
	// cannot pack what it cannot see without a sync on the hot path, and until
	// this call nothing packed it at all - every harness filled the arrays by
	// hand, which is the precise shape of a driver that cannot exist.
	status = LmRouteBuild<K3_LAYER_THREADS,K3_EXPERTS>(
		b->route_expert,rows,packed_rows,K3_TOP_K,b->group_row_offset,
		b->route_packed_row,b->route_source_token,K3_EXPERT_INTERMEDIATE * 2u,
		K3_ROUTED_EXPERT_HIDDEN,K3_LAYER_TILE_N,b->group_tile_prefix_w1,
		b->group_tile_prefix_w2,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->routed_down_weight,b->routed_down_scale,
		b->latent_bf16,rows,K3_HIDDEN,K3_ROUTED_EXPERT_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// THE GATHER IS DELETED (K3-PERF-004, roadmap D9, landed). What stood
	// here: LmGatherRowsKernel expanding latent rows into a packed copy for
	// the w1 GEMM - 2R+1W where a gather-aware A load pays 1R, ~150 MB a
	// token across the MoE stack at B16. The grouped GEMM now reads A rows
	// through route_source_token directly (route.cuh's consumer contract,
	// LmPipelineProduceIndirectA in tile.cuh): source_row_map is the map the
	// route build just wrote, activation_bytes is the UN-gathered latent with
	// rows (not packed_rows) as its extent, and the scatter side needs
	// nothing - the finalize never read the copy. The expert GEMMs stream
	// BF16 rows against MXFP4 weights; input_activations is null in the
	// checkpoint, so there is no quantise here either.
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	// THE SCALE DESCRIPTOR IS NONE, AND THAT IS THE PENDING CONTRACT, NOT AN
	// OVERSIGHT. The interleaved tensor carries its E8M0 scales in row 16 of
	// each 17-row cell; LmScaleTensorBlockUe8m0's stride math describes a
	// separate plane, so no call of it can address these scales, and pointing
	// it at the stream would misaddress every byte. The kernels wave's staged
	// cell read (the contract at the top of this function) replaces the
	// descriptor for these operands entirely. What remains reachable today is
	// the non-interleaved path, which only the host recorders exercise - a V2
	// pack always sets the flag and is refused above.
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w1;
	gemm.prefix_built = 1u;
	gemm.output_bf16 = b->gate_up_bf16;
	gemm.source_row_map = b->route_source_token;
	gemm.source_row_count = rows;
	if ( b->expert_interleave != 0u )
		status = LmGemmWeightOnlyIndirectInterleavedLaunch<
			Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
			&gemm,b->latent_bf16,b->expert_w1_weight,packed_rows,rows,
			K3_TOP_K,K3_EXPERTS,K3_ROUTED_EXPERT_HIDDEN,K3_EXPERT_INTERMEDIATE * 2u,
			multiprocessors,stream);
	else
		status = LmGemmWeightOnlyIndirectLaunch<
			Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
			&gemm,b->latent_bf16,b->expert_w1_weight,packed_rows,rows,
			K3_TOP_K,K3_EXPERTS,K3_ROUTED_EXPERT_HIDDEN,K3_EXPERT_INTERMEDIATE * 2u,
			multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// SiTU, not SwiGLU. Both betas, in the order the report gives them: 4 caps
	// the gate branch and 25 the up branch, and swapping them runs.
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), packed_rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,K3_EXPERT_INTERMEDIATE, K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	// The SiTU output is already expert-major: no gather, no quantise, the
	// rows feed the down-projection as they are. The scale descriptor stays
	// None for the reason written at the w1 launch: the scales are co-tiled
	// in the stream and no LmScaleTensor can address them today.
	gemm.source_row_map = 0;
	gemm.source_row_count = 0u;
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_tile_prefix = b->group_tile_prefix_w2;
	gemm.output_bf16 = b->gate_up_bf16;
	if ( b->expert_interleave != 0u )
		status = LmGemmWeightOnlyInterleavedLaunch<
			Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
			&gemm,b->intermediate_bf16,b->expert_w2_weight,packed_rows,rows,
			K3_TOP_K,K3_EXPERTS,K3_ROUTED_EXPERT_HIDDEN,K3_EXPERT_INTERMEDIATE,
			multiprocessors,true,stream);
	else
		status = LmGemmWeightOnlyLaunch<
			Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
			&gemm,b->intermediate_bf16,b->expert_w2_weight,packed_rows,rows,
			K3_TOP_K,K3_EXPERTS,K3_ROUTED_EXPERT_HIDDEN,K3_EXPERT_INTERMEDIATE,
			multiprocessors,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// THIS CALL WAS WRONG THREE WAYS AND COMPILED. The kernel's tail is
	// (packed, packed_row_of_token_route, weight, out, tokens, top_k, dimension)
	// and it indexes the token from blockIdx.y.
	//
	//   route_expert was passed where route_packed_row belongs - the expert id,
	//   not where this token's route landed in the packed buffer, so it indexed
	//   the expert output by expert number.
	//   tokens and dimension were swapped.
	//   the grid was 1D, so blockIdx.y was always zero and only token 0 would
	//   have been written.
	//
	// Every argument is a uint32_t and every one type-checked. glm5_2's call has
	// been correct since it was written; I did not read it before writing this.
	LM_LAUNCH((LmMoeFinalizeKernel<K3_LAYER_THREADS>), dim3((K3_ROUTED_EXPERT_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->route_packed_row,b->route_weight,b->latent_bf16, rows,K3_TOP_K,K3_ROUTED_EXPERT_HIDDEN);
	// RMSNorm between aggregation and the up-projection - the "Normalized" in
	// Normalized LatentMoE, and the report is explicit that it goes here rather
	// than after.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_ROUTED_EXPERT_HIDDEN + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->routed_norm_weight, 0,b->latent_bf16,K3_ROUTED_EXPERT_HIDDEN,K3_ROUTED_EXPERT_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->routed_up_weight,b->routed_up_scale,
		b->hidden_bf16,b->attnres_partial_bf16,rows,K3_ROUTED_EXPERT_HIDDEN,K3_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// The two shared experts run on the pre-projection hidden at full width and
	// are added to the routed result, not composed with it.
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->shared_w1_weight,b->shared_w1_scale,
		b->gate_up_bf16,rows,K3_HIDDEN,K3_SHARED_INTERMEDIATE * 2u,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,K3_SHARED_INTERMEDIATE, K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	// SEPARATE BUFFER, THEN ADD. This wrote hidden_bf16, which the routed
	// up-projection had just written, and LmGemmStore assigns rather than
	// accumulates - so the MoE output was the shared experts alone and the
	// routed branch was discarded, on 92 of 93 layers. Eq. 11 is
	// y = sum(shared) + W_up(RMSNorm(u)), an addition.
	//
	// LmAddRowsKernel was written for this - its comment says "for a shared
	// expert's contribution, which is added rather than weighted because it has
	// no gate" - and had never been called.
	// THE PARTIAL IS THE ONLY LIVE READER. Post-MLP hidden feeds nothing -
	// the next layer's first act is a retrieval that replaces it - so both
	// halves of the module output fold straight into the partial in their
	// own epilogues: routed above, shared here. The AddRows that summed
	// them into hidden, and the slice's PartialAdd that read the sum back,
	// are both gone, and so is a full-width round trip per MoE layer.
	status = K3Project<LmBf16Format>(b,b->intermediate_bf16,b->shared_w2_weight,b->shared_w2_scale,
		b->shared_out_bf16,b->attnres_partial_bf16,rows,K3_SHARED_INTERMEDIATE,K3_HIDDEN,multiprocessors,stream);
	return(status);
}

// The dense MLP, layer 0 only. first_k_dense_replace is 1: K3 has exactly one
// full-width feed-forward layer before the MoE stack begins, at intermediate
// 33792 rather than the routed 3072.
//
// Without this every layer ran LatentMoE and layer 0 was wrong - which the
// config gate said in those words, as an exemption on K3_FIRST_ROUTED_LAYER,
// rather than being discovered later.
template<class Format>
static int32_t K3LayerDenseMlp(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	// THE INPUT IS THE MLP-SIDE RETRIEVAL, IN hidden_bf16 - not the attention
	// output. The driver ran K3AttnRes over the post-attention partial before
	// calling this; reading attention_out here would compute the retrieval and
	// then ignore it, which is what this file did.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->mlp_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->dense_gate_up_weight,
		b->dense_gate_up_scale,b->gate_up_bf16,rows,K3_HIDDEN,
		K3_DENSE_INTERMEDIATE * 2u,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,K3_DENSE_INTERMEDIATE, K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	// the dense layer's module output folds into the partial the same way
	return(K3Project<LmBf16Format>(b,b->intermediate_bf16,b->dense_down_weight,
		b->dense_down_scale,b->hidden_bf16,b->attnres_partial_bf16,rows,
		K3_DENSE_INTERMEDIATE,K3_HIDDEN,multiprocessors,stream));
}

// The head, already in the split form the full-vocab price demands.
//
// Two of the three costs the naive form pays are gone. The candidate kernel
// reduces each 1024-wide vocab tile to its best (score, token) pair, so the
// full logit row - 163840 x 4 B per decode row, written and re-read every
// step - is never materialised, and the commit reduces 160 candidates a row
// instead of scanning the vocabulary. What REMAINS is the weight stream
// itself: 163840 x 7168 x 2 B = 2.35 GB read once per step, and that part is
// irreducible for an exact argmax - every vocab row must meet the hidden
// state once, which is the roadmap's D8 verdict ("exact sampling admits no
// cheat"). At TP13 it is ~180 MB per rank, 1.7% of the K3 budget.
//
// WHAT SAMPLING CONSUMES, checked before touching any of this: output_token
// and output_score only - one argmax token per row, greedy. The K3 engine
// commits tokens, no temperature/top-p path exists in this driver, and
// LmHeadSoftmaxKernel - which DOES need the full logit row - is never
// launched here. A sampler that needs the distribution would need the
// full-vocab GEMM restored and is a different contract, not a flag on this
// one. The restricted form (token_ids non-null, vocabulary = set size) is
// the exact grammar mitigation and is already wired through K3HeadRestricted.
static int32_t K3Head(const K3LayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, cudaStream_t stream)
{
	uint32_t tiles = (vocabulary + K3_HEAD_TILE - 1u) / K3_HEAD_TILE;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	LM_LAUNCH((LmHeadCandidateKernel<K3_LAYER_THREADS,K3_HEAD_TILE>), dim3(tiles,rows), K3_LAYER_THREADS, 0, stream,
		b->normed_bf16,(const uint16_t *)head_weight,token_ids, b->head_candidate_score,b->head_candidate_token,rows,K3_HIDDEN,vocabulary);
	LM_LAUNCH((LmHeadCommitKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->head_candidate_score,b->head_candidate_token,tiles, b->output_token,b->output_score,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
