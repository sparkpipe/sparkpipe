#pragma once
// Kimi K3. Shapes and constants only.
//
// WEIGHTS ARE OUT AS OF 2026-07-27, AND THIS FILE HAS NOT SEEN THEM.
// moonshotai/Kimi-K3 is gated (401); the public repository is
// moonshotai/Kimi-K3-MXFP4. Nothing below was read from its config.json - the
// corrections in this pass come from the vLLM K3 preview and Moonshot's launch
// material, which are implementers' accounts rather than the config itself.
//
// Corrected: K3_LAYERS was a guess of 72 from K2 lineage. vLLM's preview says
// 93. Still second-hand, but an implementer counting layers beats an inference
// from a previous model.
//
// Confirmed: 3 of every 4 attention layers are KDA, and the expert pool is
// top-16 of 896. Both were already right.
//
// FOUR THINGS THE RELEASE ADDS THAT ARE NOT MODELLED HERE:
//
//   AttnRes. Layers retrieve representations from earlier layer BLOCKS rather
//   than reading one uniformly accumulated residual stream. This is the same
//   class of change as DeepSeek V4's hyper-connections - a shape change, not a
//   kernel - and the two should be designed together. See
//   docs/MODEL_SUPPORT.md item 7.
//
//   The KDA output gate and post-normalisation. vLLM's fused decode kernel
//   performs "the short convolution, KDA state update, output gate, and
//   normalization" as one. kernels/linear_attn.cuh has the first two. Qwen 3.6
//   needs an attention output gate too, so this is one kernel serving both.
//
//   MXFP4 weights, group 32, E8M0 scales. WEIGHT-ONLY: the checkpoint's
//   quantization_config sets input_activations to null and output_activations
//   to null. The tech report's deployment section says activations were
//   computed in MXFP8, and that describes Moonshot's own serving stack, not
//   what the released checkpoint asks an inference engine to do.
//
//   I recorded the report's sentence here and then quantised activations to
//   MXFP4 - not even the MXFP8 the report mentions - because the GEMM takes one
//   Format for both operands. The checkpoint asks for neither. The production
//   path is BF16 activation against a streamed MXFP4 weight, decoded to BF16
//   registers at the tile, BF16 MMA with FP32 accumulation.
//   GLM52_MXFP4_GROUP is currently exempted from the coverage gate as "a
//   supported format with no checkpoint using it". There is now a checkpoint
//   using it.
//
//   Stable LatentMoE, which is not plain top-k routing over 896 experts, and
//   native vision, which is out of scope for a text decode path.
//
// MOST OF THESE ARE GUESSES AND SAY SO. The old tree marked them and the marking
// is carried forward deliberately: a constant inferred from a lineage is not the
// same kind of fact as one read from a published config, and losing that
// distinction is how an inferred number becomes a load-bearing assumption.
//
// The architecture is the interesting part. Three of every four layers use Kimi
// Delta Attention - a recurrent state, no growing cache - and the fourth uses
// gated MLA over a KV cache. So this model needs BOTH pools and a per-layer
// dispatch, which is the first real test of kernels/kv.cuh's claim that a
// recurrent state is just a pool that does not grow.
#include <stdint.h>
#include "inference/kernels/layer_kind.cuh"

// Read from moonshotai/Kimi-K3 config.json, 2026-07-27. Everything below is
// DISCLOSED unless marked otherwise, and the guesses this replaced are recorded
// where they were wrong, because which ones failed is the useful information.
//
//   right:  hidden 7168, vocab 163840, rms eps 1e-5, experts 896, top-k 16,
//           first dense layer count 1, KDA head dim 128, the 3:1 period
//   WRONG:  shared experts 1 -> 2
//           expert intermediate 2048 -> 3072
//           dense intermediate 18432 -> 33792
//           routed scale 2.5 -> 1.0
//           KDA heads 64 -> 96
//
// Five of thirteen. Every wrong one was a K2-lineage inference, and every one
// of those would have produced a model that built, ran, and was wrong.

#define K3_HIDDEN 7168u
#define K3_LAYERS 93u
#define K3_VOCAB 163840u
#define K3_RMS_EPSILON 1e-05f
// The lora norms take KimiRMSNorm's constructor default; only the layer norms
// are passed config.rms_norm_eps. q_a_layernorm and kv_a_layernorm use this.
#define K3_LORA_RMS_EPSILON 1e-06f
#define K3_MAX_CONTEXT 1048576u

// MoE. Stable LatentMoE: the routed experts run at their own hidden width and
// are projected, which is what routed_expert_hidden_size is and why it is not
// K3_HIDDEN. Router is sigmoid with noaux_tc grouping, renormalised.
#define K3_FIRST_ROUTED_LAYER 1u
#define K3_EXPERTS 896u
#define K3_TOP_K 16u
#define K3_SHARED_EXPERTS 2u
#define K3_EXPERT_INTERMEDIATE 3072u
#define K3_ROUTED_EXPERT_HIDDEN 3584u
#define K3_DENSE_INTERMEDIATE 33792u
#define K3_ROUTED_SCALE 1.0f

// MLA, on the full-attention layers. mla_use_nope and mla_use_output_gate are
// both true: the nope half carries no rotation, and the attention output is
// gated - the same gate qwen_3_6 needs, so one kernel serves both.
#define K3_MLA_HEADS 96u
#define K3_KV_LORA_RANK 512u
#define K3_Q_LORA_RANK 1536u
#define K3_QK_NOPE_DIM 128u
// NOT ROTATED. K3 is NoPE: modeling_kimi_linear.py sets rotary_emb to None and
// asserts use_nope, then splits this slice out of q and kv only to concatenate
// it back untouched. The k side is broadcast across heads, MQA style. Position
// is carried by KDA's decay, which is how the model reaches 1M tokens without
// RoPE rescaling. The name is the reference's; the behaviour is not what it
// suggests, so calling any rope kernel on a K3 layer would be wrong.
#define K3_QK_UNROTATED_DIM 64u
#define K3_V_HEAD_DIM 128u
// READ from modeling_kimi_linear.py line 359: self.scaling = self.q_head_dim
// ** (-0.5), with q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64.
//
// I had this as DERIVED, reasoning from the MLA convention, while the modelling
// file was already on disk. The number was right and the label was wrong, which
// is the worse of the two failures: a hedge on a fact that could be checked
// spends someone else's review time confirming what a grep would have settled.
#define K3_MLA_QK_SCALE 0.07216878365f       /* 1 / sqrt(128 + 64) */
#define K3_MLA_USE_NOPE 1u
#define K3_MLA_OUTPUT_GATE 1u

// -- what the modelling file says KDA and the MoE actually are ------------------
//
// LATENT MoE. The router runs on the FULL hidden, not the latent, which is the
// part a shape-only reading gets backwards:
//
//     topk, weights = gate(hidden)                 router at 7168
//     latent        = down_proj(hidden)            7168 -> 3584
//     y             = experts(latent, topk)        experts at 3584, inter 3072
//     y             = rms_norm(y)                  latent_moe_use_norm is true
//     y             = up_proj(y)                   3584 -> 7168
//     out           = y + shared_experts(hidden)   shared run on the ORIGINAL
//
// The shared experts are NOT in the latent space and take the pre-projection
// hidden at intermediate moe_intermediate * num_shared = 3072 * 2 = 6144.
//
// KDA DOES NOT FIT LmDeltaRuleKernel AS IT STANDS. The projections:
//
//     q_proj, k_proj, v_proj   hidden -> heads * head_dim = 12288, each with its
//                              OWN short convolution, each with a SiLU
//     g (forget)               f_b_proj(f_a_proj(hidden)): 7168 -> 128 -> 12288,
//                              so PER HEAD PER CHANNEL
//     beta (write)             b_proj(hidden): 7168 -> 96, per head scalar,
//                              sigmoid applied inside the kernel
//     output gate              g_proj(hidden) -> 12288, full rank, per channel
//     o_norm                   gated RMS norm with a sigmoid, at head_dim
//     A_log                    per head, log-uniform over [1,16]
//     dt_bias                  per channel
//     lower bound              -5.0 clamp on the gate
//
// linear_attn.cuh reads forget_gate[(row * key_heads) + head] - ONE SCALAR PER
// HEAD. KDA's forget gate is head_dim wide per head. The kernel cannot express
// this model's decay, and widening that argument is the first piece of K3 work
// that changes a shared kernel rather than adding one.
//
// Three separate convolutions with SiLU, not one; LmCausalConvKernel has
// no activation. The final decay arithmetic combining g, A_log and dt_bias lives
// in fla's fused_recurrent_kda and is NOT in the released modelling file, so it
// is the one piece still unread.

// KDA, on the other three in four. 96 heads at 128, a 4-wide causal convolution,
// a full-rank gate floored at -5.
#define K3_KDA_HEADS 96u
#define K3_KDA_KEY_DIM 128u
#define K3_KDA_VALUE_DIM 128u
#define K3_KDA_CONV_KERNEL 4u
#define K3_KDA_GATE_LOWER_BOUND -5.0f
#define K3_KDA_FULL_RANK_GATE 1u

// AttnRes sources at K3's shape: 93 layers in blocks of 12 gives 8 blocks, one
// partial final, and the embedding is always b_0 - so a layer late in the stack
// scores 9 candidates plus the running partial sum. The partial is not a bank
// candidate, and counting it is what made this 10 against the contract's 9
// (k3_authoritative.json: layer_block_count + embedding_representation_count).
// LmAttnResKernel takes the count at runtime because it grows with depth;
// MAX_SOURCES bounds the shared arrays.
#define K3_ATTNRES_MAX_SOURCES 9u

// THE PER-REQUEST COST OF THE BANK, which is the transport question stated as a
// number. Eight completed blocks plus the embedding is nine hidden states a
// token in the bank, with the running partial carried beside it under its own
// range, against one state for a conventional residual.
#define K3_ATTNRES_BANK_BYTES \
	(K3_ATTNRES_MAX_SOURCES * K3_HIDDEN * (K3_KV_BITS / 8u))

// THE BANK, AS THE WIRE DESCRIBES IT. pipeline_sideband validates a bank of
// source_count slots and a partial of one state, so it needs the slot count
// and one state's bytes separately rather than the product above. A slot is
// one BF16 hidden row; the slots are the candidates, and the partial - one
// more state of the same size - travels under its own flag.
#define K3_ATTNRES_BANK_SLOTS K3_ATTNRES_MAX_SOURCES
#define K3_ATTNRES_PARTIAL_BYTES (K3_HIDDEN * (K3_KV_BITS / 8u))

// AttnRes, now read from the modelling file. It is an ATTENTION over residuals,
// not a weighted sum with learned scalars:
//
//     v       = [saved block residuals ..., current prefix sum]
//     k       = v * rsqrt(mean(v^2) + eps)          RMS-normalise each candidate
//     scores  = sum(k * (norm.weight * proj.weight))
//     out     = softmax(scores) @ v
//
// Applied TWICE per layer - once before attention with self_attention_res_proj,
// once before the MLP with mlp_res_proj. A new block residual is appended every
// attn_res_block_size layers, so 93 layers at block size 12 accumulate 8 blocks
// and the candidate set reaches 9.
//
// THIS IS THE SHAPE CHANGE, AND THE NUMBER IS 9x. Every buffer in this tree
// carries hidden_bf16 as one tensor. Under AttnRes a token carries up to nine,
// and the ring moves hidden state between ranks, so the stage payload per row
// goes from 14 KiB to 126 KiB at hidden 7168. That is a transport and pool
// question before it is a kernel question. deepseek_v4's hyper-connections are
// the same class at n_hc=4 - see docs/MODEL_SUPPORT.md item 7.
#define K3_ATTNRES_BLOCK_SIZE 12u

// SiTU, from the released modeling_kimi_linear.py. IMPLEMENTED as
// LmSituMulKernel, checked numerically by tests/test_situ_activation.py:
//
//     situ_a = beta * tanh(gate / beta) * sigmoid(gate)
//     up     = linear_beta * tanh(up / linear_beta)
//     out    = situ_a * up
//
// It is SiLU-mul with both arms soft-clamped to their own beta - gate to 4,
// linear to 25 - which is what "activation control" meant. Gate is the FIRST
// half of the fused projection.
//
// The two betas are not interchangeable: swapping them clamps the gate at 25
// and the linear arm at 4, which runs and is wrong, so the gate checks the
// saturation points rather than only the values near zero.
//
// (Historical note, kept because the reasoning still applies to AttnRes below:
// before the modelling file arrived this was the one gap where a plausible
// guess was worse than an empty space. THE FORMULA WAS
// NOT PUBLISHED anywhere I can reach. Moonshot's tech blog names it and says it
// improves "activation control"; the GGUF conversion effort records it as a new
// activation and does not implement it either. The name and the two betas are
// all that is public.
//
// I am not writing this kernel from the name. "Sigmoid Tanh Unit" with betas at
// 4.0 and 25.0 admits several readings - sigmoid(beta*x)*tanh(x), x*sigmoid of a
// tanh, a two-branch gate with a beta each - and they are different functions
// that all produce fluent text. This is the one gap on the list where a
// plausible guess is worse than an empty space, because nothing downstream
// would contradict it.
//
// NOT PUBLISHED and the name admitted several readings. It came from the
// modelling file, not from reasoning about the name.)
#define K3_SITU_BETA 4.0f
#define K3_SITU_LINEAR_BETA 25.0f

// MXFP4 at group 32, and the ignore list matters: attention, shared experts,
// the dense MLP, lm_head and the vision tower are NOT quantised. Only the
// routed experts are 4-bit.
#define K3_MXFP4_GROUP 32u

// num_nextn_predict_layers is 0. This model has no MTP head, unlike glm5_2.
#define K3_MTP_LAYERS 0u

#define K3_KV_BITS 16u
#define K3_KV_PAGE_SLOTS 64u

// THE RECURRENT STATE IS FP32, AND THE CONVOLUTION WINDOWS LIVE WITH IT.
//
// Checked against SGLang's reported figure rather than derived and left alone:
// they measure one KDA state block at about 54 MB under TP=8 covering all 69
// KDA layers. Working backwards,
//
//     bf16 outer product   96 * 128 * 128 * 2 = 3 MB/layer -> 25.9 MB at TP=8
//     fp32 outer product   96 * 128 * 128 * 4 = 6 MB/layer -> 51.8 MB at TP=8
//     plus 3 conv windows  3 * 12288 * 4 * 2  = 0.28 MB/layer -> 2.4 MB
//                                                        total 54.2 MB
//
// So the state is fp32 and the three short-convolution windows are part of the
// same per-request block. This expression carried neither: it was bf16 and had
// no windows, which is half the arithmetic and none of the convolution.
//
// A recurrent state accumulates over a million tokens without renormalising, so
// fp32 is not a precision luxury - the same argument as keeping the flash
// attention output in fp32 during training. The rest of the model is 4- and
// 8-bit; this one buffer is not. The bf16 escape exists - it is half of the
// largest batch term in the model - as an admission-time option with its own
// numerics contract at K3_KDA_STATE_SLOT_BYTES_BF16 below, default OFF.
// Reference semantics (moonshotai/Kimi-K3 modeling_kimi_linear.py): the
// checkpoint's A_log tensor carries 128 heads; the model runs 96 - the
// loader takes the authoritative first-96 slice and must refuse other
// shapes. A_log and dt_bias are FP32 in the checkpoint and stay FP32
// through bind. q and k are L2-normalized IN KERNEL
// (use_qk_l2norm_in_kernel), beta passes through sigmoid in kernel, the
// gate combines g, A_log and dt_bias in kernel with the safe lower
// bound defined with the KDA block above, and the reference stores state
// TRANSPOSED (transpose_state_layout=True) - the pack/loader owns that
// flip.
#define K3_KDA_A_LOG_SOURCE_HEADS 128u
#define K3_KDA_QK_L2NORM 1u
#define K3_KDA_STATE_ELEMENT_BYTES 4u
#define K3_KDA_CONV_WINDOW_BYTES \
	(((2u * K3_KDA_HEADS * K3_KDA_KEY_DIM) + (K3_KDA_HEADS * K3_KDA_VALUE_DIM)) \
		* K3_KDA_CONV_KERNEL * (K3_KV_BITS / 8u))
// THREE NAMES, THREE USES, ONE SUM. The kernel addresses the outer-product
// slot alone - the convolution windows are separate pools with their own
// strides, because LmCausalConvKernel indexes them by channels * kernel
// and an interleaved slot cannot be reached by base + index * stride. The SLOT
// constant is what the delta rule and the replay fold are handed; the sum is
// what a capacity plan budgets per sequence per layer.
#define K3_KDA_STATE_SLOT_BYTES \
	(K3_KDA_HEADS * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * K3_KDA_STATE_ELEMENT_BYTES)
#define K3_KDA_STATE_BYTES (K3_KDA_STATE_SLOT_BYTES + K3_KDA_CONV_WINDOW_BYTES)

// THE BF16 STATE OPTION - HALF THE SLOT, DEFAULT OFF, FAIL-CLOSED TODAY.
//
// At batch the fp32 slot is the largest per-sequence stream in the model:
// 69 layers x 6.59 MB x 2 (read + write) = 909 MB per sequence per token,
// ~40% of a B64 step's bytes at the BF16 weight recipe
// (docs/PERF_ROADMAP_2026-08-01.md, "The K3 state correction"). Halving the
// slot halves exactly that term, and K3_SPEED.md:56-60 names it the single
// biggest K3 throughput lever after residency - and "a numerics question,
// not a systems one". The numerics: the slot is read and rewritten at every
// committed token, so a bf16 slot re-rounds every element once per token and
// the rounding compounds inside a recurrence that never renormalises, over a
// 1M-token context. That is the same argument the fp32 block above makes,
// and it does not go away because the bandwidth is tempting.
//
// So this is an ADMISSION-TIME option (README.md:81-84 prices the pool both
// ways), never the default, and it is gated three ways: the consumer flag is
// K3LayerBuffers::kda_state_bf16, the layer FAILS CLOSED on it while no
// launch site selects the uint16_t State instantiation (the default
// instantiation's shared tile and pool traffic are float), and
// tests/test_k3_driver_contracts.py refuses a tree where the flag can launch.
// The kernel variant exists - linear_attn.cuh's State template parameter,
// host-gated by tests/test_kda_bf16_state.py - so what lifts the gate now is
// the launch-site selection plus on-device numerics receipts, written at the
// flag.
#define K3_KDA_STATE_ELEMENT_BYTES_BF16 2u
#define K3_KDA_STATE_SLOT_BYTES_BF16 \
	(K3_KDA_HEADS * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM \
		* K3_KDA_STATE_ELEMENT_BYTES_BF16)
// Layer counts by kind, so per-layer pool arithmetic has one source. 69 + 24.
// Literals, matching generated_config.h's spelling of the same two numbers -
// pipeline_sideband.h includes both headers, and (K3_LAYERS / 4u) + 1 against
// 24u is one value spelled two ways, which the preprocessor reports as a
// redefinition.
#define K3_KDA_LAYER_COUNT 69u
#define K3_MLA_LAYER_COUNT 24u

// 1-INDEXED IN THE CONFIG, 0-INDEXED HERE. full_attn_layers is
// {4,8,...,92} plus 93; subtract one and that is {3,7,...,91} plus 92. So the
// period-4 rule holds and THE LAST LAYER IS AN EXCEPTION - 92 % 4 is 0, which
// the formula alone would call KDA. 24 full and 69 KDA, which is what the two
// lists in the config contain.
#define K3_LAYER_IS_LINEAR(layer) \
	((((layer) % 4u) != 3u) && ((layer) != (K3_LAYERS - 1u)))

#define K3_LAYER_KIND(layer) \
	(K3_LAYER_IS_LINEAR(layer) ? LM_LAYER_RECURRENT : LM_LAYER_LATENT)
