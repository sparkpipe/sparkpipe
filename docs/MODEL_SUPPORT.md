# What the models need, and what the core has

Audited 2026-07-27 against published cards and papers, not against the headers
the constants came from. That distinction matters: `qwen_3_6` had an attention
kernel instantiated for 8 KV heads at 128 dims when the checkpoint says 4 at
256, and the local header agreed with the code because both were wrong.

Six targets, not five. DeepSeek V4 ships Flash and Pro, and they differ in the
layer-kind table, which is the thing least likely to be noticed.

## The six

| | layers | hidden | attention pattern | MLP |
|---|---|---|---|---|
| glm5_2 | — | 6144 | MLA + DSA index | MoE + dense |
| qwen_3_6 27B | 64 | 5120 | 16 × (3 × GDN → 1 × gated full) | dense SwiGLU 17408 |
| mimo_2_5 | 70 | — | SWA:GA 6:1, window 128, learnable sink | 1 dense + 69 MoE, 384 experts top-8 |
| dsv4 Flash | 43 | 4096 | 2 × SWA, then CSA/HCA interleaved | all-MoE, 256+1 experts top-6 |
| dsv4 Pro | 61 | 7168 | 2 × HCA, then CSA/HCA interleaved | all-MoE |
| kimi_k3 | 93 | 7168 | 3 × KDA → 1 × gated MLA (+ last layer), AttnRes | LatentMoE 896 top-16, 2 shared |

Pro is not Flash with more layers. The first two layers are HCA rather than
SWA, so a layer-kind table written for one is wrong for the other in exactly
the two positions nobody checks.

## Attention kinds, unioned

Eight distinct kinds across six models:

1. **full / global** — everything
2. **sliding window** — mimo (6:1, window 128), dsv4 Flash (first two, window 128)
3. **compressed sparse (CSA)** — dsv4 both. Compression 4, indexer 64 heads ×
   128 dims, top-512 selection
4. **high compression (HCA)** — dsv4 both. Compression 128
5. **gated full** — qwen. Sigmoid gate on the attention output
6. **MLA / latent** — glm5_2, k3
7. **gated DeltaNet** — qwen, 48 of 64 layers
8. **Kimi Delta Attention** — k3, 54 of 72 layers

Kinds 7 and 8 are the same recurrence with different gate parameterisations,
which `linear_attn.cuh` already says in its header comment. Kinds 3 and 4 are
the same mechanism at different compression rates. So the eight collapse to
five mechanisms: softmax over a cache, softmax over a *selected* cache,
recurrent state, plus two modifiers (window, output gate).

## What the core has

```
attn.cuh        LmAttentionDecodeKernel, LmSparseScoreKernel, LmRopeYarnKernel,
                LmSparseSummaryBuild, LmRopeKernel
linear_attn.cuh LmDeltaRuleDecodeKernel, LmCausalConvDecodeKernel
project.cuh     LmSplitQkvKernel, LmRopePerHeadKernel
kv.cuh          LmKvStoreKernel, LmKvHeads/LmKvLatent/LmKvState
topk.cuh        LmTopkSmall, LmTopkHistogram, LmTopkGather
head.cuh        LmHeadCandidate, LmHeadCommit, LmHeadSoftmax
norm.cuh        LmFusedResidualRmsNorm, LmSiluMul, LmQuantiseRows, LmMoeFinalize
speculate.cuh   LmSpeculativeVerifyGreedy, LmSpeculativeVerifySampled
```

The sparse path exists. `LmSparseScoreKernel` plus `LmTopkGather` plus
`LmAttentionDecodeKernel` taking `selected_positions` is CSA, already. The
window is a position list, which is the same argument. So the core covers
kinds 1, 2, 3, 4, 6, 7 and 8 today.

## What the core is missing

Seven things. Ordered by how many models they unblock.

### 1. Per-layer kind dispatch — 5 of 6 models

Four models alternate attention kinds by layer index and there is no shared
mechanism. `qwen_3_6` uses a macro the host evaluates. `mimo_2_5` uses template
parameters. `deepseek_v4` does not do it at all — one entry point for three
kinds. `kimi_k3` will need it.

The tables differ in a way that resists a formula: qwen is `layer % 4 != 3`,
mimo is 6:1, k3 is `layer % 4 != 3` again, and dsv4 is an explicit 43- or
61-entry array whose first two entries are the exception. Only dsv4 needs a
literal table, but a table expresses all four and a formula expresses three.

**Wanted:** a `uint8_t kind[LAYERS]` in each config and one host-side selector.
Not a kernel change.

### 2. Rope pairing mode — 2 of 6, silently wrong today

`LmRopePerHeadKernel` pairs `i` with `i + rope_dim/2`. DeepSeek V4 encodes
interleaved pairs, `2i` with `2i+1`, to match the released checkpoint. Both are
"rope"; they are different rotations and the output is wrong rather than
degraded.

GLM 5.2 declares interleaved pairing for both MLA and its DSA indexer, so this is a mode on the kernel or a
permutation at pack time — not a change to the default.

### 3. Dual rope theta per layer kind — dsv4 both

theta 10000 on the compression-0 layers, 160000 with YaRN elsewhere. Both
constants are already in `config.h` and correct. Nothing can select between
them because there is one entry point. Falls out of (1).

### 4. Attention sink bias — mimo

A learnable per-head bias admitted into the softmax denominator, which is how
mimo keeps long-context quality at a 7× smaller cache. `LmAttentionDecodeKernel`
has no sink term. Small change, one extra pointer and one term in the running
sum, but it changes a kernel every model calls.

### 5. Attention output gate — qwen AND kimi_k3, now confirmed by config

Two models, not one. Qwen's reference calls its full-attention path *gated*
attention. K3's fused KDA decode kernel performs "the short convolution, KDA
state update, output gate, and normalization" together, so the recurrent path
needs the same gate plus a post-normalisation.

K3's config settles it: `mla_use_output_gate: true`. One kernel serves both,
which is the argument for building it before either model needs it separately.

### 5a. Attention output gate — DONE as a kernel, wiring blocked

`LmOutputGateKernel` is in `norm.cuh`: elementwise multiply by a sigmoid of the
gate projection, applied after attention and before the output projection.

Neither model is wired to it, and the reason is the same for both: the gate
tensor comes out of the fused QKV projection, and where it sits in that
projection is not in either config. Qwen's `QWEN36_QKV_DIM` is currently
`(24 + 2*4) * 256`, which has no room for a gate - so either the projection is
wider than the config says or the gate is a separate tensor. Guessing which
would silently reinterpret a slice of the weights.

UNBLOCKED BY: the modelling file for either model.

### 5b. SiTU activation — kimi_k3, and it blocks every layer

`hidden_act: "situ"` with two betas, 4.0 gated and 25.0 linear. The formula is
not published: Moonshot's blog names "Sigmoid Tanh Unit" and says it improves
activation control, and the GGUF conversion effort records it as new and
unimplemented.

The name plus two betas admits several readings that are different functions
and all produce fluent text. This is the one item on this list where a
plausible guess is worse than an empty space, because nothing downstream would
contradict it.

UNBLOCKED BY: `modeling_kimi_linear.py` from the released repository.

Sigmoid gate on the attention output before the output projection. The
reference calls that path *gated* attention; it is not optional. Currently
recorded as an exemption in `test_config_coverage.py` reading NOT IMPLEMENTED.

### 6. Hash routing — dsv4 both

The first three MoE layers route through a token-id → expert-id table rather
than the router. A different gate, not a different expert kernel.

### 7. Non-uniform residuals — the kernel exists, the transport does not

`LmAttnResKernel` implements the retrieval: normalise each candidate to score
it, softmax over the scores, mix the UNNORMALISED candidates. Checked on the
host against report eq. 9 at 2.2e-3, and the test fails if the values are
normalised too — which would flatten exactly the layers the mechanism weighs.

WIRED. `K3AttnRes` runs the retrieval before attention and again before the
MLP with different pseudo-queries, and `K3AttnResAccumulate` folds each layer's
output into the block in progress, closing it every 12 layers. The driver skips
layer 0, which has only the embedding in the bank.

The bank costs 143,360 bytes a token — ten hidden states against one — and 8.8
MB at B64 crossing a stage boundary. That is the transport question as a number,
and SGLang's answer is still the one to follow: generate the block
representation once at the boundary layer, share it, carry only new blocks
incrementally.

WHAT REMAINS IS NOT KERNEL WORK. A layer needs the bank threaded through it:
the block representation is a running sum reset every 12 layers, the embedding
is always b_0, and the retrieval runs TWICE per layer — before attention and
before the MLP, with separate projections. That state has to cross the pipeline
stage boundary, which is 9 hidden states per token instead of one.

SGLang's account is worth following rather than reinventing: the block
representation is generated once at the boundary layer and shared by every
subsequent layer, and the pipeline carries only newly generated blocks
incrementally. That is a scheduler and transport change, and it is where
deepseek_v4's mHC should land too.

### 7b. The old note, kept because the design argument still holds

Every block carries `n_hc = 4` streams of the hidden state across the boundary,
mixed by a Sinkhorn-Knopp normalised doubly-stochastic matrix, 20 iterations.

This is a **shape** change, not a kernel gap. `hidden_bf16` is one stream in
every buffer struct in the tree, and the residual the layer adds to is a
different object under mHC. It touches the KV pool sizing, the transport
payload per row, and the stage boundary — the ring moves hidden state between
ranks and would move four times as much.

K3's AttnRes is the same class: layers retrieve from earlier layer BLOCKS
rather than one accumulated stream. Two of six models now want a residual that
is not a single tensor, from two vendors independently, which is the same
signal that made `linear_attn.cuh` parameterise growth rather than special-case
it.

Everything else on this list is additive. This one is not, and it should be
designed for both models at once, before either is called supported.

## Known-wrong, recorded

| model | issue | where |
|---|---|---|
| qwen_3_6 | output gate not applied | config gate exemption |
| qwen_3_6 | mrope sections ignored; text-only is unaffected | `config.h` |
| dsv4 | rope pairing convention | `config.h` |
| dsv4 | layer-kind table absent | `config.h` |
| dsv4 | hash routing, hyper-connections absent | `config.h` |
| mimo_2_5 | sink bias not audited into config | this file |

## kimi_k3, after reading the modelling file

Three of the four blockers are settled and one is code.

| | state |
|---|---|
| SiTU | DONE. `LmSituMulKernel`, checked numerically against the reference |
| MLA output gate | kernel DONE and correct; `g_proj` is separate, sigmoid, before `o_proj` |
| LatentMoE | SPECIFIED. Router on the FULL hidden, experts at 3584, shared experts outside the latent |
| AttnRes | SPECIFIED. A softmax attention over 9 candidates. **9× transport** |
| KDA | SPECIFIED except the decay arithmetic, which is in `fla`, not the released file |

### The one that changes a shared kernel

`linear_attn.cuh` reads `forget_gate[(row * key_heads) + head]` — one scalar per
head. KDA's forget gate is `f_b_proj(f_a_proj(hidden))`, a low-rank projection
to `heads * head_dim`, so it is **per head per channel**. The kernel cannot
express this model's decay.

Everything else K3 needs is additive. This one widens an argument that
`qwen_3_6` also passes, so the two models have to agree on it. That makes it the
first piece of K3 work with a blast radius outside `inference/llms/kimi_k3/`.

Also: KDA runs three separate short convolutions, one each for q, k and v, every
one with a SiLU. `LmCausalConvDecodeKernel` applies no activation and the qwen
path calls it once.

### Still unread

The decay itself. `g`, `A_log` and `dt_bias` are combined inside `fla`'s
`fused_recurrent_kda` with `use_qk_l2norm_in_kernel`, `use_beta_sigmoid_in_kernel`
and a `-5.0` lower bound. The released modelling file calls that function and
does not define it, so the last piece of KDA needs the `fla` source rather than
the checkpoint.

## Order

1. per-layer kind dispatch — unblocks dsv4 ×2 and cleans qwen, mimo, k3
2. rope pairing mode — small, and dsv4 is wrong without it
3. dual theta — free once (1) exists
4. output gate — finishes qwen
5. sink bias — finishes mimo
6. hash routing — dsv4
7. hyper-connections — design first

Nothing above needs hardware. All of it is checkable by the geometry gates,
which is the argument for doing it now rather than when a ring is free.
