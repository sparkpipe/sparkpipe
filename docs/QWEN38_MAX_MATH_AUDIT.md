# Qwen 3.8 Max inference math audit - kernel formulas vs the pinned reference

Every inference and KV-cache kernel was checked against the authoritative
vendor reference: config.json (Qwen3_5MoeForCausalLM, qwen3_5_moe_text) plus
transformers modeling_qwen3_5_moe.py (Qwen3_5MoeTopKRouter /
Qwen3_5MoeSparseMoeBlock / Qwen3_5MoeGatedDeltaNet / Qwen3_5MoeAttention).
Verdict per kernel below; three real math errors were found and fixed.

## Math errors found and fixed

1. ROUTER WEIGHTS WERE RAW LOGITS OVER THEIR SUM, NOT SOFTMAX.
   Reference: routing_weights = softmax(logits); topk; renormalize over the
   selected experts. The module's GateSelectKernel and the family's
   LmTopkSmallKernel(IDENTITY + RENORMALISE) divided raw logits by their
   sum - every negative logit became a NEGATIVE mixture weight, and the
   mixture was not a probability distribution at all.
   Fix: module GateSelectKernel now softmaxes the top-k (max-subtracted
   exp, warp reduce); the family softmaxes the full logit row with
   LmHeadSoftmaxKernel (temperature 1.0) before the topk renormalize.
   Selection is unchanged (softmax is monotone); weights are now correct.

2. SHARED-EXPERT GATE WAS sigmoid(weight[d]) PER CHANNEL, NOT THE SCALAR
   GATE. Reference: shared_expert_gate = Linear(hidden, 1) (the checkpoint
   tensor is [1, 8192]); shared_expert_output *= sigmoid(gate(normed_input))
   - one scalar per token derived from the MoE input. Both the module
   (SharedGateKernel) and the family (Qwen38SharedExpertAddKernel) applied
   sigmoid to the WEIGHT ELEMENTS per channel, a token-independent
   distortion. Fix: per-row dot(normed_input, gate_weight) -> sigmoid ->
   broadcast multiply, in both implementations.

3. THE ROUTED-MOE PAIR REDUCE ACCUMULATED ONTO THE ATTENTION/GDN DELTA,
   APPLYING IT TWICE PER LAYER. The fused residual norm (verified: it writes
   hidden += delta in place and normalizes the sum) already folds the
   attention/GDN output into hidden; the pair reduce then read delta_bf16,
   added the expert mixture on top, and the final ResidualAdd(hidden, delta)
   added the attention/GDN delta a SECOND time. Fix: the module now uses
   SparkLmHostLaunchMoePairReduceOverwrite (the mixture starts from zero),
   matching the reference expert_output = sum(w x expert_out) + shared.

4. CHUNKED GDN TRIANGULAR SOLVE HAD A DATA RACE AND WRONG SUBSTITUTION.
   The reference snapshots the ORIGINAL row before updating
   (T[i,:i] = A[i,:i] + A[i,:i] x T[:i,:i]). The module read live A[row,e]
   entries while other columns of the same row were writing them - a race
   that also folds extra powers of A into the transform. Fix: stage the row
   in shared memory, barrier, then apply. (Latent: prefill-only path, which
   the module still refuses at the frame boundary.)

## Verified correct against the reference

- Fused residual + RMSNorm: hidden = hidden + delta in place, then
  rmsnorm(hidden) with eps 1e-6 - matches the two-norm decoder layout
  (input_layernorm before attention, post_attention_layernorm folded with
  the residual into the MoE input).
- Attention: per-head RMSNorm(256) on q and k; partial RoPE on the first 64
  dims (theta 1e7, half-split rotate: out_first = x1 c - x2 s,
  out_second = x2 c + x1 s); frequency theta^(-2i/d); 1/sqrt(head_dim) =
  1/16 logit scale; online softmax merge (max/den rescale, per-warp); the
  fused output gate sigmoid applied per element before o_proj.
- KV cache addressing: token record = 2 x 4 heads x 256 bf16 (K then V,
  head-major), block = 64 tokens, layer-major blocks, ceil(context/64)
  block counts, slot = block*64 + offset - all consistent between prepare
  and decode.
- Router top-k selection: NaN logits rank last (keyed as -inf), the
  ordered-key decode matches the key encode, bias selects but does not
  weigh.
- SwiGLU: silu(gate) x up, both routed and shared experts.
- GDN decode: depthwise causal conv (kernel 4, no bias, silu on the conv
  output) with a 3-tap carried tail; g = -exp(a_log) x softplus(a +
  dt_bias); beta = sigmoid(b); q = l2norm(q)/sqrt(dk), k = l2norm(k); the
  recurrent step in reference order (decay, predict kv_mem, delta =
  (v - kv_mem) x beta, rank-one update, read-out from the updated state),
  fp32 state.
- GDN chunk: q l2norm + 1/sqrt(dk) scale; intra-chunk decay cumsum; the
  strictly-lower -k_beta k^T seed with the decay mask; w = T(v x beta);
  kg = T(k x beta x e^G); v_new = w - kg S; out = (q e^G) S + (q k^T o D)
  v_new; carried state S <- S e^G_last + (k e^(G_last - G))^T v_new - all
  stage-for-stage the torch_chunk_gated_delta_rule form.
- Gated output norm: rmsnorm(core) x weight x silu(z), norm before gate.
- Head emission: final RMSNorm then exact fused matvec + argmax over the
  full 248320 vocabulary (the screened variant is common code with its own
  error-bound tests; the module uses the exact path).
- Embedding gather, residual adds, RMSNorm epsilon, attention output gate
  type (swish) - all match.

## Comments corrected
- Firmware header: GDN value width stated 6144; the checkpoint and the
  packer both pin 128 heads x 128 = 16384 (qkv total 20480).
- Decode kernel: scale comment said 1/sqrt(128); the code and the reference
  both use 1/sqrt(head_dim) = 1/16.
- MoE header comments claimed softmax while the code renormalized raw
  logits, and a per-dimension shared gate; both now describe the fixed,
  reference-verified semantics.

## Remaining verification
Numerical reference-vector comparison of the fixed kernels on device
(router softmax, shared gate, pair-reduce-overwrite, chunk solve) is the
next qualification step on spark4/fleet; this audit fixes the formulas to
match the authoritative reference, and the existing smoke + work-control
suites verify the builds and control flow.
