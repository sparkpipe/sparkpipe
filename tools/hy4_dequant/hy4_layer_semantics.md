# hy4 layer forward semantics (from AngelSlim's llama.cpp implementation)

Ground truth: `vendor/hyv4_reference.cpp` (src/models/hyv4.cpp @ llama.cpp
0cea36222, from hy4-preview-patch/0001). Config pins: hc=4, eps=1e-6,
magnitude=2.0, sigmoid routing + e_score_correction_bias, experts 8/256 + 1
shared, scale 2.827, weights_norm=true, swiglu clamp 10.0 on ROUTED experts
only (shared/dense unclamped), MLA qk 256 / v 256, DSA indexer full-every-4th,
learnable sinks, lm_head fp32.

## Block structure
```
residual[embd x 4 streams]
attn:  cur = hc_pre(residual, hc_attn_fn/scale/base)
       cur = rms_norm(cur, input_layernorm)
       attn_out = MLA_DSA(cur) * sigmoid(wqkv_gate . cur)   # gated MLA
       o = o_proj(attn_out)
       residual = hc_post(o, residual, hc_attn post-gate)
ffn:   cur = hc_pre(residual, hc_ffn_fn/scale/base)
       cur = rms_norm(cur, post_attention_layernorm)
       moe = routed(cur) + shared(cur)                       # silu, clamp routed only
       residual = hc_post(moe, residual, hc_ffn post-gate)
head:  hc_head collapse 4 streams -> rms_norm -> lm_head (fp32)
```

## hc math
```
mixes = hc_fn(cur)                     # [4*embd -> 8] (2*hc coefficients)
pre   = sigmoid(mixes[0:4]*scale_pre  + base_pre)  + eps
post  = sigmoid(mixes[4:8]*scale_post + base_post)*magnitude + eps
hc_pre_reduce(x, pre)  = sum_ih x_ih * pre_ih          # width mixing
hc_post(b, residual, post): distribute branch output back across streams
                            (per reference: residual = residual + post*? —
                            read hyv4_hc_post impl before coding)
```
NOTE: implementer must read build_hc_post's residual update line in the
vendor file exactly — the pre-reduce/post-distribute round trip is the
subtlest part of this architecture.

## MoE routing (build_moe_ffn with SIGMOID + bias + norm + scale)
```
logits = gate_inp(cur)                       # [256]
scores = sigmoid(logits)                     # elementwise, NOT softmax
scores = scores + e_score_correction_bias    # bias for SELECTION only
top-8 by biased score; selection weights = softmax? NO — with
expert_weights_norm=true: w = norm(selected sigmoid scores) then
w *= routed_scaling_factor (2.827)
expert(e, x) = down_exps[e] @ ( silu(gate_up) clamped to +/-10.0 )
sum w_e * expert_e  +  shared_expert(x)
```
(Cross-check build_moe_ffn's exact bias/norm order in llama.cpp before
coding — the selection-vs-weight bias split is the classic DeepSeek trap.)

## Attention (gated MLA + DSA)
q_a -> q_a_norm -> q_b (absorbed: q_nope + q_pe concat); kv_a_mqa ->
kv_a_norm; kv_b decomposed attn_k_b/attn_v_b; indexer: wq_b/wk/weights_proj/
k_norm -> top-k 2048 index over token scores (full indexer own kv every 4th
layer, shared layers reuse layer 0's index); sinks added per head; scale
1/sqrt(256).
