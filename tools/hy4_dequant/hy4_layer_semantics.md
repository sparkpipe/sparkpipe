# hy4 layer forward semantics (from AngelSlim's llama.cpp implementation)

Ground truth: `src/models/hyv4.cpp` as added by
`AngelSlim/Hy4-preview-GGUF@779242edccdedc2109a0b36b164263a88f015bfa`
`hy4-preview-patch/0001-hyv4-architecture.patch`, a patch against llama.cpp
`0cea36222fe9bac5ebfc45716c9eef11f37046c4`; it is not vendored because the
upstream repository states no license. Config pins: hc=4, eps=1e-6,
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

## Quant block layouts

An IQ1_M block of 256 elements is `qs` (32 bytes), `qh` (16 bytes) and
`scales` (8 bytes), 56 bytes in total; `hy4_dequant_iq1_m` reads the scales as
four `uint16_t` words after the first 48 bytes. IQ2_XXS and IQ3_XXS blocks are
66 and 98 bytes per 256 elements. These byte geometries were cross-checked
against the checkpoint itself by the solver in `tools/hy4_tp16_shard.py`.

## Vendored dequantization (`hy4_iq_dequant_vendor.h`)

`hy4_iq_dequant_vendor.h` holds the block decoders that the forward tools
and `hy4_iq_dequant_ref` in this directory and the CUDA tests in
`tools/hy4_gpu` use. It is adapted from ggml in ggml-org/llama.cpp at commit
`0cea36222`, the same commit as the ground-truth reference named at the top
of this file. The MIT license text that covers it is in the repository
`NOTICE`.

IQ decoders:

- Tables `kmask_iq2xs`, `ksigns_iq2xs`, `iq2xxs_grid`, `iq3xxs_grid` and
  `iq1s_grid` from `ggml/src/ggml-common.h`, with `NGRID_IQ1S` and
  `IQ1S_DELTA`.
- The arithmetic of `dequantize_row_iq2_xxs`, `dequantize_row_iq3_xxs` and
  `dequantize_row_iq1_m` from `ggml/src/ggml-quants.c`, as
  `hy4_dequant_iq2_xxs`, `hy4_dequant_iq3_xxs` and `hy4_dequant_iq1_m`.

K-quant and IQ4_XS decoders, taken from the same commit with renames only:
`dequantize_row_q4_K`, `dequantize_row_q5_K`, `dequantize_row_q6_K`,
`dequantize_row_iq4_xs` and `get_scale_min_k4` become
`hy4_dequant_row_q4_K`, `hy4_dequant_row_q5_K`, `hy4_dequant_row_q6_K`,
`hy4_dequant_row_iq4_xs` and `hy4_get_scale_min_k4`, with the
`kvalues_iq4nl` table.

Changes from ggml. The arithmetic is unchanged.

- `GGML_FP16_TO_FP32` is replaced by `hy4_fp16_to_fp32`, a plain IEEE
  binary16 to binary32 conversion that handles zero, subnormals, infinity
  and NaN.
- `GGML_TABLE_BEGIN` and `GGML_TABLE_END` are defined locally as plain
  `static const` arrays.
- The three IQ decoders take a raw byte pointer and a block count instead
  of a block-struct pointer and an element count. They step fixed strides
  of 66 (IQ2_XXS), 98 (IQ3_XXS) and 56 (IQ1_M) bytes per 256 values.
  IQ2_XXS and IQ3_XXS read the fp16 scale at the start of the block with
  `memcpy`. IQ1_M reads its 8 scale bytes at offset 48 of the block as four
  16-bit words and assembles the fp16 scale from their top 4 bits.
- The K-quant and IQ4_XS decoders keep the ggml signature: block pointer,
  element count `k`, `assert(k % QK_K == 0)`. Their block structs are
  redeclared with `uint16_t` for the fp16 fields, and `_Static_assert`s pin
  the sizes at 144 (Q4_K), 176 (Q5_K), 210 (Q6_K) and 136 (IQ4_XS) bytes.

Block geometry:

- The sizes used for types 12, 13, 14, 16, 18 and 29 (144, 176, 210, 66, 98
  and 56 bytes per 256 values) match `source_precision.ggml_type_geometry`
  in `model_contracts/hy4_ud_iq1m_authoritative.json`.
  `tools/hy4_tp16_shard.py` solves that table from the GGUF's tensor
  offsets; `--dry-census` prints it without writing bundles.
- Type 23 is an open question. `hy4_generate` and `hy4_stack_forward`
  decode it as IQ4_XS at 136 bytes per 256 values, but the contract lists
  type 23 at 72 bytes per 256 values, as does the sharder's `GGML_TYPES`
  table, and the contract notes that type ids follow the publisher's patched
  llama.cpp build. Settle which is right before trusting a type-23 decode.
