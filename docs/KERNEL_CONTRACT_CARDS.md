# Speculation Kernel Contract Cards

Registry of every speculation kernel in the tree, each as a filled **Part-1
contract card** from `docs/KERNEL_PLAYBOOK.md`. Model agents edit these cards
when they request kernel work; the CUDA-KERNELS agent implements against them
and updates `current_measured` whenever a number lands.

Card status legend: **`measured`** = a number exists in `PERFORMANCE_STATUS.md`
or a qualification receipt; **`NOT_MEASURED`** = no per-kernel or end-to-end
number yet (assembler/host/validator-tested only). No speculation path has a
measured decode number yet: every DSV4 and GLM52 receipt in
`PERFORMANCE_STATUS.md` is explicitly "no speculation"
(`PERFORMANCE_STATUS.md:102`, `581`, `586`).

Sources verified and cited `file:line` against the unified-branch clone at
`.agents/cuda-kernels/` (HEAD `5da43a5`).

---

## 0. Drafter shape table (the one place the shapes live)

The drafter shape is a compile-time target selected in
`model-families/common/include/sparkpipe/spark_dspark_drafter.h:1-89`. When the
K3 or Pro session requests its draft kernels, these are the starting-point
shapes (already pinned for GLM52/K3; partially declared for DSV4 Pro 0813).

| Field | GLM 5.2 | Kimi K3 | DSV4 Pro 0813 |
| --- | --- | --- | --- |
| block size (drafts) | 8 | 7 | 5 |
| verify positions | 8 (= block+1) | 8 (= block+1) | 8 (spec step+1) |
| aux/tap layer ids | {8,23,39,55,70} | {7,23,51,67,83} | {58,59,60} |
| draft layers | 5 | 5 | 3 |
| attention heads | 64 | 64 | **0 → pin mtp.1 (§4.8)** |
| KV heads | 64 | 16 (GQA) | **0 → pin mtp.1 (§4.8)** |
| head dim | 64 | 64 | **512 (expect; confirm §4.8)** |
| intermediate | 12288 | 14336 | **0 → pin mtp.1 (§4.8)** |
| markov rank | 256 | 256 | 512 |
| mask/noise token | 154856 | 163824 | 128799 |
| rope theta | 8000000.0 | 10000.0 | compressed 160000.0 |
| max spec tokens | 7 | 8 | 8 |

Sources: GLM52 `spark_glm52_model.h:55-67` + `spark_dspark_drafter.h:22-40`;
K3 `inference/llms/kimi_k3/dspark.h:17-83` + `spark_dspark_drafter.h:41-63`
(KV-head 16 is "the one that would not have announced itself", `dspark.h:29-31`);
DSV4 Pro 0813 `spark_dspark_drafter.h:64-86` +
`model_contracts/dsv4_pro_authoritative.json:32-42` (markov 512, noise 128799,
taps {58,59,60}; "Draft attention heads/intermediate are not declared by the
contract yet - zero until the Pro session pins them",
`spark_dspark_drafter.h:68-69`). The four Pro "0" cells are the DRAFT (mtp)
model's own geometry — distinct from the main model's 128 heads / 1 KV head /
head dim 512 / expert intermediate 3072 (`spark_dsv4_pro_model.h:31-36`) — and
are pinned from the GA mtp layer weight shapes, not the main constants (§4.8).

---

# 1. GLM52 DSpark draft backend (drafter)
`modules/glm52_dspark_draft_backend/source/spark_glm52_dspark_draft_backend.cu`
(11 kernels). All BF16 throughout (`draft_dtype BF16`,
`verifier_hidden_dtype BF16`, `spark_glm52_dspark_draft_backend.cu:179-180`);
weights are BF16 safetensors (`spark_glm52_dspark_draft_backend.cu:962-965`);
matrix multiplies go through `cublasGemmEx` (BF16, OP_T weights, COMPUTE_32F,
tensor-op, `spark_glm52_dspark_draft_backend.cu:1326-1362`), so only the
elementwise/norm/attention/argmax pieces are hand kernels here.

Model/backend constants: hidden 6144, vocab 154880, context 1048576, rms eps
1e-5, rope theta 8000000, mask token 154856, aux 5, draft layers 5, block 8,
max-spec-tokens 7, intermediate 12288, heads 64, KV heads 64, head dim 64,
markov rank 256, max anchors 1024 (`spark_glm52_model.h:5-67`); backend
threads 256, head threads 64, confidence dim = hidden+markov = 6400, fused input
dim = aux×hidden = 30720, attention dim = heads×head_dim = 4096, max lanes 1024
(`spark_glm52_dspark_draft_backend.cu:19-23`,
`spark_glm52_dspark_draft_backend.h:12-26`).

---

### G52-D-01 `SparkGlm52DsparkRmsNormRowsKernel`
`spark_glm52_dspark_draft_backend.cu:323-369`

- **requestor:** glm52 model agent (owns `modules/glm52_dspark_draft_backend`)
- **op:** RMSNorm one row (`row` of `dimension`), fp32 reduction, bf16 out.
- **shapes:** grid = `row_count` (one block/row, 256 threads); `dimension` =
  hidden 6144. Called for input norm, post-attention norm, and final norm
  (`spark_glm52_dspark_draft_backend.cu:1906-1915`, `2008-2017`, `2225-2233`).
- **dtypes:** in/weight/out bf16 (`uint16_t`); accumulation fp32; shared partials
  fp32 (`spark_glm52_dspark_draft_backend.cu:330`).
- **precision route:** bf16 in → fp32 square-sum → rsqrt((sum/dim)+eps) → bf16
  round of `value*inverse`, then bf16 round of `value*weight` (two bf16 rounding
  points, `spark_glm52_dspark_draft_backend.cu:361-367`); eps 1e-5.
- **target number:** NOT_MEASURED (contributes to GLM52 TP8 single-stream decode;
  6.91 tok/s B1 is the bandwidth-bound floor without speculation,
  `PERFORMANCE_STATUS.md:594-599`).
- **current measured:** NOT_MEASURED.
- **reference:** drafter `hidden_norm.weight` / `norm.weight` /
  `input_layernorm.weight` in `RedHatAI/GLM-5.2-speculator.dspark`
  (`spark_glm52_dspark_draft_backend.cu:26-27`, `1037-1039`, `1101-1102`).

### G52-D-02 `SparkGlm52DsparkAddBf16Kernel`
`spark_glm52_dspark_draft_backend.cu:371-387`

- **requestor:** glm52 model agent.
- **op:** in-place residual add, `destination += addition`, elementwise bf16.
- **shapes:** flat `element_count` = rows×hidden (rows = lane_count×block);
  256 threads; used after o_proj and after down_proj
  (`spark_glm52_dspark_draft_backend.cu:2073-2078`, `2112-2117`).
- **dtypes:** bf16 in/out; fp32 add with bf16 round on store.
- **precision route:** bf16 in → fp32 sum → bf16 round.
- **target number / current measured:** NOT_MEASURED.
- **reference:** residual add in the drafter forward; token-exact against the
  GLM52 speculator checkpoint's layer arithmetic.

### G52-D-03 `SparkGlm52DsparkSwigluRowsKernel`
`spark_glm52_dspark_draft_backend.cu:389-406`

- **requestor:** glm52 model agent.
- **op:** SwiGLU elementwise: `out = swish(gate) * up`, `swish(x)=x/(1+exp(-x))`.
- **shapes:** flat `element_count` = rows×intermediate (intermediate 12288);
  gate/up arrive interleaved as separate tensors (gate tensor, up tensor).
- **dtypes:** bf16 in/out; `expf` and multiply in fp32; gate swish rounded to
  bf16 before the multiply (`spark_glm52_dspark_draft_backend.cu:401-405`).
- **precision route:** bf16 → fp32 → bf16(swish) → fp32×up → bf16 out.
- **target number / current measured:** NOT_MEASURED.
- **reference:** drafter `mlp.gate_proj`/`up_proj`/SiLU activation in the
  speculator checkpoint (`spark_glm52_dspark_draft_backend.cu:1121-1126`).

### G52-D-04 `SparkGlm52DsparkGatherStageTapsKernel`
`spark_glm52_dspark_draft_backend.cu:408-430`

- **requestor:** glm52 model agent.
- **op:** gather tap rows: `stage_tap[stage][c] = tap_arena[tap_row_indices[stage]][c]`.
- **shapes:** `stage_count` rows × fused-input-dim 30720 (aux 5 × hidden 6144);
  grid = ceil(stage_count×30720 / 256), 256 threads
  (`spark_glm52_dspark_draft_backend.cu:1674-1686`).
- **dtypes:** bf16 in/out; u32 row indices.
- **precision route:** pure copy (bit-preserving).
- **target number / current measured:** NOT_MEASURED.
- **reference:** tap arena layout = `[lane][aux_layer][hidden]`
  (`spark_glm52_dspark_draft_backend.cu:1581-1591`); aux layers {8,23,39,55,70}.

### G52-D-05 `SparkGlm52DsparkScatterContextBatchKernel`
`spark_glm52_dspark_draft_backend.cu:432-463`

- **requestor:** glm52 model agent.
- **op:** scatter staged K/V into per-(lane,layer) context cache at the staged
  sequence position.
- **shapes:** `stage_count` rows × attention-dim 4096 (K and V, two outputs);
  destination `[lane][layer][max_context][attn_dim]` (`maximum_context_token_count`
  runtime, ≤ 1048576).
- **dtypes:** bf16; u32 lane/position indices.
- **precision route:** pure copy (bit-preserving).
- **target number / current measured:** NOT_MEASURED.
- **reference:** drafter self-attention KV layout in the speculator checkpoint.

### G52-D-06 `SparkGlm52DsparkBuildQueryBlockBatchKernel`
`spark_glm52_dspark_draft_backend.cu:465-488`

- **requestor:** glm52 model agent.
- **op:** build the draft block's hidden: position 0 = embedding of the anchor
  token; positions 1..block = embedding of the MASK token.
- **shapes:** `lane_count`×block rows × hidden 6144; grid = ceil(.../256).
  mask token id 154856 (`spark_glm52_model.h:66`).
- **dtypes:** bf16 embedding lookup (u32 token ids in).
- **precision route:** exact gather, no arithmetic.
- **target number / current measured:** NOT_MEASURED.
- **reference:** `embed_tokens.weight` `[vocab 154880 × hidden 6144]`
  (`spark_glm52_dspark_draft_backend.cu:1031-1033`); DSpark mask-token block
  construction (`docs/research-dspark/DSpark_paper.pdf`).

### G52-D-07 `SparkGlm52DsparkHeadNormRopeBatchKernel`
`spark_glm52_dspark_draft_backend.cu:490-554`

- **requestor:** glm52 model agent.
- **op:** per-head RMSNorm (q_norm / k_norm weight) then half-split RoPE.
- **shapes:** grid = (heads 64, row_count), 64 threads (one thread per head dim
  element, `head_threads = head_dim = 64`, `spark_glm52_dspark_draft_backend.cu:20-21`).
  Applies to both query and key, rows_per_lane = block (8) or 1
  (`spark_glm52_dspark_draft_backend.cu:1952-1975`, `1722-1734`).
- **dtypes:** bf16 in/out; fp32 norm/RoPE; `cosf/sinf` rounded to bf16.
- **precision route:** bf16 → fp32 norm → bf16 round → fp32 RoPE (half-split
  pairing) → bf16 out; rope theta 8000000.0, frequency
  `theta^(-2·i/head_dim)` (`spark_glm52_dspark_draft_backend.cu:533-553`).
- **target number / current measured:** NOT_MEASURED.
- **reference:** drafter `self_attn.q_norm`/`k_norm` weights
  (`spark_glm52_dspark_draft_backend.cu:1112-1115`).

### G52-D-08 `SparkGlm52DsparkBlockAttentionBatchKernel`
`spark_glm52_dspark_draft_backend.cu:556-700`

- **requestor:** glm52 model agent.
- **op:** block self-attention: online softmax over context tokens + block keys,
  then weighted sum over context + block values; three passes (max, denom, out)
  with the qk scale folded as `score·0.125` (`1/sqrt(head_dim)=1/sqrt(64)`).
- **shapes:** grid = (heads 64, row_count), 64 threads; attention dim 4096
  (64 heads × 64), context ≤ max_context_tokens.
- **dtypes:** bf16 q/k/v/out; fp32 online-softmax accumulation; `expf`.
- **precision route:** bf16 → fp32 dot (qk) → ×0.125 → bf16 round of score →
  fp32 online softmax → bf16 weight round → fp32 value accumulate → bf16 out
  (explicit bf16 rounds at score and weight,
  `spark_glm52_dspark_draft_backend.cu:623-624`, `692-693`).
- **target number / current measured:** NOT_MEASURED.
- **reference:** drafter `self_attn.{q,k,v,o}_proj` in the speculator checkpoint
  (`spark_glm52_dspark_draft_backend.cu:1103-1118`).

### G52-D-09 `SparkGlm52DsparkGatherMarkovBatchKernel`
`spark_glm52_dspark_draft_backend.cu:702-728`

- **requestor:** glm52 model agent.
- **op:** gather the Markov embedding: `embed[lane][r] = markov_w1[token][r]`,
  token = last_token (proposal 0) or generated_token[proposal-1].
- **shapes:** grid = `lane_count`, 256 threads; `markov_rank` = 256 elements
  per lane.
- **dtypes:** bf16 in/out; u32 token ids.
- **precision route:** exact gather.
- **target number / current measured:** NOT_MEASURED.
- **reference:** `markov_head.markov_w1.weight` `[vocab × 256]`
  (`spark_glm52_dspark_draft_backend.cu:1044-1046`); low-rank learned bigram bias
  `bias = W2(W1[token])` (`inference/llms/kimi_k3/dspark.h:70-74`).

### G52-D-10 `SparkGlm52DsparkArgmaxBatchKernel`
`spark_glm52_dspark_draft_backend.cu:730-797`

- **requestor:** glm52 model agent.
- **op:** greedy argmax of `block_logits + markov_logits` per lane and proposal
  index, over full vocab or a restricted-id table.
- **shapes:** grid = `lane_count`, 256 threads; candidate_count = 154880
  (or `restricted_token_count`); output `[lane][max_spec_tokens]` u32.
- **dtypes:** bf16 logits upcast to fp32; sum rounded to bf16 before compare
  (`spark_glm52_dspark_draft_backend.cu:764-770`); tie-break by lower token id.
- **precision route:** bf16+bf16 → fp32 → bf16 round → argmax (deterministic tie
  by id).
- **target number / current measured:** NOT_MEASURED.
- **reference:** `lm_head.weight` + `markov_head.markov_w2.weight`
  (`spark_glm52_dspark_draft_backend.cu:1041-1049`).

### G52-D-11 `SparkGlm52DsparkConfidenceBatchKernel`
`spark_glm52_dspark_draft_backend.cu:799-855`

- **requestor:** glm52 model agent.
- **op:** confidence head: `sigmoid(dot([hidden|markov_embed], weight) + bias)`,
  emitted in milli-units (×1000) for the scheduler.
- **shapes:** grid = `lane_count`, 256 threads; confidence dim = hidden+markov =
  6144+256 = 6400 (`spark_glm52_dspark_draft_backend.cu:22-23`); output
  `[lane][max_spec_tokens]` f32.
- **dtypes:** bf16 features/weight/bias; fp32 dot; f32 out.
- **precision route:** bf16 → fp32 dot → +bias → sigmoid → f32 (milli conversion
  on host, `spark_glm52_dspark_draft_backend.cu:2329-2336`).
- **target number / current measured:** NOT_MEASURED.
- **reference:** `confidence_head.proj.weight`/`bias`
  (`spark_glm52_dspark_draft_backend.cu:1050-1053`); `confidence_head_with_markov
  = 1` (input = hidden CONCAT markov, `inference/llms/kimi_k3/dspark.h:76-83`).

---

# 2. DSV4 Flash DSpark kernels (drafter)
`modules/dsv4_resident_decode_stage/source/spark_dsv4_dspark_kernels.cuh`
(5 kernels). Runs on ONE rank (the final head rank holding the full lm_head and
full-width hidden taps), so every kernel is full-width and communication-free
(`spark_dsv4_dspark_kernels.cuh:1-4`). Reference: `inference/model.py`
`DSparkBlock` / `DSparkAttention` / `DSparkMarkovHead`,
DeepSeek-V4-Flash-0731 @ 7872f01b (`spark_dsv4_dspark_kernels.cuh:6-7`).

Model constants: hidden 4096, vocab 129280, head dim 512, query heads 64, KV
heads 1, sliding window 128, hc streams 4, rms eps 1e-6
(`spark_dsv4_model.h:19-62`); dspark block 5, spec step 7, markov rank 256,
noise token 128799, target layers 3 (first 40)
(`spark_dsv4_model.h:36-43`); heads-per-CTA 4, CTA threads 256, CTA warps 8
(`spark_dsv4_sparse_attention_split.h:5`, `spark_lm_kernels.cuh:30-32`).

---

### D4-D-01 `SparkDsv4DsparkAttentionKernel`
`spark_dsv4_dspark_kernels.cuh:17-212` (launcher 214-237)

- **requestor:** dsv4-flash model agent (owns `modules/dsv4_resident_decode_stage`).
- **op:** draft attention: online softmax over the sequence's sliding-window ring
  (all 128 slots, rotation-free) PLUS the block's own 5..7 KV vectors, with the
  learned `attn_sink` added to the softmax DENOMINATOR only (no causal mask
  inside the block, no sink in the numerator — exact reference semantics,
  `spark_dsv4_dspark_kernels.cuh:9-14`).
- **shapes:** grid = (block_size, ceil(head_count/4)); `block_size` = spec step
  7 (`spark_dsv4_resident_decode_stage_module.c:4327,4419`), `head_count` 64,
  `head_dim` 512, `window_tokens` 128, 256 threads
  (`spark_dsv4_resident_decode_stage_module.c:4414-4422`).
- **dtypes:** bf16 q/kv/block_kv/out; f32 sink; fp32 online-softmax accumulation.
- **precision route:** bf16 → fp32 qk dot (fma) → ×scale (1/sqrt(512)) → online
  softmax (fp32) → sink into denominator (fp32) → value sum → bf16 out.
- **target number:** NOT_MEASURED (contributes to the 50 tok/s TP4 target,
  `PERFORMANCE_STATUS.md:276-278`).
- **current measured:** NOT_MEASURED.
- **reference:** `inference/model.py` `DSparkAttention`
  (`spark_dsv4_dspark_kernels.cuh:6-7`).

### D4-D-02 `SparkDsv4DsparkMarkovBiasAccumKernel`
`spark_dsv4_dspark_kernels.cuh:243-264` (launcher 266-281)

- **requestor:** dsv4-flash model agent.
- **op:** Markov logits bias: `bias = markov_w2[vocab_shard][rank] · embed[rank]`,
  accumulated into the draft logits (`logits_f32 = bf16 logits + bias`).
- **shapes:** grid = `multiprocessor_count`, 256 threads; `shard_count` = vocab
  shard (129280 across TP), `rank` = markov 256, `position` = 0..spec_step-1
  (`spark_dsv4_resident_decode_stage_module.c:4248-4253`).
- **dtypes:** bf16 logits/markov_w2/embed; f32 out (fp32 fma over rank).
- **precision route:** bf16 → fp32, rank-length fma dot, added to bf16 logit
  upcast → f32.
- **target number / current measured:** NOT_MEASURED.
- **reference:** `inference/model.py` `DSparkMarkovHead`
  (`spark_dsv4_dspark_kernels.cuh:6-7`); `markov_w2` `[vocab × rank]`
  (`spark_dsv4_stagepack_format.h:264`).

### D4-D-03 `SparkDsv4DsparkArgmaxKernel`
`spark_dsv4_dspark_kernels.cuh:284-323` (launcher 325-335)

- **requestor:** dsv4-flash model agent.
- **op:** greedy argmax over a full-vocab-shard f32 logits row (no logits tensor
  materialized; in-place tree reduce).
- **shapes:** 1 block, 256 threads; `shard_count` = vocab shard; output u32
  token id + f32 score (`spark_dsv4_resident_decode_stage_module.c:4255-4259`).
- **dtypes:** f32 logits in; u32 id + f32 score out.
- **precision route:** fp32 compare, tie not broken (strict `>`), deterministic
  first-winner.
- **target number / current measured:** NOT_MEASURED.
- **reference:** `inference/model.py` DSpark greedy head.

### D4-D-04 `SparkDsv4DsparkTapMeanKernel`
`spark_dsv4_dspark_kernels.cuh:339-358` (launcher 360-378)

- **requestor:** dsv4-flash model agent.
- **op:** hidden tap: mean over the 4 hyper-connection streams, per row
  (reference `h.mean(dim=2)`).
- **shapes:** grid = min(ceil(rows×dim/256), multiprocessor_count); `rows` =
  continuation rows, `stream_count` 4, `dimension` 4096
  (`spark_dsv4_resident_decode_stage_module.c:3770-3772`).
- **dtypes:** bf16 in/out; fp32 sum then /stream_count → bf16.
- **precision route:** bf16 → fp32 sum over streams → /4 → bf16.
- **target number / current measured:** NOT_MEASURED.
- **reference:** `inference/model.py` DSparkBlock tap (h.mean).

### D4-D-05 `SparkDsv4DsparkExpandStreamsKernel`
`spark_dsv4_dspark_kernels.cuh:383-399` (launcher 401-419)

- **requestor:** dsv4-flash model agent.
- **op:** expand `[rows × dim]` → `[rows × streams × dim]` (each stream a
  copy); the draft block's input expansion (reference
  `x.unsqueeze(2).repeat(1,1,hc_mult,1)`, `spark_dsv4_dspark_kernels.cuh:380-382`).
- **shapes:** `rows` = spec step 7, `stream_count` 4, `dimension` 4096
  (`spark_dsv4_resident_decode_stage_module.c:4361-4362`).
- **dtypes:** bf16 in/out.
- **precision route:** exact copy (bf16 preserved).
- **target number / current measured:** NOT_MEASURED.
- **reference:** `inference/model.py` DSparkBlock input expansion.

---

# 3. Shared verifier kernels
`inference/kernels/speculate.cuh` (2 kernels). These are the
target-agnostic accept/reject step shared by MTP and DSpark: "the drafter is a
policy and the verifier is a kernel" (`speculate.cuh:12-13`).

---

### V-01 `LmSpeculativeVerifyGreedyKernel`
`inference/kernels/speculate.cuh:42-64`

- **requestor:** SUBSYSTEM speculation agent (shared; consumed by model agents).
- **op:** greedy verification: longest prefix of the draft the target agrees with;
  write accepted tokens, the bonus token (target's own argmax at the divergence
  point), the accepted count, and roll the context length back
  (`speculate.cuh:34-41`, `55-63`).
- **shapes:** one block per sequence, `THREADS` (single thread does the walk);
  `draft_length` = gamma (block drafts); `context_length` u32 per sequence.
- **dtypes:** u32 draft/target tokens, u32 accepted count/committed/context.
- **precision route:** exact comparison (`==`); no float.
- **target number:** correctness: speculation must never lose (bonus token
  guarantees ≥1 token/step even at acceptance rate 0, `speculate.cuh:55-59`).
- **current measured:** NOT_MEASURED (external SGLang reference: accept length
  ≈2.7 on chat traffic, `inference/llms/kimi_k3/dspark.h:98-100`).
- **reference:** DSpark/MTP verify loop; `docs/research-dspark/DSPARK_VERIFY_LOOP_REPORT.md`;
  `docs/research-dspark/DSpark_paper.pdf`.

### V-02 `LmSpeculativeVerifySampledKernel`
`inference/kernels/speculate.cuh:76-130`

- **requestor:** SUBSYSTEM speculation agent.
- **op:** sampled verification preserving the target distribution: accept draft
  token `t` with prob `min(1, p_target(t)/p_draft(t))`; on rejection resample
  from the normalised positive part of `(p_target - p_draft)`
  (`speculate.cuh:66-75`, `105-107`).
- **shapes:** one block per sequence, `THREADS`; `draft_length` (gamma);
  `vocabulary` full-vocab row width; f32 draft/target probabilities and uniform
  draws.
- **dtypes:** u32 tokens; f32 probabilities/uniform; f32 block reduction
  (`LmBlockSum`).
- **precision route:** fp32 comparison; `fmaxf(pd,1e-20f)` guards the ratio;
  residual resample over `fmaxf(0, p_t - p_d)` (NOT the raw target — raw
  target over-proposes drafter-preferred tokens, `speculate.cuh:105-107`).
- **target number / current measured:** NOT_MEASURED.
- **reference:** modified rejection sampling (standard spec-decode); greedy is
  NOT a special case of sampled and must not be collapsed into it
  (`speculate.cuh:15-20`); `docs/research-dspark/DSpark_paper.pdf`.

---

---

# 4. DSV4 Pro DSpark native pass (P-D-01..06) — reviewed
Requested by the dsv4-pro agent in `docs/PROPOSAL_DSV4_PRO_DSPARK_PASS.md`
(landed `e9abfc2`); reviewed and completed by the CUDA-KERNELS agent below.
All six are **NOT_MEASURED** and every card is **blocked on the §0 draft-head/
intermediate pin** (P-D-03 names `head_count`); see §4.8. Pro draft context:
3-layer mHC block (`mtp_layer_count 3`, `dsv4_pro_authoritative.json:9`),
hidden 7168, markov 512, noise 128799, taps {58,59,60}
(`dsv4_pro_authoritative.json:32-42`), draft KV BF16 with no rotary
(first-light, `PROPOSAL_DSV4_PRO_DSPARK_PASS.md:44-46`).

### P-D-01 mean-reduce (tap capture)
- **requestor:** dsv4-pro model agent.
- **op:** mean over the 4 hyper-connection streams of the post-layer hidden for
  tap layers 58-60 → `[3][7168]` mean taps (reference `h.mean(dim=2)`).
- **shapes:** rows 5 (block), dim 7168, 3 taps; 4 hc streams
  (`spark_dsv4_pro_model.h:39`).
- **dtypes:** bf16 in/out; fp32 sum then /4 → bf16.
- **precision route:** bf16 → fp32 sum over streams → /4 → bf16 (RNE).
- **target number / current measured:** NOT_MEASURED (proposal headline: main-only
  12-13 tok/s → draft ~45-60 tok/s, `PROPOSAL_DSV4_PRO_DSPARK_PASS.md:3-4`).
- **reference:** `inference/model.py` DSparkBlock tap.
- **review:** REUSE `SparkDsv4LaunchDsparkTapMean` (D4-D-04) with `dimension`
  7168, `stream_count` 4 — the Flash kernel is runtime-parameterized
  (`spark_dsv4_dspark_kernels.cuh:339-358`). No new kernel.

### P-D-02 main-KV write
- **requestor:** dsv4-pro model agent.
- **op:** `kv_norm(wkv(main))` → rolling main-KV window `[128][512]` at
  `seq % 128` (the draft attention's window source).
- **shapes:** window 128 slots × head-dim 512 bf16; one row per step.
- **dtypes:** bf16 in/out; kv_norm = per-head RMSNorm, fp32 → bf16.
- **precision route:** bf16 → fp32 norm → bf16 → window write (bit-preserving store).
- **target number / current measured:** NOT_MEASURED.
- **reference:** `inference/model.py` main KV + kv_norm; `mtp` checkpoint tensors.
- **review:** NEW — Flash has no standalone rolling-window writer; its draft reads
  the main model's ring directly (`spark_dsv4_dspark_kernels.cuh:103-107`). Small
  write kernel.

### P-D-03 draft attention
- **requestor:** dsv4-pro model agent.
- **op:** online softmax over the main-KV window (128) + the draft's own KV (5),
  with `attn_sink` in the denominator; scale `1/sqrt(head_dim)`.
- **shapes:** `head_count` = **TBD (§4.8)**, `head_dim` 512, window 128,
  block/rows 5, 256 threads (mirrors D4-D-01).
- **dtypes:** bf16 q/kv/out; f32 sink; fp32 online-softmax accumulation.
- **precision route:** bf16 → fp32 qk (fma) → ×1/sqrt(512) → online softmax → sink
  into denominator → value sum → bf16.
- **target number / current measured:** NOT_MEASURED.
- **reference:** `inference/model.py` DSparkAttention.
- **review:** REUSE `SparkDsv4LaunchDsparkAttention` (D4-D-01) with
  `head_count`=pinned, `head_dim` 512, `window` 128, `block` 5. Two caveats:
  (1) the accumulator is sized off `SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION` (512)
  so it is compatible with Pro (`spark_dsv4_dspark_kernels.cuh:32-34`);
  (2) Flash attention is NON-causal in-block (`spark_dsv4_dspark_kernels.cuh:13`)
  while P-D-03 wants CAUSAL draft KV — needs a `causal` flag or a Pro variant.

### P-D-04 markov bias
- **requestor:** dsv4-pro model agent.
- **op:** `logits += markov_w2[vocab_shard][512] · markov_embed[512]` accumulated
  into the draft head logits (f32).
- **shapes:** vocab shard (129280 across TP) × rank 512; position 0..4.
- **dtypes:** bf16 logits/w2/embed; f32 out (fma over rank).
- **precision route:** bf16 → fp32, rank-length fma, added to bf16 logit upcast → f32.
- **target number / current measured:** NOT_MEASURED.
- **reference:** `inference/model.py` DSparkMarkovHead; `markov_w2` `[vocab × 512]`
  (`PROPOSAL_DSV4_PRO_DSPARK_PASS.md:16-17`).
- **review:** REUSE `SparkDsv4LaunchDsparkMarkovBiasAccum` (D4-D-02) with
  `rank` 512 (runtime arg). The "head U64 max" combine is real (vocab-sharded
  head), NOT a draft-attention/FFN all-reduce.

### P-D-05 confidence
- **requestor:** dsv4-pro model agent.
- **op:** `sigmoid(dot([hidden | markov_embed], w) + b)`, one scalar per position.
- **shapes:** input dim = hidden + markov = 7168 + 512 = 7680 (matches the
  proposal's `1 x 7680`); output 1 f32.
- **dtypes:** bf16 features/weight/bias; fp32 dot; f32 out.
- **precision route:** bf16 → fp32 dot → +bias → sigmoid → f32.
- **target number / current measured:** NOT_MEASURED.
- **reference:** `confidence_head.proj` / `bias`; confidence-with-markov
  (`inference/llms/kimi_k3/dspark.h:76-83`).
- **review:** NEW — Flash has no confidence kernel. Port GLM52
  `SparkGlm52DsparkConfidenceBatchKernel` (G52-D-11) with input 7680
  (`spark_glm52_dspark_draft_backend.cu:799-855`, `22-23`).

### P-D-06 draft mHC layer forward
- **requestor:** dsv4-pro model agent.
- **op:** 3 × (draft attention + FFN: bias-gate top-6 mtp experts) over the 5
  draft rows, mirroring the main layer kinds.
- **shapes:** mirrors the main layer geometry per layer kind; 5 rows.
- **dtypes:** mixed (main-layer precision: FP8 experts MXFP4, BF16 spine, per
  `spark_dsv4_pro_model.h:47-65`).
- **precision route:** main-layer continuation arithmetic on `state->mtp_layers[i]`.
- **target number / current measured:** NOT_MEASURED.
- **reference:** `SparkDsv4ModuleContinueLayers` over `state->mtp_layers[i]`
  (`PROPOSAL_DSV4_PRO_DSPARK_PASS.md:53-56`).
- **review:** NOT a new kernel — reuse the existing main-layer continuation on
  `state->mtp_layers[i]` (the proposal's §3 step 3 already says "continuation
  mirroring `SparkDsv4ModuleContinueLayers`"). The "FFN-side all-reduce" is the
  existing TP reduce and is inapplicable if the draft is single-rank (§4.8).

---

### 4.8 Answers to the pro agent's open questions

**Q1 — what should pin the four 0 constants (`spark_dspark_drafter.h:82-85`)?**

They are the DRAFT (mtp) model's own transformer geometry, distinct from the main
model's 128 heads / 1 KV head / head dim 512 / expert intermediate 3072
(`spark_dsv4_pro_model.h:31-36`). Pin them from the GA checkpoint's mtp layer
WEIGHT SHAPES — the only authoritative source, never prose:

- `DRAFT_ATTENTION_HEAD_COUNT` = `mtp.1.self_attn.q_proj.weight` output dim ÷ head_dim
- `DRAFT_KV_HEAD_COUNT`      = `mtp.1.self_attn.k_proj.weight` output dim ÷ head_dim
- `DRAFT_HEAD_DIMENSION`     = head dim (expect **512**: the draft attends the
  main's 512-wide KV window — the proposal already uses scale 1/sqrt(512) and
  window 128×512)
- `DRAFT_INTERMEDIATE_DIMENSION` = `mtp.1.mlp.gate_proj.weight` output dim

Land it DRY/generator-consistent so it cannot silently drift (the K3 "64 vs 16 KV
heads" lesson, `inference/llms/kimi_k3/dspark.h:29-31`): (1) add the four fields
to the `dsv4_pro_authoritative.json` dspark block (`:32-42`); (2) extend
`tools/generate_dsv4_contracts.py` to emit them — today it emits only
block/layer/markov/noise (`:222-228`, `:274-279`) and requires only those in the
Pro block (`:166-170`), so `--check` will NOT reproduce a hand-edited header;
(3) write the four into `spark_dspark_drafter.h:82-85`; (4) add a pinning test
mirroring `tests/test_dspark_drafter_pin.c` so the Pro table is asserted, not
assumed.

**Q2 — launcher surface vs the Flash dspark kernels?**

Reuse the Flash kernels wherever they are runtime-parameterized (they are), and add
a new launcher only where Flash has nothing:

| Card | Disposition | Target |
| --- | --- | --- |
| P-D-01 | reuse | `SparkDsv4LaunchDsparkTapMean` (dim 7168, streams 4) |
| P-D-02 | new | no Flash rolling-window writer |
| P-D-03 | reuse + `causal` flag | `SparkDsv4LaunchDsparkAttention` (heads=pin, dim 512, window 128, block 5) |
| P-D-04 | reuse | `SparkDsv4LaunchDsparkMarkovBiasAccum` (rank 512) |
| P-D-05 | new (port G52-D-11) | no Flash confidence kernel |
| P-D-06 | reuse (wiring only) | `SparkDsv4ModuleContinueLayers` on `mtp_layers[i]` |

**Q3 (flagged, not asked) — single-rank vs all-reduce.** The Flash draft runs on
ONE rank, communication-free (`spark_dsv4_dspark_kernels.cuh:1-4`). The proposal's
P-D-03 "attn-side all-reduce (5 rows)" and P-D-06 "FFN-side all-reduce" therefore
do not apply unless the Pro draft is deliberately TP-sharded — a divergence from
Flash that needs explicit justification. The only real cross-rank collective on the
draft path is the vocab-sharded head U64 maxloc (P-D-04). Resolve this before
landing P-D-03/P-D-06.

---

## 5. How model agents edit these cards

1. Change `requestor` to your lane, keep `op` one line.
2. Replace `target number` with the concrete number to beat and its baseline
   (leave `NOT_MEASURED` in `current measured` until a receipt exists).
3. When K3 or DSV4-Pro sessions request their draft kernels, copy the relevant
   row of §0 into `shapes` and fill the "0 (not declared)" fields first —
   those are blocking unknowns, not defaults.
4. Every card must stay contract-complete; a field you cannot yet state is
   written "TBD (reason)", never silently inferred (the "correct at rows==1"
   bug class, `inference/kernels/norm.cuh:81-90`).
