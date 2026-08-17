# DSpark MTP verify/acceptance loop — exact semantics from community implementations
Researched 2026 from: HF reference inference/model.py (forward_spec, DSparkBlock, DSparkMarkovHead,
DSparkConfidenceHead, DSparkAttention, lines ~744-962); vLLM PR #46995 (benchislett) + current vLLM main
(v1/worker/gpu/spec_decode/dspark/speculator.py, dflash/speculator.py, rejection_sampler.py,
rejection_sampler_utils.py, v1/worker/gpu/input_batch.py, sample/gumbel.py,
models/deepseek_v4/nvidia/dspark.py, config/speculative.py); joesinvestments TP4 recipe @1339c163;
tonyd2wild 1M-NVFP4 recipe; antirez/ds4 issue #468; DSpark paper (arXiv 2607.05147); ktransformers #2118.

## Checkpoint facts (DeepSeek-V4-Flash-0731 config.json)
dspark_block_size=5, dspark_target_layer_ids=[40,41,42], dspark_noise_token_id=128799,
dspark_markov_rank=256, vocab_size=129280, num_hidden_layers=43 (draft layers 43/44/45 = stages 0/1/2),
sliding_window=128, compress_ratios ends ...,4,128,4,0,0,0 (draft layers use the raw uncompressed SWA path),
n_mtp_layers absent in config (vLLM defaults to 3). Draft = 3 full MoE decoder layers
(attn + 256 routed + 1 shared expert) under mtp.{0,1,2}.*, plus mtp.2.markov_head.{markov_w1,markov_w2}
and mtp.2.confidence_head.proj.

## 1. What k=7 means
In the community serving builds k = num_speculative_tokens = number of DRAFT tokens per step, in ONE
semi-autoregressive block, verified by ONE target forward of (k+1) positions. It is NOT "5+2", NOT
"5 draft + 2 verified": 7 draft tokens, 8 slots per request in the verify batch, tok/step ceiling k+1=8.
- joesinvestments RECIPE.md: --speculative-config '{"method":"dspark","num_speculative_tokens":7,
  "draft_sample_method":"probabilistic"}'; 6.019 tok/step = 1 + 7 x 0.717 (aggregate acceptance 71.7%);
  max-num-batched-tokens 8264 = 8192 + (k-1) x seqs; max-cudagraph-capture-size = seqs x (k+1).
- vLLM PR #46995 DSparkSpeculator: num_query_per_req = num_speculative_steps (not block_size);
  "anchor + N-1 noise" query layout; sample_from_anchor=True; the draft model never reads
  dspark_block_size. A 7-wide draft = a wider single block. Positions 5-6 extrapolate beyond the
  trained block_size 5; joe measured 79.4% acceptance at position 6, so it works.
- tonyd2wild README confirms the split: vLLM 0.21.x+B12X hardcodes draft width = dspark_block_size=5
  (k=7 rejected by a divisibility guard or silently drafts 5); the anemll 0.25.2 image
  (ghcr.io/anemll/dspark-vllm-gx10:0.1.1, joe's) sizes the block from num_speculative_tokens.
  "Omit num_speculative_tokens and you get k=1, not k=5."
- The DSpark paper trains block y=7 drafts (deepseek-ai/dspark_qwen3_4b_block7); vLLM's e2e test uses
  num_speculative_tokens:7 with that draft.

## 2. Exact acceptance algorithm
Acceptance is STANDARD speculative-decoding rejection sampling (Leviathan et al. 2023) — NOT a
confidence threshold, NOT Gumbel-equality. The confidence head is DROPPED in vLLM
(load_weights: "confidence_head." -> return None) and unused by joe's build
(VLLM_DSPARK_CONFIDENCE_SCHEDULER=off exists in tonyd2wild's overlay but is off); it is used only by
the paper's load-aware verifier (chooses per-request verification length, a scheduling decision —
verification itself is still lossless rejection sampling).

Per step, per request (P = position of the last emitted token = anchor; k = num_speculative_tokens):
A. DRAFT (propose):
  1. Query tokens: query_pos = last_valid_pos+1+query_off; query 0 = last_sampled (anchor token) at
     its real position P; queries 1..k-1 = noise_token_id 128799 at P+1..P+k-1. sample_pos = query_pos+1
     for ALL queries (anchor predicts the first draft token). Draft tokens predicted at P+1..P+k.
  2. Draft backbone forward over the k query positions (one parallel pass, non-causal sparse SWA
     attention over [window(128) context KV + all block positions]).
  3. base_logits = lm_head(norm(hc_head(draft_hidden))) — full vocab, gathered once for all k positions.
  4. Sequential Markov loop (the "semi-autoregressive" stage), prev = anchor:
     for i in 0..k-1: bias_i = markov_w2(markov_w1(prev)); logits_i = base_logits[i] + bias_i;
       draft_i = sample(logits_i); prev = draft_i.
     sample = gumbel-max with temperature (probabilistic) or argmax (greedy); gumbel key = P+i
     (the predecessor/slot position). Draft logits stored pre-temperature for the verifier.
B. VERIFY:
  1. Target forward over [anchor at P] + [k drafts at P+1..P+k] (ONE forward, k+1 slots/request;
     positions P..P+k; logits at slot i predict position P+i+1).
  2. For i in 0..k-1 (stop at first rejection):
     Greedy (temp==0):  accept draft_i iff target_argmax(slot i) == draft_i; on first mismatch emit
       target_argmax and stop.
     Probabilistic:   u_i = tl_rand32(req_seed, slot_pos, includes_zero=False); accept iff
       u_i < p_t(draft_i)/q_d(draft_i), where p_t = softmax(processed target logits at slot i),
       q_d = softmax(draft_logits[i] / temp). On first rejection, emit a token resampled from the
       residual r(x) proportional to max(p_t(x) - q_d(x), 0) at that slot and stop. (vLLM
       rejection_sampler_utils _rejection_kernel: accepted = target_logprob > log(u) + draft_logprob;
       _resample_kernel: residual_logits = target_logprob + log1p(-exp(draft_logprob-target_logprob))
       where ratio<1.)
  3. If all k accepted: bonus token = sample from target logits at slot k (the distribution at
     position P+k+1 — the target never needs an input token there).
  4. Emit num_sampled = accepted + 1 tokens (accepted prefix + replacement-or-bonus).
C. NEXT STEP: anchor = last emitted token (bonus or replacement). Rejected suffix (k - accepted
   positions) is NOT emitted; the next block re-drafts from the new anchor (the previously rejected
   positions are re-covered by the new draft block — vLLM passes num_rejected so the input kernel
   excludes them from "valid context"; draft placeholders -1 only appear as padding).
Loop structure: exactly ONE target forward per step (the verify), plus ONE draft forward; the
sequential markov loop is pure GPU compute on the draft head (k small GEMMs, no host launches).
All-accept case: next draft rides the accepted prefix ONLY via the anchor token + accumulated
context KV — the full draft backbone forward is recomputed for the new block (semi-autoregressive
design; no draft-token reuse).

Exactness: the ratio test with u~U(0,1) is Leviathan's rejection sampling (lossless); the residual
max(p-q,0) resample reproduces the conditional target distribution on rejection; greedy equality is
exact by construction. vLLM asserts output-identity: e2e GSM8K accuracy at temp=1 with DSpark >= non-spec
baseline (0.801 mean); ds4/Metal reports token-identical temp=0 output; joe's bst_parity uses per-position
metrics. IMPORTANT implementer notes: (a) u is keyed by (req_seed, slot position) for deterministic CUDA-graph
replay — any fresh per-position uniform works; (b) draft and target must share temperature semantics:
draft_logits are stored PRE-temperature and q_d applies /temp, p_t uses the post-sampling-params target
logits; (c) only temperature (not top-k/top-p) is applied to the draft (vLLM: "may slightly degrade
acceptance, does not affect the output distribution").

## 3. Draft logits under TP4 vocab parallelism
- vLLM: markov_w1 = nn.Embedding(vocab x 256) and markov_w2 = ParallelLMHead(256 x vocab, disable_tp=True)
  are REPLICATED on every rank ("Sharding them would add an all-reduce and a full-vocab gather to each
  position"). The k sequential bias steps therefore do ZERO communication. The base head (shared
  target lm_head, vocab-sharded 32320 rows/rank at TP4) is all-gathered ONCE per step to full vocab
  (LogitsProcessor gather of [num_reqs x k, 32320]) — a single block-wide gather, then all k markov
  biases are added and sampled locally on full logits (position-seeded Gumbel -> identical tokens
  on every rank). Sampling happens on every rank with the same gathered logits.
- Reference model.py (single-node demo) instead does full_logits=True all_gather per head call:
  1 base + k markov = 6 gathers/block at k=5 (unoptimized).
- tonyd2wild knobs: VLLM_DSPARK_REPLICATE_MARKOV_W1=1 (replicate w1), VLLM_DSPARK_LOCAL_ARGMAX=1
  (greedy draft argmax via local shard + reduce instead of gathering logits),
  VLLM_DSPARK_EXPORT_DRAFT_PROBS=1 (export draft probs for the rejection sampler).
- For C/CUDA TP4: replicate markov_w1 (129280 x 256) and markov_w2 (256 x 129280) on all 4 ranks
  (~66 MB each in bf16; negligible on GB10 128 GB) and do ONE vocab all-gather per step for the base
  logits; or local-argmax + all-reduce for greedy drafts. Do NOT gather per sampled position.
- Aux hidden states: target layers 40/41/42 outputs are mean-pooled over hc_mult copies, then
  sp_all_gather (sequence-parallel all-gather over hidden dim) to replicate, concatenated (dim*3),
  main_x = main_norm(main_proj(concat)); main_proj is ReplicatedLinear (replicated weights).

## 4. Draft forward placement + KV/markov state
- Placement: NOT restricted to a last TP rank. vLLM runs the draft as a separate TP-parallel model on
  ALL ranks (PP must be 1 — NotImplementedError otherwise). Draft layers are TP-sharded
  DeepseekV4DecoderLayer (same MoE/quant as target); embed_tokens and lm_head are ALIASED (shared
  pointers) to the target's. Step order per decode step: target forward (captures aux hiddens) ->
  combine main_x -> precompute_and_store_context_kv (each draft layer derives its SWA context KV
  from main_x via fused wq_a|wkv -> kv_norm -> RoPE -> quant -> insert into its sliding-window
  cache) -> draft forward (k query tokens) -> sequential markov sampling -> (next step) target
  verify forward.
- KV: the draft layers keep their OWN sliding-window KV cache (window=128, uncompressed path),
  populated from MAIN-branch KV at every target position each step (precompute over all target
  slots, including the current drafts). Block-position KV of the draft is transient (in-kernel
  concat, never cached). The reference model.py writes main-branch KV at one position per
  forward_spec call (kv_cache[start_pos % win] = main_kv). Draft attention is NON-CAUSAL within the
  block in vLLM (index list = trailing window + all block positions; topk_width 256 for (128,5));
  the reference uses the same index-driven sparse attn with the block in the top-k list.
- Markov state: first-order only — prev = previously sampled token (anchor at block start); no
  persistent recurrent state, no cross-sequence state, no cross-block state except the anchor token.

## 5. Reported acceptance rates
- joe (TP4, k=7, probabilistic draft, temp-0 target): aggregate 71.7%; per-position
  95.0/92.2/91.2/91.0/90.3/86.0/79.4 (pos 0-6); 6.019 tok/step; 48.83 ms/step; 123.13 tok/s.
- techmd (TP4, k=6, greedy draft, temp-0): aggregate 79.6%; 96.2/90.0/83.5/76.3/69.5/62.3;
  5.777 tok/step; 104.17 tok/s. joe: "his curve collapses with depth; ours holds".
- Greedy vs probabilistic: joe's RESULTS.md — greedy draft = point mass => acceptance collapses to
  p_target(argmax), roughly flat; probabilistic matches the target distribution and the curve decays
  gracefully, which is what makes k=7 pay. Their production bug: DeepSeek served with greedy draft
  ran 27-48% acceptance (measured live 48.1%, 3.41 tok/step at default temp 1.0) vs ~80% with
  probabilistic. (Note: on SOME vLLM builds draft_sample_method is a no-op — tonyd2wild's 0.21.x
  overlay — so the knob only bites on the PR-46995/anemll lineage.)
- tonyd2wild (TP2, 0731, k=5, patched shared-expert): best-case structured 98.9% (5.95/6);
  pooled ~35 min agent traffic 56.1% (mean accepted 4.0-5.0 of 5); per-position after fix
  0.826/0.725/0.572/0.471/0.399; prose reasoning 33.7-37.8%; k=5->k=7 bought +3.3% tok/step for +33%
  draft cost. Unpatched (draft shared-expert uninitialised): 0.631/0.282/0.181/0.114/0.067.
- ds4 Metal (Q4_K draft): ~2.2-3.2 accepted of 5; 2.3 (temp0) / 2.1 (temp1) on a broad corpus.
- vLLM e2e (Qwen3-4B + block7 draft, temp 1.0): acceptance_rate 0.428, acceptance_len 3.994
  (= 1 + 7 x 0.428). Paper (Qwen3-4B, y=7, temp 1.0): accepted length tau incl. bonus 3.29-6.11.
- 34.3% vs 26.5% (design doc) — RESOLVED in follow-up: these are joe's production-shape
  measurements (agentic tool-calling traffic, default temp 1.0) in RECIPE.md, `draft_sample_method`
  table: greedy 26.5% accept / 2.86 tok/step vs probabilistic 34.3% / 3.40 tok/step (~19% more
  output speed). The greedy-bug range 27-48% and ds4 2.1-2.3/5 (42-46%) corroborate.

## 6. Noise fill + block boundaries
- First block after prefill: reference forward_spec(start_pos==0) returns after the KV-only pass
  (no drafting); the first real draft block anchors on the last prefill token. vLLM additionally
  splices next_prefill_tokens during chunked prefill. Noise fill is exactly: query 0 = anchor (real
  token), queries 1..k-1 = noise_token_id 128799.
- Block boundaries: the draft block NEVER overlaps the verified prefix — it always predicts the next
  k positions after the anchor. Each step re-runs the ENTIRE draft backbone (fresh noise, fresh
  parallel forward); no draft-token carryover. Cross-block state = anchor token + accumulated
  main-branch context KV (sliding window 128) in the draft layers' own caches. Markov state resets
  to the anchor at each block (first-order).
- All-k-accepted: next block anchors on the bonus token (P+k+1) and drafts P+k+2..P+2k+1 — no
  re-verification of the accepted prefix; the verify forward always covers exactly [anchor + k new
  drafts].

## Key source URLs
- HF reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py
- vLLM PR: https://github.com/vllm-project/vllm/pull/46995
- joe TP4 recipe: https://github.com/joesinvestments/DeepSeek-V4-Flash-0731-TP4-4x-DGX-Spark
- tonyd2wild 1M recipe: https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
- DSpark paper: https://arxiv.org/abs/2607.05147 (HTML: https://arxiv.org/html/2607.05147)
- Review: https://www.zhongzhuzhou.org/blog/2026-07-08-dspark-technical-review-en/
- ds4 Metal thread: https://github.com/antirez/ds4/issues/468
- ktransformers (sglang): https://github.com/kvcache-ai/ktransformers/issues/2118
- vllm.cpp reference impl: https://github.com/mudler/vllm.cpp/blob/main/tests/vllm/models/test_deepseek_v4_mtp.cpp

---

## 7. SPECULATOR QUANTIZATION (follow-up)

### 7.1 Checkpoint precision layout (what ships in the weights)
- Draft experts: already MXFP4 (config `expert_dtype: fp4`, same family as the base model; `w1/w3/w2` with per-block e8m0 scales).
- Draft projections/linears (attention `fused_wqa_wkv`, `wo_a/wo_b`, `wq_b`, hc_head_fn, main_proj): fp8 e4m3 with ue8m0 scales (128x128 blocks) — the base `quantization_config`.
- Small heads: markov_w1/markov_w2 and confidence_head.proj are stored bf16/fp32 by design (reference: DSparkConfidenceHead uses fp32 Linear; vLLM loads them as fp32/bf16 params).
- vLLM PR #46995 loads the draft with the TARGET's quant_config (`if not self.quantization: self.quantization = self.target_model_config.quantization`) — linears fp8, experts MXFP4. No vLLM build quantizes the draft linears to 4-bit.

### 7.2 Who quantizes the draft, and measured effects
- **ds4 / Metal (lobanov et al.) — the only full 4-bit draft in the wild:** Q4_K GGUF of the ENTIRE draft (3 layers ≈ 19.7B params; q8 ≈ 20 GB, q4 ≈ 11 GB, iq2 ≈ 6 GB; markov+confidence+norms ≈ 70 MB). Measured: **Q4_K vs F16/F32 draft acceptance difference < 0.4% on the F32 numpy oracle** ("even full-precision mixed-tensor DSpark drafter does not perform better"); the dominant acceptance killer was quantizing the TARGET (q2 base perturbs the layer-40/41/42 hiddens the draft consumes). **F32 accumulation over Q4_K weights** recovered Metal acceptance 1.53 → 2.2 (accumulation precision matters more than weight quant). Q2_K draft only 2.5% worse than Q4_K (~5 GiB); mixed-quant knee ~7.5 GiB ~0.6% worse. Metal draft validation gate: max-abs logit diff < 1e-2 @ fp16.
- **tonyd2wild:** deliberately keeps draft weights STOCK ("Edited draft weights would land you back in acceptance collapse" — their measured collapse: shared-expert loader bug dropped acceptance 60.2% -> 25.7%, per-pos 0.826/.../0.399 -> 0.631/0.282/0.181/0.114/0.067). Their "NVFP4" = **KV cache dtype only** (`kv_cache_dtype=nvfp4_ds_mla`), not draft weights.
- **No AWQ/GPTQ/INT4-SqQuant work on the DSV4 DSpark draft found in the community** (vLLM, sglang, ktransformers, vllm.cpp all run the draft in checkpoint precision).
- **Acceptance-gate safety:** the exact-token gate protects the target; every community build confirms a degraded draft only costs speed, never output correctness (tonyd2wild: "output quality was perfect — no garble... Only speed changed"; ds4: token-identical at temp 0 across draft quants).

### 7.3 vLLM PR #47584 — "Rowwise-fp8 draft lm_head for DSpark (opt-in)" (bird, open)
- **Mechanism:** `VLLM_DSPARK_FP8_DRAFT_HEAD=1` (SM89+): at load, quantize the local lm_head shard once to rowwise fp8-e4m3 (`w8 = w*(448/rowmax)`, `row_scale = rowmax/448`, +~0.5 byte/param ≈ 130 MB at TP1 for 64640x4096); at draft time, dynamic per-token activation quant + `torch._scaled_mm(a8, w8.T) * row_scale * (amax/448)` then the SAME TP gather + vocab-padding slice as LogitsProcessor._get_logits; the VERIFY path and target logits never see the fp8 copy (acceptance unchanged, proposals argmax-identical on their eval set).
- **Measured (GB10, ~235 GB/s unified):** draft-head GEMM **2.73 ms -> 1.45 ms per draft step (-47%)**; end-to-end single-stream **+3-5%**; per-position acceptance unchanged. Opt-in because on compute-rich datacenter parts the head GEMM is a much smaller share of the step.
- **Why it transfers to 4-bit:** the draft head/backbone is weight-read-bound on GB10; PR #47584 proves the pattern — quantize a draft-only path, keep the target's logits path untouched, gather/argmax semantics preserved. fp8->fp4 halves the reads again (~2x on the head; more on the always-read dense parts).

### 7.4 tonyd2wild NVFP4-KV, decoded
- `nvfp4_ds_mla` = 4-bit NVFP4 **KV cache** dtype (not draft); measured **no acceptance change** ("fp8_ds_mla vs nvfp4_ds_mla KV | no change to acceptance — pick for pool size"); KV pool ~2x token capacity (their 1M-context claim rests on it).
- Caveats: 4-bit KV "can collapse into salad under long, heavy agentic context; fp8 KV stays clean" (their long-context fallback path); their Stage C uses a padded 584-byte envelope workaround for a kernel-layout issue.
- Other DSpark knobs in their overlay: `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`, `VLLM_DSPARK_LOCAL_ARGMAX=1`, `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`, `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1`, `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`, `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0`.

### 7.5 Paper (arXiv 2607.05147) draft precision
- The paper does NOT quantize: draft = 3 MoE layers with mHC + SWA-128, gamma=5, Markov head, confidence head + STS calibration, sharing the target's (frozen) embedding and LM head; runs in the DeepSeek-V4 fp8/fp4 production stack. No weight-precision experiments are reported; precision is not part of the method.

### 7.6 Measured draft latencies (community, for comparison)
- GB10 / vLLM: draft-head GEMM bf16 2.73 ms -> fp8 1.45 ms per step (PR #47584). Full decode step at k=7 (verify+draft): joe 48.83 ms (techmd 55.46).
- M5 Max / Metal (Q4_K draft): draft backbone (3 layers, 5 tokens) **5.0 ms; total draft ~7.2 ms** (5.0 backbone + ~2.2 head/markov); another build ~10 ms/5 tokens; draft sidecar 3.60 ms per emitted token; batch verify L=5 ~75-80 ms per cycle (37 ms at block=2 vs 64 ms at block=5); plain decode 28-35.5 ms/tok; MTP-1 (2-token) = 2 ms draft + 33 ms fast verify.
- Paper: Markov loop FLOPs = 2*5*256*128000 = 0.33 GFLOPs/request/round; serial-loop overhead 0.2-1.3% of full-round latency at batch 128 when gamma 4->16.

### 7.7 Concrete spec for OUR 4-bit draft (target untouched)
1. **Draft MoE experts: keep MXFP4 as shipped** (already 4-bit; do not re-quantize — the checkpoint's fp4 experts + e8m0 scales load directly; re-quantizing risks the tonyd2wild shared-expert loader failure mode).
2. **Draft linears (attention projections, main_proj, hc_head_fn, and the always-read dense parts): quantize to 4-bit** — NVFP4 (MXFP4, ue8m0) or INT4-GPTQ/AWQ (group 128) are both defensible; prefer NVFP4 for a unified fp4/MXFP4 kernel path and fp32 accumulation. Expected: ~2x reduction in draft weight-read traffic vs fp8 (~4x vs bf16) on the always-read tensors, directly cutting the bandwidth-bound draft backbone on GB10 (~235 GB/s). Budget the acceptance gate: ds4's <0.4% (whole-draft Q4_K vs F16 at oracle) bounds the loss; treat it as a measured knob.
3. **Keep in fp8/bf16 (do NOT 4-bit): markov_w1 (vocab x 256), markov_w2 (256 x vocab), confidence_head, main_norm, and the RMSNorm/head params** — tiny (~70 MB total) but executed sequentially per draft position and they directly determine the markov bias; ds4's bf16-small-head failure mode (silent draft disable) and the sequential-bias sensitivity make low precision here the highest-risk choice. F32 accumulation for all 4-bit GEMMs is the single highest-leverage precision decision (ds4: 1.53 -> 2.2 acceptance).
4. **Draft KV and context projection stay in the draft's own fp8 path** (precompute_and_store_context_kv) — quantizing main_x-derived KV is optional and only affects acceptance.
5. **Never quantize the target** — q2 base dropped acceptance to ~2.2 (ds4) because the draft consumes target layer-40/41/42 hiddens; the exact-token gate protects the target but the draft's INPUT features come from the target, so target precision directly controls acceptance.
6. Expected draft-latency reduction: ~1.6-1.9x on the draft backbone/head (fp8->fp4 at constant bandwidth; c.f. bf16->fp8 gave 1.9x on the head GEMM). At joe's step budget (48.83 ms/step at k=7) a draft that is ~3-7 ms shrinks by ~1.5-3 ms — worth roughly +3-6% end-to-end single-stream, more if the draft were latency-critical.

## 8. SPECULATION DEPTH / ADAPTIVE DEPTH (follow-up)

### 8.1 Community k sweeps (all measured; per-position acceptance is NOT invariant to k)
- **joe (TP4, 2048-token code bench, temp 0, probabilistic; c=1):**
  | k | accept | tok/step | ms/step | tok/s c=1 | tok/s c=4/stream |
  |---|---|---|---|---|---|
  | 7 | 72.3% | 6.059 | 49.53 | 122.27 | 60.64 |
  | 8 | 64.1% | 6.130 | 53.15 | 115.16 | 58.80 |
  | 10 | 46.5% | 5.650 | 55.03 | 102.57 | 53.07 |
  "Asking the draft head to go deeper degrades acceptance at EVERY position, not just the new ones" (72.3 -> 64.1 -> 46.5). He built a model that reproduced k=7 to 3 decimals and predicted 6.745 tok/step at k=10; actual 5.650 (-19%). **You cannot extrapolate a deeper draft from a shallower measurement.**
- **joe (105K-context bracket, one boot each):** k=3: 40.6 tok/s, 69.9 ms/step, accept 61.2%; k=5: 49.3, 70.3, 51.9%; k=7: 68.6, 54.7, 39.4% (kept); k=8: 55.2, 59.7, 28.9%. Note: shallower k had SLOWER steps on his image (kernels specialized for k=7 shapes) — "k is a property of your ENGINE BUILD, not of the model."
- **Per-position curves:** k=5 (tonyd2wild patched, TP2): 0.826/0.725/0.572/0.471/0.399 (unpatched 0.631/0.282/0.181/0.114/0.067); k=6 (techmd, greedy): 96.2/90.0/83.5/76.3/69.5/62.3; k=7 (joe): 95.0/92.2/91.2/91.0/90.3/86.0/79.4.
- **tonyd2wild:** k=5 -> k=7 bought +3.3% tok/step for +33% draft cost; k=3 costs ~24% decode vs k=5.
- **Paper (Qwen3-4B family, gamma in {4,8,12,16}):** DSpark beats DFlash at every length; gap widens with gamma (gamma=7: +16% math/+15% code/+18% chat; gamma=15: +30%/+26%/+22%); serial-loop latency overhead only 0.2-1.3% of round latency at batch 128. (Their block-7 drafts are trained for gamma=7 — the curve holds because the head was trained at that depth; the DSV4 checkpoint is trained at block 5, so deeper k extrapolates.)
- **ds4 (Metal, replay-constrained architecture):** block=5 (16% full-commit, verify 64 ms) was WORSE than block=2 (85% full-commit, verify 37 ms) — their depth optimum was architecture-driven (per-token replay cost), not acceptance-driven. Not directly transferable to an engine with flat verify cost.

### 8.2 The paper's load-aware (confidence-scheduled) scheduler — exact mechanism
- Confidence head: c_k = sigma(w^T [h_k; W1[x_{k-1}]]) — conditional survival probability at position k given the prefix was accepted; trained against the analytical label c*_k = 1 - 1/2*||p_d - p_t||_1 (TV distance); calibrated with Sequential Temperature Scaling (STS: per-position 1D grid search minimizing ECE of the CUMULATIVE product; ECE 3-8% -> ~1%).
- Objective: maximize Theta = tau * SPS(B), where per request a_{r,j} = prod_{i<=j} c_{r,i} (prefix survival), tau = sum_r (1 + sum_j a_{r,j}) (expected accepts incl. bonus), B = sum_r (1 + l_r) (verification batch in tokens), SPS(B) = profiled engine steps/sec vs batch size (one-time cost table).
- Algorithm: greedy global admission — sort all candidate draft slots (r,j) by a_{r,j} descending; incrementally admit, recompute Theta via the SPS table; early-stop when Theta drops (the early-stop is what preserves losslessness: the admission decision must not depend on future candidate tokens — Appendix A gives a selection-bias counterexample).
- Production adaptation (Section 5.2): (1) asynchronous — the batch-capacity limit K is decided from confidence outputs TWO STEPS PRIOR (stale) so it doesn't stall the CUDA-graph/ZOS pipeline; the current step's slots are still ranked by up-to-date confidence; the 2-step lag is the causal barrier that preserves exactness while removing the early-stop; (2) variable-length verified prefixes executed by flattening all tokens and conveying intra-sequence structure via a marker tensor in the sparse-attention kernels (only index-attention and compress kernels modified on DeepSeek-V4).
- Measured gains (production, vs MTP-1 baseline): V4-Flash — +51% aggregate at 80 tok/s/user SLA; nominal +661% at 120 SLA (baseline near collapse; "frontier extension"); +60-85% per-user speed at matched throughput. V4-Pro — +52% at 35 SLA; +406% at 50; +57-78% matched. Under load the scheduler shrinks per-request verification length (Fig 8 c/d). Offline threshold sweep (Qwen3-4B): static threshold raises per-step acceptance from 45.7% (chat) / 76.9% (math) / 67.6% (code) toward 95.7%/92.5%/92.0% by pruning low-survival suffixes.
- **vLLM's equivalent:** AdaptiveVerificationManager (v1/worker/gpu/spec_decode/adaptive_verification.py) — per-step draft budget = argmax((sampled_requests + cumsum(survival_prob)) / (draft_cost + verify_cost)) with survival = cumprod of confidence probs from the previous step (CPU-stale copy, mirroring the 2-step lag), draft/verify cost curves profiled from real step timings (median_curve over cudagraph samples), budget enforced via a top-k admission kernel on GPU. Wired behind speculator.enable_adaptive_verification (off for the PR DSpark — the confidence head is dropped), and the tonyd2wild overlay ships VLLM_DSPARK_CONFIDENCE_SCHEDULER=off by default.
- **joe's simpler ladder:** num_speculative_tokens_per_batch_size [[1,4,7],[5,8,5],[9,12,3]] (k=7 at C<=4, k=5 at C 5-8, k=3 at C 9-12) — measured: costs ~7% single-stream for ~7% aggregate win at C=12; removed for his 1.3-2.4 avg-concurrency workload.
- **ds4/lobanov:** STS-derived confidence scheduling + bonus-token reuse gained ~5% on a broad prompt corpus (verifier-dominated setup).

### 8.3 Greedy vs probabilistic depth interactions
- At temp 0: greedy = probabilistic (joe: acceptance +0.2 pt, within noise — "the draft distribution at temp 0 is so peaked that sampling lands on the argmax essentially every time"). So at temp 0, pick whichever is cheaper (greedy local-argmax avoids the full-vocab gather entirely: VLLM_DSPARK_LOCAL_ARGMAX).
- At temp > 0: probabilistic is REQUIRED (greedy collapses: 26.5% vs 34.3% acceptance / 2.86 vs 3.40 tok/step, production shape; the greedy-bug range 27-48%). Deeper k interacts only through the per-position curve: at temp>0 the aggregate acceptance is much lower (joe production k=7: 39.4% vs 72.3% on the temp-0 bench), so the tok/step peak shifts lower — depth choices measured at temp 0 do not transfer to temp 1.
- The paper's scheduler assumes probabilistic verification (c*_k labels come from the TV distance of the actual draft distribution).

### 8.4 Recommended depth policy for OUR engine (verify = 24.8 ms for 1 row + 2.2 ms/extra row; draft latency TBD)
- **Fixed k=7 as the default**, based on: (1) joe's identical-hardware optimum (k=7 beat k=5, k=8, k=10; k=8's tok/step is only +1.2% higher than k=7 while step time rises, and the per-position curve degrades at k>=8 for this block-5-trained checkpoint); (2) the checkpoint's trained reach (block 5 + markov; positions beyond ~6 extrapolate and the community curve holds only to k=7).
- **Decision math:** tok/s(k) = [1 + sum_i a_i(k)] / [draft(k) + 24.8 + 2.2k]. Plugging joe's k-dependent sums (5.059/5.130/4.650 accepted at k=7/8/10) and draft ~3-4 ms at k=7 growing ~0.3 ms/extra position: k=7 ~140 tok/s > k=8 ~134 > k=10 ~111. The margin is robust: even with a perfectly flat free draft, k=8 stays below k=7 because tau(k=8) barely rises; only a k-invariant per-position curve (which joe disproved) would make deeper k pay with your flat verify.
- **Measure in-engine before finalizing:** k is engine-build-dependent (joe: "k is a property of your ENGINE BUILD"). Run k=6/7/8 for one boot each (joe's protocol: same prompt, proper warmup, server-side decode-only tok/s). If in-engine per-position acceptance at k=8 keeps the k=7 prefix (you measured it, don't assume), k=8 wins with the flat verify curve (6.130 vs 6.059 tau); joe's evidence says it won't.
- **Adaptive depth: only for high concurrency.** At low/medium concurrency (joe's 1.3-2.4 avg; single-stream is your default), fixed k beats adaptive (joe's ladder lost 7% single-stream; the paper's scheduler's +51%/+60-85% gains are aggregate-throughput numbers under multi-user SLA load vs an MTP-1 baseline). If the engine serves C >= 8-12 sustained: implement the confidence-scheduled verifier — requires loading + STS-calibrating the confidence head (vLLM drops it; you must wire it), profiling SPS(B), and the 2-step-stale top-K admission; expected +50% aggregate at moderate SLA (paper) or, minimally, the batch-size ladder (joe: ~7% aggregate win at C=12).
- **Greedy/probabilistic:** run probabilistic unless the serving temperature is pinned to 0 (then greedy + local-argmax is free and identical). The 34.3% vs 26.5% design-doc numbers are the temp-1 production-shape gap — they are why probabilistic is mandatory above temp 0.

## 9. Engine implementation status (dsv4-dspark-speculative branch, 2026-08-16)

### Verified semantics (against reference model.py @ 7872f01b)

- KV-slot convention: the KV slot at position p holds the KV of the token at
  position p-1 (the row at p projects its INPUT token's KV into slot p). The
  engine's decode submission position = the new token's position; the row
  position = the KV slot. The draft's block row 0 (the anchor at
  anchor_position+1) is therefore the anchor's KV slot, matching the
  reference's query_pos convention.
- Verify frame: rows at anchor_position+1 .. anchor_position+8 (8 rows = the
  anchor + k=7 drafts at their KV slots). Row i's logits predict position
  (anchor_position+1)+1+i = the token AFTER the row's input. Acceptance:
  accept draft_i iff output[i] == draft_i for i in 0..6 (output[0] = the
  base token vs draft_0); all-7-accept adds the bonus output[7].
- Emit: output[0..a] (1+a tokens, positions P+1..P+1+a); lane_next advances
  by a; the next anchor = output[a] at P+1+a; the next draft block starts at
  P+2+a (the anchor's KV slot).
- The verify frame's intra-frame KV visibility is free: CacheScatter writes
  all 8 rows' KV before SparseAttn reads, and the position-causal mask gives
  the reference's block-causal semantics.
- The CSA/HCA running state is position-keyed (state[position % ratio]) so 8
  rows (< ratio 21) never race.

### Engine changes landed (commit acc6783a, PR #662)

- B8 admission + frame-shape gates accept {1,1} when DSpark is enabled; the
  expansion upgrades to the bucket-width island shape internally.
- Verify expansion fixes: physical-page resolution per row, the exact
  3x u32 + 3x u64 staging layout, page initialization, and the draft block's
  input tokens/positions staged BEFORE the embedding gather (two pre-existing
  staging bugs fixed).
- Single-lane prefills + draft-failed decodes pad 8 duplicate rows at the
  same position (identical rows = identical KV + identical outputs; only
  row 0 emits).
- Completion: greedy Leviathan acceptance, burst emission (1+accepted), lane
  + cache advance by accepted, anchor tap publish from the accepted row
  (per-row verify tap buffer).
- Engine fixes: TP-fanout completions may carry status only from non-final
  ranks (runtime/model_serving_adapter.c); the residentd decode lease
  advances by the COMPLETION token count (partial-accept bursts).

### Known gap before the exact gate can pass

- CSA/HCA boundary EMISSION rollback: when a rejected row lands on the
  compressed-emission boundary ((position+1) % ratio == 0), its compressed
  window (which includes rejected tokens' contributions) is written into the
  paged compressed cache. Plan: save the emission target region before the
  verify, restore if the boundary row was rejected; the accepted-prefix
  state is position-keyed so a restore + no re-emit is exact. Probability
  ~8/21 per frame.
- The stale speculative KV beyond the accepted prefix is overwritten before
  it is ever read (the next frame's rows re-scatter the same slots), so no
  page-content rollback is needed; the page-list commit truncates at the
  accepted context via CompleteLane.

## 10. Session results 2026-08-17 (fleet: spark8-f down; dev system on spark4)

- Cluster restored: the init route_not_found was a STALE hidden_transport.so on
  the ranks (the deploy scp silently failed on the busy file). A clean
  rm+scp+restart brought all 4 ranks ready. Fleet notes: the second rail
  re-negotiated 200G->100G; new roceP2p1s0f0/f1 NICs appear/disappear on the
  nodes; spark8-f (GLM band + old build host) are down for days.
- End-to-end DSpark pipeline RUNS on B8: prefills stream, the verify
  expansion stages 8 rows, the acceptance loop runs on every rank
  (accepted=0/1 observed - the first cross-rank acceptances), bursts emit
  1+accepted tokens, the residentd lease + the TP-fanout completion
  validation accept partial-accept yields.
- Fixes landed this session: 7-row native decode shape whitelisted (the draft
  head's Linear rejected the serving block); the release completion carries no
  tokens (the adapter gate); EVERY rank D2Hs the REDUCED head tokens for the
  verify acceptance (the non-head ranks previously accepted on stale host
  buffers - the lane store diverged across the TP group); the speculative
  completion lower bound relaxed to 1.
- Measured: the O24 batch completes (status 0) at 8.35 tok/s with the
  fallback-heavy path; the gate hash still mismatches - see the remaining gap.
- REMAINING GAP (exactness): the CSA/HCA compressed-attention state. The CSA
  (ratio 4, overlapped, 21 layers, per-layer position-keyed ring state) gets
  speculative rows' contributions and a boundary-row ring SHIFT during the
  verify frame. The fix = per-frame save of the compressed states + the
  emission slots, then restore + re-apply the ACCEPTED rows (the replay needs
  the rows' per-layer compressor projections, which the recorded islands
  cannot expose - either a verify-specific island variant with the compressor
  writes to a scratch state, or the host-side replay with the projections
  saved by a modified kernel). The scaffolding (the slot save buffers) is
  committed.
- Next: the compressor rollback, then the draft latency (the per-markov-step
  host syncs + the tap syncs dominate the ~120 ms steps).

## 11. Rollback probes 2026-08-17 (continued)

- Draft-disabled probe (the pure pad path, 0 drive entries): the emitted
  stream matches the 8-lane baseline EXACTLY for the first 11 tokens
  (48582,223,2892,201,223,20,28,539,223,21,28), diverging at the 12th - the
  first CSA boundary frame. So the pad path is exact and the remaining
  divergence is confined to the compressor rollback.
- The draft-ENABLED run degenerates to token 20 from the 2nd decode: the
  draft forward's execution corrupts the verify frame's shared slot state.
  The draft uses the shared slot buffers (reduced/normalized/mixes/kv/etc),
  which the islands overwrite - the persistent corruptor is yet to be
  isolated (the bisect candidate: the draft's CacheScatter ring writes and
  the markov embedding gather on slot->input_token_ids).
- The CSA previous-window restore is INCOMPLETE for the mixed-boundary case
  (the first boundary accepted + the second rejected): the restore must
  replay the ACCEPTED rows' state updates (the position writes + the
  accepted boundary's shift + its emission) on top of the pre-frame state.
  The replay needs the rows' per-layer compressor projections, which the
  recorded islands overwrite - the fix is a CompressStep kernel extension
  that also copies its inputs to a per-slot projections save region (the
  signature change re-records the islands, which the existing capture
  already supports), then the host-side per-layer replay: memcpy the
  accepted rows' projections into the scratch + CompressStep(rows=accepted+1)
  + KvEmission against the restored state and the real cache.
- The baseline check: the 8-lane batch on the same build completes (status
  0) with the per-request stream matching the DSpark pad path; the pinned
  hash oracle remains unavailable on the current fleet (the old control
  binaries no longer initialize).
