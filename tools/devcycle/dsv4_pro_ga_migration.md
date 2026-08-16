# DSV4 Pro GA migration and required kernels

## Version finding: we packed the PREVIEW, not the GA

Evidence (all verified live against huggingface.co on 2026-08-16):

| Fact | Preview (what we packed) | GA |
| --- | --- | --- |
| Repo | deepseek-ai/DeepSeek-V4-Pro | deepseek-ai/DeepSeek-V4-Pro-0813 |
| lastModified | 2026-06-22 | 2026-08-13T16:28Z |
| Local download date | 2026-08-08 | downloading now (background) |
| Shards | 64 | 66 |
| Total size | 864,704,792,696 B | 892,727,580,904 B (+28 GB) |
| Tensors | 145,116 | 149,782 (+4,666) |
| compress_ratios | 62 entries, tail [4,0] | 64 entries, tail [4,0,0,0] (61 real layers identical) |
| MTP | mtp.0.{e,h}_proj + norms + hc_head (V3-style) | mtp.0.main_norm + main_proj; mtp.1 = full DSpark draft layer + markov/confidence heads |
| README | "preview version" wording | GA repo |

The README in the packed repo itself says "preview version", the PDF is dated
2026-05-06, and HF shows the repo last modified 2026-06-22. The GA shipped
2026-08-13 as a SEPARATE repo. Everything we staged (full pack, 16 rank
packs, single-spark validations) is built from PREVIEW weights and must be
rebuilt from the GA before any ring run. The GA config is Pro-correct
(7168/61/384/top6/128h/1536/16/1024/3hash/1mtp/129280/2.5 - no Flash mix-up
in the current listing).

## GA MTP = DSpark speculative decode stage

dspark_block_size 5, dspark_target_layer_ids [58,59,60],
dspark_markov_rank 512, dspark_noise_token_id 128799.

- main_proj: FP8 linear 3x7168 -> 7168 over the concatenated hidden states
  of layers 58-60; main_norm: RMSNorm(7168). (mtp.0.*)
- mtp.1 = a full mHC block (DSparkBlock): attention with compress_ratio 0
  (draft queries attend over the MAIN-stream KV plus the draft window via
  the same sparse_attn primitive), its own MoE/shared experts, norms, HC
  params - all stored under the mtp.1 namespace.
- Draft head: hc_head -> norm -> the SHARED main head, plus a Markov
  n-gram bias (markov_w1 vocab x 512 embedding, markov_w2 512 x vocab) and
  a confidence head (concat[hidden, markov_embed] -> 1).
- Each spec step: 1 main token + 5 draft tokens = 6 tokens per main-model
  forward - the GA's built-in answer to the decode-rate question (this is
  P2 from the performance plan, in the checkpoint itself).

## Packer changes required

1. PRO_RATIOS: append the GA tail ([4,0] -> [4,0,0,0]); the 61 real-layer
   kinds are unchanged.
2. New record kinds for the MTP set: MAIN_NORM, MAIN_PROJ (FP8+scale),
   MTP_NORM (mtp.1.norm), MARKOV_W1 (vocab x 512 embedding),
   MARKOV_W2 (512 x vocab), CONFIDENCE_PROJ (BF16 7169->1... dim = hidden +
   markov_rank = 7680 -> 1), plus the mtp.1 full layer via the existing
   layer kinds at the MTP layer index, and the mtp.1 hc_head_* globals.
   Remove the preview's MTP_E_PROJ/H_PROJ/ENORM/HNORM/FINAL_NORM records.
3. The stage-pack format header needs the new kind ids (they must not
   collide with existing ids) and the module's coverage expectations must
   match.

## Module changes required

- SparkDsv4MtpWeights -> main_norm, main_proj, mtp layer weights (a full
  SparkDsv4LayerWeights), norm, markov_w1/markov_w2, confidence_proj,
  hc_head_*; ratio table tail update; coverage bits for the new kinds.
- MTP execution: after the final main layer, gather layers 58-60 hiddens
  (the module already streams per-layer hiddens; capture at 58-60) ->
  main_proj+norm -> draft loop (block_size=5): embed draft ids -> hc-expand
  -> DSparkBlock forward (attention over main KV + draft KV) -> hc_head ->
  norm -> head logits + markov bias -> sample -> confidence -> accept.

## Required CUDA kernels

Reuse (no new math): embedding gather (markov_w1), dense FP8/BF16 linears
(main_proj, markov_w2, confidence), RMS norms, the existing sparse_attn
primitive (DSpark attention = same primitive over main-stream KV), the
full layer kernel set (mtp.1), screened-argmax head.

New kernel work:
1. **FP8-E4M3 expert kernel variant** - SparkLmSm121FusedExpertW13/W2 with
   FP8 weight loads (LmFp8 format, F32 per-128 scales) instead of the
   MXFP4 loaders; the grouped-tile schedule is codec-agnostic. Needed for
   the fp8-expert packs from tools/dsv4_pro_expert_requant.py.
2. **DSpark attention lane** - a module-side cache mode feeding the MAIN
   stream KV to the draft block's sparse attention (the python reuses
   sparse_attn; the module needs the main-KV cache lane plumbing, not a new
   tensor-core kernel).
3. **Draft head assembly** - markov bias add + confidence projection +
   draft sampling loop (small kernels over 512/7680 dims; the heavy head
   reuses the screened argmax).

## Sequencing

1. GA download (running, background, ~8 h).
2. Packer GA updates + full pack rebuild + verify + shard + re-stage.
3. Module GA updates (structs, coverage, ratio tail).
4. DSpark MTP execution path (kernels 2-3) + single-spark validation.
5. FP8 expert kernel variant (kernel 1) + fp8 pack conversion from GA.
6. Re-run val4/valtail on GA packs; re-pin all identities; preflight.
7. Ring run.

Until this migration lands, the staged deployment is PREVIEW-baseline and
must not be used for the final measured decode.

## Status log

- R11: packer + module load the GA MTP (3 draft layers, main-proj,
  markov/confidence heads, ratios 64); record dry-run vs the GA index:
  211 records, 0 missing; module b1 builds clean.
- R11: DSpark CUDA kernels land and compile (in
  spark_dsv4_resident_decode_stage_cuda.cu, guarded by
  SPARK_DSV4_MODEL_MTP_LAYER_COUNT > 0):
  SparkDsv4DSparkLaunchMeanReduction (hc-stream mean capture for the
  target layers), SparkDsv4DSparkLaunchMainKvWrite (rolling main-KV
  window write), SparkDsv4DSparkLaunchAttention (draft attention over the
  main-KV window + causal draft KV, online softmax + sink). First-cut
  approximations (documented in the kernels): BF16 draft KV, no rotary on
  the draft path, no fp8 activation quantization - safe because drafts are
  verified against the main model before acceptance.

## DSpark module wiring spec (next implementation step)

State (per slot, allocated from the ledger): dspark_capture_bf16[3] (layers
58-60 post-layer hc means), dspark_main_bf16 (main_proj + main_norm
output), dspark_main_kv_bf16[3] (128 x 512 BF16 rolling windows),
dspark_draft_q/kv/o buffers (5 rows), dspark_draft_streams (5 x 4 x 7168),
dspark_valid_counts (129..133), dspark_draft_ids + dspark_draft_tokens.

Chain (final stage, decode frames, after the main head emission):
1. main_proj: dense FP8 linear (21504 -> 7168) over the concatenated
   captures -> RMS main_norm -> dspark_main_bf16. (Existing linear + norm
   launches.)
2. Draft ids: [accepted token, noise x 4]; embed via the existing
   embedding gather -> 5 x 7168; replicate to 4 hc streams.
3. For each draft layer i (mtp.0/1/2):
   a. main_kv = kv_norm(wkv_i(main_bf16)) -> SparkDsv4DSparkLaunchMainKvWrite
      into the rolling slot (sequence position % 128). (wkv = existing
      dense FP8 linear, kv_norm = existing RMS.)
   b. Draft attention side: HcEnter(mtp hc attn params) -> attn_norm ->
      wq_a/wq_b (draft q) -> wkv + kv_norm (draft kv) ->
      SparkDsv4DSparkLaunchAttention(window, draft_kv, valid_counts,
      mtp sink, 1/sqrt(512)) -> wo_a/wo_b -> attn-side hidden all-reduce
      (5 rows) -> HcPost.
   c. FFN side: HcEnter(mtp hc ffn) -> ffn_norm -> gate route (bias gate)
      -> routed experts (top-6, mtp experts) -> shared -> ffn-side
      all-reduce -> HcPost.
   d. Continue with the existing TP continuation machinery (a draft
      continuation mirroring SparkDsv4ModuleContinueLayers with
      state->mtp_layers[i]).
4. Draft head: HcEnter/mix + HcHeadReduce with mtp.2's hc_head params ->
   RMS mtp.2.norm -> the SHARED head (screened argmax, 5 rows,
   vocab-sharded) -> U64 maxloc pack -> 5 draft tokens + (later) the
   Markov bias and confidence head (both reuse existing kernels; markov
   w2 = head-style GEMV over 512, confidence = 1x7680 GEMV).

The acceptance/verification policy lives client-side (compare draft logits
with the main model per position); the module only emits drafts + (later)
confidence.

