# DSV4 Pro — DSpark native pass: implementation plan + first code (backlog item 10)

The real decode lever: main-only ~12-13 tok/s -> DSpark gamma=5 ~45-60 tok/s
(accurate-slow -> match/exceed SOTA). Scouting confirmed the Pro draft kernels do
not exist (`grep SparkDsv4DSparkLaunch` is empty) and execution is refused
(`spark_dsv4_resident_decode_stage_module.c:47-48`). The MTP weights already load
(`..._module.c:1066-1074`). This doc is the plan + the kernel contract cards +
the first-code skeleton the cuda-kernels harness and the Pro lane wire.

## 0. What already exists (reuse, don't reinvent)
- **Flash DSpark drafter** — 5 kernels in `spark_dsv4_dspark_kernels.cuh`
  (`docs/KERNEL_CONTRACT_CARDS.md:250-346`): Attention(online softmax + sink),
  MarkovBiasAccum, Argmax, TapMean, ExpandStreams. Block 5, spec step 7,
  markov 256, hidden 4096 — **Flash geometry, not Pro**.
- **Pro MTP weights loaded** into `state->mtp` + `state->mtp_layers[3]`
  (`..._module.c:1066-1074`, struct at :361-362), plus markov_w1/w2 (vocab x 512)
  and confidence_proj (1 x 7680).
- **Draft state buffers** already declared (`..._module.c:366-377`:
  `dspark_ring_bf16`, `dspark_tap_store_bf16`, `dspark_lane_anchor/position`).
- **Shared head + screened argmax + embedding gather** exist for the main model
  and are reused by the draft head (ga_migration "reuse (no new math)").

## 1. The draft shapes (pin these first — they are 0 today)
`spark_dspark_drafter.h:82-85` declares `DRAFT_INTERMEDIATE_DIMENSION=0`,
`DRAFT_ATTENTION_HEAD_COUNT=0`, `DRAFT_KV_HEAD_COUNT=0`,
`DRAFT_HEAD_DIMENSION=0`. From the GA checkpoint the Pro session must pin:
- hidden 7168, 3 draft layers, block 5, markov 512, noise 128799 (contract
  `dsv4_pro_authoritative.json:32-42`); draft attention heads/intermediate are
  the missing values — read them from the GA `mtp.1` mHC block config, then
  write them into `spark_dspark_drafter.h` and the contract.

## 2. Kernel contract cards (Pro draft path, request to cuda-kernels)

| Card | Op | Shape (Pro) | Precision | Combine |
|---|---|---|---|---|
| **P-D-01** mean-reduce | capture layers 58-60 post-layer hc means | rows=5, dim=7168, 3 taps | bf16->fp32 sum->bf16 | none (per tap) |
| **P-D-02** main_kv write | kv_norm(wkv(main)) -> rolling window | window 128 x 512 bf16, seq%128 | bf16 | none |
| **P-D-03** draft attention | qk over main-KV window (128) + causal draft KV (5), online softmax + sink, scale 1/sqrt(512) | q heads=?/dim=?, 5 rows | bf16 | attn-side all-reduce (5 rows) |
| **P-D-04** markov bias | logits += markov_w2 . markov_embed | vocab-shard x 512 | bf16/fp32 | head U64 max |
| **P-D-05** confidence | sigmoid(dot([hidden|markov_embed], w) + b) | 1 x 7680 | bf16/fp32 | none |
| **P-D-06** draft mHC layer fwd | 3 x (attn + FFN: top-6 mtp experts) | mirrors main layer kinds | mixed | FFN-side all-reduce |

These mirror the existing Flash cards but with hidden 7168 / markov 512 / 3 draft
layers / block 5 and the Pro draft attention heads (to pin). The draft KV is BF16
(no rotary on the draft path, first-light approximation per
`dsv4_pro_ga_migration.md:122-124`).

## 3. Module wiring (the chain, from ga_migration.md:126-163)
Sequence after the final main-layer head emission (decode frames):
1. `main_proj` (FP8 21504->7168) over concat of taps 58-60 -> `main_norm` ->
   `dspark_main_bf16`.
2. Draft ids = [accepted, noise x 4]; embed -> 5 x 7168; replicate to 4 hc streams.
3. Per draft layer i in {0,1,2}: main_kv write (P-D-02) -> draft q/kv -> draft
   attention over window+draft KV (P-D-03) -> attn-side reduce -> FFN (bias gate
   top-6 mtp experts, P-D-06) -> ffn-side reduce -> continuation mirroring
   `SparkDsv4ModuleContinueLayers` on `state->mtp_layers[i]`.
4. Draft head: HcEnter/HcHeadReduce (mtp.2 params) -> norm -> SHARED head screened
   argmax (5 rows, vocab-sharded) -> U64 maxloc -> 5 draft tokens + markov bias
   (P-D-04) + confidence (P-D-05). Acceptance policy is client-side.

## 4. First code (skeleton — the contract + launcher surface)

### 4.1 Pin the drafter table (edit, ~6 lines)
`spark_dspark_drafter.h:82-85`: replace the four `0u` with the pinned Pro
draft values once read from the GA mtp.1 config (draft intermediate, draft
attention heads, draft kv heads, draft head dim).

### 4.2 New launcher surface (the kernel contracts, to land in the module .cu)
```c
/* Pro DSpark draft path - guarded by SPARK_DSV4_MODEL_MTP_LAYER_COUNT > 0. */
cudaError_t SparkDsv4DSparkLaunchMeanReduction(
    cudaStream_t stream, const void *tap_bf16 /* [3][rows][7168] */,
    void *mean_bf16 /* [3][7168] */, uint32_t rows, uint32_t streams);
cudaError_t SparkDsv4DSparkLaunchMainKvWrite(
    cudaStream_t stream, const void *main_bf16 /* [7168] */,
    void *window_bf16 /* [128][512] */, uint32_t seq_pos);
cudaError_t SparkDsv4DSparkLaunchAttention(
    cudaStream_t stream, const void *q, const void *window_kv,
    const void *draft_kv, const void *sink, void *attn_out,
    uint32_t rows, uint32_t valid_count, float scale);
cudaError_t SparkDsv4DSparkLaunchMarkovBias(
    cudaStream_t stream, float *logits_f32, const void *markov_w2,
    const void *markov_embed, uint32_t vocab_shard, uint32_t rank /*512*/);
cudaError_t SparkDsv4DSparkLaunchConfidence(
    cudaStream_t stream, const void *hidden_markov /* [7680] */,
    const void *conf_proj /* [7680] */, float *conf_out);
```

### 4.3 Module chain stub (the per-slot draft forward)
```c
static SparkStatus SparkDsv4ModuleRunDsparkDraft(
    SparkDsv4ModuleState *state, uint32_t slot, uint32_t draft_count /*5*/)
{
    /* 1. gather taps 58-60 hc means -> main_proj+norm -> dspark_main_bf16
     * 2. draft ids [anchor, noise x4] -> embed -> 5x7168 -> 4 hc streams
     * 3. for i in 0..2: main_kv write; draft attn (P-D-03); HcPost;
     *    bias-gate top-6 mtp experts (P-D-06); HcPost
     * 4. mtp.2 hc_head -> norm -> shared screened argmax -> markov bias
     *    (P-D-04) -> confidence (P-D-05) -> publish drafts.
     * Acceptance/verification stays client-side (compare vs main model). */
    (void)state; (void)slot; (void)draft_count;
    return SPARK_STATUS_UNSUPPORTED; /* until kernels land */
}
```

## 5. Sequencing + verification
1. Pin draft shapes in the contract + `spark_dspark_drafter.h` (section 4.1).
2. cuda-kernels agent lands P-D-01..06 behind
   `SPARK_DSV4_MODEL_MTP_LAYER_COUNT > 0` (mirroring the Flash draft kernels).
3. Pro lane wires `SparkDsv4ModuleRunDsparkDraft` into the decode frame
   (after the main head emission, `..._module.c` final stage).
4. Validate: single-spark valtail slice (has the full MTP record set,
   `tools/devcycle/dsv4_pro_single_spark_receipts.md:35-39`) emits 5 drafts +
   confidence; then the token-hash gate vs the main model on the ring.
