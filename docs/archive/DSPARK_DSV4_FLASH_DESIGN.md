# DSpark backend design — DSV4 Flash, native driver

Status: design. Supersedes the disabled dspark entry in the checkpoint contract
(`execution_supported: false`) once implemented and measured.

## 1. Why DSpark is the B1 answer

The no-spec hill-climb is numerically pinned: the exact-token gate rejects any
reassociation of the target math (C3/C4), and the ~5.5 ms/token idle is the
measured network floor of 130 serial collectives (C1/C6). DSpark is the one
lever that survives both constraints:

- **Output-preserving by construction.** Every draft token is verified against
  the exact target with rejection sampling, so the emitted stream is
  *identical* to the no-spec target. The O24/O128 token-hash gate applies
  unchanged; acceptance affects speed, never output.
- **It amortizes the whole decode cost over k+1 rows.** One verification
  forward streams the fixed weights once for k+1 positions and runs the 130
  collectives once for k+1 rows. For B1 this converts serial-latency-bound
  decode into batched verification. Community result on our exact hardware
  class (4× DGX Spark, TP4): **123.13 tok/s at k=7** vs 104.17 best published
  no-spec build (joesinvestments/DeepSeek-V4-Flash-0731-TP4-4x-DGX-Spark @
  `1339c163`, `anemll/dspark-vllm-gx10:0.1.1`).
- **The draft is the checkpoint's own.** No external predictor: the
  `dspark` namespace ships 3 draft layers with the model, so the integration
  is a packer + driver problem, not a training problem.

## 2. The draft contract (from dsv4_flash_authoritative.json)

```json
"dspark": {
  "checkpoint_namespace": "mtp",
  "layer_count": 3,
  "block_size": 5,
  "noise_token_id": 128799,
  "target_layer_ids": [40, 41, 42],
  "markov_rank": 256,
  "tensor_counts": [1568, 1565, 1572],
  "packed": false,
  "execution_supported": false
}
```

- Three draft layers attach to main layers **40, 41, 42**. Each draft layer
  consumes the hidden state of its target layer, the current input-token
  embedding, and the recurrent **Markov state** (rank 256), and predicts a
  **block of 5 next tokens** through 5 output heads (semi-autoregressive).
- The **noise token (128799)** fills the unused leading positions of the
  first block after the prompt; the Markov state carries the draft's own
  context across blocks, which is why deeper drafting degrades per-position
  acceptance (community: k=7 acceptance 72.3%, k=10 46.5% — the curve is NOT
  invariant to k).
- tensor_counts are per draft layer; the exact tensor catalog must be derived
  from the checkpoint index at pack time (the packer already carries a
  single-MTP slot: `SPARK_DSV4_STAGEPACK_MTP_LAYER`, `mtp_layer_count`).

## 3. Engine shape decisions (start points, all re-measured on our engine)

| Parameter | Start | Evidence / rule |
| --- | --- | --- |
| Draft depth k | **7** | community k=7 > k=8 (115.2) > k=10 (102.6); k is a property of the ENGINE build (kernel shapes), not the model — sweep on ours |
| Draft sampling | **probabilistic** | single biggest community win: 34.3% vs 26.5% acceptance (2.86→3.40 tok/step); greedy collapses to p_target(argmax) at temp>0 |
| Verify rows per step | k+1 | one batched target forward |
| Graph capture rows | **max_seqs × (k+1)** | B1 → 8 rows; B8 → 64 rows; under-sizing silently drops to eager decode (community: capture 36 truncates at 4 seqs) |
| Prefill budget | nominal − (k−1)×max_seqs | draft slots subtract from the prefill token budget |
| KV | BF16 (unchanged) | capacity trades are orthogonal; revisit NVFP4 MLA only as a capacity experiment, never for the correctness path |
| Target math | untouched, BF16 spine + FP8 linears + MXFP4 experts | the exact-token gate is the acceptance authority |
| Draft precision | checkpoint-declared; quantizing the draft is allowed | every proposal is verified by the exact target (handoff rule); community fp8 draft lm_head (vLLM #47584) is the pattern |

## 4. Native driver execution plan

### 4.1 Packing

Extend the stagepack to the full dspark namespace: 3 draft layers ×
[tensor_counts], plus the Markov-state weights (rank 256) and the noise-token
handling. The pack header already has `mtp_layer_count` and an MTP slot;
this becomes 3 MTP layers with per-layer tensor catalogs and a generated
draft pack manifest (tensor-name/offset/codec), per the archived ledger's
rule: a dspark pack must pass bit-exact load receipts for every fused
projection before acceptance is measured.

### 4.2 Draft forward (per draft step, per sequence)

```text
input: accepted prefix hidden states (from the verify forward), markov state
for each of 3 draft layers:
  emb = embedding(draft_input_token)          # tiny gather
  hidden_l = target_hidden[l] ⊕ markov_l ⊕ emb   # per attached layer 40/41/42
  logits_l = draft_head_l(hidden_l)           # 5 next-token heads
block = sample(probabilistic, logits)         # 5 tokens (noise-filled first block)
update markov_l from hidden_l                 # recurrent rank-256 state
```

Kernel budget: three small dense layers (~1.5-1.6K tensors each) + 5-way
heads + a 256-rank recurrent update. This is a small-M GEMM workload — the
archived ledger's Marlin lesson transfers: profile every tiny-M draft GEMM
epilogue and reduction separately; prefer fused atomic accumulation over a
repeated global reduction (the GLM quantized draft went 6.5 → 44.6 tok/s on
that one change).

### 4.3 Verify loop (per engine step)

```text
1. verify forward: run the MAIN model over (k+1) rows — the accepted prefix
   position plus the k draft positions, batched (weights streamed once)
2. per-position acceptance: compare target sampling vs draft token with
   rejection sampling; stop at first mismatch
3. commit accepted tokens; retain the surviving draft block for the next step
4. the draft forward for the next step rides the verify completion event
```

For B1: verify step ≈ one decode step over 8 rows. Fixed stream is read once;
the expert stream grows with coverage (6-of-256 experts over ~8 rows ≈ 2-3×
the B1 expert bytes — to be measured, not assumed). At community tok/step
6.06 (k=7, code traffic) and a ~50 ms verify step that is ~120 tok/s; at
production acceptance ~3.4 tok/step it is ~60-70 tok/s. Both are the
targets to beat 40.4.

### 4.4 Batch scaling (the B8 and TP4×PP4 path)

- **B8**: verify rows = 8 × (k+1) = 64. Capture rows must be sized for it
  (64), and the graph set prewarmed at that width; the scheduler never lets
  a wider shape fall back to eager silently.
- **TP4×PP4**: target layers 40-42 live in PP stage 3 (layers 33-42).
  Placement: the draft forward executes on the stage-3 TP4 ranks, which own
  exactly the hidden states it needs; the verify runs pipeline-wide as a
  normal micro-batch of (k+1) positions; the draft state round-trips with the
  stage-3 micro-batch boundary. Reference for DSpark-under-PP: the
  allover326/deepseek-v4-cmp170hx build (vLLM does not support this
  upstream — it is a SparkPipe differentiator).
- **Draft concurrency**: drafts for different sequences may overlap the
  verify of others; the scheduler's existing cohort machinery covers it, and
  "drafts ride decode-completion events" means zero extra fabric round-trips.

## 5. Measurement and acceptance protocol

1. **Gate 1 — output identity.** With speculation on, the O24/O128 streams
   must hash exactly to the no-spec control (rejection sampling preserves
   the target distribution). Any deviation fails the candidate.
2. **Gate 2 — tok/step and per-position acceptance**, not aggregate
   acceptance (79.6% over 6 positions loses to 71.7% over 7). Report
   tokens-per-step, per-position acceptance, and ms/step.
3. **Protocol**: 2 full warmups + 6 discarded + 10 measured (community saw an
   11 tok/s warmup effect); metric = delta generation tokens / delta decode
   seconds from server counters; snapshot counters before/after and reject
   cells with foreign traffic.
4. Sweeps (one variable per boot): k (5/7/8/10), draft sampling, draft
   precision, capture rows, prefill budget, and the Markov-state handling.

## 6. Phasing

| Phase | Deliverable | Gate |
| --- | --- | --- |
| P1 | dspark pack: 3 MTP layers + markov state + manifest, bit-exact load receipts | pack validation |
| P2 | draft-forward kernels (3 layers, 5 heads, markov update, noise token) | exact O24 with draft disabled; draft-layer numerical receipt vs reference |
| P3 | verify loop in the resident driver (k+1-row batched forward + acceptance) | exact O128 with speculation ON (output identity) |
| P4 | B1 measurement: k sweep, tok/step, vs the 40.4 control | speed gate |
| P5 | B8 capture-rows scaling + scheduler cohort integration | batch gate |
| P6 | TP4×PP4 stage-3 draft placement | pipeline gate |

## 7. Risks

- The per-position acceptance curve is not invariant to k; our k must be
  measured on our engine and our traffic (community: 6.06 tok/step at k=7 on
  code traffic, ~3.2 on deep agent traffic).
- Draft numerics do not affect output (verified), but draft SPEED does: the
  small-M draft GEMM epilogue is the first profile target (Marlin lesson).
- The Markov-state exactness matters for acceptance quality; pin a reference
  implementation receipt in P2.
- Stage-3-only draft placement under PP must not starve the other stages'
  micro-batch slots (measure, don't assume).
