# GLM52 DSpark Backend, WMMA Prefill, and the B256 Chunking Wall - 2026-07-04

## Measured sweep recap (spark2, 7dcce21, includes the weight-stationary kernel)

```text
B1    slowest 0:6    39.6 ms   25.3 tok/s   (see measurement caveat below)
B4    ~42.5 ms/stage             94-103 tok/s
B64   slowest 72:6   66.14 ms   967.7 tok/s   (was ~100 ms / 640 tok/s)
B128  slowest 60:6   112.74 ms  1135.4 tok/s  (+17%)
B256  slowest 30:6   215.30 ms  1189.0 tok/s  (+4.7%)
```

B64 at 967.7 tok/s validates the weight-stationary projection kernel
prediction from GLM52_B128_SCALING_ROOT_CAUSE_AND_MTP_20260704.md. The new
problem is B128 -> B256: 1.91x the time for 2x the tokens - batch scaling is
dead again above 128.

**B1 caveat:** the B1/B4 sweeps ran WARMUP_RUNS=0 MEASURE_RUNS=1, so the
measured pass includes CUDA graph capture. The historical 20.7 ms B1 stage
number was a warmed best-of. 39.6 ms is not evidence of a B1 regression;
re-measure with warmup >= 1 before concluding anything.

## The B256 wall: chunked MoE launches cap weight amortization at 128

`SparkGlm52B12xLaunchChunked`
(modules/glm52_sm121_b12x_compiled_backend/source/spark_flashinfer_b12x_compiled_moe_backend.cu:1594,
added in 7d1a85e) splits token_count into largest-available-bucket chunks and
launches the MoE primitive once per chunk. At >= 128 tokens every chunk
activates ~all 256 experts, so **every chunk re-streams the full ~5.4 GB/layer
expert sweep**. Tokens-per-weight-sweep is capped at the chunk size, which
caps batch scaling by construction.

The measured 215.3 ms = 1.91 x 112.74 ms is exactly the two-sweep signature.
Two candidate mechanisms produce it; both have the same fix-shape:

1. The AOT bundle used (`build/glm52_b12x_aot_b256_static`) actually contains
   generated buckets only up to 128, so B256 chunks as 2 x 128.
2. A true B256 bucket exists but the generated primitive internally tiles
   tokens at 128 granularity and re-reads expert weights per token-tile.

Discriminator on spark2 (minutes): dump the generated manifest bucket list
(`SparkGlm52Sm121B12xGeneratedManifestInstance` bucket token_upper_bound
values in the generated kernel table source), or count MoE kernel launches
per layer in Nsight for one B256 stage pass.

## The fork that decides the next 4x

Effective throughput is ~7 TFLOPS flat from B128 to B256 (0.776 TFLOP /
112.7 ms and 1.55 TFLOP / 215.3 ms). Two stories fit:

- **Bandwidth story:** the MoE sweep is ~fixed cost (~106-110 ms/stage) and
  B256 pays it twice because of chunking. Then restoring one-sweep execution
  (true B512/B1024 primitives or a weight-stationary dynamic path) gives
  roughly: B512 ~ 4100 tok/s, B1024 ~ 7000 tok/s per the fixed-sweep +
  ceil(B/128) projection model. That is the 10x lever.
- **Compute story:** the generated primitive tops out near 7 TFLOPS effective
  (bf16 MMA after NVFP4 dequant), and batch scaling past ~B128 is compute
  bound. Then the lever is the kernel itself: sm_121 block-scaled FP4 tensor
  core MMA consumes NVFP4 operands natively at ~4x the bf16 MMA rate and
  removes the dequant stage. Check what
  third_party/flashinfer/.../moe_dynamic_kernel.py emits for the MMA dtype.

Discriminating experiment: one true unchunked B256 stage pass. ~120 ms means
the bandwidth story (scaling alive, build the big buckets); ~215 ms means
compute bound (build the FP4-MMA primitive). Do this before investing in
either.

## DSpark draft backend (modules/glm52_dspark_draft_backend)

Implements the `SparkGlm52DsparkDraftFunction` boundary from
the historical DSpark implementation as a self-contained CUDA module:
C safetensors loader over spark_json (mmap, per-tensor shape/dtype checks,
device upload), qwen3 draft layers per the manifest-validated contract
(per-head q/k/v rms norm, rope, SwiGLU), five-tap fusion fc over a
backend-owned lane-major tap arena, block-window KV of BLOCK_SIZE positions,
restricted-vocabulary greedy argmax (full-vocab fallback), markov rank-256
low-rank logit correction, sigmoid confidence head. Weights ~7.5 GB bf16
resident. `TapOutputPointers` hands the five per-lane device buffers +
lane stride for `frame_context.dspark_hidden_tap_output_bf16` wiring;
`StageLane` records per-lane identity, position, last token, tap generation.

### Seam register (unverified against the real checkpoint - fix order)

1. Tensor names: two-alias lookup (`name`, `model.name`). Any miss prints
   the full checkpoint tensor inventory to stderr and fails. First real
   Initialize run against RedHatAI/GLM-5.2-speculator.dspark produces the
   complete rename list.
2. `SparkGlm52DsparkAssumedQwen3RopeTheta` (1e6) and
   `SparkGlm52DsparkAssumedQwen3RmsNormEpsilon` (1e-6): read the real
   config.json values.
3. `SparkGlm52DsparkComposeStepInputKernel` residual fusion: step input =
   embed(token) + (first step ? fc(concat(taps)) : previous pre-norm
   hidden). Verify against the speculators-framework DSpark reference.
4. Markov formula: logits[v] += markov_vocab[v] . (markov_hidden @ h), both
   orientations of the vocab projection supported. Verify.
5. Confidence: sigmoid(w . h + b). `confidence_head_with_markov` semantics
   not implemented beyond this. Verify.
6. Anchors (max_anchors 1024) unused in v1; draft attends only within its
   speculation window, conditioning on context through the taps.
7. Batching: Draft is per-sequence serial; at bf16 the 5 layers stream
   ~3.3 GB/token (~12 ms/token). Weight traffic is the cost, so the batched
   entry point (same kernels with a batch dimension) is the required v2
   before wide-batch speculative serving.

Bring-up sequence: manifest tool -> Initialize (fix names from inventory) ->
StageLane + Draft on a captured tap vector -> compare tokens against an HF
reference forward of the speculator.

Serving-adapter tap wiring (populating frame_context tap fields on
TAP_CAPTURE dispatches and calling StageLane after decode) is still absent -
the adapter has no dspark references today. The backend exposes everything
it needs.

## WMMA paged prefill attention

The bf16 paged prefill attention kernel ran 4-query x 2-key tiles per block
iteration, warp-per-pair scalar dots, per-thread serial AV with expf in the
inner loop - CUDA-core throughput on ~2.8 TFLOP of attention per 4k prompt.
Replaced with a 16x16 tensor-core flash kernel (shared-memory bf16 Q/K/V
tiles, wmma score chain, per-row online softmax, wmma PV with the running
accumulator row-rescaled in shared f32). Identical paging, causal masking,
variable-length, and output-layout semantics; the scalar bf16 kernel is
deleted (git history is the fallback); the fp8-KV prefill path keeps its
scalar kernel and is the follow-up once the bf16 kernel validates.

Verification: tests/test_glm52_exact_pp13_prefill_hidden.py is the numeric
oracle; then a wall-clock prefill measurement on a >= 2k prompt.

## Kernel audit shortlist (remaining structural wins, in expected order)

1. One-sweep large-batch MoE (chunking fix or FP4-MMA primitive per the fork
   above) - the 10x path continues here.
2. Batch projection kernel polish: vectorized shared-memory loads and
   K-slab double buffering (est. 1.2-1.5x on the projection phase), and
   M-fast grid rasterization so B >= 256 M-blocks hit L2 on shared weight
   strips.
3. Stage 72:6 was the B64 bottleneck: the final-token stage carries the
   fused restricted-logits epilogue; audit its batch scaling before the
   B512 sweep (restricted head is 256 x 6144 per token - cheap, but the
   epilogue also runs MTP verify/commit and event plumbing).
4. Decode MLA attention online kernels: scalar but context-length bound;
   irrelevant at short contexts, revisit for >= 8k serving.
