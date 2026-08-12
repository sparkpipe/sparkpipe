# Future dSpark optimization ledger

Status: research notes only.  dSpark/speculative decoding is disabled for the
current DeepSeek V4 Flash B1 qualification.  These ideas may be enabled only
after the no-speculative B1 result has a persisted correctness vector, step
latency breakdown, and reproducible baseline.

## Source pins and scope

These are public operator results, not SparkPipe measurements:

- `joesinvestments/DeepSeek-V4-Flash-0731-TP4-4x-DGX-Spark` at
  `1339c16375703280ca8e4bd70e49865dba560dd1`;
- `joesinvestments/GLM-5.2-QuantTrio-TP4-DCP2-4x-DGX-Spark` at
  `1ced1efc8729e6e08b25f37ddc2e998ba8250927`;
- `joesinvestments/gx10-bench-optimizer` at
  `2add44d8fe72e09271a4391618a07bda7ba5229c`.

The DeepSeek repository reports 123.13 tok/s for a warmed, approximately
2K-prompt, single-stream TP4 test with dSpark `k=7`.  It reports materially
different rates for deep agent traffic.  It is therefore neither a no-spec B1
roofline nor evidence that SparkPipe has reached that rate.  The GLM repository
reports 44.6 tok/s only for quantized MTP plus the Marlin atomic-reduction flag.
Both figures are hypotheses and comparison points until reproduced with the
same model, request shape, counters, and hardware state.

## Measurement tricks to adopt first

1. Measure decode using model-server counters, not SSE chunk count or total
   request wall time.  The public recipe uses
   `generation_tokens_delta / request_decode_seconds_delta`, then checks a
   post-TTFT client-side rate as a secondary result.
2. Record prompt length, output length, concurrency, cache state, thinking,
   sampling, checkpoint, precision, topology, and speculation on every result.
   A naked tok/s number is not comparable.
3. Use two full-output warmups plus enough discarded repetitions to stabilize
   graphs and kernels.  The DeepSeek operator measured an 11 tok/s difference
   between under-warmed and warmed runs.
4. Fail closed on foreign traffic.  Snapshot completed-request counters before
   and after a cell and reject the cell unless the delta equals the harness's
   own completions.
5. Cache-bust cold-prefill probes and separately measure warm shared-prefix
   traffic.  Prefix-cache hits must be recorded, not inferred.
6. Label the output-content class.  Speculative acceptance changes with code,
   prose, structured tools, temperature, and context depth.
7. Report distributions and step time: median, min, max, standard deviation,
   p50/p95/p99 inter-token latency, tokens/step, and per-position acceptance.
8. Change one variable per boot, preserve the exact launch argv and binary
   hashes, and reject a faster cell if deep-prefill or concurrency stress fails.

## dSpark and MTP tricks

### Draft depth is an engine-shape parameter

- Sweep `k`; do not copy it from another engine.  The public DeepSeek build was
  specialized for `k=7`, where it beat `k=3`, `k=5`, and `k=8`.  A different
  GLM workload preferred `k=2` over `k=4` at deep context.
- Compare tokens/step and per-position acceptance, not aggregate acceptance.
  A higher aggregate percentage can still produce fewer accepted tokens per
  verifier step.
- Re-measure every position for each `k`.  Increasing `k` can reduce acceptance
  at earlier positions; the shorter curve cannot safely predict the longer one.
- An adaptive batch-size ladder has a per-step control cost.  It won aggregate
  throughput at sustained high concurrency but lost about 7% on the cited
  single-stream DeepSeek test.  Gate it on the workload distribution.
- Probabilistic versus greedy sampling is model-, temperature-, and content-
  dependent.  The public DeepSeek recipe observed a material production-shape
  win from probabilistic sampling, while one temperature-zero GLM ablation was
  noise.  Both paths require an isolated accuracy/acceptance sweep.

### Graph and scheduling geometry

- Size speculative graph capture for the expanded batch:
  `capture_rows = max_num_seqs * (k + 1)`.  Too small a capture can silently
  drop higher concurrency into eager execution while B1 remains fast.
- If a scheduler reserves draft positions from the token budget, compensate
  explicitly.  The cited vLLM relation is
  `effective_prefill = max_num_batched_tokens - (k - 1) * max_num_seqs`.
- Measure prefill-chunk size against decode interruption.  Larger chunks improve
  cold TTFT but can stall all active decode streams for longer.
- Graph prewarm must run before readiness and must not mutate KV.  Runtime graph
  fallback must be explicit in telemetry; production qualification fails if a
  supposedly captured shape executes eagerly.

### Quantized draft small-M reduction

The GLM TP4 operator isolated a severe quantized-draft regression to Marlin's
small-M reduction: atomic accumulation disabled, FP32 global reduction, repeated
once per draft forward.  Setting `VLLM_MARLIN_USE_ATOMIC_ADD=1` changed the
reported quantized-draft result from 6.5 to 44.6 tok/s, with 42.4 and 46.9 tok/s
repeats; the BF16 draft did not benefit.

SparkPipe does not use Marlin, so the environment variable itself does nothing.
The transferable experiment is:

1. profile every tiny-M draft GEMM epilogue and reduction separately;
2. compare fused atomic accumulation with workspace/global reduction;
3. validate numerical error, ordering sensitivity, contention, and determinism;
4. retain the atomic path only for shapes where end-to-end verifier step time
   improves.

Do not apply this conclusion blindly to the no-spec native MXFP8-by-MXFP4 W13/W2
kernels; their reduction geometry is different.

### Quantized draft tensor mapping

The GLM public patch adds `qkv_proj` and `gate_up_proj` to the draft quantizer's
`packed_modules_mapping`.  Without the mapping, enabling
`quantization: compressed-tensors` can leave fused draft tensors incorrectly
loaded or unquantized while startup still succeeds.  A SparkPipe dSpark pack must
therefore carry a complete, generated tensor-name/offset/codec manifest and pass
bit-exact load receipts for every fused projection before acceptance is measured.

## Attention, KV, and context tricks

- Quantized MLA KV is a capacity-versus-speed choice, not a universal win.  The
  cited GLM data found NVFP4 useful for capacity/deep prefill but slower on a
  high-acceptance shallow decode.  SparkPipe must sweep BF16, lossless packed,
  FP8, and NVFP4 encodings at equal contexts and include reference-vector drift.
- Context ceilings can change the sparse-indexer working set and B1 decode even
  when the live prompt is short.  Benchmark the actual context ceiling, not only
  the current sequence length.
- The cited GLM DSA stack exposed a survivable product near
  `max_model_len * max_num_batched_tokens = 262144 * 1024` on its TP4 build.
  This value is not portable, but the delayed-failure pattern is: boot and short
  prompts pass, then the first deep cold prefill fails.  Qualification needs a
  max-context cold-prefill stress cell.
- MTP may overrun a block table by one block when context length is exactly
  aligned.  The public patch adds one block of draft headroom.  Test boundary
  lengths on both sides of every KV block multiple and under concurrency.
- One cited stack had two inconsistent block-table alignment calculations and
  failed when `max_model_len` was not divisible by 64.  Centralize this geometry
  in one generated contract rather than duplicating formulas.
- Prefix caching is central for long-lived agents.  Preserve prior-turn content
  exactly, group shared prefixes, record hit rate, and avoid chat-template
  behavior that rewrites history.

## Fabric and process tricks

- Discover the sole lower-case `(Up)` RDMA device from `ibdev2netdev` on every
  host after every boot.  Do not hardcode a device or GID.  Refuse the launch if
  ranks disagree on the resolved fabric identity.
- Verify every rank and exact argv before declaring the API healthy.  A head
  endpoint may answer health/model queries while a worker is absent, with the
  first collective failing later.
- Start workers before the API/head rank and persist an all-rank readiness
  receipt.
- Ensure locked-memory and file-descriptor limits are sufficient.  The cited
  TP4 recipe warns that missing IPC lock/memlock can force a silent TCP path and
  that TP socket counts require a high `nofile` limit.
- Treat collective-wedge mitigation as a measured trade.  NCCL rail/channel/QP
  flags improved or harmed small-message latency depending on the stack.  For
  SparkPipe, retain the native transport and measure each allreduce phase,
  queueing delay, bytes, rail, and fallback count directly.
- An HTTP-only watchdog is insufficient.  Detect the state where HTTP responds
  while both prefill and decode counters stop; dry-run the recovery launcher and
  compare its exact config with the live process.

## Serving tricks

- Use streaming clients and cancel server work when a client disconnects.
  Non-streaming timeouts plus retries can leave zombie generations consuming
  their full output budget.
- Separate deep-session, cold-admission, sustained batch, and shallow B1 lanes.
  A single configuration need not win all four.
- Default thinking and temperature explicitly.  Thinking consumes the output
  budget; sampling changes speculative acceptance.
- Drop or account for host page cache before loading hundreds of GB on unified
  memory systems so startup memory checks measure the intended GPU/runtime
  allocation rather than stale weight pages.

## SparkPipe experiment order

1. Close no-spec B1 correctness and latency on TP4 and TP16.
2. Persist per-kernel, per-layer, allreduce-phase, graph-launch, and KV timings.
3. Add one dSpark depth with exact draft-pack manifests and no dynamic fallback.
4. Validate target-token equality against the no-spec reference path before
   measuring acceptance.
5. Sweep draft depth, sampling, graph rows, quantized-draft reduction, KV codec,
   context ceiling, and prefill chunk one variable at a time.
6. Run deep cold-prefill and mixed-concurrency stability gates before accepting
   any throughput winner.

The no-spec baseline remains a permanent control cell after dSpark lands.
