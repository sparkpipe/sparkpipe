# DSV4 Flash TP4 benchmark matrix — 2026-08-25

Full-matrix attempt against the live DSV4 Flash TP4 B1 serving stack
(spark4..spark7, control :18480). B1 cells measured in full with throughput
AND end-to-end latency. B4/B16 cells are hard-blocked by two independent,
receipt-backed limitations (below). Receipts: ./receipts/.

## Stack identity (verified before measurement)

- Runtime root all ranks: /tmp/dsv4-integrated-lean-3d962820-runtime
- residentd binary sha256 b4118d6b4b824384... (lean_residentd == sparkpipe_model_residentd)
- model_driver.so 4675e35fa5a20349... (= /tmp/devcycle-build-lean artifact)
- Serving env SPARKPIPE_RELEASE_GIT_COMMIT = ef8fa302ad8f545e...
  ("feat(dsv4): retain exact TP4 B1 winners and handoff") — the exact handoff-
  selected tip; all five handoff source pins re-verified byte-identical.
- Canonical O128 reproduced live BEFORE the matrix: decode 40.59 tok/s, exact
  pinned token hash a9385d0b296ca083...

## Methodology

- Client: tools/model_stream_decode_benchmark.py wrapping ROOT/bin/sparkpipe_model_batch.
- Per cell: 1 discarded warmup + 3 measured runs; mean over runs reported.
- Latency columns (all means):
    ttft_s   process-start -> first token (client-inclusive upper bound;
             includes ~0.3-0.4 s spawn+ssh+connect overhead)
    ttfb_s   'accepted' event -> first token (transport-pure prefill + first step)
    e2e_s    process-start -> last terminal event (total completion time)
- Prefill effective rate = prompt tokens / ttfb_s. Marginal rate computed from
  the ctx512 -> ctx2048 delta (removes any residual constant).

## MEASURED MATRIX — B1 (live stack)

cell            B ctx   kind     dec tok/s  ttft_s   ttfb_s   e2e_s
decode_ctx512   1  512  decode      39.15   13.95   13.56    17.19
decode_ctx2048  1 2048  decode      37.44   55.76   55.29    59.16
prefill_ctx512  1  512  prefill     n/a     13.91   13.60    13.95
prefill_ctx2048 1 2048  prefill     n/a     55.74   55.28    55.78

Derived rates:
- prefill effective  ctx512 : 37.6 tok/s;  ctx2048: 37.0 tok/s
- prefill marginal (1536-token delta): 36.8 tok/s  -> flat ~37 tok/s
- decode: 39.15 tok/s @ctx512, 37.44 @ctx2048 (-4.4% attention cost)
- first-token latency is prefill-dominated at these contexts (55 s @ctx2048);
  decode adds only ~3.3 s for the full 128-token completion at B1.

## Cache ingestion

Prefill IS the KV ingestion path ("resident cached prompt KV"). Measured
ingestion rate = ~37 tok/s, flat across 512/2048 (marginal == absolute within
noise). Identical re-submissions of the same sequence_id+prompt always paid
full prefill again: NO prefix-hit shortcut is observable through the batch
client API at this generation. A ctx4096 probe is impossible by contract
(max_sequence_tokens 4096 leaves no room for an output token).

## WIDE CELLS (B4/B16) — BLOCKED, with evidence

1. The serving width-1 pair rejects capacity>1 batches client-side:
   status=2, requests rejected before submit.
2. Bucket-4 and bucket-16 drivers were built specifically for this matrix
   from the exact serving commit ef8fa302 plus a mechanical, additive-only
   bucket-ladder extension (generated dense firmware descriptions b4/b16 via
   tools/generate_dsv4_contracts.py, their sha256 contract defines, adapter
   ladder branches, native-width whitelist {1,4,8,16,1024}; no kernel or
   arithmetic changes). Both compile; module-b4 variant build succeeds.
   At adapter_initialize every rank fails deterministically:
       dsv4_stage cuda_error site=tp_graph_projection_island error=invalid argument
       model_residentd initialize=internal_error status=17 phase=adapter_initialize
   i.e. width!=1 TP graph islands are not capturable by this source
   generation (native whitelist was {1,8,1024} for good reason).
   Artifact hashes: db99c48cf3256439 (b4 driver), a1a1049414b94876 (b16 driver),
   receipts/wide-artifact-hashes.txt.
3. Mid-session, a concurrent b8 bench (/tmp/dsv4-bucket8-runtime) took the
   standard collective topology on ranks 1 and 3, so further exclusive
   retries were impossible without disrupting that run.

Conclusion: B4/B16 cells require upstream graph-capture work at width 4/16
(the handoff's next-step #4), not configuration. They are NOT measurable on
this stack generation.

## vLLM reference comparison

spark3:8124 was UNREACHABLE during the session (direct curl and on-host curl
both fail). Recorded measured baseline (docs/ARCHITECTURE_MAP.md):
- prefill ~740 tok/s flat across batch  (~20x DSV4 Flash TP4 B1 prefill)
- decode 16 -> 226 tok/s scaling by batch (DSV4 B1 measures 37-40 aggregate)
A fresh apples-to-apples vLLM run needs that endpoint restored.

## Fleet state at close / how to restore

Mid-session contention left the b8 bench owning :18480 on ranks 1 and 3.
To bring the user's lean stack back after the b8 bench vacates, run on each
rank i in 0..3 (script staged at /tmp/restore_live.sh on every rank):

    bash /tmp/restore_live.sh i

which relaunches bin/lean_residentd from
/tmp/dsv4-integrated-lean-3d962820-runtime with the snapshotted environment
(GIT_COMMIT ef8fa302..., RELEASE_ID dsv4flash-serving-rank-i). Then verify:
canonical O128 must return 40-41 tok/s with hash a9385d0b296ca083....

## Restoration record (same day, after the b8 bench vacated)

The concurrent bucket-8 bench stopped its loop and left the band; the
always-on lean stack was restored from the staged snapshot script on all
four ranks and re-verified:

    fleet probe at close: spark4-7 DSV4 Flash serving again
    (18480 / 62620-62623 / 59700); qwen band idle; spark8-f free;
    sparkc unreachable.
    Post-restore canonical O128: 40.55 tok/s decode, exact pinned token hash
    a9385d0b296ca083e577e715d2f6335067691dce0e0dd5ab1394a102a3d3631f
    (receipts/restore-final-o128.json).

The band is back to always-on service.
