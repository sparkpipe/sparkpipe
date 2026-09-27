# MGR HANDOFF — 2026-09-27

Read with docs/ROADMAP.md (the 8-milestone program), TECHDEBT.md, and the PR
threads named below. I am the hands; claude/opus is the architect and delivers
bundles (BRANCH.bundle + PR_DESCRIPTION_ITER*.md); my loop is: open PR → dispatch
CI (#1111 stall workaround: `gh api .../dispatches -f ref=BRANCH`) → build on
spark0 via the pinned queue → deploy fleet → measure → merge with receipts →
comment results on the PR.

## Program state

- **Main at 992880f6** (i17 merged). Station serves **glm on b108de0c (i17
  stack) + the i17 API binary**: 6/6 deterministic reference, ~130 tok/s
  aggregate at 8 streams (best verified), tokens identical, zero failures.
- **i18 (#1229, glm/decode-chains @ 042d6883) is HELD, not merged.** Deployed,
  measured, three defects found, rolled back cleanly. Evidence on the PR.
- Cumulative arc: 12.81 tok/s B1 (Sep-22) → 33.6 B1 / 129.9 aggregate.
  Harness (one-rank, real kernels): B256 = 1128 tok/s (i11r2), B128 = 608.
  M1 = 80% of memory roofline: B8 step ≤26 ms, B1 ≤10 ms/token.

## THE THREE OPEN DEFECTS (all evidence on PR #1229)

1. **Prefix-checkpoint token divergence — PRE-EXISTING, reproduces on i17.**
   A request reusing a published 128-token block checkpoint gets a DIFFERENT
   first token than the cold path (2403 cold vs 68947 cached). Repro: clean
   API restart + long300 alone → identical; replay the capture sequence
   (prefix100 then long300) → token0 flips, `cached_prompt_tokens=128` in the
   API log. NOT filed as its own issue yet — needs an issue + fix. All
   before/after token captures MUST isolate cache state per prompt (API
   restart or unique prompts); the i18 "token-0 diff" was this bug plus a
   flawed capture protocol on my side.

2. **i18 wedge:** after ~16 requests the FIFO head froze in BUSY; resident
   rejected all prefills `status=15 ... last_id=<frozen>`; wave lines showed
   `busy=104/0/0/0/0` → `busy=105/0/0/0/0` — ONLY the chain ('C') counter
   climbing, zero release frames. Same class as the i16 busy-head starvation.

3. **i18 chains never engaged:** zero `CHAIN-TIME path=graph steps>=2`;
   decode ran one step/frame throughout.

## RETEST STATUS (mid-flight — where I stopped)

The architect's mixed-bucket hypothesis: my retest DISPROVED part of it and
the wedge cause is now UNRESOLVED again:
- NO deployed adapter .so contains a `SPARK_BATCH_BUCKET` string (rank or
  API side). The `SPARK_BATCH_BUCKET=1024u` string I reported came from a
  STALE top-level build artifact in the i18 source tree, not the deployed
  binary — that finding on the PR is WRONG and needs correcting.
- Traced the K computation (runtime/model_batch_engine.c ~1935):
  `chain_tokens = max_output_token_count / lane_count`, capped at
  `SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE` =
  `SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE` = **8, hardcoded in
  spark_model_driver.h**. `max_output_token_count` =
  `SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT` =
  `SPARK_BATCH_BUCKET` (compile-time, per-build). So even a 1024-bucket
  API adapter asks K = min(8, 128, budget, context, block-remaining) ≤ 8 —
  **bucket mismatch does NOT explain chains-not-engaging or the wedge.**
- The rebuilt bucket-8 x86 adapter is at
  rtx5090:/home/spec/glm-i18src/build/.../libglm5_next_serving_adapter_fp8.so
  (unverified — no string marker exists to confirm bucket; needs a different
  verification: read the descriptor's max_output_token_count at runtime, or
  check the module-id in COMPONENT data).
- **Next for whoever picks this up:** (a) correct the record on #1229 (the
  bucket-string finding was a stale artifact); (b) find the real wedge cause —
  leading suspects: the linear single-step completion path under the new
  engine publish rule, or GatherSteps/SettleStep never being reached since
  chains didn't engage; (c) figure out why chains didn't engage — engine only
  requests K>1 when the adapter descriptor has RESIDENT_DECODE_CHAIN; verify
  the deployed i18 adapter actually exposes it (the capability is compiled
  into the adapter .so the API loads on rtx5090 — check the API's resident
  handshake log line or add a descriptor dump); (d) the wedge busy=C pattern
  with NO chains running means something else took chain claims — check
  SparkGlm5NextClaimTpChain callers on the 042d6883 build.

## Rollback state (how to restore i18 for the retest)

- glm registry (station.json) → b108de0c; unit files for 042d6883 still
  exist on all nodes; release dirs glm-serving-042d6883/ deployed on 16/16.
- i18 API binary backed up at rtx5090:/home/spec/sparkpipe-recovery-a2fbb5f4/
  bin/sparkpipe_model_api.i18 and runtime/lib/model_serving_adapter.so.i18;
  active binaries are the i17 ones (.bak-i17 restored).
- To retry: swap .i18 files back, registry to 042d6883, stop b108de0c units +
  untrack (queue demands verifiably-stopped before untrack), start glm.

## Deploy laws (accumulated — each paid for)

- Roll sets: weightd+weightd_warm+driver+adapter+measurement tools together on
  mesh-ABI changes; glm unit ExecStartPre warm path repoints per core
  generation; the mesh ladder binary carries the schema too.
- Driver-only rolls: new glm-serving-<sha> dir (packs/config copied, absolute
  station-manifest), residentd from the release tarball's bin/, warm from the
  live core. Untrack old persistent owners (must be verifiably stopped —
  `sudo systemctl stop` first, then `untrack`). Absolute paths in manifests
  ALWAYS (relative paths fail verification).
- x86 API-side adapter on rtx5090 built natively (CUDA_HOME=/usr/local/
  cuda-13.3, sm_120); new recovery root or in-place swap + manifest regen.
- NEW (from i18): record the module bucket per release and verify BOTH sides
  built at it before swap — despite the hard cap making K safe anyway, mixed
  buckets are still a latent hazard. NEW: cold-cache token protocol for all
  before/after captures.
- Token protocol: greedy, ≥64 tokens; for multi-chunk prompts isolate cache
  state per prompt (restart API or unique prompts).

## Fleet/station state

- Station core: c0d54638 (i13 timing weightd + c0d54638 warm). MESH0005.
- glm: lane 0, B8 profile, 128-row prefill, streaming/priority/deadline/
  overload/audit API (i12), seeded sampling (i13). weightd timing
  instrumentation live (WD-MESH-TIMING/WAVE per i13/i15).
- **Six other families DOWN since the iter-4 weightd roll** (stale drivers;
  q27rescue build was cancelled). Rescue = rebuild drivers from current main
  per family. Serving-capable after rescue: qwen27 (B1 worked), gemma4 (4
  requests then #1200 wedge). Others fail first inference (their fixes are
  roadmap items). k3 re-init still fails; qmax needs admission stub.
- The rig (10.20.0.1) not observed recently; treat sudden unaccounted GPU
  processes as it returning.

## Measurement inventory (use, don't rebuild)

- Harness: build/glm5_next_batch_roofline (checkout /home/spark0/srcdata/
  sparkqueue/i18/042d6883*/ on spark0); --batches ≤256 (mesh cap ×2),
  --index-cp 16, ROOFLINE/PHASES/EXPERTS/FLEET lines.
- Load: /tmp/long_output_test.py (8×512-token streams, on rtx5090);
  /tmp/bench_glm6_det.py (6×32 reference); /tmp/concurrency_test.py.
- Reports: tools/mesh_timing_report.py (WD-MESH-TIMING), tools/
  wave_timeline_report.py (G5N-WAVE-TIMING), CHAIN-TIME path= split one-liner
  (on the PRs). Before-token captures: /tmp/i18_before.json (FLAWED protocol
  for long300 — cache-state mixing), /tmp/i17_before_tokens_keep.json.

## Priority queue

1. Correct the #1229 record (stale-artifact bucket string) — architect's
   next iteration depends on true facts.
2. File the prefix-checkpoint divergence bug (pre-i18 repro recipe on PR).
3. Root-cause i18 wedge + chains-not-engaging (leads above); retest with
   matched builds + clean token protocol.
4. M1 1a second half per ROADMAP: prefill graph capture vs decode-ahead
   (i17 data: linear chunks 161ms, host_submit 76.6ms = launch-bound).
5. Rescue the 6 down families (qwen27 first).
6. M1 1b-1d per ROADMAP; b16 variant deploy still staged (build_b16.sh).

## UPDATE — 2026-09-27 afternoon (supersedes "RETEST STATUS" above)

- **#1229 record corrected** (comments 5851936994 + 5852110456): no bucket mismatch ever
  existed — probed via dlopen/GetInterface: API `.so.i18`, API i17, node 042d6883, node
  b108de0c adapters are ALL max_output=1024 (tarballs ship the Makefile:128 lastword
  archive). Zero ADMIT9-RESIDENTD status=2 in 172k journal lines. 8-lane K=8 chains RAN
  and completed during load (waves≈rows/8, steps≈8/wave; 123 chains, 25–51 ms/step).
- **Wedge anatomy**: chain finishes (CHAIN-TIME status=0, claim+slot released) but the
  route never completes residentd-side (active_owner held → status=15 storm, frozen
  last_id). Zero failure/heartbeat lines — v1 has no CHAIN-HEARTBEAT. Leads: parked-
  completion pool has no independent drain (module.c:4261; drains only from the next
  CompleteOnWorker or MTP-resolve, which is off in production); async-slot-reuse race
  (module.c:4247).
- **NEW: the same wedge reproduces on i17** — API restart → ref176 OK → restart →
  long400 (4-chunk) → HTTP 500 → frozen last_id=1000122 → restarted API hits
  engine_connect_deadline until glm units are restarted on all 16 nodes. The 6/6 det
  bench (single-chunk prompt) cannot catch it. Station restored + 6/6 verified.
- **Issue #1230 (prefix cache) = the worse branch**: long300.prompt[100:128] vs
  prefix100.tokens[0:28] → 0/28 match; the reused block's KV held prefix100's
  degenerate [2435,8850] generation. Checkpoint identity is keyed on too little.
- **v2 expected** (hello-ack max_output compare, bucket×8 capacity, hang fix). I asked
  that the hang fix target the shared completion-delivery path — both lineages carry it.
- **Retest kit staged** for v2: rtx5090 /tmp/cold_capture.py + /tmp/retest_prompts.json
  (restart-per-prompt, retry-with-restart on 500, unit-restart unwedge). Still open:
  1-stream K=8 token equality (first test was polluted by #1230).
- Law added: zsh does not word-split unquoted variables — "restart all nodes" loops
  silently no-op unless the node list is literal.
