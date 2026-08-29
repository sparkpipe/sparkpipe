# DSPARK Scratch-KV Handoff - DSV4 Flash dspark-k7 (k=7) correctness

## Goal
Reach the locked no-spec baseline O128 hash `a9385d0b296ca083e577e715d2f6335067691dce0e0dd5ab1394a102a3d3631f` (driver `3d962820`, 40.19 tok/s). Current spec hash is `f717f68a3a7afda4` - WRONG.

## Current token stream vs baseline
- Baseline (lean control, CORRECT): `48582, 223, 2892, 201, 223, 20, 28, 539, ...`
- DSpark (current, WRONG):   `48582, 223, 20282, 24, 201, 223, 20, 28, 539, ...`
- Divergence at TOKEN 3: the verify's anchor prediction for token 223 @ pos 129 is `20282`, the true token is `2892`.

## What is LANDED and CORRECT (keep)
All in `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c` (uncommitted, +448/-51):
1. **Anchor broadcast**: head D2H gate `owns_final_head -> participates_final_head` (~line 3700). Anchor is now `223` on all ranks (was `0`).
2. **Draft maxloc all-reduce**: `SparkDsv4ModuleReduceDraftMaxloc` packs (score,token) into a u64, `SparkTpDeviceCollectiveSubmitU64Max`, unpacks, per markov step; plus a `phase==FREE` poll (the collective credit release is async vs `cudaStreamSynchronize`). Draft tokens are now IDENTICAL across ranks (rank0==rank3).
3. **Per-row causal attention**: `RunAttentionTail` loops `RunCompressorPost` + `RunIndexerCore` + `RunAttentionRows` one row at a time (wave 0..7) for `prefill==0 && rows>1`.

These three took the hash from `59ff947f` (degenerate 20,20,20...) to `f717f68a` (diverse stream).

## The scratch-KV machinery (option 2) - IMPLEMENTED, BAKED, but EXONERATED
Implemented exactly as the coordinator specified:
- `dspark_scratch_pool` = second raw-KV page pool sized `physical_page_capacity x page_stride_bytes` (allocated in the paged-cache init path).
- `dspark_main_pool` = `device_page_pool` (fixed main base).
- Seed: full `main->scratch` `cudaMemcpyAsync` before each verify, in `SparkDsv4ModuleRunFrame` (gated on `continuation->dspark_verify`).
- Second attention-island capture: `CaptureTpIsland` captures the ATTENTION island twice; the scratch capture sets `state->kv_cache_bf16 = scratch_pool` and `dspark_capture_scratch = 1`, stored in `island->scratch_executable`.
- Replay switch: `ReplayTpIsland` launches `scratch_executable` for ATTENTION islands when `dspark_scratch_active != 0`.
- **Per-row split** (in `RunAttentionTail`): `row_cache = (dspark_capture_scratch && row==0) ? main_cache : cache`. Row 0 (anchor) -> MAIN, rows 1..7 (drafts) -> SCRATCH. The compressor/indexer post get `row_cache` too. The anchor ALSO double-scatters into the scratch (`SparkDsv4LaunchCacheScatter` to `cache`) so drafts can read it.
- Commit: in `ContinueHeadMax`, after `accepted` is known, copy rows 1..accepted (accepted drafts only; the anchor is already in main) from scratch -> main per layer.

**Capture-level proof the split is real**: `dspark_split row0 main=0xe9b7a0000000 scratch=0xe9b760000000` fires during capture - the anchor uses the main pool, drafts use the scratch pool.

### WHY IT IS EXONERATED (the critical handoff fact)
Token 3 is `20282` under ALL THREE KV regimes: (a) no scratch, (b) full-scratch (all 8 rows in scratch), (c) correctly-baked two-base split (anchor in main, drafts in scratch). The anchor's prediction is INDEPENDENT of the KV-cache state, so the KV ring-clobber (cross-layer AND cross-frame) is NOT the cause of the 20282-vs-2892 divergence. My earlier slot analysis over-attributed it.

## The REAL remaining bug (unresolved)
The divergence is in the verify's 8-row COMPUTATION itself, independent of the KV cache. Candidates: embedding gather, Q/KV projection, RoPE, attention-score math, MoE routing, or the head - any batch-dependent (8-row vs 1-row) divergence vs the baseline's 1-row decode of token 223 @ pos 129.

## Next step (tensor-diff, needs a fresh agent)
Diff the verify's anchor (row 0, token 223 @ pos 129) against the lean baseline's decode of token 223. Dump, at layer 0 (and progressively deeper): the anchor's Q, post-RoPE K, attention logits, and post-attention hidden state, and compare to the same tensors from the baseline decode. The FIRST tensor that differs identifies the buggy kernel. The baseline lean runtime is pinned at `/tmp/dsv4-integrated-lean-3d962820-runtime` (driver `3d962820608fbad251aa50b7650dba2ab4b1d19ec378251c0e0ee36922e7fce4`).

## Temporary traces still in the code (strip on hash match)
`dspark_verify_pos`, `dspark_csa_gate`, `dspark_verify_tokens` (ExpandDsparkVerify); `dspark_head_tokens` (ContinueHeadMax); `dspark_tail_causal` (RunAttentionTail); `dspark_scratch_seed` (RunFrame); `dspark_scratch_replay` (ReplayTpIsland); `dspark_split` (RunAttentionTail); `dspark_maxloc_reduce_failed` (draft maxloc).

## Build/deploy/run loop
`./tools/devcycle.sh build lean-dspark-k7 8 7` (NAME BUCKET SPEC_STEP) -> driver in `/tmp/devcycle-build-lean-dspark-k7/model_driver.so`. Deploy: `devcycle driver dspark-k7 <so>`; restart: pkill residentd on spark4-7, sleep 30, `devcycle start`, `devcycle ready`; run: `devcycle run dspark-k7 1`. Baseline: `devcycle run lean 1`. Current driver `3969c80fe72c772a`.

## Fleet
spark4-7 live TP4 band; runtime `dspark-k7` at `/tmp/dsv4-dspark-k7-runtime`; port 18480. Layer kinds: 0-1 SWA, 2-41 alternating CSA(4)/HCA(128), 42 CSA. SLIDING_WINDOW_TOKENS=128, PAGED_POOL_BLOCK_TOKENS=128. Model: 43 layers, hidden 4096, vocab 129280, ATTN_HEAD_DIMENSION=512, ATTN_KV_HEAD_COUNT=1, ATTN_QUERY_HEAD_COUNT=64.