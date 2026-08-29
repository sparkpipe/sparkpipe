# DSpark Verify Handoff - DSV4 Flash dspark-k7 (k=7) correctness

Goal: reach locked no-spec baseline O128 hash `a9385d0b296ca083e577e715d2f6335067691dce0e0dd5ab1394a102a3d3631f` (driver `3d962820`). Current spec hash `f717f68a3a7afda4` - WRONG. Baseline stream `48582,223,2892,201,...` vs DSpark `48582,223,20282,24,...`; divergence at TOKEN 3 (anchor 223 @ pos 129 -> 20282 vs 2892).

## (1) Landed-correct fixes (KEEP)
All in `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c`:
1. **Anchor broadcast**: head D2H gate `owns_final_head -> participates_final_head`. Anchor is `223` on all ranks (was `0` on ranks 0-2).
2. **Draft maxloc all-reduce**: `SparkDsv4ModuleReduceDraftMaxloc` packs (score,token) into a u64, `SparkTpDeviceCollectiveSubmitU64Max`, unpack, per markov step; plus a `phase==FREE` poll (collective credit release is async vs `cudaStreamSynchronize`). Draft tokens now IDENTICAL across ranks.
3. **Per-row causal attention**: `RunAttentionTail` loops `RunCompressorPost` + `RunIndexerCore` + `RunAttentionRows` one row at a time (wave 0..7) for `prefill==0 && rows>1`.

These three took the hash from `59ff947f` (degenerate 20,20,20...) to `f717f68a` (diverse stream).

## (2) Scratch-pool machinery state - verdict: STRIP (exonerated)
Implemented exactly per the coordinator's option 2, but does NOT fix token 3:
- Allocation: `dspark_scratch_pool` (second raw-KV page pool, `physical_page_capacity x page_stride_bytes`) + `dspark_main_pool` = `device_page_pool`.
- Seed: full `main->scratch` `cudaMemcpyAsync` before each verify in `SparkDsv4ModuleRunFrame` (gated on `continuation->dspark_verify`).
- Capture: `CaptureTpIsland` captures the ATTENTION island twice; the scratch capture sets `kv_cache_bf16 = scratch_pool` and `dspark_capture_scratch = 1`, stored in `island->scratch_executable`.
- Replay: `ReplayTpIsland` launches `scratch_executable` for ATTENTION islands when `dspark_scratch_active != 0`.
- Per-row split (`RunAttentionTail`): `row_cache = (dspark_capture_scratch && row==0) ? main_cache : cache`; row 0 -> MAIN, rows 1..7 -> SCRATCH; compressor/indexer follow the per-row base; anchor double-scatters into scratch (`SparkDsv4LaunchCacheScatter`) so drafts read it.
- Commit: in `ContinueHeadMax`, copy rows 1..accepted (accepted drafts only) scratch->main per layer.

**STRIP-OR-KEEP: STRIP.** The machinery is harmless but does not move token 3. On hash match, remove all of it (allocation, seed, capture, replay switch, per-row split, commit) along with the temporary traces, keeping only the three fixes in section (1).

## (3) Exoneration evidence (KV ring-clobber is DEAD)
- Token 3 = `20282` under ALL THREE KV regimes: (a) no scratch, (b) full-scratch (all 8 rows in scratch), (c) correctly-baked two-base split (anchor in main, drafts in scratch).
- Capture-level proof the split is real: `dspark_split row0 main=0xe9b7a0000000 scratch=0xe9b760000000` fires during capture - the anchor uses the main pool, drafts use the scratch pool.
- Conclusion: the anchor's prediction is INDEPENDENT of the KV-cache state, so the cross-layer and cross-frame ring-clobber are NOT the cause. My earlier slot analysis over-attributed it.

## (4) Next investigation spec - tensor-diff at the FIRST divergence
Diff the verify anchor (row 0, token 223 @ pos 129) against the baseline 1-row decode of token 223. For the first diverging layer, dump and compare:
1. anchor Q (post WQ projection),
2. anchor K post-RoPE,
3. attention logits (Q.K scores),
4. post-attention hidden state.

Compare each to the SAME tensor from the lean baseline decode of token 223 @ pos 129. The FIRST tensor that differs names the buggy kernel (embedding gather / QKV projection / RoPE / attention-score / MoE / head). The baseline lean runtime is pinned at `/tmp/dsv4-integrated-lean-3d962820-runtime` (driver `3d962820608fbad251aa50b7650dba2ab4b1d19ec378251c0e0ee36922e7fce4`).

## Temporary traces still in the code (strip on match)
`dspark_verify_pos`, `dspark_csa_gate`, `dspark_verify_tokens` (ExpandDsparkVerify); `dspark_head_tokens` (ContinueHeadMax); `dspark_tail_causal` (RunAttentionTail); `dspark_scratch_seed` (RunFrame); `dspark_scratch_replay` (ReplayTpIsland); `dspark_split` (RunAttentionTail); `dspark_maxloc_reduce_failed` (draft maxloc).

## Build/deploy/run loop
`./tools/devcycle.sh build lean-dspark-k7 8 7` -> `/tmp/devcycle-build-lean-dspark-k7/model_driver.so`. Deploy `devcycle driver dspark-k7 <so>`; restart: pkill residentd on spark4-7, sleep 30, `devcycle start` + `devcycle ready`; run `devcycle run dspark-k7 1`; baseline `devcycle run lean 1`. Current driver `3969c80fe72c772a`.

## Fleet / model facts
spark4-7 live TP4 band; runtime `dspark-k7` `/tmp/dsv4-dspark-k7-runtime`, port 18480. Layers 0-1 SWA, 2-41 alternating CSA(4)/HCA(128), 42 CSA. SLIDING_WINDOW_TOKENS=128, PAGED_POOL_BLOCK_TOKENS=128, hidden 4096, vocab 129280, ATTN_HEAD_DIMENSION=512, ATTN_KV_HEAD_COUNT=1, ATTN_QUERY_HEAD_COUNT=64. Repo uncommitted (+448/-51 in the one module file); no commits/pushes per scope.