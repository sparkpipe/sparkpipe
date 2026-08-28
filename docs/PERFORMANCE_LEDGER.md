# Performance Ledger

The authoritative record of what is measured, on what code, on what hardware.
Every number here has a receipt. Numbers not in this file are not claims.

Updated: 2026-08-28 (UTC)

## SCOREBOARD (machine-read; rendered every coordinator sweep)

One row per model: best MEASURED decode and prefill, with the cell that
produced them. "—" = not yet measured (the sweep prints these gaps).
tools/perf_scoreboard.py parses this section; edit HERE, never downstream.

<!-- scoreboard:start -->
| Model | Best decode (tok/s) | Decode cell | Best prefill (tok/s) | Prefill cell | Code | Date |
|---|---|---|---|---|---|---|
| Qwen 3.8 27B | 24.5 (spec) / 8.45 (no-spec) | B1 TP1 DFlash2 k8 / B1 TP1 71.1GiB cfg | ~21.7 | p256 B16 PFR8 | main | 2026-08-28 |
| Qwen 3.8 27B aggregate | RETRACTED-WAS-UNITS-BUG: true curve from same receipts = 8.31/8.31/13.5/24.1/41.3/41.3 (B1-32) — the sweep tool's awk multiplied the already-batch-total token count by B (operator caught it via roofline: 1469 needed 79 TFLOPS on ONE spark). Batching gains ~5x to ~41 tok/s saturation, not 174x. Re-derivation + flat-B2 anomaly + step-ms re-check pending | B1-32 TP1, spark3, corrected from raw wall+tokens | — | — | receipts valid, tool bug | 2026-08-28 |
| DSV4 Flash | 40.46/40.35/40.19 main+fix (O128 cell, exact hash 211462f2; lean 39.81/39.77); exact-32K cell tops ~29 (lean 29.37 — the old 40.4-at-32K attribution was WRONG; 32K costs ~10 tok/s in attention scaling); main was BROKEN at the cell (9 tokens, FAILURE_CONTINUE_LEASE), not slower — fixed by PR #731 (dspark gate + lease mirror) | B1 TP4, O128 + exact-32K cells | — | not measured | main+PR731 | 2026-08-28 |
| GLM 5.2 (donor) | 6.91 | B1 TP8 (historical); CORRECTNESS RESTORED on current code: full validator PASS + 8-rank B1 e2e decode (PR #728, timing not claimed) | — | not measured | main b313cd5 | 2026-08-28 |
| GLM 5.2 aggregate (donor) | 75.55 | B16 TP8 | — | not measured | pre-audit code | historical |
| Qwen 3.8 Max | 1.29 | B1 TP4xPP4 anchors | — | not measured | anchors | 2026-08 |
| K3 | 18.0 | B1 single-stage; MODULE CHAIN VERIFIED through collective on real deployed pack (fleet bring-up = first fleet number) | — | not measured | single-stage + main fixes | 2026-08-28 |
| Qwen 3.8 Flash | — | M5 whole-stack TP4 gate PASSED (bit-exact, cross-node); M6 live smoke + hc/indexer/PLE ports next | — | — | main e3b795a | 2026-08-28 |
| GLM 5.3 Flash | — | FLEET LIVE: 16/16 TP16 ranks READY on real packs (simultaneous wave, ready-lines verified, API connected) — one isolated blocker to first tokens (API batch-engine submit path); M5 exact-32K B1 next (PR #727 feeeae7) | — | — | lane/glm53 | 2026-08-28 |
| DSV4 Pro | — | 16 TP4xPP4 rank packs built+verified (~93GB each, DSpark draft replicated), 14/16 deployed; old packs were never corrupt (verifier mismatch by construction); driver rebuild pending ready line (PR #729) | — | — | lane/dsv4pro merged | 2026-08-28 |
<!-- scoreboard:end -->

## DSV4 Flash (DeepSeek V4 Flash, FP8, TP4)

| Metric | Value | Code | Status |
|---|---|---|---|
| B1 decode (no spec) | 33.55 tok/s | merged main | **VERIFIED** — last clean main qualification |
| B1 decode (no spec) | 40.15–40.48 tok/s | lean branch `3d962820` | **BRANCH-LEVEL** — 8 interleaved A/B experiments (2026-08-14/15), exact-token hash `211462f2…` pinned, re-confirmed 40.19 on 08-17. NOT on main. |
| Internal vLLM reference | 37.79 tok/s | retained plain-vLLM TP4 | sanity check |
| Public incumbent (FP8-KV DSpark) | ~90 tok/s | community | target to beat |
| Public incumbent (NVFP4-KV) | 123 tok/s | community | stretch target |

**Gap**: the 40 tok/s lean stack's commit is not an ancestor of main. The lean
delta needs to be cherry-picked, rebuilt, requalified on merged main, and the
contract flipped from `NOT_MEASURED` to measured. The lean runtime and packs
are deployed on spark4–7; the receipts are in
`qualification/dsv4/performance/tp4_b1_20260815/`.

## Qwen 3.8 27B (FP8, TP1, DFlash2)

| Metric | Value | Code | Status |
|---|---|---|---|
| Spec decode (DFlash2 k=8, O512) | 24.5 tok/s wall | merged main (renamed) | **VERIFIED** — stream `d7f798801a6e43a6`, 77 rounds, E≈5.66 |
| No-spec decode (O512) | 7.7 tok/s | merged main (renamed) | **VERIFIED** — stream `5d6ee525deb999f5` |
| No-spec HWM (B1) | 8.03 tok/s | mixed/FP8 29.9GB pack | dashboard |
| Aggregate B16 | ~9 tok/s | measured (B16 gate) | p256×B16, 1088 frames |
| Prefill throughput | ~21.7 tok/s | measured (spark8 events) | p256×B16, PFR=8 |
| Public incumbent (no-spec FP8) | 7.88 tok/s | community | **BEATEN** (8.03) |
| Public incumbent (DSpark k14) | 58.5 tok/s | community | target |
| Public incumbent (peak) | 75 tok/s | community | stretch |

**Known bottleneck** (2026-08-27, event-timed on spark8): prefill is
weight-streaming bound at small row counts. At PFR=8, each 8-row chunk
streams the full 28.5 GB weight set (~114ms floor at 250 GB/s). At PFR=32,
the FFN GEMM row-scaling defect made larger chunks *slower* (464ms vs 100ms
per layer-frame). Fix committed: WS kernel row ceiling lifted, frame graphs
default-on, per-row fallback replaced with hard-fail. **Measurement in
progress** on spark1 (PFR 8/32/64 A/B).

## Qwen 3.8 27B Single-Node SOTA Ladder

Current → Target:
- No-spec: 8.03 → **8.7** (gap: 8%) — within reach
- Spec: 24.5 → **64** (gap: 2.6×) — requires prefill fix + k-tuning
- Prefill: ~22 → **target TBD** after FFN fix lands

## GLM 5.2 (FP8, TP8) — Deprecated for serving, kernel donor

| Metric | Value | Status |
|---|---|---|
| B1 single-stream | 6.91 tok/s | measured |
| B8 aggregate | 43.46 tok/s | measured |
| B16 aggregate | 75.55 tok/s | measured |

## Qwen 3.8 Max (~2.4T FP8)

| Metric | Value | Status |
|---|---|---|
| B1 per-request | 1.29 tok/s | measured anchors, TP4xPP4 |

## K3 (MXFP4)

| Metric | Value | Status |
|---|---|---|
| B1 single-stage | 18.0 tok/s | measured, single-stage |
| Roofline | 20.6 tok/s | calculated |

## New Models (in onboarding)

| Model | Stage | Lane |
|---|---|---|
| Qwen 3.8 Flash (qwen4_exp) | M1–M4 complete, M5 blocked on kernel ports | pilot agent |
| GLM 5.2 → 5.3 (full) | M1 freeze running | glm lane agent |
| GLM 5.3 Flash | Separate architecture, future lane | not started |

## Measurement Infrastructure Notes

- All 27B measurements: spark2 (prod) or spark8 (bench), GB10 sm_121a, TP1
- Token stream hashes pinned for bit-exact regression detection
- Event-timed per-phase GPU profiling (CUDA events, no profiler dependency)
- `make test` gate restored and passing (except one continuation-lease test)
- Frame graphs measured at 78% replay coverage, zero failures

## Honest Gaps

1. DSV4 Flash 40 tok/s is **not on main** — the lean commit `3d962820` is
   branch-level. Requalification in progress.
2. The 27B FFN row-scaling fix is **committed but not yet measured** — the
   A/B is running on spark1.
3. No frozen qualification receipts for the 27B family (numbers live in
   docs/commit messages, unlike dsv4's receipt discipline).
4. GLM 5.2 numbers are from the pre-audit-fix code; the current code has
   correctness fixes that may change throughput.
5. qwen38_max has no validation harness (audit finding).

## Qwen 27B TP4×PP4 Projected Performance (not yet measured)

The 24.5 tok/s current production is TP1 B=1 with DFlash2. The projected
numbers for TP4×PP4 deployment on 16 sparks:

| Configuration | Single-stream spec | Aggregate spec (continuous batching) |
|---|---|---|
| TP1 B=1 (current) | 50 tok/s | 50 tok/s |
| TP4×PP4 B=1 | 199 tok/s | 199 tok/s |
| TP4×PP4 B≥4 (pipeline full) | 794 tok/s | 794 tok/s |
| TP4×PP4 B≥16 (batched) | — | ~8,900 tok/s |
| TP4×PP4 B≥64 (theoretical max) | — | ~50,000 tok/s (weight-stream bound) |

The 150 "ceiling" is B=1 single-stream latency on TP4×PP4 (28.5 ms through
4 pipeline stages). The aggregate ceiling is set by weight streaming
(28.5 GB per pass through 4,000 GB/s total HBM = 7.1 ms) — NOT by compute.
Continuous batching amortizes the weight stream across all pending
requests. The same principle as MoE expert grouping, applied to dense:
memory-bound → compute-bound by increasing tokens per weight load.

**These are projections, not measurements.** The TP4×PP4 deployment
requires: the pack path fix on spark4, TP4 rank packs (which the DFlash2
spec path needs), the pipeline runtime on 4 nodes, and the collective
standalone bypass (or a 4-node TP collective). None of these are
currently deployed for the 27B (which runs TP1 on spark2).
