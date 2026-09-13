# Qwen38 spark3 prefill: submission-row cap dominates wall time (2026-08-25)

Scope: single-node `qwen38.fp8.tp1` runtime on spark3 (GB10), new binaries
deployed 2026-08-25 11:11 (`model_serving_adapter.so` acceptance fix), spec
ON (`DRAFT_COUNT=8`, dflash2), cell `ctx512_b1` (512-token prompt).
All walls are warm-run client-measured, GPU-idle-gated.

## TL;DR

The bench cells pin `max_prefill_rows_per_submission: 1`, which turns the
512-token prompt into 512 one-row full-weight frames (~131 ms each, ~65.5 s).
Raising it to 64 cuts prefill to ~4.9 s and the full E2E request to 12.2 s.
No engine or kernel change is required; outputs stay bit-identical (same
first token 1596 across all configs; incident doc verified O512 hash at =8).

## Measured scaling (prefill-only, `output_token_budget=1`)

| rows/submission | prefill wall | rate |
|---|---|---|
| 1 (cell files today) | 65.6 s | 7.8 tok/s |
| 8 (engine default) | 10.7 s | 47.9 tok/s |
| 32 | 6.2 s | 82.9 tok/s |
| 64 | 4.9 s | 104.5 tok/s |
| 128 | 4.9 s | saturated |

Frame-count signature at rows=1: exactly 512 prefill frames in
`SPARK_QWEN36_PROFILE=1` output; multi-row chunks stream the weight set once
per chunk (~615 ms per 64-row chunk) instead of once per row.

## End-to-end paired runs (spec ON, 128 generated tokens)

| config | warm wall | effective rate | first_draft_miss |
|---|---|---|---|
| rows=1 | 74.6 s | 5.9 tok/s | - |
| rows=64 | **12.2 s** | **10.5 tok/s** | 0 |

For reference, the pre-fix morning headline (old binaries, rows=1, spec
acceptance broken at mean_accepted=0.175) was 82.3 s = 1.55 tok/s.
The combined binaries + config fix is worth ~6.7x on this benchmark.

## Decode context (why spec now helps)

Old binaries: drafter degenerated (`drafts=[16,220,16,220]`, 90% full-miss
rounds, 3 frames/round ~ 690 ms for 3.2 credits = 4.7 tok/s) - plain no-spec
was FASTER (8+ tok/s). New binaries accept 7/8 steadily at ~370 ms/round;
no-spec control measured 8.8 tok/s vs 14.2 tok/s decode-only with spec ON.
Speculation is now net-positive; adaptive gating is not needed at current
acceptance.


## Speculation gating (rolling acceptance threshold) - 2026-08-25 PM

Implemented an env-gated rolling-acceptance speculation gate in the serving
adapter (sparkpipe-consolidated qwen36 generation; source backup at
`spark_qwen36_serving_adapter.c.pre_gate`, gated build deployed as
`lib/model_serving_adapter_gated.so` + `config/model_resident_gated.json`).
Knobs: `SPARK_QWEN36_SERVING_SPEC_GATE=1`, `_WINDOW` (4),
`_MIN_ACCEPT` (3), `_PROBE_ROUNDS` (12). Gate arm/trip verified live:
`gate armed window=4 min_accept=3 probe=12` then `gate off total=7 window=4`
fired exactly on the observed low-acceptance ramp.

**Baseline quantification (stock adapter, fresh daemon + fresh sequence,
ctx512/budget128/rows64):** 12.09 s wall, 26 rounds, acceptance dist
{0:5, 1:4, 2:1, 3:1, 4:2, 5:2, 6:1, 7:10} - 11 of 26 rounds (42%) accept
<4 of 7. The wasted-ramp regime the gate targets is real on every
daemon-cold sequence.

**Blocker found:** the first PLAIN decode frame after a mid-sequence
spec->plain transition fails (`frame_status status=1` ->
`adapter_submit status=invalid_argument`). Stock operation never exercises
this transition (the capacity-exceeded fallback assumes coverage was NOT
extended), so the module-level state left by prior verify/replay rounds
(draft-chain KV coverage / GDN snapshot slots) lacks a teardown path.
Completing the gate needs a chain-teardown helper (lane release + re-cover,
mirroring completion semantics) before plain frames can follow spec rounds.
Not fixable safely in a timeboxed pass; handed back to engine owners with
the gated build + patch kept as reference (`/tmp/dsh_gate_patch.py`,
`/tmp/gated_adapter.c` on spark3).

## Recommendations

1. Ship/generate bench and gateway cell files with
   `max_prefill_rows_per_submission: 64` (or drop the key to inherit the
   engine default 8). Keep =1 only where the B16 lane-interleaving workaround
   applies (see QWEN38_B16_INCIDENT_FINDINGS.md).
2. Fixed variant persisted on spark3:
   `/tmp/sp2_cells/ctx512_b1_rows64.sparkbatch.json` (/tmp is volatile -
   land cell generators in-repo when convenient).
3. Remaining prefill headroom vs glm-dev's ~190 tok/s chunked prefill:
   ~1.8x kernel-level (causal attention reads, KV-write path) - separate,
   non-blocking optimization.
4. Fleet hygiene note: spark3 currently hosts an aggressive watchdog that
   restarts residentd every few minutes and rejects interleaved clients
   (`status=4`, single-client `last_submission_id`). Benchmark windows must
   be exclusive; see COORDINATION.md measurement rules.
