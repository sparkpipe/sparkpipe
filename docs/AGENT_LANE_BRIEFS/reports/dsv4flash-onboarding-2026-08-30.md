# DSV4 Flash lane onboarding + plan — 2026-08-30

Lane: `lane/dsv4flash-dev` (model dev: dsv4-flash). Branched from
`origin/main` @ 879fc5a. This is the standing plan until stagepacks and
the spark task queue are declared live by the coordinator; no GPU work
is dispatched from this lane before that.

## Inherited state (from ledger + dsv4-40-bisect report, 2026-08-28)

- **The 40 tok/s cell is fixed and on main+PR731.** O128 B1 TP4 cell:
  40.19–40.46 tok/s, exact hash `211462f2...`. The regression was
  unconditional DSpark machinery on the no-spec path (tap kernel +
  syncs at layers 40–42, head-max acceptance sync, staging fprintf);
  the fix is the `SPARK_DSV4_DSPARK=1` gate. Lean source `ef8fa302`
  is in tree.
- **Exact-32K cell tops ~29 tok/s** (lean 29.37). The old
  40.4-at-32K attribution was wrong — 32K costs ~10 tok/s in attention
  scaling. Prefill is row-serial and position-scaled: one 32K run
  costs 70–90 min wall.
- **Packs**: v3 (156G, 1328 tensors) and v4 w/ DSpark MTP (167G, 1409
  tensors) full packs verified; TP4 rank shards byte-equal to
  lean-era. But per the fleet roadmap the on-node packs may be stale
  (Aug-11 era vs current adapter) — pack freshness is a coordinator
  item, not mine to rebuild.
- Contract truth: `model_contracts/dsv4_flash_authoritative.json`;
  source weights `/mnt/model-warm/deepseek-v4-flash-0731` (156G).

## Plan (ordered, perf hill climb)

- **P0 — wait-state discipline.** Stagepack placement and the spark
  queue are coordinator-run and still in progress. Until the guide's
  assumptions hold ("stagepacks are there; queue is the only GPU
  path"), this lane does CPU-only work and files blockers, never
  hand-launches fleet/daemons.
- **M1 — verify the lane baseline on the live queue.** First queued
  task once the queue is live: reproduce the O128 B1 TP4 cell
  (40.x tok/s, exact hash 211462f2) from current main to re-pin the
  ratchet. Priority 5 (routine), `--ttl-min` sized for pack load +
  127 decode intervals. Telemetry receipt (dashboard + nvidia-smi)
  in the report.
- **M2 — DSpark spec path, gated ON.** The gate landed disabled
  (`SPARK_DSV4_DSPARK=1`). The hill: arm it, verify exactness FIRST
  (any mismatch = RED stop), then measure acceptance/tok-s vs the
  40.x baseline at O128. v4 packs carry the MTP weights — if the
  deployed packs are v3, flag the v4 pack placement to the coordinator.
- **M3 — exact-32K attention scaling.** 29 tok/s is the honest number;
  the cost is position-scaled attention. Candidate work: the R2c bulk
  path's decode-side scaling (split-K/flash-decode family already has
  precedent in glm lanes), targeted at the 32K decode intervals. Each
  claim via queued cell with kill-switch discipline; timing only after
  bit-exactness.
- **M4 — first-cell quality gate (COMPSEC-17)** on whatever cell is
  declared "usable" (per GOALS near-term #5), before any broader
  claims.

## Standing hygiene

- Every perf claim: queued cell, exactness before timing, receipts in
  `docs/AGENT_LANE_BRIEFS/reports/dsv4flash-*.md`, numbers with
  context/batch/topology/precision.
- 5-min wakeup timer (this lane): check pending PRs, driver progress,
  realign tasks against GOALS; submit queue work only when the queue
  is live.
