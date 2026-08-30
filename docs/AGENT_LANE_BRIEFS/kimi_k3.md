# Kimi K3 lane (lane/kimik3-dev) — model dev brief

Owner: Kimi K3 model dev. Branch `lane/kimik3-dev` off origin/main
(2026-08-30). This brief supersedes nothing — it continues the
lane/k3-finish work (see K3_FLEET_LAUNCH_STATE.md and
reports/k3-finish-*).

## Current state (inherited from lane/k3-finish)

- ALL FOUR STAGE PACKS DEPLOYED (16 ranks, TP4xPP4, receipts in
  K3_FLEET_LAUNCH_STATE.md). Stage-2 was rebuilt as 47_23 after the
  48_23 defect. Binaries/configs uniform from lane/k3-finish tip.
- Queue note `k3finish-launch-ready-note`: 16-rank wave launch-ready,
  holding for the next exclusive window (K3 needs ALL 16 nodes
  exclusively — ~96-97 GiB/rank of the 110 GiB ceiling).
- Onboarding constraint (coordinator, 2026-08-30): stagepacks + task
  queuing still settling; no GPU launches until the coordinator
  declares the queue live for lanes.
- Dashboard 2026-08-30 ~05:07Z: 0/16 nodes busy, no residentd live,
  glm5 lease apparently lapsed — window may be near, but the queue's
  exclusive-window protocol (reserve all 16 → check → launch) goes
  through the coordinator-owned cadence.

## Plan (dependency order)

1. **M1 — merge parity.** Confirm lane/k3-finish tip == main or PR its
   delta; the launch state doc says main b993b7d + lane commits. Until
   the runtime trees' branch tip is merged, keep staging recipes
   referencing lane/k3-finish.
2. **M2 — first fleet number (window-gated).** When the coordinator
   grants the exclusive window: `spark_queue.py reserve` all 16 for
   lane-kimik3, `k3_fleet_wave.sh check` (must be green incl. memory +
   ceph MDS check on spark2/3), `launch`, `status` to ready-line, then
   B1 decode tok/s through the fleet; verify via dashboard +
   node-local nvidia-smi.
3. **M3 — quality gate.** Start the staged /v1 front on spark0 (port
   8433, recipe in K3_FLEET_LAUNCH_STATE.md §blockers-3); K3 fixtures
   are pre-tokenized (qualification/ds4_eval/quality-fixtures-kimi-k3.json,
   92 cases) — run COMPSEC-17 subset through the live endpoint.
4. **M4 — perf hill climb** (docs/K3_PERF.md order, remaining items):
   (a) reduce-scatter+all-gather for the slot-encoded full-width AR
   (halves per-rank wire bytes); (b) move the f32 head exchange off the
   host tier; (c) TILE_K=32 INTERLEAVED_B GEMM variant to unlock TP16
   packs; (d) per-submission payload width. Baseline: 18 tok/s B1
   decode (55.5 ms/stage), roofline 20.6 tok/s.

## Rules I follow

Queue-only GPU work at priority 5; staged scripts; exactness before
timing (mismatch = RED stop); PRs with offline-gates exit-code green +
report in docs/AGENT_LANE_BRIEFS/reports/ as kimi-k3-*.md.
