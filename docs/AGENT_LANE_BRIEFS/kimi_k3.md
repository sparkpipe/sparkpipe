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

## Plan (operator reprioritized 2026-08-30: "TP16 first")

0. **TP16 wave (IN FLIGHT).** Full 93-layer PP1 expert_tile_k=32 pack
   building on spark8 (/home/sparkf out — sparkf+sparkb have degraded ceph
   warm reads 4.5/7.1 MB/s vs 226-345 elsewhere; keepalive supervises).
   Chain CPU-proven on a 1-layer probe (all 16 rank packs cross-verify
   PASS, receipts in reports/kimi-k3-tp16-2026-08-30.md). PR #757 carries
   the code: device-tier head exchange (u64 winner pack), TP16 adapter
   config load, keepalive tile_k. NEXT: verify full pack → slice 16 on
   spark8 with SLICE-DEPLOY-DELETE (16 rank packs don't fit beside the
   stage pack) into sparkdata/k3.mxfp4.tp16/packs/ → stage TP16 runtime
   (needs the PR's adapter fix built) → exclusive window wave → first
   TP16 number vs the 18 tok/s TP4xPP4 baseline (roofline 20.2 @ 49.5 ms).
1. **Merge parity.** lane/k3-finish runtime delta → main (pre-existing M1).
2. **TP4 fleet first number + COMPSEC-17** (window-gated, as before).
3. **Deployed-pack CPU audit** (QUEUED): tools/k3_deployed_audit.py staged;
   16 deploy digests collected (stages 0-2 from deploy logs, stage-3 from
   reports/k3-packs-2026-08-27.md); re-hash + manifest-config check pending.
4. **Perf hill climb** remainder (docs/K3_PERF.md): reduce-scatter+all-gather
   for the slot AR; head exchange DONE (pending GPU exactness); per-submission
   width DONE in common code.

## Rules I follow

Queue-only GPU work at priority 5; staged scripts; exactness before
timing (mismatch = RED stop); PRs with offline-gates exit-code green +
report in docs/AGENT_LANE_BRIEFS/reports/ as kimi-k3-*.md.
