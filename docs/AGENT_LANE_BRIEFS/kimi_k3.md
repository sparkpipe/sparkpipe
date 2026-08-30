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
   config load, keepalive tile_k. Binaries BUILT on sparke (k3tp16-src,
   PATH=/usr/local/cuda/bin; adapter carries the maxloc symbols); TP16
   adapter configs generated (/home/sparke/k3tp16-configs). Deploy
   preflight 2026-08-30 ~15:2x: pack dirs made on 15/16 nodes — spark0
   ssh banner-timeout (transient? retry), sparke has 92G free < the 98G
   rank-14 pack (queue note k3tp16-sparke-space-note filed; deploy will
   run ranks 0-13,15 first). NEXT: verify full pack → k3_tp16_deploy.sh
   (slice-verify-ship-delete) → k3_stage_runtime.sh sparke
   /home/sparke/k3tp16-src/build 16 /home/sparke/k3tp16-configs →
   exclusive window wave → first TP16 number vs the 18 tok/s TP4xPP4
   baseline (roofline 20.2 @ 49.5 ms).
1. **Merge parity.** lane/k3-finish runtime delta → main (pre-existing M1).
2. **TP4 fleet first number + COMPSEC-17** (window-gated, as before).
3. **Deployed-pack CPU audit** (QUEUED): tools/k3_deployed_audit.py staged;
   16 deploy digests collected (stages 0-2 from deploy logs, stage-3 from
   reports/k3-packs-2026-08-27.md); re-hash + manifest-config check pending.
4. **Perf hill climb** remainder (docs/K3_PERF.md): reduce-scatter+all-gather
   for the slot AR; head exchange DONE (pending GPU exactness); per-submission
   width DONE in common code.

## Bug-ledger items landed this lane (2026-08-30, operator-assigned)

- C-CANCEL residual gap FIXED: a client disconnecting while the
  request was still QUEUED got submitted anyway (full budget burned,
  no cancel). Orphaned now implies done (never submitted), and a
  worker/connector mutex handshake cancels exactly once across the
  snapshot→submit window (node/model_api.c; api_orphan_cancel_after_
  submit). Complexity-ceiling FLAT (7.87 held: helper below mean).
- K3 runner memory-contract ratchet repaired (the head-exchange sites
  went to sizeof()-based sizing; parked list pruned 22→18 memcpy) —
  this was failing at lane tip, unobserved because test_code_size's
  venv caveat aborted the suite before it.
- C-CANCEL main wiring (40ca88c) and the O(n²) parse one-pass
  (f175099) were already on main — ledger rows corrected.
- P1 module port: DESIGN FILED (reports/kimi-k3-p1-port-plan-2026-08-
  30.md) — k3 declares no ASYNC_COMPLETION so the landed async drain
  still serializes us; the staged port (adapter slots → LaunchHostFunc
  completion → chain depth) executes at the window per the program's
  closeout gate.

## Rules I follow

Queue-only GPU work at priority 5; staged scripts; exactness before
timing (mismatch = RED stop); PRs with offline-gates exit-code green +
report in docs/AGENT_LANE_BRIEFS/reports/ as kimi-k3-*.md.
