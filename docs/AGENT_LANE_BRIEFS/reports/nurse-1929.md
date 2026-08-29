# Nurse sweep 4 — 2026-08-29 19:26-19:30 UTC (lane/nurse)

Sweep 4 of 9. First sweep under the extended charter (queue + GPU). Standing
form read from origin/main (docs/AGENT_LANE_BRIEFS/nurse.md). Read-only; no
re-arm owed; nothing killed/cancelled.

## WATCH (not ESCALATE — within bring-up window, re-checks next sweep)

- **r2prefill-exact32k-cell-t2 BrokenPipeError**: running task (spark4-7,
  started 19:24:34Z). spark4 `/tmp/r2prefill-cell.log` last lines are a
  python `BrokenPipeError: [Errno 32]` in `os.write` (mtime 19:27Z), but the
  task's `/tmp/sparkqueue-r2prefill-exact32k-cell-t2.exit` does NOT exist ->
  wrapper still up. GPU on spark4: 1 GPU visible, 0% util, mem [N/A] — 3.5
  min after dispatch = bring-up window, NOT ghost work yet. Next sweep:
  pgrep r2prefill-cell on all 4 nodes + log growth; escalate if log frozen
  >5 min with no .exit (stuck-launch class).
- **r2prefill-t2 TTL overrun risk**: leases expire **19:39:34Z** (ttl 15).
  1.5x overrun mark ~19:47Z. If the cell needs longer, the dispatcher will
  free its nodes mid-run and g5dsa-wave2 (needs all 16) dispatches onto it.
  Watching both.

## 1. Queue state (controller = LOCAL mac; runs/ is live here, not on spark5)

- Dispatcher **UP**: one supervisor loop pid 21845 (parent init, started
  19:25:02Z — coordinator-restarted after the 19:11-19:24 churn), dispatch
  passes landing in /tmp/sparkqueue-dispatcher.log (mtime fresh, passes every
  ~5s).
- `spark_queue.py list`: running = r2prefill-exact32k-cell-t2
  [spark4,5,6,7]; queued = g5dsa-wave2 [all 16, blocked: nodes busy —
  CORRECT, r2prefill holds 4 of them], plus 3 note-type tasks
  (r2prefill-spark5-request, r2prefill-probe-fix-note,
  k3finish-launch-ready-note) waiting on lease-free nodes.
- **Coordinator context correction (process-table truth)**: g5dsa-wave1 is
  NOT running — it was CANCELLED (results.jsonl: exit=-1, note "cancelled",
  finished 19:23:58Z; dispatched 19:13:38Z). g5dsa-wave2 is its queued retry
  (submitted_by lane-glm5dsa). No bring-up window is in progress for wave1;
  the only GPU task is r2prefill-t2 (see WATCH).
- Failed/cancelled reconciliation (6c), one line each:
  - `g5dsa-wave1` cancelled 19:23:58Z (coordinator action; wave2 re-queued by
    lane-glm5dsa — prerequisite chain intact).
  - `w1loader-bench` + `w1loader-gate` cancelled 19:11:05Z (both
    submitted_by coordinator; owner = w1-loader lane,
    reports/w1-loader-2026-08-29.md — coordinator routes any re-queue).
  - `r3flash-glm52-exact-cell` exit=0, `weightd-vmm-verify-gpu` exit=0,
    `launch-mechanics-probe` exit=0 — clean.
- No STUCK LAUNCH (r2prefill log grew 19:27Z), no DISPATCHER DOWN, no
  STALE LEASE (no blocked task's holder is missing), no OVERRUN yet, no
  idle-node alarm (4 nodes busy; wave2 blocking is correct).

## 2. Remote pipelines (spark5)

- bf16 place **PASS1 COMPLETE** (16/16 "placed" by 04:26 KST; log 328 B,
  last "rank 12 -> sparkc" 04:25:30 KST — all 16 lines present by sweep
  close); pass2 (the spec-mismatch-forced full re-push, sweep 2) expected to
  start ~04:26 KST + 30s sleep, ETA ~05:25 KST (20:25Z). Chain pids alive.
- validate.log unchanged (VOID rc=0; no re-validation scheduled — flag
  stands). 25/25 staging labels published + dest PRESENT. No fetch/verify
  processes; nothing wedged.

## 3. Local lanes

All healthy; no lane quiet>60m with dirty. Working: cellrunner (dirty 2,
quiet 0m, NEW report cell-runner-2026-08-29.md), p1d2 (dirty 6, 19m),
glm5dsa (dirty 13, 16m), toksidecar (dirty 13, quiet 4m — uncommitted 79m,
still touching files), debts (dirty 8, 4m). Dead-lane set unchanged.

## 4. Reservations

LIVE state changed at 19:28:10Z: the 16-node lane-glm5attractor manual
reservation (18:11Z, TTL 90 -> 19:41Z) is RELEASED (queue sweep). Current:
**only spark4-7** held by `task:r2prefill-exact32k-cell-t2` (19:24:34Z,
TTL 15 -> 19:39:34Z). No expired TTLs held; nothing stale.

## Actions

Read-only + this report. Dispatcher healthy => no re-arm needed (6b).
