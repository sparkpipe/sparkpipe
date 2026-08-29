# Nurse sweep 6 — 2026-08-29 19:47-19:52 UTC (lane/nurse)

Sweep 6 of 9. Read-only; nothing killed/cancelled by nurse.

## WATCH-LOUD: spark8 ssh DEGRADED (hangs at banner) while it hosts the only running task

Two consecutive ssh attempts to spark8 (19:48Z ConnectTimeout=8 and
19:50-19:52Z ConnectTimeout=20) stalled at banner exchange and had to be
killed. spark8 hosts `qwen27b-inference-smoke2` (RUNNING since 19:47:48Z) —
if its ssh stays degraded, the dispatcher's 10s-timeout reap checks cannot
verify the task (same orphan path that produced the r2prefill zombie).
Next sweep decides: smoke2 log growing + ssh recovered = fine; ssh still
hanging = escalate as zombie-in-making (coordinator owns cancel/retry).
Note the qwen smoke family already burned 3 attempts (known-good exit=2,
inference-smoke pid=None, now smoke2) — spark8 state itself may be the cause.

## ESCALATIONS: both sweep-5 items RESOLVED by coordinator (verified)

- r2prefill-exact32k-cell-t2 zombie: cancelled -> "reaped exit=0 (nodes
  released)" 19:38:11Z; entry gone from queue; its 4 leases freed.
- qwen27b-inference-smoke (pid=None, 0 B log): replaced by smoke2 (queued
  behind wave2, then dispatched 19:47:48Z — see WATCH-LOUD).

## Queue state

- RUNNING: qwen27b-inference-smoke2 [spark8] only. TTL 25 -> expires
  20:12:48Z; 1.5x overrun mark ~20:25Z.
- **g5dsa-wave2: coordinator-CANCELLED 19:47:45Z** after ~7 min (exit=-1;
  spark0 log ends "ready: 0/16 -> Terminated / WAVE-FAIL" — bring-up never
  reached rank readiness within the cancel window). Pattern note for the
  coordinator: that is two consecutive wave cancels on bring-up
  (wave1 19:23:58Z, wave2 19:47:45Z); the bring-up window may just be
  shorter than the cancel patience, or the adapter-abi fix isn't in yet.
  lane-glm5dsa owns the next re-queue (their worktree is active, dirty=13).
- **weightd-vmm-verify-gpu-t2: died "dispatch: pid gone without exit file"**
  (dispatcher-reaped 19:48:20Z, exit=-1). Reconciliation (6c): owner = the
  weightd lane (w2-weightd / w2b-weightd reports); task needs re-queue by
  its owner — coordinator routes.
- Queued: 3 note-kind tasks (r2prefill-spark5-request, r2prefill-probe-fix-
  note, k3finish-launch-ready-note) — verified `kind=note`, no cmd:
  deliberately not dispatched (messages, not executable tasks). Dispatcher
  NOT buggy; idle-node alarm NOT triggered (smoke2 running; notes are not
  executable tasks).
- Dispatcher UP (passes every ~5 s; "nothing dispatched" is correct state).

## Remote pipelines (spark5)

- bf16 place **PASS2 in flight**: rank0 rsync -> spark0 alive at 12.6 min,
  68/98 GB landed on spark0 (~90 MB/s, --inplace), spark0 has 1.4 T free and
  sparkf 2.8 T — no disk-full risk. Pace ~13 min/rank under load => pass2
  ETA ~22:00-22:45Z (will run past my sweep 9; I'll verify landed ranks by
  size and report the remainder). PASS1-DONE rc=0 19:34:31Z. The spec
  mismatch (86,528 B) guarantees every rank re-pushes — the "already placed"
  proof will only appear if someone re-runs with the TRUE size
  98019454976. validate rc=0 still VOID (no re-validation scheduled).
- 25/25 staging labels published+dest PRESENT; no fetch/verify processes;
  nothing wedged on spark5.

## Local lanes

No lane quiet>60m with dirty => no flags. p1d2 pushed a fresh report
(p1d2-steploop-2026-08-29.md). Active: p1d2 (dirty 12, quiet 0m), debts
(dirty 9, quiet 0m), glm5dsa (dirty 13, quiet 37m — working on the wave
fix), toksidecar (dirty 13, quiet 2m), k3finish (dirty 1, quiet 5m),
cellrunner (dirty 1, quiet 9m). Dead-lane set unchanged.

## Reservations

Only spark8 held (task:qwen27b-inference-smoke2, 19:47:48Z, TTL 25). All
other nodes lease-free. No stale holders, no expired TTLs.

## Actions

Read-only + this report. No re-arm owed (dispatcher healthy; spark8 issue
is diagnosis-only pending next sweep's evidence).
