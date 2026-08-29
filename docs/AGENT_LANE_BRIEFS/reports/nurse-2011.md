# Nurse sweep 8 — 2026-08-29 20:10-20:13 UTC (lane/nurse)

Sweep 8 of 9. Read-only; nothing killed/cancelled by nurse.

## ESCALATE

**qwen27b-inference-smoke4 = terminal-error task still holding spark5 and
BLOCKING the p0 g5dsa-wave3.** Evidence (spark5, 20:11Z):
- `/tmp/sparkqueue-qwen27b-inference-smoke4.log` grew ONCE at 20:06:14Z to
  128 B and is frozen since (>5 min): last lines are
  `RESIDENTD-NOT-READY` / `usage: ./bin/sparkpipe_model_residentd
  --deployment PATH --rank-index N` / `model_residentd deployment=io_error`.
  That is the smoke's residentd invocation failing on its deployment arg/path
  (usage banner = bad args, or unreadable deployment) — the 5th consecutive
  failed smoke attempt (known-good exit=2, smoke2 cancelled, smoke3
  cancelled, smoke4 io_error).
- Queue still shows it `running`, hold=spark5, TTL 25 (expires 20:26:14Z);
  GPU 0% [N/A]. Nothing is computing on that lease.
- `g5dsa-wave3` (p0, all 16 nodes) sits queued with "blocked: nodes busy" —
  spark5 is the only node it lacks.
Likely cause: the smoke's launch script passes a wrong/missing
--deployment path (or points at a deployment the node can't read) — an
owner-side defect, not infra. Coordinator fix (nurse never cancels):
cancel smoke4 and route the io_error diagnosis to the smoke's owner; wave3
then dispatches immediately instead of waiting for the 20:26:14Z TTL sweep.

## Flag per charter (with exculpatory evidence)

- **lane-glm5dsa formally crosses "quiet >60min with uncommitted changes"**
  (quiet=60m, dirty=13, last commit 111m ago). NOT calling it dead: the lane
  demonstrably acted minutes ago — it submitted g5dsa-wave3 ("attempt 3,
  artifacts deployed+hand-verified") to the queue ~20:07Z. Worktree file
  tree is idle because the wave work is remote/queue-side. Downgraded to
  WATCH; will flag only if wave3 also stalls AND the worktree stays silent.

## Queue state

- running: smoke4 only (see ESCALATE). queued: g5dsa-wave3 [all 16, blocked
  on spark5], weightd-vmm-verify-gpu-t3 [spark0, priority barrier for
  wave3], 3 notes. Dispatcher UP (barrier logic correct: "held:
  weightd-vmm-verify-gpu-t3 — priority barrier for g5dsa-wave3 (p0)").
- No new results lines since smoke3's cancellation (20:01:12Z).

## Remote pipelines (spark5)

- bf16 place PASS2: rank0 -> spark0 done; rank1 re-push in flight (place.log
  unchanged since the rank0 line — pace ~15-25 min/rank => full pass2 past
  ~23:30Z). Nothing wedged (rsync verified alive sweep 6; re-check sweep 9).
- validate rc=0 VOID flag stands. 25/25 staging labels published+dest
  PRESENT. No fetch/verify processes.

## Local lanes

- glm5dsa: see flag above. toksidecar: quiet 6m, dirty 14 — alive,
  uncommitted-only (123m since last commit; files still touched).
- debts pushed debts-2026-08-29.md last sweep; k3finish dirty 2 quiet 9m.
  No other flags; dead-lane set unchanged.

## Reservations

Only spark5 (smoke4, expires 20:26:14Z). No stale holders, no expired TTLs.

## Actions

Read-only + this report. Dispatcher healthy — no 6b re-arm.
