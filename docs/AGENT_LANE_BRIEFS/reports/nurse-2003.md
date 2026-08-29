# Nurse sweep 7 — 2026-08-29 20:02-20:06 UTC (lane/nurse)

Sweep 7 of 9. Read-only; no re-arm owed; nothing killed/cancelled.

## WATCH (decision at sweep 8)

- **qwen smoke churn, attempt 4**: smoke2 CANCELLED 19:52:46Z (during the
  spark8 ssh degradation), smoke3 CANCELLED 20:01:12Z, **smoke4 dispatched
  spark5 20:01:14Z** (TTL 25 -> expires 20:26:14Z). At check (20:03Z, 2 min
  in): `/tmp/sparkqueue-qwen27b-inference-smoke4.log` = 0 B, GPU 0% [N/A] —
  within bring-up window. If the log is still 0 B at sweep 8 (~12 min in),
  that is a confirmed stuck launch -> ESCALATE for cancel. Four attempts
  without a success line suggests the smoke's own bring-up needs looking at
  (owner: the qwen/weightd side that keeps re-queueing it).
- **glm5-dsa lane approaching flag**: worktree quiet 53m with dirty=13,
  last commit 103m ago, nothing pushed since its wave2 retry got cancelled
  (19:47:45Z). If quiet crosses 60m with dirty unchanged => mid-edit-death
  flag per charter. No wave3 in the queue yet; 15/16 nodes idle (only
  spark5 held by smoke4). Not an idle-node ALARM (no executable task
  queued), but fleet utilization is effectively zero while the coordinator
  presumably wants wave3 or k3 16-rank work — coordinator's routing.

## Resolved / verified this sweep

- spark8 ssh RECOVERED (SSH-OK at 20:02Z after two hang-kills at 19:48-
  19:52Z). smoke2's log no longer exists (entry cancelled) — the WATCH-LOUD
  from sweep 6 closes with the cancellation, cause unproven (host stress
  during wave2 bring-up is the likely contributor).
- Dispatcher UP, passes clean. Notes (kind=note x3) correctly skipped.

## Remote pipelines (spark5)

- bf16 place PASS2: rank0 -> spark0 re-push COMPLETED ("rank 0 -> spark0:
  placed" ~20:05Z, ~25 min for the first rank). 15 ranks to go at that pace
  => pass2 likely runs past ~23:30Z; PASS1 data for ranks 5-15 was already
  byte-correct, ranks 0-4 are the truncated ones being repaired in order.
  My sweep 9 will spot-check landed ranks by exact size (98019454976).
- validate rc=0 VOID flag stands (no re-validation scheduled).
- 25/25 staging labels published+dest PRESENT; no fetch/verify processes;
  nothing wedged.

## Local lanes

No flags yet. Active: debts (dirty 12, quiet 5m, NEW report
debts-2026-08-29.md pushed), k3finish (dirty 2, quiet 9m), toksidecar
(dirty 14, quiet 7m — still touching files, uncommitted 115m), glm5dsa
(see WATCH), cellrunner/cfgaudit/w3weightd clean. Dead-lane set unchanged.

## Reservations

Only spark5 (task:qwen27b-inference-smoke4, 20:01:14Z, TTL 25 -> 20:26:14Z).
No stale holders.

## Actions

Read-only + this report.
