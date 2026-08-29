# Nurse sweep 9 (FINAL) — 2026-08-29 20:18-20:21 UTC (lane/nurse)

Sweep 9 of 9 — final report before lane COMPLETE. Read-only throughout the
shift; two ESCALATEs raised (both acted on by coordinator); zero kills,
zero cancels, zero source edits, one documented-re-arm class used: none.

## ESCALATE (carried, strengthened)

**qwen27b-inference-smoke4: log frozen 13+ min with a terminal error and
still holding spark5** (`RESIDENTD-NOT-READY` / `model_residentd
deployment=io_error`, last write 20:06:14Z; queue: running, hold=spark5,
TTL expires 20:26:14Z). It now ALSO gates a newly queued
`q27b-build-1` (after=[smoke4], lane/qwen27b-serve f5a8cbc) — cancelling
smoke4 unblocks both wave3 (p0) and the build. Owner action unchanged:
cancel + route the io_error (bad/missing --deployment path in the smoke's
residentd invocation) to the smoke's owner. 5th failed attempt of the
family; the smoke script itself needs the fix, not another retry.

## Reconciliation (6c) — one line each

- `r3flash-glm52-exact-cell-t3` **exit=1 "dispatch reaped" 20:09:42Z** —
  real task failure (not a cancel). Owner: r3-flash lane
  (reports/r3-flashdecode-2026-08-29.md). Owner must diagnose + re-queue;
  not done until then.
- Earlier, already routed: wave1/wave2 cancels (glm5-dsa; wave3 now queued
  with hand-verified artifacts), w1loader-bench/gate cancels (coordinator),
  weightd-vmm-verify-gpu-t2 pid-gone (t3 already queued by owner),
  smoke2/3/4 (see ESCALATE).

## HANDOFF TRUTH — glm53full bf16 placement is INCOMPLETE (do not trust rc=0)

Fleet byte spot-check at 20:19Z (the check the place chain's wrong
`--bytes` spec cannot give):

- spark0 rank0: **98,019,454,976 B = SIZE-OK** (pass2 re-push landed
  byte-perfect).
- spark1 rank1: 90.6 GB (92.5%) = mid-re-push; spark2 rank2: 68.2 GB (69.5%)
  = pass1 copy was itself incomplete — pass1's "placed" lines did NOT all
  mean complete files. Pass2 (running, sequential) re-pushes every rank from
  the complete source, so the end state self-heals — at ~15-25 min/rank it
  finishes ~23:00-24:30Z, well after this shift.
- Successor verification command (run per node after PLACE-PASS2-DONE):
  `ssh <node> stat -c%s sparkdata/glm53full.bf16.tp16/packs/<rank>.glm52sp`
  — all 16 must equal **98019454976** (NOT the chain's 98019368448 spec).
  Then re-run the 16 validators: the log's `VALIDATE-DONE rc=0` (03:48Z)
  ran MID-WRITE and is VOID; no re-validation is scheduled by the chain.

## Remote pipelines (spark5)

- bf16 build COMPLETE 19:02:43Z (16x 98019454976 B + receipts); place pass2
  in flight (see handoff); fetch/publish/refreeze all DONE with their DONE
  lines verified earlier in the shift. 25/25 staging labels phase=published
  + destination PRESENT all shift — zero zombies, zero re-arms owed.
- No fetch/verify processes alive (correct). Nothing wedged on spark5.

## Queue / GPU

- Dispatcher UP all shift (one supervisor loop; passes every ~5 s; correct
  barrier + note-kind handling throughout).
- running: smoke4 only (see ESCALATE); GPU 0% on its node = consistent with
  its terminal error, no ghost work beyond it. wave3 + weightd-vmm-t3 +
  q27b-build-1 + 3 notes queued. No idle-node alarm criteria met (smoke4
  holds a node; remaining idle nodes have no executable task except those
  gated/blocked correctly).

## Local lanes

- glm5dsa: 60m+dirty flag from sweep 8 **RESOLVED** — committed+pushed 2 min
  before this sweep (dirty 13->3, quiet 2m); wave3 attempt 3 in the queue.
- toksidecar: still uncommitted-only (dirty 14, quiet 6m, last commit 131m)
  but files touched minutes ago — alive; carries the largest uncommitted
  delta on the fleet (watch item for the next nurse).
- k3finish (dirty 2, pushed), debts (new report last sweep), p1d2
  (new report sweep 6), cellrunner/cfgaudit/w3weightd clean. Dead-lane set
  unchanged all shift: 3 informational not-in-main cases
  (drywave1 dirty=8 detached, dsv4flash clean, qwen38max-shard 1 unpushed
  commit); everything else verified merged.

## Reservations

Only spark5 (smoke4, expires 20:26:14Z). No stale holders, no expired TTLs
all shift; the 16-node attractor manual reservation released itself at
19:28Z as observed in sweep 4.

## Shift summary (9/9 sweeps, 18:52-20:21Z)

Reports: nurse-1857/1907/1917/1929/1939/1949/2003/2011/2019.md, all pushed
to lane/nurse. Key catches: (1) bf16 validate+place raced the pack build —
VOID rc=0 + truncated early placements flagged before anyone trusted them,
with the exact fleet byte-truth check to finish verification; (2) place
chain's --bytes spec off by 86,528 B (fp8's true-size run proved the
pattern) — "already placed" proof impossible, pass2 forced full re-push;
(3) r2prefill-t2 zombie (ESCALATED, coordinator cancelled); (4) qwen smoke
family: pid=None stuck launch -> io_error root cause visible in log
(ESCALATED, pending owner fix); (5) wave1/wave2 bring-up cancel churn
pattern surfaced; (6) charter extension executed every sweep since 1929Z:
queue, dispatcher, GPU, reconciliation, reservations.
