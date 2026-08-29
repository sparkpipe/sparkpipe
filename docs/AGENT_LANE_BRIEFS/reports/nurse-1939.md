# Nurse sweep 5 — 2026-08-29 19:36-19:41 UTC (lane/nurse)

Sweep 5 of 9. Read-only. Nothing killed/cancelled by nurse.

## ESCALATE

**1. r2prefill-exact32k-cell-t2 = ZOMBIE TASK (queue says running; nodes are
empty).** Since 19:27:22Z (log frozen 10+ min at check time).
Evidence (one command class, run 19:37-19:39Z):
- `pgrep -af r2prefill` on spark4/spark5/spark6/spark7 => **no task process on
  any node** (spark4 matches are another lane's unrelated staging: md5sum +
  scp of dsv4_flash_stage.spstage).
- spark4:`/tmp/r2prefill-cell.log` = 2549 B unchanged since 19:27:22Z; last
  lines are a terminal python `BrokenPipeError: [Errno 32] Broken pipe` in
  `os.write`.
- spark4:`/tmp/sparkqueue-r2prefill-exact32k-cell-t2.exit` DOES NOT EXIST —
  the wrapper's completion write was lost with the session.
- `python3 tools/spark_queue.py list` still shows it `running`,
  hold=spark4,spark5,spark6,spark7; leases TTL-expire **19:39:34Z**.
Likely cause: the dispatcher's ssh session to spark4 broke at ~19:27:22Z
(BrokenPipe = connection drop); the remote pipeline died with it and the
orphaned queue entry stayed `running` (the pid=None zombie class).
Coordinator fix (nurse never cancels): cancel the zombie entry; the
r2-prefill lane re-queues (its staging feeds are alive — the scp/md5sum
activity is theirs). Cost of delay: 4 nodes idle-held + g5dsa-wave2 (needs
all 16) blocked behind a corpse until 19:39:34Z.

**2. qwen27b-inference-smoke = probable STUCK LAUNCH (pid=None class).**
Dispatched 19:36Z; dispatcher line literally `dispatched qwen27b-inference-smoke
nodes=spark8 pid=None`. spark8:`/tmp/sparkqueue-qwen27b-inference-smoke.log`
= **0 bytes** at 19:38Z; no matching process on spark8; GPU 0%/[N/A].
Predecessor `qwen27b-known-good-smoke` reaped exit=2 seconds earlier, so the
retry was created into a known-failing context. Within the 5-min grace at
check time (1-2 min old) — if the log is still 0 B at sweep 6, this is a
confirmed stuck launch for cancel+retry. Likely cause: launch-time ssh/exec
failure, pid never captured.

## 1. Queue state (controller local)

- Dispatcher UP (continuous passes; held/barrier logic visible and correct:
  "held: weightd-vmm-verify-gpu-t2 — priority barrier for g5dsa-wave2 (p0)").
- running: r2prefill-t2 [zombie, see ESCALATE], qwen27b-inference-smoke
  [spark8, see ESCALATE].
- queued: g5dsa-wave2 [all 16, blocked — correct while r2prefill held 4;
  unblocks at 19:39:34Z lease expiry], weightd-vmm-verify-gpu-t2 [spark0,
  held by priority barrier], + 3 note tasks.
- No new results since 19:25Z (w1loader-bench/gate + wave1 cancellations
  already reconciled in sweep 4).

## 2. Remote pipelines (spark5)

- bf16 place **PASS1-DONE rc=0 (16/16 placed)** at ~04:26 KST; **PASS2
  RUNNING** (4 place processes; expect full 16-rank re-push due to the
  86,528 B spec mismatch — the pass that should land byte-correct packs on
  all 16 nodes; ETA ~20:25Z). validate rc=0 still VOID (unchanged).
- 25/25 staging labels published+dest PRESENT. No fetch/verify processes.
  Nothing wedged on spark5.

## 3. Local lanes

No lane quiet>60m with dirty => no flags. Working: debts (dirty 8, quiet 1m),
cellrunner (dirty 1, quiet 0m), p1d2 (dirty 8 — grew, quiet 8m), toksidecar
(dirty 13, quiet 7m — files still being touched, uncommitted 91m), glm5dsa
(dirty 13, quiet 28m). Dead-lane set unchanged.

## 4. Reservations

spark4-7 task:r2prefill-exact32k-cell-t2 TTL 15 -> **expired 19:39:34Z**
(during this sweep; zombie entry may linger — see ESCALATE). All other nodes
lease-free. No stale holders.

## Actions

Read-only + this report. Dispatcher healthy (6b re-arm not needed).
