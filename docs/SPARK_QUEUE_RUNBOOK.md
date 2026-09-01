# Spark task queue — parallel-dev runbook (2026-09-01)

One queue, one state, one dispatcher. This is the contract every lane
follows for ANY spark-touching work; the discipline is what makes
parallel dev safe.

## The invariants (audited + tested 2026-09-01)

1. **ONE machine-global state** at `/tmp/sparkqueue` (the tool honors
   `SPARK_QUEUE_STATE` for tests). The stale per-checkout `runs/`
   copies are dead — if you find a checkout whose `tools/spark_queue.py`
   says `RUNS = os.path.join(ROOT, "runs")`, it is OLD: pull main.
   The split-brain (two queues, mutually invisible reservations,
   silent node collisions) is exactly what this prevents.
2. **ONE dispatcher**: the mac daemon (5 s loop) running
   `/Users/mac/sparkpipe/tools/spark_queue.py dispatch`. Manual
   `dispatch` calls are safe (lock-serialized) but normally
   unnecessary — submit and let the daemon serve you.
3. **Priority: lower number runs first** (0 = highest, default 5).
   Tasks older than 2 h auto-elevate to 0 (anti-starvation).
4. **Node sets are exclusive** for the task's duration; disjoint sets
   run in parallel; `--ttl-min` is the expected duration (declare it —
   the lease and the stale-reaper are sized by it).
5. **The reaper releases nodes** via the task's exit sentinel
   (`/tmp/sparkqueue-<id>.exit` on the first node) — crashed tasks free
   their nodes within seconds of the next dispatch pass. `cancel`/
   `done` on a running task TERM its process GROUP (the wrapper
   self-reports its pid to `/tmp/sparkqueue-<id>.pid`; the fleet proxy
   eats `$!` stdout — never rely on it).
6. Results persist in `/tmp/sparkqueue/results.jsonl`;
   `status --id` finds finished entries; per-task logs live at
   `/tmp/sparkqueue-<id>.log` on the task's first node.
7. `--cmd-file PATH` submits long scripts without quoting hell.
   Denylist: no reboot/shutdown/kill -9/rm -rf in submitted commands.

Regression gate: `tests/test_spark_queue.py` (hermetic, no sparks) is
in `make test` — it pins dispatch exclusivity, priority order,
dependency gates, reaper release, cancel-kills-remote, duplicate-id
refusal, and no-double-dispatch under concurrent passes.

## EXPERIMENTAL MODE (the tight loop)

Goal: maximum hypothesis-kills per hour on the sparks, zero contention
with other lanes.

0. **Weights stay resident** (weightd / a live serving set on your
   nodes) — the loop's speed IS the one-minute debug cycle. Never
   teardown-and-relaunch between experiments.
1. **Design the batch before touching the cluster**: half a dozen
   tests that answer questions, each with a crisp expected outcome
   ("if X then row 0 matches; if Y it diverges at element N"). Write
   the hypothesis in the task `--notes` — the queue log becomes the
   experiment ledger.
2. **Each test = one executable + one queue task**: dedicated test
   programs, scripts, or an instrumented driver build. Stage the
   binary (git branch or scp), then:
   `add --id lane-x-q3 --nodes spark4 --ttl-min 10 --cmd-file run.sh`
   Use `--after` to chain (build → cell → report). Prefer SINGLE-node
   tasks; claim the fleet-wide exclusive window only for waves, and
   keep it short with `--ttl-min` honest.
3. **While waiting**: audit your code (DRY, delete the instrumentation
   scaffolding you no longer need, shrink the diff), and draft the
   NEXT batch from both possible outcomes. Do NOT idle-poll the nodes.
4. **Read results, iterate** (`status --id`, task logs). One round
   should take minutes, not hours.

Conventions that keep the lane legible to others:
- id prefix = lane (`g5n-…`, `k3-…`, `r2-…`); `--by` = owner.
- `kind=note` for coordination ("holding for window X"), not chatter.
- Never ssh around the queue for GPU work; read-only journal/log peeks
  are fine.

## PR PHASE (the formal loop)

1. Clean implementation branch (no diag prints, no dead experiments).
2. **Host gates first, sparks second**: `make offline-gates` in a clean
   worktree (build-all, run-tests incl. the new gates you added,
   package manifest) — catches host-side breakage without spending
   cluster time. Re-pin ratchets (`test_code_size`,
   `test_complexity_ceiling`) with justification.
3. **Validation build through the queue**: one task (single node for
   module-side changes; fleet-wide only for wave/collective changes)
   that builds + validates + reports. For kernel-path PRs, include a
   GPU-cell equivalence test (the tier3 pattern: multi-row vs
   sequential, bit-exact or oracle-tolerance) in the validator — the
   publish gate then enforces your claim forever.
4. Positive receipt → PR → merge → update the shared checkout
   (`git pull`) so every agent runs the same tools.
5. Negative → back to experimental mode with the sharpest new
   instrument; keep the failed hypothesis in the notes.

## Fleet-vs-experiments policy

Serving holds its nodes through the queue (a long-lease task or manual
`reserve`), so experiments see them busy and pick free nodes. When a
lane needs the serving fleet (a wave), it queues an exclusive task and
waits — the queue is the only place where "who owns the sparks right
now" is decided. If serving must be interrupted, that is an operator
call, recorded as a note first.
