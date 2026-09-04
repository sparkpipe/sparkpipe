# Spark task queue — parallel-dev runbook (2026-09-01)

## DEV QUICKSTART (the whole contract in six commands)

    # 0. once, first thing in your session:
    python3 /Users/mac/sparkpipe/tools/spark_queue.py doctor

    # 1. look before you take:
    ... list                          # who holds what right now

    # 2. submit your test (id = YOURLANE-name; ttl honest; script via file):
    ... add --id mylane-q7 --nodes spark4 --ttl-min 3 --by yourname \
            --notes "hypothesis: if X, row 0 matches" --cmd-file run.sh

    # 3. the daemon dispatches you when the nodes free up; watch:
    ... status --id mylane-q7         # includes the log tail

    # 4. take it back early if you must (kills the remote process):
    ... cancel --id mylane-q7

    # 5. coordination is notes, not chat:
    ... add --id mylane-hold-note --nodes spark5 --kind note \
            --notes "holding for window after k3 wave"

Where `...` is `python3 /Users/mac/sparkpipe/tools/spark_queue.py`.
Rules that matter: lower `--priority` runs first (0 highest, default 5);
nodes are exclusive for the task's TTL; never ssh around the queue for
GPU work; logs live at /tmp/sparkqueue-<id>.log on the task's first
node; results persist (results.jsonl) even after entries leave the list.

---

One queue, one state, one dispatcher. This is the contract every lane
follows for ANY spark-touching work; the discipline is what makes
parallel dev safe.

## The window model (operator rulings 2026-09-01, 2026-09-04)

A task's node claim is ONE WINDOW, **3 minutes by default, 15 max**
(enforced at submit). The deadline is STRICT:

- **At the deadline the reservation is gone** — nodes released
  unconditionally, and if the task's processes are still running they
  are **KILLED** (process-group SIGKILL plus a pattern sweep for
  setsid escapees). No bug, hang, or runaway benchmark prevents the
  next dev's test from starting.
- If the kill cannot even be delivered (node unreachable), the node is
  **FENCED**: dispatch skips it and probes it every pass until it
  answers; the queue itself never blocks on it. `doctor` lists fenced
  nodes.
- **Weights stay resident between windows** (weightd with lazy expert
  loading), so there is NO swap-in time — a window is pure test time.
  Technically you may load all of your model's experts, but that
  spends your own 3 minutes and LRU-pressures the other lanes' expert
  pools; invoke ≤10% of experts and everything CUDA-level validates
  and hill-climbs fine.
- **Batch many short tests per window** — one cmd that runs them in
  sequence, or `after=`-chained tasks.
- When your tasks finish, **the next dev's window starts** — no
  standing claims. To hold your turn between tests, keep your NEXT
  test queued (a queued higher-priority task blocks overlapping
  lower-priority work from dispatching in your gap).
- Manual `reserve` holds live under the same law: default ttl 3
  minutes, expiry is automatic — no hand-release needed.

## The invariants (audited + tested 2026-09-01)

1. **ONE machine-global, durable state** at `~/.sparkpipe/queue`
   (migrated from /tmp — macOS cleans /tmp; `SPARK_QUEUE_STATE`
   overrides start EMPTY and never inherit production). The stale per-checkout `runs/`
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
4. **Node sets are exclusive** for the task's duration (one window:
   default 3 minutes, hard cap 15, enforced); disjoint sets run in
   parallel; `--ttl-min` is the expected duration within that cap.
5. **The deadline is strict**: at ttl the task's processes are KILLED
   (group SIGKILL + pattern sweep) and the nodes released — no
   condition keeps a window alive past its lease. The reaper also
   releases nodes via the task's exit sentinel
   (`/tmp/sparkqueue-<id>.exit` on the first node) as soon as a task
   finishes; `cancel`/`done` on a running task kill its process GROUP
   (the wrapper self-reports its pid to `/tmp/sparkqueue-<id>.pid`;
   the fleet proxy eats `$!` stdout — never rely on it). A node whose
   kill is undeliverable is fenced and probed, never a queue blocker.
6. Results persist in `~/.sparkpipe/queue/results.jsonl`;
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

0. **Weights stay resident** (weightd lazy-expert residency across ALL
   models on every node) — the loop's speed IS the one-minute debug
   cycle. Never teardown-and-relaunch between experiments; a window is
   pure test time because nothing swaps. Touch ≤10% of your model's
   experts: full-expert sweeps spend your own window and evict other
   lanes' experts (LRU).
1. **Design the batch before touching the cluster**: half a dozen
   tests that answer questions, each with a crisp expected outcome
   ("if X then row 0 matches; if Y it diverges at element N"). Write
   the hypothesis in the task `--notes` — the queue log becomes the
   experiment ledger.
2. **Each test = one executable + one queue task**: dedicated test
   programs, scripts, or an instrumented driver build. Stage the
   binary (git branch or scp), then:
   `add --id lane-x-q3 --nodes spark4 --ttl-min 3 --cmd-file run.sh`
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
