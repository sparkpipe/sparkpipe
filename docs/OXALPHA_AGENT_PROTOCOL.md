# Ox Alpha paired-agent development protocol

This protocol governs temporary Ox Alpha implementation capacity. It is
designed for unreliable APIs and prevents an agent from writing the canonical
SparkPipe checkout.

## 1. Authority model

- The canonical checkout is coordinator-owned and is the only integration
  source.
- Every agent receives an immutable base commit and works in an isolated clone.
- Agents never commit, push, open PRs, use repository credentials, or edit the
  canonical checkout.
- An implementer may change only the task's declared write set.
- An auditor receives a fresh clone with the implementer's patch applied. The
  auditor does not receive the implementer's mutable workspace.
- The auditor never fixes code. It tries to invalidate the patch and emits a
  structured verdict.
- The coordinator does not inspect a candidate patch until the independent
  auditor returns `APPROVE`.
- Approval queues a candidate for coordinator review; it does not merge it.

## 2. Unit of work

One task contract contains:

```text
task id and title
base commit
objective and non-goals
declared write set
required source files and contracts
dependencies and input artifact hashes
acceptance criteria
required positive, negative, and regression tests
target-hardware requirements
evidence/report schema
estimated effort and retry budget
```

Tasks should fit one focused agent session. A broad component is decomposed
until each patch has a reviewable semantic boundary and disjoint write set.

## 3. Pair state machine

```text
BLOCKED_DEPENDENCY
  -> READY_IMPLEMENTER
  -> IMPLEMENTING
  -> IMPLEMENTER_RETRY_WAIT
  -> IMPLEMENTER_COMPLETE
  -> PREPARING_AUDIT
  -> AUDITING
  -> AUDITOR_RETRY_WAIT
  -> AUDIT_REJECTED -> READY_IMPLEMENTER (new attempt)
  -> AUDIT_APPROVED -> READY_COORDINATOR
  -> INTEGRATING
  -> INTEGRATED | COORDINATOR_REJECTED | SUPERSEDED
```

No transition skips audit. Each attempt has its own implementation patch hash,
test evidence, audit hash, logs, and provider session ID.

## 4. API failure handling

API errors are expected. The controller classifies failures before changing
task state.

### Retryable

- HTTP 408, 409 caused by provider session contention, 425, 429, 500, 502,
  503, or 504;
- connect, DNS, TLS, read, write, or empty-response failures;
- truncated streaming/JSON events before a terminal event;
- provider overload, endpoint unavailable, or model temporarily unavailable;
- local CLI crash where the workspace and session record remain intact; and
- a stale worker heartbeat with no child process.

### Not automatically retryable

- authentication or account suspension;
- task/schema errors in the controller;
- a permission violation or write outside the declared workspace;
- an agent changing files outside its write set;
- an auditor modifying tracked source;
- deterministic test failure; and
- exhausted attempt, wall-clock, or token budgets.

### Resume strategy

1. Persist every JSON event before interpreting it.
2. If a provider session ID exists, resume the same session with a compact
   recovery prompt describing the last durable state.
3. If resume fails twice, start a replacement session in the same workspace.
4. If workspace integrity fails, recreate it from the base commit and replay
   the last accepted patch/checkpoint.
5. Use exponential backoff with full jitter, provider-wide rate-limit cooling,
   and a circuit breaker.
6. Never count a transport retry as a code-attempt failure.

Default delay is `min(300 seconds, 2^n * 2 seconds)`, randomized from zero to
that ceiling. A provider circuit opens after five correlated failures in 60
seconds and probes with one request after the cooling interval.

## 5. Implementer output contract

The controller captures the git diff mechanically. The implementer response
must include:

```json
{
  "status": "READY_FOR_AUDIT",
  "summary": "...",
  "changed_paths": ["..."],
  "tests": [
    {"command": "...", "exit_code": 0, "evidence": "..."}
  ],
  "known_limits": ["..."],
  "hardware_claims": [
    {"claim": "...", "class": "MEASURED|SIMULATED|ANALYTICAL|UNVERIFIED"}
  ]
}
```

The controller rejects an empty diff, a dirty submodule, a changed path outside
the write set, generated drift, secrets, credentials, private addresses where
forbidden, binary artifacts, or missing required tests.

## 6. Auditor output contract

The auditor starts from the task contract, not the implementer's conclusion.
It must read the diff, inspect surrounding code, rerun required tests, add
adversarial tests where practical, and verify semantic/non-goal boundaries.

```json
{
  "verdict": "APPROVE|REJECT|BLOCKED",
  "patch_sha256": "...",
  "findings": [
    {
      "severity": "P0|P1|P2|P3",
      "path": "...",
      "line": 0,
      "title": "...",
      "evidence": "..."
    }
  ],
  "tests": [
    {"command": "...", "exit_code": 0, "evidence": "..."}
  ],
  "scope_verified": true,
  "tracked_source_unchanged_by_auditor": true
}
```

Approval requires zero P0/P1/P2 findings, all mandatory gates green, matching
patch hash, and no tracked source edits by the auditor. P3 findings may approve
only if the task contract explicitly permits deferred cleanup.

## 7. Coordinator review

After approval the coordinator:

1. verifies base and patch hashes;
2. reads the task, implementation, audit, and diff;
3. reproduces risk-proportionate tests in a fresh integration branch;
4. resolves overlap with already integrated work;
5. applies style, architecture, and security review;
6. commits and opens a PR through the repository authentication wrapper; and
7. performs zero-drift target deployment before a production claim.

The coordinator may reject an auditor-approved patch. That rejection becomes a
new task input; it is never silently edited into a different patch.

## 8. Global task scheduler

The development scheduler is a DAG executor, separate from SparkPipe's product
scheduler. It provides:

- dependency and artifact-hash readiness;
- write-set collision exclusion;
- role separation and no self-audit;
- provider and model concurrency limits;
- fair queues by critical-path slack and priority;
- retry/backoff/circuit-breaker state;
- resumable provider sessions;
- token and wall-clock budgets;
- stale-worker detection and process adoption after controller restart;
- immutable event history; and
- coordinator integration queue ordering.

Priority is ordered by blocking critical-path length, then explicit program
priority, then age. Two tasks with overlapping write sets do not implement
concurrently even if their DAG dependencies are independent.

## 9. Real-time display

The controller exposes a localhost dashboard and a JSON snapshot. The display
is derived from persisted state, not process names or hand-edited markdown.

Required views:

- aggregate counts by pair state, provider, retry reason, and workstream;
- one row per implementer/auditor pair;
- current task and attempt;
- role state (`working`, `waiting for implementer`, `retry backoff`,
  `auditing`, `ready for coordinator`, and terminal states);
- queued task count for each role and pair;
- API retries, next retry time, circuit status, and last heartbeat;
- provider session IDs and elapsed/token usage without secrets;
- last durable event and artifact links; and
- PERT readiness/blocked counts.

The page refreshes from an event endpoint at least once per second and shows a
stale warning when the controller heartbeat is older than ten seconds. The
same information is available through `status --json` for scripts.

## 10. Launch policy

Concurrency ramps rather than jumps:

1. one pair proves implementation, patch capture, independent audit, and
   restart recovery;
2. four pairs prove write-set locking and provider backoff;
3. eight pairs measure stable error and throughput rates;
4. then increase by four pairs while 429/5xx rate remains below the configured
   threshold; and
5. shrink automatically when the provider circuit reports overload.

The number of configured pairs is not the same as simultaneous API requests.
Auditors normally wait while implementers run, so `N` pairs begin with at most
`N` active requests. Provider limits, not ambition, set the live ceiling.

## 11. Security

- Credentials remain in the provider CLI's credential store.
- Logs redact authorization headers, environment values matching secret-name
  patterns, and bearer-like tokens.
- Prompts never include credentials, private `.env` files, or unredacted
  customer data.
- Workspaces have no GitHub credentials or writable canonical remote.
- Network and target-hardware access are task-scoped.
- A worker exceeding its filesystem or command policy is terminated and its
  patch is rejected.

Ox Alpha is a temporary, anonymous-provider preview. Only repository content
that is safe to disclose to that provider may be placed in its prompts or
workspaces.

## 12. Controller operations

The reference controller is `tools/oxalpha_fleet.py`. Durable state and agent
workspaces belong outside the checkout; `/private/tmp/sparkpipe-oxalpha-live`
is the default operational location used by the coordinator.

```sh
python3 tools/oxalpha_fleet.py init \
  --state-dir /private/tmp/sparkpipe-oxalpha-live \
  --repo . --graph orchestration/platform_tasks.json --pairs 4

python3 tools/oxalpha_fleet.py run \
  --state-dir /private/tmp/sparkpipe-oxalpha-live \
  --pairs 1 --pool host --once

python3 tools/oxalpha_fleet.py serve \
  --state-dir /private/tmp/sparkpipe-oxalpha-live \
  --host 127.0.0.1 --port 8765

python3 tools/oxalpha_fleet.py status \
  --state-dir /private/tmp/sparkpipe-oxalpha-live --json
```

The controller injects a restrictive OpenCode permission policy, strips
non-OpenCode credential environment variables, disables external directories,
plugins and subagents, removes the Git remote in every clone, and mechanically
rejects out-of-scope paths, binary patches and secret-like values. It needs
access to OpenCode's own credential store when launched.

`--pool host` dispatches only host work. Live profile tasks name explicit pools
such as `spark-tp4`, `spark-single`, `spark-tp16`, or `spark-fleet`; the
controller will not dispatch those tasks to a host worker. A second controller
cannot acquire the same state directory lock.

After coordinator inspection and integration, advance the canonical base and
unlock dependents with:

```sh
python3 tools/oxalpha_fleet.py mark-integrated PERF-001 \
  --state-dir /private/tmp/sparkpipe-oxalpha-live \
  --repo . --commit EXACT_CANONICAL_HEAD
```

An API transport failure resumes the same provider session twice before a
replacement session is created in the same isolated workspace. It does not
consume a code attempt. A controller restart terminates the ownership epoch,
requeues interrupted work from the immutable base, and keeps its prior events
and feedback; no interrupted patch becomes coordinator-visible.
