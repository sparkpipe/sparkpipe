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
2. If a native session has an unanswered user/tool turn, resend that exact
   durable context without appending a duplicate recovery prompt. The optional
   OpenCode adapter resumes its provider session with a compact recovery prompt.
3. If resume fails twice, start a replacement session in the same workspace.
4. If workspace integrity fails, recreate it from the base commit and replay
   the last accepted patch/checkpoint.
5. Use exponential backoff with full jitter, provider-wide rate-limit cooling,
   and a circuit breaker.
6. Never count a transport retry as a code-attempt failure.

Default delay is `min(300 seconds, 2^n * 2 seconds)`, randomized from zero to
that ceiling. A provider circuit opens after five correlated failures in 60
seconds and probes with one request after the cooling interval.

### 4.1 Per-turn provider racing

`tools/oxalpha_race.py` is the provider-neutral turn racer used by the native
Codex harness in `tools/oxalpha_harness.py`. The harness preserves one logical
session and one mutable workspace. For each model turn, the racer sends the
identical messages, tools, and sampling contract to `R` distinct provider
failure domains; only the provider-specific model ID and credential differ.
OpenCode can use the loopback-compatible endpoint as an optional adapter, but
it is neither the context owner nor a dependency of AIML, Venice, OpenRouter,
or any other upstream.

The first structurally complete choice-zero response wins. A valid later choice
cannot hide a malformed choice zero. Only the successful OpenAI-compatible
terminal reasons `stop` and `tool_calls` are accepted. A first byte, partial
SSE, malformed tool arguments, empty content, an unknown or truncation finish
reason, malformed JSON, missing terminal marker, or provider error cannot win.
The proxy buffers each stream through its terminal marker, publishes only the
winner, closes loser connections, and immediately refills a failed hedge from
the next healthy provider. This avoids duplicate tool execution or concurrent
filesystem mutation.

Winner affinity is keyed to the implementer or auditor context. The next turn
starts with that winner and a rotated healthy backup. Providers sharing a
failure-domain label never occupy two hedge slots. Each provider declares its
entire known chain (for example, its gateway plus a shared OpenRouter upstream),
not merely its account vendor. A correlated gateway may be tried sequentially
after a failed leg settles, but it is not counted as an independent concurrent
hedge. Launch selection computes the same maximum disjoint set reported by the
dashboard rather than using a greedy first fit. Every launch atomically
rechecks current circuit state, so a candidate captured by a concurrent request
cannot bypass a newly opened cooldown. One failure opens a short provider
cooldown; a whole-request deadline opens it before asynchronous
connection cleanup completes. First-byte, stream-idle,
and whole-request deadlines are independent. Start with `R=2`; raise to `R=3`
only after at least three truly independent authorized provider chains and
measured quota headroom exist.

Pool files contain endpoint and credential references, never keys. Static
secret/token headers and credential-looking values are rejected. Both the full
header value and raw credential are redacted for every configured auth prefix
before an upstream or transport error can enter health/events. Credential files must not be
group/world-readable, and only providers explicitly marked
`hedging_authorized` participate. The
proxy binds to loopback, uses a random per-controller bearer token, and exposes
non-secret live health, in-flight, win, failure, cancellation, latency, and
circuit state to the controller dashboard. Status callbacks run on a bounded
dispatcher queue and cannot delay inference. Dispatcher lag and snapshot age
are explicit status fields, and per-race winner events precede loser-settlement
events. `in_flight` remains nonzero until the worker really settles, worker
startup failure rolls accounting back, and loopback request bodies have an
absolute deadline that trickle uploads cannot extend.

### 4.2 Durable native Codex sessions

Each implementer or auditor is one local session directory under the fleet
state directory. `state.json` is the provider-neutral conversation checkpoint;
`journal.jsonl` is the compact event ledger; `responses/` stores every complete
winning provider response; and `tool-results/` stores raw tool output. Large
tool output is represented in the API context by a bounded preview, hash, byte
count, and artifact name. The agent can page the raw artifact explicitly.

Tool calls are checkpointed before execution and run sequentially in the one
workspace. A completed result is durably archived before its tool message is
added. After a process interruption, archived results are replayed into context
without re-executing the tool. An interrupted mutating tool with no receipt is
reported as side-effect-uncertain and is never automatically replayed.

The controller records the native session independently of the worker pair and
pins every attempt to its original base commit, so a controller restart can
resume the same task workspace even if another pair claims it. Required tests
must have durable `run_command` receipts whose workspace fingerprint still
matches the final candidate, including ignored as well as ordinary untracked
files; the controller reopens each artifact, recomputes
its digest, and compares it with durable session state. Model-authored test
claims alone cannot advance a native candidate. Implementers and auditors
complete with a typed `finish_task` tool call, avoiding free-form or fenced JSON
parsing. Context is compacted to hashed local archives at 16 MiB, command output
is bounded while the process runs, and each session has a 1 GiB artifact budget
plus a persisted 1 GiB workspace-growth ceiling. Storage accounting runs while
commands execute; every exception path kills and reaps the owned process group.

Declared local tests run fail-closed in an OS sandbox: macOS uses
`sandbox-exec`, Linux requires Bubblewrap, the clone and private scratch paths
are the only writable trees, controller credentials and other home files are
unreadable, and networking is denied. Read access is limited to immutable
system paths and explicitly resolved Python/Git toolchain roots, never broad
`/opt` or `/Library` trees. If the required sandbox is unavailable,
the test fails instead of falling back to an ordinary shell. Target-hardware
qualification is a separate trusted runner with an explicit hardware/network
policy; anonymous agent-authored code never inherits controller network access.

## 5. Implementer output contract

The controller captures the git diff mechanically. The implementer calls
`finish_task` with:

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
adversarial tests where practical, verify semantic/non-goal boundaries, and
submit this object through `finish_task`.

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

The seven initial model drivers use explicit durable lane affinity rather than
the generic pull queue. Each `model-driver:<code>` lane owns one logical
implementer/auditor pair across that model's task sequence so its
task-local provider-neutral context survives provider changes, API retries,
and controller restart. A newly admitted task starts from the repository and
the lane's already integrated artifacts rather than inheriting unreviewed chat
state. Both roles still use the configured provider race, the auditor still
starts from a fresh immutable workspace, and the lane cannot claim another
model's work. Shared-runtime or cross-model write sets leave the lane and
return to coordinator-controlled ownership. A planned lane is not an active
agent until its pair, admitted task, provider supply, lock, and heartbeat are
all present in durable state.

## 9. Real-time display

The controller exposes a localhost dashboard and a JSON snapshot. The display
is derived from persisted state, not process names or hand-edited markdown.

Required views:

- aggregate counts by pair state, provider, retry reason, and workstream;
- one row per implementer/auditor pair;
- current task and attempt;
- role state (`working`, `waiting for implementer`, `retry backoff`,
  `auditing`, `ready for coordinator`, and terminal states);
- queued task count for each role and pair; unbound workers show the global
  ready pull queue, while a dedicated model pair shows only its exact lane,
  with both split by implementer/auditor role;
- API retries, next retry time, circuit status, and last heartbeat;
- provider session IDs and elapsed/token usage without secrets;
- last durable event and artifact links; and
- PERT readiness/blocked counts; and
- one row per dedicated model-driver lane, including assigned logical pair,
  active task, lane queue depth, provider-race width, and truthful planned,
  blocked, idle, working, or stale state.

The page refreshes from an event endpoint at least once per second and shows a
stale warning when the controller heartbeat is older than ten seconds. A
controller heartbeat also persists a live provider snapshot independently of
event delivery; snapshot age and callback lag make a blocked status dispatcher
visible. The same information is available through `status --json` for scripts.

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

- Upstream credentials remain in controller-side environment variables or
  private credential stores; agent children receive only a loopback token.
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
  --pairs 1 --pool host \
  --harness codex \
  --provider-pool orchestration/oxalpha_providers.json \
  --provider-env-file ~/.env --once

python3 tools/oxalpha_fleet.py serve \
  --state-dir /private/tmp/sparkpipe-oxalpha-live \
  --host 127.0.0.1 --port 8765

python3 tools/oxalpha_fleet.py status \
  --state-dir /private/tmp/sparkpipe-oxalpha-live --json

# Repeat with one idle pair for each of q27,d4f,glm,k3,d4p,qmax,h3.
python3 tools/oxalpha_fleet.py bind-model-lane \
  --state-dir /private/tmp/sparkpipe-oxalpha-live \
  --pair pair-002 --lane model-driver:q27
```

The default native Codex harness exposes only workspace-confined reads and
protected-file-filtered searches, declared test commands, a small exact set of
read-only Git inspections, text patches inside the write set, and
session-artifact reads. It removes the Git remote in every clone and
mechanically rejects every path reported by Git's full patch parser when it is
out of scope, plus binary patches, symlinks, submodules, and secret-like
candidate values. Test commands are additionally confined by the fail-closed OS
sandbox described above. Provider credentials remain only in the controller
process; they are stripped from all test-command environments. A private `~/.env` may
hold provider keys, but only credential variable names declared by the provider
pool are loaded. OpenCode remains available only through `--harness opencode`
for compatibility testing.

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
requeues an interrupted implementer at implementation and an interrupted
auditor directly at audit, always on the same attempt, immutable base, session,
and role-specific workspace. A recovered rejected attempt advances to the next
attempt, while a legacy approved transition goes directly to the coordinator
queue. Approval no longer has an intermediate crash-strandable state. Recovery
keeps prior events and feedback; no interrupted patch becomes
coordinator-visible.
