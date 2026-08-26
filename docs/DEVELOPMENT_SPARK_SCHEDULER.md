# Development Spark scheduler

`tools/spark_development_scheduler.py` is the single development-control-plane
owner for Spark assignments used by model-driver agents. It is a planner, not
an SSH or file-management tool. The coordinator emits generation-fenced plans;
the files/execution agent performs the swap and returns an exact receipt.

## Invariants

- A logical model-driver pair is durably bound to one `model-driver:*` lane.
- At most one big-model development lease is active in the initial shared
  sixteen-Spark pool.
- `small-models-current` is drained and restored as one atomic group. Its
  original exact snapshot and digest remain durable through any number of
  non-restoring big-model handoffs.
- A lease clock starts only after the `files-agent` proves all activation-plan
  steps were applied to the exact union of requested big-model Sparks and
  every Spark in the captured small-model snapshot.
- A lease lasts 3,600 seconds. Code changes, tests, profiles, heartbeats, and
  invalid inference do not renew it.
- A numerically valid first measurement establishes a baseline only for a lease
  explicitly admitted as `ESTABLISH_IF_NONWORKING`. `REQUIRE_GAIN` can record
  observations but can never create an accepted baseline. Each later renewal
  requires at least a 1.00% gain against an independently accepted best.
- Qualifying cells are B1, B8, or B64 with exactly 32,768 prompt tokens, 256
  output tokens, disabled prefix cache, at least one warmup, and at least three
  measured samples. Checkpoint, quality, topology, hardware, and timing
  protocol are immutable parts of the lease contract.
- Batch, prompt, output, generation, sample-count, and warmup fields are exact
  JSON integers; booleans and floating-point representations are rejected.
- Both `measured_at` and the scheduler's receipt-acceptance time must be no
  later than the current lease deadline. A result received after expiry cannot
  renew a lease even if it was measured earlier. `measured_at` must also be no
  later than receipt acceptance; future-dated results have no tolerance.
- Every benchmark receipt names one regular source artifact beneath the
  evidence root and carries its descriptor-verified fingerprint. Missing,
  symlinked, changed, or mismatched evidence fails closed before admission.
- Expiry first emits a newer-generation fence/checkpoint plan. The next queued
  model cannot receive the Sparks until the file agent acknowledges that fence.
- Equal-priority requests are FIFO. Higher-priority ready requests run first.

The initial baseline exception exists so a model explicitly classified as
nonworking can earn its first hour. It applies only while that model has no
accepted baseline. Every accepted baseline carries receipt-level provenance;
legacy or malformed best rows without an establishing receipt are ignored and
removed. Changing quality or topology does not manufacture progress.

Accepted benchmark values are persisted both as displayable SQLite numbers
and bounded decimal text. Renewal compares the decimal values exactly: the new
value must be strictly greater and `new * 100 >= accepted * 101`. Thus an
unchanged subnormal value can never qualify because of floating-point rounding.

## SOTA display contract

The live dashboard contains one row for every supported model at B1, B8, and
B64. Each row lists SparkPipe's accepted best prefill/output rate and the exact
public SOTA value plus its 110% target, source, date, accelerator, and count.

A public value is shown only when the primary-source ledger contains the exact
32,768-prompt/256-output fixed workload, matching batch, disabled prefix cache,
passed output parity, and the same metric unit and direction. Full performance
gap is withheld until the entire comparison tuple is certified comparable.
Missing values render as `N/A`; a shorter-context result is never relabeled.

An eligible row must also identify the observation, checkpoint name and
revision, topology kind and positive integer TP/PP/EP sizes, accelerator name
and positive integer count, and metric timing boundary. Its source must be an
HTTPS URL with a valid `retrieved_utc` timestamp and either a publication date
or immutable source revision. Malformed JSON, partial nested records,
out-of-domain integers, and numeric conversion overflow are ignored per row as
`N/A`; they cannot crash or partially populate the dashboard.

At bootstrap time the committed ledger has no exact cells for this display, so
all public SOTA cells correctly start as `N/A`. Daily SOTA scanning can fill
them only by adding source-backed observations through the audited ledger
pipeline.

## Durable workflow

Initialize scheduler state under the OxAlpha fleet state directory:

```console
python3 tools/spark_development_scheduler.py \
  --state-dir /path/to/fleet-state/spark-development init
```

When first initialized with parent `fleet.sqlite3`, scheduler tables live in
that same database. Otherwise they live in the standalone scheduler database.
The selected mode is durably recorded in `spark-development/database-mode`.
Both stores existing, a persisted integrated store disappearing, or a fleet
database appearing beside a persisted standalone store is an error; mode never
switches implicitly. Pair affinity in `pairs.agent_lane` and scheduler affinity
in `lane_affinity` are updated by one `BEGIN IMMEDIATE` transaction. A failed
insert or process crash therefore leaves both views unchanged; there is no
compensating second commit. Standalone scheduler tests still use
`spark_development.sqlite3` when no fleet database exists.

Releasing a model lane marks its affinity row `RELEASED` instead of deleting
it. Historical requests retain their lane foreign key, while only `BOUND` rows
participate in active pair uniqueness and dashboard affinity. The released
pair and lane can therefore be rebound after unfinished work is cleared.

Import a full sixteen-node observation without changing the fleet:

```console
python3 tools/spark_development_scheduler.py \
  --state-dir /path/to/fleet-state/spark-development \
  observe --file orchestration/execution_state.example.json
```

Then bind the dedicated model pair, enqueue a validated model recipe request,
and ask for the next plan. `ack` accepts only executor identity `files-agent`,
the same plan ID, generation, requested Sparks, exact affected-Spark union,
small-model snapshot and digest, execution-contract digest, and exact ordered
step list. An applied receipt must equal `steps`; a rollback receipt must equal
`rollback_steps` and repeat the exact restored snapshot states. Duplicate steps
are invalid.

A non-restoring `FENCE_AND_HANDOFF` does not clear the suspended small-model
snapshot. The next activation inherits that original snapshot even when its
fresh physical observation truthfully reports no resident small models. The
snapshot is cleared only in the same transaction that accepts a `files-agent`
receipt proving activation rollback or final small-model restoration.

The scheduler must have an evidence root for acknowledgement. In integrated
fleet mode it derives the root from the fleet's immutable `canonical_repo`
binding; standalone CLI use supplies
`--evidence-root /absolute/repository/path`. Receipt
`evidence_paths` are sorted, unique, normalized repository-relative paths.
The scheduler opens the root once, walks each parent relative to that descriptor
with `O_DIRECTORY|O_NOFOLLOW`, and opens the final regular file with
`O_NOFOLLOW`. It hashes the fstat-verified descriptor, so concurrent parent
renames or symlink swaps cannot redirect validation outside the root. Receipts
are bounded to 64 files, 8 MiB per file, and 32 MiB total.
For each file, the scheduler builds this record:

```json
{"path":"qualification/executor/plan.json","bytes":123,"sha256":"..."}
```

It sorts records by path, serializes the array as canonical JSON (sorted keys,
ASCII, and no insignificant whitespace), and computes SHA-256 over those UTF-8
bytes. The lowercase hexadecimal digest must exactly equal
`result_fingerprint`; a caller-supplied but unverified 64-character digest is
not evidence. `benchmark` uses the same descriptor-safe one-file fingerprint
for `source_path`/`source_fingerprint` and rejects stale generations, expired or
future measurements, absent evidence, and mismatched benchmark contracts.

Every applied or rolled-back executor receipt atomically advances a durable
generation/timestamp barrier and invalidates the previous observation. A next
plan requires a complete observation whose timestamp is strictly newer and
whose adopted barrier generation matches. This ordering survives restart and
also applies to newer-generation fence receipts; replaying an identical receipt
does not advance the barrier again.

Scheduler metadata uses namespaced keys including
`spark_scheduler_schema_version`, `spark_scheduler_generation`,
`spark_scheduler_request_sequence`, `spark_scheduler_observation_epoch`, the
executor barrier pair, and the suspended-snapshot pair, so the shared SQLite
database cannot confuse scheduler state with fleet counters.

The first live adoption and every post-receipt adoption must re-probe every
Spark immediately before importing the observation. An observation collected
before a state-changing receipt cannot authorize the next plan. A repository
example is a schema fixture, not evidence about current processes, files,
mounts, or model ownership.
