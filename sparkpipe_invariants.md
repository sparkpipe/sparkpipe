# SparkPipe invariants

This is the required firmware contract, not a claim that the current code
already satisfies it. An invariant is not optional for a driver. A missing
required operation or violated invariant must produce an explicit error and
block functional acceptance. A flag, stub, fallback, reduced test, or document
cannot turn missing behavior into an accepted implementation.

GLM 5.3 Flash is the current implementation and optimization focus. Other
drivers must follow the same contract; their qualification is not implied by
changes to common code. Track implementation evidence in
[driver acceptance](docs/DRIVER_ACCEPTANCE.md), the
[GLM hill-climbing log](docs/GLM_FLASH_HILLCLIMB.md), and
[performance gates](docs/GLM_PERFORMANCE_GATES.md).

## 1. One fixed serving contract

- **I01 — Required means required.** Drivers implement every required operation
  and its behavior. Missing functions fail explicitly, naming the missing
  operation. Success-returning stubs, unused callbacks and capability claims
  do not establish functionality.
- **I02 — Drivers implement inference, not serving policy.** Common code owns
  scheduling, admission, batching, cache transactions, ownership, cancellation,
  release, progress orchestration and engine lifecycle. A driver cannot choose
  a different policy because it is easier to implement.
- **I03 — No opt-outs from correctness or required functionality.** Descriptor
  bits, environment variables, configuration fields and compile-time switches
  cannot waive required behavior. Unsupported geometry or hardware fails
  explicitly; it does not silently select an incomplete path.
- **I04 — Configuration describes the deployment.** Model geometry, formats,
  topology, backend and real resource budgets are explicit, validated inputs.
  They are not permissions to skip work. Capacity declarations must match
  allocated resources and qualified execution limits, not hide debug clamps.
- **I05 — One public ABI.** Drivers, loaders, probes and tests use current
  shared headers. Incompatible binaries fail loading. Do not copy private
  struct definitions or preserve obsolete semantics through compatibility
  aliases. ABI 22 removes driver-selected slot reuse entirely.

## 2. Shared implementation and simplicity

- **I06 — Share at the broadest valid boundary.** Universal policy is preferred;
  topology algorithms, mathematical operations shared across model families,
  hardware backends, codecs and fabrics are additional overlapping boundaries.
  These are not a rigid inheritance hierarchy.
- **I07 — One algorithm, narrow hooks.** Use the qsort pattern: common algorithms
  with small callbacks and opaque context for actual model/backend differences.
  Model hooks supply math, geometry and complete state layout. Backend hooks
  supply allocation, kernels, copies, events and transport operations.
- **I08 — Reuse before adding machinery.** Extend existing common primitives.
  Do not copy a scheduler, cache manager, transaction protocol or collective
  state machine into each driver. Do not add a framework or alternate path
  without a concrete requirement. Keep hot mathematical specialization direct.
- **I09 — Hardware independence is real.** Universal policy contains no CUDA,
  NCCL, Metal or host-specific assumptions. CUDA and future Metal implementations
  obey the same ownership and completion rules. Portability requires testing;
  an abstraction alone does not qualify the Mac Studio backend.
- **I10 — Verify semantics before changing code.** Read the actual fields,
  signatures, units, strides, layouts and lifetimes. Keep code compact and
  dependency ordered; avoid hot-path allocation and unnecessary scans or
  synchronization. Prefer the smallest implementation that satisfies the rule.

## 3. Engine, sequence and completion ownership

- **I11 — A deployed engine persists.** Keep one engine per deployed
  code/configuration version across requests and benchmark phases. Weights,
  transport sessions and reusable cache survive request completion. Replace
  the engine for a deployment/configuration change or unrecoverable failure,
  not because an HTTP client disconnects or a batch finishes.
- **I12 — Release precedes reuse.** Common code releases a bound sequence before
  assigning its resident slot to another owner. Position zero grants no
  exception. Drivers cannot advertise implicit reuse. Request identity,
  sequence identity and generations must match every ownership transition.
- **I13 — Device completion determines lifetime.** A submission returning, a
  callback being queued, or a timeout expiring does not mean device work ended.
  Buffers, pages, expert mappings, slots, credits and communication workspaces
  stay owned until every operation using them is terminal.
- **I14 — Completion is safe for immediate reuse.** Publish final state and
  release execution claims in the required order before notifying consumers.
  Completion data must survive callbacks that immediately reuse the slot.
  No producer may access a recycled submission record after notification.
- **I15 — Cancellation and failures preserve ownership.** Stop new work for
  the request, drain or quarantine outstanding operations, then release safely.
  Do not report success for failed work or return an unsafe resource to a pool.
  A failed drain remains an explicit failure, not permission to free memory.
- **I16 — Reset is coordinated.** Quiesce admission and device work, invalidate
  old generations, clear all sequence-dependent state consistently, and only
  then resume. Normal sequence reset does not reload unchanged weights.
  Partial reset failure must not reopen admission into mixed state.
- **I17 — BUSY represents recoverable pressure.** There must be an outstanding
  operation or resource transition that can make retry useful. An impossible
  ownership state must fail diagnostically, not spin indefinitely. Liveness,
  readiness and demonstrated request progress are distinct observations.

## 4. Continuous batching and weight reuse

- **I18 — Arbitrary batch sizes work.** Admit arrivals and retire completions
  continuously, including odd sizes, mixed positions and partially filled
  batches. Internal tiles or buckets must not exclude eligible requests or
  change semantics. A power-of-two kernel shape is not a serving restriction.
- **I19 — Batching executes together.** Group eligible dense and expert work
  so weight reads serve multiple requests. Repeating B1 serially and reporting
  B3 outputs is not batched execution. Verify launch shapes and actual reuse.
- **I20 — Admission is bounded and fair.** Resource limits may queue work;
  they must not lose it or starve it. Chunk prefill fairly with decode, bound
  temporary storage, and exercise arrivals, cancellation and completion under
  pressure. Logical batch policy remains common across kernels and topologies.
- **I21 — Scaling is measured.** Pursue near-linear aggregate output growth
  with occupancy until measured compute, memory or communication limits
  intervene. Exercise occupancies through roughly 100 and beyond the measured
  crossover when resources permit. Neither 100 nor linear scaling is assumed
  to be a proven hardware limit or guaranteed result.
- **I22 — No hidden production clamps.** Diagnostic changes to execution shape
  or behavior belong under `#ifdef DEBUG` with visible diagnostics. DEBUG
  still does not waive required functionality. Production resource bounds are
  explicit and validated; a formerly crashy path must be fixed, not clamped.

## 5. Complete cache state and transactions

- **I23 — JIT and prefix reuse are required common behavior.** A driver cannot
  disable them or substitute full recomputation. A normal cache miss may compute
  the missing prefix; a benchmark explicitly requiring a hit must fail on a miss.
- **I24 — Restore the whole computation state.** Cache payloads include every
  relevant KV/index tensor, recurrent state, convolution window, sequence
  position and continuity field. For GLM this includes KDA state. KV alone is
  insufficient when other state influences the next token.
- **I25 — Logical identity and physical residency are separate.** Resolve real
  physical mappings, pin pages during use, and preserve immutable shared
  prefixes with copy-on-write. Eviction cannot reclaim referenced or pinned
  state. A fixed identity mapping is not JIT cache support.
- **I26 — Publication is transactional.** Prepare, commit, claim, finish and
  abort use common ownership rules and complete identities/generations.
  Publish a reusable prefix only after all matching payloads are complete.
  Partial failures roll back safely; stale transactions cannot publish or
  release a newer owner's state.
- **I27 — Restored execution matches uninterrupted execution.** Prove hits,
  misses, eviction, movement, copy-on-write, slot changes and reset with real
  model state. Descriptor declarations and byte-copy tests alone are insufficient.

## 6. Experts, fleet access and deployment

- **I28 — Lazy loading is strict.** Missing, corrupt or incompatible required
  `.experts` data fails explicitly. Never fall back to eager/direct loading
  because lazy-loading prerequisites are absent. Any deliberate resident
  benchmark mode must be selected explicitly and reported as such.
- **I29 — Expert residency is bounded.** Load the working set within the
  declared budget, protect in-flight mappings, and reclaim only unowned data.
  Reuse the common loader/weight daemon rather than driver-specific substitutes.
- **I30 — Developers can debug independently.** Assigned-node component tests
  use strict lazy loading and the automatic verified rsync/build path.
  Independent jobs interleave within real resource budgets. Full-fleet
  reservations are for tests that actually require the full topology, including
  distributed correctness and isolated performance measurements.
- **I31 — One authoritative resource ledger.** Queue jobs and persistent
  engines both count toward memory/device reservations. An empty queue does
  not prove idle hardware. Notes, stale locks or observation timeouts cannot
  reserve resources forever or prove a running job has stopped.
- **I32 — Cleanup is scoped and verified.** Stop only owned work. Release a
  reservation after authoritative process/cgroup and device-lifetime evidence,
  not merely after a client exits. Reconcile interrupted attempts without
  duplicate launches or killing unrelated developers' jobs.
- **I33 — Deployment evidence has zero drift.** Use a reviewed PR, merge to main,
  pull/sync the exact merged source on participating Sparks, rebuild/install,
  restart the changed engine and then validate. Record source, artifact and
  pack hashes. Dirty trees, copied driver hotpatches and unmerged binaries do
  not qualify deployment behavior.
- **I34 — CI and Spark testing proceed independently.** Run focused checks and
  review, let CI run, and perform merged-main Spark validation in parallel.
  Do not serialize hardware testing behind CI completion. Retain and resolve
  failures from both; neither a CI pass nor a Spark pass cancels the other gate.

## 7. Topology and communication

- **I35 — Topology policy is shared.** TP, PP and hybrid orchestration own
  partitioning, dependencies, collective identity, credits and progress in
  common topology code. Model hooks supply stage math and boundary geometry.
- **I36 — Use the required batch-dependent collective strategy.** B1 uses
  contribution broadcast with local reduction; B2+ uses tree reduction with
  independent compute overlap. Preserve every contribution and numerical
  semantics. Prove overlap on the execution timeline, not from asynchronous
  API names or stream counts.
- **I37 — Overlap cannot break ownership.** Collective identity is deterministic
  across ranks, steps and generations. Workspaces and credits remain valid
  through GPU and transport completion. Measure exposed communication, rank
  imbalance and weight reuse lost through splitting; splitting is not free.
- **I38 — Qualify each topology.** GLM's default is TP4xPP4; TP16 is the speed
  topology. A TP16 component pass cannot qualify TP4xPP4 boundaries or a filled
  pipeline. A single-rank probe with collectives disabled is component evidence.

## 8. Numerical and performance acceptance

- **I39 — Functional and performance verdicts are separate and mandatory.**
  A correct but slow driver is functionally qualified and performance-lagging.
  An incomplete or numerically wrong driver fails functionality regardless of
  speed. Neither state is “done.” Required evidence cannot be waived by paperwork.
- **I40 — Numerical gates test the real model.** Use pinned references and
  qualified tolerances for prefill, decode, mixed batches, long contexts,
  restored state and each topology. Repeatable nonsense is still wrong.
  Acceptance tools must reject bad/nonfinite values and propagate failures.
- **I41 — Separate timing phases.** Report load, prefill, TTFT, warm decode and
  batch tails separately. Seed a complete reusable cache entry, verify its hit,
  and benchmark decode on the persistent engine. HTTP wall time is not pure
  decode time. Report per-sequence latency alongside aggregate output tokens/s.
- **I42 — Compare matched workloads.** Match context, occupancy/traffic, timing
  boundaries, weight precision, speculation and hardware. Current GLM targets
  are non-speculative. Cite and verify community measurements before treating
  them as baselines; record uncertainty and normalization assumptions.
- **I43 — Normalize bytes and hardware honestly.** FP8 at half a 4-bit rate can
  be comparable only under an explicit bandwidth/traffic assumption. Account
  for actual weights, scales, expert traffic, cache traffic and quantization
  work where available. Four times the nodes must not disappear from the comparison.
- **I44 — Prove the GLM targets.** Establish a numerically correct TP4 baseline
  approaching measured sustainable memory roofline. Target TP16 at 3.5 times
  matched TP4 throughput and a filled TP4xPP4 pipeline at approximately four
  times TP4 serving capacity. These are acceptance targets, not claimed results;
  queued requests alone are not demonstrated capacity.
- **I45 — Hill-climb from evidence.** Profile the actual critical path, optimize
  the measured bottleneck, then repeat unprofiled measurements and numerical
  checks. Record bandwidth efficiency, exposed communication, occupancy,
  variability and regressions. Summed overlapping kernel times are not elapsed time.
- **I46 — Evidence is reproducible and scoped.** Preserve commands, source/build/
  pack identities, hardware/network topology, budgets, queue attempts, outputs,
  latency traces and verdicts. Protect credentials and private infrastructure
  details. A merge, compile, local smoke test or zero sanitizer errors cannot
  stand in for distributed functional and performance acceptance.
- **I47 — Reviews enforce these invariants.** Developer PR feedback identifies
  the violated rule, the existing common primitive to use, the required change
  and the behavioral evidence needed. Record GLM fixes and measured lessons
  so subsequent drivers follow the proven path rather than inventing variants.
- **I48 — No comments in code.** Put rationale, design explanations and
  implementation notes in documentation and PRs. Express the implementation
  clearly through names and structure. Tests verify executable behavior and
  contracts, never the presence or wording of comments. Do not restore or add
  a comment to satisfy a test; fix the test.

- **I49 — Model EOS is mandatory.** Common generation loads termination tokens
  from authoritative model deployment metadata. Missing or invalid EOS blocks
  engine startup. Caller stop tokens are additive and cannot replace model
  EOS. An EOS token in the prompt does not terminate prefill; a generated EOS
  completes the request through common release and slot-lifetime handling.
  No environment variable or driver option enables no-EOS serving.

- **I50 — Collective policy preserves the logical batch.** Every collective
  submission carries the full logical request count separately from its
  execution-row count. Missing logical metadata is an error. Splitting a
  B2+ batch into one-row chunks must not select the B1 direct algorithm.
  Direct B1 transfers must fit registered payload capacity; larger prefill
  transfers use the bounded tree path. Algorithm selection belongs in common
  collective code, and callers carry the count from the original frame.

## Applying this document

For each affected invariant, a PR states the behavior changed and the evidence
that exercises it. Keep missing evidence and known failures visible. Fix the
implementation or leave acceptance failed; do not weaken the invariant to
make a driver pass. Prefer fewer choices, fewer states and one correct shared
implementation over a growing collection of driver conveniences.
