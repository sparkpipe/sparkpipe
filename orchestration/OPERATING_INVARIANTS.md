# SparkPipe operating invariants

These are correctness rules, not preferences. Stop and fix the responsible
scheduler, harness, task, or display whenever one is violated.

0. **Coordinator accountability:** Codex obeys every invariant here and enforces
   all of them on every Luna, OxAlpha, runner, integration, and status display.
   Every agent receives these invariants before acting. Output that violates an
   invariant is rejected; urgency, apparent success, and prior approval do not
   excuse a violation.
1. **PERT provenance:** Every BigGulp is a PERT node. Every derived bite names
   exactly one existing PERT parent. No runner-only or orphan work exists.
2. **One atomic queue:** Physical Luna workers receive no hardcoded task. A
   fungible worker atomically claims the highest-priority dependency-ready
   BigGulp, executes one useful turn, transitions it, and claims again.
3. **No slot held while waiting:** A Luna parks a BigGulp only on a named live
   child or external event, then immediately claims other ready work.
4. **Bite-sized OxAlpha work:** One request answers one question or implements
   one narrow solution, normally in 10-30 minutes, with one bounded write set,
   one decisive acceptance command, and at most 80 new production lines unless
   the coordinator records a concrete exception.
5. **Coordinator defines the grammar; foreman steers:** The coordinator defines
   production bites, safety boundaries, and the parameterized exploratory
   hill-climb grammar. A Luna may instantiate the next experimental cell from
   that grammar using current evidence, but may not invent production
   architecture or broad write sets. It reports a decomposition gap when the
   grammar cannot express the needed action.
6. **No ownership bypass:** Every OxAlpha request has a registered BigGulp
   parent before the first provider call. Parentless calls fail closed.
7. **Work conservation:** Keep every safe Luna slot and every independent
   provider race busy while dependency-ready, conflict-free work exists.
8. **Parallel by default:** Run independent BigGulps, bites, provider legs,
   builds, and hardware experiments concurrently. Serialize only a proven
   dependency, write conflict, scarce resource, or safety boundary.
9. **Results, not activity:** Progress is an executed experiment, analysis tied
   to its output, a valid Spark result, or integrated tested production code.
   Plans, inventories, API calls, receipts, and lines written are not progress.
10. **Code-size penalty:** Score accepted work as `validated solutions / net
    production LOC squared`. Failed or rejected code scores zero and appears
    only in the AI-slop counter. Tests that exercise production code are not
    production LOC.
11. **No camouflage:** Never hardcode agents, assignments, counters, success,
    or evidence to make a broken scheduler or dashboard look healthy. Reproduce
    the underlying bug, fix it, and add a regression test.
12. **Truthful telemetry:** Queue ownership, provider streams, deployments,
    benchmarks, gains, failures, and tokens come from their authoritative raw
    state. Orphans and stale processes are explicit errors.
13. **Hardware evidence:** A Spark deployment or benchmark counts only after a
    saved command, node/config/checkpoint identity, exit status, and raw output.
    Lease renewal additionally requires a numerically valid exact-32K B1/B8/B64
    prefill or output gain of at least 1 percent.
14. **Data ownership:** Only a fileadmin-owned bite may copy, move, shard, encode,
    or activate model data. Launchers and coordinators consume fenced receipts.
15. **Audit proportionality:** Exploratory work needs a decisive real result and
    no independent audit. Production code needs deterministic tests, independent
    approval of the exact patch, coordinator review, and then integration.
16. **Provider continuity:** Race independent providers, preserve agent context
    on disk, retry transient API errors, switch at the first provider failure,
    and retain per-provider latency, token, win, and error statistics.
17. **Source of truth:** The canonical public repository and atomic queue are
    authoritative. Agent workspaces submit results; they never silently mutate
    canonical state or claim integration.
18. **Wall-clock efficiency:** Minimize elapsed time from dependency readiness
    to the next useful result. Track queue wait, claim-to-dispatch,
    dispatch-to-first-byte, claim-to-experiment, and experiment-to-next-action.
    Polling, paperwork, repeated inventory, and idle waiting must not occupy the
    critical path. If useful-result throughput stalls, diagnose and fix the
    bottleneck before producing more work.
19. **Effective Spark utilization:** Maximize useful Spark-seconds divided by
    leased Spark-seconds, not superficial GPU activity. Useful time is a
    reasonable-EV launch, correctness run, benchmark, profile, or validated
    optimization on the assigned PERT critical path. Every assigned idle Spark
    is a scheduler fault: dispatch eligible work immediately or record the exact
    blocker, owner, wait age, and next event, then release/rotate the lease.
    Never run a low-value experiment merely to make hardware look busy; every
    run must answer a named question needed to advance production code. Leases
    last one hour and renew only on the exact-32K, numerically valid B1/B8/B64
    prefill or output gain required by invariant 13.
20. **Context isolation:** One Luna generation processes exactly one BigGulp
    turn. It starts with only the invariant prompt and that BigGulp's disk-backed
    state, persists its result, releases or parks the claim, and terminates. A
    fresh Luna generation reuses the physical pool slot for the next claim. No
    conversation context crosses BigGulps, and agents never summarize one task
    into the next task's prompt.
21. **Bite-level parallelism:** A BigGulp claim coordinates its bites; it does
    not serialize them. In each turn the foreman launches every dependency-ready,
    conflict-free bite concurrently. A live child blocks only its descendants
    and overlapping write sets or hardware leases. Park the BigGulp only when no
    runnable bite remains. Locks are scoped to a bite, write set, or scarce
    resource, never to unrelated work in the same BigGulp.
22. **Bite-scoped Spark queue:** Every hardware action is a job in one atomic
    Spark usage queue and names its PERT parent, bite, exact nodes, resource
    types, runner role, expected-value question, task spec, and result path
    before dispatch. Only that bite waits for the result. Jobs conflict only
    when both node sets and resource types overlap: storage transfer may run
    beside GPU compute, while service activation can declare both resources.
    The queue reports submit, wait, run, and receipt ages. No direct or
    BigGulp-wide hardware reservation may bypass it.
23. **Result-to-next-action closure:** A foreman turn does not end at the first
    result. It must validate the result, update the bite state, compute every
    newly dependency-ready predefined bite, and dispatch or enqueue all
    conflict-free next actions before releasing its claim. A state such as
    `next_ready_not_dispatched` is an invariant violation unless a named queue,
    resource, or safety blocker prevented dispatch.
24. **Decision-grade bite evidence:** Every bite declares the exact useful data
    before dispatch: fields, units, control or baseline, raw artifact paths,
    identity and provenance, success and failure observations, and the decision
    each outcome unlocks. Narrative status or a generic receipt is not
    acceptance. Missing required data produces a correction run, not an
    optimistic summary.
25. **Bounded event-backed waits:** `waiting` is valid only for a named child or
    external event with a live heartbeat and an explicit deadline. Telemetry
    shows both total wait age and event-heartbeat age. A missing runner or event
    heartbeat older than one hour is terminalized as stale and wakes the owning
    BigGulp. Any total wait older than one hour is re-queued once for a foreman
    value/retry review even when its event remains live. A decomposition gap,
    missing executable bite, or coordinator decision is `blocked`, never
    disguised as waiting.
