# Model-driver inference hill-climb

This is the operating loop for every Luna model foreman. Model geometry,
checkpoint identity, legal sharding, numerical tolerances, and bottlenecks
differ; the search, evidence, and retain rules do not.

The objective is the shortest evidence-backed path from slow accurate
inference to a zero-drift production driver. Optimize quality-preserving,
end-to-end throughput over legal batch, context, topology, KV, and API cells.
GPU activity, an isolated kernel result, and a large table are not objectives.

The score is:

```text
validated Solutions / (production LOC^2)
```

Less production code and simpler measured solutions win. A failed or
unmeasured idea scores zero. A negative experiment is progress when it closes
a real branch of the search.

## Foreman contract

The Luna owns the model contract, oracle, production write-set, accuracy gate,
experiment queue, receipt, and next bite. It may choose the next power-of-two
or bisection cell, topology, collective, kernel, fusion, KV, batching, API, or
speculation probe; submit it to the bite-scoped Spark queue; and close failed
branches.

It may not change checkpoint or tokenizer identity, lower the numerical gate,
invent production architecture, move model data, mutate a live service outside
the scheduler, or turn an exploratory hotpatch into shipping code.

Every action is one of two phases:

- **Exploratory:** answer one named unknown using the real production path.
  The probe needs no independent audit. It still needs a raw command, exact
  identity, and a truthful measured result.
- **Production:** implement the smallest measured solution in a bounded write
  set. Audit it independently, compare it with clean main, merge, pull,
  rebuild/install, restart, and repeat the retained cells before release.

Plans, inventories, status prose, and receipt preparation are not progress.
Progress is a runnable bite, an executed result, analysis that chooses the
next bite, or tested production code.

## PERT execution mapping

`orchestration/program_pert.json` is the project graph and
`orchestration/pert_bites.json` is its executable refinement. Validate the
refinement with `python3 tools/pert_bite_catalog.py check`. Every PERT node has
one node record; `dependency_ready` and `ready_bites` identify its current
frontier without treating completion, readiness, or activity as synonyms.

Each bite names one parent PERT id, one acceptance atom, dependencies, exact
write and resource sets, inputs, command, pass/fail predicates, output,
required data, and a maximum 45-minute active-work budget. A foreman selects
only catalogued ready bites, launches every conflict-free bite, and updates
the catalog inputs from measured results. It does not silently invent a child
task or serialize unrelated atoms behind a model-wide queue.

Model work below an integration gate may use generated test-only exploratory
probes on independent fronts. Those probes change no production source and
need no audit. Their decisive result selects a real hardware probe, a bounded
production correction, or a rejected branch. Production changes retain the
implementer-plus-independent-auditor gate.

## Durable lane state

Keep one compact disk-backed record per model lane. Every value has units,
timing boundary, sample count, raw artifact path, and identity. Missing proof
is `UNKNOWN`, never zero or inferred success.

The record contains:

- exact checkpoint, tokenizer, prompt recipe, weight codec, compute/
  accumulation/KV precision, source commit, build/config hashes, and rank-pack
  hashes;
- hardware, device, rank-to-node map, TP/PP/EP/KVP topology, fabrics, clocks,
  power, and service exclusions;
- authoritative oracle, numerical tolerance, and current correctness gate;
- legal and rejected batch/context/output/KV cells with byte proofs;
- retained prefill, decode, TTFT, ITL, memory, power, capacity, and API
  controls;
- current no-drop profile, exposed-wall bottleneck, SOTA comparator, active
  hypotheses, queued jobs, rejected bites, and patches awaiting audit; and
- exactly one next decision with its falsifiable question.

## Parallel fronts

Start every front whose prerequisites are satisfied. Independent exploratory
fronts run concurrently; they do not imply that their results integrate.

| Front | Can run concurrently with | True dependency or integration gate |
| --- | --- | --- |
| Identity/oracle | Pack/capacity, fixture work, topology feasibility, profiling harness | Exact checkpoint, tokenizer, geometry, tensor inventory, and oracle lock all comparisons. |
| Pack/capacity | Identity/oracle, topology feasibility, KV/context byte model, API harness | Complete content-addressed packs and legal rank-byte cells gate real-weight runs. |
| Runtime correctness | Identity/oracle, pack fixtures, collective smoke tests, API harness | Real-weight operator and complete-layer parity gate full-model timing. |
| Collectives/topology | Identity/oracle, capacity, synthetic and real-message transport probes | All-rank pack/config identity, stage boundaries, failure propagation, and full-model parity gate scale claims. |
| KV/context | Pack/capacity, runtime correctness, batching/API harness | Per-rank KV ownership, resolution, admission, park/restore, and byte proof gate context cells. |
| Profiling/kernels/fusion | A retained B1 control, identity/oracle, transport probes | A micro or layer win is only a hypothesis until full-request anchors pass. |
| Batching/API | B1 runtime smoke, KV/context, pack/capacity | Lifecycle, admission, prefill/decode, streaming, cancellation, and overload gate serving claims. |
| Speculation/SOTA | No-spec control, profiling, batching/API, SOTA refresh | Output/parity, drafter cost, verification/rollback, capacity, and end-to-end anchor wins gate retention. |

Do not serialize the fronts merely to produce status. Do serialize their true
gates: identity before comparison; real-weight correctness before throughput;
capacity/topology/KV before scale; full-request anchors before retention; and
production audit plus clean-main rebuild before release.

## Integration gates

These are the only gates that stop a claim or promotion:

1. **G0 identity/oracle:** exact source, checkpoint, tokenizer, recipe,
   geometry, tensor inventory, and authoritative implementation are frozen.
2. **G1 pack/capacity:** every rank pack is complete, atomic, hash-identified,
   loader-validated, and legal cells have rank/KV/workspace byte proofs.
3. **G2 runtime correctness:** real-weight dense, attention/state, FFN/MoE,
   router, grouped GEMM, norm, residual, position, output head, and one full
   layer match the oracle; then deterministic no-spec B1 matches end to end.
4. **G3 topology/collectives/KV:** all ranks use identical pack/build/config;
   stage seams, collectives, slowest rank, KV ownership, and failure behavior
   pass on the declared topology.
5. **G4 serving frontier:** exact-32K `B1`, `B8`, and `B64` anchors are
   measured or explicitly rejected; context, chunked prefill, decode, API,
   batching, and supported speculation cells have receipts.
6. **G5 production:** a bounded change passes unchanged correctness, the 1%
   retain gate, protected-cell policy, and independent audit; merged main is
   pulled on every target, rebuilt/installed, restarted, and remeasured.

An unsupported or unmeasured cell is rejected explicitly. It never silently
falls back to B1, shorter context, another checkpoint, or another topology.

## Qualification staircase

Use the lowest unmet gate first; do not chase speed above an open correctness
gate.

1. Freeze identity and oracle (`MOD-*-001`).
2. Prove recipe, sharding, packs, topology feasibility, and capacity
   (`MOD-*-002/003/012`). Separate storage bits from compute and accumulation
   precision.
3. Validate real-weight components and one complete layer
   (`MOD-*-009/010/011/004`).
4. Establish deterministic no-spec full-model B1 (`MOD-*-005`).
5. Prove production topology, collectives, failure propagation, and KV
   ownership (`MOD-*-006`).
6. Establish context, chunked prefill, continuous decode, batching, and API
   cells (`MOD-*-013/014/015/016/007`).
7. Measure speculation only after a strong no-spec control.
8. Hill-climb the largest exposed wall and refresh the comparable SOTA
   control (`MOD-*-017`, `PERF-001..010`).
9. Execute G5 zero-drift release (`MOD-*-008`).

## Batch and context frontier

Use powers of two `B1,B2,...,B1024` and context anchors through 256K, but never
assume the Cartesian maximum fits. Separate prompt tokens, current context,
requested output, and parked KV.

The first comparable anchors are exact-32K `B1`, `B8`, and `B64`. Each cell
records admitted/rejected sequences, exact shape, output identity, failure,
prefill tok/s and TTFT excluding decode, aggregate and per-user decode tok/s,
ITL distribution, GPU/host/NVMe/KV/workspace bytes, power, warmups, randomized
sample order, raw samples, clocks, topology, precision, speculation, and the
timing boundary.

Explore adaptively:

1. Prove B1 at a diagnostic context, then exact 32K.
2. Prove exact-32K B8 and B64, or retain the exact blocker.
3. At fixed context, double B until rejection or throughput collapse; bisect
   the nearest useful cliff.
4. At retained B, double context until byte proof or measurement rejects it.
5. Fill B2/B4 and B16/B32 around latency/throughput knees. Fill B128-B1024
   only where memory and demand make them legal.
6. Re-run all protected anchors after every retained topology, KV, kernel,
   fusion, batching, or speculation change. Never extrapolate a rate.

## No-drop profile and hypothesis choice

Account for the whole request wall without double counting. Report GPU busy
union and exposed GPU idle; dense/GEMM, attention/state, MoE/router,
norm/activation, output, sampling, weight decode, KV, graph, fusion,
collectives and their non-overlapped portion, host scheduling/launch gaps,
allocation/copies/storage waits, API time, and overlap.

Select the largest exposed wall that can change the target cell:

- memory bandwidth: reduce bytes, improve layout/reuse, or cheaper codec decode;
- compute: improve tile/atom, occupancy, vectorization, grouped work, or
  precision route without lowering accuracy;
- launch/host: graphs, persistent work, batching, allocation removal, or a
  simpler queue;
- collective: fewer bytes, better sharding/algorithm/chunking, or real overlap;
- KV/storage: layout, resolution, prefetch, queue depth, or admission; and
- scheduler: batch formation, chunked prefill, fairness, or pipeline bubbles.

The next bite asks one falsifiable question and changes the smallest variable
that separates explanations. Declare its control, required data, success and
failure decisions, rollback, and Spark resources before dispatch.

## Topology, kernels, fusion, and speculation

Change one topology dimension at a time. For each candidate record rank bytes,
per-rank KV, collective bytes and algorithm/chunk, fabric route, overlap,
stage bubbles, slowest rank, and end-to-end scaling efficiency. Microbenchmark
real message sizes, but an isolated collective win remains a hypothesis.

Use production call sites and shapes for kernels. The kernel contract states
dimensions, strides, layouts, dtypes, accumulation, rounding, target, control,
and oracle. Fuse only adjacent exposed work. Estimate bytes and launches
removed; reject spills, occupancy loss, duplicated work, or changed rounding.
Promote a microkernel through layer, full-model, API, and anchor A/B gates.

Keep a no-spec control. For speculation measure drafter cost, proposal depth,
acceptance by position, verification, rollback/state-copy, bonus tokens,
capacity, latency, and final per-user plus aggregate throughput. Acceptance
rate alone is not a win. Greedy output stays exact; sampled verification must
preserve the target distribution.

## Retain or reject

Compare candidate and clean main with identical identities and at least three
randomized interleaved full-request pairs when variance permits.

Retain only if:

- the unchanged numerical gate passes;
- aggregate end-to-end gain is at least 1% on the target cell;
- no protected anchor materially regresses;
- memory, power, TTFT, ITL, and capacity stay within policy; and
- it is the simplest solution under `Solutions/(production LOC^2)`.

Exploratory probes need no independent audit. A production change gets a
bounded write set and independent audit. A rejected result records its tested
identity and becomes the control for the next bite.

## Per-turn decision algorithm

Run this loop on every Luna turn:

1. Load durable state and raw receipts. Mark stale, mixed, impossible, or
   missing fields `UNKNOWN`; do not repair claims by inference.
2. Find the lowest unmet integration gate. If one is open, choose the smallest
   real-path bite that can close it and stop performance work at that gate.
3. Otherwise compute the current exposed-wall ranking and the nearest useful
   unmeasured cell. Choose one falsifiable question, one primary variable, and
   one clean control.
4. Launch all conflict-free ready bites across the parallel fronts. Share no
   mutable service, pack, or output path without declaring the exclusion.
   `storage_io` may overlap `gpu`; service activation declares every resource
   it excludes.
5. Execute immediately through the assigned Spark runner. Capture raw output
   before summarizing it. A hardware or runner block is `BLOCKED_HARDWARE`,
   not a pass or zero.
6. For each result, classify `PASS`, `FAIL`, `REJECTED_CELL`, or
   `INCONCLUSIVE`; write the required receipt; update retained/rejected state.
7. Apply the failure-to-next-bite table. Enqueue every newly unblocked bite,
   including a negative-result follow-up, before releasing the queue slot.
8. If a production change passes the retain gate, freeze its bounded write
   set, audit it, merge, pull/rebuild/restart clean main, and repeat the
   protected anchors. Otherwise keep the current control.

## Required receipt fields

The raw run artifact comes first; receipt creation must not delay the next
experiment. A decision-grade receipt has these fields:

```text
receipt_id, model, phase, question, hypothesis, decision
checkpoint_id, tokenizer_id, recipe_id, prompt_id, oracle_id
source_commit, build_hash, config_hash, pack_manifest_hash, rank_pack_hashes
host_ids, device_ids, device_firmware, mount_or_storage_source
rank_map, TP, PP, EP, KVP, fabric, collective, clocks, power_limit
B, prompt_tokens, context_tokens, output_tokens, KV_format, precision
speculation, API_mode, control_id, candidate_id, changed_variable
command, runner_job_id, start_utc, end_utc, timing_boundary, warmups, samples
prefill_tps, decode_tps_aggregate, decode_tps_per_user, TTFT, ITL
gpu_bytes, KV_bytes_active, KV_bytes_parked, host_bytes, workspace_bytes
power_watts, exposed_wall_breakdown, output_hash, token_count, error_status
oracle_result, tolerance, result_class, retain_or_reject, next_bite
raw_artifact_path, receipt_path
```

`UNKNOWN` is valid for a not-yet-measured field; blank, guessed, or copied
values are not. A production receipt additionally names the bounded write set,
independent auditor, merged-main commit, rebuilt artifact hash, restart proof,
and post-restart anchor results.

## Failure to next bite

| Observed failure | Do not infer | Next bite |
| --- | --- | --- |
| Checkpoint, tokenizer, pack, or build hash differs | Performance or parity is comparable | Rebuild the exact control; rerun identity/oracle before any claim. |
| Oracle mismatch on a component | Tolerance or sampling is wrong | Reduce to the smallest operator/real-weight fixture; then one layer. |
| Missing, duplicate, or short rank pack | Loader or runtime is slow | Regenerate one rank pack, verify manifest/content hashes, then reload. |
| OOM or byte-capacity rejection | The next larger cell is close enough | Prove rank/KV/workspace bytes; test the nearest legal neighbor and reject the cell if needed. |
| Invalid launch only at B>1 | Long context or a kernel regression | Reproduce B1 then B2 with the same weights; capture rows, routes, scales, and launch geometry. |
| Collective timeout, skew, or rank disagreement | The model math is broken | Run real-size all-rank smoke; isolate rank, fabric, algorithm, or chunk before changing model code. |
| KV/context mismatch or park/restore failure | More context is supported | Run B1 with exact KV byte/ownership checks, then the next context boundary. |
| API timeout, retry cascade, duplicate, or global request failure | Decode is slow | Run one client and one B1 request; check lifecycle, deadlines, prefill chunking, and daemon exclusivity. |
| Variance hides the sign | A noisy win is a win | Interleave control/candidate pairs, preserve clocks and order, and repeat the same bite. |
| End-to-end gain is below 1% or an anchor regresses | An isolated micro win should ship | Record the negative result; return to the largest exposed wall or nearest unmeasured cell. |
| Speculation parity, rollback, capacity, or latency fails | Acceptance rate is sufficient | Return to no-spec; inspect first rejection and state-copy cost at depth one. |
| Hardware or runner unavailable | Zero throughput or correctness | Mark `BLOCKED_HARDWARE`, preserve the command and identity, and dispatch the next independent front. |

## Model examples

### DSV4 Flash

Freeze the GA checkpoint, tokenizer, and DSV4 Flash oracle independently of
DSV4 Pro. Use TP4 no-spec B1 as the control, then exact-32K B1/B8/B64. In
parallel, profile exposed collective/idle time and measure the existing DSpark
speculation path. A DSpark bite records proposal/verification/rollback cost and
output hash; retain it only when the no-spec numerical gate and all protected
anchors pass with at least 1% end-to-end gain. If the exposed wall is
collective-shaped, send the next bite to collectives/topology rather than
adding fusion code.

### GLM 5.2

The first integration target is the exact real stagepack and complete-layer
oracle; a synthetic validator is not readiness. Run pack ownership, dense/
attention/MoE correctness, and API harness probes concurrently once identity is
frozen. After real-weight B1 parity, measure batching and the single-stream
bandwidth wall before writing kernels. Speculation is a measurement only when
the exact checkpoint supplies a drafter/MTP path; otherwise record the blocker
and take the batching bite.

### Q27 (Qwen 3.8 27B)

Freeze Qwen 3.8 27B separately from Qwen 3.8 Max and Qwen 3.6. Establish the
one-Spark B1 control, then B2/B4/B8/B16 and exact-32K B1/B8/B64 wherever the
topology and KV bytes are legal. If a B16 run produces deadline/retry or
`NOT_FOUND` cascades, reproduce with one client and an exclusive clean daemon;
test lane-major prefill chunking, deadlines, and request lifecycle before
touching kernels. A clean B1/B4 result plus a failed B16 is a batching/runtime
next bite, not evidence of a Q27 math failure.

The foreman's output is measured answers, useful experiments, and retained
simple solutions—not turns, requests, documents, or lines.
