# SparkPipe Technical Debt

This file contains only unfinished work against
[`ARCHITECTURE.md`](ARCHITECTURE.md). Completed work is removed rather than
retained as a progress diary.

## Dual-fabric topology contract

- Replace the legacy ring/single-switch/dual-switch topology modes with one
  schema that represents the CRS804 rail plus eight pairwise direct links.
- Generate supported four-, eight-, and sixteen-Spark profiles from the same
  schema, requiring complete direct pairs at every size.
- Generate pinned interface, address, direct-partner, and communicator tables
  for all sixteen ranks from that schema.
- Remove obsolete topology examples and release switches after all consumers
  use the combined-fabric contract.

## Adaptive all-reduce

- Land topology-aware recursive halving/doubling as the medium-payload mode.
- Generate independent two-crossover profiles for TP8 and TP16, for every
  production datatype and relevant concurrent-collective pressure level.
- Prove route and byte balance with interface counters for every rail and
  direction, including failure diagnostics down to rank, rail, phase, stripe,
  and chunk.
- Remove remaining per-chunk CPU dispatch, whole-tensor barriers, and any
  progress path that can serialize one rail behind the other.
- Replay collective correctness and performance from clean merged `main` and
  retain the profile as a release artifact.

## TP collective control plane

- Replace the per-collective host callback/submission chain with one
  model-neutral, predeclared collective program per resident slot. A token must
  have exactly one control plane: never layer mapped graph semaphores over the
  callback chain.
- Pre-register complete send and receive slabs, build packet and work-request
  templates at initialization, and pre-arm a rolling credit window before the
  producing kernel runs. No memory registration, packet rebuilding, or
  fixed-capacity table scan belongs in the steady-state token path.
- Advance the immutable program from transport completions and publish one
  terminal completion per token rather than one callback per collective.
- Separate receive readiness from send-buffer reuse so local reduction can
  begin when all receive completions arrive without waiting for unrelated send
  completions.
- Use one program catalog for B1-B1024. Select the collective algorithm from TP
  degree, datatype, effective row count, payload bytes, and the measured
  hardware profile without changing the model driver or resident weights.
- Remove the graph-island controller only after the replacement produces exact
  tokens and beats its retained merged-main B1 and saturated-batch receipts.

## Steady-state decode hot-path audit

- Replace completion-queue polling followed by fixed 64-entry send, striped
  completion, and receive scans with work-completion-indexed ready queues. The
  current TP4 B1 path spans six transport sessions per rank and repeats those
  scans throughout every collective.
- Collapse the current TP4 B1 accounting of 389 payload sends, 389 credit-return
  sends, and 778 receive reposts per rank per token. Across TP4 that is 6,224
  verbs posts plus matching completions for only about 3 MiB of payload per
  rank. Piggyback credits and reuse prebuilt work requests rather than paying a
  second message stream for buffer ownership.
- Replace six directional session/QP control objects per rank with one
  bidirectional peer connection per route and one completion context per rail.
  RC queue pairs are bidirectional; direction-specific state must not duplicate
  connection setup, polling, packet construction, or credit bookkeeping.
- Build immutable packet fields and receive templates once. The current path
  rebuilds packet metadata on both sides, including receive packets that are not
  consumed by the data plane, and performs repeated string comparisons in
  steady-state progress.
- Remove per-poll timeout clocks, disabled-profile array clears, and exact-length
  memory-region lookup from the steady-state path. These belong in admission,
  setup, a completion-driven timer wheel, or a slab registry.
- Predicate DSV4 compressor emission before RMSNorm, RoPE, Hadamard, quantize,
  and scatter work. A non-boundary token currently launches work for zero
  emission; static schedule accounting identifies about 221 useless compressor
  post launches per average token.
- Replace the approximately 780 CUDA event record/wait operations per token with
  dependency edges at true data hazards. In particular, KV post-processing and
  query projection must not inherit unrelated attention/projection barriers.
- Construct deterministic attention indices once per token or position range,
  not once per each of 43 layers. Remove repeated Hc residual copies and the
  other tiny host/device transfers only after bitwise output comparison.
- Remove batch- and topology-identity fallbacks. Unsupported topology values
  must fail compilation, and B1-B1024 must share one runtime descriptor plus a
  specialization cache rather than silently selecting PP13 or loading a
  different resident driver.
- Accept each removal independently: exact token parity first, then at least
  three unprofiled end-to-end cached-prefill B1 runs. Do not stack candidates
  until the preceding candidate beats the retained 33.6647 tok/s floor.

## Resident TP4 x PP4 execution

- Generate the sixteen-rank TP4 x PP4 deployment directly from the final
  hardware and model contracts.
- Keep stage-local weights, KV, communicators, graphs, and workspaces stable
  while switching B1-B1024 execution widths and speculation policy.
- Complete shared-prefix B8 scheduling with asynchronous DSpark proposal work
  and starvation bounds.
- Complete throughput scheduling that forms dense stage-local microbatches from
  a larger dynamic agent population without request-wide lockstep.
- Add bounded gang scheduling for a co-resident TP16 dense model without
  communicator ordering hazards or network contention.

## Model residency and storage

- Implement one catalog that keeps every configured frontier model addressable
  while tracking resident, warm, promotable, and unavailable states.
- Partition and mount each 4 TB internal NVMe as 2.5 TB hot KV, 1 TB active
  model shards, and 0.5 TB system/runtime space with startup validation.
- Reserve at least 1 TB of every external NVMe for direct rank-local model
  access and combine the remaining capacity into the selected striped,
  failure-aware model-data pool.
- Measure and close at least 20 Gb/s useful reads from the pooled model store
  for complete rank-shard promotion workloads.
- Implement atomic model promotion, prewarm, and publication in at most 60
  seconds without disrupting unrelated resident requests.
- Preserve resumable KV and request ownership across model eviction and
  reactivation, subject to explicit capacity and retention policy.

## Dynamic batching

- Publish one logical resident model driver with prewarmed B1-B1024
  specializations rather than batch-specific resident identities.
- Select the smallest validated specialization for effective rows, including
  speculative verification rows, while preserving sequence and KV identity.
- Qualify mixed arrivals, priorities, prompt lengths, shared prefixes, cache
  pressure, cancellation, and starvation bounds.
- Make priority and deadline enforcement span admission, prefill, decode,
  speculation, gang scheduling, model promotion, and storage I/O.

## Model contracts

- Add exact checkpoint-derived contracts and native execution packages for
  MiniMax H3, Qwen 3.8 Max, and Qwen 3.8 27B.
- Remove legacy model names from generated release inventories and operator
  surfaces when their replacement contracts land.
- Retain independent numerical, transport, memory, and performance gates for
  every model and precision route.
- Complete and qualify the K3 BF16-activation/MXFP4-weight asymmetric GEMM,
  route gather, in-load E8M0 decode, and full expert-path comparison.
- Bind GLM 5.2 dense gate, up, down, and router-logit tensor-core linear plans
  at startup before required-stage validation.

## Packaging and provenance

- Add the upstream implementation commit to stage-pack provenance at the next
  pack format revision.
- Publish one synchronized pack-environment manifest for Python, CUDA, and
  model conversion dependencies.
- Generate compact deployment specifications for every released model package.

## Runtime completion

- Add bounded cancellation and drain for terminal client I/O failures so every
  resident sequence slot is released.
- Produce one immutable qualification bundle containing merged commit, release
  generation, package and driver hashes, all-rank identities, token stream,
  accuracy, performance, route counters, and drained queue state.

## Production qualification

- Repeat accepted transport and model measurements from clean merged `main`,
  rebuild the exact release on Spark hardware, and retain all receipts.
- Close exact-checkpoint numerical parity and end-to-end service gates for each
  model before reporting it production-ready.

## DGX Station deployment

- Define exact 1x, 2x, 4x, and 8x Station hardware profiles, including memory
  bandwidth, interconnect topology, power envelope, and storage.
- Implement standalone placement and the mixed Station-plus-Spark execution
  plan for every supported Station count without introducing model-specific
  runtime branches.
- Generate and calibrate Station collective profiles from exact model payloads
  for each supported Station count.
- Measure each Station-count profile against the selected DGX B300 comparison
  workload and close the roughly one-half-throughput objective for the
  four- and eight-Station largest-model workloads.
- Validate office power, cooling, startup, failure recovery, and service
  operations as part of the deployment receipt.

## Incremental expansion

- Automate expansion from 4 to 8 to 16 Sparks while preserving package
  identities, catalog state, priority policy, and resumable request metadata.
- Generate model placement and storage rebalance plans before nodes join the
  ready set; never improvise redistribution in the request path.
- Support adding a Station as standalone capacity or as an explicit Spark
  fabric enhancement under the same API and scheduler.
- Retain upgrade and rollback receipts so a failed expansion returns to the
  prior ready deployment without mixed topology state.
