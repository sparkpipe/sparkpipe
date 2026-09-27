# SparkPipe Technical Debt

This file contains only unfinished work against
[`ARCHITECTURE.md`](ARCHITECTURE.md) and the system described in
[`README.md`](README.md). Completed work is removed rather than retained as a
progress diary.

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

## Mesh collectives

- Every hardware-wait round crosses weightd's CPU relay twice: the local
  doorbell sweep posts the RDMA writes and the remote sweep completes the
  waiter. Replace it with GPU-initiated RDMA so kernels post work requests
  and ring the NIC doorbell themselves; a 16-rank 8 KiB all-reduce should
  then cost tens of microseconds rather than the measured 167 us p50.
- Two collective substrates coexist: the residentd-owned hidden transport
  (`ring/transport/tp_collective.c`, recursive doubling and split rings),
  which k3's runner still creates, and the weightd mesh through
  `ring/transport/tp_device_collective.c`, which every other TP module
  opens. Converge on one substrate with one algorithm selector and one
  measured crossover profile.
- The mesh reads only degree, rank, width, rows, the operation timeout, the
  band, the lane and the host combines from `SparkTpDeviceCollectiveConfig`.
  The rest of that struct (hosts, ports, identifiers, credits and bindings,
  rails, algorithm masks and thresholds), the stub calls around it
  (`ApplyTopology` copies the degree, `ProbeMemoryMode`,
  `CreditBindingRouteCount`) and the adapters' `tp_collective` stage-config
  section that feeds them predate the mesh. Remove them together with a
  fleet stage-config migration. k3's relay and TP4 combine callbacks serve
  the same pre-mesh algorithms.
- The weightd mesh is wired on one interface (the switched rail). Build the
  pair-link hierarchical all-reduce (pair sum over `rank XOR 1`, 8-way
  switched exchange, pair return) designed in
  `docs/GLM5_NEXT_ROOFLINE.md` and select it for prefill and large batches.
- Overlap communication with compute: two micro-batches in flight per
  engine, one computing while the other's collectives complete. Measure the
  extra weight reads that splitting costs against the exposed wait it hides.

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
- glm5_next: the graph path waits for the whole graph on residentd's thread
  (`SparkGlm5NextGraphStep` through `SparkStageModuleCudaWaitFor`) before it
  finishes the chain, and the completion worker then waits on the same stream
  again. Move the stuck-graph timeout and the error checks into the completion
  path so residentd keeps serving its socket during the replay.
- glm5_next: prefill chunks never run as a graph, and a decode wave that
  arrives behind a chunk waits for the chunk's whole GPU time
  (`decode_wait_ms` in `G5N-WAVE-TIMING`). Capture prefill graphs per chunk
  shape, or order decode waves ahead of queued prefill chunks.
- glm5_next: the chain state machine still synchronizes the stream after
  every collective round and twice more per routed layer (expert lease, then
  release). It runs only where a linear chain cannot: experts not pinned,
  MTP, speculative verify, the T1 trace, `SPARK_GLM5_NEXT_GRAPH_RECORD_OPS`,
  or a collective without hardware-wait rounds. Pin experts wherever the
  arena fits them, and move MTP and speculative verify onto the linear walk.
- glm5_next: between two waves every rank ends and restarts mesh activity
  (two weightd round trips each way), rank 0 broadcasts a new chain epoch that
  the other ranks spin on, and two host callbacks run. Resident decode chains
  pay this once per K decode steps instead of once per step; prefill chunks,
  publish frames and single-step waves still pay it every time. Keep the
  activity and the epoch across the waves of a session.
- glm5_next: a resident decode chain feeds each step's tokens back through the
  host. Between steps it synchronizes the stream, copies the sampled tokens
  into the pinned input arrays and advances the positions. On the linear path
  that wait runs on residentd's thread, which a single-step linear chain never
  blocks. Feed the next step on the device (copy the head's tokens into the
  input rows and advance the positions in a kernel) so K steps are one enqueue
  or one graph.
- glm5_next: the engine ends a chain at the nearest block end of any lane in
  the batch, so chains shorten as the batch grows: a mean of about 5.4 steps
  at 8 lanes against 8 at one lane, by simulation of random lane offsets. Each
  lane's block end is also still a separate publish frame. Publishing block
  checkpoints from inside a chain would keep chains at 8 steps and remove
  those frames.
- glm5_next: a decode chain holds the rank for all of its steps, so a prefill
  chunk that arrives mid-chain waits up to 8 decode steps. The engine picks K
  without looking at queued prefill; shorten chains while prefill waits if
  time to first token suffers.
- glm5_next: the exact DSA top-k (`LmTopkExactKernel`) runs four radix passes
  and a block-scan compaction in one CTA per row. At 32K context a row has
  8,192 pools, and the kernel's time there is unmeasured. If it shows in
  `run_us` at long context, compact with warp ballots instead of the
  Hillis-Steele scan.

## Resident TP4 x PP4 execution

- GLM 5.3 Flash source audit at main `371ae9e`: the TP4xPP4 JSON generator
  exists, but the shipped serving adapter hard-codes TP16, rejects other TP
  degrees, requires `tp_rank == stage_index`, and declares parallel fanout.
  Its node context starts at layer zero and the firmware defaults to all 45
  layers. Complete the hybrid runtime topology, stage-local layer spans and
  ownership, boundary forwarding, and grouped collectives before calling the
  generated TP4xPP4 deployment runnable. Reuse the shared
  `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP` path in
  `runtime/model_serving_adapter.c`; changing the collective degree alone is
  insufficient. Prove token parity through all four pipeline stages.
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

## KV sharding

- MLA/DSA models store the full shared latent KV and indexer keys on every
  TP rank. GLM 5.3 Flash at TP16 holds 16.5 KiB per token per rank of
  replicated DSA state; only its KDA recurrent state is head-sharded.
- Context-parallel indexer scoring (`dsa_index_context_parallel`) splits the
  indexer reads exactly but is opt-in and off on the fleet, and it does not
  shard storage. Qualify it on long prompts and make it the default.
- Build context-parallel latent attention: each rank stores and attends
  over its own tokens and the head owners merge partial softmax states. The
  query all-gather and partial exchange (about 64 KiB each per row per DSA
  layer) exceed the current 256 KiB slot budget at B64; land it after
  GPU-initiated RDMA or as a 4 head-group x 4 context-shard split.
- Replace per-driver KV and index pools with one node-level pool shared by
  all resident drivers, admitted against resident demand.

## Dynamic batching

- GLM Flash's shared batch scheduler already selects arbitrary counts up to
  its configured limit; a power-of-two kernel bucket is not an admission rule.
  Both GLM deployment generators currently set active/resident capacity to 16.
  Qualify explicit larger capacities with arrivals/completions and memory gates.
  The current module reserves every sequence's full context in its internal KV
  and index pools. At 32K context its allocation formulas reserve 10.685 GiB
  for 16 sequences or 66.779 GiB for 100, per full-model rank, including KDA
  state/windows but excluding weights, workspaces, transport and metadata.
  These are code-derived sizes, not live memory measurements. See
  `SparkGlm5NextBuildPageTable` and the cache allocation code in
  `spark_glm5_next_resident_decode_stage_module.c`. Reuse the shared paged-cache
  contracts to admit against resident demand; do not merely raise the limit.
- The KV page cache keeps every published block until an allocation finds the
  logical pool full, then evicts one. It picks the victim by scanning every
  entry, so once the pool is full each new page costs a pass over the whole
  table. Keep unreferenced entries on an LRU list instead. Nothing reports how
  full the pool is either; add used pages, retained entries and evictions to
  the wave timeline.
- Publish one logical resident model driver with prewarmed B1-B1024
  specializations rather than batch-specific resident identities.
- Select the smallest validated specialization for effective rows, including
  speculative verification rows, while preserving sequence and KV identity.
- Qualify mixed arrivals, priorities, prompt lengths, shared prefixes, cache
  pressure, cancellation, and starvation bounds.
- Make priority and deadline enforcement span admission, prefill, decode,
  speculation, gang scheduling, model promotion, and storage I/O.
- Dense projections above eight rows leave the skinny kernel for
  tensor-core GEMM with a different accumulation order, and batched latent
  attention sums in a different order from the per-head kernel, so prefill
  and batched-decode logits are not bitwise equal to B1. Extend the
  row-blocked kernels that keep the skinny order wherever they match the
  tensor-core throughput.
- Provider-network replay verification needs batched rows to be bitwise
  equal to B1: a verifier replays a request alone and must reproduce what a
  provider served at B8 or B256. TensorFold (MIT, `ashhart/TensorFold`) shows
  the rules: the reduction split is fixed by the weight's shape and never by
  the row count, attention tiles are fixed by absolute key position with an
  fp32 online softmax merged in key order, router ties break by token id, and
  cross-GPU partial sums are added in rank order. It also compares multi-row
  with one-row output at load and turns drafting off when they differ. Adopt
  the same rules, and make that load-time comparison the gate for verified
  serving.
- A prefix-cache hit reuses KV and KDA state computed however the source
  request ran: one-row prefill for prompt tokens, batched decode rows for
  generated ones. Until batched rows equal B1, a warm and a cold run of the
  same prompt can differ (#1230). Make them equal, or keep checkpoints of
  generated tokens out of reuse for verified requests.
- glm5_next: a row's attention still depends on the other rows in its wave.
  The wave's longest row decides split-KV for every row, and a row at or
  below 2,048 tokens, in a wave whose longest row is past 2,048, attends
  through the 2,051-slot selected list (all its complete pools, then its
  tail) instead of densely. Either way its partitions and summation order
  differ from a one-row wave's. Graph and eager waves now agree with each
  other, but not with the row run alone. Attention tiles fixed by absolute
  key position (the TensorFold rule above) remove this.
- glm5_next: no GPU test compares a replayed decode graph's logits with an
  eager wave's at the same context. Host tests cover the choices (split-KV,
  DSA selection, pool expansion) and run the selection kernels. Add the
  logits comparison to the CUDA validation at contexts 63, 64, 2,048, 2,049
  and 4,099.

## Model contracts

- Add exact checkpoint-derived contracts and native execution packages for
  MiniMax H3, Qwen 3.8 Pro, and Qwen 3.8 27B.
- Remove legacy model names from generated release inventories and operator
  surfaces when their replacement contracts land.
- Retain independent numerical, transport, memory, and performance gates for
  every model and precision route.
- Complete and qualify the K3 BF16-activation/MXFP4-weight asymmetric GEMM,
  route gather, in-load E8M0 decode, and full expert-path comparison.
- Bind GLM 5.2 dense gate, up, down, and router-logit tensor-core linear plans
  at startup before required-stage validation.

## Speculation

- The GLM 5.3 Flash MTP draft layer keeps a one-page KV per execution slot
  and attends only within its current draft chain, not the sequence
  history. Give it per-sequence draft KV and measure the acceptance change.
- Wire tree verification (`spark_speculation_tree.h`) into the common
  acceptance engine; it currently resolves chains only.
- Build the tournament provider over the provider slot, with per-drafter
  acceptance telemetry, after the single-drafter agreement matrix is
  measured.
- glm5_next resident decode chains run no MTP draft: a frame of more than
  one step skips `SparkGlm5NextMtpDriveDraft`, and the engine asks for chains
  whenever the adapter offers them, so with MTP enabled drafts only run on
  single-step frames. Choose between MTP and chains per batch in the engine,
  or verify drafts inside a chain.
- glm5_next has no DFlash2 source. Its hidden-tap capture (#931, `e60f690b`:
  HC-mean of the post-MLP residual at global layers 5/14/24/33/42 into a
  per-lane host ring) left the module with the whole-step graph engine
  (`096f45ed`). The ring, its gate and the adapter's DFlash2 offer outlived
  it; they are gone too, so a DFlash2 request is refused as unavailable.
  Restore the capture inside the graph engine, with the ring and gate from
  `e60f690b`, when a GLM 5.3 Flash DFlash2 drafter is to be qualified.

## Packaging and provenance

- Add the upstream implementation commit to stage-pack provenance at the next
  pack format revision.
- Publish one synchronized pack-environment manifest for Python, CUDA, and
  model conversion dependencies.
- Generate compact deployment specifications for every released model package.
- Replace the 33 per-family `tools/*stagepack*.py` packers with the
  universal packer (`docs/UNIVERSAL_PACKER.md`): one CLI, one codec table,
  per-family byte-compatible emitters, each gated on byte identity with its
  existing packs.

## Driver consolidation

- Adopt the common parameterized modules in
  `docs/COMMON_MODULE_ARCHITECTURE.md` and delete the near-copy code they
  replace (measured at about 26,000 lines across the families), each
  migration proved by byte or behaviour identity.
- Publish one driver per model with prewarmed row-count specializations
  instead of one module ID per `SPARK_BATCH_BUCKET`.
- Every decode module now builds for sm_121a and links as a driver in
  `tools/cuda13_sm121a_compile_gate.sh`, and every TP module opens the
  weightd mesh in `tests/test_tp_collective_open.py`. Most of them have
  changed since they last ran on hardware. Requalify each of them:
  - qwen4_flash now runs the common GDN decay kernel with its sharded-pack
    layout (`SPARK_LLM_GDN_DECAY_REPLICATED 0`), and the common expert
    launchers, which now also take NVFP4;
  - hy4 links the stage-module lifecycle it calls, and did not link before;
  - dsv4, muse_glimmer and qwen4_flash compile the common mesh kernels that
    `tp_device_collective.c` calls, and did not link before;
  - dsv4 could not open its TP collective at all (a local-host check the
    mesh rewrite left unsatisfiable), and dsv4 and muse_glimmer never
    attached the mesh, so every TP round would have been refused;
  - glm52, laguna, qwen38_max and qwen4_flash attached the mesh only with a
    lazy pack, and now map it themselves without one;
  - dsv4, minimax, muse_glimmer, qwen38_max, qwen4_flash and qwen38_27b now
    reduce through the common combines, which sum all ranks in FP32 and
    round once, where their private kernels summed rank by rank in BF16.
    Record a precision receipt with each requalification.
- glm5_next still carries host code its driver never reaches: the per-layer
  attention graph wrapper `Glm5NextLayerAttentionBf16Graphed`, the
  `LayerAttentionBf16` entry in
  `family/glm/spark_glm_unity_glm5_next_ling.cuh` (ling carries it too) and
  `SparkGlm5NextLaunchEpochSample` with its kernel. Deleting them leaves
  every other kernel's SASS identical but changes GCC's inlining in four
  live host launch functions, so it waits for a measured glm5_next
  deployment.
- dsv4's `.cu` carries launchers that no dsv4 driver build calls. `QuantSim`,
  `Hadamard`, `HeadArgmax` and `HeadScreenedArgmax` are the GPU validator's
  unfused reference paths for fused kernels the driver runs: move them into
  the validator's translation unit. `QueryHeadRmsRope` is the query fusion
  the B1 stack rejected, still validated; `ExpertUp`, `HcSplitSinkhorn`,
  `MoePairReduceStrided`, `SwigluClamp` and `HeadScreenedArgmaxSharded` are
  kept only by source tests. Delete those with their checks.
- Pack synthesizers for dsv41_flash, gemma4 and muse_glimmer do not use
  `spark_pack_synthesize_common.h`, and the dsv4 and k3 batch-tuning headers
  keep their own bucket ladders. `tests/test_template_adoption.py` lists
  them.

## Runtime completion

- Add bounded cancellation and drain for terminal client I/O failures so every
  resident sequence slot is released.
- A failed residentd route whose driver cache abort also fails stops the
  residentd so its unit restarts, because the driver's transaction state for
  those slots is unknown. Give the driver a per-slot reset so one slot can be
  recovered without restarting the unit.
- The serving completion ABI accepts statuses up to UNSUPPORTED, but weightd
  can fail a glm5_next expert lease with NO_LANE or EVICT_DENIED, and the
  chain passes that status up. residentd now names it
  (`COMPLETION-REJECTED ... completion_status=20`) and fails the route with
  invalid_argument. Map those statuses in the adapter, to BUSY where the step
  can be retried without having advanced any recurrent state, otherwise to
  CAPACITY_EXCEEDED.
- ROUTE-STUCK reports a route once per state. Repeat it on an interval with
  the route's committed-FIFO position and slot claims.
- Pipeline-parallel stages still wedge after a failure on another rank. When
  one rank fails a submission's COMMIT or frame, the next stage's route has
  already posted its hidden-transport receive and waits in WAIT_INPUT for data
  that never comes. Only a deadline moves it, and the engine sets none, so the
  route keeps its slot claim and blocks the reset the reconnect needs. The
  in-tree test transport completes receives without a peer, so no test sees
  this. On client-generation change, cancel abandoned WAIT_INPUT/WAIT_OUTPUT
  routes through the transport before releasing their boundary buffers, and
  add a test transport that needs a real peer. Tensor-parallel deployments
  (glm5_next TP16) post no inter-stage receives and are not affected.
- A failed route that the driver had already run (for example a PP output
  send that failed) releases its slot claims but leaves residentd's `bound`
  flag as it was, so the driver and residentd can disagree about the slot
  until the next session reset. Settle the slot the way an error completion
  does.
- Produce one immutable qualification bundle containing merged commit, release
  generation, package and driver hashes, all-rank identities, token stream,
  accuracy, performance, route counters, and drained queue state.

## Serving API

- Sampling is temperature-only and only glm5_next implements it; other
  adapters answer `400 sampling_unsupported`. Add top-k/top-p and logprobs,
  which need a cross-rank log-sum-exp, and port the sampled head
  (`LmHeadSampledCandidate*Kernel`) to the other families.
- Sampled rows take the full-vocab BF16 head instead of the certified FP8
  B1 head, run eagerly instead of replaying a CUDA graph, and get no MTP
  drafts: the certified screen prunes with un-noised bounds, graphs freeze
  the head choice, and drafts are keyed by relative positions. Noise the
  screen bounds, key drafts by absolute position, and capture sampled
  graphs to lift all three.
- Draw sampled tokens as the argmax of logit/T plus Gumbel noise keyed by
  (seed, absolute position, token id), as TensorFold's exact sampler does, so
  a seeded sampled stream replays exactly and a draft is accepted exactly
  when it equals the serial draw.
- Carry `deadline_ms` into the batch engine and the serving submission
  (`deadline_time_ns` exists but is not populated) so the scheduler, not
  only the API, orders work by deadline.
- The engine's position limit comes from the deployment's optional
  `runtime_limits.max_sequence_positions`, while each adapter enforces its
  own stage config's `max_sequence_positions`. Nothing compares them, so a
  deployment without the member, or with a larger value, still fails
  mid-decode instead of at admission. Report the adapter's limit in the
  residentd hello and let the engine take the smallest across ranks.
- Positions are sized for every resident sequence at full length
  (`tools/spark_serving_profile.py`: B8 is 8 × 512 positions in 1,024
  pages). A request cannot use the pages its neighbours leave idle. Size
  positions for one long sequence and let paged admission share the pool.

## Production qualification

- Repeat accepted transport and model measurements from clean merged `main`,
  rebuild the exact release on Spark hardware, and retain all receipts.
- Close exact-checkpoint numerical parity and end-to-end service gates for each
  model before reporting it production-ready.
- Nine Python tests stay outside `make test`:
  - five drive the fleet over ssh: `test_expert_io_perf`,
    `test_jit_kv_page_fault`, `test_lossless_doorbell`,
    `test_memory_bandwidth_budget` and `test_transport_stability`;
  - three read whole synthetic packs into memory and fail with
    `MemoryError` on a 7 GB host: `test_dsv41_flash_layer0_anchor`,
    `test_dsv41_verify_pack` and `test_glm52_validate_pack_stage`;
  - `test_stagepack_mtp_strip_qwen36sp` fails removing its temporary
    directory in a sandbox.

  Give the fleet tests a runner, stream the packs, and register all nine.

## Hardware independence

- The device layer exists only as `SparkMemoryBuffer`
  (`include/sparkpipe/spark_memory_buffer.h`). Build the rest of the memory
  interface (register, map_file, make_resident/evict) and then streams,
  events and launch, per `docs/INFERENCE_OS_DESIGN.md`.
- Modules call the CUDA runtime directly. Move them behind the device layer
  so a module target can be `host.*`, `metal.*` or `rocm.*`.
- Add rank, island and link-class descriptors and derive collective
  decomposition from the link matrix instead of a configured backend.
- Qualify the Metal backend on the Mac Studios and keep the host backend as
  a CI oracle for common policy, not only for kernels.

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

## Multi-model serving (parallel inference on shared Sparks)

Assessed 2026-09-13 against the lane/mesh/weightd architecture (PRs
#959-#973). Handled today: per-model lanes (weightd-assigned, fail-closed),
weight arenas shared by content identity (same model never loads twice),
per-connection collectives (two engines of one model get distinct mesh
pathways), swap-in loads that run daemon-side (a serving model does not
stop while another loads), and per-engine continuous batching at
MAXBATCHSIZE=128 with per-request completion. The gaps below are required
for seamless production multi-model.

- Lane-priority over-subscription policy: eviction is LRU plus epoch with
  no lane awareness. Required rule: lane X may swap in only by evicting
  weights whose owning lane is greater than X; when the evictable set
  belongs to a lane currently executing, wait for its batch boundary, then
  evict; when nothing greater than X is resident, the acquire errors
  rather than thrashes.
- Request queueing behind not-yet-live models: lane exhaustion fails the
  residentd load closed and requests for that model see connection
  refused at the router. Required: warm request queue that drains when the
  model finishes loading (swap starts, requester waits, serving model
  continues).
- Load bandwidth fairness: swap-in reads run at full readahead with no
  QoS; the serving model's page-cache and mesh traffic compete
  unbounded. Required: a fair-share cap on loader throughput while any
  lane is serving.
- Output chunking for GPU fairness: decode chains hold the GPU for whole
  tokens and cross-model sharing relies on driver time-slicing (no MPS).
  Required: bounded output quanta analogous to prefill chunks so a lane
  cannot lock the GPU for many tokens.
- Pause and resume of output batches: token-boundary resume is structural
  (sequence positions, prefix cache) but there is no pause/resume API, the
  NVMe KV tier is disabled (kv_backing_maximum_bytes=0), and active-KV
  protection is transaction-scoped rather than batch-scoped. Required:
  resume without recomputation, with boundary-flushed KV protected from
  eviction until the batch completes.
- Same-model multi-engine has never been exercised end-to-end: two
  residentds on one model (shared arena via identity match, distinct
  lanes per connection) needs a live verification run, including the
  MTP-debug use case.
- Fleet tooling is single-model: the agent accepts multiple runtime roots
  but release sync, health, and measurement lanes are per-root; no
  multi-model deploy or update has been tested.
- Deployed lane count is two; the eight-lane geometry is blocked on the
  engine-side cudaHostRegister invalid-argument at the 4 GB mapping (lane
  handoff Addendum 54).

## Topology-aware lane sub-allocation

- LANE_ACQUIRE today hands one whole lane (two mesh bands, sixteen rank
  slots) to one engine. Extend the acquire to carry topology and rank
  range: a residentd states TP degree plus the contiguous rank range it
  occupies, and the allocator packs sub-ranges into bands (4xTP4, 2xTP8,
  TP8+2xTP4 per lane), constraining TP8 ranges to start at rank 0 or 8.
- Doorbell entries, mesh slots, and per-round ring positions are already
  rank-indexed within a band, so sub-range packing is an allocator change
  plus collective band/rank wiring; four TP4 drivers in one band must not
  share sequence spaces (derive chain keys per sub-range owner).
- Lane count verification: two lanes (1 GB page) proven; four lanes is one
  define change and untested; the eight-lane attempt fails engine-side
  cudaHostRegister with invalid argument on the 4 GB mapping while the
  same registration shape succeeds standalone — isolate the in-engine
  condition before assuming a size ceiling.

## MPS evaluation on GB10

- The CUDA MPS control and server binaries are present on the sparks.
  Run a live evaluation: start the control daemon, run two CUDA
  processes concurrently, confirm overlapping kernel execution and
  per-process contexts. If GB10 supports MPS, cross-driver GPU
  concurrency replaces driver time-slicing and the output-chunking
  requirement shrinks to memory-bandwidth fairness.
