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
  logical pool full, then evicts the least recently used unreferenced entry
  from the head of its LRU list. Making a page resident when the resident pool
  is full still scans every resident slot for a victim. Nothing reports how
  full the pool is; add used pages, retained entries and evictions to the wave
  timeline.
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
- The GPU validator digest (`SPARK_<FAMILY>_CUDA_VALIDATOR_SHA256`, computed
  in each module Makefile, validate script and publish wrapper) hashes only
  the validator's `.cu`. The validators include templates from
  `include/sparkpipe/family/validation/`, so a template edit changes what
  validates a pack without changing the digest its receipt records. Hash the
  `.cu` together with the templates it includes, in one helper every pin
  calls.

## Driver consolidation

- Move the model-lineage code still filed under shared paths into its family
  and rename driver-named family templates after what they do. The files are
  the `PENDING` list in `tests/test_dry_law.py`: `common/common_glm_cuda_tree`,
  `common/common_glm_stage_module`, `common/common_kv_geometry.h`,
  `common/glm_resident_stage_wrapper.mk`, and the dspark drafter and qwen38
  serving-adapter headers in `model-families/common`. Family templates under
  `include/sparkpipe/family/` named after drivers (for example
  `spark_module_glm5_next_laguna.h`) take the name of their behaviour.
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
  - qwen4_flash refused every prefill frame from 2026-09-19 to 2026-09-28:
    its KV frame wrapper demanded a decode batch before asking whether the
    KV tier was on. The same check refused qwen38_max decode frames that
    carry no frame context (single-stage runs, unqualified execution, the
    T1 harness);
  - muse_glimmer's JIT KV tier (`SPARK_MUSE_GLIMMER_STAGE_KV_STORE`) now
    runs the common KV frame;
  - with the JIT KV tier on, qwen38_max, qwen4_flash and muse_glimmer could
    restore two blocks into one slot in a frame that evicted twice. Claimed
    slots are now pinned and the eviction cursor advances; exercise the
    tier with a pool smaller than the working set;
  - qwen38_max's FP8 grouped-expert launcher refuses views whose rows per
    expert or input width are off the 128 block again;
  - dsv4, gemma4, glm52, laguna, ling, minimax, muse_glimmer, qwen38_27b,
    qwen38_max and qwen4_flash now report a failed TP combine launch as
    `INTERNAL_ERROR` or `CAPACITY_EXCEEDED` with a log line, as glm5_next
    does, instead of a silent `IO_ERROR` the engine treated as transport
    trouble;
  - qwen4_flash, muse_glimmer and the gemma4 26B could not load a driver
    compiled from their firmware description: adapter initialize failed
    with `TARGET_MISMATCH`. qwen4_flash's and muse_glimmer's adapters sent
    the package contract hash as the description hash. qwen4_flash's
    description named another model id and revision, and the 26B's
    revision was a geometry string. muse_glimmer's description was an
    unported copy of qwen38_max's: model, revision, target, module and
    metadata. Its adapter also named a target no module builds, which
    `tools/muse_gen_deployment.py` does not deploy. Each description now
    carries what its adapter sends, and qwen4_flash's module Makefile builds
    its adapter where `tools/module_build_release.sh` looks for it. The
    muse description says `NOT_MEASURED` where the copy said
    `GPU_VALIDATED`; record a GPU receipt before changing it back;
  - every adapter that loads a driver now reports a weightd lease failure
    (`NO_LANE`, `EVICT_DENIED`) as `CAPACITY_EXCEEDED`, which fails the
    request. Before, residentd rejected such a completion and failed the
    route as `INVALID_ARGUMENT`. The qwen38 template returned the raw status
    from submit, and residentd's IPC cannot encode a status above
    `UNSUPPORTED` in a submit result either.
  - the glm52, glm5_next, laguna and ling GPU validators now compile at the
    batch bucket of the archive they validate (`SPARK_MODULE_BATCH_BUCKET`).
    Before, they compiled at the header default of 1024, so
    `publish_variants` validated every tighter variant with a b1024
    validator.
- `SPARK_DSV4_MODEL_DSPARK_SPEC_STEP` is the last build setting a production
  header still defaults with `#ifndef` (7, from the dsv4 contract). The dsv4
  module Makefile passes it only for a k-sweep build (`DSPARK_SPEC_STEP`),
  and nothing passes it to the dsv4 adapters, tests or tools, so
  `tools/devcycle/build_remote.sh` compiles a k-sweep module at the
  requested step and its adapter at 7. Pass the step from every dsv4 build
  and delete the default. `tests/test_no_build_defaults.py` lists it as the
  one pending exception. Every other build setting is now named by the
  build that compiles it: the batch bucket (each family Makefile names 1024
  for the default archive, and the GPU validators compile at the bucket of
  the archive they link), qwen38_27b's serving TP degree, the
  pack-synthesizer and pack-load template hooks, the legacy
  `inference/llms` layer thread counts and the dsv41_flash attach probe's
  TP degree. Platform shims (`_POSIX_C_SOURCE`, `MSG_NOSIGNAL` and the like)
  and test-harness paths are not build settings.
- `tools/module_build_release.sh` runs `make archive adapter` in the
  module directory and compiles
  `examples/model_descriptions/<module>_<codec>_firmware.json` unless
  `FIRMWARE_JSON` names another description:
  - such a description exists only for gemma4 31B (bf16), glm52 (seven
    codecs), glm5_next (fp8) and minimax (bf16); laguna, ling, qwen38_max,
    qwen4_flash and glm5_next's other codecs need `FIRMWARE_JSON`;
  - dsv4, muse_glimmer and qwen38_27b build their adapters in the root
    Makefile or a family release script, and dsv41_flash and hy4 have no
    serving adapter. None of their module Makefiles names an
    `ADAPTER_SOURCE`, nor does the gemma4 26B's `Makefile.moe`, so
    `make adapter` refuses and the script cannot release them.
- glm5_next and laguna compare four driver descriptor fields in their
  own load functions and skip `model_description_sha256`, which
  `serving_adapter_template.c` checks for the other adapters. Moving them
  onto the template needs each build to pass its description hash, as the
  ling module does through `MODEL_DESCRIPTION`; the tree holds a glm5_next
  description for fp8 only.
- `tools/module_build_release.sh` defaults the ling firmware description to
  `examples/model_descriptions/ling_resident_decode_stage_<codec>_firmware.json`,
  which does not exist; a ling release must pass `FIRMWARE_JSON` (the
  script refuses without it). Commit the per-codec ling descriptions or
  point the default at `ling_resident_decode_stage_firmware.json`.
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
  them. dsv4's ladder adds buckets 6, 9 and 11. k3's is the common ladder,
  but `spark_batch_variant_tuning_common.h` can be instantiated once per
  translation unit and `tests/test_batch_variants.py` compiles the glm52,
  k3 and dsv4 headers together; no k3 build includes k3's header.

## Runtime completion

- Add bounded cancellation and drain for terminal client I/O failures so every
  resident sequence slot is released.
- A failed residentd route whose driver cache abort also fails stops the
  residentd so its unit restarts, because the driver's transaction state for
  those slots is unknown. Give the driver a per-slot reset so one slot can be
  recovered without restarting the unit.
- Adapters map a weightd lease failure (`NO_LANE`, `EVICT_DENIED`) to
  `CAPACITY_EXCEEDED` through `SparkModelServingCompletionStatus`, which
  fails the request. A family whose lease failure provably comes before any
  recurrent state advances could return `BUSY` instead, so the engine
  retries the step.
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

- The API applies the GLM chat template in shared code (`node/model_api.c`:
  `[gMASK]<sop>` and GLM role markers for every model). The chat template
  belongs to the model: carry it in the deployment's tokenizer or model
  description and let the API render whatever the model declares, then drop
  `node/model_api.c` from the `PENDING` list in `tests/test_dry_law.py`.
- `text/tokenizer.c` knows split regexes only by exact string. It knows
  the GLM digit-run pattern, the Qwen letter-and-mark pattern, and two
  letter-class patterns (MiMo, Qwen3.8-27b nvfp4, Ling, the last with
  possessive quantifiers). Every other `Split` is skipped without an
  error, and the text is BPE-encoded whole. A 2026-09-28 survey of
  `/mnt/model-warm/*/tokenizer.json` found these unhandled:
  - laguna, whose newline split precedes the letter pattern;
  - dsv4 and dsv4.1, with three splits;
  - muse-glimmer's case-aware letters;
  - gemma4's `Replace` plus `Split " "`.

  Implement them, then make an unknown `Split` a load error.
- The four known letter-class splits classify code points by Unicode
  class (`text/unicode_class_tables.h`, generated from Python
  `unicodedata` 16.0 and checked range for range against the Oniguruma
  classes of HF tokenizers 0.23.2 by
  `tests/test_tokenizer_unicode_split.py`). The legacy GPT-2 `ByteLevel`
  regex path (`use_regex: true` with no `Split`) still classifies by byte
  and treats every byte >= 0x80 as a letter. Port it to the same
  classifier before a model that uses it serves non-ASCII text.
- The tokenizer applies an `NFC` normalizer (Ling, Qwen3.8, MiMo). It
  skips any other normalizer without an error: gemma4's `Replace`, and
  `Sequence`, `NFKC` and `Lowercase` if a model declares them. Implement
  those, then make an unknown normalizer a load error. A tokenizer with
  NFC cannot be saved in the compiled format, which has no field for it.
- The ling driver applies no SwiGLU clamp. The publisher serving code
  (sglang and vLLM `bailing_moe_v3`) clamps `silu(gate)` to at most L and
  `up` to [-L, L] on layers 34-41, using `expert_swiglu_limit_list` and
  `share_expert_swiglu_limit_list`; the HF modeling and our contract treat
  the lists as inert. The three reference prompts cannot tell the two
  apart (docs/T1_REFERENCE_COMPARE.md). Add the clamp to the routed and
  shared expert activation for those layers, and make the contract name
  the lists as required behaviour.
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

## Provider network

Of the provider network in `README.md`, the tree has only the LiteLLM front
door and the static pages and playground in `site/`.

- **Tailnet.** No tailnet exists yet. It needs a control server that scales
  to hundreds of providers at a sane cost (Tailscale or a self-hosted
  Headscale), tagged auth keys issued at registration, access rules that
  let only the router and the reference nodes reach a provider's model API
  port (and providers nothing of each other), key rotation, and removal of
  a provider whose bond is forfeited.
- **Provider agent.** `sparkpipe provider register` and an agent that
  joins the tailnet, publishes the provider's offers (models, prices,
  limits, the resale switch) to the router and signs completion receipts.
- **Owner-first scheduling.** The batch engine has priorities but no
  preemptible class. Add a network class that runs only on capacity owner
  traffic leaves idle, and that yields at frame boundaries.
- **Hand-off.** A network request either finishes within the bound the
  router set when it placed it, or resumes on another provider. Resuming
  mid-request needs KV transfer between installations; until then, a
  preempted network request restarts elsewhere.
- **Router on LiteLLM.** Each offer becomes a LiteLLM deployment at the
  provider's tailnet address, with the provider's price as its per-token
  cost. Missing:
  - adding and removing deployments as offers change, without a restart;
  - a routing strategy by price, latency, load and verification record;
  - prefix affinity, so requests that share a prompt prefix reach the
    provider holding that KV cache;
  - pinning to one driver build and hardware type for buyers who need
    seeded replays to match bit for bit;
  - virtual keys and spend logs, which need LiteLLM's Postgres database;
  - a logging callback that feeds the audit sampler.

  Two limits of the door as it stands, found with LiteLLM 1.74 and a mock
  upstream (`docs/LITELLM_FRONTEND.md`, browser clients):
  - the committed `config/litellm-config.yaml` uses `vllm/` deployments,
    which serve only the token-ID passthrough; LiteLLM's chat route fails on
    them, so chat clients, the playground included, need the `openai/`
    deployments that `tools/generate_litellm_config.py` writes;
  - LiteLLM consumes a request's `priority` for its own scheduler and does
    not forward it, so the batch engine's priorities do not cross the door.
    `seed`, `temperature` and `deadline_ms` do.
- **Metering, billing, payouts and bonds.** Bill buyers from signed receipts
  at the serving provider's price, pay providers 85% after the challenge
  window, and hold and forfeit bonds.
- **Audit service.** It needs:
  - a cost-weighted secret sampler over real completed requests (about 2% of
    tokens);
  - a replay scheduler, and reference nodes for each hardware type and
    driver build;
  - a comparator: exact tokens for greedy traffic, logprobs within a stated
    tolerance for sampled traffic;
  - the challenge and dispute state machine;
  - weight fingerprints.
- **Exact replays need batch-invariant numerics.** Batched decode is not
  bitwise equal to B1 (see Dynamic batching). Until the batch kernels agree
  bitwise, replays of batched requests can only compare logprobs within a
  tolerance, which is weaker against mild quantization.
- **Logprobs do not exist yet** (see Serving API), so sampled traffic cannot
  be audited.
- **Every driver update splits audit cohorts.** Replays compare only against
  the same driver hash, so the router must track each provider's build, and
  releases need per-model cohort changeovers.
- **sparkpipe.ai is static.** `site/` has the landing page, the provider page
  and the playground. The catalog, the buyer console and the provider
  dashboard need the router and the ledger behind them.
- **Payments.** Provider identity checks, tax reporting and payout rails
  (fiat, crypto or both) are undecided.

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

## Mac Studio deployment

The design is in
[`docs/HARDWARE_TOPOLOGY.md`](docs/HARDWARE_TOPOLOGY.md#mac-studio-pool).

- Define hardware profiles for one to eight Studios: memory, bandwidth, the
  Thunderbolt 5 island wiring, the bridge to the Sparks, power, and storage.
- Add Thunderbolt 5 RDMA as a link class. Apple's verbs API (TN3205) offers
  only send and receive, on at most ten unreliable-connection queue pairs,
  with no hardware acknowledgements and no routing, so the transport must
  detect loss, retransmit, and forward between islands itself.
- Place models on the pool: one replica per Studio when the model fits, TP
  inside an island, TP4 x PP2 across the islands.
- Serve disaggregated: a Spark TP group prefills and streams the KV cache
  layer by layer to a Studio group, which decodes. This needs one KV wire
  format per model, and the resumable request moves with its cache.
- Run PP2 across hardware classes once routes waiting on transport input can
  be cancelled when a peer rank fails.
- Measure each bridge when the Studios arrive: Studio 10GbE into the fabric;
  IP over Thunderbolt from a Studio to the RTX 5090 host, on that host's
  controller; the host's ConnectX-6 into the CRS804. Record the winners as
  named link classes in the deployment JSON.
- Stand up reference nodes for every hardware combination the provider
  network sells (Studios alone, and Spark prefill with Studio decode),
  because different kernels produce different bits.
- Validate office power, cooling, startup, failure recovery, and service
  operations as part of the deployment receipt.

## Incremental expansion

- Automate expansion from 4 to 8 to 16 Sparks while preserving package
  identities, catalog state, priority policy, and resumable request metadata.
- Generate model placement and storage rebalance plans before nodes join the
  ready set; never improvise redistribution in the request path.
- Support adding Mac Studios one at a time, as standalone replicas or as the
  decode pool of a Spark fleet, under the same API and scheduler.
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
