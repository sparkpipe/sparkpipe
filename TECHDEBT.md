# SparkPipe Technical Debt

This file contains only unfinished work against the system described in
[`README.md`](README.md) and the contracts in [`SPEC.md`](SPEC.md) and
[`sparkpipe_invariants.md`](sparkpipe_invariants.md). Completed work is
removed rather than retained as a progress diary.

## Dual-fabric topology contract

- Replace the legacy ring/single-switch/dual-switch topology modes with one
  schema that represents the CRS804 rail plus eight pairwise direct links.
- Generate supported four-, eight-, and sixteen-Spark profiles from the same
  schema, requiring complete direct pairs at every size.
- Generate pinned interface, address, direct-partner, and communicator tables
  for all sixteen ranks from that schema.
- Remove obsolete topology examples and release switches after all consumers
  use the combined-fabric contract.

## Mesh collectives

- Every hardware-wait round crosses weightd's CPU relay twice: the local
  doorbell sweep posts the RDMA writes and the remote sweep completes the
  waiter. Replace it with GPU-initiated RDMA so kernels post work requests
  and ring the NIC doorbell themselves; a 16-rank 8 KiB all-reduce should
  then cost tens of microseconds rather than the measured 167 us p50.
- Two collective substrates coexist: the residentd-owned hidden transport
  (`ring/transport/tp_collective.c`, recursive doubling and split rings),
  which only k3's runner still creates (`SparkTpCollectiveCreate` in
  `spark_k3_resident_decode_stage_runner.cu`), and the weightd mesh through
  `ring/transport/tp_device_collective.c`, which every other TP module
  opens. Move k3 onto the mesh and delete the hidden transport; its own
  control-plane debt (per-collective host callbacks, credit-return sends,
  per-direction sessions) goes with it.
- The wait mode picks the algorithm, against I36. With
  `SPARK_TP_WAIT_MODE=hardware` every payload runs chunked direct rounds
  (`SparkTpLaunchMeshHardware` is always called with one logical row), with
  reduce-scatter plus all-gather over slice routes for sums of at least
  `SPARK_TP_MESH_RSAG_MIN_ELEMENTS` (49,152) elements at degree 4 or more.
  With spin wait, a single-sequence payload that fits one slot takes the host
  round and everything else takes the tree (`SparkTpDeviceCollectiveHostRound`,
  `SparkTpDeviceCollectiveRunDeviceRounds`). Select from the logical batch
  and payload in both modes.
- The reduce-scatter crossover is a constant, not a measured profile per
  degree, operation and datatype. Measure it on the mesh, retain the profile
  as a release artifact, and select from it.
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
- weightd has 16 mesh lanes (`4f0e339`). CUDA refuses to host-register the
  daemon's RDMA-registered mesh region (`cudaErrorInvalidValue`); since
  `5814bf2` (PR #1135) the collective logs `MESH-REGISTER-SKIP` and runs
  unregistered. Find out why CUDA refuses those pages, and measure the
  unregistered path against a registered one.
- Hardware waits were deployed fleet-wide without distributed fault
  qualification. The real daemon/NIC path has no receipt for rank skew, a
  missing peer, timeout, cancellation, a failed Begin/End or source-slot
  reuse. The `MESH-REGISTER-SKIP` and `MESH-DEVICE-ALIAS-IDENTITY` paths were
  never exercised on a daemon-owned region, and the production link
  `rocep1s0f1` was not covered by the two-host check
  ([`docs/TP_STREAM_MEMOP_QUALIFICATION.md`](docs/TP_STREAM_MEMOP_QUALIFICATION.md),
  Production status). Qualify GPU cancellation, drain and event-driven mesh
  activity on that path too, so that rearming a wait cannot cancel unrelated
  work on the rank.
- Teardown after a register skip is broken by code reading:
  `SparkTpDeviceCollectiveReleaseRegion`
  (`ring/transport/tp_device_collective.c`) still calls `cudaHostUnregister`
  on a mapping whose registration was skipped, which ends in
  `MESH-UNREGISTER-FAIL` with a retained cleanup-only owner. Record the skip
  per mapping.
- NCCL leftovers after `b31761e`, which deleted the NCCL backend:
  `SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL`, the `nccl` parsers in
  `runtime/serving_adapter_template.c` and
  `modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c`, the
  NCCL branches in the glm5_next and laguna modules, and
  `tools/qwen38_tp4_nccl_bench.c`.

## Steady-state decode hot path

- Accept each hot-path change independently: exact token parity first, then
  at least three unprofiled end-to-end cached-prefill B1 runs. Do not stack
  candidates until the preceding candidate beats the current GLM 5.3 Flash
  TP16 B1 receipt in [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md)
  (36 tok/s on 2026-09-28).
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
  or a collective without hardware-wait rounds. Working-set graphs (Model
  residency and storage) remove the unpinned case without pinning every
  expert; move MTP and speculative verify onto the linear walk.
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
- Validation and admission fanout run on the per-frame path (perf-program
  rock R5, from the archived `PERF_PROGRAM2.md`): 12 + 9 `Validate` call
  sites remain on the client paths, and the engine SHA-probes inside
  `Progress`. Move them off the per-frame path.
- qwen38_27b: `SparkQwen38_27bServingUploadBlockTable`
  (`modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c`)
  copies the full `max_active_sequence_count` x `blocks_per_lane`
  block-index table plus the counts on every submission (rock R7). Add dirty
  tracking.
- dsv4: island chaining, RA joins and an event diet (rock R6). Do this only
  while DSV4 stays in the driver order.
- glm5_next: the exact DSA top-k (`LmTopkExactKernel`) runs four radix passes
  and a block-scan compaction in one CTA per row. At 32K context a row has
  8,192 pools, and the kernel's time there is unmeasured. If it shows in
  `run_us` at long context, compact with warp ballots instead of the
  Hillis-Steele scan.

## Placement beyond TP16

- GLM 5.3 Flash serves only at TP16. Measure B1 latency and aggregate
  throughput from B8 to B1024 for TP16, TP4 x PP4 and PP16 once each runs,
  and choose placement from those numbers (README, Placement). The items
  below are the TP4 x PP4 work; PP16 needs the same stage-local layer spans
  and boundary forwarding.
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

- Relocatable expert graphs (saved graphs) are designed but not built. `8adebc6`
  proposed capturing a graph once, recording which kernel arguments are
  expert pointers and patching them on load, as a dynamic linker does, and
  `360c0ee` and `1a674be` deleted the union-lease workaround on that
  assumption. No code patches or updates an instantiated graph's kernel
  arguments, and nothing stores a captured graph for reuse, so every engine
  captures its graphs again and the captured pointers must never move.
- Working-set graphs. Because the pointers cannot move, a whole-chain GLM
  graph refuses to run unless all 12096 routed experts are leased
  (`SparkGlm5NextGraphClaimExperts`, since `78c2c21`), and production pins
  every expert (`G5_PIN_EXPERTS=1`). Without the pin only eager chains run;
  the fleet's eager, spin-wait configuration measured 5.9 tok/s B1 earlier
  on 2026-09-28 (lead-dev measurement). Full pinning breaks I29's bounded
  residency, and the memory it holds is unavailable to co-resident drivers.
  On top of relocation, capture over a leased working set and send a miss
  to an explicit, reported refill.
  - The pin takes 12096 keys in 24 leases per rank; the daemon held
    20,874 MiB of device memory on spark6 on 2026-09-28
    ([`docs/WEIGHTD_DESIGN.md`](docs/WEIGHTD_DESIGN.md#glm-graph-residency-today)).
  - With fewer experts leased, `SparkGlm5NextGraphClaimExperts` returns
    `UNSUPPORTED` and the wave runs eager instead. That is logged, but it is
    still a mode fallback.
  - This is the known exception in
    [`docs/DRIVER_ACCEPTANCE.md`](docs/DRIVER_ACCEPTANCE.md) and the first
    item of ROADMAP M6. Exit: GLM graph decode on a bounded expert working
    set, with the same greedy tokens as the pinned build
    ([`docs/GLM_LAZY_DRIVER_INTEGRATION.md`](docs/GLM_LAZY_DRIVER_INTEGRATION.md),
    [`docs/WEIGHTD_DESIGN.md`](docs/WEIGHTD_DESIGN.md#open-relocatable-graphs)).
- Partial residency (S3) and queue warm hints with pin-on-dispatch (S4) from
  the archived `WEIGHTD_RESIDENT_CACHE_DESIGN.md` are not implemented; S2 is
  partial.
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

## KV tiers

- JIT-KV W3 is open (from the archived `docs/archive/JIT_KV_RESPONSE.md`).
  The three spill mechanisms (`cache/kv_page_store.c`,
  `runtime/spark_kv_backing.c`, `cache/nvme_tier.c`) were never collapsed
  into one. `runtime/spark_kv_backing.c`'s only consumer is
  `tools/spark_kv_backing_test.c` (Makefile targets). `SparkKvPagerInitialize`
  (`cache/kv_pager.c`) has no production caller; only
  `tests/test_jit_kv_slice.c`, `test_jit_kv_c3c4.c` and `test_jit_kv_c5w2.c`
  call it, so the pager that README's KV tiers describe is not wired into
  any module.

## Prefix reuse (I23)

Adapters without `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE` get no
cached prefixes and log `prefix_reuse=off` at engine connect
([`docs/DRIVER_ACCEPTANCE.md`](docs/DRIVER_ACCEPTANCE.md), Prefix reuse
capability). Each is an open I23 gap:

- qwen38_27b has a GDN snapshot borrow for prompt checkpoints
  (`SparkQwen38_27bServingPrefixBorrow`) but cannot declare the capability:
  a borrow miss logs `recomputing` and prefills over unrestored KV blocks and
  GDN state; decode-lane checkpoints are never snapshotted (only prefill frames
  call `SparkQwen38_27bServingPrefixPublish`), so indexed generated blocks
  would always miss; a publish that finds no free entry or more than 64 blocks
  returns OK without storing; the eight snapshot entries evict independently
  of the engine's index; a borrowed partial last block is shared without
  copy-on-write.
- gemma4 keeps a per-slot block allocator and has no borrow path.
- glm52 and ling resolve prefix pages through `SparkKvPageCachePrepareLane`
  but attend through an identity page table; laguna shares glm5_next's page
  table upload. Each needs a restored-versus-uninterrupted proof (I27) before
  it declares the capability.
- k3, minimax, muse_glimmer, qwen38_max and qwen4_flash have no borrow path.

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
- Batch weight amortization (perf-program rock R4): take the WS/native path
  from two rows up, with k-tile pipelining, so a batch reads each weight
  once.
- `SparkContinuousBatchStep` (`scheduler/continuous_batch.c`) has no caller
  outside `tests/test_continuous_batch.c`. Delete it or wire it.
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
- No gate checks batch invariance. The COMPSEC-17 run of 2026-09-28
  (`qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/REPORT.md`)
  sent the same 17 prompts concurrently four times: each run differed from
  the sequential completions in one or two cases, and the runs differed from
  each other. Add a gate that replays a fixed prompt set sequentially and
  concurrently and requires byte-identical completions, and run it with
  every serving release until the batch kernels pass it.
- JIT KV admission does not prefetch. The serving `prefetch` hook runs
  cache-prepare admission only when residentd receives a submission
  (`SparkModelServingAdapterPrepareSubmission` in `node/model_residentd.c`),
  so a restore starts at dispatch, not when the engine queues the request.
  A submission that cannot be prepared answers `BUSY`, and the batch engine
  retries it with a 10 to 200 ms backoff, up to 10,000 times
  (`runtime/model_batch_engine.c`). The deadline-ordered restore in
  `cache/kv_pager.c` (`SparkNvmeTierRequestDemandDeadline`) has one
  consumer, dsv4 (`spark_dsv4_jit_kv.h`), whose packs are deleted; glm5_next
  does not use the pager. Issue restore demand from enqueue, admit against
  restore bandwidth, and dispatch a lane only after its restore completes.
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

- Add an exact checkpoint-derived contract for MiniMax H3; `model_contracts/`
  has none.
- Remove legacy model names from generated release inventories and operator
  surfaces when their replacement contracts land.
- `tools/generate_recipe.py` (`MODELS`, via `glm52.json`) and
  `examples/recipes` still generate GLM 5.2 (`zai-org/GLM-5.2`) recipes,
  although the owner said on 2026-09-28 that GLM 5.2 weights are deprecated.
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
- Tree verification is not wired. The policy engine resolves trees
  (`SparkSpeculationPolicyResolveVerifierTree`), but no caller passes one.
  Sequence state stores chains, the seam collapses DFT3 trees to one chain
  (`SparkSpeculationSeamExtractChain`), and no verify frame applies a
  tree-attention mask. `spark_speculation_tree.h` has no production includer.
- Build multi-drafter composition over the seam, after the single-drafter
  agreement matrix is measured: verify the DFT3 tree instead of collapsing
  it, allow local plus remote sources together (qwen38_27b rejects the mix),
  and attribute acceptance per `source_bit`.
- `SPARK_QWEN38_27B_SPECULATORS=0x4` (local DFlash2) fails initialization
  with `SCHEMA_ERROR`, with or without a bridge, because the seam classes
  DFLASH2 as a remote tap source.
- qwen38_27b remote drafting is synchronous: a decode frame, then a blocking
  20 ms `DraftRemoteChain`, then verify. Pipeline it one round ahead.
- glm5_next accepts `draft_bridge_host`/`draft_bridge_port` and the tap-free
  remote sources but has no `DraftRemoteChain` call site.
- `inference/kernels/speculate.cuh` is included by four unity files and
  launched nowhere.
- `Makefile:791` passes a vestigial `-DSPARK_DSPARK_TARGET_GLM52=1` to
  `spark_speculation_policy.o`, which no longer includes the header that
  reads it.
- The qwen38_max provider-slot shim is the only user of
  `spark_speculation_provider.h`; move it onto the seam and delete the slot.
- `tests/test_speculation_tree_resolve.c`,
  `tests/test_speculation_headers_coexist.c` and
  `tests/test_qwen38_27b_remote_spec.c` are not built by the Makefile.
- `tools/fleet_serve.sh` defaults `SPARK_GLM5_NEXT_MTP=1` (`:66`, `:96`),
  which fails glm5_next initialization at TP > 1.
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
  universal packer ([`docs/DRY_PACKBUILDER_PROPOSAL.md`](docs/DRY_PACKBUILDER_PROPOSAL.md)): one CLI, one codec table,
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
  replace (estimated by the 2026-09-13 SEAM surveys at about 26,000 lines
  across the families), each migration proved by byte or behaviour identity.
- Two copies of each model's constants: make `spark_<family>_model.h` a shim
  that includes `llm_defines.h` for dsv41_flash, glm52, glm5_next,
  muse_glimmer and qwen38_27b, or tie the copies together with
  `_Static_assert`. glm5_next, for example, states hidden 4096 and 288
  experts in both files. k3's model header includes
  `spark_k3_llm_defines.h`, and nothing consumes k3's generic
  `llm_defines.h`.
- glm5_next assigns its combine wrappers field by field instead of calling
  `SPARK_FAMILY(ModuleRegisterCombines)`, and its `internal.h` re-declares
  the `SparkTpLaunch*` prototypes from `spark_tp_mesh_register.h`.
- k3 has no TP16 adapter descriptor: `K3ServingDescriptor` is `k3-tp4pp4`
  only, so a TP16 PP1 deployment cannot load
  ([`docs/K3_PERF.md`](docs/K3_PERF.md)).
- k3's `ServingPrefetch`, `ResolvePrefetch`, `Progress`, `Quiesce` and
  `Reset` are no-ops that return success (I01).
- The `capture_graphs` keys that `tools/k3_gen_adapter_configs.sh`,
  `tools/k3_multidev_lane.py` and
  `modules/k3_resident_decode_stage/configs/*.json` emit have been read by
  nothing since `a29ea53`.
- qwen4_flash: `ServingSeamInterface` in
  `spark_qwen4_flash_serving_adapter.c` sets no `prefetch`,
  `resolve_prefetch` or `reset`, so
  `SparkModelServingAdapterValidateInterface` rejects the adapter.
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
    codecs), glm5_next (fp8), laguna (bf16) and minimax (bf16); ling,
    qwen38_max, qwen4_flash and glm5_next's other codecs need
    `FIRMWARE_JSON`. laguna's module Makefile refuses `make adapter` for a
    codec without a description. Its fp8 and nvfp4 descriptions wait for
    per-layer expert codecs: poolside's FP8 and NVFP4 releases keep the
    routed experts of layers 44-47 and 40-47 in BF16, and the module
    takes one expert codec for every layer;
  - dsv4, muse_glimmer and qwen38_27b build their adapters in the root
    Makefile or a family release script, and dsv41_flash and hy4 have no
    serving adapter. None of their module Makefiles names an
    `ADAPTER_SOURCE`, nor does the gemma4 26B's `Makefile.moe`, so
    `make adapter` refuses and the script cannot release them.
- glm5_next and ling compare four driver descriptor fields in their own
  load functions and skip `model_description_sha256`, which
  `serving_adapter_template.c` checks for the other adapters. Moving them
  onto the template needs each build to pass its description hash; the
  tree holds a glm5_next description for fp8 only.
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
- `node/model_api.c` calls `SparkModelBatchEngineReopenAdmission` before
  every submit. After a latched collective failure it keeps feeding the dead
  collective instead of failing loudly.
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

  Limits of the door as it stands, found with LiteLLM 1.74 and a mock
  upstream (`docs/LITELLM_FRONTEND.md`, browser clients):
  - LiteLLM consumes a request's `priority` for its own scheduler and does
    not forward it, so the batch engine's priorities do not cross the door.
    `seed`, `temperature` and `deadline_ms` do.
  - `tools/generate_litellm_config.py` emits `num_retries: 1`, which retries
    a failed upstream call once and can duplicate a generation on the fleet;
    the old hand-written config used 0.
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
- `tools/glm5_next_driver_compare.py` runs resident baselines, which always
  fail since `c67be23` removed the module's direct pack loader, so it never
  writes `RESULT.json`. Replace the baseline with a lazy or path-parity one.
- Four Python tests stay outside `make test` because they drive the fleet
  over ssh: `test_expert_io_perf`, `test_jit_kv_page_fault`,
  `test_lossless_doorbell` and `test_transport_stability`. Give them a
  runner and register them.
- `test_hy4_driver_acceptance` has a build rule but stays out of
  `TEST_NAMES`: it holds the behaviour a complete hy4 driver must show and
  fails on the current stub module. Register it with the hy4 driver.
- Four K3 test sources have no build rule and no runner:
  `test_k3_serving_adapter_smoke.c` needs an adapter configuration argument,
  and `test_k3_interleave_gemm.cu`, `test_k3_layer_probe.cu` and
  `test_k3_swizzle_probe.cu` are sm_121a probes. Each needs an nvcc-gated
  rule and a Spark runner, or deletion.
- Eleven C tests are neither registered nor run anywhere. Ten should be
  deleted: `test_cache`, `test_sideband`, `test_group_gemm_workspace` and
  `test_state_pool` test headers no product includes (`cache/cache.h`,
  `ring/sideband.h` and `runtime/workspace.h` go with them;
  `spark_state_pool.h` is still used by `test_k3_kv_cache`);
  `test_continuous_batch_decode`, `test_multi_row_prefill`, `test_pack`,
  `test_dequant` and `test_reference` (with `tests/reference.h`) test local
  reimplementations; `test_graph_replay_kernel_abi` checks a mesh-kernel
  marker the module build already verifies. `test_graph_replay_correctness`
  is a GPU rig with private kernel prototypes and no build rule; it needs an
  nvcc-gated rule on the shared kernel headers and a Spark runner.
- `make test` fails on a host with a CUDA toolkit. There the model-common
  library carries no CUDA stub, but `ring/transport/tp_device_collective.c`
  calls the mesh launchers (`SparkTpLaunchMesh*`), which only a module's
  CUDA object (`spark_tp_mesh_kernels.cuh`) or the stub defines. An adapter
  that pulls the transport from the library is left with them undefined: on
  the x86 hub at main `bbf5432`, `build/libdsv4_tp4_pp4_serving_adapter.so`
  fails `dlopen` with `undefined symbol: SparkTpLaunchMeshHardware`, and
  `make test` stops at `test_dsv4_tp4_pp4_serving_adapter`. Move the
  launcher calls behind the module boundary, or give host links one object
  that defines them, rather than linking the stub case by case (#1258 did
  that for two tests).
- The production serving configuration is not in the repository. GLM 5.3
  Flash at 36 tok/s B1 needs `G5_GRAPH_PATH=1`, `G5_PIN_EXPERTS=1`,
  `SPARK_TP_WAIT_MODE=hardware` and `G5_WARMUP=0`, set only in each node's
  untracked drop-in
  `~/.config/systemd/user/fleet-agent.service.d/20-serving.conf`;
  `tools/fleet-agent.service` sets none of them. Without the drop-in,
  residentd refuses to start (`SPARK_GLM5_NEXT_GRAPH_PATH` unset) and warmup
  runs; with only `G5_GRAPH_PATH=0`, the node falls to eager chains, spin
  wait and unpinned experts. Move these settings into the deployment contract as validated
  fields (I04) and record their hash in every receipt (I33).

## Fleet and queue tooling

Found while verifying [`docs/FLEET_RELEASE_RUNBOOK.md`](docs/FLEET_RELEASE_RUNBOOK.md)
and the multidev docs on 2026-09-28. Line numbers are in
`tools/fleet_node_agent.sh` unless another file is named.

- `restart_scope` (`:311-316`) matches `config/model_resident.json`, but a
  root's deployment file is the top-level `model_resident.json`, so a
  deployment-only change never restarts the root.
- An installed but unrecycled weightd makes `ensure_weightd` return 1 on
  every pass while an engine runs (`:440-444`, `:565-569`), which stops all
  root convergence.
- `apply_manifest` never re-verifies on-disk files while the `MANIFEST` is
  unchanged (`:279`), so an agent copied in by `fleet_sync.sh start`
  persists.
- Heartbeats are sent only on change (`:105-118`), so the epoch is not
  liveness. Add a periodic report.
- `fleet-agent.service` runs with the default `KillMode=control-group`, so
  any agent stop, restart or exit kills weightd and the engine. Set
  `KillMode=process` in the drop-in or record why not.
- `fleet-agent.service` has no finite `MemoryMax`; give it one and track it
  with `spark_queue.py track --scope user`.
- `tools/publish_core.sh:4` ignores `SPARKPIPE_BUILD_TREE`, and
  `tools/publish_local.sh:32` takes `model_driver.so` from `~/sparkdata/out`,
  which the current build never writes. Fix or retire both.
- The janitor's log message (`:410`) is inverted relative to what it kills.
- The `install_core` and `tools/weightsd_announce.sh` messages still say
  weightsd owns the restart.
- `tools/fleet_release_hygiene.sh:16` defaults `HUB` to sparkf; the hub is
  the rtx5090.
- `tools/fleet_ready_poll.sh` reads heartbeats from sparkf instead of the
  rtx5090 hub, waits on an `UPDATE` sentinel the agent no longer uses, and
  miscounts ready nodes when it greps several files.
- Neither the `20-serving.conf` drop-in (Production qualification) nor
  `sparkpipe-hub-route.service` is in the repository or shipped by
  `fleet_sync.sh`. Production GLM should also pin its lane with
  `SPARK_WEIGHTD_LANE=0` in that drop-in.
- The queue ledger holds 32 stale persistent owners
  (`sparkpipe-weightd-shared.service`,
  `sparkpipe-glm-serving-dd3526b2.service`) that block `gpu-shared`
  admission. Untrack them.
- Bump `SPARK_WEIGHTD_IPC_ABI_VERSION` whenever the mesh layout changes.
- The family wrappers default to `/run/sparkpipe-weightd-shared/weightd.sock`,
  which no Spark provides.
- `tools/devcycle/lane_assignments.json` and `lane_budget_calc.py` still
  assume 8 lanes, and `lane_assignments.json` gives lane 0, which production
  GLM should hold, to a GLM development lane. `tools/inference_smoke.py` accepts lanes
  0-7 only.
- A stale, idle `sparkpipe_weightsd` still runs on spark6.
- Operations: linger is missing for the fleet user on spark8, spark9,
  sparka to sparkd and sparkf. Run `sudo loginctl enable-linger` there.

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
- `tools/fleet_swap.sh` still drives the system unit
  `sparkpipe_model_residentd` through sudo (`start_model`, `stop_model`),
  not the `fleet-agent` user unit that serves. Running it writes `/etc`
  drop-ins and starts a system-level residentd outside the fleet-agent
  cgroup: a fleet-scope swap would start a second residentd beside the
  agent's on all 16 nodes, which the agent's janitor does not reap because
  it only matches its own roots. The registry it reads
  (`tools/devcycle/fleet_registry.json`) has no GLM 5.3 Flash entry and
  marks both DSV4 models removed. The fleet_swap procedure is obsolete:
  delete the script and the registry, together with their remaining
  callers `tools/devcycle/deploy_pro.sh` and
  `tools/devcycle/first_decode_pro.sh`, or rebuild model swaps on the
  agent's release roots.
- Device allocation budgets are incomplete. GB10 `MemoryMax` does not
  contain every CUDA allocation, and the driver ledger omits some direct
  allocations and graph and context overhead. Enforce complete budgets
  before co-resident drivers rely on them.

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
## MPS evaluation on GB10

- The CUDA MPS control and server binaries are present on the sparks.
  Run a live evaluation: start the control daemon, run two CUDA
  processes concurrently, confirm overlapping kernel execution and
  per-process contexts. If GB10 supports MPS, cross-driver GPU
  concurrency replaces driver time-slicing and the output-chunking
  requirement shrinks to memory-bandwidth fairness.
