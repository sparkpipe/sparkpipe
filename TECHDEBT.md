# SparkPipe Technical Debt

This file contains only unfinished work against the system described in
[`README.md`](README.md) and the contracts in [`SPEC.md`](SPEC.md) and
[`sparkpipe_invariants.md`](sparkpipe_invariants.md). Completed work is
removed rather than retained as a progress diary.

Required behaviour that is not implemented is recorded here as "Left out on
purpose", with the date, what is missing, its consequence and the fleet proof
that closes it; no other way of not implementing a required feature is
accepted. The 2026-10-02 entries come from the JIT KV audit, the prefix-reuse
audit and the production-code findings of the 2026-10-01 test-honesty audit,
each verified against branch `kv/sequence-shard` at `b41ed891f`; their line
citations refer to that commit.

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
- Left out on purpose (2026-10-02): The wait mode picks the algorithm, against
  I36. With `SPARK_TP_WAIT_MODE=hardware`
  (`ring/transport/tp_device_collective.c:1491`) every payload runs chunked
  direct rounds through `SparkTpLaunchMeshHardware`, always called with one
  logical row (`:972-983`), and BF16 sums of at least
  `SPARK_TP_MESH_RSAG_MIN_ELEMENTS_WIDE` (18,432) elements at degree 16 or
  more, or `SPARK_TP_MESH_RSAG_MIN_ELEMENTS` (49,152) at degree 4 to 15, run
  reduce-scatter plus all-gather (`SparkTpDeviceCollectivePhases`,
  `:897-911`). With spin wait the capability read returns 0 (`:882-883`). A
  single-sequence payload that fits one slot then takes the host round
  (`SparkTpDeviceCollectiveHostRound`, `:1036-1040`), and everything else runs
  the tree (`SparkTpLaunchMeshTree`, `:984-993`), so one submission runs a
  different algorithm and phase count in each mode. The fix: select from
  degree, operation, datatype, logical batch and payload in both modes. Prove
  it on the fleet with a collective probe that records the same algorithm and
  phase count for every regime-table case in spin and hardware mode, plus GLM
  Full T1 and B16 in both modes.
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
- Left out on purpose (2026-10-02): the TP all-reduce does not use the
  pairwise links for bandwidth. weightd wires a pair QP to `rank XOR 1` since
  #1404, but the production collective sends every exchange over the one
  switched 100G port, so a large all-reduce (6.3M elements = B1024 or a
  prefill wave, 12 MB) takes about 3 ms, bound by that port, and the pair link
  carries only the partner's 1/15 share. The pair-first hierarchy designed in
  `docs/GLM5_NEXT_ROOFLINE.md` (pair reduce-scatter, 8-group reduce-scatter
  and all-gather, pair all-gather; rank-order fp32 sum, bit-exact) is built on
  the unmerged branch `perf/pair-first-allreduce` with a weightd bulk route:
  on 14 ranks with the pair link on its own PCIe x4 it measured 3030 to 2355
  us at 512 rows (-22%), 750 to 715 us at 128 rows, and was slower (142 to 185
  us) at 16 rows and when the pair link shared the switch NIC's x4. The branch
  is quarantined: spark8's root superblock was overwritten with a page of
  process memory during that branch's two-band test on 2026-10-01 16:01Z
  (cause under investigation), and the second x4 pair address is runtime-only
  (no boot unit). Close it by finding that root cause, then merging the
  pair-first all-reduce for payloads of two or more 1M-element chunks, proven
  by a 16-rank fleet ladder and GLM-5.3 Full T1, accuracy gate, B16, B256+ and
  TTFT A/B against the switch-only collective with root-filesystem checksums
  unchanged on every node.
- Left out on purpose (2026-10-02): communication is never overlapped with
  compute. Every decode step and prefill wave is one serial chain (compute,
  all-reduce, compute), so every collective wait is exposed: at B1 a GLM-5.3
  Full token takes 36.7 ms, of which 12.8 ms (35%) are peer waits over 236
  exchanges (p50 52 us, p99 262 us) and 25.5 ms kernels against a 19.4 ms
  memory floor; at B16 the waits are 21% of the step; at large batch the 3 ms
  bandwidth-bound all-reduces sit on the critical path. Every all-reduce is
  also a barrier, so one rank's hiccup stalls all sixteen. The design keeps
  two micro-batches in flight per engine, one computing while the other's
  collectives run on the NIC. Close it by splitting each wave into two
  micro-batches whose chains interleave on separate streams, proven by B1, B16
  and B256 fleet A/B with peer-wait time on the critical path reported per
  step, tokens identical, and the extra weight reads of the split measured
  against the hidden wait.
- Left out on purpose (2026-10-02): the mesh region can attach unregistered
  (`MESH-REGISTER-SKIP`, `MESH-DEVICE-ALIAS-IDENTITY`). Both paths now require
  the device to report coherent pageable access through host page tables, and
  fail attach otherwise (lane `mesh/contracts`). Still open: why CUDA refuses
  to host-register a shared weightd's RDMA-registered pages, and a Spark
  measurement of the unregistered path against a registered one
  ([`docs/TP_STREAM_MEMOP_QUALIFICATION.md`](docs/TP_STREAM_MEMOP_QUALIFICATION.md)).
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
- NCCL leftovers after `b31761e`, which deleted the NCCL backend:
  `SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL`, the `nccl` parsers in
  `runtime/serving_adapter_template.c` and
  `modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c`, the
  NCCL branches in the glm5_next and laguna modules, and
  `tools/qwen38_tp4_nccl_bench.c`.
- Awaiting fleet proof (lane `mesh/contracts`): a release module or the
  adapter template now refuses `collective_identifier` 0 at TP>1
  (`TP-COLLECTIVE-IDENTIFIER-REQUIRED`). The policy waiver is deleted, and the
  collectives-off probe needs a `-DDEBUG` module build. Proof: a GLM Full TP16
  load whose stage config carries identifier 0 fails on every rank, and the
  same build passes T1 and the accuracy gate.
- Left out on purpose (2026-10-03): module publish validation for TP>1
  modules runs one rank alone. `TP_STANDALONE=1` at TP>1 now builds a
  `-DDEBUG` module into a `-debug` build directory, a release module refuses
  the variable (`TP-STANDALONE-REFUSED`), and `publish` refuses a standalone
  build (`MODULE-PUBLISH-REFUSED`). The qwen38_27b and qwen38_max release
  scripts (`tools/qwen38_27b_lane_build_release.sh`,
  `tools/qwen38max_multidev_build_artifacts.sh`,
  `tools/qwen38max_multidev_run_family.sh`) therefore stop at publish until
  module publish can validate a TP>1 module with its collective open across
  ranks. Close it with a multi-rank publish validation, proven by a TP4 publish
  whose validator compares all four ranks' outputs with the CPU oracle.
- Awaiting fleet proof (lane `mesh/contracts`): attach now fails with
  `MESH-CAPABILITY-MISSING` when the local weightd lacks a route the regimes
  use, and the routes are fixed at attach, so no later read of the wait cell
  can change the algorithm. Proof: with one rank on a weightd without the
  bits, attach fails on every rank; the matched fleet passes the hardware
  collective probe and GLM Full T1.
- Left out on purpose (2026-10-02): weightd does not range-check the
  client-supplied `source_offset`, `length`, `remote_offset` or `lkey` of
  `MESH_WRITE` and `MESH_BROADCAST` requests (`runtime/spark_weightd.c`, the
  `SPARK_WEIGHTD_IPC_KIND_MESH_WRITE` and `MESH_BROADCAST` handlers;
  `node/weightd_mesh.c` post paths). A client bug or a stale request can
  overwrite another lane's cells inside a peer's registered mesh region,
  silently corrupting another engine's collective. Found by the spark8
  incident investigation (2026-10-02); it cannot reach memory outside the
  registration. Close it by validating every offset and length against the
  requesting lane's band and the registered region and refusing the request
  with a named error, proven by a weightd fault-injection run on two Sparks
  whose out-of-range requests are refused while in-range rounds stay
  bit-exact.
- Left out on purpose (2026-10-02): weightd's QP repair moves an error-state
  queue pair through RESET to RTS with PSN 0 (`node/weightd_mesh.c`, the
  repair path around `IBV_QPS_RESET`), which can drop pending sends without
  completions, and a sticky `transfers[].failed` then blocks `LaneConfigure`;
  spark9 logged endless WD-STUCK after the 2026-10-01 incident. Close it by
  draining or failing every outstanding transfer with a completion before
  repair, resynchronizing PSNs with the peer, and clearing per-transfer
  failure state on a successful rewire, proven by killing one rank's weightd
  mid-collective on the fleet and seeing every peer complete or fail its
  rounds and rewire without a restart.
- Recorded incident (2026-10-02; owner: treat as a one-off and act only if it
  recurs): on spark8 at 2026-10-01 16:00:49Z a PCIe completion timeout on CX-7
  function 0000:01:00.1 was followed within about 85 ms by a stalled SMMU0
  command queue (`CMD_SYNC timeout`). The kernel kept running, recycled IOVAs
  while lazy (DMA-FQ) translations were stale, and the NVMe wrote other
  processes' pages over the root superblock and GDT blocks before a hung-task
  panic. It was recovered via PXE rescue and e2fsck from the backup
  superblock. No other node shows AER, SMMU, NVMe or ext4 errors. If it
  recurs, the prepared responses are a runtime kmsg guard that panics on the
  first SMMU `CMD_SYNC timeout` or uncorrectable CX-7 AER, `iommu.strict` for
  the NVMe and CX-7 groups (a boot-path change for the owner), the guarded
  pair reproduction, and a report to NVIDIA and Canonical (investigation
  workflow wf_2545220b-a65).
- Awaiting fleet proof (lane `mesh/contracts`): every collective logs
  `MESH-PATH` (operation, path, phases, elements, rows, capture) on each regime
  change and again for each capture. `tools/glm53full_lane.sh` keeps one rank
  log per run (`logs/residentd-<run id>.log`, refuses to overwrite, `archive`
  collects and removes). The other lane tools (`ling_lane.sh`, `k3_lane.sh`,
  `gemma4_lane_resident.sh`, `laguna_multidev_decode.sh`,
  `qwen38_27b_lane_launch.sh`) still overwrite `residentd.log`. Proof: a
  fleet run whose rank logs name the path of every chain and survive a second
  run on the same lane.

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
- The exact DSA top-k selects rows of waves of 64 rows or fewer through
  16,384-score chunks (`LmTopkExactLaunch`): at 1M context a B1 step takes
  10.7 ms of top-k for 78 layers on one GB10 (was 147 ms). Prefill waves keep
  one CTA per row, which fills the GPU; their selection is 0.48 s of a
  1,024-row wave at 64K for 78 layers.
- The DSA indexer runs on every TP rank over the whole replicated index
  cache. B16 decode at 16K context reads 5.2 GB of bf16 index keys per step.
  With the context split, each rank scores its own 1/tp of the context and
  the ranks merge their top-k candidates.
- KV pages are just in time: physical pages must hold one full lane and
  logical pages every lane, so lanes share one resident pool and pages past
  it park in the backing store. Fleet record (2026-10-02, 0b5371e, TP16, 2 x
  262,144 positions on a 32 GiB pool = 5,637 pages, 20 GiB backing): two
  concurrent 124,997-token prompts each retrieved their pass key (TTFT
  450 s for both); pass keys retrieved at 8K to 60K; B1 30.3 tok/s. Parking
  copies synchronously under the binding mutex (see the JIT KV plan, G6),
  but the engine reserves each request's full prompt and output pages when
  it binds a lane, so active lanes never overcommit the pool; a request that
  does not fit waits in the queue.
- Left out on purpose (2026-10-02): glm52 graph regimes key long contexts on
  4,096-token buckets to 16K and four buckets per octave above, in the fixed
  72-regime table (`spark_glm52_graph_regime.h`): a 1,048,576-position
  deployment uses 60. Captured graphs are never evicted: a long prompt
  captures one graph per (context bucket, row bucket) it crosses.
  Relocatable graphs (lane R) close it. Fleet record (2026-10-02, 04256ca,
  TP16, 2 x 131,072 positions, replicated KV): T1 exact; B1 30.9 tok/s;
  TTFT 1.06 s at 994 tokens, 24.3 s at 16,308; pass keys retrieved at 8K,
  16K, 32K, 60K, 100K and 130K (key at position 110,464 of 129,991, TTFT
  243 s); B1 decode at those contexts about 20 tok/s. Graph and linear
  chains differ after the first two tokens of a 16K prompt because prefill
  waves padded to a row bucket run different GEMM shapes, so long-context
  output is judged by the accuracy gate, not token equality.
- Prefill waves past the 2,048 DSA selected-token count attend through the
  tensor-core prefill kernel over the union of each 4-row block's selected
  positions with a per-row membership mask (`LmPrefillSparseUnionKernel`):
  on one GB10 a 1,024-row layer at 16K takes 1.45 ms (was 5.96 ms through
  the per-row decode kernel), with outputs equal to bf16 rounding. Waves of
  64 rows or fewer stay on the per-row kernel, where the two are equal.
- A 1,024-row TP16 prefill wave spends about 440 ms of its ~1.0 s waiting in
  collective rounds (`SparkTpMeshHardwareGuardKernel`) and about 400 ms in
  compute, serially (rank-15 profile, 2026-10-02). Split waves into halves
  whose collectives overlap the other half's compute.

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
- Left out on purpose (2026-10-02): KV does not survive model eviction and
  reactivation, which `README.md:157-158` requires. Unloading GLM-5.3 Full
  destroys the binding
  (`modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:2490`),
  and the binding destroys its anonymous `O_TMPFILE` spill store
  (`runtime/stage_kv_binding.c:176`, `:266`; `cache/kv_page_store.c:179-185`)
  without saving anything, so every resident and spilled prefix is gone when
  the model returns. Close it by saving every published, unsaved chain to the
  snapshot store before the binding is destroyed and restoring through the
  store after reactivation. Fleet proof: publish a prompt, evict GLM-5.3 Full,
  reactivate it, resend the prompt, and see `cached_tokens > 0` with tokens
  identical to the first run.
- Left out on purpose (2026-10-02): Nothing validates the KV backing directory
  at startup beyond a non-empty string (`runtime/stage_kv_binding.c:209-213`)
  and a successful `O_TMPFILE` open (`cache/kv_page_store.c:174-185`). No code
  checks that the directory is on the hot-KV NVMe partition or that the
  partition has `kv_backing_maximum_bytes` free, and the page store reserves
  no space, so a misplaced or full directory shows up later as write-back
  failures that drop pages. Close it with a startup check (mount point,
  filesystem, free bytes against every resident driver's backing quota) that
  fails the load with a named error, proven on one Spark by loading against a
  directory off the partition and against a quota larger than the free space,
  both refused.
- Left out on purpose (2026-10-02): weightd accepts a KV reserve and never
  applies it. `--kv-reserve-bytes` and `SPARK_WEIGHTD_KV_RESERVE_BYTES` are
  parsed (`node/weightd.c:175-186`, `:344-356`), checked only against the
  ceiling (`:359-366`) and copied into
  `SparkWeightdServerConfig.kv_reserve_bytes` (`:371`).
  `runtime/spark_weightd.c` never reads that field: arena and expert-pool
  admission compares against `device_bytes_max` alone
  (`runtime/spark_weightd.c:996-997`, `:1449-1450`, `:1834-1835`). An operator
  who sets a reserve expects device memory to be held back for KV, but weightd
  can lease all of it to weights, and the engine's KV `cudaMalloc` then fails
  or crowds out co-resident drivers. Close it by subtracting the reserve from
  the admission ceiling, or by removing the flag until weightd owns KV pools.
  The node proof is a weightd started with a reserve that refuses arena growth
  past `device_bytes_max - kv_reserve_bytes` while the engine's KV allocation
  succeeds.
- Left out on purpose (2026-10-02): `SparkWeightdAttachRequested`
  (`runtime/spark_weightd_attach.c:37-48`) answers `BUSY` when
  `SPARK_WEIGHTD_SOCKET` is unset and `SPARK_WEIGHTD_ATTACH` is not `1`: a
  retryable status for a missing configuration, with no operation outstanding
  (I17). k3 returns it from `SparkK3StageRunnerInitialize`
  (`modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_runner.cu:1153-1164`)
  and adapter initialize (`spark_k3_serving_adapter.c:605-607`), and
  `tests/test_k3_attach_contract.c:159-162` asserts `BUSY`. Other modules read
  the same `BUSY` as 'attach not requested' and load directly (next entry).
  Return a non-retryable configuration status from
  `SparkWeightdAttachRequested`, update every caller, and prove it with a
  residentd start on a Spark from a deployment without weightd that fails with
  that status.
- Left out on purpose (2026-10-02): A deployment may omit the `weightd` member
  (`runtime/model_resident_deployment.c:563-568`); residentd then skips
  weightd (`node/model_residentd.c:3238-3248`) and
  `SparkWeightdAttachRequested` answers `BUSY`. The shared `LazyOpen`
  (`include/sparkpipe/family/module/spark_module_lazy_open.h:10-12`, used by
  glm52 and laguna), qwen38_max
  (`spark_qwen38_max_resident_decode_stage_module.c:450-452`), qwen4_flash
  (`spark_qwen4_flash_resident_decode_stage_module.c:455-457`) and dsv4
  (`spark_dsv4_resident_decode_stage_module.c:1055-1056`) treat that as
  success, and their pack loaders copy the whole stage pack to device
  (`spark_glm52_resident_decode_stage_module.c:529-545`,
  `model-families/common/include/sparkpipe/spark_pack_load_common.h:196-202`).
  `c67be235e` made attach mandatory because direct full-pack loads by several
  drivers kill Sparks, but only glm5_next refuses this case (`:731-733`); the
  GLM-5.3 Full lane renders `weightd` (`tools/glm53full_lane.py:127`), so it
  is latent there. Make the `weightd` member required, make every module fail
  initialization when attach is not configured, and prove it with a GLM-5.3
  Full residentd start from a deployment without `weightd` that fails before
  any device allocation.

## KV sharding

- Left out on purpose (2026-10-02): GLM-5.3 Flash (glm5_next) refuses TP16
  without `kv_shard`
  (`modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:482-487`).
  With it, each rank stores 1/tp of the latent KV and indexer keys through the
  shared context split (`include/sparkpipe/spark_kv_shard.h`,
  `inference/kernels/attn_shard.cuh`). Below TP16, `kv_shard` is optional, and
  `tools/glm5_next_gen_tp4pp4_deployment.py` sets neither `kv_shard` nor
  `dsa_index_context_parallel`, so a TP4xPP4 glm5_next deployment stores the
  full latent KV and indexer keys on every rank of each TP group, against
  README:268-274. It closes when `kv_shard` is required at every TP degree the
  shard check accepts
  (`model-families/glm5_next/include/sparkpipe/spark_glm5_next_kv_shard.h:103-110`),
  proven by a TP4xPP4 fleet run with T1 parity. GLM-5.3 Full replicates at
  every degree; see the glm52 entries below.
- Left out on purpose (2026-10-02): `dsa_index_context_parallel` is required
  whenever `kv_shard` is set
  (`modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:493-497`),
  and `tools/glm5_next_gen_deployment.py:136-138` sets both at TP16. Below
  TP16 it is optional, and the TP4xPP4 generator leaves it off.
- Replace per-driver KV and index pools with one node-level pool shared by
  all resident drivers, admitted against resident demand.
- Left out on purpose (2026-10-03): GLM-5.3 Full (glm52) splits every
  sequence's latent KV and DSA index keys across the TP ranks by context
  (owner of position p is p % tp; `SparkGlm52ModuleConfigure`, binding
  `context_shard` in `runtime/stage_kv_binding.c`), and refuses TP > 1 without
  the collective or without all-to-all. The change is not yet fleet-proven.
  It closes with a TP16 fleet run whose binding log shows `page_bytes=380928`
  per rank with the `-shard16r<rank>` layout, the `GLM52-KV-SHARD` sizing
  line, T1 and the accuracy A/B against the bf16 floor, and B1, B16, 16K and
  128K TTFT measured against 0b5371e. Known costs that stay after the proof:
  decode waves (scatter mode) add a query all-gather and a partial all-to-all
  per layer, with the index candidates riding the gather on the 21
  full-indexer layers; folding the query into the projection gather and
  sending bf16 partials are the planned cuts. Decode attention is not
  bit-exact against the replicated split kernel (a different summation
  order); the host and device oracles compare against the replica-view shard
  math, and the index selection and prefill attention are bit-identical
  (`tests/cuda/index_shard_cuda.cu`, `tests/host_cuda/index_shard_host.cu`).
  Prefill waves (gather mode) carry one sequence, so packed multi-span
  prefill is off under the split. They exchange only the latent keys of
  the context before the wave, in balanced chunks, and read them in place
  through a per-position remap; the wave's own rows come from each rank's
  local copy, and a row digest folded into one 2-word MAX all-reduce per
  wave fails the request loudly if any rank's local rows differ. DSA
  selection on full-indexer layers past 2,048 positions is sharded: each
  rank scores every row only against the keys it owns, keeps its exact
  top-k, sends the candidates to the rank that owns the row (one
  all-to-all), that rank merges them into the exact replicated top-k, and
  one all-gather returns every row's selection to every rank. On one GB10
  this cut per-layer selection from 20.5 ms to 2.0 ms at 64K and from
  112 ms to 5.7 ms at 256K for 1024 rows (the replicated layout repeats the
  same selection on all 16 ranks). Graph mode sizes the latent exchange and
  the selection keep from the regime bound. Execution rows must be a
  multiple of the context-split degree (refused at allocation otherwise).
  Still open: fetching only the selected union's latents, once its size is
  measured on the fleet. Scatter waves are capped at 64 rows.
- Left out on purpose (2026-10-02): KV memory is owned by each engine, not by
  the node. `SparkStageKvBindingInitialize` allocates the KV regions and the
  page table with `cudaMalloc` through the module's own ledger
  (`runtime/stage_kv_binding.c:231`, `:234`;
  `runtime/stage_module_common.c:744-786`). The backing directory reaches the
  binding through the driver's adapter and module
  (`spark_glm52_serving_adapter.c:666-667`,
  `spark_glm52_resident_decode_stage_module.c:327-328`). weightd has no KV
  pool, lease or prefix-share code (`runtime/spark_weightd.c`,
  `node/weightd*.c`). The pools die with residentd, so a restart loses every
  resident prefix, co-resident drivers cannot share KV memory, and no
  node-wide KV budget is enforced across engines. Close it by having weightd
  own KV pools, leases, refcounted prefix shares, copy-on-write forks and
  per-node budgets, with the binding as its client. The fleet proof is a
  residentd restart that reattaches the device pages and serves a prefix hit
  with no NVMe reads, with the per-engine private KV bytes reported as zero.

## KV tiers

- Left out on purpose (2026-10-02): JIT-KV W3
  (`docs/archive/JIT_KV_RESPONSE.md:47-49`) is open. The tree holds five KV
  stores and two unwired residency layers: the anonymous page store
  `cache/kv_page_store.c`, the slot file `runtime/spark_kv_backing.c`, the
  digest-checked slot index `cache/nvme_tier.c` (no file I/O of its own,
  `include/sparkpipe/spark_nvme_tier.h:57-64`), the prefix snapshot files
  `cache/kv_snapshot.c`, the external provider client `cache/store/kv_store.c`
  with `cache/store/stage_kv_client.c`, plus the pager `cache/kv_pager.c` and
  the header-only `LmCache` in `cache/cache.h` that only `tests/test_cache.c`
  includes. `kv_snapshot.c` is linked but `SparkKvPageCacheAttachSnapshot`
  (`cache/kv_page_cache.c:1480`) has no production caller, and the provider
  store stays off unless `SPARK_<FAMILY>_STAGE_KV_STORE` names one
  (`include/sparkpipe/family/module/spark_module_open_kv_tier.h:27-31`), so
  production spills only to the page store, the one with no persistence and no
  integrity check. Close it by building one store on the `kv_snapshot.c`
  format behind the pager and the binding and deleting the rest, proven on the
  fleet by a spill, residentd restart and restore run that hits through that
  store alone.
- Left out on purpose (2026-10-02): The pager and its stores have no
  production caller. `cache/kv_pager.c` is in no library (`sources.mk:72-81`)
  and is built only into the `test_jit_kv_*` targets (`Makefile:1646-1667`);
  `cache/nvme_tier.c` is linked into the model common library
  (`sources.mk:81`) but is called only by `kv_pager.c` and
  `scheduler/topology_switch.c`, which only `test_topology_switch` builds
  (`Makefile:1671-1672`); `runtime/spark_kv_backing.c` is built only into
  `tools/spark_kv_backing_test.c` (`Makefile:1267-1268`); and
  `modules/dsv4_resident_decode_stage/source/spark_dsv4_jit_kv.c` is not a
  dsv4 module source (`modules/dsv4_resident_decode_stage/Makefile:15-31`).
  Whole-lane park and restore, the park budget and restore-bandwidth admission
  run only in host tests, so no served model can park a lane. Close it by
  wiring one pager into the common KV binding (`runtime/stage_kv_binding.c`)
  and deleting the unwired copies, proven by a fleet backpressure run at 2x
  device pages where parked lanes restore bit-exact at B1 and B16.
- Left out on purpose (2026-10-02): No production code attaches a KV snapshot
  store, so GLM-5.3 Full keeps no prefix across a residentd restart or a model
  unload (req 15). `SparkKvPageCacheAttachSnapshot`
  (`cache/kv_page_cache.c:1480`) and `SparkKvSnapshotStoreOpen`
  (`cache/kv_snapshot.c:530`) are called only from `tests/test_kv_snapshot.c`
  and `tests/test_kv_snapshot_cuda.c`, and `SparkStageKvBindingInitialize`
  (`runtime/stage_kv_binding.c:198-259`) opens only the anonymous `O_TMPFILE`
  spill store (`runtime/stage_kv_binding.c:176`,
  `cache/kv_page_store.c:179-185`), which disappears with the process. The
  PREPARE restore (`cache/kv_page_cache.c:2006-2014`) and the saves at
  completion and release (`cache/kv_page_cache.c:2184`, `:2072`) are gated on
  `cache->snapshot` and never run. Close it by opening the store and attaching
  it in `SparkStageKvBindingInitialize` from a required deployment field whose
  absence fails the load, after saves move off the CUDA host callback and the
  engine can find stored prefixes after a restart (both below). Fleet proof:
  send a 4K prompt, restart residentd on all 16 ranks, resend it, and see
  `cached_tokens > 0`, `restore_count > 0`, `restore_failure_count == 0` and
  tokens identical to the first run.
- Left out on purpose (2026-10-02): The KV snapshot key carries no model,
  pack, codec or shard identity for GLM-5.3 Full. The binding's
  `layout_fingerprint` is the constant string
  `latent-bf16-page-major-index-bf16-layer-major-v1`
  (`modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:749`),
  and nothing reads it: `SparkStageKvBindingFillTable` copies it with
  `model_id` and `model_revision` into `SparkKvModelTable`
  (`runtime/stage_kv_binding.c:193-195`), and `SparkKvBackendInitialize`
  (`cache/kv_model_table.c:40-90`) ignores all three. The snapshot key is
  `layout_sha256` plus a SHA-256 of the prompt tokens alone
  (`cache/kv_page_cache.c:1557-1563`, `runtime/model_batch_engine.c:447-452`),
  and `layout_sha256` is whatever the attacher writes into
  `SparkKvPageCacheSnapshot` (`include/sparkpipe/spark_kv_page_cache.h:77`),
  so once a store is attached, files written under a different pack, contract,
  expert or KV codec, driver binary, TP rank or owner count restore as hits
  with wrong KV. Close it by computing `layout_sha256` in the common binding
  from model id and revision, pack SHA-256, contract SHA-256
  (`GLM_CONTRACT_SHA256`), expert and KV codecs, driver binary SHA-256,
  `owner_rank`, `owner_count`, block size and page geometry, and deleting the
  unused string fields. Fleet proof: save a prefix, restart with a different
  expert codec or rank-to-node assignment and see the resent prompt miss the
  store (`restore_miss_count` rises, tokens exact by recompute), then restart
  on the original build and see it hit.
- Left out on purpose (2026-10-02): Attaching a snapshot store to the GLM-5.3
  Full binding would run synchronous CUDA copies inside a CUDA host function.
  `SparkGlmStageEnqueueAsyncCompletion` queues `SparkGlm52CompleteAsync` with
  `cudaLaunchHostFunc`
  (`common/common_glm_stage_module/spark_glm_stage_module.h:161`, called from
  `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:1821`);
  the callback calls `SparkStageKvBindingFinish`
  (`spark_glm52_resident_decode_stage_module.c:1849`), which for a PUBLISH
  lane reaches `SparkKvPageCacheSaveSequence` (`cache/kv_page_cache.c:2184`),
  `SaveChain` (`:1658`), `SaveEntry` (`:1635`), `SnapshotPage` (`:1609`) and
  the binding's `cudaMemcpy` (`runtime/stage_kv_binding.c:17-19`). CUDA does
  not permit CUDA API calls in a host function, so the first save after
  attachment breaks the execution stream's completion path. The release-path
  save (`cache/kv_page_cache.c:2072`) also copies device to host synchronously
  on the admission thread while holding the binding mutex
  (`runtime/stage_kv_binding.c:315`), which every completion needs. Close it
  by having the callback and the release path only mark the terminal entry for
  saving, and a common binding worker copy it with an event and
  `cudaMemcpyAsync` into the snapshot ticket outside the callback and the
  mutex. Fleet proof: with the store attached, a B16 run of published prompts
  ends with `save_count > 0` and `save_failure_count == 0`, and B1 decode step
  time stays within run-to-run noise of the build before the change.
- Left out on purpose (2026-10-02): PR #1278 (`5815cf10f`, merge base
  `30cccaaf7`, 2026-09-28) holds the only snapshot wiring and prefetch-join
  code, is not in this tree, and cannot be merged as is. It puts the store
  open, the layout digest and the snapshot directory in glm5_next driver code
  (`SparkGlm5NextSnapshotLayout` and `SparkGlm5NextSnapshotInitialize` in the
  module, `kv_snapshot_directory` and `kv_snapshot_maximum_bytes` in the
  adapter), against central KV ownership. Its adapter treats both keys as
  optional and serves with `kv_snapshot=off` when they are absent, which is an
  opt-out of required persistence. Its PENDING retry
  (`SparkModelBatchRequeuePrefetchWave`) doubles a 10 ms backoff up to 200 ms
  for at most 10,000 tries instead of waiting on a restore deadline. Close it
  by porting only `cache/kv_snapshot.c`, the `cache/kv_page_cache.c` prefetch
  and join, and the engine PENDING path onto `runtime/stage_kv_binding.c` and
  `runtime/model_batch_engine.c`, with a required snapshot field, a
  deadline-bounded wait and no driver snapshot code. Fleet proof: GLM-5.3 Full
  restores a saved prefix with no glm52 snapshot code, and the engine log
  shows each PENDING wait ending at the restore's completion or its deadline.
- Left out on purpose (2026-10-02): No report shows whether a snapshot restore
  or save happened or what it cost. `SparkKvPageCacheSnapshot` keeps save,
  restore, miss, corrupt and failure counts and nanosecond totals
  (`include/sparkpipe/spark_kv_page_cache.h:83-93`, updated at
  `cache/kv_page_cache.c:1644-1645` and `:1865-1881`), but nothing in
  `runtime/`, `node/` or the glm52 module reads them; the one reporter,
  `55a9241a6`, prints them from the glm5_next module and is not in this tree.
  A restart benchmark therefore has no `restore_count` or save time to check.
  Close it by exporting the snapshot counters through the common binding into
  the driver runtime snapshot and the engine measurement view that
  `node/model_api.c` reports. Fleet proof: the restart run's report shows
  `restore_count > 0`, `restore_failure_count == 0`, and save and restore time
  per page on every rank.
- Left out on purpose (2026-10-02): Crash recovery of the KV snapshot store
  has never run in production. The store writes each file to a `.kvs-writing-`
  temporary opened with `O_EXCL` and mode 0600, fsyncs it, renames it and
  fsyncs the directory (`cache/kv_snapshot.c:331-358`), and
  `SparkKvSnapshotStoreOpen` deletes leftover temporaries before indexing the
  store (`cache/kv_snapshot.c:501-505`, `:563`). Nothing outside the tests
  opens the store, so this path is unreachable on the fleet and no
  kill-during-write case has run. Close it with the production store
  attachment, then run the crash case on the fleet: kill -9 residentd on one
  rank while saves are queued, restart it, and check that no `.kvs-writing-`
  file remains, `removed_temporary_count` equals the leftovers, and the resent
  prompt restores with tokens identical to an uninterrupted run.
- Left out on purpose (2026-10-02): GLM-5.3 Full takes its KV backing
  directory through a driver-private path. The adapter copies
  `kv_backing_directory` and `kv_backing_maximum_bytes` into the glm52 node
  context
  (`modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c:666-667`,
  fields at
  `modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h:83-84`)
  and the module reads them from there
  (`modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:327-328`),
  although the common host services already carry both values
  (`include/sparkpipe/spark_module_abi.h:43-44`, filled by
  `runtime/pack/driver_compiler.c:591-592` from
  `runtime/serving_adapter_template.c:559-562`). No snapshot directory field
  exists in the deployment, the host services or the binding configuration, so
  KV storage policy stays in the driver and a persistence store has no common
  source. Close it by reading the backing and snapshot directories only from
  host services inside `SparkStageKvBindingInitialize`, adding a required
  `kv_snapshot_directory` and byte budget to the deployment node and host
  services, and deleting the glm52 node-context fields. Fleet proof: GLM-5.3
  Full loads with the glm52 fields removed, and the binding log on every rank
  names the host-service backing and snapshot directories.
- Left out on purpose (2026-10-02): `tools/glm52_gen_deployment.py:112-113`
  renders `kv_logical_page_capacity` equal to `kv_physical_page_capacity` (16
  x 512 pages), so a GLM Full deployment rendered by it has zero spill pages
  (`runtime/stage_kv_binding.c:138`) and every eviction under device pressure
  discards cached KV. The lane renderer already adds `kv_backing_bytes //
  KV_PAGE_BYTES` spill pages (`tools/glm53full_lane.py:133-134`, c7edad09e)
  and refuses a non-positive backing size (`:164-165`); the TP8 generator was
  never updated. Close it by rendering logical > physical in
  `glm52_gen_deployment.py` the same way, or by deleting it in favour of the
  lane renderer, proven by the binding load line
  (`runtime/stage_kv_binding.c:256-257`) showing logical_pages >
  physical_pages on every rank of the rendered deployment.
- Left out on purpose (2026-10-02): Evicting a prefix-cache entry destroys it
  in every tier: `SparkKvPageCacheEvictEntry`
  (`cache/kv_page_cache.c:395-432`) calls `SparkKvPageCacheDiscardLogicalPage`
  (`:326-352`), which invalidates the page-store copy (`:334-345`) and frees
  the logical page. It runs when logical pages or entries run out (`:434-441`,
  `:443-460`) and when a page cannot be made resident (`:899-910`); the only
  demotion is the arena's per-page write-back into the process-lifetime page
  store (`cache/kv_cache.c:1186-1229`). Once logical pages are exhausted an
  evicted prefix is gone and its next request recomputes it. Required:
  eviction demotes an entry to the next tier and discards only from the last
  one. Close it by moving entry eviction onto the tier chain, proven on the
  fleet by a run at 2x device pages where a prompt evicted from the device and
  logical pools is restored from the lower tier with `cached_tokens` covering
  it and tokens equal to an uninterrupted run.
- Left out on purpose (2026-10-02): There is no host-memory KV tier. The
  binding's only host KV memory is one pinned staging page
  (`runtime/stage_kv_binding.c:87-91`, `:182-183`), and a page leaving the
  device pool goes straight to the page-store file (`:172-176`), while
  `README.md:261-263` says pages move between GPU memory, host memory, NVMe
  and an external store. Pages that fit in host DRAM go to disk or are
  discarded, and README describes a tier that does not exist. Close it with a
  bounded host DRAM tier between the device pool and the NVMe store, sized by
  a deployment field whose absence fails the load, proven on the fleet by a
  spill run whose demoted pages are restored from host memory with the
  page-store read count unchanged and tokens equal to an uninterrupted run.
- Left out on purpose (2026-10-02): The arena has one evict hook and two
  owners: the common binding installs the page store's
  `SparkKvPageStoreWriteback` (`runtime/stage_kv_binding.c:172-173`), and
  `SparkKvPagerInitialize` refuses with `BUSY` when any other hook is set
  (`cache/kv_pager.c:340-344`) before installing its own (`:371-372`). The
  pager therefore cannot attach to any binding arena, and wiring it means
  choosing one owner of eviction. Close it by making the pager the only evict
  hook with the surviving store as its backing, proven on the fleet by a spill
  run whose write-backs are counted by the pager's statistics, not the page
  store's.
- Left out on purpose (2026-10-02): A deployment node with no
  `kv_backing_directory` still spills KV under `/tmp` in three drivers:
  glm5_next
  (`modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:1677-1685`),
  laguna
  (`modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_module.c:790-798`)
  and ling
  (`modules/ling_resident_decode_stage/source/spark_ling_resident_decode_stage_module.c:648-656`).
  71c2a7692 made the common binding refuse the load
  (`runtime/stage_kv_binding.c:209-213`), but the deployment loader still
  accepts a missing directory (`runtime/model_resident_deployment.c:204-206`,
  `:656`), so these drivers silently put KV outside the KV partition. Close it
  by making `kv_backing_directory` required in the deployment loader and
  deleting the three fallbacks, proven by a residentd load on a Spark with the
  field removed that fails with a named error for every driver. glm5_next and
  laguna also size that backing to `page_count * payload_bytes` and ignore
  `kv_backing_maximum_bytes`
  (`spark_glm5_next_resident_decode_stage_module.c:1686`,
  `spark_laguna_resident_decode_stage_module.c:799`), so spilled pages land
  outside the deployment's declared storage path and budget.
- Left out on purpose (2026-10-02): Every production page store is anonymous:
  the common binding (`runtime/stage_kv_binding.c:176`) and the glm5_next
  (`modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:1581`,
  `:1673`), laguna
  (`modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_module.c:786`),
  ling
  (`modules/ling_resident_decode_stage/source/spark_ling_resident_decode_stage_module.c:644`)
  and dsv4
  (`modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:1326`)
  modules set `SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS`, and the store then opens
  an unnamed `O_TMPFILE` (`cache/kv_page_store.c:174-185`). Spilled KV dies
  with the process, so a residentd restart or crash loses every spilled prefix
  and the next request recomputes it. The named-path helper
  `SparkKvPageStoreBuildPath` (`cache/kv_page_store.c:75-101`) has no caller.
  Close it by spilling into the persistent store, proven on the fleet by a
  prompt spilled before a residentd restart that hits after it with tokens
  equal to an uninterrupted run.
- Left out on purpose (2026-10-02): The page store verifies nothing it reads
  back: a page counts as valid when its flag and generation match
  (`cache/kv_page_store.c:1131-1136`), and the write path records only the
  generation (`:424-440`), with no per-page digest. A torn write, bad sector
  or stray write to the backing file restores wrong KV that attention consumes
  without error. `cache/kv_snapshot.c` already hashes each segment with
  SHA-256 on write (`:323`) and checks it on read (`:945-946`, `:985-986`),
  and `cache/nvme_tier.c:1272-1291` checks a SHA-256 on landing. Close it by
  storing a SHA-256 per spilled page and answering NOT_FOUND on mismatch,
  proven on one Spark by corrupting a spilled page and observing a counted
  digest mismatch followed by a recompute with correct tokens.
- Left out on purpose (2026-10-02): Eviction ignores request priority and
  deadline. Victims are the least recently used entry
  (`cache/kv_page_cache.c:354-361`), the oldest resident entry (`:363-393`),
  or in the arena the block with the lowest reference count and then reuse
  value or recency (`cache/kv_cache.c:1094-1116`); the frame priority is
  copied into the admission request (`cache/kv_page_cache.c:2137`) only to be
  compared on retry (`:1904`). Pages pinned by a running transaction are
  protected (`:317-324`, `cache/kv_cache.c:1147-1151`), but a high-priority
  request's cached state is evicted as readily as a low-priority one's,
  against `docs/archive/JIT_KV_DESIGN.md:81-86`. Close it by ranking victims
  by owning-request priority and deadline before reuse value, proven on the
  fleet by an oversubscribed run with two priority classes where the higher
  class keeps its hits and its TTFT stays flat.
- Left out on purpose (2026-10-02): A full backing store neither tightens
  admission nor logs. When the page store has no free slot it returns
  `CAPACITY_EXCEEDED` (`cache/kv_page_store.c:786-793`); the arena then drops
  an unreferenced block's contents and bumps `write_back_degraded_block_count`
  (`cache/kv_cache.c:1201-1210`), or the page cache discards its least
  recently used entry and retries (`cache/kv_page_cache.c:899-910`). No
  production code reads or prints that counter or `evicted_entry_count`
  (`:430`), so a full store silently turns cached prefixes into recomputes.
  Required (`docs/archive/JIT_KV_DESIGN.md:114-115`): a full store queues new
  work and logs the transition. Close it by feeding store occupancy into
  engine admission with a logged backing-full transition, proven on the fleet
  by a run with a small `kv_backing_maximum_bytes` that logs the transition,
  queues requests and completes every request without wedging.
- Left out on purpose (2026-10-02): No code budgets or reports NVMe write
  endurance. The page store and snapshot store count written bytes
  (`cache/kv_page_store.c:439`, `cache/kv_snapshot.c:390`), but nothing reads
  those counters and no limit stops spill writes once they pass the
  drive-writes-per-day budget (`docs/archive/JIT_KV_RESPONSE.md:26-28` sets
  0.1-0.3 DWPD). A thrashing workload wears out the KV NVMe with no signal.
  Close it with a per-drive write budget in the deployment, enforced by
  refusing further spill writes (recompute instead) once the rolling budget is
  spent and reported in the wave timeline, proven on one Spark by a run with a
  small budget that reports the write rate, stops spilling at the limit and
  keeps serving.
- Left out on purpose (2026-10-02): Copy-on-write of a partial prefix page and
  snapshot restore run synchronously on the residentd submission thread while
  the KV binding lock is held. `SparkStageKvBindingAdmit` holds
  `binding->mutex` (`runtime/stage_kv_binding.c:315-328`) around the lane
  prepare, which clones the page through `SparkKvPageCacheCloneMutable`
  (`cache/kv_page_cache.c:972-1010`, called at `:1063`) into
  `SparkKvPageStoreCopyResidentPage` (`cache/kv_page_store.c:629-663`):
  blocking `cudaMemcpy` calls (`runtime/stage_kv_binding.c:17-19`) move the
  whole page device to host to device through the store's one staging page
  under the store mutex, and the copy answers `BUSY` while any spill transfer
  is queued (`kv_page_store.c:649-651`). `SparkKvPageCacheRestorePrefix` is
  called inline in the same prepare (`cache/kv_page_cache.c:2006-2014`); it
  does not run today only because no production code attaches a snapshot
  (`SparkKvPageCacheAttachSnapshot`, `:1480`, has test callers only). Nothing
  answers `PENDING`; the engine polls `BUSY` with a 10 to 200 ms backoff
  (Dynamic batching, JIT KV admission does not prefetch). Close it by making
  copy-on-write a device-to-device copy ordered on the execution stream and by
  reading restores outside the lock, with prepare answering `PENDING` until
  the pages are in place; the fleet proof is a B16 decode run that admits
  copy-on-write and restored prefixes mid-run with unchanged per-step decode
  time and tokens identical to an uninterrupted run.
- Left out on purpose (2026-10-02): The glm52 completion host function waits
  on the same lock as admission. `SparkGlm52CompleteAsync` runs as a
  `cudaLaunchHostFunc` host function
  (`common/common_glm_stage_module/spark_glm_stage_module.h:161`) and calls
  `SparkStageKvBindingFinish`
  (`modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:1849`),
  which locks `binding->mutex` (`runtime/stage_kv_binding.c:578`); the submit
  path takes the same lock (`:512`, `:533`). While an admission holds it for a
  copy-on-write copy, or for a restore once a snapshot is attached, every
  finishing wave's host function blocks, and its stream runs nothing enqueued
  after it until it returns. The stall has not been measured for GLM Full.
  Close it by keeping restore and copy work out from under the lock the
  completion path takes; the fleet proof is B16 per-step decode time with
  prefix admissions arriving during decode, equal to the same run without
  admissions.
- Left out on purpose (2026-10-02): a block whose spill write-back fails is
  degraded (`cache/kv_cache.c`, `SparkKvCacheArenaEvictResidentBlock`): its
  contents are dropped, its backing is marked invalid, and a later restore of
  it answers NOT_FOUND so the engine recomputes, never a wedge. The page-cache
  entry that owns the block stays VALID, so every later request whose chain
  matches it reaches the same missing page and recomputes again, and a
  persistent disk error turns every spill into a recompute with no signal
  beyond `write_back_degraded_block_count`, which nothing reports. (From
  2026-10-01 to 2026-10-02 the degrade was limited to unreferenced blocks,
  which made a failed write-back fail the request instead; that broke the B1
  drop-and-recompute contract tested by `tests/test_jit_kv_wire.c` scenario 5
  and was reverted.) Close it by evicting the owning page-cache entry and its
  descendants when a write-back fails, and reporting the degraded count,
  proven by a fleet run with injected write errors in which the first affected
  request recomputes, later requests miss cleanly without touching the bad
  page, and tokens match an uninterrupted run.
- Left out on purpose (2026-10-02): The rule for a missing, corrupt or
  unreadable KV page is not written down (JIT KV plan owner decision 5), and
  the spill store GLM Full uses cannot detect corruption. A missing prefix
  answers `NOT_FOUND`, and the engine tombstones the prompt and recomputes it
  (`runtime/model_batch_engine.c:772-835`); the snapshot path turns a SHA-256
  mismatch into `NOT_FOUND` (`cache/kv_page_cache.c:1863-1867`), but no
  production code attaches a snapshot. The page store keeps no digest:
  `SparkKvPageStoreExecuteRead` (`cache/kv_page_store.c:332-350`) copies
  whatever `pread` returns into the KV pool, so a corrupt spill page is
  attended as valid KV. A spill read error returns `IO_ERROR`, the engine
  fails the request while every rank is connected (`model_batch_engine.c:882`,
  `:901-902`), and the page keeps its backing record
  (`kv_page_store.c:424-456`), so every later request whose chain matches it
  reads the same page and fails again. Close it once the owner confirms the
  rule: a per-page digest in the spill store whose mismatch evicts the entry
  and its descendants and answers `NOT_FOUND`, plus the confirmed treatment of
  read errors applied to the entry; the fleet proof injects a corrupt page and
  a read error and shows recompute or failure exactly as the rule says, with
  no later request reusing the bad page.
- Left out on purpose (2026-10-02):
  `include/sparkpipe/family/module/spark_module_open_kv_tier.h:27-31` treats
  an unset `SPARK_<FAMILY>_STAGE_KV_STORE` as provider `none`: it opens a
  disabled store client (`cache/store/stage_kv_client.c:15-16`) and returns OK
  with the tier off. qwen38_max, qwen4_flash and muse_glimmer, the three
  modules that include it
  (`spark_qwen38_max_resident_decode_stage_module.c:596`,
  `spark_qwen4_flash_resident_decode_stage_module.c:685`,
  `spark_muse_glimmer_resident_decode_stage_module.c:372`), therefore load
  with no JIT KV tier unless an operator sets the variable, and with the tier
  off `spark_module_kv_prepare_frame.h:8-9` returns OK without resolving,
  evicting or restoring a block. An environment default waives required JIT
  behaviour (I03, I23). Close it by deleting the `none` provider so a missing
  store fails module open with an error naming the variable, and before any of
  these drivers loads again prove on the fleet that a pool smaller than the
  working set evicts and restores blocks with tokens equal to an uninterrupted
  run.
- Left out on purpose (2026-10-02): The serving adapters overwrite the store
  variable with `none` before loading the module:
  `model-families/common/include/sparkpipe/spark_qwen38_pp_serving_adapter_common.h:111-115`
  (called at `:737`) does it for gemma4, minimax, muse_glimmer, qwen38_max and
  qwen4_flash, and
  `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:682-686`
  does it for qwen38_27b, whose module then opens a disabled client
  (`spark_qwen38_27b_resident_decode_stage_module.c:1919-1923`). Because
  `setenv` overwrites, no deployment can turn the JIT KV tier on for these six
  drivers through the serving path, and minimax's module reads no store
  variable at all. All six are refused at load today because none declares
  PREFIX_REUSE (`runtime/model_serving_adapter.c:195-198`), so the waiver is
  latent. Close it by deleting the `STAGE_KV_*` assignments in both adapters,
  and before a driver loads again prove on the fleet that its log shows
  `kv_tier_enabled` (`cache/store/stage_kv_client.c:46`) with a real provider.
- Left out on purpose (2026-10-02):
  `modules/gemma4_resident_decode_stage/source/spark_gemma4_resident_decode_stage_module.c:405-421`
  has no JIT KV tier: an unset `SPARK_GEMMA4_STAGE_KV_STORE` becomes `none`
  and returns OK, and any real provider fails with `kv_provider_unsupported`.
  gemma4 serves only from the blocks the shared adapter allocator gives each
  lane and can never evict or restore a block (I23, I25). Close it by moving
  gemma4 onto the common KV frame or the common binding
  (`runtime/stage_kv_binding.c`) with no `none` path, and before it loads
  again prove on the fleet that evicted blocks restore with tokens equal to an
  uninterrupted run.
- Left out on purpose (2026-10-02): The common KV frame names each stored
  block `kv/<model_fp>/<layout_fp>/r<rank>/s<sequence_id>/b<block>`
  (`cache/store/stage_kv_client.c:50-56`). `model_fp` is a 64-bit FNV-1a hash
  (`runtime/stage_module_common.c:2098-2107`) of the expected stage-pack
  header
  (`include/sparkpipe/family/module/spark_module_open_kv_tier.h:41-42`), and
  the qwen38_max, muse_glimmer and qwen4_flash headers hold shapes only, with
  no weight revision or content digest
  (`spark_qwen38_max_stagepack_format.h:64-94`,
  `spark_muse_glimmer_stagepack_format.h:38-68`,
  `include/sparkpipe/spark_stagepack_format.h:83-113`). Two model revisions
  with the same shapes write under the same key prefix, and the key names the
  request's sequence id rather than its token prefix, so one request can never
  restore a block another request stored. Close it by keying blocks on the
  model revision and contract digest plus the engine's prefix identity, with a
  cryptographic digest in place of FNV, and prove it on the fleet with two
  revisions sharing one store without cross-hits and a second request
  restoring the first request's prefix.
- Left out on purpose (2026-10-02):
  `include/sparkpipe/family/module/spark_module_kv_prepare_frame.h:8-9`
  returns OK and leaves the frame untouched when the tier is off, and with the
  tier on `:10-11` refuses any frame without a decode batch. qwen4_flash's
  prefill path calls it with a prefill context
  (`spark_qwen4_flash_resident_decode_stage_module.c:1993`), and a prefill
  frame carries no decode batch (c9649ff02), so turning the tier on fails
  every qwen4_flash prefill with `INVALID_ARGUMENT` and the tier-off early
  return is the only working path. Close it by preparing prefill rows from the
  prefill view in the template and deleting the tier-off return once the tier
  is mandatory, then prove on the fleet that qwen4_flash prefills and decodes
  with the tier on and a pool smaller than the working set, with tokens equal
  to an uninterrupted run.
- Left out on purpose (2026-10-02): With the tier on, the GDN record size is
  the constant `MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES` (4096 in
  `spark_qwen38_max_resident_decode_stage_module.c:41`,
  `spark_muse_glimmer_resident_decode_stage_module.c:34`,
  `spark_qwen4_flash_resident_decode_stage_module.c:51`), which sizes the plan
  and the staging buffer
  (`include/sparkpipe/family/module/spark_module_open_kv_tier.h:58`, `:73`).
  The only eviction call passes `include_gdn_state` 0
  (`common/common_kv_frame.h:168`) and every queued restore sets
  `gdn_nonresident` 0 (`:316`), so no GDN recurrent or convolution state is
  ever stored or restored. A hybrid sequence can be paged only while its
  recurrent state stays in its slot; it cannot be moved, restored after
  release or shared as a prefix (I24). Close it by sizing the record from the
  real per-sequence GDN state and storing it with the sequence's blocks, and
  prove on the fleet that a hybrid sequence evicted whole and restored
  mid-decode matches the uninterrupted run token for token.

## Prefix reuse (I23)

Prefix reuse is required and non-compliant adapters are refused at load
([`docs/DRIVER_ACCEPTANCE.md`](docs/DRIVER_ACCEPTANCE.md), Prefix reuse is
required).

Left out on purpose (2026-10-02): the GLM-5.3 Full builds that have run on the
fleet recompute every prompt from position 0. The deployed lanes predate the
restore code: they ran with the `prefix_reuse` opt-out until 2026-10-01, and
the glm52 restore (real page tables, index keys in the payload, prefix-aware
continuity, the kernel view bound by the physical pool, the
final-partial-block publish) exists only on the unmerged branches
`fix/prefix-reuse-required` (#1416) and `kv/sequence-shard`. Every repeated
system prompt and every follow-up turn is prefilled in full. Close it by
passing the TP16 fleet I27 gate on those branches (below), merging them and
deploying that build, proven by `usage.prompt_tokens_details.cached_tokens` >
0 on repeated prompts in production logs with T1 exact. From 2026-09-28 to 2026-10-01 two opt-outs (the
`SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE` skip in the batch engine
and the deployment `prefix_reuse` field) let these adapters serve while
silently recomputing every prompt; GLM-5.3 Full ran that way. The load
check reads only the `PREFIX_REUSE` descriptor bit
(`runtime/model_serving_adapter.c:195-199`) and asks for no I27 proof. Each
adapter below lacks real restore, an I27 proof, or both:

- Left out on purpose (2026-10-02): glm52 (GLM-5.3 Full) has restored prefixes
  through the common binding since `c7edad09e` (branch `kv/sequence-shard`).
  Lanes get real page tables (`SparkStageKvBindingClaim` and
  `SparkStageKvBindingUploadPageTables` at
  `spark_glm52_resident_decode_stage_module.c:2274` and `:2283`, read by the
  wave at `:861`), and the DSA index keys are page payload region 1
  (`:737-739`). A lane restored mid-sequence on a fresh slot passes continuity
  (`runtime/stage_kv_binding.c:462-469`). The adapter declares `PREFIX_REUSE`
  (`spark_glm52_serving_adapter.c:185`), so it loads without the I27 proof
  this section requires. The only proof is `tools/glm52_prefix_probe.c`, a
  single-GPU rank-0 run with collectives off that prints `rank-local
  computation only` (`:345`). Until a TP16 fleet I27 run passes, restored
  GLM-5.3 Full prefixes are unproven on the real lane. That run needs cold vs
  warm token parity at B1 and B16, a copy-on-write mid-block prefix, abort,
  reset, and an evicted prefix that comes back NOT_FOUND and is recomputed.
- qwen38_27b has a GDN snapshot borrow for prompt checkpoints
  (`SparkQwen38_27bServingPrefixBorrow`), but a borrow miss logs `recomputing`
  and prefills over unrestored KV blocks and GDN state; decode-lane
  checkpoints are never snapshotted; a publish that finds no free entry or
  more than 64 blocks returns OK without storing; the eight snapshot entries
  evict independently of the engine's index; a borrowed partial last block is
  shared without copy-on-write.
- dsv4 restores paged KV, index and compressor state but speculates without
  the cache-publish work kind, so generated blocks after a multi-token step
  were never published.
- ling attends through an identity page table and keeps no KDA state per
  cached prefix.
- laguna has a real page table and transactions but no I27 proof, no state
  capture hook and logical = physical page capacity.
- Left out on purpose (2026-10-02): gemma4, minimax, muse_glimmer, qwen38_max
  and qwen4_flash share the lane block allocator in
  `model-families/common/include/sparkpipe/spark_qwen38_pp_serving_adapter_common.h:218-249`,
  which gives each lane private blocks from a pool sized to resident capacity
  x blocks per lane (`:731-732`, logical = physical); they have no borrow
  path, and the same header forces the JIT KV tier off (`:111-115`). k3 has no
  borrow path either.

Related common-code debt:

- Left out on purpose (2026-10-02): the publish of a reply's final partial
  block has no fleet proof, and multi-token chains that stop early never
  publish their tail. A completed request with an unpublished tail queues
  `CACHE_PUBLISH` before its release when its last step's tokens were all
  accepted (`runtime/model_batch_engine.c`,
  `SparkModelBatchHandleDecodeCompletion`), the common binding serves the
  publish-only frame (`SparkStageKvBindingPublishFrame`,
  `runtime/stage_kv_binding.c`), glm52 declares `CACHE_PUBLISH`, and every
  adapter must declare it (`runtime/model_serving_adapter.c`,
  `SparkDescriptorCheckRequiredCacheOperations`). When a multi-token chain
  (glm5_next) stops on EOS mid-chain, the lane has already advanced past the
  tail and its KDA recurrent state belongs to that later position, so
  publishing the tail would store mismatched state; the tail stays unpublished
  and a follow-up turn recomputes it. Close the chain case by capturing or
  rolling back recurrent state to the tail through the common recurrent-state
  hook (I-05), and prove both cases with fleet I27 sessions whose second turn
  extends a reply that ended on EOS mid-block, with `cached_tokens` covering
  the whole first reply and tokens identical to an uncached run.
- Left out on purpose (2026-10-02): residentd's slot claim checks slot
  ownership and the lane's request id, generation and sequence id
  (`node/model_residentd.c:714-783`), not position continuity. Continuity is
  checked only inside adapters: the common `SparkStageKvBindingContinuity`
  (`runtime/stage_kv_binding.c:507`) has one caller, glm52
  (`spark_glm52_resident_decode_stage_module.c:2223`). An adapter that skips
  the call accepts a submission whose position jumps ahead in a resident
  sequence (I02). The fix: run the continuity check in residentd or the engine
  for every adapter. It is closed by a fleet run in which a skipped-position
  submission to a GLM Full lane gets an explicit continuity error.
- Left out on purpose (2026-10-02):
  `include/sparkpipe/family/module/spark_module_glm5_next_laguna.h` is a
  family template named after the two drivers that include it
  (`spark_glm5_next_resident_decode_stage_module.c:702`,
  `spark_laguna_resident_decode_stage_module.c:270`). Besides pack-range and
  manifest checks, it carries a second copy of KV code that
  `runtime/stage_kv_binding.c` now owns: `PrefixRestorePending` (`:96-99`, the
  predicate at `stage_kv_binding.c:421-424`) and `UploadPageTables`
  (`:135-152`, the upload at `stage_kv_binding.c:546-570`). Its T1 trace
  prints `G5N-T1` for laguna too (`:108`). `tests/test_dry_law.py:40-44`
  checks family templates only for glm52, kimi, k3, qwen, dsv4, deepseek and
  mimo25 tokens, so the file passes. Close it with the glm5_next and laguna
  move onto the binding: delete the two duplicated functions, rename what
  remains after its behaviour, and add `glm5_next`, `laguna` and `G5N` to
  `FAMILY_TEMPLATE_TOKEN`. The proof is the dry-law test failing on the old
  file and passing on the renamed one.
- `build/libdsv4_pro_tp4_pp4_serving_adapter*` do not compile
  (`SPARK_DSV4_MODEL_DSPARK_SPEC_STEP` undeclared), and
  `build/libdsv4_tp4_pp4_serving_adapter.so` cannot be opened on GPU hosts
  (undefined `SparkTpLaunchMeshHardware`).
- Left out on purpose (2026-10-02): GLM Full declares prefix reuse on this
  branch
  (`modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c:185`)
  but has no fleet I27 proof. Its only restore evidence is
  `tools/glm52_prefix_probe.c`. The probe runs the rank-0 pack on one GPU with
  collectives disabled (`:98-114`), uses two lanes, and sets logical pages
  equal to physical so nothing spills (`:109-110`). It prints "rank-local
  computation only" (`:345`), prints its swapped-prefix sensitivity control
  without failing when no lane changes (`:300-309`), and no Makefile target or
  gate builds or runs it. The tree has no I27 session, scorecard or
  `sessions/` file, so nothing compares restored with uninterrupted execution
  at TP16 for B1 and B16, spill eviction and readback, a failed write-back,
  abort mid-prefill, reset or residentd restart. Close it with a fleet I27
  session on the 16-node lane that runs each case against an uninterrupted
  control and requires identical tokens, `cached_tokens` above zero on every
  expected hit, and exact T1.
- Left out on purpose (2026-10-02): A reconnect or restart of residentd throws
  away the engine's prefix index, and nothing rebuilds it from a durable
  store. residentd gives each client connection a new generation
  (`node/model_residentd.c:1232`), the pipeline folds it into the session
  fingerprint (`runtime/model_pipeline_client.c:716-728`), and the engine
  resets its prefix index when the fingerprint changes
  (`runtime/model_batch_engine.c:2485-2499`, `SparkPrefixCacheReset` at
  `:2471`); a restarted engine also starts empty (`:1256`), and
  `cache/prefix_cache.c` has no load or rebuild path. The engine marks a lane
  PREFIX only from its own index lookup
  (`runtime/model_batch_engine.c:767-770`, `:2046-2051`), and PREPARE restores
  only PREFIX lanes (`cache/kv_page_cache.c:2009`), so after a restart no
  request reaches the snapshot store and every prompt is recomputed. Close it
  by rebuilding the engine index from the ranks' snapshot store keys after a
  reconnect, or by letting an index miss probe the store with the same
  token-chain identity. Fleet proof: publish a 4K prompt, restart residentd on
  all ranks, resend it, and see `cached_tokens` equal to the published prefix
  with tokens identical to the first run.
- Left out on purpose (2026-10-02): `SparkQwen38_27bServingPrefixPublish`
  (`modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:901-958`)
  returns OK without storing when the prefix spans no block, more than 64
  blocks or more blocks than the lane holds (`:906-907`), and when all eight
  entries (`spark_qwen38_27b_resident_decode_stage_firmware.h:22`) are
  referenced (`:927-928`). The caller treats OK as stored (`:1223-1232`): when
  the lane borrowed an entry earlier in the request (`:978`), it points
  `GDN_PREFIX_SNAPSHOT_OUT` at that entry, and the module writes the longer
  prompt's GDN state into it
  (`spark_qwen38_27b_resident_decode_stage_module.c:2727-2728`) while the
  entry keeps the shorter prefix's identity and token count. A later hit on
  that prefix restores the wrong recurrent state. The adapter is refused at
  load (no PREFIX_REUSE, `spark_qwen38_27b_serving_adapter.c:343-345`), so
  this is latent. Close it by returning an explicit status when nothing is
  stored, never aiming a snapshot at a borrowed entry, and moving the entries
  under the engine's prefix index, and prove it with a fleet I27 run whose
  second request extends a borrowed prefix and whose third request hits the
  original prefix with tokens equal to a cold run.
- Left out on purpose (2026-10-02): On a borrow miss
  `SparkQwen38_27bServingCoverSubmission` logs `qwen38_27b_prefix miss ... -
  recomputing`, disarms the restore and continues
  (`modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:1001-1006`).
  Nothing recomputes: the engine sends a prefix lane's rows from the cached
  position (`runtime/model_batch_engine.c:767-770`), and
  `SparkQwen38_27bServingCoverLane` (`:1027`) gives the lane fresh blocks for
  the prefix positions, so the suffix attends over unwritten KV blocks and an
  unrestored GDN state. The miss happens because the eight snapshot entries
  evict on their own LRU (`:913-925`), not with the engine's prefix index.
  Close it by failing the submission with `NOT_FOUND`, which the engine
  already turns into a full prefill (`runtime/model_batch_engine.c:772-800`),
  and prove it with a fleet I27 run that evicts a snapshot entry the engine
  still indexes and checks tokens against a cold run.
- Left out on purpose (2026-10-02): `tools/glm5_next_bench_wrap.py --api-log`
  computes `all_requests_have_prefix_hits`
  (`tools/glm5_next_bench_wrap.py:164`) but never uses it: `valid` comes from
  the stream checks (`:58`, `:132-174`) and the exit code follows `valid`
  (`:191`). A run in which every request missed the prefix cache
  (`cached_prompt_tokens` 0) reports valid and exits 0, so a warm-cache
  measurement silently becomes a cold one, against I23 (a benchmark requiring
  a hit fails on a miss). Close it with a required-hit mode that sets `valid`
  false and names every request with `cached_prompt_tokens == 0`, used by
  every warm-cache session, and prove it with a fleet warm session against a
  freshly restarted engine that fails and names the missed requests.
- Left out on purpose (2026-10-02): Score-dump row keys are not forgotten when
  a lane is restored. `SparkScoreDumpKeysAdvance`
  (`src/spark_score_dump.c:67-87`) keeps one key chain per resident slot and
  continues it for any position up to the slot's known length. No function
  resets a slot (`include/sparkpipe/spark_score_dump.h:103-144`), and neither
  caller tells it that a restored prefix now owns the slot
  (`spark_glm52_resident_decode_stage_module.c:1540`,
  `spark_glm5_next_resident_decode_stage_module.c:4390`). A restored lane
  whose first row is at position P can land on a slot that last held a
  different sequence of at least P tokens. That row gets a key chained from
  the other sequence and the KEY_VALID flag, although
  `docs/SCORE_DUMP.md:31-33` says it is written keyless. The end record then
  undercounts keyless rows, and the probe lookup uses a wrong key; A/B runs
  are safe only because the corpus rule prevents hits and
  `tools/ab_receipt.py:129-133` refuses cached tokens. Close it by keying the
  chain on the sequence id as well as the slot, or by forgetting the slot when
  a lane binds a new sequence at a non-zero position. The proof is a fleet
  score-dump run that restores a prefix onto a slot last used by a different
  sequence, with the end record counting those rows as keyless.
- Left out on purpose (2026-10-02): glm5_next and laguna were not moved onto
  `runtime/stage_kv_binding.c`, so each keeps a private copy of the KV
  plumbing. That copy covers model table, arena and page store setup
  (`spark_glm5_next_resident_decode_stage_module.c:1596-1724`,
  `spark_laguna_resident_decode_stage_module.c:726-829`), page-table builds
  (`:1418-1440`, `:656-677`), lane transactions (glm5_next `:1873`, `:5861`,
  `:6515`, `:6623`; laguna `:860`, `:1467`, `:1652`) and the upload in the
  driver-named family header. Fixes made in the binding do not reach these
  copies. Both drivers still fall back silently to
  `/tmp/sparkpipe_<model>_kv_<revision>` when the deployment leaves
  `kv_backing_directory` null (glm5_next `:1677-1685`, laguna `:790-798`; ling
  the same at `spark_ling_resident_decode_stage_module.c:648-656`), which
  `runtime/model_resident_deployment.c:205-206` allows. The glm5_next module
  also sizes its backing quota from its page count (`:1686`) instead of the
  stored deployment `kv_backing_maximum_bytes` (`:464`). The binding has no
  hook for per-prefix recurrent state
  (`include/sparkpipe/spark_stage_kv_binding.h:20-46`), and glm5_next's KDA
  layers need one before it can move. Close it by adding that hook, moving
  both drivers onto the binding and deleting the private copies; the proof is
  each driver refusing a deployment with no `kv_backing_directory` and passing
  its TP fleet I27 run (cold vs warm token parity at B1 and B16).
- Left out on purpose (2026-10-02): `SPEC.md:223` says the orchestrator does
  not understand KV layout or JIT-KV policy and that both belong inside model
  firmware. `SPEC.md:157-159` lets a module own resident KV pages, and
  `SPEC.md:252` forbids forcing KV internals into the orchestrator.
  `sparkpipe_invariants.md` gives cache transactions and ownership to common
  code (I02, `:23-26`) and makes JIT and prefix reuse common behaviour (I23,
  `:120-122`), and the tree implements them in `runtime/stage_kv_binding.c`
  and `cache/`. The two contracts that AGENTS.md and this file point to
  therefore disagree on who owns KV. The fix: rewrite SPEC.md section 6 and
  the module ABI paragraphs so that KV pages, cache transactions and JIT-KV
  policy belong to common code, as the invariants say.

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
  `cache/kv_pager.c` (`SparkNvmeTierRequestDemandDeadline`, called at `:691`)
  has no production consumer: `modules/dsv4_resident_decode_stage/source/spark_dsv4_jit_kv.c`
  is built only into `tests/test_jit_kv_wire.c` (`Makefile:1652-1653`), not
  into the dsv4 module, and calls no pager function; glm5_next and the common
  binding do not use the pager. Issue restore demand from enqueue, admit
  against restore bandwidth, and dispatch a lane only after its restore
  completes.
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
- Left out on purpose (2026-10-02): Admission does not account for restore
  bandwidth. The engine bounds lanes by physical pages and in-flight page
  demand only (`runtime/model_batch_engine.c:1758-1786`); the rule that admits
  work only while queued restore bytes over measured drive bandwidth fit the
  slack (`docs/archive/JIT_KV_RESPONSE.md:21-24`) exists only in the unwired
  pager (`cache/kv_pager.c:474-493`, bandwidth estimate at `:78-94`), where a
  zero `restore_slack_microseconds` also switches it off. Nothing bounds how
  many restore bytes are admitted at once. Close it by measuring the backing
  drive's sustained read rate at startup and admitting against it in the
  common engine with no off switch, proven on the fleet by a burst of
  spilled-prefix requests whose admitted restore bytes per second stay at or
  under the measured rate while the decode step time of running lanes is
  unchanged.
- Left out on purpose (2026-10-02): `SparkModelBatchRefreshQueuedPrefix`
  (`runtime/model_batch_engine.c:1578-1583`) fails a request with
  `CAPACITY_EXCEEDED` when its cached prefix ends mid-block and the matched
  pages, plus a copy page and a new page, exceed `kv_physical_page_capacity`.
  It does not fall back to the block-aligned part of the match, or to
  recomputation, so a request that fits the pool when computed is lost (I20,
  I23). `tests/test_model_batch_engine_mock.c:760-791` asserts this failure as
  expected. The fix: when the copy page does not fit, truncate the match to
  whole blocks (or to zero), recompute the tail and report the smaller
  `cached_tokens`. It is closed by a fleet run on a lane with a minimal
  physical page pool in which every repeated partial-block prompt completes.
- Left out on purpose (2026-10-02): When a rank rejects a submission with
  `CAPACITY_EXCEEDED`, the batch engine fails every request in it
  (`runtime/model_batch_engine.c:882-899`). Only `BUSY`, or `IO_ERROR` while
  ranks are disconnected, is retried. The engine sizes prefill spans from the
  block boundary and the `MULTI_BLOCK_PREFILL` bit alone
  (`SparkModelBatchPrefillSpan`, `:1591-1602`). It never shrinks a refused
  span or requeues it after an eviction, so a request that fits in smaller
  spans is lost (I20). The fix: requeue a capacity-refused prefill with a
  smaller span, and fail only a one-block span that cannot fit an empty pool.
  It is closed by a fleet run with long prompts on a small physical pool in
  which every request completes.
- Left out on purpose (2026-10-02): `SparkModelBatchPrefillSpan`
  (`runtime/model_batch_engine.c:1600-1602`) lets a descriptor bit choose
  common chunking policy. With
  `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_MULTI_BLOCK_PREFILL`, a span grows
  to `max_prefill_rows`. Without it, every prefill span stops at the next
  cache block boundary. Only glm52 declares the bit
  (`spark_glm52_serving_adapter.c:184`), so glm5_next prefills one block per
  span whatever the deployment's prefill row bound (I02, I20). The fix: remove
  the bit, size spans from `max_prefill_rows` and page demand for every
  adapter, and require every adapter to execute multi-block spans. It is
  closed by a glm5_next fleet run with output tokens identical to the
  one-block baseline and TTFT recorded for both.

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
- Left out on purpose (2026-10-02): there is no context-lookup drafter, so a
  reply that re-emits text already in the request pays full decode cost for
  every token. Prompt-lookup (n-gram) speculation is a proven technique. Its
  target case is code and document edits, where the reply repeats most of the
  input with small changes and drafts are accepted for long stretches. A
  common, model-neutral draft source in the engine
  (`runtime/model_batch_engine.c`) matches the longest suffix of the request's
  own token history (prompt plus generated) and proposes the tokens that
  followed it last time, with no drafter model and no GPU cost. The draft
  length adapts: it grows while drafts are accepted and drops to none after a
  miss. All k drafted tokens are verified in one multi-row wave, not serially.
  The wave commits the longest prefix whose greedy tokens equal the draft,
  plus the model's own token at the first mismatch, and rejected positions are
  rolled back (free for attention KV; KDA and GDN state need the common
  recurrent-state hook, I-05). It is built for the core drivers first (glm52,
  glm5_next, k3) and shares the common speculation seam and the multi-row
  verify head with the built-in MTP drafters. The verify math on closed PR
  #1398 (k+1 certified rows, accepted-prefix commit) is the starting point.
  Multi-row verify numerics differ from a 1-row decode step by the reordering
  floor, so speculative output is judged against the floor like batched
  output. Close it with a fleet session on GLM-5.3 Full of edit-style requests
  (a long file re-emitted with small changes), reporting the acceptance-length
  distribution, verify-wave cost and tok/s against plain decode, the accuracy
  gate as excess over the floor, and plain prompts with no matches showing no
  slowdown beyond noise.

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
- Left out on purpose (2026-10-02): No GLM-5.3 Full TP16 deployment is checked
  in. The fleet lane renders its tree at run time from
  `tools/glm53full_lane.py` (TP16 at `:8`; pages derived from
  `--max-sequence-positions` at `:104`; logical = physical + backing spill at
  `:133-134`). The renderer is driven by `tools/glm53full_lane.sh`, which
  requires `GLMFULL_POSITIONS`, `GLMFULL_ROWS`, `GLMFULL_SEQUENCES` and
  `GLMFULL_INFLIGHT` from the environment (`:10-13`), so the deployed context
  limit is recorded nowhere in the repository. Two older generators remain:
  `tools/glm52_gen_deployment.py` (TP8 band by default at `:15-18`, `tp8` pack
  and backing names at `:57` and `:81` whatever the TP) and
  `tools/glm53full_gen_deployment.py`. Both write `max_sequence_positions`
  4096 (`:58`, `:98`) but size the pool for 32768 positions with logical =
  physical (`:112-113`, `:152-153`). That is eight times the pages 4096
  positions can address, with no spill headroom.
  `tests/test_deployment_config_drift.py:205-213` covers only the TP8
  generator. Close it by deleting the two older generators (or deriving their
  pages from positions), checking in the TP16 lane deployment and pinning it
  in the drift test. The proof is the drift test passing on the checked-in
  tree and a fleet load of that tree logging the expected `kv binding
  logical_pages= physical_pages=` line.

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
- Left out on purpose (2026-10-02): `K3ServingReset`
  (`modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c:921-926`)
  returns OK without touching the stage runner, and `K3ServingPrefetch`,
  `K3ServingResolvePrefetch`, `K3ServingProgress` and `K3ServingQuiesce`
  (`:872-902`) are also success-returning no-ops (I01). On a client reset
  residentd calls `reset`, takes OK as done, zeroes its own slot table and
  sets `reset_done` (`node/model_residentd.c:2998-3014`), while nothing clears
  the K3 runner's slots, so their KDA and KV state from the previous client
  stays in place (I16). k3 is refused at load today (no PREFIX_REUSE,
  `:26-30`, `:49-51`). Close it by having reset release every runner slot
  through `SparkK3StageRunnerResetSlots` (used today only for RELEASE at
  `:780`) and refuse stale-generation submissions, and by making each other
  hook do its work or return `UNSUPPORTED` naming itself; prove it on the
  fleet with a client reconnect after a completed request, after which a
  request on the same slot matches a fresh-process run token for token.
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
- glm5_next compares four driver descriptor fields in its own load
  function and skips `model_description_sha256`, which
  `serving_adapter_template.c` checks for the other adapters. Moving it
  onto the template needs its build to pass its description hash, as the
  ling module does through `MODEL_DESCRIPTION` and the laguna module through
  `LAGUNA_MODEL_DESCRIPTION_SHA256`; the tree holds a glm5_next description
  for fp8 only.
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
- Left out on purpose (2026-10-02):
  `modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_module.c:241-242`
  and
  `modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_resident_decode_stage_module.c:260-261`
  read `SPARK_<FAMILY>_STAGE_DEBUG_SKIP_GDN` and `_SKIP_MOE` in every build.
  Any value skips the whole token-mixer step, full attention as well as GDN
  (qwen38_max `:1131-1132`, qwen4_flash `:1464-1465`), or the MoE block
  (qwen38_max `:1133-1134`, qwen4_flash `:1476-1477`), with no log line, and
  the step still returns OK.
  `spark_qwen4_flash_resident_decode_stage_module.c:277-283` reads
  `SPARK_QWEN4_FLASH_STAGE_ALLOW_MISSING_PLE` in every build and drops the PLE
  tensors from the required pack geometry (`:327`, `:669-670`). A pack without
  them then loads, and the PLE layer runs without the n-gram injection
  (`:1457`). A stray variable in a serving environment produces wrong tokens
  with a successful status (I03, I22). The fix: put all three under `#ifdef
  DEBUG` or delete them, and make a release pack without PLE tensors fail to
  load. Prove it on a Spark: release qwen4_flash and qwen38_max loads refuse
  the variables and pass their T1 gates.
- Left out on purpose (2026-10-02): About 50 C, C++ and CUDA sources outside
  `tests/` still carry comments, against I48 and AGENTS.md. Examples:
  `runtime/spark_weightd.c` (54 comment lines),
  `include/sparkpipe/spark_kv_page_store.h`, `spark_kv_page_cache.h`,
  `ring/transport/tp_device_collective.c:2020-2032` (the host-register skip
  rationale), `node/model_residentd.c:400` and
  `runtime/model_batch_engine.c:2316`. `inference/kernels/route.cuh:11` is a
  marker comment that `tests/test_cuda_performance_contracts.py:378` requires
  word for word, so the test keeps a comment alive. The fix: move each
  rationale into documentation, delete the comments and the wording assertion,
  and add a source lint that fails on any comment in a C, C++ or CUDA file
  outside tests. It is closed when that lint passes on the whole tree.

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
- Left out on purpose (2026-10-02): When a glm5_next graph replay is stuck or
  its wait times out, the module clears `graph_path_enabled`
  (`modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:4686`,
  `:4734`), fails that frame, and runs every later wave eager until the
  process restarts. The degraded state appears only as `graph_path=2` in the
  `G5N-WAVE-TIMING` log line (`:2848-2853`, `:5928`); `degrade_graph_stuck` is
  never read and readiness is unchanged, so the rank keeps serving on the slow
  path while reporting ready (I22). Report the degraded path in the snapshot
  and readiness, fail benchmark verdicts taken on a degraded rank, and prove
  it by forcing a graph wait timeout on a Spark and seeing the rank leave the
  ready set.
- Left out on purpose (2026-10-02): While any rank is disconnected, the batch
  engine retries an `IO_ERROR` rejection up to 10,000 times, with a backoff
  capped at 200 ms, about 33 minutes (`runtime/model_batch_engine.c:874-892`).
  The engine never sets a submission deadline, so nothing fails a request held
  by a dead rank sooner. `GET /health` answers `{"status":"ok"}` from the
  served count and tokenizer state alone (`node/model_api.c:1431-1437`), so
  readiness never shows the missing rank (I17). The fix: report connected
  ranks and request progress in `/health`, mark the engine not ready while a
  rank is missing, and fail a held request at its deadline with an error
  naming the rank. It is closed by a fleet run that stops one rank's residentd
  mid-run: `/health` turns not-ready, held requests fail at their deadline,
  and new requests complete once the rank returns.

## Serving API

- The API applies the GLM chat template in shared code (`node/model_api.c`:
  `[gMASK]<sop>` and GLM role markers for every model). The chat template
  belongs to the model: carry it in the deployment's tokenizer or model
  description and let the API render whatever the model declares, then drop
  `node/model_api.c` from the `PENDING` list in `tests/test_dry_law.py`.
- Ling TP16 prefill runs one row per sequence per wave (the round-major
  wave rule), so a prompt costs one full 86-collective chain per token:
  298 prompt tokens take 5.1-8.2 s to the first token. Chunked KDA
  prefill (many rows of one sequence per wave) is the fix.
- `SparkLingRoundMajorWaveRows` clamps every wave to one row, so a
  decode step of 8 sequences runs 8 full chains (8-stream aggregate
  equals B1, about 52 tok/s). Removing the clamp gives waves of several
  sequences, and those produce wrong tokens after the first decode step;
  the Q-projection stride and the KDA state index were two of the
  multi-row defects and are fixed, at least one remains. Find it with a
  T1 route comparison of a multi-row wave against the same prompts solo,
  then drop the clamp.
- The ling TP chain advances from host callbacks and keys the single
  device collective per chain, so a ling lane runs one submission in
  flight (`tools/ling_lane.py` renders `max_inflight_submissions` 1).
- Ling keeps no KDA state per cached prefix, so it is refused at load
  until it captures KDA state at block boundaries (Prefix reuse (I23)).
- laguna's TP chain has the shape ling had before 2026-09-28: it keeps the
  adapter's stack-allocated batch view and frame context across
  asynchronous collectives and takes no collective chain key, so a TP>1
  laguna lane should expect a dead-stack read on multi-wave prefill and
  CAPACITY_EXCEEDED after 1024 collectives. Port the ling fixes
  (chain-owned views, `SparkTpDeviceCollectiveChainKey` per execute).
- The ling T1 stream dump prints hidden plus the reduced MLP delta, which
  does not equal the fixture's residual stream at early layers (layer 0
  norm 1.1 against 226); `tools/t1_ling_log_assembly.py` also expects a
  prefix field that raw residentd logs do not carry. Route ids and head
  tokens compare; streams need the dump brought back to the fixture's
  definition.
- `text/tokenizer.c` knows split regexes only by exact string. It knows
  the GLM digit-run pattern, the Qwen letter-and-mark pattern, the
  DeepSeek digit/ideograph sequence (V4, V4.1 Flash, Hy4), and two
  letter-class patterns (MiMo, Qwen3.8-27b nvfp4, Ling, the last with
  possessive quantifiers). Every other `Split` is skipped without an
  error, and the text is BPE-encoded whole. A 2026-09-28 survey of
  `/mnt/model-warm/*/tokenizer.json` found these unhandled:
  - laguna, whose newline split precedes the letter pattern;
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
- Left out on purpose (2026-10-02): `node/model_api.c:1529-1537` lowers the
  engine's `max_prefill_rows_per_submission` below the deployment's validated
  `runtime_limits.max_input_row_count` (`:1528`) whenever
  `SPARK_MODEL_API_MAX_PREFILL_ROWS` is set. This happens in every build,
  writes no log line, and silently ignores a non-numeric value. A production
  API run with the variable set prefills in smaller submissions than the
  deployment declares, so prefill and TTFT receipts measure a shape the
  deployment does not describe (I22, I04). The 2026-09-28 hub unit set it to 8
  (`docs/FLEET_RELEASE_RUNBOOK.md:582`) to avoid a spin-mode host-round
  failure that #1255 fixed (`docs/GLM5_NEXT_ROOFLINE.md:610-617`). The fix:
  delete the env read so the bound comes only from the deployment, and log the
  bound the engine receives. Close it with a GLM Full TP16 API run whose unit
  carries no such variable, whose log shows the deployment's
  `max_input_row_count`, and which passes the T1 and TTFT gates.

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
- Left out on purpose (2026-10-02): `tools/glm5_next_driver_compare.py` cannot
  produce a receipt. Its probe gives the driver a TP16 context with collective
  identifier 0 and no `KV_SHARD` flag (`tools/glm5_next_driver_probe.c:95`,
  `:100`), which the module has refused at configure since `43ddc3ee4`
  (`spark_glm5_next_resident_decode_stage_module.c:482-491`) in resident and
  lazy mode alike; resident mode would fail anyway because weightd attach is
  mandatory (`:731-733`). The compare tool also requires the literal
  `full-vocabulary-logits=unavailable` (`:45`), so it rejects any probe that
  compares full-vocabulary logits and can never record I27 logit parity. Give
  the probe a node context the real module accepts, compare full-vocabulary
  logits and require `exact`, and prove it with a RESULT.json from a Spark run
  on the GLM-5.3 Flash rank-0 pack.
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
- Left out on purpose (2026-10-02):
  `include/sparkpipe/family/module/spark_module_initialize_gate.h:3-10`
  qualifies the gemma4, minimax, muse_glimmer, qwen4_flash, qwen38_max and
  qwen38_27b modules by one environment variable,
  `SPARK_<FAMILY>_ALLOW_UNQUALIFIED_EXECUTION=1`, with no validation receipt
  behind it. Every serving adapter for those modules sets the variable
  unconditionally before it loads the driver:
  `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:641`
  (called at `:2209`) and
  `model-families/common/include/sparkpipe/spark_qwen38_pp_serving_adapter_common.h:99`
  (called at `:737`) for gemma4, minimax, muse_glimmer, qwen4_flash and
  qwen38_max. The gate therefore never returns `MODULE_NOT_VALIDATED` in
  serving, and a module build that never passed validation loads and serves
  (I03, I39). The GPU validators also demand the bypass:
  `modules/qwen38_27b_resident_decode_stage/validation/validate_qwen38_27b_resident_decode_stage_cuda.sh:45`,
  `modules/qwen38_max_resident_decode_stage/validation/validate_qwen38_max_resident_decode_stage_cuda.sh:38`,
  `modules/muse_glimmer_resident_decode_stage/validation/validate_muse_glimmer_resident_decode_stage_cuda.sh:49`
  and
  `modules/qwen4_flash_resident_decode_stage/validation/validate_qwen4_flash_resident_decode_stage_cuda.sh:41`.
  The fix: bind the gate to a pinned validation receipt for the module hash,
  and delete both `setenv` calls together with the Makefile and validator
  plumbing. Prove it on a Spark: initializing a module without a receipt
  returns `MODULE_NOT_VALIDATED`, and the receipted build loads and passes its
  T1 gate.
- Left out on purpose (2026-10-02): The gemma4 GPU validator skips
  `SparkGemma4ValCheckChainSliding`, its only chained-layer check, when
  `SPARK_GEMMA4_VALIDATION_CHAIN=0`
  (`modules/gemma4_resident_decode_stage/validation/spark_gemma4_resident_decode_stage_cuda_validation.cu:1960-1964`),
  and still prints PASS and exits 0 (`:1965-1967`). `make publish`
  (`modules/resident_decode_stage_rules.mk:203-222`) treats exit 0 as
  validation (`runtime/pack/module_library.c:938-954`), so a gemma4 module can
  be published with a validation receipt for a layer chain that never ran.
  Delete the switch so the chain check always runs, and prove it with a gemma4
  publish on a Spark whose validator log shows the chain check.
- Left out on purpose (2026-10-02): The muse_glimmer GPU validator runs its
  module tier (`SparkMuseGlimmerValCheckModule`, the only check that executes
  driver decode) only when `SPARK_MUSE_GLIMMER_VALIDATION_MODULE_TIER` is set
  (`modules/muse_glimmer_resident_decode_stage/validation/spark_muse_glimmer_resident_decode_stage_cuda_validation.cu:529-534`),
  and nothing in the tree sets it. The validator then prints PASS after the
  kernel-level norm, window and gate checks alone (`:540`), so every
  muse_glimmer publish is accepted without one decoded token. Make the module
  tier unconditional and fail when weightd attach is unavailable, and prove it
  with a Spark validator run that attaches through weightd and prints the
  `module_decode` check.
- Left out on purpose (2026-10-02): The qwen4_flash GPU validator turns a
  decode-versus-prefill token mismatch into a log line and continues when
  `SPARK_QWEN4_FLASH_VALIDATION_TOKEN_PARITY=warn`
  (`modules/qwen4_flash_resident_decode_stage/validation/spark_qwen4_flash_resident_decode_stage_cuda_validation.cu:154-160`),
  then prints PASS (`:718-719`). A qwen4_flash module whose decode disagrees
  with its own prefill passes validation and can be published (I03, I40).
  Delete the warn mode so a mismatch always fails, and prove it with a Spark
  validator run on a qwen4_flash rank pack that reports `bit_exact=1`.
- Left out on purpose (2026-10-02): The dsv4 GPU validator compares against
  reference outputs only for stage 0 with the three-layer slice starting at
  layer 0
  (`modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu:1488-1518`).
  Every other stage passes when any hidden output element is nonzero
  (`:1456-1457`), and the final stage passes when its token ids are below the
  vocabulary size (`:1458-1461`), so repeatable wrong output is published as
  validated (I40). Compare every stage's boundary hidden state and the final
  tokens with pinned reference outputs within a qualified tolerance, and prove
  it with a per-stage validator run on a DSV4 pack before DSV4 serves again.
- Left out on purpose (2026-10-02):
  `modules/glm52_resident_decode_stage/validation/glm52_prefill_rows_parity.cu:590-596`
  labels every `regime_split=0` wave and every wave wider than
  `SparkGlm52ExactWaveRows()` (8 rows) DIFFER-EXPECTED and prints `max_abs`
  without bounding it, and `:520-523` does the same for head tokens. The
  exact-row limit comes from the module under test
  (`spark_glm52_resident_decode_stage_cuda.cu:18-21`), so GLM-5.3 Full prefill
  waves above 8 rows, which the module accepts up to its execution row
  capacity (`spark_glm52_resident_decode_stage_module.c:2160-2172`), pass the
  rig with any error. The rig can also be downgraded by argument: `argv[2]` of
  0 drops the exact checks and a negative value returns 0 after the reference
  run (`:548`, `:582-583`). Pin a qualified per-width max_abs and token
  tolerance in the rig, fail any wave above it, remove the argument downgrade,
  and prove it with `run_glm52_prefill_rows_parity.sh` on a Spark covering the
  widest wave GLM-5.3 Full serves.
- Left out on purpose (2026-10-02): The glm5_next GPU validator checks DSA
  attention only for run-to-run determinism
  (`modules/glm5_next_resident_decode_stage/validation/spark_glm5_next_resident_decode_stage_cuda_validation.cu:1668-1670`),
  states at `:1672` that distributed, routed-MLP and multi-row numerical
  checks are still required, and prints PASS at `:1673-1675` anyway. A GLM-5.3
  Flash module whose attention or routed experts compute wrong values
  repeatably is published as validated (I40). Add DSA attention, routed-MLP
  and multi-row tiers compared with a pinned reference within qualified
  tolerances, fail the validator until they exist, and prove it with a
  validator run on a Spark against the GLM-5.3 Flash TP16 rank pack.
- Left out on purpose (2026-10-02): `tools/glm5_next_driver_probe.c:95` and
  `:100` configure the glm5_next driver at TP16 with collective identifier 0
  and no `KV_SHARD` flag, a context the real module refuses
  (`spark_glm5_next_resident_decode_stage_module.c:482-491`), so the probe
  computes nothing on a real driver. `tests/test_glm5_next_driver_probe.c:25`
  asserts that same context against `fake_create`, and
  `tools/cuda13_sm121a_compile_gate.sh:171-174` runs it in CI, so CI stays
  green while the probe is dead; the CLI also accepts only 1, 3 or 5 rows
  (`:547-549`). Configure the probe with a context the real module accepts,
  accept any row count up to the execution capacity, make the test assert the
  real module's acceptance rules, and prove it with a probe receipt from a
  Spark run.
- Left out on purpose (2026-10-02): `tools/qwen38max_tp16_rank_verify.py`
  accepts the placed tp4pp4 legacy form, whose directory repeats the last
  inventory tensor's shape (`:183-184`, `:241-255`), and prints `verdict=PASS`
  (`:359`) after a structure check; the content pass refuses that form
  (`:276-280`) but runs only with `--checkpoint`, which
  `tools/qwen38max_multidev_pack_emit.sh:126-128` never passes. A `--receipt`
  path that does not exist is skipped (`:339`) and absent receipt fields are
  not compared (`:353`), so a defective pack verifies (I05, I28). Refuse the
  stale-shape form and repack, require the receipt file and every receipt
  field when `--receipt` is given, and prove it with the verifier rejecting a
  placed legacy rank pack and passing a repacked one with `--checkpoint`.
- Left out on purpose (2026-10-02): The qwen38_27b GPU validator's module tier
  compares decode only with its own prefill
  (`modules/qwen38_27b_resident_decode_stage/validation/spark_qwen38_27b_resident_decode_stage_cuda_validation.cu:124-145`)
  and a fresh instance with itself (`:155-184`), skips the MTP draft check
  when `SPARK_QWEN38_27B_STAGE_MTP` starts with `0` (`:148-152`), and prints
  PASS (`:213-214`). A module that is consistently wrong passes and can be
  published (I40). Add a pinned external reference (tokens and boundary hidden
  state from the checkpoint) to the module tier, run the MTP check whenever
  the pack carries an MTP layer, and prove it with a Spark validator run on
  the qwen38_27b rank pack.
- Left out on purpose (2026-10-02): The hy4 module's Prepare logs `ready ...
  execute=UNSUPPORTED` and returns OK
  (`modules/hy4_resident_decode_stage/source/spark_hy4_resident_decode_stage_module.c:95-111`),
  Admit accepts every request (`:121-132`), and Execute returns `UNSUPPORTED`
  (`:113-119`). A host that loads hy4 sees a ready driver that accepts work
  and computes nothing (I01). Make Initialize return `UNSUPPORTED` until
  Execute runs the model, register `test_hy4_driver_acceptance` with the real
  driver, and prove it with an hy4 rank-pack run that produces reference
  tokens.
- Left out on purpose (2026-10-02): The qwen38_max GPU validator runs its GDN
  step, gated-norm and chunk checks only at `tp_degree` 1
  (`modules/qwen38_max_resident_decode_stage/validation/spark_qwen38_max_resident_decode_stage_cuda_validation.cu:315`,
  `:389`, `:606`) and prints PASS (`:1168-1169`). `59b082ded` deleted the TP4
  rank-local GDN check because it could not fail, and nothing replaced it, so
  the head-sharded GDN launches qwen38_max serves with have no numerical gate
  (I40). Add a rank-local GDN tier at the serving TP degree compared with a
  sharded host reference, and prove it with a validator run on a Spark.
- Left out on purpose (2026-10-02): At TP16, GLM-5.3 Full packs several
  sequences into one prefill wave (`SparkGlm52PackedPrefill`,
  `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:794-797`,
  with the table set at `:850`). No GPU validator or CUDA test sets
  `prefill_block_table`: `tests/cuda/attn_prefill_cuda.cu:142,151,221` pass a
  null table, and nothing under
  `modules/glm52_resident_decode_stage/validation/` references it. The graph
  parity rig builds only one-row waves (`glm52_chain_graph_parity.cu:40-66`),
  so padded graph-bucket replays have no numerical check either. A packed wave
  in which one sequence attends to another's keys, or a padded row that writes
  wrong KV, therefore passes every gate (I18, I40). The fix: add a validator
  case that compares a packed wave of K sequences, and a padded bucket replay,
  against K single-sequence linear waves, bit for bit on tokens and written
  KV. It is closed when that validator passes on a Spark and its receipt is
  pinned.

## Fleet and queue tooling

Found while verifying [`docs/FLEET_RELEASE_RUNBOOK.md`](docs/FLEET_RELEASE_RUNBOOK.md)
and the multidev docs on 2026-09-28. Line numbers are in
`tools/fleet_node_agent.sh` unless another file is named.

- Owner decision (2026-10-03): nothing SparkPipe-specific runs at boot.
  After a reboot, `tools/fleet_post_reboot.sh HOST...` checks the node and its
  mesh peers. The serving settings now in the hand-installed `20-serving.conf`
  drop-in (and the lane pin `SPARK_WEIGHTD_LANE`) move into the deployment
  contract (lane A06), and `sparkpipe-hub-route.service` stays a hub-side
  runtime step listed in `docs/INCIDENT_RECOVERY_PLAYBOOK.md`.
- Bump `SPARK_WEIGHTD_IPC_ABI_VERSION` whenever the mesh layout changes.
- `tools/devcycle/lane_assignments.json` and `lane_budget_calc.py` still
  assume 8 lanes, and `lane_assignments.json` gives lane 0 to a GLM development
  lane while production GLM runs on lane 6. Replace both with the lane
  allocator of lane B04 (TD287/TD288).
- A stale, idle `sparkpipe_weightsd` still runs on spark6.
- Operations: confirm linger on all 16 nodes once they are up; `tools/fleet_post_reboot.sh` checks it on every rebooted node.

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
- Left out on purpose (2026-10-02): KV and weights use separate residency
  mechanisms. KV pages move through the arena
  (`SparkKvCacheArenaMarkBlockResident`, `cache/kv_cache.c:1361`) and its
  evict hook, weights through weightd leases, and the device layer offers only
  allocate, free and copy (`include/sparkpipe/spark_memory_buffer.h:31-40`),
  while `docs/INFERENCE_OS_DESIGN.md:72-76` requires one
  `make_resident`/`evict` for both. The only coordination is a static
  `--kv-reserve-bytes` carve-out in weightd (`node/weightd.c:175-186`,
  `:359-371`), so KV pressure cannot reclaim idle weight memory and the
  reverse. Close it by building `make_resident`/`evict` in the device layer
  and moving the KV pager and weightd lease eviction onto it, proven on the
  fleet by a run where KV growth and expert demand both resolve through that
  one operation against one logged node budget.

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
