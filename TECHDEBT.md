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
- A prepare waiting on its snapshot restore answers `PENDING` and the engine
  polls it every 2 ms within the in-flight budget, but there is still no
  completion signal, and a prepare that waits on a park or a full slot table
  answers `BUSY` with the 10 to 200 ms backoff. Close it with a
  restore-complete and park-complete notification from residentd to the
  engine, so a lane is dispatched as soon as its pages are in place.
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

- Store the replicated spine losslessly compressed (DFloat11 idea, Zhang et
  al. 2025, `LeanModels/DFloat11`): BF16 sign and mantissa stay raw and the
  8-bit exponent is Huffman-coded, so the spine shrinks to about 70% while
  every decoded weight stays bit-identical. The spine is replicated on all 16
  ranks, so each GB saved per rank frees 16 GB of fleet memory for KV and
  expert residency. Write our own encoder (pack time) and GPU decoder (a
  lookup-table decode into the layer's BF16 working buffer just before its
  GEMM/GEMV, overlapped with the previous layer), reading only the paper;
  no code from the repository. Gate: weights decode bit-exact against the
  BF16 pack, logits identical, and B1/B16 decode tok/s within noise of the
  uncompressed spine. An unfused decode adds a write and read of each layer's
  BF16 weights per step, which can cost more than the bytes saved on
  memory-bound decode. If the gate fails, try decoding inside the GEMV.
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
- KV survives model eviction and reactivation in code (`README.md:157-158`):
  destroying the binding saves every published, unsaved chain to the snapshot
  store first (`SparkStageKvBindingSaveAtDestroy`, bounded by
  `SPARK_STAGE_KV_DESTROY_SAVE_TIMEOUT_NS`), parked prefix pages queue their
  saves when they spill, and the weightd pool keeps the sealed resident pages
  for a reattach. Not yet fleet-proven: publish a prompt, evict GLM-5.3 Full,
  reactivate it, resend the prompt, and see `cached_tokens > 0` with tokens
  identical to the first run.
- The KV page store reserves its whole backing quota at open (`fallocate`),
  so a full partition or a quota larger than the free space fails the load
  with `CAPACITY_EXCEEDED`, and each deployment node names its `kv_partition`:
  residentd refuses a backing or snapshot directory on another filesystem, or
  a partition on tmpfs, ramfs, NFS, Ceph, FUSE, CIFS/SMB or 9p. The Sparks
  keep KV on the root NVMe (`/`), so the check does not separate KV from the
  OS. Close it on one Spark by loading a deployment whose backing directory is
  on `/mnt/model-warm` (Ceph) and seeing the refusal.
- Left out on purpose (2026-10-02): GLM-5.3 Flash (glm5_next) requires
  `kv_shard` (and with it `dsa_index_context_parallel`) at every TP degree the
  shard check accepts (8 and 16), so each rank holds 1/tp of the latent KV and
  indexer keys. At TP2 and TP4 the shard check refuses the split (the 64-head
  and 64-slot page geometry at the index grain), so a TP4xPP4 deployment still
  stores the full latent KV on every rank of a TP group, against
  README:268-274. Close it by extending the context split to degrees 2 and 4,
  proven by a TP4xPP4 fleet run with T1 parity. GLM-5.3 Full: see the glm52
  entry below.
- KV pools resize by chunk inside the node reserve with resident demand
  (`runtime/stage_kv_binding.c` pool policy, weightd `KV_POOL_RESIZE`/
  `KV_POOL_STATUS`), and `tests/test_stage_kv_pool_resize.c` moves chunks
  between two bindings in one process under the CUDA stub. Not yet proven on
  the fleet: close it with two co-resident engines on one Spark (two GLM Flash
  lanes on a shared reserve) whose `kv pool grew` and `kv pool shrank` lines
  follow the load, with identical outputs to unshared runs and no
  `KV-PARK-FAILED`.
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
- Left out on purpose (2026-10-04): weightd owns each engine's KV pool
  (`SparkWeightdKvPoolMap`, `runtime/spark_weightd_kv_pool.c`; the binding
  carves its regions and page table from it), keeps it across an engine
  restart and lets a clean restart adopt the sealed resident prefix pages, but
  pools are private to one engine: there are no cross-engine leases,
  refcounted prefix shares or copy-on-write forks, so two co-resident engines
  serving the same model hold duplicate prefixes. Close it with shared pages
  in weightd that several bindings map read-only, proven by two engines on one
  node serving one prefix from one set of device pages.

## KV tiers

- Left out on purpose (2026-10-04): JIT-KV W3
  (`docs/archive/JIT_KV_RESPONSE.md:47-49`) is partly done: the slot file, the
  lane pager with its dsv4 frame ops and the header-only `LmCache` are
  deleted, and the binding uses two stores, the per-engine spill page store
  (`cache/kv_page_store.c`, digest-checked, quota reserved at open) and the
  persistent snapshot store (`cache/kv_snapshot.c`). A parked prefix page
  whose entry is not yet saved queues its snapshot save when the arena parks
  it (`SparkStageKvBindingPark`, `SparkKvPageCacheSaveParked`), and the save
  reads the spill copy, so spilled prefixes reach the persistent store
  within the write budget. Still left: the spill store itself stays anonymous
  for pages of running sequences (not reusable after a restart);
  `cache/nvme_tier.c` remains for the unwired topology switch;
  and the external provider client (`cache/store/`, `common/common_kv_frame.h`)
  stays inside the qwen38_max, qwen38_27b, qwen4_flash and muse_glimmer
  modules until they move onto the binding. Close it by making the spill
  store write the snapshot format and moving those families onto the
  binding, proven on the fleet by a spill, residentd restart and restore run
  that hits through the snapshot store alone.
- Oversubscription is fleet-unproven. The engine admits lanes up to the
  physical pages, and the arena parks the pages of lanes outside the running
  wave one at a time through the page store and brings them back when the
  lane rejoins, so a lane is parked and restored page by page rather than as
  one contiguous transfer. Close it with a fleet backpressure run at 2x device
  pages where parked lanes restore bit-exact at B1 and B16.
- Crash recovery of the KV snapshot store has not run on the fleet. The store
  writes each file to a `.kvs-writing-` temporary, fsyncs, renames and fsyncs
  the directory, and its open deletes leftover temporaries; the binding opens
  it in production, but no kill-during-write case has run. Close it on the
  fleet: kill -9 residentd on one rank while saves are queued, restart it, and
  check that no `.kvs-writing-` file remains, `removed_temporary_count`
  equals the leftovers, and the resent prompt restores with tokens identical
  to an uninterrupted run.
- Eviction now prefers entries the snapshot store already holds (or cannot
  hold), and queues a save for the older unsaved entries it passes over
  (`SparkKvPageCacheSelectVictim`, `cache/kv_page_cache.c`), so a saved prefix
  evicted from the device and logical pools restores from the snapshot store.
  When no safe entry is in the scan window the oldest unsaved entry is still
  discarded (counted as `evicted_unsaved` in the store close line) rather than
  stalling admission on the save thread. Close it on the fleet with a run at
  2x device pages where `evicted_unsaved` stays zero and an evicted prompt
  restores with `cached_tokens` covering it and tokens equal to an
  uninterrupted run.
- Spilled prefixes on the common binding (GLM Full, GLM Flash, K3) reach the
  snapshot store: a parked page of an unsaved prefix entry queues its save
  from the spill copy, and released sequences already queue theirs. Not yet
  fleet-proven: close it with a prompt spilled before a kill -9 of residentd
  that hits after the restart with tokens equal to an uninterrupted run. The
  laguna
  (`modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_module.c:831`),
  ling
  (`modules/ling_resident_decode_stage/source/spark_ling_resident_decode_stage_module.c:645`)
  and dsv4
  (`modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:1327`)
  modules keep their own anonymous page stores, so their spilled KV still dies
  with the process until they move onto the binding (B07–B11).
- Prefix-cache eviction ranks victims by the highest priority of the requests
  that published or reused an entry, then prefers entries the snapshot store
  holds, then age (`SparkKvPageCacheSelectVictim`,
  `SparkKvPageCacheResidentVictim`), but deadlines are not considered and the
  arena's choice of which resident page to park is still reuse value and
  recency. Close it with deadline-aware ranking, proven on the fleet by an
  oversubscribed run with two priority classes where the higher class keeps
  its hits and its TTFT stays flat.
- A full backing store fails the admission: after the page cache relieves
  backing (releasing an idle restored page's record or evicting an unused
  entry) and still finds no slot, the admission answers `CAPACITY_EXCEEDED`
  and logs `KV-BACKING-FULL` (`cache/kv_page_cache.c`,
  `SparkKvPageCacheBackingOutcome`). Close it by feeding store occupancy into
  engine admission so a full store queues new work instead of failing it.
- The KV NVMe write budget covers only the engines on the common binding. weightd
  hands each KV pool a share of `--kv-write-budget-bytes-per-day`, and the
  binding stops snapshot saves and discards parked pages for recompute once
  its share is spent (`cache/kv_page_cache.c`, `SparkKvPageCacheDiscardForBudget`),
  reported in `kv_store_report`. The non-core families' own page stores
  (laguna, ling, dsv4) write unbudgeted until they move onto the binding, and
  the budget has not run on a Spark. Close it with those families on the
  binding and a one-Spark run with a small budget that reports the write
  rate, stops spilling at the limit and keeps serving.
- Copy-on-write of a partial prefix page needs a device copier attached to
  the page cache (`SparkKvPageCacheAttachDeviceCopy`). glm52 (through the KV
  binding) and glm5_next attach one; dsv4, laguna and ling do not, so a
  mid-page prefix admission on them answers `UNSUPPORTED`
  (`KV-COPY-ON-WRITE-UNAVAILABLE`). Close it by moving those drivers onto the
  KV binding.
- The glm5_next completion still finishes cache lanes inside its
  `cudaLaunchHostFunc` host function under `kv_mutex`
  (`SparkGlm5NextFinishCacheLanes`), the same lock admission takes, so an
  admission holding the lock stalls the finishing wave's stream. glm52 hands
  its completion to the KV binding's completion thread
  (`SparkStageKvBindingFinishAsync`). Close it by moving glm5_next onto the KV
  binding.
- A page whose park keeps failing stays resident with `PARK_FAILED`
  (`cache/kv_cache.c`, `SparkKvCacheArenaEvictResidentBlock`); once every
  resident page has failed, the last-resort retry also fails and the
  admission answers `IO_ERROR` with `KV-PARK-STALLED`. A persistent disk
  error therefore stops admissions that need a park. Close it by failing the
  backing store over to a second path or degrading to recompute-only prefix
  reuse with a logged transition.
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
  `include/sparkpipe/spark_stagepack_format.h:78-108`). Two model revisions
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
  path, and the same header forces the JIT KV tier off (`:111-115`).

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
  (`node/model_residentd.c`), not position continuity. Continuity is checked
  inside adapters by the common `SparkStageKvBindingContinuity`, which GLM
  Full, GLM Flash and K3 call; families not yet on the KV binding accept a
  submission whose position jumps ahead in a resident sequence (I02). The fix:
  move those families onto the binding, or run the continuity check in
  residentd for every adapter. It is closed by a fleet run in which a skipped-position
  submission to a GLM Full lane gets an explicit continuity error.
- `build/libdsv4_pro_tp4_pp4_serving_adapter*` do not compile
  (`SPARK_DSV4_MODEL_DSPARK_SPEC_STEP` undeclared), and
  `build/libdsv4_tp4_pp4_serving_adapter.so` cannot be opened on GPU hosts
  (undefined `SparkTpLaunchMeshHardware`).
- Left out on purpose (2026-10-02): GLM Full declares prefix reuse
  (`modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c`,
  `PREFIX_REUSE`) and restores prefixes through the common binding, but no
  fleet I27 session has passed at TP16; the only restore evidence is the
  single-GPU, collectives-off `tools/glm52_prefix_probe.c`. Run
  `tools/i27_session.py` against the GLM Full TP16 API with
  `--restart-command` and `--writeback-fault-command` hooks and pin its PASS
  receipt: B1 and B16 cold/warm parity, a copy-on-write mid-block prefix,
  abort mid-prefill, eviction and recompute, residentd restart and a failed
  write-back, each against an uninterrupted control with identical tokens and
  `cached_tokens` above zero on every expected hit, plus exact T1.
- Left out on purpose (2026-10-02): the engine keeps its prefix index across
  rank-session changes and reloads it from `prefix_index_path` at start, but
  no fleet run has shown a restored prefix after a residentd restart. Fleet
  proof: publish a 4K prompt, restart residentd on all ranks, resend it, and
  see `cached_tokens` equal to the published prefix with tokens identical to
  the first run.
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
- Left out on purpose (2026-10-02): laguna was not moved onto
  `runtime/stage_kv_binding.c`, so it keeps a private copy of the KV plumbing:
  model table, arena and page store setup, page-table builds, lane
  transactions, and the device page copy, prefix-restore predicate and
  page-table upload in its own module source
  (`modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_module.c`).
  Fixes made in the binding do not reach this copy. laguna and ling still fall
  back silently to `/tmp/sparkpipe_<model>_kv_<revision>` when the deployment
  leaves `kv_backing_directory` null, which
  `runtime/model_resident_deployment.c` allows. Close it by moving laguna onto
  the binding and deleting the private copy; the proof is laguna refusing a
  deployment with no `kv_backing_directory` and passing its TP fleet I27 run
  (cold vs warm token parity at B1 and B16).
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
- Making a page resident when the resident pool is full still scans every
  resident slot for a victim instead of taking the least recently used
  unreferenced entry; pool fullness, retained pages, evictions and spill
  traffic are now reported in the API's `kv_store_report`.
- Publish one logical resident model driver with prewarmed B1-B1024
  specializations rather than batch-specific resident identities.
- Batch weight amortization (perf-program rock R4): take the WS/native path
  from two rows up, with k-tile pipelining, so a batch reads each weight
  once.
- Select the smallest validated specialization for effective rows, including
  speculative verification rows, while preserving sequence and KV identity.
- Qualify mixed arrivals, priorities, prompt lengths, shared prefixes, cache
  pressure, cancellation, and starvation bounds.
- Priority and deadline reach admission, prefill and decode scheduling (the
  batch engine selects by priority, then earliest `deadline_ms`, and a prefill
  or decode submission carries its latest lane deadline to residentd as
  remaining time) and KV eviction ranks by request priority. Still open:
  speculation, gang scheduling, model promotion and storage I/O ignore both.
- `LmSkinnyGroupedExperts` (grouped FP8 gate/up experts) does not match the
  per-pair expert kernel bit for bit at 8 rows on a GB10:
  `tests/test_skinny_dependent_cuda.py` fails with "rows8: grouped experts
  differ from the per-pair experts" on main and on branches that do not touch
  the kernel. Find the race or order difference in the grouped kernel and
  make the device test pass.
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
- Restore hints are fleet-unproven. When the engine first finds a cached
  prefix for a queued request it sends every rank a `CACHE_HINT` (resident
  IPC 22); residentd hands it to the adapter's `cache_hint`, and the core
  drivers pass it to the binding as a `CACHE_HINT` admission, which queues the
  restore so it runs while the request waits for dispatch. Non-core adapters
  have no hook and count the hint as unsupported. Close it on the fleet: a
  queued request whose prefix was evicted to the snapshot store shows a TTFT
  that drops by the restore time against a run with hints dropped.
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
- Restore bandwidth is bounded only by construction: one binding worker reads
  restores one job at a time, so restore reads never exceed one sequential
  stream per engine, but nothing measures the drive or admits against it.
  Close it on the fleet with a burst of spilled-prefix requests whose restore
  bytes per second stay at or under the drive's measured rate while the decode
  step time of running lanes is unchanged.
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
- `tools/gen_geometry_header.py` has no `--check` gate: it is absent from
  `Makefile` and `tools/gates.sh`. Its glm5_next and qwen4_flash outputs differ
  from the tracked headers in code (the `REPLAY_ROWS_MAX` and `MISS_RING_*`
  lines, and the `llm_defines.h` include), and only the two qwen38_27b outputs
  are byte-identical, with nothing keeping them so. Close it in A02: make all
  four outputs byte-identical and add a `--check` test for each; if the
  generator moves or is renamed, keep `GENERATOR`, `FAMILIES` and
  `ADAPTER_CONSTANTS` in `tests/test_no_source_comments.py` matching.

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
- Left out on purpose (2026-10-02): GLM Full (glm52) and K3 have no
  context-lookup drafter; GLM Flash has one in its verify regime
  (`SPARK_GLM5_NEXT_VERIFY_ROWS`, drafter `lookup`). On the other two, a
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

- Third-party material that NOTICE does not yet cover (2026-10-03; lane L01,
  scheduled after A01). The repository is public and has no license of its
  own, so every copy is redistributed. NOTICE covers exllamav3 and llama.cpp
  only. Not covered:
  - vLLM copies: `docs/research-dspark/vllm-*.py` (12 of 14 carry
    `SPDX-License-Identifier: Apache-2.0`) and `pr46995.diff`. Apache-2.0
    requires a copy of the license and the retained notices.
  - Text with no license recorded: `docs/research-dspark/model.py`,
    `generate.py`, `joe-*`, `tony-*`, `ds4-issue468-comments.md`,
    `DSpark_paper.pdf`, `paper-html.txt`, `zhong-review.*`, and the copied
    posts, READMEs and release pages in `docs/research-dflash2/`.
  - HF modeling references in `model_contracts/references/` keep their
    Apache-2.0 headers but have no NOTICE entry; `README.md` there gives no
    commit for `modeling_qwen4_exp.py` and no entry for
    `modeling_muse_glimmer.py`.
  - `text/unicode_nfc_tables.h` and `text/unicode_class_tables.h` are
    generated from Unicode 16.0.0 data (`tools/gen_unicode_nfc_tables.py`);
    the Unicode License v3 requires its notice.
  - `tools/hy4_dequant/hy4_tokenize.py` ports the hyv4 pre-tokenizer of the
    AngelSlim patch, whose license is unknown.
  - Chat templates and tokenizer files copied from model repositories, and
    the serving obligations of the model licenses.
  Close it with NOTICE entries (upstream, full commit, license text) for
  everything kept, deletion or links for everything without a redistribution
  right, and the `tests/test_third_party_notices.py` markers extended to
  these paths. The repository-wide audit ledger feeds the lane.
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
- The production GLM-5.3 Full TP16 lane tree is checked in at
  `deployment/glm53full_tp16_lane6/`, rendered by `tools/glm53full_lane.py`
  from the arguments in its `render.json` (lane 6, fp8, 262,144 positions,
  1,024 rows, 2 sequences, 1 in flight, a 32 GiB physical and 20 GiB backing
  KV budget), and `tests/test_glm53full_lane.py` fails when the tree drifts
  from that render. Not yet fleet-proven: load that tree and see the
  `kv binding logical_pages=64567 physical_pages=8192` line on every rank.

## Driver consolidation

- Adopt the common parameterized modules in
  `docs/COMMON_MODULE_ARCHITECTURE.md` and delete the near-copy code they
  replace (estimated by the 2026-09-13 SEAM surveys at about 26,000 lines
  across the families), each migration proved by byte or behaviour identity.
- glm5_next assigns its combine wrappers field by field instead of calling
  `SPARK_FAMILY(ModuleRegisterCombines)`, and its `internal.h` re-declares
  the `SparkTpLaunch*` prototypes from `spark_tp_mesh_register.h`.
- K3 now resets every runner slot and the KV binding on a client reset and
  refuses stale-generation submissions, but no fleet run has proved it: run a
  client reconnect after a completed request, after which a request on the
  same slot matches a fresh-process run token for token.
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
## Runtime completion

- A client that hangs up mid-reply is cancelled and drained in the host tests
  (`test_model_api_text` client disconnect: four abandoned streams reach zero
  live requests within 10 s and the engine keeps serving); a release that a
  missing rank cannot take is bounded by the engine's in-flight budget, and
  that rank's slots are cleared by the session reset on its reconnect. Not yet
  fleet-proven: drop clients mid-stream under load on a GLM Full lane and see
  `live_requests` return to zero with every resident slot free.
- A failed residentd route whose driver cache abort also fails resets the
  rank's session: residentd drops the client and resets every lane before the
  next hello, and only a failed reset stops it so its unit restarts. An abort
  that finds the lanes already empty succeeds, and an abort answered BUSY is
  retried for 5 s. Still open: a per-slot driver reset, so one slot recovers
  without dropping the other live sequences on that rank.
- glm5_next and K3 map a weightd lease failure (`NO_LANE`, `EVICT_DENIED`)
  to `CAPACITY_EXCEEDED`, which fails the request: their KDA recurrent state
  advances layer by layer and is restored only after a prefix attach, so a
  retried frame would run on half-updated state. Restore the lanes' recurrent
  state before a retried frame, then map the failure to `BUSY` as GLM Full
  does.
- Pipeline-parallel stages no longer wedge after a failure on another rank in
  the host tests: the hidden transport has a `cancel` operation (ABI 6;
  `host_staged_tcp` drops the posted receive and any queued frame), and
  residentd cancels a route's transfer when the route is abandoned, its client
  generation changed or its deadline expired. A route cancelled while waiting
  for input fails through the failed-route path, which aborts its committed
  cache transaction, so the session reset can run. The in-tree test transport
  has a peer mode (`SPARK_TEST_TRANSPORT_PEER_DIRECTORY`) in which a receive
  completes only after the peer sent, and `test_model_pipeline_client` proves
  the reconnect with it. Not yet fleet-proven: fail one stage of a K3
  TP4xPP4 lane mid-run and see the next stage log `ROUTE-TRANSPORT-CANCEL`
  and the lane serve again after the engine reconnects.
- Produce one immutable qualification bundle for a release candidate with
  `tools/qualification_bundle.py`: merged commit, release generation, package
  and driver hashes, all-rank identities, token stream, accuracy, performance,
  route counters and drained queue state. No release has one yet.
- A glm5_next rank whose graph replay got stuck or timed out keeps serving on
  the eager path; its snapshot now carries `degraded_flags`, `/health` answers
  503 with `degraded_rank` and `"degraded_path":"eager"`, and `compsec17.py`
  gives no verdict when the API is not healthy at the end of a run. Not yet
  fleet-proven: force a graph wait timeout on a Spark and see the rank named
  in `/health` and the qualification bundle refused.
- Rank loss is reported but not fleet-proven: `/health` answers 503
  `degraded` with `connected_ranks` and `missing_rank` while a rank is gone or
  a failure is latched, a held request retries only within the engine's
  in-flight budget and fails naming the missing rank, and a latched failure
  stays until the rank session changes. Close it with a fleet run that stops
  one rank's residentd mid-run: `/health` turns degraded, held requests fail
  at their deadline, and new requests complete once the rank returns.

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
- Only GLM Full, GLM Flash and K3 sample (temperature, top-k, top-p) and
  return logprobs; the other adapters answer `400 sampling_unsupported`.
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
- GLM Flash positions are sized for every resident sequence at full length
  (`tools/spark_serving_profile.py`: B8 is 8 × 512 positions in 1,024
  pages), so a request cannot use the pages its neighbours leave idle. GLM
  Full already sizes its physical pool by a byte budget its lanes share
  (`tools/glm53full_lane.py --kv-physical-bytes`). Size GLM Flash the same
  way and let paged admission share the pool.
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
- **Every driver update splits audit cohorts.** Replays compare only against
  the same driver hash, so the router must track each provider's build, and
  releases need per-model cohort changeovers.
- **sparkpipe.ai is static.** `site/` has the landing page, the provider page
  and the playground. The catalog, the buyer console and the provider
  dashboard need the router and the ledger behind them.
- **Payments.** Provider identity checks, tax reporting and payout rails
  (fiat, crypto or both) are undecided.

## Production qualification

- `tools/sparkpipe_weightd_vmm_verify.sh` prints `SKIP` and exits 0 when CUDA
  or a GPU is missing, so a receipt run on the wrong host reads as a pass.
  Make both cases exit nonzero with the named reason.
- Repeat accepted transport and model measurements from clean merged `main`,
  rebuild the exact release on Spark hardware, and retain all receipts.
- Close exact-checkpoint numerical parity and end-to-end service gates for each
  model before reporting it production-ready.
- Left out on purpose (2026-10-02): `tools/glm5_next_driver_compare.py` cannot
  produce a receipt. Its probe gives the driver a TP16 context with collective
  identifier 0 and no `KV_SHARD` flag (`tools/glm5_next_driver_probe.c:95`,
  `:99`), which the module has refused at configure since `43ddc3ee4`
  (`spark_glm5_next_resident_decode_stage_module.c:482-491`) in resident and
  lazy mode alike; resident mode would fail anyway because weightd attach is
  mandatory (`:731-733`). The compare tool also requires the literal
  `full-vocabulary-logits=unavailable` (`:45`), so it rejects any probe that
  compares full-vocabulary logits and can never record I27 logit parity. Give
  the probe a node context the real module accepts, compare full-vocabulary
  logits and require `exact`, and prove it with a RESULT.json from a Spark run
  on the GLM-5.3 Flash rank-0 pack.
- `test_hy4_driver_acceptance` has a build rule but stays out of
  `TEST_NAMES`: it holds the behaviour a complete hy4 driver must show and
  fails on the current stub module. Register it with the hy4 driver.
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
- Left out on purpose (2026-10-02): `glm52_prefill_rows_parity.cu` now fails
  any wave above the exact-row limit, or without the regime split, whose
  boundary `max_abs` exceeds 1/64 of the reference's largest magnitude, and any
  head run whose differing tokens exceed 1/64 of the positions; it covers
  widths up to the build's `SPARK_BATCH_BUCKET` and has no argument
  downgrade. The bounds are not qualified: run
  `run_glm52_prefill_rows_parity.sh` on a Spark at the serving bucket, record
  the measured `max_abs` and token differences, and pin the qualified bounds
  with that receipt.
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
  `:99` configure the glm5_next driver at TP16 with collective identifier 0
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
- Left out on purpose (2026-10-02): The qwen38_27b GPU validator's module tier
  compares decode only with its own prefill and a fresh instance with itself
  (`modules/qwen38_27b_resident_decode_stage/validation/spark_qwen38_27b_resident_decode_stage_cuda_validation.cu`,
  `SparkQwen38_27bValCheckModule`), then prints PASS. A module that is
  consistently wrong passes and can be published (I40). Add a pinned external
  reference (tokens and boundary hidden state from the checkpoint) to the
  module tier, and prove it with a Spark validator run on the qwen38_27b rank
  pack.
- The hy4 driver refuses initialization (`UNSUPPORTED`) because Execute
  does not run the model. Implement Execute, register
  `test_hy4_driver_acceptance` with the real driver, and prove it with an hy4
  rank-pack run that produces reference tokens.
- Left out on purpose (2026-10-02): The qwen38_max GPU validator runs its GDN
  step, gated-norm and chunk checks only at `tp_degree` 1
  (`modules/qwen38_max_resident_decode_stage/validation/spark_qwen38_max_resident_decode_stage_cuda_validation.cu:315`,
  `:389`, `:606`) and prints PASS (`:1164-1165`). `59b082ded` deleted the TP4
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
