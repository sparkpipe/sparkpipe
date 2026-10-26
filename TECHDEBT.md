# SparkPipe Technical Debt

This file contains only unfinished work against the system described in
[`README.md`](README.md) and the contracts in [`SPEC.md`](SPEC.md) and
[`sparkpipe_invariants.md`](sparkpipe_invariants.md). Completed work is
removed rather than retained as a progress diary.

Required behaviour that is not implemented is recorded here as "Left out on
purpose", with the date, what is missing, its consequence and the fleet proof
that closes it; no other way of not implementing a required feature is
accepted. Line citations in the 2026-10-02 entries refer to `b41ed891f`.

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

- Every hardware-wait round still crosses weightd's CPU relay once: the GPU
  publishes its payload and doorbell cell, and weightd's doorbell sweep
  (`SparkWeightdMeshDoorbellSweep`, `node/weightd_mesh.c`) posts the RDMA
  writes through its own mlx5dv send queues. Since `af0cc3f8d` the GPU waits
  on the peer tails itself for direct and RSAG rounds
  (`SparkTpMeshHardwareArrivalKernel`). GPU-initiated RDMA is not pursued
  (owner, 2026-10-15): a GB10 SM cannot ring the CX-7 doorbell, and
  GPU-built work requests with a CPU doorbell measured within about 1 us of
  the relay per round, under 1% of a token. The remaining per-round costs
  are the engine's kernels around the wire (publish, arrival, combine).
  Measured on 16 ranks on 2026-10-08: a 132-element all-gather takes 16.1 us,
  and a 7,168-element all-reduce takes about 38 us direct or 33.1 us as
  reduce-scatter plus all-gather.
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
- The mesh reads only the backend kind, degree, rank, width, rows, the
  operation timeout, the band, the lane, the wait mode and the combine
  callbacks from `SparkTpDeviceCollectiveConfig`. Several pieces predate the
  mesh: the rest of that struct (hosts, ports, identifiers, credits and
  bindings, rails, algorithm masks and thresholds); the stub calls around it
  (`ApplyTopology` copies the degree and wait mode, `ProbeMemoryMode`,
  `CreditBindingRouteCount`, `DumpOperations`); the `device_collective`
  stage-config members that feed them (`StageServingLoadDeviceCollective`,
  `inference/runner/stage_serving_adapter.c`); and the `tp_collective` parser,
  which no production code calls
  (`SparkServingAdapterTemplateLoadTpCollective`,
  `runtime/serving_adapter_template.c`). Remove them together with a fleet
  stage-config migration. The collective never reads the relay and TP4 combine
  callback fields, though the stage runner still sets the TP4 one
  (`StageRunnerCombineTp4Bf16`, `inference/runner/stage_runner.cu`).
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
- Left out on purpose (2026-10-02): decode communication is not overlapped
  with compute. Single-sequence prefill waves of 512 rows or more run as two
  half-wave lanes on their own streams, whose collectives overlap the other
  half's compute: `StageRunnerLanesSplit` (`inference/runner/stage_runner.cu`)
  for GLM Full, GLM Flash and the Qwen 3.8 hybrid, and `K3LaunchSlicePair`
  (`inference/llms/kimi_k3/slice.cuh`) for K3. K3 also splits some rounds into
  Begin and Finish. Every other decode step and multi-sequence wave is still
  one serial chain (compute, all-reduce, compute), so every collective wait is
  exposed. At B1 on 2026-10-02 a GLM-5.3 Full token took 36.7 ms. Of that,
  12.8 ms (35%) were peer waits over 236 exchanges (p50 52 us, p99 262 us) and
  25.5 ms were kernels, against a 19.4 ms memory floor. At B16 the waits were
  21% of the step, and at large batch the 3 ms bandwidth-bound all-reduces sit
  on the critical path. Every all-reduce is also a barrier, so one rank's
  hiccup stalls all sixteen. Close it by splitting each decode wave into two
  micro-batches whose chains interleave on separate streams. Prove it with B1,
  B16 and B256 fleet A/B runs that report the peer-wait time on the critical
  path per step, keep tokens identical, and measure the split's extra weight
  reads against the hidden wait.
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
  `SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL`
  (`include/sparkpipe/spark_tp_device_collective.h`), the `nccl` parsers in
  `inference/runner/stage_serving_adapter.c` and
  `runtime/serving_adapter_template.c`, the `nccl` branch of
  `tools/dsv41_flash_gen_deployment.py`, `tools/qwen38_tp4_nccl_bench.c`, and
  the tests that feed them `nccl` configs (`tests/test_serving_tp_config.c`,
  `tests/test_tp_allreduce_fuzz.c`).
- Awaiting fleet proof (lane `mesh/contracts`): attach now fails with
  `MESH-CAPABILITY-MISSING` when the local weightd lacks a route the regimes
  use, and the routes are fixed at attach, so no later read of the wait cell
  can change the algorithm. Proof: with one rank on a weightd without the
  bits, attach fails on every rank; the matched fleet passes the hardware
  collective probe and GLM Full T1.
- weightd refuses a `MESH_WRITE` or `MESH_BROADCAST` whose source, destination
  or sequence word lies outside the registered mesh region, with a named log
  line (`SparkWeightdMeshRangeAllowed`, `runtime/spark_weightd.c`). For a
  connection that declared its lane (mesh activity or a lane acquisition), it
  also refuses ranges outside that lane's two bands; a connection that has
  declared no lane gets only the region check. Not yet fleet-proven: a weightd
  fault-injection run on two Sparks whose out-of-range requests are refused
  while in-range rounds stay bit-exact.
- weightd's QP repair resynchronizes PSNs with the peer: a repaired queue pair
  bumps the node's `wire_generation` in its mesh record, the peer re-transitions
  its own queue pairs to PSN 0 and echoes the generation, and the mesh stays
  unready until the echo arrives (`WD-QP-REPAIR`, `WD-QP-RESYNC`,
  `WD-QP-RESYNC-DONE`; record magic MESH0007). Outstanding transfers to the
  peer are failed with `PEER_RESET` before the rewire, and a lane reset clears
  them. Not yet fleet-proven: kill one rank's weightd mid-collective and see
  every peer complete or fail its rounds and rewire without a restart.
- Recorded incident (2026-10-01 and 2026-10-02): on spark8, a PCIe completion
  timeout on a CX-7 function was followed by a stalled SMMU0 command queue
  (`CMD_SYNC timeout`). The NVMe then wrote other processes' pages over the
  root superblock and group descriptors on 10-01, and over four
  group-descriptor blocks on 10-02. Both incidents happened during
  two-collective (two-band) tests, and each was recovered via PXE rescue and
  e2fsck from the backup superblock. Mitigations in force: the DMA guard
  resets a node on the first event on all 16 nodes
  (`tools/devcycle/spark_dma_guard.py`, hub listener
  `spark-dma-guard-listen.service`), each engine runs one collective in
  flight, and the Lenovo nodes run IOMMU identity mode. The owner rejected
  `iommu.strict` (2026-10-02). Still open: the root cause, a report to NVIDIA
  and Canonical, and the untested case of several fast engines prefilling on
  one Lenovo node
  ([`docs/CX7_DMA_INVESTIGATION_2026-10.md`](docs/CX7_DMA_INVESTIGATION_2026-10.md)).
- Awaiting fleet proof (lane `mesh/contracts`): every collective logs
  `MESH-PATH` (operation, path, phases, elements, rows, capture) on each
  regime change and again for each capture (`SparkTpDeviceCollectiveLogPath`,
  `ring/transport/tp_device_collective.c`). `tools/k3_lane.sh`,
  `stage_lane.sh` and `fleet_serve.sh` keep one rank log per run
  through `tools/lane_run_log.sh`. Each run writes `logs/residentd-<run
  id>.log` and refuses to overwrite it; the helper moves an older launcher's
  `residentd.log` aside, and `archive` collects and removes the logs.
  `fleet_node_agent.sh` moves a non-empty `residentd.log` aside with a
  timestamp before each start. Proof: a fleet run whose rank logs name the
  path of every chain and survive a second run on the same lane.

## Steady-state decode hot path

- Accept each hot-path change independently: exact token parity first, then
  at least three unprofiled end-to-end cached-prefill B1 runs. Do not stack
  candidates until the preceding candidate beats the current GLM 5.3 Flash
  TP16 B1 receipt in [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md)
  (36 tok/s on 2026-09-28).
- The stage runner waits for each wave on residentd's thread.
  `StageRunnerGraphLaunch` synchronizes the stream after every graph replay
  (`SparkTpDeviceCollectiveGraphSettle`,
  `ring/transport/tp_device_collective.c`). `SparkStageRunnerSubmit`
  (`inference/runner/stage_runner.cu`) then synchronizes again to verify the
  deferred rounds and copy the step's tokens before it returns to residentd's
  progress loop (`SparkModelResidentdSubmitAdapter`,
  `node/model_residentd.c`). Move the round checks and token copies into a
  completion path so residentd keeps serving its socket during the replay.
- Prefill waves never run as a graph: the stage runner captures only waves
  with one row per sequence and at most 16 rows (`StageRunnerGraphEligible`,
  `inference/runner/stage_runner.cu`). A decode wave queued behind a prefill
  wave waits for that wave's whole GPU time, because residentd submits one
  wave at a time and the runner returns only when the wave has finished.
  Capture prefill graphs per wave shape, or order decode waves ahead of queued
  prefill waves.
- When the routed experts are not all resident in one fixed pool, the stage
  runner synchronizes the stream on every routed layer to copy the routed
  group offsets to the host before it leases that layer's experts
  (`StageRunnerExpertWeightsService`, `inference/runner/stage_runner.cu`).
  Such a step runs neither as a graph nor as two prefill lanes. In spin wait
  mode every collective round also synchronizes the stream
  (`SparkTpDeviceCollectiveReadControl`,
  `ring/transport/tp_device_collective.c`). Working-set graphs (Model
  residency and storage) remove the unpinned case without pinning every
  expert.
- Between two waves every rank ends and restarts mesh activity, with one
  weightd mesh-activity round trip each way
  (`SparkTpDeviceCollectiveEndChain`, `SparkTpDeviceCollectiveChainKey`,
  `ring/transport/tp_device_collective.c`). Rank 0 also broadcasts a new chain
  epoch that the other ranks spin on (`StageRunnerChainBegin`,
  `inference/runner/stage_runner.cu`). Resident decode chains pay this once
  per chain of K steps; prefill waves and single-step waves still pay it every
  time. Keep the activity and the epoch across the waves of a session.
- A resident decode chain feeds each step's tokens to the next step on the
  device (`StageRunnerChainAdvanceKernel`,
  `inference/runner/stage_runner.cu`). But after every step the runner still
  synchronizes the stream to verify the deferred collective rounds and to copy
  that step's tokens and scores to the host (`StageRunnerStepOutputs`), and
  each step is a separate graph launch. Move those checks and copies to the
  end of the chain so K steps are one enqueue or one graph.
- The batch engine ends a decode chain at the nearest block end of any lane in
  the batch (`SparkModelBatchBuildDecodeRows`,
  `runtime/model_batch_engine.c`), so chains shorten as the batch grows. A
  chain never carries a publish or prefix lane (`StageServingChainSteps`,
  `inference/runner/stage_serving_adapter.c`), so each lane's block end is a
  separate publish frame. Publishing block checkpoints from inside a chain
  would keep chains at full length and remove those frames.
- Validation and admission fanout run on the per-frame path (perf-program
  rock R5, from the archived `PERF_PROGRAM2.md`): 12 + 9 `Validate` call
  sites remain on the client paths, and the engine SHA-probes inside
  `Progress`. Move them off the per-frame path.
- A prepare waiting on its snapshot restore answers `PENDING` and the engine
  polls it every 2 ms within the in-flight budget, but there is still no
  completion signal, and a prepare that waits on a park or a full slot table
  answers `BUSY` with the 10 to 200 ms backoff. Close it with a
  restore-complete and park-complete notification from residentd to the
  engine, so a lane is dispatched as soon as its pages are in place.
- The shared runner splits a single-sequence prefill wave into two lanes whose
  collectives overlap the other lane's compute (`StageRunnerLanesStep`,
  `inference/runner/stage_runner.cu`), and K3 pairs its half-waves
  (`K3LaunchSlicePair`, `inference/llms/kimi_k3/slice.cuh`). The
  mixture-of-experts families refuse a lane below 1,024 rows (`GLM_LANE_ROWS`,
  `G5N_LANE_ROWS`, `QH_MOE_LANE_ROWS`) because each lane reads every expert
  weight again, which cancelled the overlap on GLM Full (8K prompt: 9,140 ms
  whole, 9,088 ms in two 512-row lanes). GLM Full and GLM Flash deploy with a
  1,024-row execution capacity, so their prefill waves never split and still
  wait about 440 ms of a ~1.0 s wave in collective rounds
  (`SparkTpMeshHardwareGuardKernel`, rank-15 profile, 2026-10-02) serially
  with about 400 ms of compute. Overlap the collectives of a 1,024-row MoE
  wave without reading the expert weights twice.

## Prefill

- Prefill reaches at most 5% of the fleet's compute: Kimi K3 prefills an 8K
  prompt at 590 tok/s and a 137,700-token prompt in 299 s
  ([`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md#prefill)). The target is
  80% of the first resource to saturate. The overlap of prefill-wave
  collectives under Mesh collectives is part of it; the prefill accelerators
  under Fleet-hours take the long prompts.

## Placement beyond TP16

- GLM 5.3 Flash serves only at TP16. Measure B1 latency and aggregate
  throughput from B8 to B1024 for TP16, TP4 x PP4 and PP16 once each runs,
  and choose placement from those numbers (README, Placement). The items
  below are the TP4 x PP4 work; PP16 needs the same stage-local layer spans
  and boundary forwarding.
- GLM 5.3 Flash runs only as one TP16 stage on the common runner.
  `Glm5NextStage::Open`
  (`model-families/glm5_next/source/glm5_next_stage_model.cuh`) refuses a
  stage count other than one or a TP degree other than 16, and its `Sideband`
  refuses a pipeline boundary. The common adapter refuses a `tp_degree` other
  than the model's (`inference/runner/stage_serving_adapter.c`). The GLM Flash
  adapter
  (`modules/glm53_flash_tp16_resident_decode_stage/source/spark_glm53_flash_tp16_stage_serving.c`)
  declares parallel fanout without
  `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP`.
  `tools/glm5_next_gen_tp4pp4_deployment.py` still writes TP4xPP4 configs for
  the deleted glm5_next module target. The runner already gives the embedding
  to the first stage and the head to the last (`owns_embedding`,
  `owns_final_head` in `inference/runner/stage_runner.cu`). Complete
  stage-local layer spans and ownership, boundary forwarding and grouped
  collectives in the runner before calling a TP4xPP4 deployment runnable;
  changing the collective degree alone is insufficient. Prove token parity
  through all four pipeline stages.
- Generate the sixteen-rank TP4 x PP4 deployment directly from the final
  hardware and model contracts.
- Keep stage-local weights, KV, communicators, graphs, and workspaces stable
  while switching B1-B1024 execution widths and speculation policy.
- Complete shared-prefix B8 scheduling with asynchronous DSpark proposal work
  and starvation bounds.
- Complete throughput scheduling that forms dense stage-local microbatches from
  a larger dynamic agent population without request-wide lockstep.

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
- Relocatable expert graphs are built but not wired.
  `runtime/spark_graph_reloc.c` and `runtime/spark_graph_reloc_cuda.c`
  register regions, walk a captured graph, validate two captures against each
  other and patch an instantiated graph's pointer arguments in place
  ([`docs/RELOCATABLE_GRAPHS.md`](docs/RELOCATABLE_GRAPHS.md)), but no driver
  calls them. The common runner (`StageRunnerGraphCapture` in
  `inference/runner/stage_runner.cu`) captures its graphs again in every
  engine, nothing stores a captured graph for reuse, and the captured pointers
  must never move.
- Working-set graphs. Because no runner graph relocates, the common runner
  captures decode graphs only after weightd reports every expert group present
  in a fixed pool (`StageRunnerCheckResident`, `STAGE-RESIDENT`,
  `StageRunnerGraphEligible` in `inference/runner/stage_runner.cu`). With a
  bounded expert pool it runs eager waves that lease each layer's routed
  experts. Full residency breaks I29's bounded residency, and the memory it
  holds is unavailable to co-resident drivers. On top of relocation, capture
  over a leased working set and send a miss to an explicit, reported refill.
  Exit: graph decode on a bounded expert working set for GLM-5.3 Flash,
  GLM-5.3 Full and K3, with the same greedy tokens as the fully resident build
  ([`docs/WEIGHTD_DESIGN.md`](docs/WEIGHTD_DESIGN.md#open-relocatable-graphs)).
- The swap controller (`tools/serving/model_swap.py`) and the chat layer
  (`tools/serving/chat_frontend.py`) keep every configured model addressable.
  `/v1/models` lists them all, `/v1/sparkpipe/status` reports the active model
  and the swap state, and a request for an inactive model waits in its warm
  queue while the controller swaps it in. Neither reports, before a swap is
  attempted, whether an inactive model can be swapped in (its packs present on
  all 16 nodes and its projected memory free). A swap to a model that cannot
  load fails only at that point.
- Enforce the internal NVMe tiers as validated directories with byte budgets
  on the existing root filesystem (owner 2026-10-03: no repartitioning). KV
  has its tier: each deployment node names its `kv_partition`, residentd
  refuses KV directories off it, and `kv_backing_maximum_bytes` and
  `kv_snapshot_maximum_bytes` bound its stores (`node/model_residentd.c`,
  `runtime/model_resident_deployment.c`). The active model shard tier (1 TB)
  has no named directory, byte budget or startup check, and nothing reserves
  the 0.5 TB of system and runtime space against KV and shard growth.
- Reserve at least 1 TB of every external NVMe for direct rank-local model
  access and combine the remaining capacity into the selected striped,
  failure-aware model-data pool.
- Measure and close at least 20 Gb/s useful reads from the pooled model store
  for complete rank-shard promotion workloads.
- The swap controller (`tools/serving/model_swap.py`) promotes one model on
  demand. It stops the other models, releases their KV pools, starts the
  target's engines and waits for every rank. It warms weights where the model
  has a warm step and serves the model only once its API connects every rank.
  On the current build every catalog model swaps in: the medians of the swaps
  logged on 2026-10-13 and 2026-10-14 run from 16 s (Gemma 4, Muse Glimmer) to
  38 s for GLM-5.3 Full and 46 s for K3 (`~/sparkpipe-swap/swap.log` on the
  hub; README, Models). Still open: thirteen catalog models in
  `deployment/hub/model_swap.json` have no warm step. Nothing holds a swap to
  60 seconds (the rank wait alone allows 300 s), and K3 took up to 79 s on
  2026-10-14 with a CPU tenant loading every node. A failed swap leaves no
  model serving because the previous one is already stopped. Close it with a
  bound that fails a swap past 60 seconds and a failed start that leaves the
  previous model serving.
- A model swap waits at most `drain_seconds` (30 s by default) for the stopped
  model's live requests (`drain` in `tools/serving/model_swap.py`) and then
  stops its engines, so a reply still running at that point fails instead of
  resuming when the model returns. The KV side is the next item. Close it by
  holding such requests, with their resumable KV and positions, across the
  swap, proven by a fleet run that swaps away mid-reply and back and completes
  the reply with tokens equal to an uninterrupted run.
- KV survives model eviction and reactivation in code. Destroying the binding
  first saves every published, unsaved chain to the snapshot store
  (`SparkStageKvBindingSaveAtDestroy`, bounded by
  `SPARK_STAGE_KV_DESTROY_SAVE_TIMEOUT_NS`), and parked prefix pages queue
  their saves when they spill. The swap controller drops a stopped model's
  weightd KV pool (`KV_POOL_DROP`), so a reactivated model restores its
  prefixes from the snapshot store. The 2026-10-08 swap cycle restored 4,224
  cached tokens of a 4.2K document on every return
  ([`docs/LITELLM_FRONTEND.md`](docs/LITELLM_FRONTEND.md)). Only the return to
  K3 compared output, and GLM-5.3 Full ran its pre-runner build. Close it on
  the common-runner GLM-5.3 Full: publish a prompt, swap away and back, resend
  it, and see `cached_tokens > 0` with tokens identical to the first run.
- The KV page store reserves its whole backing quota at open (`fallocate`),
  so a full partition or a quota larger than the free space fails the load
  with `CAPACITY_EXCEEDED`, and each deployment node names its `kv_partition`:
  residentd refuses a backing or snapshot directory on another filesystem, or
  a partition on tmpfs, ramfs, NFS, Ceph, FUSE, CIFS/SMB or 9p. The Sparks
  keep KV on the root NVMe (`/`), so the check does not separate KV from the
  OS. Close it on one Spark by loading a deployment whose backing directory is
  on `/mnt/model-warm` (Ceph) and seeing the refusal.
- Left out on purpose (2026-10-02): GLM-5.3 Flash splits its latent KV and
  indexer keys by context at the adapter's TP degree (`context_shard` in
  `modules/glm53_flash_tp16_resident_decode_stage/source/spark_glm53_flash_tp16_stage_serving.c`).
  `SparkKvShardValid` (`include/sparkpipe/spark_kv_shard.h`) accepts degrees 2
  and 4, but the stage runs only at TP16, so the split has never run at degree
  4 and a TP4xPP4 deployment cannot hold 1/16 of the KV per node. Close it
  with the TP4xPP4 runtime above, proven by a TP4xPP4 fleet run with T1 parity
  in which each rank holds 1/4 of its group's latent KV and indexer keys.
- KV pools resize by chunk inside the node reserve with resident demand
  (`runtime/stage_kv_binding.c` pool policy, weightd `KV_POOL_RESIZE`/
  `KV_POOL_STATUS`), and `tests/test_stage_kv_pool_resize.c` moves chunks
  between two bindings in one process under the CUDA stub. Not yet proven on
  the fleet: close it with two co-resident engines on one Spark (two GLM Flash
  lanes on a shared reserve) whose `kv pool grew` and `kv pool shrank` lines
  follow the load, with identical outputs to unshared runs and no
  `KV-PARK-FAILED`.
- Left out on purpose (2026-10-03): GLM-5.3 Full splits every sequence's
  latent KV and DSA index keys across the 16 ranks by context, where the owner
  of position p is p % tp (`context_shard` in
  `modules/glm53_full_tp16_resident_decode_stage/source/spark_glm53_full_tp16_stage_serving.c`,
  `GlmModel::Open` in `model-families/glm52/source/glm52_stage.cuh`). The
  stage runs only as one TP16 stage with 4 heads per rank and a row capacity
  that is a multiple of 16. The split is not yet fleet-proven on the common
  runner. It closes with a TP16 fleet run that shows, on every rank,
  `context_shard=<rank>/16` in the binding line and the stage's sizing line
  (`scatter_units`, `gather_latent`, `select_rows`), passes T1 and the
  accuracy A/B against the bf16 floor, and measures B1, B16, 16K and 128K TTFT
  against 0b5371e. Known costs that stay after the proof: decode waves
  (scatter mode) add a query all-gather and a partial all-to-all per layer,
  with the index candidates riding the gather on the full-indexer layers.
  Folding the query into the projection gather and sending bf16 partials are
  the planned cuts. Decode attention is not bit-exact against the replicated
  split kernel, because the summation order differs; the host and device
  oracles compare against the replica-view shard math
  (`tests/cuda/index_shard_cuda.cu`, `tests/host_cuda/index_shard_host.cu`).
  Prefill waves (gather mode) carry one sequence and exchange the latent keys
  of the whole earlier context. A row digest folded into a MAX all-reduce per
  wave fails the request loudly if any rank's local rows differ. Still open:
  fetching only the selected union's latents, once its size is measured on the
  fleet. Scatter waves are capped at 64 rows (`GLM_SCATTER_ROWS`).
- Cross-engine shared prefix pages are in code
  ([`docs/KV_SHARED_PREFIX.md`](docs/KV_SHARED_PREFIX.md)):
  - weightd's `--kv-shared-window-bytes` gives each KV layout a window
    that every binding of that layout maps after its private pages;
  - new pages come from the window, and published prefixes are found
    through a shared index in weightd's memfd;
  - engines read their peers' prefix indexes
    (`model_api --peer-runtime-root`).

  Still open:
  - Bindings with recurrent state (GLM Flash, K3) keep their prefixes
    private, because their prefix state lives in a per-engine state store.
    Close it by keeping the state record in the window too.
  - Window pages are never spilled. A reclaimed slot's prefix survives
    only through its owners' snapshot saves.
  - The window size is one node-wide byte count per layout.
  - Fleet proof: two GLM Full lanes on one node, where the second lane's
    prompt hits the first lane's prefix with no extra device pages and
    identical tokens.

## KV tiers

- The 24-hour retention outside a running model (weightd's purge thread and
  its release of detached KV pools, the draft servers' context expiry, the
  batch engine's slot clearing and the prefix index's record expiry) has no
  fleet proof yet. Proof: backdate an inactive model's snapshot store while
  another model serves and see `weightd kv purge` empty it within a minute;
  swap a model away and see `reason=detached_retention` an hour later; see
  `CONTEXT-EXPIRED` and `prefix index expired` on a backdated hub. The
  retired API directories on the hub keep prefix index files that no engine
  opens; they need deleting by hand.

- Left out on purpose (2026-10-04): JIT-KV W3
  (`docs/archive/JIT_KV_RESPONSE.md:47-49`) is partly done. The slot file, the
  lane pager and the header-only `LmCache` are deleted, and every driver
  serves through the binding's two stores: the per-engine spill page store
  (`cache/kv_page_store.c`, digest-checked, quota reserved at open) and the
  persistent snapshot store (`cache/kv_snapshot.c`). A parked prefix page
  whose entry is not yet saved queues its snapshot save when the arena parks
  it (`SparkStageKvBindingPark`, `SparkKvPageCacheSaveParked`), and the save
  reads the spill copy. Still left: the spill store is an `O_TMPFILE` file, so
  pages of running sequences cannot be reused after a restart.
  `cache/nvme_tier.c` remains only for the unwired topology switch
  (`scheduler/topology_switch.c`). The external provider client
  (`cache/store/kv_store.c`, `cache/store/stage_kv_client.c`) is still built
  into the common library although no driver calls it. Close it by making the
  spill store write the snapshot format and deleting the provider client,
  proven on the fleet by a spill, a residentd restart and a restore that hits
  through the snapshot store alone.
- Oversubscription is fleet-unproven. The engine admits lanes up to the
  physical pages, and the arena parks the pages of lanes outside the running
  wave one at a time through the page store and brings them back when the
  lane rejoins, so a lane is parked and restored page by page rather than as
  one contiguous transfer. Close it with a fleet backpressure run at 2x device
  pages where parked lanes restore bit-exact at B1 and B16.
- The format-2 KV snapshot tier (page files naming their parent, separate
  state files, leaf-first eviction, ordered writes) is host-tested only.
  Prove it on the fleet: a K3 -> GLM -> K3 swap that restores a 270K
  conversation in under 15 s, a store of 8 GiB per rank that keeps the newest
  conversations whole while older ones go tip-first, and an open that removes
  the format-1 files of the previous release once.
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
- Spilled prefixes on the common binding reach the snapshot store: a parked
  page of an unsaved prefix entry queues its save from the spill copy, and
  released sequences already queue theirs. Every driver now serves through the
  binding (`inference/runner/stage_serving_adapter.c`). Not yet fleet-proven:
  close it with a prompt spilled before a kill -9 of residentd that hits after
  the restart with tokens equal to an uninterrupted run.
- KV eviction follows request priority and deadline.
  - Prefix-cache entries rank by the highest priority of the requests that
    published or reused them, then by whether the snapshot store holds
    them, then by age (`SparkKvPageCacheSelectVictim`,
    `SparkKvPageCacheResidentVictim`).
  - Each admission stamps its lanes' pages with the wave's priority and
    deadline, and a sequence's release clears them.
  - The arena parks the lowest priority first, then the latest deadline,
    then by reuse value and recency
    (`SparkKvCacheBlockIsBetterEvictionVictim`).

  Two limits remain: a page shared across waves keeps the stamp of the
  last admission that used it, and the deadline is the wave's latest.
  Not yet fleet-proven: close it with an oversubscribed run with two
  priority classes, where the higher class keeps its hits and its TTFT
  stays flat.
- A full backing store queues new work instead of failing it. The page
  cache first relieves backing: it releases an idle restored page's record
  or evicts an unused entry. If the store is still full (`KV-BACKING-FULL`)
  for an admission whose lanes are all new sequences, and a live sequence
  outside the wave could free backing, the lane transactions answer `BUSY`
  (`SparkKvLaneTransactionsPrepare`, `cache/kv_page_cache.c`). The engine
  then retries with backoff inside the in-flight budget; the binding's
  arena line counts these as `backing_full_queued`. Two cases still answer
  `CAPACITY_EXCEEDED`: nothing outside the wave is live, or a running lane
  needs backing for its growth or restore. Waiting there could leave waves
  stalled on each other for the whole budget. Not yet fleet-proven: close
  it with a one-Spark run on a small backing quota where new requests wait
  and complete as running ones finish, and no running request fails.
- KV NVMe writes are metered, not limited. Every engine on the common binding
  counts every spill, recurrent-state and snapshot write and raises one
  `KV-WRITE-ALERT` per hour above 20 GiB/hour
  (`include/sparkpipe/spark_kv_write_meter.h`). The alert shows in
  `kv_store_report` and as `kv_write_alert_rank` on `/health`, but nothing
  forwards it beyond the API's `/health` yet. Close it with a Spark run that
  writes past the threshold and shows the alert line, the report fields and
  the health rank.
- Recurrent state is checkpointed sparsely (`cache_checkpoint_token_count` in
  the adapter descriptor; K3 every 16384 tokens, GLM Flash every 4096). The
  engine publishes every block as KV but keeps state only at stride multiples,
  at the last block boundary of a prompt and at a finished reply's last
  computed token, and it resumes only at those points. The binding sizes the
  state store from the backing budget (`SparkStageKvBindingStateBudget`). A
  stateful publish that deduplicates onto an existing stateless entry keeps
  the stateless entry, so that checkpoint is lost and the next resume
  recomputes (`stateful_dedup_lost_count`); the store has no record move
  between logical pages. Follow-up turns resumed after the previous reply on
  K3 and GLM Flash on the fleet on 2026-10-07 (b3d47b07f, before the common
  runner). Close it with a record move in `SparkKvPageStore` and a repeat of
  the follow-up-turn fleet run on the common-runner builds.
- A store whose parks fail degrades to recompute-only prefix reuse.
  - The first last-resort park failure (`KV-PARK-STALLED`) logs
    `KV-STORE-DEGRADED` (`cache/kv_cache.c`).
  - From then on, the page cache discards unused prefix pages instead of
    spilling them (`SparkKvPageCacheDiscardInsteadOfSpill`), so they are
    recomputed on their next use.
  - New work whose admission still stalls waits for running sequences
    (`BUSY`).
  - The next successful park logs `KV-STORE-RECOVERED`.
  - The binding's arena line reports `degraded`, `degraded_entries`,
    `degraded_discards` and `park_stall_queued`.

  A running lane that needs a park still fails with `IO_ERROR`, and
  snapshot saves to the same disk fail on their own. Not yet
  fleet-proven: close it on one Spark by remounting the backing directory
  read-only mid-run; requests should keep completing through recompute,
  and the log should show the degrade and the recovery.

- **Binding test race.** `tests/test_stage_kv_binding.c` can fail on a busy host (1 of 300 runs in one loop, 0 of 300 in the next): a page's residency count reads one high right after a copy-on-write admit, while a save or spill still holds a reference. The test should read the count after the save settles without draining the copy it is checking.

## Prefix reuse (I23)

Prefix reuse is required and non-compliant adapters are refused at load
([`docs/DRIVER_ACCEPTANCE.md`](docs/DRIVER_ACCEPTANCE.md), Prefix reuse is
required).

The load check reads only the `PREFIX_REUSE` descriptor bit
(`runtime/model_serving_adapter.c`,
`SparkDescriptorCheckRequiredCacheOperations`) and asks for no I27 proof.
Production replies report prefix hits (`cached_tokens` above zero in the hub
chat log, 2026-10-14), but no adapter on the stage runner has passed a fleet
I27 session. GLM Full is listed under the common-code debt below; these
adapters also need one:

- ling3 (`modules/ling3_resident_decode_stage`) attends through the common
  binding's page tables and stores its KDA state and convolution window per
  cached prefix through the recurrent-state hook (`RecurrentCopy`,
  `model-families/ling/source/ling_stage.cuh`), but has no fleet I27 proof.
- laguna (`modules/laguna_tp16_resident_decode_stage`) restores prefixes
  through the common stage binding
  (`inference/runner/stage_serving_adapter.c`, `runtime/stage_kv_binding.c`)
  with real page tables and separate logical and physical page capacity, but
  has no fleet I27 proof: cold vs warm token parity at B1 and B16 on its TP16
  lane.

Related common-code debt:

- Left out on purpose (2026-10-02): the publish of a reply's final partial
  block has no fleet proof, and multi-token chains that stop early never
  publish their tail. A completed request with an unpublished tail queues
  `CACHE_PUBLISH` before its release when its last step's tokens were all
  accepted (`runtime/model_batch_engine.c`,
  `SparkModelBatchHandleDecodeCompletion`). The stage serving adapter serves
  the publish-only frame through `SparkStageKvBindingPublishFrame`
  (`inference/runner/stage_serving_adapter.c`, `runtime/stage_kv_binding.c`),
  and every adapter declares `CACHE_PUBLISH`
  (`runtime/model_serving_adapter.c`,
  `SparkDescriptorCheckRequiredCacheOperations`). When a decode chain of a
  model with recurrent state (GLM Flash, K3, Ling, the Qwen 3.8 hybrids) stops
  on EOS mid-chain, the stage runner has already run the remaining chain steps
  (`inference/runner/stage_runner.cu`). The lane's recurrent state then
  belongs to a later position, so the tail stays unpublished and a follow-up
  turn recomputes it. Close the chain case by capturing or rolling back
  recurrent state to the tail through the binding's recurrent-state hook
  (`SparkStageKvBindingRecurrentCapture`). Prove both cases with fleet I27
  sessions whose second turn extends a reply that ended on EOS mid-block, with
  `cached_tokens` covering the whole first reply and tokens identical to an
  uncached run.
- Left out on purpose (2026-10-02): residentd's slot claim checks slot
  ownership and the lane's request id, generation and sequence id
  (`node/model_residentd.c`), not position continuity. Every adapter now
  checks continuity through the common `SparkStageKvBindingContinuity`
  (`inference/runner/stage_serving_adapter.c`), but no fleet run has shown it.
  It is closed by a fleet run in which a skipped-position submission to a GLM
  Full lane gets an explicit continuity error.
- Left out on purpose (2026-10-02): GLM Full declares prefix reuse
  (`modules/glm53_full_tp16_resident_decode_stage/source/spark_glm53_full_tp16_stage_serving.c`,
  `PREFIX_REUSE`) and restores prefixes through the common binding
  (`inference/runner/stage_serving_adapter.c`, `runtime/stage_kv_binding.c`),
  but no fleet I27 session has passed at TP16. Run `tools/i27_session.py`
  against the GLM Full TP16 API with `--restart-command` and
  `--writeback-fault-command` hooks and pin its PASS receipt. It covers B1 and
  B16 cold/warm parity, a copy-on-write mid-block prefix, abort mid-prefill,
  eviction and recompute, residentd restart and a failed write-back. Each case
  runs against an uninterrupted control with identical tokens and
  `cached_tokens` above zero on every expected hit, plus exact T1.
- Left out on purpose (2026-10-02): the engine keeps its prefix index across
  rank-session changes and reloads it from `prefix_index_path` at start, but
  no fleet run has shown a restored prefix after a residentd restart. Fleet
  proof: publish a 4K prompt, restart residentd on all ranks, resend it, and
  see `cached_tokens` equal to the published prefix with tokens identical to
  the first run.

## Dynamic batching

- Kimi K3 and GLM 5.3 Full serve at most 8 sequences at once and GLM 5.3
  Flash 16. GLM Flash's shared batch scheduler selects arbitrary counts up to
  its configured limit; a power-of-two kernel bucket is not an admission rule.
  The GLM deployments set active and resident capacity to 16 or less
  (`deployment/glm5_next_tp16/model_resident.json` and the GLM records in
  `deployment/lanes.json`), while the stage
  serving adapter allows up to `SPARK_STAGE_SERVING_MAX_LANES` (128). KV is
  paged from the weightd pool through `runtime/stage_kv_binding.c`, so a
  sequence no longer reserves its full context. Qualify explicit larger
  capacities with arrivals/completions and memory gates before raising the
  limit.
- Making a page resident when the resident pool is full still scans every
  resident slot for a victim instead of taking the least recently used
  unreferenced entry; pool fullness, retained pages, evictions and spill
  traffic are now reported in the API's `kv_store_report`.
- Prewarm the decode graphs a deployment admits at load. The stage runner
  captures a graph per row count the first time that count runs, and only up
  to 16 rows with one row per sequence (`STAGE_RUNNER_GRAPH_ROWS`,
  `StageRunnerGraphEligible`, `StageRunnerStep` in
  `inference/runner/stage_runner.cu`). The first wave at each count pays the
  capture, and wider waves run eager.
- Qualify mixed arrivals, priorities, prompt lengths, shared prefixes, cache
  pressure, cancellation, and starvation bounds.
- Priority and deadline reach admission, prefill and decode scheduling (the
  batch engine selects by priority, then earliest `deadline_ms`, and a prefill
  or decode submission carries its latest lane deadline to residentd as
  remaining time), and KV eviction ranks by request priority. Still open:
  speculation, gang scheduling and KV snapshot I/O ignore both; the binding's
  restore worker and saver run in queue order (`runtime/stage_kv_binding.c`).
- Dense projections above sixteen rows (`LM_SKINNY_ROWS_WIDE`,
  `inference/kernels/skinny.cuh`) leave the skinny kernel for tensor-core GEMM
  with a different accumulation order. The tensor-core latent attention for
  wide waves (`inference/kernels/attn_shard_prefill.cuh`) also sums in a
  different order from the per-row decode kernel. Prefill and batched-decode
  logits are therefore not bitwise equal to B1. Extend the row-blocked kernels
  that keep the skinny order wherever they match the tensor-core throughput.
- No gate checks batch invariance. The COMPSEC-17 run of 2026-09-28
  (`qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/REPORT.md`)
  sent the same 17 prompts concurrently four times: each run differed from
  the sequential completions in one or two cases, and the runs differed from
  each other. Add a gate that replays a fixed prompt set sequentially and
  concurrently and requires byte-identical completions, and run it with
  every serving release until the batch kernels pass it.
- Restore hints are fleet-unproven. When the engine first finds a cached
  prefix for a queued request, it sends every rank a `CACHE_HINT` (resident
  IPC kind `CACHE_HINT`). residentd hands it to the adapter's `cache_hint`,
  and the stage serving adapter (`StageServingCacheHint`,
  `inference/runner/stage_serving_adapter.c`) passes it to the binding as a
  `CACHE_HINT` admission. The admission queues the restore so it runs while
  the request waits for dispatch. Close it on the fleet: a queued request
  whose prefix was evicted to the snapshot store shows a TTFT that drops by
  the restore time against a run with hints dropped.
- A prefix-cache hit reuses KV and KDA state computed however the source
  request ran: one-row prefill for prompt tokens, batched decode rows for
  generated ones. Until batched rows equal B1, a warm and a cold run of the
  same prompt can differ (#1230). Make them equal, or keep checkpoints of
  generated tokens out of reuse for verified requests.
- GLM Flash: no GPU test compares a replayed stage-runner decode graph's
  logits (`StageRunnerStep`, `inference/runner/stage_runner.cu`) with an eager
  wave's at the same context. `tests/test_glm5_next_graph_regime.py` checks
  only `spark_glm5_next_graph_regime.h`, which no production code includes.
  Add the logits comparison to the CUDA validation at contexts 63, 64, 2,048,
  2,049 and 4,099.
- Restore bandwidth is bounded only by construction: one binding worker reads
  restores one job at a time, so restore reads never exceed one sequential
  stream per engine, but nothing measures the drive or admits against it.
  Close it on the fleet with a burst of spilled-prefix requests whose restore
  bytes per second stay at or under the drive's measured rate while the decode
  step time of running lanes is unchanged.

## Model contracts

- Add an exact checkpoint-derived contract for MiniMax H3; `model_contracts/`
  has none.
- Remove `deployment/minimax_text_tp4`; the MiniMax driver is gone.
- `model_contracts/glm53_full_authoritative.json` carries no rope theta or
  interleave flags, so `tools/generate_recipe.py` reads them for the GLM Full
  recipes from `spark_glm52_model.h`. Add them to the contract's geometry at
  its next re-freeze (a contract edit changes the `CONTRACT_SHA256` every GLM
  Full build pins) and read them from there.
- Retain independent numerical, transport, memory, and performance gates for
  every model and precision route.
- GLM 5.3 Full serves Z.ai's FP8 release (`glm53full.fp8lin.tp16`), and the
  BF16 release fits one fleet. #1558 builds the stage with BF16 codecs on the
  BF16 packs (`glm53full.bf16.tp16`) and scores both against the CT-short
  corpus through the API; its result decides the served precision.
- `model_contracts/glm53_full_authoritative.json`'s top-level `model_id` names
  `RadixArk/GLM-5.3-NVFP4`, which no lane serves, and
  `tools/glm52_model_contract.py --print-build-identity bf16` prints the FP8
  revision. Re-freeze the contract with the served source on top and give each
  codec its own source's revision.
- K3's BF16-activation/MXFP4-weight expert paths run in production. Decode
  uses the skinny experts. Wide waves use `LmCellMmaExperts`
  (`inference/kernels/cell_mma.cuh`) and `LmGemmKernel<LmBf16Format, LmMxfp4>`
  (`inference/llms/kimi_k3/unity.cu`), with UE8M0 scales decoded in the load
  and the route build in `inference/kernels/route.cuh`. Only the skinny path
  has a device test against an f64 reference
  (`tests/cuda/skinny_mxfp4_cuda.cu`); the host tests stand a shim in for
  `LmCellMmaExperts`. Qualify the tensor-core expert paths and the full expert
  path (route, gate/up, down, combine) against a dequantised reference at
  prefill widths.

## Korean models

The sources are pinned. K-EXAONE 2.0 serves; A.X K2 and Solar Open 2 do not yet.

- **K-EXAONE 2.0** (`LGAI-EXAONE/K-EXAONE-2.0-750B-A37B@fec0c8dd`, BF16, 1.50 TB): 78 layers with 64 query and 8 KV heads of 128; 20 layers attend globally without position encoding, 57 a 128-token window and one a 4,096-token window; 256 routed experts (top 8) plus one shared, after two dense layers; one MTP layer. Its DSpark drafter is `model-dspark.safetensors` of `LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-DSpark@e42409db`, whose other files are the base weights.
  - **Serving:** contract, driver (`kexaone2` on the Gemma stage), packs and lane 11.
  - **Measured:** 23.0 tok/s at B1 without speculation and 324 tok/s prefill at 6,850 tokens, with every expert split over the ranks by rows. Giving each rank 16 whole experts measured 12.0 and 273: a token's 8 experts landed unevenly. Both layouts score the same on the A/B corpus (McNemar p 0.39).
  - **Missing:**
    - The numpy reference (`tools/kexaone2_reference.py`) agrees with the served tokens at all 92 positions of six prompts (served logprobs within 0.11, median under 0.005). With reasoning off the model answers 17 × 23 as 401 in the reference too, so the UI smoke's one miss is the model's answer.
    - Speculation (its MTP layer or the DSpark drafter) is not wired.
- **A.X K2** (`skt/A.X-K2@d0805cfd`, block-128 FP8, the precision it was trained in, 0.69 TB): 61 layers of latent attention (query rank 1,536, KV rank 512) with top-2,048 sparse attention through a 64-head indexer, a gated attention output and a gated norm; 256 routed experts (top 8) plus one shared; YaRN to 256K. Drafter `skt/A.X-K2-DSpark@0b278334`. GLM 5.3 Full has the same attention and expert design.
- **Solar Open 2** (`upstage/Solar-Open2-250B@9190fbe6`, BF16, 0.50 TB): 48 layers, three linear-attention layers (short convolution 4, 64 heads) for each GQA layer, no position encoding, 320 routed experts (top 8) plus one shared, 1M context. The Upstage Solar License asks that derivatives be named Solar and show "Built with Solar". The nearest served design is the linear-attention path of Kimi K3 and Ling.

Each needs, in order: the checkpoint contract, the driver on the common stage runner, the packer emitter and packs, a lane record, the release gates and qualification, speculation with its drafter, then the perf climb that every core model gets.

## Speculation

- Tree verification is not wired. The policy engine resolves trees
  (`SparkSpeculationPolicyResolveVerifierTree`), but no caller passes one.
  Sequence state stores chains, the seam collapses DFT3 trees to one chain
  (`SparkSpeculationSeamExtractChain`), and no verify frame applies a
  tree-attention mask. `spark_speculation_tree.h` has no production includer.
- Build multi-drafter composition over the seam, after the single-drafter
  agreement matrix is measured: verify the DFT3 tree instead of collapsing it,
  allow local plus remote sources together, and attribute acceptance per
  `source_bit`. `SparkSpeculationDrafterMix`
  (`src/spark_speculation_drafter_mix.c`) only switches between a primary and
  a fallback source and has no production caller.
- Speculation serves on Kimi K3, GLM 5.3 Full and DeepSeek V4.1 Flash, whose
  adapters declare `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATIVE_VERIFY`.
  Still to do:
  - verify frames for GLM 5.3 Flash and the twelve other models, with GLM 5.3
    Flash's recurrent state restored to the accepted row through the
    binding's recurrent-state hook;
  - graphs for verify frames that carry several sequences: the runner
    captures a graph for one row per sequence or for one sequence's verify
    frame (`StageRunnerGraphEligible` in `inference/runner/stage_runner.cu`),
    so shared verify frames run eager;
  - multi-row verify numerics differ from a 1-row decode by the reordering
    floor, so speculative output is judged against the floor like batched
    output. Add the accuracy gate as excess over the floor for edit-style
    requests, and show that plain prompts with no matches lose no speed
    beyond noise.

## Packaging and provenance

- License follow-ups (2026-10-18; the code is Apache-2.0 and NOTICE covers
  every third-party file kept in the tree):
  - The HY4 reference tools and module were first written against
    AngelSlim's hyv4 patch, which states no license. NOTICE now credits the
    MIT implementation llama.cpp merged at 49c0dc82 (src/models/hy-v4.cpp);
    a line-by-line check of the tools and module against it is open.
  - The serving obligations of each model license (K3's display and revenue
    conditions, the Qwen 3.8 licenses, OpenMDW for Laguna, the NVIDIA Open
    Model Agreement for NVIDIA checkpoints) are not yet written down in one
    place.
  - The ds4-eval fixtures keep GPQA questions in plain text, which GPQA's
    dataset card asks users not to publish; a decision on keeping only ids
    is open.
- Add the upstream implementation commit to stage-pack provenance at the next
  pack format revision.
- Publish one synchronized pack-environment manifest for Python, CUDA, and
  model conversion dependencies.
- Replace the per-family packers (the stage runner's `tools/*_tp16_pack.py`
  and `tools/k3_pack.py`, and the remaining `tools/*stagepack*.py`) with the
  universal packer
  ([`docs/DRY_PACKBUILDER_PROPOSAL.md`](docs/DRY_PACKBUILDER_PROPOSAL.md)):
  one CLI, one codec table, per-family byte-compatible emitters, each gated on
  byte identity with its existing packs.

## Driver consolidation

- Kimi K3's device code (its slices and unity kernels) lives in
  `inference/llms/kimi_k3/`, outside `model-families/k3/` where every other
  model's math lives. Move it there.
- Every stage-runner model, K3 included, resets every runner slot and the KV
  binding on a client reset and refuses stale-generation submissions
  (`StageServingReset` and the `reset_generation` check in
  `inference/runner/stage_serving_adapter.c`), but no fleet run has proved it:
  run a client reconnect after a completed request, after which a request on
  the same slot matches a fresh-process run token for token.
- The stage runner is one driver per model and takes any row count up to its
  `max_input_row_count`, but it captures its per-row-count decode graphs (1 to
  16 rows, `STAGE_RUNNER_GRAPH_ROWS` in `inference/runner/stage_runner.cu`) on
  the first step of each row count, so the first request at a new batch width
  pays the capture. Prewarm them at load. The root `Makefile` still passes
  `-DSPARK_BATCH_BUCKET=1024u` (`DEFAULT_BATCH_FLAGS`), which no source reads;
  drop it.

## Runtime completion

- A submission that fails residentd's cache validation is retried until its
  deadline, and every attempt resets the session on all ranks: on 2026-10-14
  two GLM-5.3 Full requests with a 60 s and a 3 s deadline (failing on the
  deadline mismatch fixed in `cache/kv_page_cache.c`) caused about 2,950
  session resets on spark0, roughly 50 a second, until their deadlines ran out
  (`model_residentd session_reset ... abort=validation_failed`,
  `runtime/model_batch_engine.c`). A request without a deadline would retry
  for the engine's in-flight budget. Fail the request after a small number of
  validation failures of the same route, report the cause to the client, and
  prove it on one Spark with an injected validation failure: the request fails
  with its cause and the rank resets once.
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
- The stage runner returns a weightd lease failure (`NO_LANE`, `EVICT_DENIED`)
  from a non-resident expert lease (`StageRunnerExpertWeightsService` in
  `inference/runner/stage_runner.cu`) unmapped, and `StageServingSubmit` in
  `inference/runner/stage_serving_adapter.c` returns it as the submit status,
  which residentd's IPC cannot encode (`runtime/model_resident_ipc.c` refuses
  statuses above `UNSUPPORTED`). `SparkModelServingCompletionStatus` and
  `SparkModelServingCompletionStatusStateless` in
  `include/sparkpipe/spark_model_serving_adapter.h` are called only by tests.
  Recurrent lane state (KDA, GDN) advances layer by layer and is restored only
  after a prefix attach (`SparkStageKvBindingRecurrentRestore` in
  `runtime/stage_kv_binding.c`), so a retried frame would run on half-updated
  state. Map the failure in the adapter to `CAPACITY_EXCEEDED` for models with
  recurrent state, restore the lanes' recurrent state before a retried frame,
  then map it to `BUSY` for every model.
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
- The stage runner runs a row count eagerly when its graph capture fails
  (`STAGE-GRAPH-CAPTURE-FAILED` in `inference/runner/stage_runner.cu`) but its
  snapshot never sets `SPARK_MODEL_DRIVER_DEGRADED_EAGER_PATH`, so `/health`
  (`node/model_api.c`: 503 with `degraded_rank` and `"degraded_path":"eager"`)
  never names the rank and `tools/compsec17.py` gives a verdict on a degraded
  run. Set the flag in the runner snapshot, then force a graph capture failure
  on a Spark and see the rank named in `/health` and the qualification bundle
  refused.
- Rank loss is reported but not fleet-proven: `/health` answers 503
  `degraded` with `connected_ranks` and `missing_rank` while a rank is gone or
  a failure is latched, a held request retries only within the engine's
  in-flight budget and fails naming the missing rank, and a latched failure
  stays until the rank session changes. Close it with a fleet run that stops
  one rank's residentd mid-run: `/health` turns degraded, held requests fail
  at their deadline, and new requests complete once the rank returns.

## Serving API

- `text/tokenizer.c` knows split regexes only by exact string: the GLM
  digit-run pattern, the Qwen letter-and-mark pattern, the DeepSeek
  digit/ideograph sequence (V4.1 Flash, Hy4), the letter-class patterns (MiMo,
  Qwen3.8-27b nvfp4, Ling, the last with possessive quantifiers), laguna's
  leading newline-run split and muse-glimmer's cased letters. Any other
  pre-tokenizer fails the load with an error naming the file. gemma4's
  `Replace` plus `Split " "` loads: the split is a no-op once the normalizer
  has replaced every space, but this has not been checked against gemma4's own
  file.
- The tokenizer implements the `NFC` normalizer (Ling, Qwen3.8, MiMo) and
  the metaspace `Replace`. Any other normalizer (`Sequence`, `NFKC`,
  `Lowercase`, another `Replace`) fails the load with an error. Implement
  one when a model declares it.
- Sampled rows take a second, eager head pass after the step: the full-vocab
  BF16 head (`StageRunnerDistribution` in `inference/runner/stage_runner.cu`)
  instead of the certified FP8 head greedy rows use (`StageRunnerHeadRows`),
  outside the step's CUDA graph, because the certified screen prunes with
  un-noised bounds. Noise the screen bounds and capture the sampling pass in
  the step graph to lift both.
- The chat layer silently turns a request for no reasoning into the lowest
  effort for models whose template has no off switch. GLM-5.3 Full and Flash
  map `minimal` to `reasoning_effort: low` and Muse Glimmer to
  `reasoning_strength: low` (`deployment/hub/chat_frontend.json`). GLM Flash
  and Muse Glimmer still reasoned with reasoning off, so `tools/ui_smoke.py
  --reasoning both` fails that case for them; GLM Full happened to pass it.
  Qwen 3.8 Max maps no `minimal` at all and answers it with a 400, and Laguna
  returned no reasoning with reasoning on (release qualifications of
  `e05394f94` and `1d89e0377`, 2026-10-14). Answer reasoning off with a clear
  error, or a documented per-model capability in `/v1/models`, where the
  template cannot turn it off, and find out why Laguna does not reason when
  asked.
- Every stage-runner lane, GLM Flash included, is sized for every resident
  sequence at full length by default: `tools/stage_lane.py` sets
  `kv_physical_pages` to `sequences * kv_pages` (each sequence's
  `max_sequence_positions` is `kv_pages` pages) unless `--kv-physical-pages`
  (`STAGE_KV_PHYSICAL_PAGES` in `tools/stage_lane.sh`) names a smaller pool,
  and it takes that pool as a page count. Size the physical pool from a byte
  budget the lanes share and let paged admission share it.

## Fleet-hours

Of the fleet-hours product in `README.md`, the tree has the terms (`site/terms/`, published by `tools/site_terms.py`), the Fleet List, the privacy policy, the price and rate table `site/pricing.js`, and a checkout preview (`site/buy.html`) that keeps everything in the browser. Nothing else exists. The business flow is settled by the legal review of the terms, so building waits for it.

- **Accounts.** Ethereum-address accounts; acceptance of the terms as an EIP-712 message signed over the text's hash; business details and automatic sanctions screening; the business check (company documents, owners of 25% or more) with its 720-hour holding limit; recovery addresses.
- **Payments.** The payment processor's USDC invoices and confirmations, refunds of VAT that was not owed, USD wires from abroad for two-year customers, and VAT invoices: tax invoices for Korean businesses and zero-rating evidence for wires. The processor is not approved yet.
- **Ledger.** An append-only, hash-chained file of every payment, issue of hours, reservation, displacement, credit and consumption; a published daily root; per-customer proofs. Balances must come only from replaying it.
- **Reservations.** One writer, the per-fleet calendar and its API, the 7-day hourly horizon, displacement of hourly hours more than 6 hours ahead with an email, consumption and expiry, and the two-year tranches, gates and window elections.
- **Fleet access.** Access tokens scoped to an account's reserved hours, the gate in front of the fleet that admits only the current hour's customer, the customer-facing API (sparkpipe.ai has no API today), and loading the reserved model before the hour starts.
- **`availability.txt`.** Generated from the calendar and fleet state every 30 seconds.
- **Credits.** Automatic credits for downtime over 15 minutes and for performance below the threshold, from the fleet's own records.
- **Won payments.** Electronic tax invoices (세금계산서) issued for each won order, matching of incoming bank transfers to orders, and won amounts in the ledger beside the hours they buy.
- **Korean site.** The checkout preview (`site/buy.html`) is English only; the console that replaces it needs Korean from the start. The Korean address list behind the language redirect (`tools/site_kr_networks.py`, `docs/LITELLM_FRONTEND.md`) is rebuilt by hand; APNIC changes it daily, so it needs a monthly refresh job.
- **ESTIMATED values on sparkpipe.ai.** The site shows projections marked ESTIMATED (the home page Speed and Roadmap sections, `fleets.html#estimates`). Each is replaced by a fleet measurement, with its receipt in `PERFORMANCE_STATUS.md`:
  - prefill accelerators: boxes of four RTX 5080 cards (PCIe reduce inside a box, 100 Gb/s per box over the 32-port 400 Gb/s switch) prefilling a 128K-token prompt, with 4 / 8 boxes: Kimi K3 37 / 18 s, Qwen 3.8 Max 4-bit 35 / 17 s, K-EXAONE 2.0 35 / 18 s, A.X K2 16 / 8 s; a 1M-token Kimi K3 prompt in about 17 / 9 minutes. Assumes weights streamed once per prompt at 85% of 50 / 100 GB/s, cards at 40% of dense peak (FP8 450, BF16 225 TFLOPS) including attention, and short prompts kept on the fleet. Needs the boxes, the switch and an accelerator prefill path;
  - decode estimates take a token's time as each node's share of the weights at the measured 245 GB/s plus 0.44 ms per layer, the fixed cost Kimi K3 measures (GLM 5.3 measures 0.31 ms). Fleet 1: K-EXAONE 2.0 19 tok/s, A.X K2 28. Fleet 2 (eight M5 Ultra Mac Studios, 90% of 1.2 TB/s, the same cost per layer): Kimi K3 23 tok/s, K-EXAONE 2.0 23, A.X K2 33; it needs the Metal backend and the Thunderbolt 5 link class. Lowering the per-layer cost raises every figure, and the site follows each measurement;
  - Qwen 3.8 Max's 4-bit build serves at 10.4 tok/s on Fleet 1, which the per-layer model does not explain, so it has no Fleet 2 estimate; FP8 across Fleets 1 and 2 needs a two-fleet pipeline;
  - speculation at 1.5 times plain decode on agent coding turns, as measured on short edits; tree drafts carry no estimate until they are built and measured;
  - each candidate model also needs a driver, qualification and a license check: K-EXAONE 2.0, A.X K2 and Solar Open 2 (see Korean models), Nemotron 3 Ultra, Mistral Large 3, Inkling (BF16 across two fleets, or an FP8 build that passes the accuracy A/B) and Muse Spark Open (gated license under review). Nemotron 3 Ultra and Muse Spark Open get site estimates once their layer counts and formats are checked against the per-layer model.
- **LiteLLM door limits.** LiteLLM consumes a request's `priority` for its own scheduler and does not forward it, so the batch engine's priorities do not cross the door. `tools/generate_litellm_config.py` emits `num_retries: 1`, which can duplicate a generation; the hub's hand-written `config/litellm-config.yaml` uses 0. The LAN door on port 4000 has no accounts, keys, quotas or spend logs.

## SparkPipe Chat

- **AI label on downloaded chats.** Copied replies end with the bilingual AI label (`ui.watermark`); chats downloaded as files carry none, and Open WebUI v0.12.0 has no setting for it.
- **Notice before sign-up.** Open WebUI's sign-up page cannot show text. The AI and privacy notice appears on the waiting screen right after sign-up, before the account is approved or used.
- **Field search.** None of README's field search exists. Each field is an Open WebUI tool beside `tools/serving/open_webui/tools/github.py`, shown as a checkbox and attached per chat, and every answer cites the documents it used. Sources to evaluate first: Korean statutes and court decisions (the national law information center's open API), EU law (EUR-Lex), US federal law (eCFR and the Federal Register), medicine (PubMed), Korean company filings (Open DART) and patents (KIPRIS). A tool whose source is abroad sends the query out of Korea and must say so before the user ticks it.
- **Memory switch state.** The toolbar shows the memory setting it last sent from this browser; a `/memory` typed by hand or sent from another device is not reflected. A chat-layer endpoint that reports a chat's memory settings would let the toolbar read the real state.
- **Local mode needs an account.** Local mode reaches the models through the user's Open WebUI session; access by API key without an account is #1549.
- **Tunnel port 18080.** `sparkpipe-ai-door` forwards VPS 127.0.0.1:18080 to the hub's port 4001, where nothing listens, and the VPS key may listen on 18080. Removing both is an SSH change for the owner.

## Production qualification

- Fold the engine-evidence checks of the closed end-to-end suite (#1491) into the release qualification: assert `cached_prompt_tokens`, `stale_prefix_recomputes`, `status` and `finish_reason` from each request's `request_measurements` line, so prefix sharing, KV restore and park/resume pass only on engine evidence.
- `scripts/release_rollout.sh` captures the old and the new build through the
  live prefix cache, so a capture can be served from KV another build wrote:
  in the 2026-10-25 release most old captures read their long prompts from
  cache entries of earlier builds. Clear the model's cached state on every
  node before each capture and fail the comparison unless every capture
  reports `cached_tokens` 0.
- `tools/sparkpipe_weightd_vmm_verify.sh` prints `SKIP` and exits 0 when CUDA
  or a GPU is missing, so a receipt run on the wrong host reads as a pass.
  Make both cases exit nonzero with the named reason.
- Repeat accepted transport and model measurements from clean merged `main`,
  rebuild the exact release on Spark hardware, and retain all receipts.
- Close exact-checkpoint numerical parity and end-to-end service gates for each
  model before reporting it production-ready.
- `tools/glm5_next_driver_compare.py` and
  `tests/test_glm5_next_driver_probe.c` outlived the glm5_next module and
  probe that `82e63a909` deleted: the compare tool launches a probe binary
  nothing builds, and the test `#include`s the deleted
  `tools/glm5_next_driver_probe.c`, so neither can run. Delete both, with
  `docs/GLM_REAL_DRIVER_PROBE.md`, the probe paragraph of
  `docs/DRIVER_ACCEPTANCE.md` and the `glm5_next_driver_probe` entry in
  `tools/test_serving_reliability_host.py`.
- Hy4 runs on the common runner (`modules/hy4_tp16_resident_decode_stage`,
  `model-families/hy4/source/hy4_stage.cuh`) and serves as `hy4-preview`, but
  it is not checked against a reference. The one teacher-forced run of
  `tools/hy4_reference.py` (2026-10-11, before the epsilon fix `c723234a3` and
  the round-to-nearest-even mesh sum) agreed on 426 of 433 served tokens, with
  six hard disagreements beyond near ties (story, translate, science,
  long_recall), and its report is not in the tree. Rerun the reference on the
  current build, explain or fix every hard disagreement, and pin the passing
  report under `qualification/hy4/`.

## Fleet and queue tooling

Found while verifying [`docs/FLEET_RELEASE_RUNBOOK.md`](docs/FLEET_RELEASE_RUNBOOK.md)
and the multidev docs on 2026-09-28. Line numbers are in
`tools/fleet_node_agent.sh` unless another file is named.

- The lane pin reaches residentd only as the environment variable
  `SPARK_WEIGHTD_LANE`, which the `lane.sh` rendered by `tools/stage_lane.sh`
  sets and `inference/runner/stage_serving_adapter.c` and
  `ring/transport/tp_device_collective.c` read with `getenv`. Move it into the
  deployment contract beside `wait_mode`. `sparkpipe-hub-route.service` stays
  a runtime step, but only `docs/FLEET_RELEASE_RUNBOOK.md` (section 1)
  describes it: `docs/INCIDENT_RECOVERY_PLAYBOOK.md` does not list it and
  `tools/fleet_post_reboot.sh` does not check the route.
- `tools/devcycle/lane_assignments.json` and `lane_budget_calc.py` still
  assume 8 lanes, and `lane_assignments.json` gives lane 0 to a GLM development
  lane while production GLM runs on lane 6. Replace both with the lane
  allocator of lane B04 (TD287/TD288).
- The Ceph canary (`tools/hub/ceph_canary.py`) only records problems on the
  hub. They reach a person only while a session runs its `watch` command
  from the workstation. Nothing sends mail or a push on its own, and the
  hub's own outage shows only as the watcher's stale-state exit.

## Hardware independence

- The device API (`include/sparkpipe/spark_device.h`, `runtime/device/`)
  has buffers in the five spaces, register and map_file, queues, events and
  kernel launch, with host, CUDA and Metal backends built as one shared
  library each. Only the contract tests and the Metal stage runner use it.
  Production modules call the CUDA runtime directly and own their streams,
  graphs and allocations; move them behind the device layer so a module
  target can be `host.*`, `metal.*` or `rocm.*`. The CUDA backend then needs
  kernel launch (it refuses kernel libraries today) and graph capture and
  replay.
- The stage-runner model interface
  (`include/sparkpipe/spark_stage_runner_model.h`) still passes queues and
  device addresses type-erased: `void *stream` in `step`, `recurrent_copy`,
  `reset_slot` and `sideband`, and raw pointers in `SparkStageRunnerStep` and
  `SparkStageRunnerKv`. The Metal model reads them as a `SparkDeviceQueue *`
  and device addresses. Convert them to `SparkDeviceQueue *` and
  `SparkDeviceAddress` in one ABI change with every consumer.
- The Metal kernels (`inference/kernels/metal/`) implement the operations of
  the CUDA kernels (`inference/kernels/`) a second time, in Metal. The
  formulas they share are moving into `inference/kernels/kernel_math.cuh`
  ([`docs/METAL_BACKEND.md`](docs/METAL_BACKEND.md#shared-kernel-math)):
  bf16 conversion, sigmoid and SiLU (29 CUDA copies), softplus with the
  gated-delta decay, the tanh GeLU and the softcap are there. Still a copy
  per kernel or per backend:
  - RMSNorm scales, rotary angles and rotation, the delta-rule cell update,
    online-softmax merges and the log-probability reduction;
  - copies whose expression differs from the shared one, each a numerical
    change that needs the affected model's gate: the experts' tanh GeLU in
    `moe_local.cuh` computes `0.044715 * g * g * g`, while the shared one
    (and PyTorch) cube first, so Gemma 4's dense and expert FFNs round
    differently; the head-gate softplus uses `log1pf(__expf(x))`, the
    gated delta `__logf(1 + __expf(x))`; the causal convolution's swish is
    `x * sigmoid(x)` on CUDA and `x / (1 + exp(-x))` on Metal;
  - exp flavor: CUDA kernels use `__expf` in some places and `expf` in
    others, Metal the precise functions everywhere, so the backends differ
    by the intrinsic's error until one policy is chosen and measured;
  - the bf16 conversion carries CUDA's semantics on both backends: a NaN
    whose payload sits only in the low 16 bits rounds to infinity. The
    hardware conversion (`cvt.rn.bf16.f32`) keeps NaNs and is one
    instruction; switching changes every kernel's machine code, so it needs
    a fleet speed check.
  Kernel bodies (the schedules) stay per backend.
- Add rank, island and link-class descriptors and derive collective
  decomposition from the link matrix instead of a configured backend.
- Keep the host backend as a CI oracle for common policy, not only for
  kernels. The Metal backend is qualified on one M5 Ultra for the device
  contract, every kernel Qwen3.8-27B uses, and that model end to end
  ([`docs/METAL_BACKEND.md`](docs/METAL_BACKEND.md)); no other model and no
  multi-Studio topology is.
- Left out on purpose (2026-10-02): KV and weights share weightd but not one
  budget. weightd maps KV device memory as KV pools that grow and shrink in
  chunks (`runtime/spark_weightd_kv_pool.c`, driven from
  `runtime/stage_kv_binding.c`) and weights through leases. The weight budget
  is `device_bytes_max` minus the static `--kv-reserve-bytes` carve-out
  (`node/weightd.c`, `runtime/spark_weightd.c`), so KV pressure cannot reclaim
  idle weight memory and the reverse. KV pages still enter the arena through
  `SparkKvCacheArenaMarkBlockResident` (`cache/kv_cache.c`), and the device
  API (`include/sparkpipe/spark_device.h`) has no `make_resident`/`evict`,
  while `docs/INFERENCE_OS_DESIGN.md` requires one for both.
  Close it by building `make_resident`/`evict` in the device layer and moving
  KV pool sizing and weightd lease eviction onto it, proven on the fleet by a
  run where KV growth and expert demand both resolve through that one
  operation against one logged node budget.

## Mac Studio deployment

The design is in
[`docs/HARDWARE_TOPOLOGY.md`](docs/HARDWARE_TOPOLOGY.md#mac-studio-pool); the
Metal backend and the single-Studio stage are in
[`docs/METAL_BACKEND.md`](docs/METAL_BACKEND.md).

- Serve the single-Studio stage. `inference/runner/device_stage_runner.c`
  runs Qwen3.8-27B on one M5 Ultra through the stage-runner model interface,
  but no serving process hosts it: `model_residentd` and the stage serving
  adapter are CUDA builds. A served Studio replica still needs:
  - a device-backend serving adapter that drives the device runner from the
    batch engine, with admission, continuous batching, cancellation and
    release in common code (I02, I18-I20);
  - JIT and prefix KV reuse (I23-I27): the runner holds one identity page
    table over a fixed pool and no page store, so every request recomputes
    its prompt. Its pages must come from the common page cache with
    copy-on-write prefixes, and cached request state from the snapshot tiers
    under the 24-hour retention rule;
  - state save and restore: the model's `recurrent_copy` packs its recurrent
    state in the CUDA model's order, but no runner path exports or restores a
    sequence's KV pages and state, and decode after a restore has not been
    compared with uninterrupted decode;
  - generation policy: EOS from deployment metadata (I49), stop tokens and
    sampling beyond greedy;
  - weight residency: the runner maps the pack in its own process; weightd,
    the arena that survives a code-only redeploy, is CUDA VMM only;
  - a lane record, model description and swap-controller entry for the
    Studio target (`metal.apple10`).
- Map the numerics gate's resolution at BF16. The gate now judges the Studio
  stage against the reference's own reduction-order floor (mean KL at most
  2 times, top-1 disagreement at most 1.5 times, a paired block test at 3
  standard errors, both reference variants), and it passes Metal while
  failing two controls by wide margins: two output rows of `ffn_down`
  swapped in all 64 layers (4.4 times the floor's KL, paired test at 95
  standard errors) and a whole tensor offset by one row (15 times, 188).
  What it cannot see: the same two-row swap in one layer only
  (`layers.31.ffn_down`; 0.79 and 1.00 times the floor's KL, paired test at
  -4.8 and +0.2 standard errors), a fault whose effect on the output is
  smaller than BF16 rounding noise. Between those two sizes the threshold is
  not measured; controls at 2, 8 and 32 layers, and more positions to shrink
  the block standard error, would place it
  ([`qualification/metal/qwen38_27b/`](qualification/metal/qwen38_27b/)).
- Batch past the per-sequence state on a Studio. Decode reads the weights
  once per step for up to 64 rows, but per-sequence work then dominates: at
  B64 the gated delta rule reads and writes 3.1 MB of FP32 state per layer
  and sequence (about 27 ms of a 145 ms step) and decode attention reads each
  KV head once per query head (about 14 ms at 1,100 positions). A delta-rule
  kernel near the state's streaming bound and a decode attention that serves
  a KV head's query heads from one read raise B32 and B64; above 64 rows the
  projections fall to the prefill GEMM, which reads the weights once per 128
  rows. Prefill and decode mixed in one step are not measured (I18, I19,
  I21), and speculative verify frames have no cost curve yet.
- Make long verify frames cheaper. A verify frame must compute its rows
  exactly as one-row decode does, so it runs the GEMV in passes of at most
  eight rows: on one M5 Ultra 1 to 4 rows cost the same 49-50 ms as a decode
  step, 8 rows 63 ms, and every further 8 rows another pass (16 rows 122 ms,
  64 rows 474 ms). A 16-row GEMV pass spills registers (over 1 s). A GEMV
  that holds more rows per pass without changing each row's order of sums
  would let whole-file rewrites run frames of 16 to 32 drafts. Verify frames
  over several sequences in one step, and speculation through a served
  request (the serving adapter is CUDA), are not built.
- Close the B1 gap to the streaming roofline (90.4%, 20.12 of 22.27 tok/s).
  The projections stream at about 1,100 GB/s; the rest of the 49.8 ms step
  is about 710 dependent dispatches with a floor of about 5 us each, the
  residual RMSNorm (14 us a call, 128 calls; it loads 2 bytes per thread per
  step), the two-kernel delta rule (33 us) and the convolution (9 us). The
  levers are fewer dispatches: the norm in the next projection's prologue,
  convolution, preparation and delta rule as one kernel, rotary embedding
  with the KV store, and the gate projection in the same launch as the mix
  projection (it has 96 outputs and runs at 105 GB/s on its own).
- The kernel library builds with Metal 4.0 and its prefill GEMM uses the
  Metal 4 tensor operations, so it loads only on Apple family 10 GPUs (M5).
  Another Studio generation needs its own qualified kernel set.
- Only Qwen3.8-27B has a device model. Each further model is a shape, a
  tensor list and a layer schedule over the device stage and layer kinds
  (`inference/runner/device_stage_model.c`, `device_layers.c`), plus a TP1
  pack and its own gates. The layer kinds cover gated delta, attention with a
  sigmoid output gate, the SwiGLU FFN and residual RMSNorm; ungated and
  sliding-window attention, sandwich norms, GeGLU and experts (with expert
  residency on unified memory) are still to add, each by the first model that
  needs it. Order: Gemma 4 26B A4B, Gemma 4 31B, Muse Glimmer 30B, MiMo V2.6
  Flash, Qwen 3.8 Flash Next, each merged before the next starts.

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

One model holds a fleet at a time, and loading another takes about a
minute (79 s for Kimi K3). Several models sharing a fleet's Sparks need the
items below.

- Lane-priority over-subscription policy: the explicit `EVICT` request
  enforces lane order (`SparkWeightdServerOnEvict` in
  `runtime/spark_weightd.c` denies a requester lane that is not below the
  target lane), but only `tools/multi_dev_smoke.c` sends it. The eviction an
  acquire triggers (`SparkWeightdServerReclaimCold`,
  `SparkWeightdServerEvictColdGroups`) takes the least recently used arena and
  group with no lane awareness. Required rule: lane X may swap in only by
  evicting weights whose owning lane is greater than X; when the evictable set
  belongs to a lane currently executing, wait for its batch boundary, then
  evict; when nothing greater than X is resident, the acquire errors rather
  than thrashes.
- Load bandwidth fairness is enforced but not yet configured or measured.
  weightd paces pack streams, spine preloads and expert lease loads with one
  token bucket (`runtime/spark_weightd_pacer.c`) while a connection of
  another process has had its lane active within the last second (an engine's
  own transport connection never paces its own expert loads), at
  `--load-pace-bytes-per-second` (`SPARK_WEIGHTD_LOAD_PACE_BYTES_PER_SECOND`).
  Without the setting such a load is refused, so the fleet agent's
  environment must carry the value before a second model loads beside a
  serving one. Close it with a fleet run that loads a pack beside a serving
  GLM Full lane and shows the serving lane's B1 step time unchanged and the
  load's bytes per second at the cap.
- weightd serves one queued request at a time on its worker. Mesh activity,
  writes and status run on the main loop, so a pinned serving engine keeps
  running while another engine's pack loads; but expert lease acquires and
  releases and KV pool status and resize calls of every other engine wait
  behind the whole load (and the pacer lengthens that wait). Required: loads
  run off the request path, with the arena table locked against them, so
  lease and KV pool requests are answered during a load. Prove it with a host
  test where a lease acquire on one arena completes while another connection's
  attach is still streaming.
- Output chunking for GPU fairness: decode chains hold the GPU for whole
  tokens and cross-model sharing relies on driver time-slicing (no MPS).
  Required: bounded output quanta analogous to prefill chunks so a lane
  cannot lock the GPU for many tokens.
- Pause and resume of output batches: token-boundary resume is structural
  (sequence positions, prefix cache, the NVMe snapshot tier in
  `cache/kv_snapshot.c`, reply checkpoints marked by
  `SparkPrefixCacheMarkResume` in `runtime/model_batch_engine.c`), but there
  is no pause/resume API. A paused request must be sent again and reuses only
  what the prefix cache kept, and active-KV protection is transaction-scoped
  (`SparkKvPageCacheBeginPinnedLaneTransaction`, `cache/kv_page_cache.c`)
  rather than batch-scoped. Required: resume without recomputation, with
  boundary-flushed KV protected from eviction until the batch completes.
- Same-model multi-engine ran end-to-end once, on the deleted glm5_next
  module: on 2026-10-03 eight GLM-5.3 Flash TP16 engines on lanes 1-5 and 7-9
  shared one pinned-expert arena per node and gave token-identical outputs. It
  has not run on the common stage runner (`inference/runner/stage_runner.cu`),
  and the MTP-debug use case has never been exercised.
- Device allocation budgets are incomplete. GB10 `MemoryMax` does not
  contain every CUDA allocation, and the driver ledger omits some direct
  allocations and graph and context overhead. Enforce complete budgets
  before co-resident drivers rely on them.
