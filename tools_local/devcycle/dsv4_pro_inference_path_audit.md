# DSV4 Pro TP4xPP4 inference path audit

Code audit of the complete inference path (request intake through token
emission), with focus on the communication layers. Method: source reading of
the batch engine, pipeline client, serving adapter, decode-stage module,
TP device collective, and the host-RDMA verbs transport. No ring access was
used; latency figures are model-derived estimates, not measurements.

## 1. Path map

1. **Intake** — sparkpipe_model_batch (spark0) connects by TCP to all 16
   ranks' control endpoints (mgmt network, port 20480). Each submission is
   Prepare/Continue'd to ALL 16 ranks (runtime/model_pipeline_client.c
   SparkModelPipelineClientSubmit). The tool loop is: EngineProgress (drains
   up to 8 messages per rank, dispatches up to
   maximum_new_submissions_per_progress new submissions) -> poll() on all 16
   sockets -> repeat. Event-driven, no sleep.
2. **Adapter (per rank)** — SparkDsv4ServingSubmit: validate -> reserve one
   of 4 pending slots -> driver admission (KV pages) -> program submit.
   Slots limit concurrency to max_inflight_submissions = 4.
3. **Module (per rank)** — one CUDA stream per slot. Per layer:
   HcEnter -> attn norm -> attention -> hidden all-reduce (attn side,
   overlapped with the next WQ_B weight read-ahead) -> FFN -> hidden
   all-reduce (FFN side) -> next layer. Final stage adds final norm ->
   vocab-sharded screened argmax head -> U64 max cross-rank reduce -> token.
   PREFILL frames run attention in round-major waves
   (SparkDsv4ModuleRunCausalAttention): for a single-lane prompt each wave is
   1 row, so prefill attention degrades to per-row launches while MoE/dense
   GEMMs and the collectives still batch over the whole frame.
4. **TP collective** — SparkTpDeviceCollective: created once at module init;
   routes (per-step hidden-transport edges) opened once, receive credits
   pre-registered once (persistent-credit handshake, 120 s window). A
   dedicated progress thread busy-spins: poll transport completions ->
   advance each credit's phase machine (reserve -> send -> transfer ->
   consume -> publish -> release). Algorithm by payload: <= 81920 B direct
   all-to-all (3 peer sends per rank at TP4); >= 655360 B counter-rotating
   split ring (3 steps: step 0 on the paired 200G rail, steps 1-2 on the
   100G all-to-all rail); otherwise recursive doubling. 8 credits
   (4 slots x 2/slot) of concurrency.
5. **Boundaries (PP)** — 2 hidden-transport edges per rank (previous/next
   stage), opened eagerly at boot with 120 s timeouts (observed solo:
   rank7_to_rank11_hidden receiver wait). Per token the mHC mixed stream
   crosses each boundary once (3 hops at PP4).
6. **Completions** — stream-ordered driver completion -> adapter -> TCP
   completion message to the batch tool; token events stream to stdout.

## 2. Communications inventory (per decode token)

| Traffic | Count | Payload | Network | Notes |
| --- | --- | --- | --- | --- |
| TP hidden all-reduce | 122 (61 layers x 2) | 14 KB (1 row x 7168 BF16) | TP4 group, RDMA | direct all-to-all at this size; zero-copy on GB10: credit buffers are cudaHostAlloc(Mapped) in unified LPDDR5X, verbs NIC reads the same memory |
| Head cross-rank U64 max | 1 | 8 B | TP4 group (stage 3) | vocab-sharded argmax |
| PP boundary hop | 3 | mHC mixed stream (<= 4 x 7168 BF16) | adjacent stages, RDMA | 2 edges per rank |
| Control prepare | 16 | submission struct | TCP mgmt | fan-out to all ranks |
| Control completion | 16 | completion struct | TCP mgmt | fan-in from all ranks |
| Engine progress tick | ~1-2 | - | TCP mgmt | poll-wakeup + drain |

## 3. Footguns

### F1 (HIGH, correctness risk, first-time-on-ring): CUDA-graph TP prewarm
The TP4xPP4 adapter REQUIRES cuda_graph_count_by_pp_stage =
[49,46,46,46] (3L+1 islands; the adapter rejects any other value and the
module validates the same). SparkDsv4ModulePrewarmTpGraphs therefore
captures ~185 graph islands per slot x 4 slots WITH the TP collectives and
the host progress thread on the first ring boot. The stream-ordered
completion design is meant to support capture, but this path has never run
(hardware validation used tp=1 + graphs=0; solo boots blocked before module
init). If capture hangs, the whole ring stalls at boot. There is NO config
escape hatch: the graph counts cannot be zeroed without code changes.
Mitigation: watch /tmp/fleet-swap-dsv4-pro-*.log during the boot window for
the graph-prewarm progress lines; treat "silent 120 s" as this failure.

### F2 (HIGH, operational): fail-stop collective + coarse deadlines
Any collective failure latches the whole collective (fail-stop, no per-step
recovery), and each operation's deadline is operation_timeout_milli =
120000 ms. A lost credit, a hung QP, or one stalled peer therefore kills
the entire 16-rank decode and takes up to 2 minutes to surface per
operation. Mitigation: enable tp_collective profiling on ring day
(SPARK_TP_DEVICE_COLLECTIVE_PROFILE=1 if supported) and keep the fleet swap
logs.

### F3 (MED): 120 s boot handshake window
Boundary edges and the collective credit handshakes all open within a
single 120 s window per rank. fleet_swap starts ranks sequentially
(~1-2 s/rank) and pack loads (~60 GB reads) happen after, so the window
should hold, but any slow NVMe or ssh hiccup during start could strand a
rank mid-handshake. The runbook's rank-liveness check catches the result
but not the race.

### F4 (MED, fixed): 1-row prefill submissions
The inherited flash batch used max_prefill_rows_per_submission = 1, so the
128-token prompt ran as 128 sequential submissions: 15,616 latency-bound
collectives plus 128 control round trips, with no GEMM batching across
prefill rows (expert weight reads are NOT amortized). The module supports
multi-row prefill frames (wave-chunked attention), so a 128-row single
submission amortizes MoE weights ~128x and cuts prefill collectives to
122. FIXED: tools/devcycle/batches/dsv4pro_o128_128row_batch.json (same
prompt, same budget - the token stream is unchanged; the first run pins the
baseline either way), shipped to spark0:/tmp/dsv4pro-o128-128row-batch.json
and wired into the runbook.

### F5 (LOW-MED): decode dispatch serialization
Autoregressive decode is 1 token per submission by construction; each token
pays 32 control-plane TCP messages plus the engine's progress tick before
the next token dispatches. Estimated ~0.5-2 ms/token of control latency in
the critical path. Acceptable at the expected 30-60 tok/s, but worth noting
if the ring shows an unexpectedly low ceiling.

### F6 (LOW): KV backing claim vs disk
kv_backing_maximum_bytes = 4 TiB on /home; sparke has only ~740 GB free.
The page store's anonymous+direct-io writeback path could in principle
grow backing files; the O128 test's actual KV footprint is tiny. Monitor
sparke's disk during the run.

### F7 (LOW): progress-thread busy-spin
SparkTpDeviceCollectiveProgressMain loops with no sleep, burning one CPU
core per rank at 100%. Latency-optimal (no wakeup latency) but consider
core pinning/affinity on ring day to keep it off the kernel-dispatch cores.

## 4. Performance model (per decode token, estimates)

| Component | Estimate | Basis |
| --- | --- | --- |
| Expert weight streaming | 16-40 ms | top-6 experts ~132 MB/layer x 61 layers ~ 8 GB/token from LPDDR5X at a few hundred GB/s effective; dominates |
| Dense + shared weights | 2-6 ms | per-layer wq/wkv/wo/shared/gate streams, ~2-4 GB/token |
| TP collectives | 2-6 ms | 122 x ~15-50 us each (RDMA wire + progress thread + consume kernel; zero-copy on GB10 unified memory) |
| PP boundaries | 0.1-0.3 ms | 3 hops, each one RDMA transfer + progress |
| Control plane | 0.5-2 ms | 32 TCP messages + engine tick per token |
| **Expected decode rate** | **~25-60 tok/s** | consistent with flash TP4 control at 40 tok/s |

Bottleneck ordering: expert-weight DRAM streaming >> TP collectives ~
control plane > boundaries. Prefill with the 128-row batch amortizes the
first three components across all 128 rows (single traversal); decode
cannot batch and is weight-bound.

## 5. Recommendations

1. **DONE** - 128-row prefill batch (F4).
2. **DONE** - runbook now passes --profile-stages, so the ring-day receipt
   captures per-stage queue/service/elapsed ns on stderr for every
   submission - the first real per-stage latency data.
3. Ring day: keep the fleet-swap logs; if boot stalls at graph prewarm
   (F1), the workaround is code-level (disable prewarm), not config-level -
   the graphs are mandatory for the TP path.
4. Post-baseline follow-ups: (a) GPUDIRECT transport variant exists
   (libhidden_transport_spark_gpudirect_rdma_verbs.so) but is unlikely to
   help on GB10's unified memory - the mapped-host mode is already
   zero-copy; (b) raise maximum_messages_per_rank_per_progress to reduce
   decode dispatch serialization; (c) SPARK_TP_DEVICE_COLLECTIVE_PROFILE
   for per-collective latency histograms; (d) consider 2-token decode
   chains (the module supports chain_step_count > 1) to halve the
   per-token control/collective overhead.
