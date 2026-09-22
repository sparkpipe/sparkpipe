# CUDA stream memory waits and qualification

The isolated prototype has qualified the primitive described below. The shared
collective now also has an explicit `SPARK_TP_WAIT_MODE=hardware` path under
qualification; `spin` remains the default. Selecting hardware requires supported
64-bit stream operations and a valid CUDA alias for the registered mesh mapping.
An unsupported configuration fails rather than switching wait implementations.
The shared GPU numerical path has passed the bounded single-device fixture
below; no distributed model-throughput pass is claimed here.
The original local gate prototype compiled and ran on Spark0 with CUDA 13.0.88
for GB10 `sm_121a`. Source commit `6db0dd333ebb089336cd8d9362c5e6a6d36ff202`,
receipt `spark0:/tmp/sparkpipe-hw-wait-6db0dd33/run.log`, reported 4,258.53 ns per
91-node update replay (46.80 ns/node) and all three local GPU cases passed.
Source SHA256 was `307b5b21fea6cd4b022794e43eefd435ed4151a988120c4d270c32b9c68e4425`;
binary SHA256 was `e10ad2bc35c8881b2a3a39019529e37816c6379672a59417d8c300355a064dc3`.
The two existing weightd/resident processes and their measured GPU allocations
were unchanged after the run. This was a shared host timing sample, not an
isolated performance benchmark or GLM inference result.

The memfd/RDMA extension was executed at source commit
`cd344a64440d075552a7f4eb3e3e01c830ef4b7d`, using
`/tmp/sparkpipe-hw-rdma-cd344a64` on Spark0 and Spark1. The source SHA256 was
`e044fd958a4187a8b63284571ce48c1ed38e51d33c06322fefa10e0d88d5f25c` and wrapper
SHA256 was `1316b5443f05e61f1588b5d0b5bd7410113a41c3338c1c4cf948f8a346cfcb14`.
Receiver binary SHA256 was
`3064edc6f2e8759df9d88b896e8afee13bdc05a15ad7c8c9f65bb283b68b5cc5`;
sender binary SHA256 was
`a6048bc899ae09637c959aa60bf376faea2d4408ab26a220362f706e5a219776`.

Spark0 received into an 8,192-byte shared memfd mapping, with portable/mapped
CUDA registration and the actual device alias. Spark1 sent over the explicitly
selected RoCE v2 link. Eight repetitions of all three cases passed: 24 GPU
launches with 4,096-byte NIC-written payloads, stale/delayed readiness,
error-before-release cancellation, and recovery. Both processes exited 0;
combined elapsed time was 2.077 seconds. The receiver observed GPU entry before
its stale check and GPU completion before any CUDA call following NIC release.
The sender checked every signaled work completion before reusing source memory.

The 1,000 repetitions of 91 node-value updates averaged 4,249.49 ns per replay
(46.70 ns/node). The separate local memfd run passed all three GPU cases and
reported 4,078.29 ns per replay (44.82 ns/node). These are host update timings
on shared machines, not collective latency or serving throughput. Local receipt
bundle `/private/tmp/sparkpipe-pr1082-receipts/cd344a64-rdma` contains
`result.json`, `receiver.log` and `sender.log`; the local GPU receipt is
`spark0:/tmp/sparkpipe-hw-rdma-cd344a64/local-memfd.log`.

## Safe invocation

The wrapper and compiled binary default to help without CUDA or RDMA calls. Compilation
requires an explicit toolkit and target plus libibverbs headers/library. Any CUDA context creation requires
`--run`; the additional `--gpu-waits` flag permits three graph launches.

```sh
python3 tools/tp_stream_memop_probe.py
python3 tools/tp_stream_memop_probe.py --compile --cuda-root /usr/local/cuda --arch sm_121a
python3 tools/tp_stream_memop_probe.py --run
python3 tools/tp_stream_memop_probe.py --run --gpu-waits
```

Run only in an assigned isolated GPU window. `--run` creates a context, captures
91 wait nodes plus an entry marker and guarded consumer, instantiates the graph, and times 1,000
sets of 91 node-value updates without launching a graph. `--gpu-waits` also
checks stale/delayed readiness, cancellation before release, and recovery. A
30-second process alarm bounds the standalone experiment. The guard asserts
that cancellation never consumes the payload. The local mode involves no network or service.

## Two-host NIC visibility check

Both processes require `--run` and explicit device, port, GID, IPv4 address and
unique TCP port. The receiver binds that address; the sender connects to it.
`--iterations` repeats all three cases 1..128 times (default 8), within the same
30-second process alarm. The sender creates no CUDA context. Optional receiver `--memfd` creates a private
memfd with `MAP_SHARED`, registers it with CUDA portable/mapped flags, and uses
the actual alias returned by `cudaHostGetDevicePointer`. There is no allocation
fallback. Normal cleanup destroys the QP, deregisters the NIC MR, destroys the
terminal graph/stream, unregisters CUDA, then unmaps and closes the memfd.
Without `--memfd`, the receiver explicitly uses `cudaHostAllocMapped`. Each process owns
one RC QP, one 8-entry CQ and one MR; at most three WRs are outstanding. Every
WR is signaled and checked before source reuse. Normal teardown destroys the
QP before deregistering its MR; failed/partial posts terminate the isolated
probe rather than reusing memory. The OS releases only this process's resources
on the bounded alarm. No daemon, persistent config or existing QP is modified.

For every trial the receiver first observes a GPU-written entry marker, proving
the graph has started. The sender then RDMA-writes a stale generation. The receiver
requires the started graph to remain incomplete with untouched output after
that write. The sender then posts payload, error and current readiness in that
order on the same RC QP. The GPU consumes all 512 words and produces a checksum;
the receiver computes its expectation from the trial number, without reading
the received payload on CPU before GPU completion. After the sender releases
readiness, the receiver observes the GPU-written completion status before making
any further CUDA call; synchronization only confirms terminal cleanup afterward.
Cancellation must consume
nothing, and recovery must observe the subsequent NIC-written error clear.
The receiver never writes a successful readiness value or transport ACK into
the MR. TCP messages coordinate phases only; NIC writes release the graph.

Read-only sysfs inventory on 2026-09-22 found `rocep1s0f0`, port 1, GID index 3
active on both hosts, RoCE v2, IPv4-mapped GIDs `10.10.200.0` (Spark0) and
`10.10.200.1` (Spark1). Selection is explicit, with no interface/GID fallback;
active MTU below 4096 is rejected. An assigned run can use these commands after
checking that the chosen TCP port is unused:

```sh
python3 tools/tp_stream_memop_probe.py --run --rdma-receive --memfd --ib-device rocep1s0f0 --ib-port 1 --gid-index 3 --address 10.10.200.0 --tcp-port 49387 --iterations 8
python3 tools/tp_stream_memop_probe.py --run --rdma-send --ib-device rocep1s0f0 --ib-port 1 --gid-index 3 --address 10.10.200.0 --tcp-port 49387 --iterations 8
```

This tests the selected CUDA-mapped host allocation on one link. The executed
`--memfd` mode matches the production allocation/registration shape. The pass
does not qualify cross-process shared ownership, allreduce,
missing-peer recovery, or model throughput. No unsupported remote FLUSH is
requested; failure of ordering or visibility is a qualification failure.

Read-only device queries on Spark0 returned success for these CUDA 13 attributes:

| Attribute | ID | Value |
| --- | ---: | ---: |
| CAN_USE_64_BIT_STREAM_MEM_OPS | 122 | 1 |
| CAN_USE_STREAM_WAIT_VALUE_NOR | 123 | 1 |
| CAN_FLUSH_REMOTE_WRITES | 98 | 0 |
| CAN_MAP_HOST_MEMORY | 19 | 1 |
| HOST_REGISTER_SUPPORTED | 99 | 1 |

These describe the sampled device, not every deployment. The tool checks 64-bit
wait support and fails explicitly if absent. It obtains the mapped device alias
with `cudaHostGetDevicePointer` and does not request unsupported remote FLUSH.
CPU publication into this mapped gate cannot qualify RDMA payload visibility.

## Shared GPU numerical qualification

Source `52d2944262eb643d901df08f7662e5272fa7351b` passed all 70 cases of
`tools/tp_mesh_hardware_probe.cu` on Spark0 with CUDA 13.0.88, `sm_121a`,
`CUDA_MODULE_LOADING=LAZY`, `CUDA_MODULE_DATA_LOADING=LAZY` and
`CUDA_DEVICE_MAX_CONNECTIONS=32`. Receipt directory:
`spark0:/tmp/sparkpipe-hardware-52d29442/build/qualification/` contains
`gpu.log`, `receipt.json` and before/after GPU/process inventories. Binary SHA256:
`7793d1b87f296cc896b2974cb1f7a99870400a5a74b063bfdfbd89e201404c8b`.

The fixture runs the actual shared GPU arithmetic, hardware waits and daemon
gate checker, with private mapped bands and CPU copying between logical ranks.
It covers TP2/3/4/8/16, B1/B2, BF16 sum with FP32 partials, U64 maximum, rank-major
gather, payload splitting, eager execution and repeated graph replay. Scheduled
stale/future tails and stale ready cannot advance the waiting phase. Cancellation
and missing-peer timeout leave output untouched, drain subsequent graph work,
and permit recovery using the same executable after a new chain is seeded.
This fixture models multiple ranks on one GPU; it does not qualify NIC transfer
completion, multi-host skew or model inference by itself.

The original `277a33f6` binary failed the first TP2 eager peer wait under lazy
loading but passed all then-existing 69 cases with eager function/data loading.
Commit `147896e26ac7fd15e4b0a24c2978d56951d02352` preloads the seven hardware
kernels with `cudaFuncGetAttributes` during preparation. The 70-case explicit
LAZY rerun passed in 3.776 seconds. This moves CUDA's potentially synchronizing
first function load before dependent waits, following NVIDIA's
[lazy-loading guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html).
The historical comparison logs remain under
`spark0:/tmp/sparkpipe-hardware-277a33f6/build/qualification{,-eager}/`.

The same final fixture killed a negative control that changed exact peer-tag
equality to accept future tags: after 66 ordinary numerical cases it failed the
scheduled no-early-phase assertion. The isolated mutant binary SHA256 was
`c0dc79041bc13ef40149f4532abfb6767728c8b5670276b533028616bee40bfa`;
receipt directory is the final build's `qualification-negative/` sibling.

For the synthetic TP16 B1 graph of 91 rounds, six samples after two warmups
reported median 15.579 ms. Median maximum-rank counters per sample were source
wait 2.177 ms, peer wait 12.306 ms, copy 0.367 ms and combine 0.214 ms. These
separately selected maxima/medians are not an additive time breakdown. Transport
was CPU copy and daemon progress shared one host; these are neither RDMA
allreduce latency nor token-throughput measurements.

## Shared collective implementation

The current transport already pushes payloads and tails by RDMA WRITE into each
receiver's mapped local host memory. GPU `ld.global.cv` polling is not an RDMA
read from another host. Hardware mode removes SM-resident waiting while keeping
the existing arithmetic and transfer protocol.

1. ABI 6 provides one 128-byte `SparkWeightdMeshWaitRequest` per rank/band in
   the existing mapped mesh region. Its producer and consumer fields occupy
   separate 64-byte halves. Source-credit and received-peer waits use this same
   record and the acknowledged mesh activity interval.
2. A stream-ordered request kernel clears `ready`, writes the condition, exact
   collective tag, peer mask, expected cancellation value and timeout, performs
   a system fence, then publishes an increasing request ID. Eager execution
   uses `cuStreamWaitValue64`; graph capture inserts an explicit batch-memory-op
   node with dependencies from that request kernel. Both wait for `ready == 1`.
   A guard kernel follows the wait before any copy, publication or arithmetic.
3. The active weightd worker checks CQ-qualified SHIPPED or exact local peer
   tails, records the terminal request ID, then publishes error before ready.
   It handles each ID once and never rewrites a terminal record, even after a
   later cancellation. The next GPU request resets ready only after the prior
   consumer finishes on the same stream. IDs persist across graph replay and
   collective instances; exhaustion is an explicit error. The fixed wait value
   needs no graph-node updates and does not require retaining the captured graph.
   Failed graphs propagate their first error through later requests so they
   can drain without claiming peer arrival or forging SHIPPED.
4. B1 keeps its existing copy/publication/combine arithmetic. B2+ splits the
   fused tree at phase boundaries, reusing `SparkTpMeshTreeRoute` and
   `SparkTpMeshTreeLevels`. Preserve FP32 partials through reduction and broadcast
   and round BF16 once. Sparse phases, logical batch selection and execution-row
   chunking must remain identical.
5. Device control derives tags from the chain epoch and dynamic phase sequence,
   so replay preserves slot ownership without host parity readback. Source
   credit requires equality with the last published full tag; tag zero is the
   sole immediate initial-credit case. An explicit graph edge orders each
   request, memory wait and guarded consumer.
6. The daemon owns readiness publication and bounded active-request timeout.
   Cancellation changes the existing mapped cancellation value; it does not
   create a second writer to ready. If daemon progress is lost, a hardware wait
   has no independent SM timer. Host drain must report failure and retain
   ownership rather than free live resources. EndChain requires terminal GPU
   ownership, and CQ-qualified source transfers must independently drain.

Hardware timing counters report source-wait and peer-wait intervals from request
to guard, including daemon progress and GPU scheduling. Copy and combine counters
measure the corresponding kernel intervals. They are cumulative chain timings,
not isolated network measurements. The active daemon still polls current request
conditions; idle daemon work sleeps.

NVIDIA documents mapped-device-pointer requirements and CUDA-visible ordering
for [stream memory operations](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html).
The original prototype's [graph parameter updates](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html)
affect future launches and require the original node to remain in its graph;
the shared constant-value gate does not use that update mechanism.
CUDA 13 [capture inspection and dependency APIs](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__STREAM.html)
include edge data; the prototype uses those current signatures.

The separately compiled negative control in
`spark0:/tmp/sparkpipe-rdma-memfd-negative-20260922` changes the wait comparison
from equality to greater-or-equal and sets every replay wait threshold to zero.
The original 91-node update loop is unchanged. Its local
`--run --gpu-waits --memfd` run exited 1 at the expected line 478 assertion that
`cudaStreamQuery` must report not-ready after GPU entry. It failed by assertion,
not by timeout, demonstrating that the scheduled-stale oracle detects a bypassed
wait. Receipt: `negative-run.log`; source and binary hashes are recorded in
`build/qualification/SHA256SUMS` in that directory. This is an isolated source
mutation, not a runtime fallback or production build mode.

## Qualification gates

The local memfd checks, two-host NIC visibility checks, shared GPU numerical
fixture and their negative controls passed their expected outcomes. Distributed
serving still requires the actual daemon/NIC path under rank skew, missing peers,
timeout, cancellation, failed Begin/End and source-slot reuse, then numerical
output and whole-chain latency with identical model/configuration inputs.
The separate primitive and single-device passes do not establish distributed
serving reliability or a token-throughput benefit.
