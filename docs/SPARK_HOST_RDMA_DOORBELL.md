# Spark Host RDMA Doorbell Path

The `hidden_transport_spark_host_rdma_verbs` module moves pipeline boundary
payloads between CUDA-mapped host allocations through ConnectX RC queue pairs.
It does not require `nvidia_peermem`, and it does not claim that separate Sparks
share physical GPU memory. The useful contract is narrower: the producing CUDA
kernel writes the registered mapped allocation, the NIC reads that allocation,
and the receiving CUDA kernel consumes the destination allocation without a CPU
payload copy or a CUDA device memcpy.

## Small-frame path

Packets at or below 256 KiB use a single RC queue pair per packet,
selected as `receive_index % lane_count` from the receive slot the
receiver advertises. The receiver posts a zero-length receive work
request on that lane and advertises a persistent registered boundary
region. The sender performs an ordered RDMA write, using
`IBV_WR_RDMA_WRITE_WITH_IMM` for the final region. The immediate value
identifies the pending receive slot, and both endpoints derive the same
lane from it without any extra control traffic. A completion-channel
file descriptor wakes the normal SparkPipe transport event loop; no TCP
transfer-complete message is sent.

One queue pair per packet is intentional for B1 and other small frames.
Striping a roughly 12 KiB hidden vector over eight queue pairs costs
more work requests and completions while losing the ordering property
needed for a single doorbell. Spreading whole doorbell packets across
lanes by receive slot keeps that per-packet ordering while letting
independent small frames use the full QP set instead of queueing
behind lane 0.

The transport returns `OK` once the send work request owns the packet and emits
one completion only after every posted write completes. The resident route keeps
its mapped output slot pinned between those events, so the next model invocation
cannot overwrite bytes still being read by the NIC. `BUSY` means no send was
accepted and the same packet must be retried.

The module does not advertise native batch submission. Its former batch
callbacks only looped over scalar sends, and every scalar send waits for local
completion. The generic transport API therefore performs the same scalar
fallback without misreporting a performance capability.

Rank daemons accept `--transport-busy-poll` for latency-critical RDMA service.
It keeps transport and resident progress on-core instead of entering `ppoll`
after an idle iteration. The option fails closed for non-RDMA transports and
consumes one CPU core per rank while enabled. On the 13-Spark B1 micro-ring,
busy progress reduced the end-to-end loop from 6.888 ms to 0.506 ms while still
verifying the final payload on every lap.

## Large-frame path

Packets above the threshold retain the existing multi-lane striped RDMA path and
TCP completion message. This preserves large-batch bandwidth while the small
decode path removes the expensive per-lane and completion-message overhead.

The compiled defaults are eight striped lanes, control-port base `55700`, IB
port `1`, GID index `0`, and a `262144`-byte doorbell threshold. Set
`SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_LANES`,
`SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_CONTROL_PORT_BASE`,
`SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_IB_PORT`,
`SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_GID_INDEX`, or
`SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_DOORBELL_MAX_BYTES` to override them.
Present values are parsed strictly: malformed or out-of-range configuration
fails initialization rather than selecting a different setting. A doorbell
threshold of `0` explicitly disables the doorbell path.

For the PP13 linear ring, the module selects the RoCE device per edge: `f0`
faces the previous rank and `f1` faces the next rank. Non-adjacent routes fail
closed unless `SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_IB_DEVICE` explicitly names the
device. This matters because each rank's input and output sessions use different
ConnectX ports.

## Registration lifetime

Mapped boundary memory regions are registered once and cached; the cache
evicts the least-recently-used registration when full rather than
failing closed. A registration is never evicted while in-flight work
references it: posted send work requests pin their source regions until
completion, and an advertised receive region stays pinned until the
receive completes, so a peer can never write to a deregistered rkey.
CUDA-visible pointers passed to the transport must therefore remain
allocated and keep the same backing allocation while any transfer
referencing them is in flight. Exact pointer-and-size cache hits
intentionally bypass repeated CUDA pointer-attribute queries under that
lifetime contract. With the PP13 builder, each edge reuses fixed hidden
and sideband pointers, so steady-state packets hit the cache and no
eviction occurs.

When `SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_DEBUG=1`, transport destruction logs
doorbell sends, striped sends, pointer-attribute queries,
memory-registration count, MR-cache evictions, and MR-cache hits. The
only other accepted value is `0`.

## Hardware validation

The isolated 13-Spark ring produced these payload-verified results:

| Shape | Progress | Average full loop |
| --- | --- | ---: |
| 12 KiB B1 | event driven | 6.888 ms |
| 12 KiB B1 | busy progress | 0.506 ms |
| 12 MiB B1024 | busy progress, 8 lanes | 15.407 ms |

The 12 MiB result is about 1.185 ms per physical link. A 100 Gbit/s link needs
1.007 ms to serialize 12 MiB, so the full-ring result is roughly 85 percent of
the wire serialization limit. Every run verifies the returned payload exactly;
the large run registered one MR per direction and reused it for every lap.

## Deployment boundary

The generated FP8 production manifest selects this module. A release is still
not accepted until a zero-drift two-Spark payload-parity and latency gate and a
complete 13-rank correctness run pass for the exact built artifact.

The existing `sparkpipe_glm52_pp13_ring_check` accepts both `--transport` and
`--transport-module`. For an isolated hardware test, set a control port base not
used by production and select the adjacent RoCE device on each node:

```sh
export SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_CONTROL_PORT_BASE=57700
export SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_DOORBELL_MAX_BYTES=262144
export SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_IB_DEVICE=rocep1s0f1
build/sparkpipe_glm52_pp13_ring_check \
    --rank 0 --ranks 2 --prev spark1 --next spark1 --laps 1000 --active 1 \
    --sideband-bytes 1024 \
    --transport build/libhidden_transport_spark_host_rdma_verbs.so \
    --transport-module spark.hidden_transport.spark_host_pinned_rdma.verbs.v1
```

The peer uses rank 1 and its RoCE device for the same physical link. The checker
alternates connection order by rank so blocking RC setup cannot deadlock. It
allocates only the boundary payload and scratch space, so it can run beside the
model pipeline without loading another checkpoint.
