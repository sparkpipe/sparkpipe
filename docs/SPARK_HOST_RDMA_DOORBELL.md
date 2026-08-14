# Spark Host-RDMA Transport

SparkPipe moves pipeline activations and tensor-parallel collective payloads
through ConnectX RC queue pairs backed by fixed CUDA-mapped host arenas. The
path does not require `nvidia_peermem` and does not claim that separate Sparks
share physical GPU memory.

## Data path

At communicator construction:

1. allocate fixed page-locked mapped send and receive arenas;
2. register each arena with the owning NIC;
3. expose device mappings to the CUDA transport and collective kernels;
4. create rail-specific queue pairs and completion resources; and
5. pre-post the complete bounded receive window.

At steady state, the producer writes a registered mapped slot, the NIC reads or
writes that slot, and the consumer GPU reduces or consumes it without a CPU
payload copy or CUDA device memcpy.

Ordered immediate data identifies communicator, generation, sequence, phase,
stripe, chunk, and receive slot. A slot returns to the credit pool only after
send, receive, and GPU ownership are all released.

## Small payloads

Small messages use one ordered queue-pair lane per message. Whole messages are
distributed across lanes by receive slot so independent requests can progress
without striping one latency-sensitive payload into extra work requests and
completions.

## Large payloads

Large messages use persistent chunk pipelines with multiple registered slots.
Every logical ring overlaps receive `k`, GPU reduction `k-1`, transmit `k-2`,
and preparation of a later slot. Direct-forward, direct-reverse,
switched-forward, and switched-reverse progress have independent resources.

Queue locking, one global completion mutex, polling one rail to exhaustion,
whole-tensor barriers, or CPU-dispatched phase transitions violate the
transport contract.

## Failure behavior

Creation fails on missing interfaces, routes, queue pairs, registrations,
credits, or topology mismatch. Runtime failure identifies the exact rank, rail,
phase, stripe, and chunk. A failed communicator closes its transport resources
and cannot be reused under a newer generation.

There is no TCP, management-network, one-port, or serialized fallback in a
production package. Reference transports exist only in explicit test packages.

## Qualification

Qualification covers:

- simultaneous TX and RX on each physical port;
- simultaneous direct and switched traffic;
- interface-counter proof of pinned routing and expected byte balance;
- delayed, duplicate, stale, and missing completions;
- repeated slot reuse and bounded credit exhaustion;
- exact payload comparison for every supported datatype and tail shape; and
- latency, throughput, CPU progress cost, and eligible-work idle gaps.

Measurements live only in [`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md).
The collective algorithms built on this transport are specified in
[`PAIRED_DUAL_LINK_ALLREDUCE.md`](PAIRED_DUAL_LINK_ALLREDUCE.md).
