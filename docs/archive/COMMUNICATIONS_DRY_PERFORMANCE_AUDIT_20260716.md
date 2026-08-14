# Communications DRY And Performance Audit

## Scope

This audit covers hidden transport, resident IPC, rank work control, final-event
return, memlink helpers, and release selection. It distinguishes correctness
defects fixed in this change from larger performance work that still requires a
measured implementation.

## Fixed Findings

### Completion overwrite

Both TCP CUDA and verbs RDMA stored only one completion plus a ready bit. A
second completed packet could replace the first before the event loop polled it.
The persistent-ring simulator already contained a FIFO, but implemented it
privately.

All three transports now use `SparkHiddenTransportCompletionQueue`, a fixed
symbolic 1,024-entry FIFO. FIFO ordering and fragmented resident IPC have direct
unit tests.

### False native batching

TCP CUDA and verbs RDMA advertised
`SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION`, but their batch callbacks only
called the scalar function in a loop. In RDMA that also repeated the synchronous
CQ wait for every packet.

Those transports no longer advertise batching or publish batch callbacks.
`SparkHiddenTransportSendBatch` and `SparkHiddenTransportPostReceiveBatch`
fail closed unless the selected backend advertises and implements one actual
batch operation.

### Resident IPC fragmentation and client aliasing

The resident daemon used one shared control payload for four clients and read a
whole header/payload in one call path. A fragmented nonblocking read lost its
offset; a blocking accepted socket could stop the event loop.

The common resident IPC module now owns an incremental reader. Each client has
independent header state, payload offset, and payload storage. Accepted sockets
are nonblocking, and partial messages resume without restarting the frame.

### Release host-selector regression guard

The example PP13 release is now tested to contain exactly the twelve non-gateway
rank-daemon hosts. This prevents duplicate or missing host entries from becoming
a generated release.

## Remaining DRY Debt

| Duplicate family | Current copies | Risk |
| --- | --- | --- |
| Socket setup and exact I/O | TCP transport, RDMA control, service backend, rank daemon, resident daemon, memlink, gateway tools | Semantics already differ for blocking, `EAGAIN`, timeout, and partial-frame handling. Consolidate by protocol type, not one universal wrapper. |
| Packet-key matching | TCP hash key plus separate RDMA pending and remote predicates | A sideband field added to one predicate can strand packets in another. Define one symbolic packet key and hash it. |
| Fixed exact file I/O | node-context JIT KV, generic KV prefetch backend, B12x plan loader | Error and short-read behavior can drift. Share one positioned exact-I/O primitive. |
| Lane partitioning ownership | Production RDMA depends on a helper named for the standalone memlink service | The algorithm is shared correctly, but its ownership name obscures that it is a transport primitive. Move it to a neutral striping module if memlink is retired. |
| Control request/response loops | gateway backend and rank daemon each serialize resident submit and await | Duplicated protocol state makes asynchronous conversion harder and preserves redundant waits. |

The audit did not find a production wire path that hardcodes the old 128-byte or
8 KiB payload sizes. Wire and sideband byte counts are derived from `sizeof`,
`offsetof`, model constants, and descriptor fields. Operational capacities such
as 64 receive slots remain symbolic macros.

## Current Performance Footguns

1. RDMA writes are all signaled and the sender busy-polls CQs until every write
   completes.
2. The builder owns one output boundary buffer, preventing safe network/compute
   overlap.
3. RDMA pending and remote receive lookup is linear over 64 slots per packet.
4. Resident submit acknowledgements synchronously stop gateway/rank control
   progress.
5. JIT KV uses O_DIRECT but synchronizes the CUDA I/O stream and then performs
   blocking `pread`/`pwrite`.
6. The resident IPC write path can wait up to five seconds for a stalled local
   client; an owned nonblocking output queue is still needed.
7. Hidden and sideband payloads can require separate RDMA writes.
8. The standalone memlink object service allocates and copies per transfer and
   is not an inference transport.

## Measurement Requirements

No speedup is claimed by this audit. Required receipts are:

1. SM121 compile for both CUDA transports and the resident archive.
2. Burst test proving every submitted completion is returned once and in order.
3. Fragmented multi-client resident IPC test against the live daemon.
4. Hidden-only and sideband ring laps for event and busy-poll progress.
5. End-to-end FP8 inference with stage timings before and after each future
   transport optimization.
