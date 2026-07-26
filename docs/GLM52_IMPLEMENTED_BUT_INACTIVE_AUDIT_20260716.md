# GLM-5.2 Runtime Activation And Missing-Performance Audit

## Status Rules

Runtime status uses only these labels:

```text
ACTIVE_MEASURED
ACTIVE_UNMEASURED
BUILT_NOT_SELECTED
UNREACHABLE
NOT_BUILT
BROKEN
```

This audit separates source reachability, generated-release selection, and live
measurement. A feature is not working merely because its code compiles or its
flag appears in a manifest.

## Integrated Paths

| Path | Status | Current source behavior | Hardware receipt required |
| --- | --- | --- | --- |
| Multi-token bulk prefill | ACTIVE_UNMEASURED | Work ABI 10 carries row-major prompt tokens and ragged row counts. The service chunks up to 256 prompt tokens and the builder launches one embedding gather for the execution rows. | Prompt accuracy, chunk latency, and ring progress at 64 and 256 tokens. |
| NVMe JIT KV | ACTIVE_UNMEASURED | The release supplies a stable `.jit` path, 1,048,576 records, 128-record I/O batches, a separate nonblocking CUDA stream, and FP8-aware record ABI 3. | Forced eviction and reload with nonzero completion counters and no stale records. |
| DSA KV fragment path | ACTIVE_UNMEASURED | DSA selection now builds 32 logical blocks from 2,048 selected tokens and issues read-only L2 prefetches for resident blocks. | Long-context DSA accuracy and stage-clock comparison. This does not yet prove selected-only external-NVMe capacity. |
| FP8 KV | ACTIVE_UNMEASURED | MLA cache bytes and scales remain compressed in device memory and are dequantized directly by attention. Expanded BF16 key/value shadow caches are not allocated in FP8 mode. | Cache byte comparison and end-to-end token parity against the serialized FP8 reference. |
| Q/KV projection overlap | ACTIVE_UNMEASURED | Production plans retain the overlap capability and use the existing query/KV streams and events. | Device-clock proof that overlap beats the sequential path at equal output. |
| Persistent-doorbell RDMA | ACTIVE_UNMEASURED | The generated release selects the verbs transport. Sends use CUDA events and host callbacks instead of synchronizing the caller stream. | Hidden-only and sideband ring laps, then inference hop and stage timing. |
| Memlink multi-lane partitioning | ACTIVE_UNMEASURED | The RDMA transport uses the shared memlink partition helper for lane striping. The standalone RAM object service is not inserted into inference. | Per-lane byte balance and aggregate fabric bandwidth. |
| DSpark release path | ACTIVE_UNMEASURED | The generated FP8 release passes `--dspark` to the gateway and resident role and supplies the checkpoint path. | Checkpoint oracle, acceptance by depth, memory headroom, and B1/B2/B16 throughput. |
| Completion-safe hidden transport | ACTIVE_UNMEASURED | Persistent ring, TCP CUDA, and verbs RDMA share one symbolic FIFO completion queue. Production transports no longer advertise native batch submission while executing scalar loops. | Burst completion ordering on CUDA hardware and hidden-ring parity. |
| Fragment-safe resident IPC input | ACTIVE_UNMEASURED | Every resident client owns an incremental header/payload reader and payload buffer. Partial messages no longer block the event loop or lose byte offsets. | Multi-client fragmented-message smoke against a live resident. |

## Important Boundaries

The DSA integration is a resident read-only L2 prefetch. The current work-control
JIT path still loads context blocks before DSA selection, so selected-fragment
external-NVMe capacity is not claimed.

The memlink integration reuses its multi-lane partition contract inside RDMA.
Inference does not route hidden states through the standalone memlink daemon.

No compatibility fallback was added. Missing required capabilities, malformed
bulk token rows, invalid FP8 cache plans, and unavailable RDMA dependencies fail
closed. Batch API calls also fail closed unless an interface truthfully
advertises and implements native batching.

## Remaining Inactive Or Unproven Code

| Item | Status | Reason |
| --- | --- | --- |
| MTP | BUILT_NOT_SELECTED | MTP request, verifier, cache-transaction, and final-stage code exists, but the generated FP8 release does not pass `--mtp` to the gateway or residents. |
| TCP CUDA hidden transport | BUILT_NOT_SELECTED | Built as a staged debug/reference transport. The generated FP8 release selects verbs RDMA. |
| Standalone memlink RAM service | BUILT_NOT_SELECTED | Its lane partition helper is active inside RDMA, but inference does not route through the object-store daemon. |
| Persistent-ring transport | UNREACHABLE | Simulation and unit-test backend only; it carries the simulation capability and cannot satisfy production requirements. |
| Native hidden-transport batching | NOT_BUILT | The API exists, but TCP and verbs previously implemented it as scalar loops. Their capability bit and callbacks are now absent until one backend posts a real batch. |
| Selected-only external DSA KV fetch | NOT_BUILT | Resident selected-block prefetch exists, but the JIT path still restores context before DSA selection. External capacity therefore remains full-record/cohort based. |
| B512/B1024 | ACTIVE_UNMEASURED | Launchers exist, but this integration does not add a ring receipt. |
| Continuous release agents | ACTIVE_UNMEASURED | Release-manager code exists; live supervision must be verified during deployment. |
| Prefix-family reuse counters in public health | UNREACHABLE | Scheduler counters exist but are not yet exposed by the service health surface. |

## Missing Performance Work

| Priority | Enhancement not implemented | Why it matters |
| ---: | --- | --- |
| 1 | Multiple owned boundary buffers with asynchronous RDMA retirement | Verbs sends wait for local completion because one builder output buffer can be overwritten by the next stage invocation. A boundary-slot ring is required to overlap NIC reads with CUDA execution safely. |
| 2 | Native verbs batch posting | Wide dispatches should build WR chains, leave intermediate writes unsignaled, and signal one tail completion instead of entering the scalar send path per packet. |
| 3 | Asynchronous resident submit-result protocol | Gateway and rank control paths synchronously await `SUBMIT_RESULT`. Keyed asynchronous acknowledgements would let control-plane work continue up to resident capacity. |
| 4 | Hash-indexed RDMA receive matching | Pending and remote receive tables are fixed arrays of 64 entries and are linearly searched for every packet. The TCP transport already has a hash-indexed pending table. |
| 5 | Selected-only DSA external fetch | Run index selection from the compact index cache first, then load only selected MLA blocks from NVMe. This is the missing long-context capacity reduction. |
| 6 | Truly asynchronous O_DIRECT KV I/O | Current JIT KV batches synchronize the CUDA copy stream and issue blocking `pread`/`pwrite` calls. `io_uring` or an owned I/O worker ring is required for real compute/storage overlap. |
| 7 | Nonblocking resident IPC output queues | Input framing is now incremental, but outbound messages still use a bounded poll-and-write path. Per-client output queues would remove the remaining control-client stall surface. |
| 8 | Hidden and sideband write coalescing | Doorbell packets may issue separate hidden and sideband writes. A contiguous registered boundary layout can reduce work requests and completions for small frames. |
| 9 | Closed-loop width/depth scheduling | Current priority and cohort selection is source-reachable, but there is no measured cost model that continuously chooses batch width, pipeline depth, MTP depth, and bubble use from live stage and acceptance counters. |
| 10 | Direct GPU-memory RDMA qualification | Current production path uses CUDA-mapped pinned host memory. A true GPU-memory verbs path should only be added if GB10/ConnectX peer-memory support is proven faster and reliable. |

## Acceptance Order

1. Build the SM121 archive and resident artifacts from merged `main`.
2. Verify burst completion ordering and fragmented resident IPC.
3. Run hidden-only and 8 KiB sideband ring checks with the RDMA transport.
4. Run the matched real-prompt FP8 accuracy receipt with detailed dumps enabled.
5. Run 64-token and 256-token bulk-prefill receipts.
6. Force JIT eviction and reload and verify the NVMe counters and cache parity.
7. Exercise DSA above 2,048 candidates and compare stage clocks.
8. Compare Q/KV overlap device clocks and keep it only if it wins.
9. Qualify DSpark and MTP separately; never infer one from the other.
10. Record B1, B4, B16, B64, B256, and B1024 end-to-end measurements where memory permits.
