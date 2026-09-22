# Completion events and polling

Payloads and readiness counters already arrive through RDMA writes into each
receiver's local mapped host memory. `SparkWeightdClientAttachLazy` maps the local
mesh file, and `PrepareReceiveBf16` passes that mapping to CUDA. A GPU `.cv` load
of the readiness counter is not an RDMA READ operation. Replacing that repeated
local load with a pushed notification is a separate change from payload direction.
CPU polling, GPU wait kernels and network transfer time need separate measurements.

The API worker now sleeps on resident sockets and a nonblocking request-queue
pipe. Enqueue and cancellation wake that pipe; only the worker mutates the batch
engine. The batch CLI flushes events immediately after progress and checks
completion before sleeping. There is no fixed 5/10 ms polling cadence in either
consumer. Submission checkpoint persistence retains its 60-second deadline.

`SparkModelResidentClientNextProgressNs`, the pipeline aggregator and
`SparkModelBatchEngineNextProgressNs` expose absolute monotonic deadlines:
zero means no timer, one means immediately runnable. Reconnect, rejected-work
backoff, circuit delay and actual inflight timeout remain timed operations.
Readiness inspection never extends a deadline. Successful dispatch resets the
per-submission timeout and clears its rejected-work retry deadline.

Validation: real API text/token-ID and system-loopback tests pass. The added
API test waits for an idle worker, queues a request and asserts both exact fixture
tokens and a response within two seconds. Removing its enqueue wake fails the
test. Batch tests verify idle has no timer, new work is immediate, inflight work
has its real timeout, BUSY has an exact retry deadline, and completed work returns
to event-only waiting. These are host lifecycle tests, not GPU throughput evidence.

## Chain and stream completion

The common collective owns a mesh activity interval from acknowledged BEGIN until
its stream is terminal and END is acknowledged. Main and hidden-channel
collectives have independent owners. The weight daemon sleeps on a condition
variable only when there are no active owners, queued NIC transfers or published
GPU doorbells still waiting to ship. All three predicates are checked under the
wake lock; a producer ending before the first scan cannot lose its last transfer.
A disconnected active owner is an explicit orphan error, not proof of GPU drain.
This changes the weightd wire ABI to 5; clients and daemon must be rebuilt together.

The common CUDA receipt uses one host callback and a monotonic condition deadline
for an owned stream. Timeout retains the receipt until the callback has actually
arrived; destruction and rearm cannot recycle its context early. The callback
only signals host state and calls no CUDA API. GLM graph completion uses this
receipt. All seven GLM eager, failure and diagnostic waits use it as well;
there are no remaining GLM host query-poll loops. The MTP CUDA callback submits
to the existing completion worker, which performs CUDA work outside the callback.
Submit-or-park is protected by the existing queue lock to prevent lost work.

Mesh tests execute the actual daemon and client paths with mocked verbs, including
multiple owners, the final queued doorbell, pending NIC completion, socket loss
and lost-wakeup mutations. CUDA receipt tests cover independent streams, early
and delayed callback delivery, timeout retention and failed destruction. These
checks establish host ownership rules; they do not prove GPU or RDMA ordering.

## Device and transport boundary

The developer's PR1077 commit 07333264 reports idle engines at 96% GPU and 0%
memory utilization. Its success-time cancellation broadcasts are unscoped:
a faster rank can cancel another rank's valid final round. Cancellation must
preserve request ownership and drain before rearming shared cells. A same-stream
CUDA completion callback cannot prove that another stream or request has drained.
The fix preserves ownership through stream drain, cancels only failed work, and
latches cancellation generations in GPU waits. Sampled 96% GPU utilization means
kernels were active during most samples; it does not establish 96% SM occupancy
or prove that the wait kernel caused the full serving delay.

GB10 reports 64-bit stream memory operations, NOR waits and mapped host memory
support, but no remote-write flush capability. The read-only attribute probe
created no CUDA context and launched no GPU work. This does not qualify replacing
mapped-host wait kernels with hardware waits. Graph replay needs correct wait
values, explicit timeout/cancellation wakeups and payload visibility ordering.

CUDA documents [stream memory operations](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html).
RDMA completion notifications are [one-shot](https://man7.org/linux/man-pages/man3/ibv_req_notify_cq.3.html)
and require rearming and draining without a lost-wakeup race. A plain incoming
RDMA write does not produce the receiver completion event needed for that design;
GPU writes to a shared host doorbell do not themselves wake a CPU file descriptor.

Mesh registration is owned per exact mapping, shared by main and hidden-channel
collectives and released only after the final owner drains. Distinct mappings
register independently. Registration failure is explicit; it cannot silently
continue with pageable memory. Failed unregister retains a cleanup-only owner.

Active mesh intervals still poll GPU-produced doorbells and send completions.
B1 and B2+ device waits still use GPU polling kernels. Removing those loops requires
a completion gate separate from real peer counters, error-before-wake ordering,
valid mapped device pointers and graph wait-value updates. Cancellation must not
forge peer completion. CUDA memory-operation support and a successful compile
alone do not prove that design works on GB10.

The [hardware wait probe](TP_STREAM_MEMOP_QUALIFICATION.md) is a standalone
qualification artifact. Exact PR source cd344a64 passed 24 NIC-to-GPU trials on
Spark0/Spark1 with shared memfd registration, stale readiness, cancellation and
recovery. A wait-bypass mutation failed the intended assertion. GPU completion
was observed before any post-release CUDA call. Updating 91 wait nodes averaged
4.25 microseconds per replay. These primitive results support integration testing;
they do not establish model throughput. See the [receipt](receipts/tp-hardware-wait-cd344a64.json).
