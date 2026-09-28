# Completion events and polling

Payloads and readiness counters already arrive through RDMA writes into each
receiver's local mapped host memory. `SparkWeightdClientAttachLazy` (GLM) or
`SparkWeightdClientMeshMap` (`SparkTpDeviceCollectiveAttachMesh`) maps the local
mesh file, and `PrepareReceiveBf16` passes that mapping to CUDA. In spin mode, a
GPU `.cv` load of the readiness counter is not an RDMA READ operation. Replacing
that repeated local load with a pushed notification is a separate change from
payload direction.
CPU polling, GPU wait kernels and network transfer time need separate measurements.

The API worker now sleeps on resident sockets and a nonblocking request-queue
pipe. Enqueue and cancellation wake that pipe; only the worker mutates the batch
engine. The batch CLI flushes events immediately after progress and checks
completion before sleeping, with no fixed polling cadence. The API worker's poll
timeout follows the batch engine's next deadline, except that while a
submission is refused as busy it retries every 5 ms (`API_BUSY_RETRY_MS`,
`node/model_api.c`). Submission checkpoint persistence retains its 60-second
deadline.

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
This change bumped the weightd IPC ABI (`SPARK_WEIGHTD_IPC_ABI_VERSION`); clients
and daemon must be rebuilt together.

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

[`TP_STREAM_MEMOP_QUALIFICATION.md`](TP_STREAM_MEMOP_QUALIFICATION.md) is the
authority for device waits. In `hardware` mode, the fleet's serving mode, GPU
waits are stream memory waits on a ready word the daemon publishes. In `spin`
mode, the default when `SPARK_TP_WAIT_MODE` is unset, they are GPU polling
kernels. In both modes the weight daemon polls while a mesh activity interval
is active and sleeps only as described above. An incoming RDMA WRITE produces no
receiver completion event, and a GPU write to a host doorbell wakes no CPU file
descriptor, so neither can replace that polling. RDMA completion notifications
are [one-shot](https://man7.org/linux/man-pages/man3/ibv_req_notify_cq.3.html)
and must be rearmed and drained without a lost-wakeup race.

Mesh registration is owned per exact mapping, shared by main and hidden-channel
collectives and released only after the final owner drains. Distinct mappings
register independently. Since 5814bf2 (2026-09-23), when `cudaHostRegister`
returns `cudaErrorInvalidValue`, as it did for a shared weightd's RDMA-registered
region in that commit's reproduction, the collective logs `MESH-REGISTER-SKIP` and continues with the
unregistered mapping; every other error fails with `MESH-REGISTER-FAIL`
(`ring/transport/tp_device_collective.c`). A failed unregister logs
`MESH-UNREGISTER-FAIL` and retains a cleanup-only owner. The skip is not
recorded, so the final owner's release still calls `cudaHostUnregister` on the
unregistered mapping, which CUDA rejects; by code reading, teardown after a
skip therefore ends in `MESH-UNREGISTER-FAIL`. The fleet's registrations
currently succeed, so this path has not been observed there.
