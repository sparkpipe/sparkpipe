# Interface contracts

This document records the caller-visible contracts of shared headers under `include/sparkpipe/`, plus two shared code paths. The headers carry no comments, so these rules live here: what a caller must do, and what the implementation guarantees in return. Each section names the symbols it covers.

## include/sparkpipe/spark_hidden_transport.h

### Fixed-buffer send

`SparkHiddenTransportSendFixedFunction` is the `send_fixed` member of `SparkHiddenTransportInterface`. Callers reach it through `SparkHiddenTransportSendFixed`.

- `remote_offset` is a byte offset into the peer's registered memory region, not an absolute address.
- `sequence` is only a completion tag. It selects no address. The host RDMA backend ignores it.
- `SparkHiddenTransportSendFixed` returns `SPARK_STATUS_INVALID_ARGUMENT` when the session or `local_buffer` is null, or when the module leaves `send_fixed` unset.
- In the host RDMA backend, `send_fixed` copies `bytes` from `local_buffer` to the start of the buffer registered with `set_fixed_local`, then calls `SparkWeightdClientMeshBroadcast` with source offset 0 and `remote_offset` as the destination offset on the peers. It returns `SPARK_STATUS_INVALID_ARGUMENT` when no local buffer has been set or `bytes` is zero.

## include/sparkpipe/spark_kda_reference.h

`SparkKdaReferenceHead` is a scalar host reference for one KDA head: a delta-rule state update with per-key retention. It is an independent oracle for validating device kernels, not a production path.

Caller contract:

- `state` is row-major `[key_count][value_count]` and is updated in place.
- `query` and `key` must already be normalized and scaled. The function does not normalize them.
- `retention` holds `key_count` linear decay factors. A caller whose gate is in log space passes `exp(log_gate)`.
- `key_count` and `value_count` must be positive. The function does not validate any argument.

One call does the following:

1. It multiplies every state row `k` by `retention[k]`. This happens exactly once, before any prediction.
2. For each value column `v`, it computes `prediction = sum_k state[k][v] * key[k]` and `correction = beta * (value[v] - prediction)`, then applies `state[k][v] += correction * key[k]`.
3. It writes `output[v] = sum_k state[k][v] * query[k]`, using the updated column.

## include/sparkpipe/spark_kv_cache.h

### Pinning a resident table

`SparkKvCacheArenaPinResidentTable(arena, logical_block_indices, block_count, resident_slot_indices)` pins a table of logical blocks and returns their resident slots.

- The arena does not lock block state internally. The caller serializes all access to one arena.
- `logical_block_indices` and `resident_slot_indices` must not overlap. If they do, the call returns `SPARK_STATUS_INVALID_ARGUMENT`. A `block_count` of zero returns `SPARK_STATUS_OK` and touches neither array.
- Each entry takes one residency pin on its block and writes the block's resident slot to `resident_slot_indices`. If a block is not allocated, not resident, or has a residency reservation, the call fails with `SPARK_STATUS_BUSY`.
- A repeated index takes one pin per occurrence. Calling `SparkKvCacheArenaUnpinResidentTable` with the same list releases each of them.
- A pinned block is never picked as an eviction victim. Freeing it, recycling it or marking it non-resident returns `SPARK_STATUS_BUSY`. Its resident slot therefore stays valid until it is unpinned. Keep the pins until the device work that uses those slots has completed.
- Keep the logical index list unchanged until the matching unpin, which takes the same list.
- On failure the call unpins only the blocks this call pinned. It returns the original error, or the unpin error if the rollback itself fails. `resident_slot_indices` is meaningful only on success.
- `SparkKvCacheArenaUnpinResidentTable` processes every entry, even after an error, and returns the first error.

## include/sparkpipe/spark_kv_page_cache.h

### State store

`SparkKvPageCacheAttachStateStore(cache, store)` attaches a second page store that holds recurrent state records, keyed by logical page. A page has a record only when a stateful publish ended on it; its backing may hold fewer records than there are logical pages.

- The call returns `SPARK_STATUS_INVALID_ARGUMENT` unless all of these hold:
  - the cache already has a page store and no state store;
  - `store` is a different object from the page store;
  - `store` has a matching ABI version and descriptor size, and an initialized transfer worker;
  - `store->logical_page_capacity` is at least the arena's logical block count.
- Attach before admitting any lane. The call returns `SPARK_STATUS_BUSY` once a sequence is live, a page has been published, or the store already holds backing pages.
- A page's state record is keyed by the page's current arena generation. Hold the page's residency pin for the whole of a state transfer, so that the page cannot be evicted or freed during the transfer.
- A publish whose lane carries `SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_STATELESS` needs no record and creates a stateless entry. A page sealed without a record also becomes a stateless entry. A stateless entry holds valid KV and can be shared, deduplicated and parked, but it is never a prefix: resolving a lane prefix that ends on it returns `SPARK_STATUS_NOT_FOUND`. The snapshot store saves it with its chain and KV page and no state segment.
- `SparkKvPageCachePromoteState(cache, logical_page)` turns the stateless entry on that page stateful once the caller has written its record. It returns `SPARK_STATUS_NOT_FOUND` unless the page holds a valid stateless entry. If the entry was already saved to the snapshot store, its file is removed and the entry is queued for a new save with its state; a full save queue skips that save, as for any other save.

### Staged reply checkpoint

A stateless publish whose lane also carries `SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_STAGED` keeps the lane's recurrent state in the stage KV binding. The flag is valid only together with `SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_STATELESS`. The batch engine sets it on every decode publication that drops state.

- The binding captures the lane's state into the lane's state buffer and remembers the published page, its generation and the sequence. Any later publish of the lane replaces that stage: a staged one with its own page, any other with none.
- When a CACHE_RELEASE admission releases the lane, the binding writes the staged state as the page's record and promotes the entry. A stage whose sequence, page generation or stateless entry no longer matches, or whose write fails, is dropped and counted in `recurrent_staged_drops`; the release itself never fails for it.
- The batch engine marks the staged block a resume point when the release completes, so the next request that carries the whole reply resumes after it.

### State demotion

`SparkKvPageCacheDemoteState` frees one state record. It walks the LRU list from the oldest entry, skips stateless entries and entries with a snapshot save pending, invalidates the first record it can, and marks that entry stateless. The KV page stays cached. It returns `SPARK_STATUS_CAPACITY_EXCEEDED` when no record can be freed, and `SPARK_STATUS_INVALID_ARGUMENT` when no state store is attached. The stage KV binding calls it before `SparkKvPageCacheEvictUnused` when a state record write finds the store full.

### Eviction

`SparkKvPageCacheEvictUnused` evicts one cache entry, which is one page at the tail of a prefix chain. The entry comes from the cache's LRU list, which holds only entries with no references (no child entry and no sequence), oldest first. The call takes the first entry on that list whose page can be discarded: the page is allocated, the cache holds the only arena reference to it, and it is neither pinned nor residency-reserved.

- Evicting an entry invalidates the page's records in the page store when the cache has one, and in the state store if one is attached. It then frees the page and drops one reference from the parent entry, which may make the parent evictable.
- If no entry qualifies, the call returns `SPARK_STATUS_CAPACITY_EXCEEDED`.

### Pinned lane transaction

`SparkKvPageCacheBeginPinnedLaneTransaction(cache, lane, logical_pages, physical_pages, page_capacity, page_count_out, mutation_flags_out)` prepares a lane, begins its transaction and pins its full page table.

- The caller has exclusive ownership of the lane's resident sequence slot. It serializes all access to the cache and its arena, because neither one locks.
- The call returns `SPARK_STATUS_BUSY` while non-resident pages of the lane's table are still being fetched from the page store. Retry the call when that happens.
- On success, the first `*page_count_out` entries of `logical_pages` and `physical_pages` hold the lane's table. Both arrays remain owned by the caller. Every listed page carries one residency pin. `*mutation_flags_out` records the `SPARK_KV_PAGE_CACHE_MUTATION_*` lane changes that this transaction made.
- Unpin the table with `SparkKvCacheArenaUnpinResidentTable(arena, logical_pages, page_count)` once device work has completed. Unpin before `SparkKvPageCacheCompleteLane`, or before `SparkKvPageCacheRollbackLaneTransaction` on abort. Both can free a mutable page: completion when it deduplicates the page against a published entry, rollback when it releases pages the transaction allocated. Freeing a pinned page returns `SPARK_STATUS_BUSY`.
- On failure the call releases the pins it took and rolls back its lane mutations. `*page_count_out` and `*mutation_flags_out` are zero.

### Lane transactions

`SparkKvLaneTransactions` tracks one `SparkKvLaneTransaction` per resident sequence slot. Each one moves through these phases:

- `SPARK_KV_LANE_TRANSACTION_EMPTY`.
- `SPARK_KV_LANE_TRANSACTION_PREPARED`, reached through `SparkKvLaneTransactionsAdmit` with `SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE`, which runs a pinned lane transaction for each lane.
- `SPARK_KV_LANE_TRANSACTION_COMMITTED`, reached with `SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT`.
- `SPARK_KV_LANE_TRANSACTION_EXECUTING`, reached through `SparkKvLaneTransactionsClaim`.
- Back to `SPARK_KV_LANE_TRANSACTION_EMPTY`, through `SparkKvLaneTransactionsFinish`.

`SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT` unpins a prepared or committed lane, rolls it back and returns it to `SPARK_KV_LANE_TRANSACTION_EMPTY`.

`SparkKvLaneTransactionsClaim(transactions, frame)`:

- The dispatch decision binds the frame to its admission. `SparkStageKvBindingAdmit` sets these fields, and `SparkModelDriverApplyAdmissionDecision` copies them into the frame:
  - `driver_dispatch_generation` is the request's `control_generation`;
  - `driver_dispatch_cookie0` is its `transaction_id`;
  - `driver_dispatch_cookie1` is its `submission_id`.
- Claim rebuilds the admission request from the frame and requires every lane to be `SPARK_KV_LANE_TRANSACTION_COMMITTED`. The rebuilt request must match the saved one field for field: identifiers, generations, deadline, counts, priority, frame flags and residency. Each lane's complete payload must also match, including the prefix, publish and block identities.
- A lane in the wrong phase returns `SPARK_STATUS_BUSY`. A content mismatch returns `SPARK_STATUS_VALIDATION_FAILED`. On success every lane moves to `SPARK_KV_LANE_TRANSACTION_EXECUTING`.

`SparkKvLaneTransactionsReset(transactions)` is a quiescent reset:

- If any lane is `SPARK_KV_LANE_TRANSACTION_EXECUTING`, the call returns `SPARK_STATUS_BUSY` before it changes anything.
- Otherwise it aborts every prepared or committed lane, releases every live sequence and evicts every unreferenced chain. If an entry is still valid afterwards, it returns `SPARK_STATUS_BUSY`.
- If a later step returns `SPARK_STATUS_BUSY` or an error, keep admission stopped and call reset again. Lanes already aborted stay in `SPARK_KV_LANE_TRANSACTION_EMPTY`, so completed cleanup is not repeated.

## include/sparkpipe/spark_kv_page_store.h

### Layered page layout

`SparkKvLayeredPageLayout` and `SparkKvPageStoreCopyLayered` describe a packed backing page that contains one page slice from each native layer.

- The host buffer holds `layer_count` consecutive slices of `layer_page_bytes` each, in layer order. `bytes` must equal `layer_page_bytes * layer_count`.
- The slice for layer `l` maps to the device address `device_base + l * layer_stride_bytes + physical_page * layer_page_bytes`, in both copy directions.
- `layer_stride_bytes` must be at least `layer_page_bytes * page_count`. A layer slab may therefore have padding after its last page. `device_bytes` must cover the last layer's pages.
- Every device access goes through the `SparkKvPageStoreCopyFunction` callback, which is the hardware boundary. The layout code has no CUDA dependency.

### Transfer worker

Each store runs one worker thread. That thread executes queued writeback, prefetch and readback jobs and performs their device copies, through the copy callback when the store has one.

`SparkKvPageStoreWaitForTransfers` blocks until no job is queued or active. It does not consume results: a completed job keeps its terminal status until the caller repeats the operation that queued it, or until `SparkKvPageStoreProgress` collects it. `SparkKvPageStoreProgress` collects completed writeback and prefetch jobs, never readback jobs.

- Keep every buffer named by a queued job alive until that job completes.
- Do not destroy the store while a wait is in progress.
- Never call `SparkKvPageStoreWaitForTransfers` from a copy callback. A queued job's callback runs on the worker thread while that job is active, so the wait would never return.

### Readback

`SparkKvPageStoreReadback(store, logical_page_index, generation, destination, bytes)` is a polled operation.

- A call with no job pending for the page queues a copy of the page's record into `destination` when a job slot is free, and returns `SPARK_STATUS_BUSY`. Repeat the identical request, with the same page, generation and destination, while it returns `SPARK_STATUS_BUSY`. Once the job is done, the next call returns its terminal status and frees the job.
- `destination` remains owned by the store until a terminal result or until the store is destroyed.
- The record must be valid for that generation, or the call returns `SPARK_STATUS_NOT_FOUND`. `bytes` must equal the store's `page_bytes`.
- Readback copies into a buffer only. It does not change KV arena residency.

### Record validation and invalidation

- `SparkKvPageStoreValidateRecord` returns `SPARK_STATUS_OK` only for a completed record of the given generation. It returns `SPARK_STATUS_BUSY` while a writeback has the page reserved, and `SPARK_STATUS_NOT_FOUND` otherwise. It never schedules a copy.
- `SparkKvPageStoreInvalidatePair(first, second, logical_page_index, generation)` locks both workers in store address order. It checks that each store can drop the page's record, then invalidates both. Each store must hold no job for the page, including a completed job whose result has not been collected, and its record must be either absent or valid for `generation`. If either check fails, neither record changes.

## include/sparkpipe/spark_model_serving_adapter.h

### Mandatory operations and capability bits

Every adapter must support prefill, decode and release work, JIT cache transactions and reset. None of these has a capability bit. Prefix reuse and cache publication are also required but do carry bits: `SparkModelServingAdapterValidateDescriptor` returns `SPARK_STATUS_UNSUPPORTED` unless the descriptor sets `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE` and `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH`.

- `SparkModelServingAdapterValidateInterface` requires all ten operations in `SparkModelServingAdapterInterface` to be set: `initialize`, `destroy`, `validate_submission`, `submit`, `prefetch`, `resolve_prefetch`, `progress`, `quiesce`, `snapshot` and `reset`.
- If any one is null, validation logs the missing operation and returns `SPARK_STATUS_INVALID_ARGUMENT`.
- `prefetch` prepares the cache for a set of submissions. `resolve_prefetch` then commits that preparation (`SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT`) or aborts it (`SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT`). `SparkModelServingAdapterResolvePrefetch` returns `SPARK_STATUS_INVALID_ARGUMENT` for any other resolution value.

Validation rejects any capability bit outside `SPARK_MODEL_SERVING_ADAPTER_KNOWN_CAPABILITIES`, whether it appears in the descriptor's `capability_flags` or in the caller's required mask.

- Bits `0x1`, `0x2`, `0x4`, `0x20`, `0x40`, `0x80` and `0x100` are retired. They stay reserved and are rejected.
- Never reassign a retired bit or accept it as a compatibility alias.

## include/sparkpipe/spark_stagepack_format.h

### 4-bit weight formats

`SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED` (8) is NVFP4 in the NVIDIA ModelOpt layout.

- Payload: e2m1 values, two per byte, `rows * columns / 2` bytes.
- Scales: one segment per expert. A segment holds that expert's e4m3 scale bytes, one per 16 values, then `SPARK_STAGEPACK_NVFP4_GLOBAL_SCALE_BYTES` (8) bytes: an F32 input scale, then the F32 weight global as the segment's last 4 bytes. `SparkStagePackFamilyScaleBytes` computes the total.
- Decoded weight: `e2m1 * e4m3 scale * F32 weight global`.

`SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1_E8M0G32` (9) is MXFP4 as the checkpoint stores it.

- Payload: e2m1 values, two per byte, `rows * columns / 2` bytes.
- Scales: one E8M0 scale byte per 32 values, `rows * columns / 32` bytes. There is no global scale.
- Both planes are copied from the checkpoint unchanged.

## include/sparkpipe/spark_tp_chain_ordinal.h

`SparkTpChainOrdinal(chain_id, lane_count, credits_per_lane, operation_capacity, operation_index, ordinal)` maps one operation of a chain to a collective ordinal:

- `lane = chain_id % lane_count`, `epoch = chain_id / lane_count`
- `generation = epoch * operation_capacity + operation_index`
- `credits = lane_count * credits_per_lane`
- `ordinal = generation * credits + lane * credits_per_lane + operation_index % credits_per_lane`

It returns `SPARK_STATUS_INVALID_ARGUMENT` for a null output, a `chain_id` of zero, or a zero `lane_count`, `credits_per_lane` or `operation_capacity`. It returns `SPARK_STATUS_CAPACITY_EXCEEDED` in three cases:

- `operation_index >= operation_capacity`;
- the generation would reach `SPARK_TP_CHAIN_MAX_GENERATION` (`UINT64_MAX >> 16`);
- the ordinal would overflow 64 bits.

`SparkTpChainIdCapacity(lane_count, operation_capacity)` returns the largest chain ID for which every operation index stays under that generation bound. It returns zero for zero arguments.

Caller contract:

- Every rank must compute the same ordinal for the same operation. All ranks pass the same `chain_id`, `lane_count`, `credits_per_lane` and `operation_capacity`. Admission must place a chain in lane `chain_id % lane_count` on every rank.
- Lane `lane` owns credits `lane * credits_per_lane` through `lane * credits_per_lane + credits_per_lane - 1`. A second chain in the same lane maps to the same credits, so only one chain may be live per lane, and it holds those credits until it completes.
- Chain IDs must increase within a lane, so that each new chain gets later generations.
- `lane_count`, `credits_per_lane` and `operation_capacity` are fixed for the session. The ordinal depends on all three, so changing one maps the same operation to a different ordinal.

## include/sparkpipe/spark_tp_device_collective.h

### Enqueue

`SparkTpDeviceCollectiveEnqueue(collective, submission, operation_kind)` runs one collective round. `operation_kind` must be at most `SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL`, or the call returns `SPARK_STATUS_INVALID_ARGUMENT`.

- Call it from a host thread. It issues CUDA calls on `cuda_stream`, and unless graph capture is armed or the submission uses stream-ordered completion it also synchronizes that stream inside the call. CUDA forbids CUDA calls inside a host callback, so never call it from one.
- The collective keeps no queue of pending submissions: the call issues the round itself. Work it queued on `cuda_stream` can still be pending when it returns: the whole round under graph capture or stream-ordered completion, and the combine into `full_device` on the host-round path (spin waits, one logical sequence, a payload that fits one mesh slot), which is queued after the synchronize.
- The call copies the submission descriptor. `local_device`, `full_device` and `cuda_stream` must stay valid until the work queued on `cuda_stream` has completed, and `completion_context` until the completion callback has run. The callback can run before that queued work has completed.
- A submission must set `completion_function`, with two exceptions: graph capture is armed, or the collective uses hardware waits and the submission sets `SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION`. Any other submission without a callback returns `SPARK_STATUS_INVALID_ARGUMENT`. An `ordinal` of `UINT64_MAX` is also rejected.
- On `SPARK_STATUS_OK` with a callback and graph capture not armed, the collective's completion thread later invokes `completion_function(completion_context, completion)`. In that completion, `status` is `SPARK_STATUS_OK`, `slot_index` and `ordinal` are copied from the submission, `credit_index` is 0 and `generation` is `ordinal + 1`. While graph capture is armed, no completion is queued.
- A non-OK return delivers no completion. During an uncaptured host round, a cancel or a missed deadline returns `SPARK_STATUS_BUSY`. The deadline is the smaller of the configured `operation_timeout_milli` and 120 s. If the collective's `SparkWeightdClient` connection is dead while that round waits, it returns `SPARK_STATUS_IO_ERROR`.
- Once `SparkTpDeviceCollectiveDestroy` has begun stopping the completion thread, the call returns `SPARK_STATUS_BUSY`. That thread delivers every completion already queued before it exits.

### Deferred verification

A submission is deferred when graph capture is not armed, the collective uses hardware waits, the submission sets `SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION` and it has no `completion_function`.

- The call queues the round on `cuda_stream` and returns without synchronizing the stream or checking the round. It counts the round as deferred.
- `SparkTpDeviceCollectiveVerifyDeferred(collective, stream)` checks every round deferred since the last check. It synchronizes `stream` once and returns `SPARK_STATUS_IO_ERROR` unless the device completed all of them without an error word. It returns `SPARK_STATUS_OK` at once when nothing is deferred.
- A caller must verify before the host reads any result that depends on a deferred round, and before it starts the next chain. A failed round leaves the values it wrote undefined, so a result read before verification can be wrong without any error.

## include/sparkpipe/spark_weightd.h

### Residency

`SparkWeightdClientResidency(client, arena_generation, residency, timeout)` reports one lazy arena that the connection is attached to: `group_count`, `present_count`, `present_bytes`, `epoch` and `fixed_pool`.

- It returns `SPARK_STATUS_NOT_FOUND` when the connection is not attached to that generation or the arena is not lazy. An arena with a sticky failure returns that failure.
- `fixed_pool` is 1 when one allocation holds the whole arena. A fixed pool never evicts a present group, and weightd frees an arena only when no client is attached and no lease is held. So while the caller stays attached, a group reported present stays present at the same offset.
- `SparkWeightdMapResident(map, timeout, resident)` sets `*resident` to 1 only when the arena has a fixed pool, every group is present and the map holds the whole pool mapping. A consumer that sees 1 may address any group at the map base plus its offset without a lease, for as long as it stays attached.
- An acquire whose groups are all present pins them and returns. It does not budget, load or synchronize the device.

## Host RDMA capability set

The deployment's `transport.mode` selects the capability mask that `node/model_residentd.c` passes to `SparkHiddenTransportLoadInterfaceFromSharedObject`. `SparkModelResidentdTransportContract` builds the mask; the module must declare every bit in it. An adapter that declares `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT` without `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP` loads no transport at all.

For `host-rdma`, the required mask is `SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS`: the production set plus host-pinned RDMA and CUDA-mapped host memory.

- The required mask is not `SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS`. That set adds batched submission, poll descriptors, multi-lane and the remote-completion doorbell.
- A required mask must list only bits the daemon cannot run without. Recommended bits must not be promoted into it.

The host RDMA backend in `ring/transport/rdma.cu` declares, in its default build:

- `SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS`;
- `SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS`;
- `SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS`.

When `SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT` is nonzero it instead declares `SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS` and `SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS`. A module must declare only capabilities it implements. This backend implements neither multi-lane nor the remote-completion doorbell, so it does not declare them.

Poll descriptors:

- A rank with a previous or next pipeline stage opens a transport session. On every iteration, the residentd serve loop calls `SparkHiddenTransportGetPollDescriptors` on each open session.
- That call returns `SPARK_STATUS_NOT_FOUND` unless the module declares `SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS` and sets `get_poll_descriptors`. The serve loop stops on any non-OK status.
- The `host-rdma` required mask does not contain `SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS`, so loading does not check it. A module without it loads, and the serve loop of a rank with an open session then stops at its first poll descriptor query.
- The host RDMA backend implements `get_poll_descriptors` and reports zero descriptors. It declares the capability so that the serve loop can run.

Batch functions:

- `SparkHiddenTransportValidateInterface` requires `post_receive_batch` and `send_batch` when `SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION` appears in either the module's `capability_flags` or the required mask.
- `SparkHiddenTransportPostReceiveBatch` and `SparkHiddenTransportSendBatch` work only when the module declares the capability and sets both functions. Otherwise, for a batch that passes packet validation, they return `SPARK_STATUS_MODULE_NOT_VALIDATED`.
- The host RDMA backend provides both functions as per-packet loops over `post_receive` and `send`, but does not declare batched submission. On this backend the batch wrappers therefore return `SPARK_STATUS_MODULE_NOT_VALIDATED`, and callers must use the single-packet calls.

## Batch engine failure circuit

`SparkModelBatchDispatchKind` in `runtime/model_batch_engine.c` suspends dispatch after repeated pipeline faults.

- If `SparkModelPipelineClientSubmit` returns an error other than `SPARK_STATUS_BUSY`, the engine increments `consecutive_pipeline_failures`.
- `SPARK_STATUS_BUSY` is backpressure, for example a full rank queue, no free pipeline transaction, or a refusal from the first rank. It is not a fault, so it does not count toward the circuit and does not reset the counter.
- At 8 consecutive failures the engine logs `batch circuit open`, resets the counter and sets `circuit_open_until_ns` to 15 s in the future.
- While the circuit is open, dispatch returns `SPARK_STATUS_BUSY` without submitting and sets `next_progress_ns` to `circuit_open_until_ns`.
- Any successful submit clears both the circuit and the counter.
