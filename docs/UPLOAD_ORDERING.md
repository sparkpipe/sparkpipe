# Upload ordering

Device work in SparkPipe runs on non-blocking streams: residentd's execution stream, the stage runner's lanes, K3's load and pair streams, pass prefill's gather and copy streams, and DSV4.1's shared stream. A non-blocking stream does not wait for the legacy default stream. So any copy, fill or kernel issued on the legacy stream is unordered against later work on those streams unless something makes the order explicit.

On a unified-memory GPU (GB10) most of that ordering happens to hold, because a host-to-device copy lands before the call returns. On a discrete GPU it does not:
- **Pageable uploads:** `cudaMemcpy` from pageable memory returns once the data is staged, and the DMA may still be pending.
- **Device copies and fills:** a synchronous device-to-device copy and a synchronous memset of device memory only queue work on the legacy stream.

A kernel launched on a non-blocking stream right after either can read the old bytes. A test that uploaded a 1M-element source this way on a discrete GPU saw half of it as zeros.

## The contract: a synchronous copy or fill has landed when it returns

`SparkGpuMemcpy` and `SparkGpuMemset` complete before they return, in every device flavour:
- **nvrm:** both always ran on the default queue and then synchronized it.
- **CUDA:** `SparkGpuMemcpy` now follows the copy with `cudaStreamSynchronize(0)` for every kind except device to host, which is synchronous already. `SparkGpuMemset` does the same after `cudaMemset`.

The wait covers only the legacy-stream work the call itself queued. The legacy stream already waits for blocking streams before such a copy starts, so the sync adds no dependency the copy did not have.

The asynchronous wrappers (`SparkGpuMemcpyAsync`, `SparkGpuMemcpy2DAsync`, `SparkGpuMemsetAsync`, `SparkGpuMemset2DAsync`, `SparkGpuEventRecord`) no longer default their stream to 0, so every caller names the stream its work is ordered in. No production call passes stream 0.

With the contract, every synchronous copy or fill is ordered before anything the host issues after it, on any stream. What remains to check per site is the other direction: the copy must not touch memory that work issued earlier on a non-blocking stream still uses.

## Classes

- **Load:** the destination was just allocated, or is filled while the model loads, before any stream has work. Only forward ordering matters, and the contract gives it.
  - Before the contract, a pageable upload of this kind was safe only on unified memory and a race on a discrete GPU.
  - A fill or device copy was formally a race on any GPU, since its legacy-stream work could still be queued when the first non-blocking-stream kernel ran.
- **Serving write after idle:** written while serving, after a check that the consumer stream is idle (a stream sync, or the collective's terminal check). Forward ordering is the same as for load.
- **Readback after sync:** a device-to-host copy of data a non-blocking stream produced, after that stream is synchronized. Safe on any GPU.
- **Monitoring read:** a device-to-host read of a counter or error word while kernels may still run. It is concurrent by design: the value is a snapshot used as a progress hint or a diagnostic, never as a completion fence. Safe on any GPU.
- **Fixed:** a race this change removes.

## Sites

The synchronous copy and fill sites in production code, by file. `tests/test_upload_ordering_sites.py` holds the same per-file counts, so a new site fails until it is classified here.

| File | Sites | Class |
|---|---|---|
| `include/sparkpipe/spark_weightd_device.h` | 1 upload | Serving write: `SparkWeightdDeviceWrite`, weightd's one write into its own mappings. `runtime/spark_weightd.c` calls it for the verify sink, the staged range loads, the lease sink and the arena epoch bump, and `runtime/spark_weightd_spine.c` for spine spans copied into a lease. Their consumers are engine processes that learn of the data from weightd's reply after the call returns. All come from pageable or staging memory, so they were cross-process races on a discrete GPU, and the contract closes them. |
| `inference/kernels/weight_prefetch.cuh` | 1 upload | Serving write after idle: the prefetch plan is published after a graph capture ends and before the graph's first launch. |
| `inference/runner/pass_prefill.cuh` | 1 fill | Load: the pass's buffers at allocation. |
| `inference/runner/stage_model_cuda.cuh` | 2 fills | Load. |
| `inference/runner/stage_runner.cu` | 1 readback | Readback after sync: the group offsets, after `SparkGpuStreamSynchronize(state->stream)`. |
| `model-families/dsv41_flash/source/dsv41_flash_stage.cuh` | 7 uploads, 11 readbacks, 1 device copy | Uploads: load (expert placement, token map, engram tables and resident engram rows, `Upload`). Readbacks: after sync (route and step dumps, the frame readback at 3039). Device copy: load, ordered on the legacy stream with the cell packing kernel it feeds, and followed by `SparkGpuDeviceSynchronize`. |
| `model-families/gemma4/source/gemma4_stage.cuh` | 1 upload | Load: the YaRN table. |
| `model-families/glm52/source/glm52_stage.cuh` | 1 upload | Load: the rope table. |
| `model-families/glm5_next/source/glm5_next_stage_model.cuh` | 1 fill | Load. |
| `model-families/k3/source/k3_resident_decode_stage_cuda.cu` | 10 fills | Load: the dispatch's pools, scratch and index scratch. Each fill's result is checked, and a failed fill fails the dispatch's creation. |
| `model-families/k3/source/k3_stage.cuh` | 2 fills | Load. |
| `model-families/k3/source/k3_streamed_prefill.cuh` | 1 upload | Load: the identity chunk slots. |
| `model-families/mimo26/source/mimo26_stage.cuh` | 2 uploads | Load. |
| `model-families/motif/source/motif_stage.cuh` | 1 upload | Load. |
| `model-families/qwen38/source/qwen38_hybrid_stage.cuh` | 4 readbacks | Readback after sync: the layer dump syncs the stream first and launches nothing in between. |
| `modules/k31_dspark_draft_stage/source/spark_k31_dspark_module.cu` | 2 uploads, 1 readback | Uploads: load, the YaRN table and the block rows' destinations at `open`. Readback after sync: `K31Readback`, the probe's reads of context rows (after the ingest that wrote them synchronized the module's stream) and of the block forward's rows and per-layer trace (after the forward's stream sync). |
| `ring/transport/tp_device_collective.c` | 11 uploads, 15 readbacks, 3 fills | See below. |
| `runtime/spark_weightd_lazy_pack.c` | 1 device copy | Load: on a device without unified memory, the bind copies a held pack from its device-read-only import into the spine it has just allocated, in 1 GiB pieces, and returns only after the copies land. No stream has work on that spine before the bind returns. |

### The mesh collective

- **`EnsureCells`**, once per collective: a fill of the arrival ring, the zeroing of the round control block, and five 8-byte seeds from pageable variables.
  - The zeroing was `SparkGpuMemsetAsync(..., 0)`, legacy-stream work that the collective's kernels on the non-blocking submission stream do not wait for. It is now `SparkGpuMemset`, which has landed when it returns.
  - Class: load. Fixed for a discrete GPU: the seeds and fills land before any round runs.
- **`SeedEpoch`**, per chain: two uploads from the pinned published cell and one fill. Every dispatch starts a chain, because the batch engine gives each submission its own request id.
  - `ChainKey` first checks that the active stream has finished.
  - When there is an active stream, the three writes are now asynchronous on it, so they are ordered before the round kernels that follow on the same stream without any host wait. Switching streams requires the old one to be idle.
  - Only the first chain, with no active stream yet, uses the synchronous copies and fill, which land before they return.
  - Before: the pinned uploads were host-synchronous, and the fill was a race on any GPU. Class: serving write after idle; per step, stream-ordered.
- **The rounds loop** (`EnqueueRoundsInternal`): two uploads of the round control words before the loop kernel launches.
  - They are now asynchronous on the submission stream, ordered before the loop kernel. The source is a pageable struct and a stack pair, which the copy stages before it returns.
  - Before: a race on a discrete GPU.
- **`ArmCapture`:** two uploads zeroing the error and diag words before a graph capture.
  - A capture can begin right after the previous step's graph was launched on the same stream, and that graph uses those words. So the zeroing was a backward race on any GPU.
  - Fixed: `ArmCapture` now synchronizes the collective's active stream, which `GraphPreLaunch` set to the launch stream, before it writes.
- **`DisarmCapture`:** two reads of the sequence words after a capture.
  - Nothing executes during a capture, and the stream was idle when the capture was armed, so the values are final. Class: readback after sync.
- **Readbacks after sync:** the rounds loop's control readback (1827), `GraphPreLaunch` (2292), and the timing readback (2676, 2679), each after a stream sync or terminal check.
- **Monitoring reads:** the wait-kernel timeout report (1024, 1025), `DeviceRoundsDone`, `GraphProgress`, the stuck, arrival and diag dumps, and `GraphError`.
- **`ClearGraphError`:** one upload of zero into the error word. No production code calls it.
- **`ReportRecords`:** one upload of six zero words that clears the fault record (`fault_phase` to `guard_poisons`) after the host has printed it.
  - It runs only when the record holds a round-loop timeout or a guard poison. A healthy round leaves the record empty, and nothing is written.
  - Every caller has just read the block back after a stream sync: the round loop's completion, the control read after rounds, the per-round readback and the capture end. No kernel of the collective runs, so the clear cannot race a later timeout's claim.
  - It runs before the caller checks the round's error word and returns that error, so the next round starts with an empty record and its first fault claims it. When the readback itself fails, there is no record to print or clear.
  - Class: readback after sync, on the fault path.

## Per-step cost

No path that runs on every decode step or verify frame issues a synchronous upload, device copy or fill. The per-step writes are `SeedEpoch` and the rounds loop, now ordered in their stream. The remaining synchronous sites run at load, at a graph capture (once per row count), in weightd's load and eviction paths, or after a fault record was printed. The synchronous readbacks keep their old cost, since a device-to-host copy was already synchronous. `test_tp_device_collective_mock` checks that a new chain issues no synchronous copy and no stream sync.

## Kernels on the legacy stream

Load-time kernels launched on stream 0 are followed by `SparkGpuDeviceSynchronize` before serving:
- **DSV4.1:** the FP8 slice kernel at bind; the two rope frequency kernels, followed by the sync at 3078; MXFP4 cell packing, followed by the sync at 4065.
- **K3 streamed prefill:** the identity page kernel, followed immediately by the sync.

## Tests

- **`tests/test_upload_ordering.c`** runs on the CUDA stub in a discrete ordering mode, `spark_stub_cuda_discrete_ordering(1)` or `SPARK_STUB_CUDA_DISCRETE=1`.
  - In this mode a pageable upload, a device-to-device copy, a fill, or any asynchronous operation on stream 0 stays pending on the legacy stream. It lands when the legacy stream or the device is synchronized, before a pinned or device-to-host synchronous copy, or before work on a blocking stream.
  - A non-blocking stream sees device memory as it is, without waiting. New device allocations are filled with `0xa5`, so a write that never landed is visible.
  - **Controls:** the test first shows that the stub models the race (a raw pageable `cudaMemcpy` has not landed when a non-blocking stream reads) and that a pinned upload lands at once.
  - **Contract:** then it checks that `SparkGpuMemcpy` (host to device and device to device) and `SparkGpuMemset` have landed when they return.
  - It fails on main's wrapper and needs no GPU.
- **`tests/test_upload_ordering_sites.py`** fails on any production call of an asynchronous copy, fill or event record on stream 0, on a stream default left on those wrappers, on a synchronous copy or fill site that the table above does not count, and on a host-visible buffer site that the table below does not count.
- **`tests/cuda/stage_runner_failed_dispatch_cuda.cu`** (`tests/test_stage_runner_failed_dispatch_cuda.py`, on a GPU):
  - Control: it first shows that an upload queued behind other work reads its pinned source when it runs, so a host write before then changes what lands.
  - Then a dispatch whose model step fails, queued behind 200 ms of other work. The test checks that the dispatch returns only after its queued work has completed, and that rewriting the runner's token staging right after the failure leaves the failed dispatch's own tokens embedded.
  - Without the settle in `SparkStageRunnerSubmit`, all three checks fail.

## Host-visible buffers

Pinned staging that an asynchronous copy reads, mapped memory that kernels read or write in place, registered regions, and host functions on a stream. A host write must follow the last device reader of the same bytes, and a host read must follow the last device writer. `tests/test_upload_ordering_sites.py` counts the sites per file: allocations, registrations, device aliases and host functions. A new one fails until it is classified here.

Classes:
- **Staging per dispatch:** the host writes it once per dispatch, copies on the dispatch's streams read it, and those streams are synchronized before the dispatch returns.
- **Readback after a wait:** a device-to-host copy into pinned memory, read after a stream sync or an event wait.
- **Monitoring word:** mapped memory a kernel sets on a fault. The host reads it after the step has completed, and clears it only then, or never when the fault is terminal.
- **Stream-ordered host function:** host code that `SparkGpuLaunchHostFunc` runs at its place in the stream, so its reads and writes are ordered with the kernels around it.
- **Ring with release events:** a writer (a host thread or a NIC) reuses a buffer only after the event recorded behind its last device reader has completed.
- **Protocol cell:** mapped or registered memory shared with peers or weightd. The mesh protocol's sequence words and fences order it, not a stream.

| File | Sites | Buffers and class |
| --- | --- | --- |
| `inference/runner/stage_runner.cu` | 13 | See below. |
| `inference/runner/pass_prefill.cuh` | 5 | See below. |
| `model-families/dsv41_flash/source/dsv41_flash_stage.cuh` | 15 | See below. |
| `model-families/glm52/source/glm52_stage.cuh` | 4 | Host staging and lane staging: staging per dispatch. `Begin` writes them once per step and uploads them on the lane's stream, inside the stage runner's dispatch. The KV error words: monitoring words, terminal. |
| `modules/k31_dspark_draft_stage/source/spark_k31_dspark_module.cu` | 5 | Staging per dispatch and one readback. The tap and position staging of `ingest`: each part of up to 256 rows is written, copied on the module's stream, and written into the context with a stream sync before the next part reuses them. The block forward's descriptor and page table: written before the forward, copied at its start on the module's stream, and rewritten only by the next forward, after the stream sync that ends this one. The draft's picks (tokens and confidences): readback after a wait, copied on the module's stream at the end of `propose` and read after its stream sync. |
| `node/model_residentd.c` | 2 | The pipeline boundary, mapped for the host-RDMA transport. It is used only by a rank with a previous or next pipeline stage, which no served lane has, and is claimed per route. |
| `ring/transport/tp_device_collective.c` | 2 | See below. |
| `model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh` | 1 | The device alias of the collective's registered receive region: protocol cells. The mesh needs host page-table access, which GB10 has and the 5090 refuses at setup. |

### The stage runner

- **Token ids, the distribution rows, positions and rules, and the lane stage:** staging per dispatch.
  - Fixed: a dispatch that failed after queuing work returned without waiting for it. The next dispatch could then rewrite staging that a queued copy had not read yet. A queued device-to-host copy could also still write a caller's logprob buffer after the call returned.
  - `SparkStageRunnerSubmit` now synchronizes the runner's stream and its lane streams before a failed dispatch returns. That wait is bounded by the queued kernels, and a queued collective wait ends at its round timeout.
  - With a device collective, the next dispatch's chain key already refused to start while the stream was busy, so only the caller's buffers were exposed. A TP1 runner, such as the 5090 drafters, has no collective and had no such guard.
- **Output tokens and scores:** readback after the dispatch's stream sync.
- **The route fault word:** a monitoring word. `SparkStageRunnerCheck` reads and clears it after the serving adapter's device wait.
- **The tap slab and chain:** mapped, and written by kernels in place. The serving adapter sends their rows after the dispatch returned and before the next one starts, on the same thread.
- **A caller's hidden input:** `StageRunnerCopy` synchronizes after each copy, so the input has been read when the dispatch returns. The cost is one stream wait per dispatch on a stage without the embedding, which no served lane has.

### Pass prefill

- **Host staging:** written after the stream sync that opens each pass, and uploaded before the pass's second sync.
- **The health words:** zeroed after the previous pass's sync, and read after this pass's sync.
- **The round sums:** readback after a wait.
- **The weight window and the row region** (registered): a ring with release events. `LmPassAwaitRelease` waits on the last reader's event before the streamer reuses a slot.

### DSV4.1

- **Engram ids and rows:** a stream-ordered host function (`D41EngramFetch`) reads the ids the hash kernel wrote and writes the rows the gather kernel reads.
- **The ahead ids and rows:** a worker thread fetches them between two stream-ordered host functions, `D41EngramAheadStart` and `D41EngramAheadJoin`. They are double-buffered per table, so the next step's hash kernels run only after this step's gather.
- **The KV error word:** a monitoring word, terminal.
- **The exchange counts:** readback after a stream sync.
- **The exchange totals:** readback after the `counts_ready` event.

### The mesh collective

- **The published cell:** `ChainKey` writes the seed words only after it finds the active stream idle, and returns BUSY otherwise. Each control readback into the cell waits for its stream.
- **The registered receive region:** protocol cells. The host writes the base cell's rank map and degree, fences, then writes the value word. Kernels and peers read the value word.
