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
| `inference/kernels/weight_prefetch.cuh` | 1 upload | Serving write after idle: the prefetch plan is published after a graph capture ends and before the graph's first launch. |
| `inference/runner/pass_prefill.cuh` | 1 fill | Load: the pass's buffers at allocation. |
| `inference/runner/stage_model_cuda.cuh` | 2 fills | Load. |
| `inference/runner/stage_runner.cu` | 1 readback | Readback after sync: the group offsets, after `SparkGpuStreamSynchronize(state->stream)`. |
| `model-families/dsv41_flash/source/dsv41_flash_stage.cuh` | 7 uploads, 11 readbacks, 1 device copy | Uploads: load (expert placement, token map, engram tables and resident engram rows, `Upload`). Readbacks: after sync (route and step dumps, the frame readback at 3039). Device copy: load, ordered on the legacy stream with the cell packing kernel it feeds, and followed by `SparkGpuDeviceSynchronize`. |
| `model-families/gemma4/source/gemma4_stage.cuh` | 1 upload | Load: the YaRN table. |
| `model-families/glm52/source/glm52_stage.cuh` | 1 upload | Load: the rope table. |
| `model-families/glm5_next/source/glm5_next_stage_model.cuh` | 1 fill | Load. |
| `model-families/k3/source/k3_resident_decode_stage_cuda.cu` | 10 fills | Load: the dispatch's pools, scratch and index scratch. Nine of them ignore the call's return value; see the follow-ups. |
| `model-families/k3/source/k3_stage.cuh` | 2 fills | Load. |
| `model-families/k3/source/k3_streamed_prefill.cuh` | 1 upload | Load: the identity chunk slots. |
| `model-families/mimo26/source/mimo26_stage.cuh` | 2 uploads | Load. |
| `model-families/motif/source/motif_stage.cuh` | 1 upload | Load. |
| `model-families/qwen38/source/qwen38_hybrid_stage.cuh` | 4 readbacks | Readback after sync: the layer dump syncs the stream first and launches nothing in between. |
| `ring/transport/tp_device_collective.c` | 12 uploads, 15 readbacks, 3 fills | See below. |
| `runtime/spark_weightd.c` | 4 uploads | Serving write: the verify sink, the staged range loads and the lease sink. Their consumers are engine processes that learn of the data from weightd's reply after the call returns. The arena epoch bump is published the same way. All come from pageable or staging memory, so they were cross-process races on a discrete GPU, and the contract closes them. |
| `runtime/spark_weightd_spine.c` | 1 upload | Same as weightd: spine spans copied into a lease before the reply. |

### The mesh collective

- **`EnsureCells`**, once per collective: a fill of the arrival ring, the zeroing of the round control block, and five 8-byte seeds from pageable variables.
  - The zeroing was `SparkGpuMemsetAsync(..., 0)`, legacy-stream work that the collective's kernels on the non-blocking submission stream do not wait for. It is now `SparkGpuMemset`, which has landed when it returns.
  - Class: load. Fixed for a discrete GPU: the seeds and fills land before any round runs.
- **`SeedEpoch`**, per chain: two uploads from the pinned published cell and one fill.
  - `ChainKey` first checks that the active stream has finished.
  - The pinned uploads were host-synchronous already. The fill was a race on any GPU and now lands.
  - Class: serving write after idle.
- **The rounds loop** (`EnqueueRoundsInternal`): two uploads of the round control words from a pageable struct before the loop kernel launches, which synchronizes its stream right after.
  - It was a race on a discrete GPU and now lands. Class: serving write after idle.
- **`ArmCapture`:** two uploads zeroing the error and diag words before a graph capture.
  - A capture can begin right after the previous step's graph was launched on the same stream, and that graph uses those words. So the zeroing was a backward race on any GPU.
  - Fixed: `ArmCapture` now synchronizes the collective's active stream, which `GraphPreLaunch` set to the launch stream, before it writes.
- **`DisarmCapture`:** two reads of the sequence words after a capture.
  - Nothing executes during a capture, and the stream was idle when the capture was armed, so the values are final. Class: readback after sync.
- **Readbacks after sync:** the rounds loop's control readback (1827), `GraphPreLaunch` (2292), and the timing readback (2676, 2679), each after a stream sync or terminal check.
- **Monitoring reads:** the wait-kernel timeout report (1024, 1025), `DeviceRoundsDone`, `GraphProgress`, the stuck, arrival and diag dumps, and `GraphError`.
- **`ClearGraphError`:** one upload of zero into the error word. No production code calls it.

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
- **`tests/test_upload_ordering_sites.py`** fails on any production call of an asynchronous copy, fill or event record on stream 0, on a stream default left on those wrappers, and on a synchronous copy or fill site that the table above does not count.

## Host-visible buffers: the next pass

Pinned staging buffers that asynchronous copies read, and mapped buffers that kernels read in place, are host writes that a kernel reads. Each is safe only if the host does not rewrite it before the copy or kernel that reads it has run. Checked so far:
- **Stage runner, dispatch staging** (the token, distribution and output staging): each dispatch uploads its inputs from pinned staging on its stream, runs, reads its outputs back on the same stream, and synchronizes that stream before it returns. So the next dispatch's host writes follow the last copy that read them.
  - A dispatch that fails before that sync returns without it. The next dispatch could then rewrite staging that a copy queued before the failure still reads.

Listed for the next pass, which checks each host write against the last reader on its stream:
- **Stage runner:** the lane staging, the route fault word, and the tap slab and chain.
- **Pass prefill:** the round sums, the host staging, the health words, and the registered row and pass regions.
- **DSV4.1:** the engram ids and rows, including the ahead buffers; the KV error word; and the exchange counts and totals.
- **GLM 5.2:** the host staging, the lane staging and the KV error words.
- **residentd:** the boundary allocation.
- **Mesh collective:** the published cell and the registered receive region.
- **TP mesh kernels:** the host pointer mapped by `spark_tp_mesh_kernels.cuh`.
