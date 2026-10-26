# Metal backend and the single-Studio Qwen3.8-27B stage

This is the authority for how SparkPipe runs on an Apple GPU: the kernel
launch part of the device API, the Metal backend's queue model, the Metal
kernel library, the single-device stage runner, the precision policy those
kernels and the reference share, and how the Qwen3.8-27B stage is qualified
on one M5 Ultra Mac Studio. Measured results are in
[`PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md#mac-studio) and the
receipts under
[`qualification/metal/qwen38_27b/`](../qualification/metal/qwen38_27b/).
Open work is in [`TECHDEBT.md`](../TECHDEBT.md#mac-studio-deployment).

## Kernel launch in the device API

`include/sparkpipe/spark_device.h` launches kernels by name from a kernel
library:

| Call | Contract |
| --- | --- |
| `SparkDeviceLibraryLoad(&library, path, owner)` | loads a backend kernel library (a `.metallib` on Metal); a missing file is `NOT_FOUND` |
| `SparkDeviceKernelCreate(&kernel, library, name)` | builds the named kernel; an absent name is `NOT_FOUND`, a kernel the device cannot build is `TARGET_MISMATCH` |
| `SparkDeviceKernelGetInfo(kernel, &info)` | the kernel's own thread limit, SIMD width and static group memory |
| `SparkDeviceKernelLaunch(queue, kernel, &launch)` | enqueues one launch: group grid, threads per group, dynamic group memory, up to 4096 bytes of constants and up to 30 range bindings |
| `SparkDeviceKernelDestroy`, `SparkDeviceLibraryUnload` | `BUSY` while a queued launch uses the kernel, or while a library has live kernels |

A binding is a buffer range, its byte count and its access (read, write or
both). `device.c` checks every launch before the backend sees it: the grid and
group are non-empty, the group fits the kernel's thread limit and the device's
group memory (`UNSUPPORTED` above either), the constants fit, every binding is
a live, device-visible range inside its buffer, and the requested access is one
the buffer grants (a file mapping cannot be bound for writing, pageable host
memory cannot be bound at all). The launch then stamps every bound buffer and
the kernel with the queue's sequence, so `SparkDeviceFree` and
`SparkDeviceKernelDestroy` refuse while the launch is in flight (I13).

On Metal the constants bind at buffer index 0 and binding `i` at index `i + 1`,
so a kernel's signature is `constant Params &p [[buffer(0)]]` followed by its
buffers in binding order. The host and CUDA backends load no kernel libraries;
`SparkDeviceLibraryLoad` returns `UNSUPPORTED` naming the backend, and
`tests/test_device_contract.c` checks that refusal on both.

## The Metal queue

A queue owns one `MTLCommandQueue`. Copies, fills, kernel launches and event
waits are encoded into the queue's open command buffer on the calling thread;
the buffer is committed when the host waits or queries, records an event,
queues a host callback, or after 64 operations, so the GPU starts on a long
step while the host is still encoding it.

- Launches go into one serial compute encoder per command buffer, and copies
  and fills into blit encoders in the same buffer. Buffers use Metal's tracked
  hazards, so a launch, a copy and a launch on one queue see each other's
  writes in order (`TestKernelOrdering`).
- Command buffers hold unretained references; the device layer's stamps keep
  every bound buffer and kernel alive until the work that uses it completes.
- A host callback runs on the queue's worker thread after everything before it
  completes, and work queued after it waits on a per-queue shared event that
  the worker signals when the callback returns, so the callback has the same
  ordering as `cudaLaunchHostFunc`.
- An event is an `MTLSharedEvent` signalled at the end of a committed command
  buffer; another queue waits on it with `encodeWaitForEvent`, so cross-queue
  order needs no host round trip. Event timestamps are the command buffer's
  `GPUEndTime`.
- A failed command buffer marks the queue and the device faulted; the next wait
  returns `IO_ERROR` with Metal's error text, and every later device call fails
  (as on the other backends).

Queues are not ordered with each other without an event: a copy on one queue
followed by `SparkDeviceCopySync` on the device's synchronous queue needs a
queue wait or an event between them.

## The kernel library

`make build/runtime_libs/sparkpipe_kernels.metallib` compiles
`inference/kernels/metal/*.metal` with `-std=metal4.0 -fno-fast-math`, so
`exp`, `rsqrt`, `cos` and `sin` are Metal's precise functions. The library
ships next to `libsparkpipe_device_metal.dylib`, where
`SparkDeviceOpsAcquire` finds it. Every kernel is model-neutral; the
parameter structs in `inference/kernels/metal/kernel_params.h` are shared with
the host code in `inference/runner/device_ops.c`, which validates shapes and
tensor sizes before each launch.

| Kernel | Operation |
| --- | --- |
| `spark_gather_rows` | embedding rows by id; an id outside the table raises the device error word |
| `spark_rmsnorm_residual` | `h = bf16(h + r)` written back, then `bf16(rmsnorm(h) * w)` with an FP32 gain |
| `spark_rmsnorm_gate` | per-head RMSNorm with a BF16 gain and a SiLU output gate read from a fused row |
| `spark_head_rmsnorm_rope` | per-head RMSNorm with an FP32 gain, then rotary embedding on the head's first dimensions (rotate-half pairs) |
| `spark_gemv_r{1,2,4,8}`, `spark_gemv_f32_r*`, `spark_gemv_gated_r*` | decode projections for up to eight rows: BF16 weights `[N, K]`, FP32 accumulation, BF16 or FP32 out, and the gated form that computes `bf16(silu(gate) * up)` from a fused input row |
| `spark_gemm`, `spark_gemm_f32` | prefill projections on the Metal 4 tensor operations (`matmul2d`, 128 x 64 tiles, FP32 accumulation, relaxed precision off) |
| `spark_causal_conv_silu` | depthwise causal convolution over a per-sequence window, BF16 sum, SiLU |
| `spark_gated_delta_prepare`, `spark_gated_delta_rule` | L2-normalised query and key, decay and write strength, then the gated delta rule over FP32 state, one SIMD group per value column |
| `spark_kv_store_paged` | key and value rows into 64-slot pages through a page table |
| `spark_attention_decode`, `spark_attention_prefill`, `spark_attention_combine` | causal grouped-query attention over paged KV, split into key partitions and merged, with the sigmoid output gate |
| `spark_silu_mul`, `spark_add_rows` | the FFN activation and the final residual add |
| `spark_logprob_rows` | argmax (lowest index on ties), log-normaliser and a target's log-probability over FP32 logits |
| `spark_stream_read` | the streaming-read bandwidth probe |

The kernels are Metal sources written for this tree. The decode GEMV has the
shape of MLX's GEMV for row-major matrices (`mlx/backend/metal/kernels/gemv.h`
at `github.com/ml-explore/mlx`): a SIMD group's 32 lanes walk the inner
dimension for a block of four output rows, reduce with SIMD operations and
stage nothing in group memory; here each lane loads 16 bytes per row per step.
The prefill GEMM uses Apple's `MetalPerformancePrimitives` `matmul2d` as its
SDK header documents it. No third-party source is copied, so NOTICE has no
entry for them.
They implement the same operations as the CUDA kernels in
`inference/kernels/`; one kernel body for both backends is open work
(TECHDEBT).

## The single-device stage runner

`inference/runner/device_stage_runner.c` hosts one stage model on one device
through the same `SparkStageRunnerModelInterface` the CUDA stage runner uses.
It initializes the device, maps the rank pack with `SparkDeviceMapFile` (on
unified memory the kernels read the pack's page-cache pages with no copy), and
owns the embedding gather, the final norm, the head projection with FP32
logits, the log-probability kernel, the row tables and one paged KV pool. Each
step validates that every sequence's rows are contiguous and continue that
sequence at its next position.

The model interface is used with these meanings on a device backend:

- `stream` is a `SparkDeviceQueue *`, and every pointer field of
  `SparkStageRunnerStep` and `SparkStageRunnerKv` is a device address, which
  the model resolves to a range with `SparkDeviceResolve`. The interface still
  passes them type-erased (`void *`), as on CUDA.
- `bind_device(model, weights, geometry)` hands the model the device address
  and size of the mapped rank pack; the model resolves each tensor by its pack
  offset. weightd's lazy pack is a CUDA VMM arena, so a device model's `bind`
  refuses (`UNSUPPORTED`).
- The runner has one execution lane and no collectives; a model's `round` or
  `lane` call is refused.

`model-families/qwen38/device/qwen38_device_stage.c` is the Qwen3.8-27B model
(64 layers, 48 gated-delta and 16 gated-attention, dense SwiGLU FFN) for this
runner. Prefill rows above eight use the tensor-operation GEMM and the tiled
prefill attention; up to eight rows use the GEMV kernels, the fused gated
down projection and the partitioned decode attention. The recurrent state is
FP32, `[layer][slot][head][key][value]`, and `recurrent_copy` packs it in the
same order as the CUDA model.

## Precision policy

Weights are the publisher's BF16. Every projection accumulates in FP32 and
rounds its output to BF16; the head projection keeps FP32 logits. Rounding to
BF16 is round-to-nearest-even and happens at these points, the same in the
Metal kernels and in the reference:

| Step | Value rounded to BF16 |
| --- | --- |
| residual | `h + attention`, `h + ffn` |
| RMSNorm | `x * rsqrt(mean(x^2) + 1e-6) * (1 + w)`, the gain folded to FP32 by the packer |
| gated delta | every projection; the convolution sum, then `silu` of it; the rule's output; in the gated norm `x * rsqrt(..)`, then `* w`, then `* silu(z)` |
| attention | every projection; the normed query and key, then the rotated query and key; the attention output, then `* sigmoid(gate)` |
| FFN | gate and up; `silu(gate) * up` |

The delta-rule state, the L2-normalised query and key, the decay and write
strength, the attention scores and softmax stay FP32. Rotary angles are
`float(position) * inverse_frequency` in FP32, with the inverse frequencies
computed in double and rounded once.

## Qualification on one M5 Ultra

The gates run on the Studio from a committed revision shipped with
`git archive`; `tools/qwen38_27b_metal_gates.sh PACK CHECKPOINT OUT REVISION`
runs all of them and `tools/qwen38_27b_metal_summary.py` writes the receipt.

**Device contract.** `build/test_device_contract_metal`: the backend-neutral
contract plus the kernel tests (refusals, kernel-copy-kernel ordering on one
queue, kernel ordering across queues through an event behind a host callback,
busy kernels and buffers, group memory).

**Per-operation tests.** `build/test_metal_kernels` runs every kernel at the
shapes the model uses and compares each output element with a float64 host
reference computed from the same BF16 inputs. Each tolerance is derived from
the operation: `u = 2^-8` for the final BF16 rounding (relative to the exact
value), `gamma_n = n * 2^-24 / (1 - n * 2^-24)` times the sum of absolute
terms for an FP32 reduction of `n` terms, a 16-ulp envelope (`16 * 2^-24`
relative) for each precise transcendental function, and, where a value is
rounded to BF16 before more work, the size of that rounding's possible flip:
the reference checks whether the exact value lies within the FP32 error of a
rounding boundary and, only then, adds one BF16 step, propagated through the
rest of the operation. The recurrent tests carry a bound that decays with the
state and grows by the per-step rounding, and the attention test propagates
the score error through the softmax. Every test then corrupts one input the
kernel reads and must find elements outside the tolerance; a test whose
control does not fail is itself a failure. The receipt lists the worst
error-to-tolerance ratio per case.

**End to end.** Four prompts of 700 to 2,040 tokens built by
`tools/qwen38_27b_gate_prompts.py` from files in this tree (5,636
teacher-forced positions). The Metal run prefills the first half of each
prompt and decodes the rest one token at a time, writing FP32 logits for
every position. The reference, `tools/qwen38_27b_reference.c`, is a separate
implementation of the same precision policy on the CPU: it reads the original
safetensors (separate q, k, v, z, b and a projections, unfused, norms folded
at load), converts the weights to FP32 once and runs the whole prompt through
Accelerate's FP32 GEMM. It runs twice per prompt with different reduction
orders: variant A uses the BLAS GEMM over the full inner dimension and
forward sums; variant B splits every GEMM's inner dimension into 1,024-wide
chunks accumulated last chunk first and reverses every sum. KL divergence
between A and B is the floor: the noise of FP32 reduction order under BF16
rounding. The Metal run passes when its position-weighted mean KL against A
is at most ten times the floor and its top-1 agreement with A is at least 99%.
`tools/pack_tensor_control.py` writes APFS clones of the pack with one
tensor corrupted, `layers.31.ffn_down` with rows 0 and 1 swapped and with
every row moved down by one; the gate is run on both. The reference also runs
once with FP32 activations (no BF16 rounding), as a diagnostic of how far
BF16 inference itself moves the distribution.

The floor is large for this model: references A and B agree on the top-1
token at 97.75% of the positions and differ by 0.0395 nats of KL on average,
most of it at a few percent of positions in long contexts where a rounding
difference changes a near tie and the divergence carries on for some tokens.
Metal sits inside it (1.11 times the floor's KL, 97.62% top-1), so the 99%
top-1 bar is not met, by Metal or by the reference against itself. The
whole-tensor control lands at 15.4 times the floor and fails; the two-row
swap lands at 1.16 times the floor and is not detected, because its effect
on the output is inside the BF16 noise. Results are in
[`PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md#mac-studio).
`tools/logit_compare.c` computes KL(ref || test), top-1 agreement and the
largest change in the target token's log-probability. Both implementations
also continue each prompt greedily for 128 tokens.

**Speed.** `build/device_bandwidth 16 PACK` measures streaming reads of a
16 GiB device-private buffer and of the mapped pack (seven timed passes after
a warm-up, the median kept). `qwen38_27b_metal_run --mode bench` measures B1
decode of 256 tokens after a 1,024-token prompt and prefill of 1,024 and
8,192 tokens in 2,048-row chunks, three timed runs after a discarded warm-up,
the median kept. Decode reads every weight but the embedding table once per
token, so its roofline is that byte count over the measured bandwidth;
prefill reads the weights once per chunk and is bounded by the tensor
operations, not memory.
