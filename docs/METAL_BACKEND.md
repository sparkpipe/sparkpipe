# Metal backend and the single-Studio stages

This is the authority for how SparkPipe runs on an Apple GPU: the kernel
launch part of the device API, the Metal backend's queue model, the Metal
kernel library, the single-device stage runner, the precision policy those
kernels and the references share, and how the Qwen3.8-27B and Gemma 4 26B
A4B stages are qualified on one M5 Ultra Mac Studio. Measured results are in
[`PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md#mac-studio) and the
receipts under
[`qualification/metal/qwen38_27b/`](../qualification/metal/qwen38_27b/) and
[`qualification/metal/gemma4_26b/`](../qualification/metal/gemma4_26b/).
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
| `spark_head_rmsnorm_rope` | per-head RMSNorm with an optional FP32 gain, then rotary embedding with rotate-half pairs over the head's first `rope_dimension` dimensions, a table of up to 256 inverse frequencies (a zero frequency leaves its pair as it is), and either one FP32 rotation rounded once or the publisher's eager rounding (`bf16(bf16(x bf16(cos)) + bf16(p bf16(sin)))`) |
| `spark_rmsnorm_branches` | sandwich norms: `h = bf16(h + bf16(rms(a) w))`, or with two branches `c = bf16(bf16(rms(a) w1) + bf16(rms(b) w2))` and `h = bf16(h + bf16(rms(c) w))`, then optionally `bf16(h * scalar)` |
| `spark_rmsnorm_scaled` | `bf16(bf16(bf16(rms(x)) s) c)`: a weightless norm, a per-channel scale and a constant, each rounded (an expert router's input) |
| `spark_gemv_r{1,2,4,8}`, `spark_gemv_f32_r*` | the GEMV family's projections for one or two rows, and its verify frames in passes of up to eight rows: BF16 weights `[N, K]`, 4, 4, 2 and 1 outputs per threadgroup at 1, 2, 4 and 8 rows, the inner dimension split across the threadgroup's four SIMD groups, FP32 accumulation, BF16 or FP32 out |
| `spark_gemm_rows16`, `spark_gemm_rows32`, `spark_gemm_reduce` | decode projections for 3 to 64 rows on the tensor operations: 128 outputs per threadgroup, the weights staged through threadgroup memory 64 inputs at a time with the next slice prefetched into registers, the inner dimension split so the launch has about 1,024 threadgroups, FP32 partials summed in split order by the reduce kernel |
| `spark_gemm`, `spark_gemm_f32` | prefill projections on the Metal 4 tensor operations (`matmul2d`, 128 x 64 tiles, FP32 accumulation, relaxed precision off) |
| `spark_causal_conv_silu` | depthwise causal convolution over a per-sequence window, BF16 sum, then `SparkMathSwishFast` (`x * sigmoid(x)`, CUDA's `LM_CONV_SWISH`) |
| `spark_gated_delta_prepare`, `spark_gated_delta_rule` | L2-normalised query and key, decay and write strength, then the gated delta rule over FP32 state, one SIMD group per value column |
| `spark_kv_store_paged` | key and value rows into 64-slot pages through a page table; a position outside its sequence's pages or a page outside the pool raises the device error word (`SparkKvPage`, shared with the attention kernels) |
| `spark_attention_{decode,prefill_blocks,combine}_d{256,512}` | causal grouped-query attention over paged KV for 256- and 512-wide heads, split into key partitions and merged, with an optional sliding window (a row sees its last `window` positions) and an optional sigmoid output gate; a key on a page `SparkKvPage` rejects raises the device error word and is skipped (decode per key, prefill per 64-key tile); decode walks 128-key partitions one key per SIMD group, prefill computes 32 (16 for 512-wide heads) query rows against 64-key pages on the tensor operations (scores and the FP32 probability-times-value product in `matmul2d`, the online softmax in FP32 between them) |
| `spark_silu_mul`, `spark_gelu_tanh_mul`, `spark_add_rows` | the gated activations (`bf16(silu(g) u)`, `bf16(bf16(gelu_tanh(g)) u)` with `SparkMathGeluTanh`) and the final residual add |
| `spark_silu_mul_routed`, `spark_gelu_tanh_mul_routed` | the experts' gated activations with the route weight folded in before the down projection, as CUDA's `LmSwigluPackedKernel` and `LmGeluTanhPackedKernel` do: `bf16(silu(g) u w)` and `bf16(bf16(gelu_tanh(g)) u w)` with `SparkMathGeluTanhScaleFirst` |
| `spark_route_topk`, `spark_route_pinned`, `spark_gemv_routed`, `spark_route_combine` | expert routing: softmax over the router's BF16 logits (a NaN or infinite logit raises the device error word), the top k in a total order (NaN below every probability, lowest index on ties) or, pinned, the experts a route override names, renormalised over a pairwise sum tree plus 1e-20 (the order of CUDA's warp top-k) and times a per-expert scale; the GEMV of each route's expert on its row (the same per-row arithmetic as `spark_gemv_r1`); and the routes' already weighted outputs summed in FP32 in route order and rounded once to BF16, as CUDA's `LmMoeLocalFinalizeKernel` does |
| `spark_scale_rows`, `spark_softcap_rows` | a BF16 row scale (an embedding multiplier) and the FP32 logit softcap `cap tanh(z / cap)` (`SparkMathSoftcap`) |
| `spark_logprob_rows` | argmax (lowest index on ties), log-normaliser and a target's log-probability over FP32 logits |
| `spark_stream_read` | the streaming-read bandwidth probe |

The kernels are Metal sources written for this tree. The decode GEMV has the
shape of MLX's GEMV for row-major matrices (`mlx/backend/metal/kernels/gemv.h`
at `github.com/ml-explore/mlx`): a SIMD group's 32 lanes walk the inner
dimension for a block of four output rows and reduce with SIMD operations;
here each lane loads 16 bytes per row per step and the four SIMD groups of a
threadgroup take interleaved quarters of the inner dimension, summed in SIMD
order through group memory. The GEMMs and the prefill attention use Apple's
`MetalPerformancePrimitives` `matmul2d` as its SDK header documents it. No third-party source is copied, so NOTICE has no
entry for them.
They implement the same operations as the CUDA kernels in
`inference/kernels/`.

### Shared kernel math

`inference/kernels/kernel_math.cuh` holds the formulas both compilers build:
bf16 conversion and rounding, the scaled sigmoid with sigmoid and SiLU as
its cases, softplus, the gated-delta decay, the tanh GeLU and the softcap.
The experts' tanh GeLU is `SparkMathGeluTanhScaleFirst`, which multiplies
`0.044715 g g g` from the left as CUDA's expert kernels always have; the
dense GeLU cubes first. Both backends use the same two, so the choice
between them is made in one place.
Each is an inline function that the CUDA kernels (through `dtype.cuh`), the
Metal kernels (through `common.metal`) and the host CUDA shim of the host
tests include. A formula is defined once, so a change to it reaches both
backends.

- The header shares formulas, not schedules. Tiling, data movement, matrix
  units, reductions and launch shapes stay per backend, because that is
  where each backend's speed is.
- The macros at the top of the header pick each backend's intrinsics:
  bit casts, `exp`, `log` and `tanh`. A `Fast` variant is CUDA's `__expf`
  and `__logf`. Metal builds with `-fno-fast-math`, so both variants are the
  precise functions there.
- A CUDA kernel that moves to the header keeps its machine code: every
  conversion reproduces the replaced expression and its exp flavor, and the
  `cuobjdump -sass` of every serving adapter is compared before and after.
  A copy whose expression differs stays until a change with its own gate
  replaces it (TECHDEBT lists them).

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

## Device stage models

A device model is a shape, a tensor list and a layer schedule over two
model-neutral layers in `inference/runner/`. A model directory holds nothing
else, so every further model tests these interfaces: an operation a model
needs that they lack is added there, for every model, and the existing
models' gates are re-run to show their output is unchanged.

- `device_stage_model.{h,c}` (`SparkDeviceStage`) is the model's lifecycle. It
  opens the rank pack and checks its topology (`SparkDeviceStageOpen`) and its
  configuration values (`SparkDeviceStageExpectConfig`), reserves scratch and
  recurrent state against the state budget (`SparkDeviceStageReserve`), fills
  the runner geometry from the stage shape and the pack's `embed`,
  `head_norm` and `head` (`SparkDeviceStageGeometry`), binds named tensors
  with their expected sizes (`SparkDeviceStageTensor`,
  `SparkDeviceStageLayerTensor`), lays out scratch slices
  (`SparkDeviceStagePlan`, `SparkDeviceStageAllocate`), validates a step's
  row tables (`SparkDeviceStageStepTables`) and serves the KV pool
  (`SparkDeviceStagePaged`). It also provides the interface entries that do
  not depend on the model: close, manifest check, the weightd, sideband and
  lane refusals, `attach_kv`, `recurrent_copy`, `reset_slot`, probe and
  report. A model struct starts with its `SparkDeviceStage`, so these entries
  take the model pointer as it is.
- Recurrent state is a set of pools, each `[layer][slot]` of one width;
  `recurrent_copy` packs pool by pool, layer by layer, and `reset_slot` zeroes
  a slot in every pool.
- `device_layers.{h,c}` are the layer kinds, each a shape struct, a bind call
  that names the layer's pack tensors and a run call:

| Layer kind | Shape | Pack tensors (`layers.N.`) |
| --- | --- | --- |
| gated delta rule (`SparkDeviceGatedDelta`) | key and value heads and dimensions, convolution taps; recurrent pools 0 (FP32 state) and 1 (BF16 convolution window) | `gdn_qkvz`, `gdn_ba`, `gdn_conv`, `gdn_a_log`, `gdn_dt_bias`, `gdn_norm`, `gdn_out` |
| attention (`SparkDeviceAttention`) | heads, KV heads, head dimension, rotary span, base and the number of rotated pairs, eager or fused rotary rounding, scale; options: a sigmoid output gate (queries and gates interleaved per head in the fused projection), a sliding window, a weightless value norm, and keys shared as values (the fused projection then holds queries and keys only) | `attn_qkv`, `attn_q_norm`, `attn_k_norm`, `attn_out` |
| gated FFN (`SparkDeviceGatedFfn`) | intermediate width and activation (SiLU, or tanh GeLU), applied by `spark_gated_mul` before the down projection | `ffn_gate_up`, `ffn_down` |
| experts (`SparkDeviceExperts`) | experts, top k, expert width, activation; the router reads the residual through `spark_rmsnorm_scaled`, the experts read it through their own norm; a step's route override replaces the top k (qualification only) | `router_proj`, `router_scale`, `expert_scale`, `pre_ff_norm_2`, `moe_w1` (gate), `moe_w3` (up), `moe_w2` (down) |
| residual RMSNorm (`SparkDeviceResidualNorm`) | the stage's hidden width and epsilon | the model names its norm weights |
| branch norms (`SparkDeviceBranchNorm`) | one or two branches, an optional layer scalar | the model names its norm weights |

A stage's KV pool holds the widest slot any of its attention layers needs
(`kv_heads x head_dimension` of the stage shape); a layer with narrower slots
uses its pages at that layer's own slot stride. A stage shape can tie the
head to the embedding table and soft-cap the logits, which the runner applies
before the log-probabilities. Other norm placements, other activations and
attention without query and key norms have no option yet; the model that
needs one adds it here.

A device model is a family's Metal backend running the family schedule
([`FAMILY_SCHEDULES.md`](FAMILY_SCHEDULES.md)) on the model's TP1 geometry; the
schedule and the geometry are the ones the CUDA module runs.
`model-families/qwen38/device/qwen38_device_stage.cpp` is Qwen3.8-27B on these
layers: 64 layers, every fourth one (from the fourth) attention and the rest
gated delta, each followed by the SwiGLU FFN, with the attention and
gated-delta projections sharing one scratch slice. The recurrent state is FP32,
`[layer][slot][head][key][value]`, and `recurrent_copy` packs it in the same
order as the CUDA model.

`model-families/gemma4/device/gemma4_device_stage.cpp` is Gemma 4 26B A4B
(text only) on the same layers: 30 layers, every sixth one (from the sixth)
full attention (16 query and 2 KV heads of 512, keys shared as values, rotary
on the first 64 of 256 pairs with base 10^6) and the rest sliding-window
attention (16 query and 8 KV heads of 256, window 1,024, the whole head
rotated with base 10^4), q and k norms weighted and the value norm
weightless, attention scale 1; then the tanh-GeLU dense FFN (2,112 wide) and
128 experts (top 8, 704 wide) on the same residual, combined by the two-branch
sandwich norm and the layer scalar. The embedding is multiplied by
`bf16(sqrt(2816)) = 53`, the head is the embedding table and the logits are
capped at 30. The pack is `tools/gemma4_tp16_pack.py --tp-degree 1`.

## Decode and prefill by row count

A device model declares its projection family in its stage shape
(`SparkDeviceStageShape.projection`); `SparkOpsLinear` then picks the kernel
from the family and the step's row count, and `SparkOpsAttention` the
attention kernel from the longest run of rows any one sequence has:

| Rows in the step | `SPARK_PROJECTION_GEMV` (Gemma 4 26B A4B) | `SPARK_PROJECTION_ROW_TILES` (Qwen3.8-27B) | Weight reads per step |
| --- | --- | --- | --- |
| 1 or 2 | `spark_gemv_r1`/`_r2` | `spark_gemm_rows16`, then `spark_gemm_reduce` | once |
| 3 to 64 | `spark_gemm_rows16` (up to 16 rows) or `spark_gemm_rows32`, then `spark_gemm_reduce` | the same | once |
| a verify frame of up to 64 rows | the GEMV in passes of 8, 4, 2 and 1 rows | the same row tiles | once per 8 rows (GEMV), once (row tiles) |
| above 64 | `spark_gemm` (prefill) | the same | once per 128 rows |

| Rows per sequence | Attention |
| --- | --- |
| at most 8, or any verify frame | `spark_attention_decode`, every row independent |
| more | `spark_attention_prefill_blocks` |

A row's result depends only on its own inputs and on which projection kernel
ran, never on the other rows or on the row count inside one kernel class:
the GEMV computes every row with the same sequence of operations whatever
its pass size, the tensor-unit kernel tiles rows in 16 or 32 with identical
arithmetic for each row (the two tile heights give the same bits), and its
split count depends only on the projection's shape.
`tests/test_metal_kernels.c` checks this bit for bit (`linear row
independence`, with and without row tiles and row-exact frames). So with row
tiles every row of every step up to 64 rows has the bits of a one-row step:
every batched sequence decodes exactly its solo tokens, and a verify frame
computes each row exactly as plain decode does while reading the weights
once. With the GEMV family one or two sequences give exactly the solo
tokens, three to 64 give the same tokens as one another but not the solo
GEMV's where a rounding difference flips a near tie, and a verify frame
(`row_exact`) runs the GEMV passes so that it stays exact.

Which family a model takes is measured, both being exact for speculation:
row tiles cost one-row steps 7 to 9% on large matrices and 35 to 110% on
small ones, and make long verify frames close to one step. For Qwen3.8-27B
that trades B1 20.1 tok/s for 18.2 and the whole-file rewrite's 103.7 tok/s
for 222.9; for Gemma 4 26B A4B it would trade B1 92.7 for 76.3 and the
Korean summary's 103.1 for 83.6 against a rewrite gain of 13%, so Gemma keeps
the GEMV:

| | GEMV family | Row tiles |
| --- | --- | --- |
| Qwen3.8-27B B1 | 20.1 tok/s | 18.2 tok/s |
| Qwen frame of 8 / 16 / 32 rows | 62.9 / 121.5 / 238.8 ms | 58.4 / 61.8 / 72.2 ms |
| Qwen rewrite / Korean, with speculation | 103.7 / 21.07 tok/s | 222.9 / 19.14 tok/s |
| Gemma 4 26B A4B B1 | 92.7 tok/s | 76.3 tok/s |
| Gemma frame of 8 / 16 / 32 rows | 23.4 / 39.3 / 68.1 ms | 25.3 / 36.5 / 57.0 ms |
| Gemma rewrite / Korean, with speculation | 323 / 103.1 tok/s | 366.0 / 83.6 tok/s |

A GEMV that holds 16 to 64 rows per pass in the one-row order of sums is
bit-exact but SIMT FP32 tops out at 6 to 7 TFLOPS here, so a 32-row Qwen
frame (1.64 TFLOP) cannot go below about 230 ms; the tensor units reach 27
TFLOPS at 32 rows. `build/metal_linear_bench LIB K N ROWS F32 ROW_TILES`
times one projection.

The split-K partials live in a 64 MiB buffer the ops instance owns; the first
queue that runs a split projection owns it, and a split projection on another
queue is refused.

## Speculation

The device runner takes the CUDA stage runner's verify contract.
`SparkDeviceStageRunnerBatch.verify` makes a step a verify frame: the model
runs it with `commit = 0`, so the recurrent state and the convolution window
are read but not written and the sequence's next position does not move.
`SparkDeviceStageRunnerKeepRows(runner, kept, sequences)` then keeps the
first `kept[s]` rows of each sequence: the model's `keep_rows` replays the
recurrent update over those rows with `commit = 1`, and the runner advances
each sequence by its kept count. KV written for rejected positions is
overwritten when the sequence reaches them again.

- `SparkDeviceStageStepTables` marks a `commit = 0` step row-exact and
  refuses one longer than the stage's verify rows (64, the replay slab).
- The gated-delta layer kind gathers each verify row's mix and gate
  projections into a per-layer slab before the convolution;
  `SparkDeviceGatedDeltaKeep` gathers the kept rows back, reruns the
  convolution and the delta rule over them with `commit = 1` and so writes
  the state and window the rows would have left one step at a time. No
  projection runs again. `SparkDeviceStageKeepTables` builds the kept-row
  list and the per-sequence row ranges on the host from the runner's
  shared tables.
- A model wires `keep_rows` to its recurrent layers; for Qwen3.8-27B that is
  the 48 gated-delta layers. The attention layers keep nothing, since their
  KV rows are already in place.

Drafts come from the request's own context, as in the CUDA serving path:
`metal_stage_run --mode spec` indexes the prompt and the generated
tokens with the common lookup index (`src/spark_speculation_lookup_draft.c`,
3-token grams, matches up to 64 tokens, 64 candidates) and sizes each frame
with the serving path's frame chooser (`src/spark_speculation_frame.c`),
seeded with the measured verify costs and learning acceptance per match
class as it goes; a step with no useful draft is an ordinary one-row step.
The chooser's first frame in a match class probes; the tool caps a probe at
the largest frame whose measured cost is within the chooser's margin of a
one-row step, so text the lookup cannot predict pays nothing for the probe.
Where no frame is that cheap (a mixture-of-experts step, where every extra
row reads more experts) the probe is one draft, about a fifth of a step
once per match class.
The accepted prefix plus the model's next token are emitted, so greedy
output with speculation is exactly greedy output without it.

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

The batch work moved no rounding point. The K-split GEMV, the split-K
tensor-unit projection with its ordered reduce, the 128-key decode
partitions and the block prefill attention (FP32 scores, FP32
probabilities times BF16 values with FP32 accumulation on the tensor
operations) only reorder FP32 sums; every value in the table above is
rounded where it was before.

The delta-rule state, the L2-normalised query and key, the decay and write
strength, the attention scores and softmax stay FP32. Rotary angles are
`float(position) * inverse_frequency` in FP32, with the inverse frequencies
computed in double and rounded once.

Gemma 4 26B A4B follows the publisher's eager PyTorch rounding, which
`tools/gemma4_reference.py` reproduces in numpy: the embedding times
`bf16(sqrt(hidden))`; every norm `bf16(x rsqrt(mean(x^2) + eps) w)` with plain
FP32 `w`; every projection; the per-head query and key norms, then the
rotation as `bf16(bf16(x bf16(cos)) + bf16(p bf16(sin)))`; the weightless
value norm; the attention output (FP32 scores, softmax and probability times
value); the sandwich norms and residual sums as in `spark_rmsnorm_branches`;
`bf16(gelu_tanh(g))`, then `* up`, for the dense FFN and each expert; the
router input `bf16(bf16(bf16(rms(x)) s) / sqrt(hidden))` and its BF16
logits (softmax, top 8 and the weights in FP32); each expert's down
projection, then `* weight`, then the BF16 sum over the routes in ascending
expert order; the FP32 logits, soft-capped at 30.

## Qualification on one M5 Ultra

The gates run on the Studio from a committed revision shipped with
`git archive`; `tools/qwen38_27b_metal_gates.sh PACK CHECKPOINT OUT REVISION`
runs all of them and `tools/metal_gate_summary.py` writes the receipt.

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
rounding.

The gate rule is relative to that floor and runs in two legs, against A with
B as the floor and against B with A as the floor. A candidate passes a leg
when all three hold:

- its position-weighted mean KL is at most 2 times the floor's;
- its top-1 disagreement rate is at most 1.5 times the floor's;
- the paired test: for each position, the candidate is "above the floor" when
  KL(reference || candidate) exceeds KL(reference || other variant). A
  candidate whose error is distributed like the reduction-order noise is
  above the floor at half the positions; the share may exceed one half by at
  most 3 standard errors, the error taken over contiguous 128-position
  blocks (`sqrt(n/(n-1) * sum (above_b - share * size_b)^2) / positions`),
  because neighbouring positions share context and are not independent.

Metal must pass both legs. `tools/pack_tensor_control.py` writes APFS clones
of the pack with tensors corrupted, and the gate runs each through the same
teacher-forced pass:

| Control | Corruption | What it shows | Required |
| --- | --- | --- | --- |
| `control_layers` | rows 0 and 1 of `ffn_down` swapped in all 64 layers | a kernel that misplaces one output does so in every layer it runs; the gate must see a fault of that reach | fails |
| `control_rotate` | every row of `layers.31.ffn_down` moved down by one | a whole-tensor layout or offset fault | fails |
| `control` | rows 0 and 1 of `layers.31.ffn_down` swapped, one layer only | a probe of the gate's resolution, reported with its numbers | none |

The probe passes: at BF16 the effect of one misplaced output pair in one of
64 layers is inside the rounding noise, and no statistic over these 5,636
positions separates it from a correct run (in the receipt for this revision
its mean KL is 0.79 and 1.00 times the floor's against A and B, and the
paired test sits at -4.8 and +0.2 standard errors). The all-layer control
fails at 4.4 and 4.8 times the floor's KL and +95 and +124 standard errors,
the whole-tensor control at 15 and 18 times and +188 and +197. The gate thus
sees a fault that every layer shares, or that moves a whole tensor, and does
not see a two-channel fault confined to one layer; where between those
sizes its threshold lies is not measured (TECHDEBT). The reference also runs
once with FP32 activations (no BF16 rounding), as a diagnostic of how far
BF16 inference itself moves the distribution.

The floor is large for this model: references A and B agree on the top-1
token at 97.75% of the positions and differ by 0.0395 nats of KL on average,
most of it at a few percent of positions in long contexts where a rounding
difference changes a near tie. An absolute top-1 bar of 99%, which an
earlier gate used, fails the reference against itself, so it could not pass
any correct implementation; top-1 is judged against the floor's 2.25%
disagreement instead, and the KL bound, once 10 times the floor, is 2 times.
The mean KL is dominated by a few near-tie positions and cannot see a small
systematic error; the paired test can, because a systematic error makes the
candidate worse than the floor at most positions rather than at a few.
Results are in [`PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md#mac-studio).
`tools/logit_compare.c` computes KL(ref || test), top-1 agreement, the
largest change in the target token's log-probability and, given the other
variant, the paired counts per block. Both implementations also continue each
prompt greedily for 128 tokens.

**Batches.** `metal_stage_run --mode ladder --batches 1,2,4,...,64`
decodes 128 tokens for each of 64 sequences alone, then for each batch size
prefills that many sequences (1,024-token prompts cut from the gate prompts
at different offsets) and decodes them together, three timed runs after a
discarded warm-up, the median kept. Every batched sequence is compared token
for token with its solo run; at the first token that differs, both runs'
top-1 and top-2 tokens and margins are recorded, so a divergence can be
shown to be a near tie. Repeated runs must give the same tokens: a batch
whose repeats differ fails the gate (`batch_ladder.deterministic`).

**Speculation.** `metal_stage_run --mode spec` measures the verify
cost curve (one sequence after a 933-token prompt, frames of 1 to 64 rows,
and the replay that keeps all of them, median of three after a warm-up),
then decodes 256 tokens greedily with and without speculation on the four
gate prompts and on `qualification/metal/qwen38_27b/spec_prompts.json`: a
whole-file rewrite of `tools/token_prompts.c` and a prose request, three
times each after a warm-up, alternating the two. Every run's tokens must be
identical with and without speculation; tok/s with and without are reported
separately, as the median with the lowest and highest.

**Speed.** `build/device_bandwidth 16 PACK` measures streaming reads of a
16 GiB device-private buffer and of the mapped pack (seven timed passes after
a warm-up, the median kept). `metal_stage_run --mode bench` measures B1
decode of 256 tokens after a 1,024-token prompt and prefill of 1,024 and
8,192 tokens in 2,048-row chunks, three timed runs after a discarded warm-up,
the median kept. Decode reads every weight but the embedding table once per
step, whatever the batch, so its roofline is that byte count over the
measured bandwidth per sequence at B1, and the batch multiplies it until the
per-sequence state (the recurrent state, read and written each step, and the
KV) and the tensor operations' compute take over;
prefill reads the weights once per chunk and is bounded by the tensor
operations, not memory. A mixture-of-experts step reads top-k of each
layer's experts (the stage counts its expert tensors when it binds them),
and a head tied to the embedding is read in full.

### Gemma 4 26B A4B

`tools/gemma4_26b_metal_gates.sh PACK REFERENCE_DIR OUT REVISION` runs the
same gates on Gemma's four prompts (built by `tools/gemma4_reference.py
gate-prompts`, 5,636 positions), with `tools/gemma4_reference.py teacher
--variants a,b,bp` as the reference: numpy, the publisher's eager rounding,
a and b in the two reduction orders.

The router makes the free-route floor useless as a gate. Its logits are
BF16, so the eighth and ninth experts of a row are often a rounding step
apart, and the reduction order alone changes the choice: in a 160-token
pass, b chose different experts than a for 7 rows at layer 1 and 63 at layer
29, and its logits sat 0.31 nats of KL from a's with 82.5% top-1 agreement.
Over the gate's 5,636 positions a and b differ at 18.3% of the layer and
position decisions (0.2% at layer 0, 36% at layer 29), and the free-route
floor is 0.31 nats. A misplaced output in every layer's FFN moves the logits
less than that, so no rule relative to this floor can see it. The gate therefore has a second
set in which every layer's experts are pinned to a's choice: the reference
writes a's experts per layer and position (`<name>.ref_a.routes`), bp is b
with those experts (its own probabilities of them, renormalised, as the
weights), and the Metal tool's teacher mode takes `--pin-routes DIR`, which
fills a route override for every step (`SparkStageRunnerStep.route_override`,
`[layer][row][rank]`) that the experts layer kind uses instead of its top-k
(`spark_route_pinned`). Pinned, a against bp is 0.063 nats.

The receipt applies the rule to both sets. Free routes: Metal against a
and b with b and a as the floor; it must pass, which shows the end-to-end
noise is that of a correct implementation. Pinned routes: Metal, the same
pass prefilled in 64-row chunks (`metal_rows64`, the skinny tensor-unit
projection the batches use) and the controls (run pinned), against a and bp
with each other as the floor; the Metal runs must pass and
`control_layers`, `control_rotate` and `control_experts` (experts 0 and 1 of
every layer's `moe_w2` swapped, the reach of a misindexed expert) must fail. Metal's own top-k is
checked by the kernel test (exact experts at 128 experts, top 8).

The fast batch ladder is not identical to solo decode for most sequences,
and its first differing tokens are not near ties, because a rounding
difference from the batch's projection kernel flips a route upstream. The
gate runs the ladder a second time with `--exact-rows`: every decode step is
a verify frame whose rows all are kept, so each row takes the row-exact
kernels, and every sequence at every batch size must reproduce its solo
tokens (`batch_ladder_exact_rows.identical_to_solo`); the Qwen3.8-27B gate
runs the same exact-rows ladder. That shows the batch
itself (row tables, KV pages, per-sequence attention, routed GEMV) is exact,
and the fast path differs only by rounding.
