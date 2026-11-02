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
| `spark_rmsnorm_residual` | the FP32 sum `s = h + r`, `h = bf16(s)` written back, and `bf16(rmsnorm(s) * w)` with an FP32 gain, the statistic and the normed value taken from the unrounded sum as CUDA's `LmFusedResidualRmsNormKernel` does (the Gemma, Qwen, Ling and GLM 5.2 add-and-norms); or weightless `bf16(rmsnorm(h))`, which may write in place (Muse Glimmer's embedding norm, CUDA's one-head `LmHeadRmsNormKernel`) |
| `spark_rmsnorm_gate` | per-head RMSNorm with a BF16 gain and an output gate read from a fused row: SiLU, or sigmoid when `sigmoid_gate` is set (Qwen 3.8 Flash), as CUDA's `LmHeadRmsNormSiluGateKernel` / `LmHeadRmsNormSigmoidGateKernel`; with `weight_f32` (GLM 5.3 Flash's KDA, `SparkDeviceChannelDeltaShape.norm_f32`) the gain is FP32 and the result is rounded once, `bf16(x r w sigmoid(g))`, as `LmRmsNormSigmoidGateKernel`; `round_norm` rounds the normed product before the gate, `bf16(bf16(x r w) sigmoid(g))` (Kimi K3's KDA) |
| `spark_head_rmsnorm_rope` | per-head RMSNorm (or, `plain`, no norm at all) with an optional FP32 gain and an optional multiplier applied after the norm's BF16 rounding and rounded again (CUDA's `head_multiply`), then rotary embedding over the head's first `rope_dimension` dimensions with rotate-half pairs or, `interleaved`, adjacent pairs `(2i, 2i + 1)` (CUDA's `LM_ROPE_INTERLEAVED`), written at `output_head_stride` per head when that is set (latent attention ropes 64 dimensions of each 192-wide query head straight into its 576-wide packed row), a table of up to 256 inverse frequencies (a zero frequency leaves its pair as it is; YaRN fills it from `SparkRopePlanBuildYarnInvFrequency`), cos and sin times the rope scale, and either one FP32 rotation rounded once, CUDA's `LmRopePair` (`SparkMathRopeLow`, `SparkMathRopeHigh`), or the publisher's eager rounding (`bf16(bf16(x bf16(cos)) + bf16(p bf16(sin)))`), which the Gemma family keeps |
| `spark_rmsnorm_branches` | sandwich norms: `h = bf16(h + bf16(rms(a) w))`, or with two branches `c = bf16(bf16(rms(a) w1) + bf16(rms(b) w2))` and `h = bf16(h + bf16(rms(c) w))`, then optionally `bf16(h * scalar)` |
| `spark_rmsnorm_scaled` | `bf16(bf16(bf16(rms(x)) s) c)`: a weightless norm, a per-channel scale and a constant, each rounded (an expert router's input) |
| `spark_gemv_r{1,2,4,8}`, `spark_gemv_f32_r*` | the GEMV family's projections for one or two rows, its verify frames, and any projection whose inner dimension is not a multiple of the 64-wide row-tile depth (Solar Open 2's 160-wide expert slices at TP8), in passes of up to eight rows: BF16 weights `[N, K]`, 4, 4, 2 and 1 outputs per threadgroup at 1, 2, 4 and 8 rows, the inner dimension split across the threadgroup's four SIMD groups, FP32 accumulation, BF16 or FP32 out |
| `spark_gemm_rows16`, `spark_gemm_rows32`, `spark_gemm_reduce` | decode projections for 3 to 64 rows on the tensor operations: 128 outputs per threadgroup, the weights staged through threadgroup memory 64 inputs at a time with the next slice prefetched into registers, the inner dimension split so the launch has about 1,024 threadgroups, FP32 partials summed in split order by the reduce kernel |
| `spark_gemv_fp8_r{1,2,4,8}`, `spark_dequant_fp8`, `spark_gemm_fp8_blocks` | E4M3 projections with an FP32 scale per row and 128 inputs (`SparkOpsLinearFp8`), in the two forms CUDA uses: up to 16 rows, and for any row-exact frame, CUDA's skinny arithmetic (each 16-weight chunk decoded once through a 256-entry threadgroup table filled from `SparkMathE4m3ToFloat`, summed per row by a sequential FMA chain, then added to the FP32 accumulator by one FMA with its block scale); above 16 rows CUDA's stream GEMM (`LmStreamGemmDense` with an FP8 format): the weights are converted exactly to BF16, each 128-input block (32 for MXFP8) is multiplied in FP32 and added to the total by one FMA with the block's scale (`spark_gemm_fp8_blocks`, `spark_gemm_mxfp8_blocks`) |
| `spark_gemv_mxfp8_r{1,2,4,8}`, `spark_dequant_mxfp8`, `spark_gemv_mxfp8_routed` | MXFP8 projections and routed experts (HY4: `fp8` E4M3 weights with a `<name>_scale` UE8M0 byte per 32 inputs; `scale_group` 32 in the GEMV parameters and in `SparkDeviceWeight`): the FP8 kernels above with the block scale `SparkMathUe8m0ToFloat(s)`, CUDA's `LmSkinnyFormat<LmMxfp8>` (16-weight chunks, one FMA per chunk with its group's scale); the routed form runs one route per grid row on its expert's rows (`SparkOpsLinearRoutedFp8` with `scale_group` 32, `SPARK_ROUTE_SKIP` writes zeros) |
| `spark_gemv_fp8_routed` | the same routed form for CUDA's FP8 experts (`LmFp8`, GLM 5.3 Flash): E4M3 weights with an FP32 scale per row and 128 inputs (`SparkOpsLinearRoutedFp8` with `scale_group` 0, `SparkDeviceExpertMatrix.scale_group` 0) |
| `spark_gemm`, `spark_gemm_f32` | prefill projections on the Metal 4 tensor operations (`matmul2d`, 128 x 64 tiles, FP32 accumulation, relaxed precision off) |
| `spark_causal_conv_silu` | depthwise causal convolution over a per-sequence window, BF16 sum, then `SparkMathSwishFast` (`x * sigmoid(x)`, CUDA's `LM_CONV_SWISH`); with a dilation the window holds `(taps - 1) dilation + 1` entries and the taps read every `dilation`-th one, oldest first, as `LmCausalConvLaunch`'s dilated form (the n-gram embedding's conv); with `weight_f32` the taps are FP32 and the sum goes to the swish unrounded (Kimi K3's KDA) |
| `spark_gated_delta_prepare`, `spark_channel_delta_prepare`, `spark_gated_delta_rule` | L2-normalised query and key, decay and write strength, then the gated delta rule over FP32 state, one SIMD group per value column. The gated form has one decay per value head (Qwen's gated delta); the channel form (`SparkOpsChannelDelta`, the KDA layers of Ling and Solar) has one per key channel, `SparkMathBoundedDecay`: `exp(lower sigmoid(exp(a) (f + b)))` as CUDA's `LmBoundedDecayKernel`, a write strength by `beta_mode`: `sigmoid(beta)` (`LmSigmoidRowsKernel`), doubled `2 bf16(sigmoid(beta))` or rounded `bf16(sigmoid(beta))` (`LmBf16SigmoidRowsKernel`, GLM 5.3 Flash), and the rule multiplies each key row of the state by its channel's decay. With `prenorm_epsilon` the query and key are L2-normalised twice, first `bf16(x rsqrt(sum x^2 + prenorm_epsilon))` and then the usual normalisation of that (Kimi K3) |
| `spark_kv_store_paged` | key and value rows into 64-slot pages through a page table; a position outside its sequence's pages or a page outside the pool raises the device error word (`SparkKvPage`, shared with the attention kernels) |
| `spark_attention_{decode,prefill_blocks,combine}_d{128,256,512,192v128,576v512}` | causal grouped-query attention over paged KV for 128-, 256- and 512-wide heads, for 192-wide queries and keys with 128-wide values, and for latent attention's 576-wide queries and keys (latent and rope) with 512-wide values (`value_dimension`; a KV slot then holds the keys then the values at their own widths), split into key partitions and merged, with an optional sliding window (a row sees its last `window` positions) and an optional sigmoid output gate, the sigmoid in FP32 (`SPARK_ATTENTION_GATE_SIGMOID`, Qwen's `LmOutputGateKernel`) or rounded to BF16 first (`SPARK_ATTENTION_GATE_SIGMOID_BF16`, Gemma's `GemmaSigmoidGateKernel`), read from the fused projection or from a separate gate projection; and optional per-head sinks (`has_sinks`), which the combine takes into the running maximum and adds as `exp(sink - max)` to the denominator, as CUDA's `LmGqaShardMergeLaunch` does; a key on a page `SparkKvPage` rejects raises the device error word and is skipped (decode per key, prefill per 64-key tile); decode walks 128-key partitions one key per SIMD group, prefill computes 32 (16 for 512-wide heads) query rows against 64-key pages on the tensor operations (scores and the FP32 probability-times-value product in `matmul2d`, the online softmax in FP32 between them); both online softmaxes rescale the running sum and accumulators at every new maximum, as CUDA's GQA softmax does |
| `spark_silu_mul`, `spark_gelu_tanh_mul`, `spark_silu_clamp_mul`, `spark_silu_gate_limit_mul`, `spark_add_rows` | the gated activations (`bf16(silu(g) u)`, `bf16(bf16(gelu_tanh(g)) u)` with `SparkMathGeluTanh`, `bf16(min(silu(g), L) clamp(u, -L, L))` with `SparkMathClampedGlu`, CUDA's `LmSiluMulLimitKernel`, and `bf16(silu(min(g, L)) clamp(u, -L, L))` with `SparkMathGateLimitedGlu`, CUDA's `LmSiluMulGateLimitKernel`) and the final residual add; `spark_situ_mul` is SiTU, `bf16(SparkMathSitu(g, u, limit, up_limit))`: `limit tanh(g / limit) sigmoid(g)` times `u`, or `up_limit tanh(u / up_limit)` when `up_limit` is set, shared with CUDA's `LmSituMulKernel` (Kimi K3's first layer and shared experts) |
| `spark_attnres_score`, `spark_attnres_mix` | attention residuals over a bank of block sums and the running partial (at most `SPARK_ATTNRES_MAX_SOURCES`, the partial last): each source's score is the FP32 dot of its RMS-normalised row with the site's BF16 query weight, then a softmax over the sources and the FP32 mix of the sources rounded once (Kimi K3's residual stream) |
| `spark_silu_mul_routed`, `spark_gelu_tanh_mul_routed`, `spark_silu_clamp_mul_routed`, `spark_silu_gate_limit_mul_routed` | the experts' gated activations with the route weight folded in before the down projection, as CUDA's `LmSwigluPackedKernel`, `LmGeluTanhPackedKernel` and `LingSiluClampPackedKernel` do: `bf16(silu(g) u w)`, `bf16(bf16(gelu_tanh(g)) u w)` with `SparkMathGeluTanhScaleFirst`, `bf16(min(silu(g), L) clamp(u, -L, L) w)`, and `bf16(silu(min(g, L)) clamp(u, -L, L) w)` (`LmSwigluPackedValue` with a finite limit: K-EXAONE's last 16 layers, HY4's experts) |
| `spark_gemv_mxfp4_routed` | each route's MXFP4 expert on its row (`SparkOpsLinearRoutedMxfp4`): E2M1 weights two per byte, low nibble first, with an E8M0 scale per 32 inputs, in CUDA's skinny arithmetic (`LmSkinnyFormat<LmMxfp4>`): each 32-weight chunk summed by a sequential FMA chain over `SparkMathE2m1ToFloat` in element order, then added to the FP32 accumulator by one FMA with `SparkMathUe8m0ToFloat` of its scale |
| `spark_route_topk`, `spark_route_pinned`, `spark_gemv_routed`, `spark_route_combine` | expert routing: softmax over the router's BF16 logits (a NaN or infinite logit raises the device error word), the top k in a total order (NaN below every probability, lowest index on ties) or, pinned, the experts a route override names, renormalised over a pairwise sum tree plus 1e-20 (the order of CUDA's warp top-k) and times a per-expert scale; the GEMV of each route's expert on its row (the same per-row arithmetic as `spark_gemv_r1`); and the routes' already weighted outputs summed in FP32 in route order and rounded once to BF16, as CUDA's `LmMoeLocalFinalizeKernel` does |
| `spark_head_gate` | a per-head scalar gate on a head's values, `bf16(v g)`: `g = bf16(softplus(z))` (`SparkMathSoftplusGate`, Laguna's `GemmaHeadSoftplusGateKernel`) or `g = bf16(sigmoid(z))` (latent attention's head gate, Ling's `LingHeadGateKernel`); with `gate_head_stride` one gate per element read from a row at a head stride and offset (A.X K2's query gate, `LmHeadSigmoidGateKernel`), and `SPARK_HEAD_GATE_SIGMOID` keeps the sigmoid unrounded, `bf16(v sigmoid(z))` |
| `spark_rmsnorm_bf16` | `bf16(bf16(rms(x)) w)` with a BF16 gain, the normed value rounded before the gain, as CUDA's `LmBf16RmsNormKernel` (MiMo V2.6) |
| `spark_route_sigmoid_topk` | sigmoid routing with a selection bias (`noaux_tc`), as CUDA's `LmTopkGroupLimitedKernel`: `sigmoid(z)` per expert; with `groups` above one, each group scored by the sum of its two best `sigmoid(z) + bias` and the best `topk_groups` groups kept (first group on ties); the top k of `sigmoid(z) + bias` in the kept groups picked one at a time (lowest index on ties), weights `sigmoid / (sum + 1e-20)` summed in pick order, or with `tree_sum` in the warp tree of CUDA's `LmTopkWarpKernel` (top 8: `((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))`, GLM 5.3 Flash), times the routed scale |
| `spark_silu_rounded_mul`, `spark_route_combine_weighted` | the gated SiLU with the activation rounded first, `bf16(bf16(silu(g)) u)`, as `MimoSwigluKernel`; and the routes' outputs times their weights, `fma(bf16(out), w, total)` in route order and rounded once, as `MimoMoeCombineKernel` |
| `spark_gemv_nvfp4_routed` | each route's NVFP4 expert on its row (`SparkOpsLinearRoutedNvfp4`): E2M1 weights two per byte, low nibble first, with a UE4M3 scale per 16 inputs times the expert's FP32 global scale, in CUDA's skinny arithmetic (`LmSkinnyFormat<LmNvfp4>`) and lane layout: 32 lanes per output row (16 when the input has fewer than 1,024 elements, 8 below 512), each lane taking 32-element chunks in ascending order, each chunk two 16-element FMA chains joined as `fma(s_hi, hi, fma(s_lo, lo, acc))`, the lanes summed by CUDA's xor butterfly; bit-for-bit CUDA's decode result |
| `spark_hyper_expand`, `spark_group_rmsnorm`, `spark_hyper_gate`, `spark_hyper_mix`, `spark_hyper_inject_weights`, `spark_hyper_inject`, `spark_hyper_key_gate`, `spark_hyper_add` | hyper-connection streams (Qwen 3.8 Flash), each CUDA kernel of `hyper_mix.cuh` with its per-element formula from `kernel_math.cuh` (`SparkMathHyperGate`, `SparkMathHyperMixGate` / `SparkMathHyperMixTerm`, `SparkMathHyperInjectWeight`, `SparkMathHyperInjected`, `SparkMathHyperKeyGate`, `SparkMathHyperAddend`): the residual copied into every stream, a per-stream RMS norm with an FP32 gain, the low-rank gate `bf16(v sigmoid(v))` of `v = bf16(x / 4)`, the stream mix `bf16(sum bf16(bf16(sigmoid(u)) n) / 4)` in stream order, the inject weights `bf16(2 bf16(sigmoid(bf16(bf16(n.w) / 4))))`, the injection `bf16(r + bf16(out w))`, the n-gram key gate and the n-gram residual add `bf16(r + bf16(g + c))` |
| `spark_hc_mix{8,4}`, `spark_hc_post` | HY4's hyper-connections (CUDA's `HyHcPreKernel`, `HyHcPostKernel`): per row, the flattened streams dotted with each mix's coefficients and with themselves in FP32, `pre_s = sigmoid(m_s r scale0 + base_s) + eps` and, with eight mixes, `post_s = 2 sigmoid(m_{4+s} r scale1 + base_{4+s}) + eps` (`r = SparkMathRmsScale` of the sum of squares), the branch input `bf16(sum_s pre_s x_s)` in stream order; four mixes collapse the streams before the head; the post update is `bf16(fma(post_s, out, x_s))`, one rounding |
| `spark_hc_site`, `spark_hc_post_comb`, `spark_stream_mean` | GLM 5.3 Flash's Sinkhorn hyper-connections (CUDA's `Glm5NextHcSiteKernel`, `LmHcPostBf16Kernel`, `LmStreamMeanRowsKernel`): per row 24 mixes, `pre_s = sigmoid(m_s r scale0 + base_s) + eps`, `post_s = 2 sigmoid(m_{4+s} r scale1 + base_{4+s})`, and a 4x4 combination from the other 16, softmax per row from its maximum plus eps, then `iterations` Sinkhorn passes (from the second pass a row normalisation, then a column normalisation, each `+ eps`) with sequential sums; the branch input `bf16(sum_s pre_s x_s)`; the post update `bf16(bf16(bf16(P) out) + bf16(sum_s fma(bf16(C_st), x_s)))` per stream `t`; `spark_stream_mean` collapses the streams as `bf16(sum_s x_s / 4)` before the head |
| `spark_sigmoid_gate_rows`, `spark_route_combine_shared` | the shared expert's gate, `bf16(v sigmoid(bf16 logit))` per row, as `QwenHybridSigmoidGateRowsKernel`; and the routes' outputs summed in FP32 in pick order with the shared expert's row added last, rounded once, as `LmMoeLocalFinalizeKernel` |
| `spark_ngram_hash`, `spark_ngram_gather_fp8`, `spark_ngram_history` | the n-gram embedding's ids, rows and carried tokens, as `LmNgramHashKernel`, `LmNgramGatherScaledKernel` and `LmNgramHistoryKernel`: each row's current and previous tokens (from the step, else the sequence's history), blocked from the first end-of-text token on, hashed by an int64 rolling XOR of token times multiplier with C's remainder per head; `bf16(e4m3 x scale)` table rows; the last tokens kept on a committed step |
| `spark_scale_rows`, `spark_softcap_rows` | a BF16 row scale (an embedding multiplier) and the FP32 logit softcap `cap tanh(m z / cap)` (`SparkMathSoftcap`) with an optional logit multiplier `m`, as CUDA's `LmStageSoftcapKernel` |
| `spark_logprob_rows` | argmax (lowest index on ties), log-normaliser and a target's log-probability over FP32 logits |
| `spark_block_qdq` | an activation's quantise-dequantise round trip in place or into another buffer, one block per thread, as DeepSeek V4.1 Flash's CUDA kernels do before every FP8 or FP4 consumer: FP8 per 32 values with the power-of-two scale `SparkMathFp8QdqScale` (`2^ceil(log2(amax / 448))` from the FP32 product's bits, amax floored at 1e-4), FP4 per 16 with an E4M3 scale `SparkMathFp4QdqScale` (`e4m3(amax / 6)`, amax floored at `6 * 2^-9`, the compressed keys), and FP4 per 32 with a power-of-two scale; each value `bf16(round(clamp(x / s)) s)` onto the E4M3 or E2M1 grid, ties to even (`SparkMathRoundE4m3`, `SparkMathRoundE2m1`) |
| `spark_route_sqrt_softplus_topk` | `spark_route_sigmoid_topk` with the score `sqrt(softplus(z))` (`SparkMathSqrtSoftplus`, CUDA's `LM_TOPK_SCORE_SQRT_SOFTPLUS`), selected by `SparkRouteParams.score` |
| `spark_hc_site_lagged`, `spark_hc_post_comb_f32`, `spark_hc_collapse` | DeepSeek V4.1 Flash's hyper-connections (CUDA's `D41HcSinkhornRowsKernel`, `D41HcPostRowsKernel`, `D41HcPreRowsKernel`): `spark_hc_site`'s mixes, but the branch input collapses the streams with the pre weights the previous site computed (a per-row FP32 buffer the kernel reads and then overwrites with its own); the post update in FP32 with one rounding, `bf16(fma(P_t, out, sum_s C_st x_s))`; the head input `bf16(sum_s pre_s x_s)` |
| `spark_stream_key_gate` | a value added to every stream through a signed square-root gate (DeepSeek's engram, CUDA's `SparkDsv41FlashEngramGateKernel`): per stream `d = (h . (q k) key) rsqrt(mean h^2 + e) rsqrt(mean key^2 + e) / sqrt(H)`, `g = sigmoid(copysign(sqrt(max(|d|, 1e-6)), d))` (`SparkMathSignedSqrtGate`), `h = bf16(h + g v)` |
| `spark_kv_compress`, `spark_kv_compress_save`, `spark_compressed_positions` | the compressed-key pool's input: a row whose position closes a group of `ratio` (one or two) pools its FP32 projections softmax-gated over the group (`SparkMathSoftmaxPool2`, CUDA's `D41CompressorRowsKernel`), the group's earlier row taken from the step or from the sequence's carried FP32 state, which a committed step's last open row saves; and per row the store index (or `SPARK_KV_SKIP`), the group's first position for its rope and the last compressed key it sees |
| `spark_kv_store_paged_skip`, `spark_attention_decode_extra_d512`, `spark_attention_combine_extra_d512` | the KV store that leaves a row whose position is `SPARK_KV_SKIP`; and 512-wide attention over two key ranges in one softmax: the window of the layer's own pages and a second paged range (`SparkOpsAttentionExtra`: the compressed keys up to each row's own last key), each decoded into its own partitions of one record buffer (`partition_base`), merged with the sinks by the combine (`extra_partitions`) |
| `spark_ngram_hash_mapped`, `spark_ngram_history_mapped`, `spark_ngram_gather_e8m0` | the n-gram kernels with a token map applied to every token (CUDA's `LmNgramToken`; a token mapped to the block id blocks), and table rows with an E8M0 scale per 32 values (`LmNgramGatherE8m0Kernel`) |
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
its cases, softplus, the head-gate softplus (`SparkMathSoftplusGate`), the
gated-delta decay, the bounded per-channel decay of the KDA layers
(`SparkMathBoundedDecay`), the clamped GLU (`SparkMathClampedGlu`), the tanh
GeLU, the softcap, the RMS scale and the rotary rotation
(`SparkMathRopeCos`, `SparkMathRopeSin`, `SparkMathRopeLow`,
`SparkMathRopeHigh`).
The experts' tanh GeLU is `SparkMathGeluTanhScaleFirst`, which multiplies
`0.044715 g g g` from the left as CUDA's expert kernels always have; the
dense GeLU cubes first. Both backends use the same two, so the choice
between them is made in one place.
Each is an inline function that the CUDA kernels (through `dtype.cuh`), the
Metal kernels (through `common.metal`) and the host CUDA shim of the host
tests include. A formula is defined once, so a change to it reaches both
backends.

The header also holds the weight-format decoders: E4M3, UE4M3 and UE8M0
(block scales) to FP32, and an E2M1 nibble to FP32. CUDA's
`LmUe4m3ToFloat`, `LmUe8m0ToFloat` and `LmE2m1ToFloatPure` call them. CUDA
decodes E4M3 and E2M1 data with the hardware conversions
(`cvt.rn.f16x2.e4m3x2`, `cvt.rn.f16x2.e2m1x2`); `test_bf16_convert_cuda`
checks every code of the shared decoders against those conversions, and
`test_metal_bf16_convert` checks every code on Metal against the formats.
A Metal projection over FP8 or FP4 weights therefore reads the same values
as the CUDA one.

DeepSeek V4.1 Flash's CUDA kernels take their block scales, the expert
score, the compressor's pair pool and the engram gate from the header too
(`SparkMathPow2CeilScale`, `SparkMathFp8QdqScale`, `SparkMathFp4Pow2QdqScale`,
`SparkMathSqrtSoftplus`, `SparkMathSoftmaxPool2`, `SparkMathSignedSqrtGate`);
the PTX of every CUDA stage model is byte-identical before and after (2,307
entries over 18 stage models). CUDA rounds onto the E4M3 and E2M1 grids with
the hardware conversions (`cvt.rn.satfinite`); `SparkMathRoundE4m3` and
`SparkMathRoundE2m1` are the Metal side's software twins (round to nearest
even by `rint` at the binade's step, saturating), checked in
`test_metal_kernels` against an enumerated grid.

- The header shares formulas, not schedules. Tiling, data movement, matrix
  units, reductions and launch shapes stay per backend, because that is
  where each backend's speed is.
- The macros at the top of the header pick each backend's intrinsics:
  bit casts, `exp`, `log`, `cos`, `sin` and `tanh`. A `Fast` variant is
  CUDA's `__expf`, `__logf`, `__cosf` and `__sinf`. Metal builds with
  `-fno-fast-math`, so both variants are the precise functions there.
  Metal has no `log1p`: `SparkMathLog1p` is `log1pf` on CUDA and
  `log(u) * (x / (u - 1))` with `u = 1 + x` on Metal, within a few FP32
  ulps of it.
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
- The runner has one execution lane; a model's `lane` call is refused.
- `round` is the collective of a tensor-parallel rank (below). A TP1 runner
  never sees one: `SparkDeviceStageAllReduce` returns at once on one rank.

## Tensor parallel ranks

A runner opened with `tp_degree` N > 1 is one rank of N. It reads its own
rank pack (the same `tp{N}.rank{NN}` packs the CUDA packers write), holds one
vocabulary slice of the embedding and head, and runs the model's collectives
through `round`, with the operations and meanings of
`spark_tp_device_collective.h`:

- The embedding gathers the rank's slice (`SparkOpsGatherSlice`: tokens of
  other ranks become +0 rows) and all-reduces, as CUDA's `StageRunnerEmbed`.
- The head projects the rank's vocabulary slice to FP32, all-gathers the
  slices and reorders them into whole rows (`SparkOpsRankRows`); every rank
  then holds the full logits, log-probabilities and argmax.
- A round moves bytes and computes nothing: the transport
  (`SparkDeviceCollective.exchange`) leaves every rank's contribution in the
  rank's lane buffer, rank-major, and a combine kernel in
  `inference/kernels/metal/collective.metal` reduces it. The sum is CUDA's
  mesh sum: an FP32 accumulator from +0 over ranks 0..N-1 in order and one
  rounding to BF16 (`SparkMathBf16ToFloat`, `SparkMathFloatToBf16`), so a
  reduction does not depend on the transport. ALL_GATHER and ALL_TO_ALL land
  in the destination directly; REDUCE_SCATTER sums the rank's segment;
  PAIR_GATHER is refused.
- Attention is head-parallel. The packers split the fused QKV rows evenly
  over the ranks, not by head, so each rank projects its rows, the ranks
  all-gather them (`SparkDeviceStageGatherRows`) and every rank stores keys and
  values for all KV heads. A rank attends with its own query heads
  (`first_head`, and `group` query heads per KV head, in
  `SparkAttentionParams`) and projects them through its `attn_out` columns;
  the partial output is all-reduced. Per head, the attention arithmetic is
  the TP1 arithmetic. CUDA's TP attention is context-parallel instead (each
  rank holds a slice of positions for every head and merges partial softmax
  records by all-to-all): its KV is 1/N per rank, Metal's is whole per rank.
- Experts: the router runs on every rank; `SparkOpsRouteLocal` renumbers the
  rank's experts from 0 and marks the rest `SPARK_ROUTE_SKIP`, for which the
  routed GEMVs write +0 rows, so the route-order combine sums only local
  experts, exactly like CUDA's `LmMoeLocalFinalizeKernel`. The dense MLP uses
  the rank's `ffn_rows`. Both partial outputs are all-reduced.

Each rank has its own queue and its own ops lane (`SparkDeviceOpsAcquireLane`:
the kernels are shared, the device error word and split-K partials are the
rank's), so ranks run concurrently on one device.

`SparkDeviceStageRunnerOpen` with `tp_degree` N and no collective opens an
in-process group: one device, N rank runners named from the rank 00 pack path,
a thread per rank, and `SparkDeviceCollectiveLocalCreate` as the transport
(each rank drains its queue, the ranks meet, each copies every rank's
contribution into its lanes and they meet again). The group answers the
ordinary runner calls (step, keep rows, reset), so `metal_stage_run --tp N`
runs every mode on N ranks of one Studio: a slow but exact model of N Studios,
whose only difference is the transport. A rank of a multi-Studio group passes
its own `SparkDeviceCollective` and `tp_rank`.

`SparkDeviceCollectiveTcpCreate` is that transport between processes: a full
mesh of TCP connections (rank r listens on base port + r and dials the lower
ranks), each round staged through shared memory, every message tagged with
its round and size so ranks that diverge fail at once instead of mixing
rounds. `metal_stage_run --tp N --rank R [--tp-hosts H0,...] [--tp-port P]`
runs one rank per process; the ranks run the same deterministic schedule
(SPMD) and only rank 0 writes outputs. N processes on one Studio over
loopback must reproduce the in-process group's logits bit for bit, and N
Studios over Thunderbolt IP run the same command with their addresses.
Speculation chooses draft lengths from measured timings, so its ranks can
diverge; it needs rank 0 to broadcast its choices before it runs as rank
processes.

### Packs larger than memory

A Metal command buffer makes every buffer it binds resident, so a pack mapped
as one buffer is pinned whole while any of its kernels run, and N rank packs
on one Studio need N packs of memory at once. With `weight_segment_bytes`
(`metal_stage_run --stream-weights`, 1 GiB) the runner maps the pack through
`SparkDeviceMapFileSegments`: one file mapping, with no Metal buffer over
it. Each binding, copy or fill on it wraps the pages it covers as a Metal
buffer of its own (about 2 us), which the command buffer's batch holds until
the GPU has finished with it, so a command buffer pins only the pages its
kernels bind and the operating system pages the rest of the pack in from the
SSD and out again as the layers go by. Residency is per buffer: on the M5
Ultra a kernel that reads one page of a 2 GiB file-backed buffer pages in all
2 GiB, so the earlier windows of two strides made a streamed decode step read
the whole pack (Solar Open 2 at TP8, 502 GB from the external SSD at about
4.6 GB/s, near two minutes per token). `SparkDeviceSegmented` tells the
layers a tensor is in such a mapping, and the routed experts then bind only
their routes' experts. The segmented allocation takes a device address of
its own outside the GPU's range, used only to resolve tensors. The host and
CUDA backends keep one mapping, which already covers every range.

## Device stage models

A device model is a shape, a tensor list and a layer schedule over two
model-neutral layers in `inference/runner/`. A model directory holds nothing
else, so every further model tests these interfaces: an operation a model
needs that they lack is added there, for every model, and the existing
models' gates are re-run to show their output is unchanged.

- `device_stage_model.{h,c}` (`SparkDeviceStage`) is the model's lifecycle. It
  opens the rank pack and checks its payload alignment and topology
  (`SparkDeviceStageOpen`) and its
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
  take the model pointer as it is. The payload starts at the header rounded up
  to the stage shape's alignment, so a pack whose manifest `format` declares
  another alignment is refused: opened at 256 bytes, a 128-byte GLM 5.3 Full
  pack read every tensor 128 bytes late and failed only at its last tensor.
- Recurrent state is a set of pools, each `[layer][slot]` of one width;
  `recurrent_copy` packs pool by pool, layer by layer, and `reset_slot` zeroes
  a slot in every pool.
- `device_layers.{h,c}` are the layer kinds, each a shape struct, a bind call
  that names the layer's pack tensors and a run call:

| Layer kind | Shape | Pack tensors (`layers.N.`) |
| --- | --- | --- |
| gated delta rule (`SparkDeviceGatedDelta`) | key and value heads and dimensions, convolution taps; recurrent pools 0 (FP32 state) and 1 (BF16 convolution window) | `gdn_qkvz`, `gdn_ba`, `gdn_conv`, `gdn_a_log`, `gdn_dt_bias`, `gdn_norm`, `gdn_out` |
| attention (`SparkDeviceAttention`) | heads, KV heads, head dimension, rotary span, base and the number of rotated pairs, eager or fused rotary rounding, scale; options: a sigmoid output gate (queries and gates interleaved per head in the fused projection, or a separate `attn_gate` projection), weightless or no query and key norms, a query multiplier, a value width other than the head width and a value scale, per-head sinks, a sliding window, a weightless value norm, and keys shared as values (the fused projection then holds queries and keys only) | `attn_qkv`, `attn_gate`, `attn_q_norm`, `attn_k_norm`, `sinks`, `attn_out` |
| gated FFN (`SparkDeviceGatedFfn`) | intermediate width and activation (SiLU, or tanh GeLU), applied by `spark_gated_mul` before the down projection | `ffn_gate_up`, `ffn_down` |
| experts (`SparkDeviceExperts`) | experts, top k, expert width, activation; the router reads the residual through `spark_rmsnorm_scaled`, the experts read it through their own norm; a step's route override replaces the top k (qualification only) | `router_proj`, `router_scale`, `expert_scale`, `pre_ff_norm_2`, `moe_w1` (gate), `moe_w3` (up), `moe_w2` (down) |
| residual RMSNorm (`SparkDeviceResidualNorm`) | the stage's hidden width and epsilon | the model names its norm weights |
| branch norms (`SparkDeviceBranchNorm`) | one or two branches, an optional layer scalar | the model names its norm weights |

A stage's KV pool holds the widest slot any of its attention layers needs
(`kv_heads x head_dimension` of the stage shape); a layer with narrower slots
uses its pages at that layer's own slot stride. A stage shape can tie the
head to the embedding table, scale and soft-cap the logits, which the runner
applies before the log-probabilities, give the post-branch norms their own
epsilon, and alias a range of token ids to one embedding row (the runner
gathers through its own row table, so the step's token ids are untouched).
Other norm placements and other activations have no option yet; the model
that needs one adds it here.

A device model is a family's Metal backend running the family schedule
([`FAMILY_SCHEDULES.md`](FAMILY_SCHEDULES.md)) on the model's TP1 geometry; the
schedule and the geometry are the ones the CUDA module runs.
`model-families/qwen38/device/qwen38_device_stage.cpp` is Qwen3.8-27B on these
layers: 64 layers, every fourth one (from the fourth) attention and the rest
gated delta, each followed by the SwiGLU FFN, with the attention and
gated-delta projections sharing one scratch slice. The recurrent state is FP32,
`[layer][slot][head][key][value]`, and `recurrent_copy` packs it in the same
order as the CUDA model.

The same backend runs Qwen3.8-27B's official FP8 checkpoint
(`tools/qwen38_27b_tp16_pack.py --codec fp8 --tp-degree 1`, 30.2 GB against
the BF16 pack's 54 GB). The layer kinds bind each projection through
`SparkDeviceStageLayerWeight`, which reads the tensor's kind from the pack: a
`bf16` tensor as before, or an `fp8_e4m3` tensor with its `<name>.scale`
sibling (an FP32 scale per row and 128 inputs, the checkpoint's 128 x 128
block scale repeated over its rows). `SparkDeviceLinearWeight` then runs FP8
weights on `SparkOpsLinearFp8`, with a dequantisation scratch the model
reserves when its pack is FP8 (`layers.0.ffn_gate_up` of kind `fp8_e4m3`).
The GDN gates and the head stay BF16, as in the checkpoint.
`tools/qwen38_27b_reference.c` reads the FP8 checkpoint by dequantising each
FP8 projection exactly (E4M3 times the BF16 block scale, held in FP32), so the
gate measures the Metal FP8 stage against the FP8 model, not the BF16 one.
Its gate passes (KL 0.78 and 1.14 times the floor against references A and
B; the all-layer and rotated controls fail at 26 to 68 standard errors;
the exact-rows ladder reproduces solo decode), and B1 decode runs at 27.1
tok/s against the BF16 pack's 18.1 (30.2 after the decode table below).
Between 3 and 16 rows the FP8 projection loops the GEMV in passes of eight
rows, which beats BF16 up to B4 but trails its tensor-operation row tiles at
B16 (TECHDEBT).

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

The same backend places Gemma 4 31B (dense, 60 layers) and Muse Glimmer 30B.
Muse is dense too: 52 layers, every fourth one (from the fourth) full
attention without rotary and the rest sliding-window attention (window 2,048,
the whole head rotated with base 5 x 10^5), all with 32 query and 2 KV heads
of 128, weightless q and k norms with the queries times 3.87, a separate
sigmoid output gate rounded to BF16, attention scale 1/sqrt(128); then the
SiLU FFN. The embedding goes through a weightless RMSNorm, token ids 200,091
and 200,092 read embedding row 0, the post-branch norms use epsilon 10^-8,
the head is its own table and the logits are multiplied by 0.196 and capped
at 20. Its pack is the same packer's TP1 output.

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
queue is refused. Each rank of a TP group has its own ops lane, so its own
partials.

Routed BF16 experts take the same split. Up to 64 rows (decode and verify
frames, and every row-exact step) each route is one GEMV over its expert, so
a row's result does not depend on the rows around it. A larger step groups
its routes by expert, CUDA's prefill shape (`LmLocalExpertsUp` /
`LmLocalExpertsDown`): the stage reads the step's routes back after one
queue wait, sorts them by local expert, gathers each expert's rows, runs one
projection per expert for gate, up and down (the row tiles or the GEMM by its
row count), folds the route weights in the sorted order and scatters the
outputs back into route order, skipped routes of other ranks taking a zero
row. A route's GEMV re-reads its expert's weights for every row, so before
the grouping MiMo V2.6 Flash prefilled at about 50 tokens per second; grouped,
each expert is read once per step. Experts in a segmented pack (below) take
the grouped path at every row count, so a step binds only the experts its
routes name; up to 64 rows, and in row-exact steps, the grouped projections
are row-exact GEMVs, the per-route GEMV's arithmetic for every output. The
FP4 experts (MiMo's MXFP4, Qwen 3.8 Flash's NVFP4) take the per-route GEMV
through `SparkDeviceRoutedFfn`, which takes BF16, MXFP4 or NVFP4 matrices
(`SparkDeviceExpertMatrix`). In a segmented pack it sorts the routes the same
way and runs the same routed kernel once per expert over that expert's
gathered rows, bound to the expert's own values and scales with an all-zero
route table, so every route's output is the per-route launch's and only the
routed experts are paged in.

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
rotation as `bf16(bf16(x bf16(cos)) + bf16(p bf16(sin)))` (the Metal stage
keeps it: CUDA's FP32 rotation took Gemma 4 26B's gate to paired +3.8 sigma
and left 31B outside at +3.5); the weightless
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
Accelerate's FP32 GEMM. It runs twice per prompt: variant A uses the BLAS
GEMM over the full inner dimension and forward sums; variant H splits every
GEMM's inner dimension into 1,024-wide chunks accumulated last chunk first,
reverses every sum (the order of the older variant B) and gives every exp and
tanh the error a GPU's has (`tools/reference_device_error.py`: on the M5 Ultra
under `-fno-fast-math`, exp is off by a relative 1.64e-7 mean and 4.35e-7
standard deviation, about 18 times FP32 rounding, and tanh by an absolute
3.7e-8; the error is a fixed function of the input, drawn from a hash of its
bits). KL divergence between A and H is the floor: FP32 reduction-order noise
plus the transcendental noise any GPU adds, under BF16 rounding. Every
reference (the Python ones too) computes A and H by default.

The floor used to be A against B, reduction order alone. It failed the
reference itself: the same numpy reference of Gemma 4 31B run on a second CPU
(lenovo7 against the hub, BLAS kernels the only difference) scored paired
+4.20 / +3.33 standard errors against it, about where Metal stood (+4.78 /
+4.32), so it could not pass a correct GPU implementation. A against H, both
computed on lenovo7, differs by a mean KL of 0.112, the same as the hub's and
lenovo7's A differ by; against it the same Metal run passes (KL 1.06 / 0.73
times the floor, top-1 1.05 / 0.98, paired +0.10 / -1.15), the hub's
reference passes (paired -0.32 / -1.24), and the all-layer and whole-tensor
controls still fail (+25.1 / +26.7 and +124 / +122 standard errors).

The gate rule is relative to that floor and runs in two legs, against A with
H as the floor and against H with A as the floor. A candidate passes a leg
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

### Gemma 4 26B A4B, Gemma 4 31B and Muse Glimmer 30B

`tools/metal_gates.sh MODEL PACK REFERENCE_DIR OUT REVISION` runs the
same gates for any of the family's Metal models on four prompts built by
`tools/gemma4_reference.py gate-prompts` from the same sources: the Gemma 4
models share a tokenizer and `qualification/metal/gemma4/` (5,636
positions); Muse Glimmer has its own tokenizer and chat template and
`qualification/metal/muse_glimmer/` (5,640 positions). Gemma 4 31B and Muse
are dense, so their gate is the Qwen gate's set: free decode against a with
b as the floor, and the three weight controls. The reference is
`tools/gemma4_reference.py teacher --variants a,b,bp` (a and b for the dense
models): numpy, the publisher's eager rounding, a and b in the two reduction
orders.

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
### Laguna S 2.1 on eight ranks

Laguna runs on the Gemma family schedule (its pre-norm path) and the Gemma
Metal backend, which covers the switches it sets: pre-norm blocks
(`AddRmsNorm`, `AddRows`), a per-head softplus gate before `attn_out`
(`spark_head_gate`: `bf16(v * bf16(softplus(g)))`, CUDA's
`GemmaHeadSoftplusGateKernel` through `SparkMathSoftplusGate`), YaRN on the
first 64 dimensions of the full layers (the inverse frequencies from
`SparkRopePlanBuildYarnInvFrequency`, cos and sin times the attention
factor), a dense SwiGLU in layer 0, and from layer 1 a sigmoid router
(`SparkDeviceSigmoidExperts`: BF16-rounded logits, top 10 on score plus
`expert_bias`, weights normalised and times 2.5) whose routed experts and
the shared expert are summed in route order and rounded once
(`spark_route_combine_shared`, CUDA's `LmMoeLocalFinalizeKernel` with the
shared rows). A layer's MLP width comes from its own `ffn_gate_up` (the dense
layer and the shared expert differ).

The BF16 checkpoint is 235 GB, so it runs as eight ranks
(`tools/laguna_tp16_pack.py --tp-degree 8`, which pads the sliding layers to
80 heads at every degree, as the geometry does) with the packs paged from
the SSD:

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh laguna-s-2.1 PACKS/laguna.tp8.rank00.pack REFERENCE_DIR OUT REVISION

The eight packs and their device state come to 258 GB on a 256 GB Studio,
and every step reads every layer in the same order, so the page cache
(about 166 GB beside 80 GB of wired memory) drops each page just before it
is needed again: a decode step pages in about 130 GB at 8.8 GB/s, 15 s per
token, while a 1,024-row prefill chunk reads the weights once for all its
rows. `SPARK_GATE_DECODE=N` (`metal_stage_run --decode-positions N`) has
the teacher passes decode only each prompt's last N positions and prefill
the rest, and `SPARK_GATE_GREEDY` shortens greedy decoding, so a gate takes
hours instead of days; the decode path is still exercised at every layer.

The gate prompts are `qualification/metal/laguna/` (the Gemma set rendered
with Laguna's full chat turn by `tools/laguna_reference.py gate-prompts`);
`tools/laguna_reference.py teacher` writes references a and b, b summing each
row's routed experts in descending instead of ascending expert order. With
`SPARK_GATE_TP` the gate runs the teacher, greedy and control passes on the
rank group and leaves out the timing ladders, the benchmark and
speculation. A control corrupts the tensor in every rank's pack, as a kernel
bug would reach every rank; corrupting rank 0's slice alone dilutes it by the
rank count before the all-reduce, which left GLM 5.3 Flash's every-layer KDA
control at 1.08 times the floor. `SPARK_GATE_CONTROL_DIR` puts the control
packs on the packs' volume, where `cp -c` clones them and only the corrupted
blocks take space.


### Ling 3.0 Flash on eight ranks

Ling 3.0 Flash runs on the Ling family schedule (`LingSchedule` in
`spark_ling_family.h`, the one CUDA steps through) with the Metal backend in
`model-families/ling/device/`. Every sixth layer is gated latent attention
(MLA), the others KDA, and layers 0 and 1 are dense. Two model-neutral device
layers serve them:

- `SparkDeviceChannelDelta` (a KDA layer): the fused `kda_qkv` projection,
  `kda_beta`, the decay logits `kda_f` and the output gate `kda_g` (each
  through a low-rank pair when the geometry has one, as Solar Open 2's), the
  causal convolution with swish over the fused rows, `spark_channel_delta_prepare`
  and the delta rule with one decay per key channel, then the per-head
  RMSNorm with a sigmoid gate (`spark_rmsnorm_gate`) and `kda_out`. The
  state and convolution windows are the two recurrent pools of the gated
  delta.
- `SparkDeviceLatentAttention` (an MLA layer), in CUDA's absorbed form:
  `mla_q`; `mla_kv_a` to the 576-wide slot (512 latent, 64 rope), the latent
  normed in place with the FP32 `mla_kv_norm`; each head's no-rope query
  through `mla_uk` written straight into its packed row; the query's and
  the slot's 64 rope dimensions rotated with interleaved pairs and theta
  6e6; the slot stored as the key with its latent as the value; attention
  at 576/512 with scale `1/sqrt(192)`; each head's latent output through
  `mla_uv`; the sigmoid head gate rounded to BF16 (`spark_head_gate`); and
  `mla_out`. The per-head projections are ordinary projections on strided
  slices, one per head. At TP the latent slot is replicated (`mla_kv_a` is
  in every rank's pack) and each rank attends its own heads, where CUDA
  shards the context.
  The shape's switches cover HY4's and the GLM variants: `query_lora` (`q_a`,
  `q_norm`, then `q_b`), `norm_bf16` (both norms with BF16 gains as
  `bf16(w bf16(x r))`, `spark_rmsnorm_bf16`; without it both take FP32 gains
  and round once, `bf16(x r w)`, as GLM 5.3 Full's CUDA projection norm
  does), `rope` 0 (a rope-free 512-wide latent that is both key and value,
  GLM 5.3 Flash), `query_concat` and `query_gate` (A.X K2: q_b reads the
  normed and raw query, and each head's query rows carry a gate for its
  values), `rope_factor` with its betas and original positions (YaRN through
  `SparkRopeYarnRampTable`), `head_gate` set to
  `SPARK_LATENT_GATE_ELEMENT` (a sigmoid gate per value element through
  `spark_head_gate` with one-wide heads), `sinks` (the attention's per-head
  sinks, sliced to the rank's heads) and `tensor_prefix` (`mla_` by default,
  none for HY4's names).

The experts are `SparkDeviceSigmoidExperts` with the router named `router`,
8 groups of 64 experts of which 4 stay open (`spark_route_sigmoid_topk`),
top 8 times 2.5, and the layer's SwiGLU limits from the geometry
(`RoutedLimit`, `SharedLimit`: 4 for the routed experts from layer 35, 5
and 7 for the shared expert from layers 34 and 40) through
`spark_silu_clamp_mul_routed` and `spark_silu_clamp_mul`. The shared expert
is computed first and joined in route order by `spark_route_combine_shared`.
The geometry's shared rows follow the TP degree (`768 / TP`, padded to a
multiple of 64: 64 rows at TP16 as before, 128 at TP8).

The BF16 checkpoint is 238 GB, so it runs as eight ranks
(`tools/ling3_tp16_pack.py --tp-degree 8`) paged from the SSD, as Laguna:

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh ling-3.0-flash PACKS/ling3_flash.tp8.rank00.pack REFERENCE_DIR OUT REVISION

The gate prompts are `qualification/metal/ling/` (the Gemma set in Ling's
chat turn with thinking off, by `tools/ling3_reference.py gate-prompts`).
`tools/ling3_reference.py teacher` writes references a and b in FP32: b
splits every matmul into 1,024-wide chunks summed last first, reverses the
norms', recurrent dot products' and softmax sums, and combines the routed
experts in descending order. The controls corrupt `kda_out`.

Ling 3.0 Flash Fin has Ling 3.0 Flash's configuration and tokenizer with its
own weights, as CUDA builds it from the same source with its own contract
(`LING3_FIN`), so `Ling3FinDevice` places `Ling3Geometry` under its own name
and tag, and its gate uses Ling's prompts (the chat templates differ only in
how they spell the thinking switch):

    tools/ling3_tp16_pack.py --model ling3_fin --tp-degree 8 ...
    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh ling-3.0-flash-fin PACKS/ling3_fin.tp8.rank00.pack REFERENCE_DIR OUT REVISION

### Solar Open 2 on eight ranks

Solar Open 2 is the Ling family schedule's GQA layout, and the Ling Metal
backend serves it (`SolarOpen2Device`). Every fourth layer from 0 is NoPE
grouped-query attention, through `SparkDeviceAttention` with no rope, no
query or key norm, a separate gate projection whose sigmoid multiplies the
output element by element (CUDA's `LmOutputGateKernel`), and the pack's
`gqa_*` tensor names (`tensor_prefix`). The other layers are KDA with
low-rank decay and gate projections (rank 128) and a doubled write strength
(`2 bf16(sigmoid(beta))`). The query and key L2 norm keeps CUDA's fixed
`1e-6` while the gate norm takes the model's `1e-5` (`l2_epsilon`). Each
rank holds every one of the 320 experts, split by their intermediate width
(`tensor_parallel` in the experts shape: no routes are skipped, and the
ranks' partial sums meet in the all-reduce). The geometry's expert and shared
widths follow the TP degree (80 and 128 at TP16 as before, 160 and 192 at
TP8), and `tools/solar_open2_tp16_pack.py --tp-degree` writes the packs.
`tools/solar_open2_reference.py` gains the Ling reference's `gate-prompts` and
FP32 `teacher` modes:

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh solar-open-2 PACKS/solar_open2.tp8.rank00.pack REFERENCE_DIR OUT REVISION

### GLM 5.3 Flash on eight ranks

GLM 5.3 Flash (45 layers of 4,096: 34 KDA layers and every fourth layer
rope-free latent attention, four Sinkhorn hyper-connection streams, 288 FP8
experts) runs on `Glm5NextSchedule` through `Glm5NextDevice`
(`model-families/glm5_next/device/`): `spark_hyper_expand` copies the
embedding into the four streams; before each branch `spark_hc_site` makes the
branch input and the post and combination weights, and `spark_rmsnorm_bf16`
norms it; the branch is `SparkDeviceChannelDelta` with the rounded write
strength and the FP32-gain gated norm (`beta_mode` rounded, `norm_f32`), or
`SparkDeviceLatentAttention` without rope (query LoRA 1,536, BF16 norms, the
512-wide latent as both key and value through the d512 attention kernel,
scale 1/16), or the dense layers 0-2, or the experts: the BF16 router in
FP32, sigmoid top 8 of 288 over the bias-corrected scores with
`LmTopkWarpKernel`'s tree sum, times 2.5, `SparkDeviceRoutedFfn` on the FP8
experts (`spark_gemv_fp8_routed`, FP32 scale per row and 128 inputs) with the
gate-limited SwiGLU at 10 and no route weight, `spark_route_combine_weighted`,
and the shared expert added (`spark_add_rows`), as CUDA's finalize and add;
every rank holds all 288 experts split by intermediate width. After each
all-reduce `spark_hc_post_comb` updates the streams, and `spark_stream_mean`
and the final BF16 norm feed the prenormed head. The indexer is not built:
a row that sees more than 2,048 keys is refused, and below that every pool
is selected and attention is dense. The FP8 checkpoint is 309 GB, so it runs
as eight ranks paged from the SSD (`tools/glm53flash_tp16_pack.py
--tp-degree 8`, which reads the checkpoint through the CUDA stagepack's
reader, so the BF16 projections are its `bf16(e4m3 x scale)`); the reference
is `tools/glm53flash_reference.py` (`gate-prompts`, and `teacher` through
the Ling reference's FP32 variants):

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh glm-5.3-flash PACKS/glm53flash.tp8.rank00.pack REFERENCE_DIR OUT REVISION

### K-EXAONE 2.0 on eight ranks

K-EXAONE 2.0 (78 layers of 6,144, BF16) is the Gemma family schedule's
pre-norm layout, and `GemmaDevice` places it as `Kexaone2Device`
(`SparkKexaone2DeviceStageModelAt`, TP 1, 2, 4 or 8). Its geometry has a
layer table (`kLayerTable`): layer 0 attends globally, layer 1 through a
4,096-key window, every fourth layer from layer 5 globally and the rest
through the 128-key window, so the backend takes each layer's window from
`L::Window` rather than from the full or sliding shape; and its last 16
layers limit the SwiGLU at 7 (`L::SwigluLimit`), so those layers' dense and
expert activations are `spark_silu_gate_limit_mul(_routed)`. The router
logits stay FP32 (`kRouterFloat`, no BF16 rounding before the sigmoid), and
every rank holds all 256 experts split by intermediate width
(`tensor_parallel`). The checkpoint is 1.4 TB, so it runs as eight ranks
paged from the SSD (`tools/kexaone2_tp16_pack.py --tp-degree 8`);
`tools/kexaone2_reference.py` gains `gate-prompts` (the full chat turn) and
`teacher` (FP32 products; variant b adds the routed experts in descending
order):

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh k-exaone-2.0 PACKS/kexaone2.tp8.rank00.pack REFERENCE_DIR OUT REVISION

### HY4 on eight ranks

HY4 (Hy4 preview, 78 layers of 6,144, MXFP8 weights) runs on its family
schedule through `HyDevice` (`model-families/hy4/device/`): the embedding
expanded into four hyper-connection streams (`spark_hyper_expand`); before
each branch `spark_hc_mix8` mixes the streams with the layer's pre and post
weights and `spark_rmsnorm_bf16` norms the branch input; the branch is
`SparkDeviceLatentAttention` with HY4's switches (query LoRA 2,048, nope 192
plus rope 64 at theta 1e7, value 256, scale `1/16`, elementwise gate, sinks)
or the dense layer 0 or the experts: the BF16 router in FP32, sigmoid top 8
times 2.827 over the bias-corrected scores (`spark_route_sigmoid_topk`),
`SparkDeviceRoutedFfn` on the MXFP8 experts with the gate-limited SwiGLU at
10 and the route weight before the down projection, and the shared expert
joined in route order (`spark_route_combine_shared`); `spark_hc_post` adds
the branch to each stream with its post weight, and `spark_hc_mix4` collapses
the streams before the head norm and the head. The sparse-attention indexer is
not built: a row that sees more than the indexer's 2,048 keys is refused,
and below that the indexer keeps every key, so attention is exactly dense. The
FP8 checkpoint is 766 GB, so it runs as eight ranks paged from the SSD
(`tools/hy4_tp16_pack.py --tp-degree 8`); `tools/hy4_reference.py` gains the
`gate-prompts` and `teacher` modes (variant b sums every product in
1,024-wide chunks last first, reverses the norms' and softmax sums and adds
the experts in descending order):

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh hy4-preview PACKS/hy4.tp8.rank00.pack REFERENCE_DIR OUT REVISION

### MiMo V2.6 Pro on eight ranks

`MimoDevice` places either MiMo geometry at TP 1, 2, 4 or 8
(`SparkMimo26FlashDeviceStageModelAt`, `SparkMimo26ProDeviceStageModelAt`):
each rank projects its rows of the fused FP8 QKV, which the shared attention
gathers before its heads run (the SWA layers' sinks sliced to them), holds
its slice of the dense layer and the vocabulary and `E / TP` whole experts,
maps the routes to its experts (`spark_route_local`, the others skipped) and
all-reduces the attention and FFN outputs. Pro is Flash's schedule with 70
layers of 6,144, 128 heads, 8 KV heads in both attention kinds, 384 experts,
a value scale of 0.612 and an epsilon of 1e-5. Its checkpoint is 566 GB, so
it runs as eight ranks paged from the SSD (`tools/mimo26_tp16_pack.py
--model pro --tp-degree 8`), against the MiMo gate prompts (the same
tokenizer and chat template):

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh mimo-v2.6-pro PACKS/mimo26_pro.tp8.rank00.pack REFERENCE_DIR OUT REVISION

Pro's single-layer controls corrupt layer 7, a full-attention layer, where
Flash's corrupt layer 11. On the gate prompts every head of Pro's SWA layers
1 and 11 puts its softmax mass on the sink: the keys carry a constant
component the query is anti-aligned with (scores of -60 to -90 against sinks
near 0.6), so those layers' `attn_out` reaches the residual only as values far
below a BF16 step, and a corrupted layer-11 `attn_out` leaves the logits
byte-identical. Shifting the layer's query rows by one breaks that alignment
and moves the logits (mean KL 4.1), and rotating layer 7's `attn_out` moves
them by a mean KL of 0.56.

### GLM 5.3 on eight ranks

GLM 5.3 (Full: 78 layers of 6,144, latent attention on every layer, 256 FP8
experts) runs on `GlmSchedule` through `Glm52Device`
(`model-families/glm52/device/`) as `Glm53FullDevice`, whose geometry is the
shared `SPARK_GLM52_MODEL_*` constants. The schedule's single wave is the
whole step: each layer's attention norms the residual sum
(`SparkDeviceResidualNorm`, FP32 gains, the previous layer's MLP partial
added) and runs `SparkDeviceLatentAttention` with GLM's switches (query LoRA
2,048, both projection norms with FP32 gains rounded once as CUDA's
`GlmProjectionNormKernel`, nope 192 plus interleaved rope 64 at theta 8e6,
value 256, scale 1/16); the MLP norms the residual plus the reduced
attention output, then runs the dense layers 0-2 or the experts: the BF16
router in FP32, sigmoid top 8 of 256 with the tree sum, times 2.5, the FP8
experts with a plain SiLU and no route weight, `spark_route_combine_weighted`
and the shared expert added, as CUDA's finalize and add. The projections stay
FP8 (`fp8_e4m3` with an FP32 scale per row and 128 inputs, CUDA's skinny FP8
arithmetic), and the last layer's partial is added and the head norm taken
in the stage (`prenormed_head`) from the FP32 sum, as CUDA's closing
`LmFusedResidualRmsNormKernel` does. The indexer is not built,
so a row past position 2,047 is refused. The 704 GB checkpoint packs to about
780 GB at TP8 (`tools/glm52_tp16_pack.py --model glm53full --tp-degree 8`,
through the CUDA stagepack's reader), paged from the SSD;
`tools/glm52_reference.py --model glm53full` gives the gate prompts in GLM
5.3's own chat turn and the FP32 teacher:

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh glm-5.3-full PACKS/glm53full.tp8.rank00.pack REFERENCE_DIR OUT REVISION

A.X K2 (61 layers of 7,168) is the same backend as `Axk2Device`, whose
geometry is `Axk2Dimensions` in the CUDA module's header (the CUDA
`Axk2Geometry` takes its values from there). It switches on what GLM 5.3
does not use: the rank-16 norm gate after the input norm of every layer and
the post norm of the routed layers (`SparkDeviceLowRankGate`: the down
projection rounded, `bf16(silu(.))` through `spark_head_gate` gating the
bottleneck by itself, the up projection rounded, then `bf16(x sigmoid(g))`
with the sigmoid unrounded, as `LmLowRankGateDownKernel` and
`LmLowRankGateApplyKernel`); the latent attention's `query_concat` (q_b reads
the normed query and the raw q_a output side by side, 3,072 wide) and
`query_gate` (each head's q_b rows end in 128 gate values, applied to the
head's values as `bf16(v sigmoid(g))` after the value projection, as
`LmHeadSigmoidGateKernel`); YaRN rope through `SparkRopeYarnRampTable`, the
CUDA stage's own table builder moved into `spark_rope_plan.h`; and routing
over 8 groups of which 4 stay open. Layer 0 is dense. The FP8 checkpoint is
647 GB (`tools/glm52_tp16_pack.py --model axk2`, `tools/glm52_reference.py
--model axk2` with its ChatML turn):

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh ax-k2 PACKS/axk2.tp8.rank00.pack REFERENCE_DIR OUT REVISION

### DeepSeek V4.1 Flash on eight ranks

DeepSeek V4.1 Flash (40 layers of 5,120, four Sinkhorn hyper-connection
streams, 64 heads of 512 over one shared key that is also the value, 384
MXFP4 experts, two engram modules) runs on its family schedule
`D41Schedule` (`spark_dsv41_flash_family.h`, the one `D41Cuda` steps
through) with the Metal backend `D41Device` in `model-families/dsv41_flash/device/`
at TP 1, 2, 4 or 8 (`SparkDsv41FlashDeviceStageModelAt`). The CUDA backend's
tail rows, landing caches and context-sharded records are its way of running
a step; the Metal backend's `LayerRows`, `Tap` and `LayerEnd` do nothing.

- Streams: `spark_hyper_expand` copies the embedding into four streams. Each
  site's `spark_hc_site_lagged` mixes the streams and collapses them with
  the previous site's pre weights (the first site of a step with
  `[1, 0, 0, 0]`, the head with the last FFN site's), as the publisher's
  `model.py` and the CUDA stage do; the branch input is normed with the FP32
  gain and rounded once (`spark_rmsnorm_residual`) and quantised for the FP8
  and FP4 consumers (`spark_block_qdq`); `spark_hc_post_comb_f32` adds the
  reduced branch back, and `spark_hc_collapse` and the final norm feed the
  prenormed head.
- Attention: q_a, the q norm, FP8 round trip, q_b (the rank's heads); kv, its
  norm, the rope of the last 64 dimensions (interleaved pairs, theta 10,000
  on layers 0 and 1, the compressed YaRN table elsewhere, through
  `spark_head_rmsnorm_rope` on a 64-wide view) and the FP8 round trip, stored
  as the layer's window key. Layers 2, 8, 14 and 20 compress: the FP32
  projections of the unquantised input, `spark_kv_compress` (pairs on layers
  2 to 19, every token from layer 20), the norm, the rope at the group's
  first position and the FP4 round trip, stored at the compressed index in
  the cache's own KV layer (40 window layers, then 4 caches; layers 2-7 read
  layer 2's, ..., 20-39 layer 20's). `SparkOpsAttentionExtra` attends the
  last 128 window keys and every visible compressed key in one softmax with
  the per-head sinks (layers 0 and 1: the window alone, `SparkOpsAttention`),
  the output tail is rotated back, rounded through FP8, projected per output
  group (`o_a`, a rank holding 8/T groups), rounded again and projected by
  the rank's `o_b` columns before the all-reduce.
- MoE: the BF16 router in FP32, top 6 of `sqrt(softplus(z)) + bias` with the
  warp tree sum times 1.5 (`spark_route_sqrt_softplus_topk`), the rank's 384/T
  experts (`spark_route_local`) through `SparkDeviceRoutedFfnQdq` (MXFP4, the
  gate-limited SwiGLU at 10 with the route weight before the down
  projection, the activation rounded through FP8), the shared expert's rank
  slice the same way, `spark_route_combine_shared` and the all-reduce.
- Engram (layers 1 and 14): `Begin` hashes both modules' 24 ids per row with
  the token map (`spark_ngram_hash_mapped`), waits for the queue and copies
  the rows from `dsv41_flash.engram.pack` beside the rank pack (203 GB,
  mapped on the host, never bound) into a staging buffer; the layer gathers
  them (`spark_ngram_gather_e8m0`), rounds through FP8, projects through the
  rank's `engram_kv` rows, gathers the ranks' rows and gates every stream
  (`spark_stream_key_gate`). `End` keeps the last three mapped tokens.

The sparse-attention indexer is not built. A row at position 512 or later
is refused: from there the ratio-1 layers see more than the indexer's 512
compressed keys, and below it the indexer keeps every key, so attention is
exactly dense. The gate prompts are therefore 496 tokens.

`tools/dsv41_flash_tp16_pack.py --tp-degree 8` writes the rank packs (about
38 GB each: the official FP8 bytes with the 32 x 32 UE8M0 block scales
repeated per row, the MXFP4 expert planes, the hash tables) and the engram
file; `tools/dsv41_flash_reference.py` gives the gate prompts in the model's
chat turn and the FP32 teacher:

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh deepseek-v4.1-flash PACKS/dsv41_flash.tp8.rank00.pack REFERENCE_DIR OUT REVISION

The FP8 and FP4 round trips make the floor large: references a and h agree
on the top-1 token at 88 to 96% of the positions and differ by a mean KL of
0.051 (a and b: 0.048). Metal is 1.10 and 0.98 times that KL from a and h,
at +2.0 and +1.2 sigma. The controls corrupt `kv`, whose rows feed every
head's keys and values: swapping rows 0 and 1 in every layer fails at 1.64
and 1.60 times the floor and +8.0 and +15.5 sigma, rotating layer 11's rows
at 2.53 and 2.35 times and +20.7 and +23.6 sigma. The same swap in layer
11 alone stays at +2.4 and +1.1 sigma, and in every layer's `o_b`, which
moves two of the 5,120 hidden channels, it stayed within 1.2 times the a
and b floor's KL and +3.1 sigma: as close to the floor as Metal itself.

### Motif 3 on eight ranks

Motif 3 (53 layers of 4,096, four mHC streams, 80 query heads in 16 groups
of four signal heads and one noise head over a 512-wide latent, 384 BF16
experts) runs on its family schedule `MotifSchedule`
(`spark_motif_family.h`, the one the CUDA `MotifStage` steps through) with
the Metal backend `MotifDevice` in `model-families/motif/device/` at TP 1, 2,
4 or 8 (`SparkMotif3DeviceStageModelAt`).

- Streams: `spark_hyper_expand` copies the embedding into four streams. Each
  site runs `spark_hc_site_clamped`, the mHC site with Motif's policy as
  CUDA's `MotifHcPolicy`: sigmoid pre and post gates of the RMS-scaled mixes
  clamped at +-10, the exp of the residual logits clamped at +-20 and 20
  Sinkhorn passes (rows, then columns) divided by sums floored at 1e-8; the
  combination is stored target-major, so `spark_hc_post_comb_f32` applies it
  as `LmHcPostF32Kernel` does (`bf16(post x out + sum_s comb_ts x_s)` in
  FP32). The branch input is normed with the FP32 gain
  (`spark_rmsnorm_residual`).
- Attention: `SparkDeviceLatentAttention` with Motif's switches. q_a, its FP32
  norm, q_b (the rank's five-head groups, 192 wide: nope 128 and rope 64);
  kv_a, its FP32 norm on the latent; rotate-half rope (`rope_half`) on the
  query's and the slot's last 64, with YaRN (factor 64 over 4,096) on the
  full layers (layer % 4 == 0) and plain theta 1e4 on the others, which see
  the last 129 keys (`window`); scales 0.1447 and 0.0722. The values pass
  `spark_differential_gate` (`differential`, CUDA's
  `LmDifferentialGateKernel`): each group's signal heads minus
  `bf16(sigmoid(lambda))` times its noise head, times
  `bf16(sigmoid(gate))`, where lambda is the replicated 64-wide projection of
  the normed input and the gate the rank's signal rows projected from the
  normed query LoRA; `wo` reads the signal heads only.
- MLPs: the dense layers 0 and 1 and the shared expert run on rank
  layer % T only (the others contribute zeros), gate and up in one
  projection, then `spark_poly_norm_mul` (CUDA's `LmPolyNormMulKernel`):
  `poly = sigmoid(w0) g^3/d3 + sigmoid(w1) g^2/d2 + sigmoid(w2) g/d1 + b`
  with `d_k` the RMS of the gate powers, times up, rounded, then halved. The
  BF16 router in FP32, sigmoid top 8 of `score + bias`, the unbiased scores
  normalised by their sequential sum (+1e-20) times 2 as
  `LmTopkGroupLimitedKernel`; the rank's 384/T experts
  (`SparkDeviceRoutedFfnPoly`) project gate and up from the fused `moe_w13`,
  apply the routed PolyNorm (bias clamped at +-0.5, the product clamped at
  +-1e6 and halved in FP32) with each expert's parameters, and project down;
  `spark_route_combine_weighted_shared` adds the weighted outputs and the
  shared expert in FP32 and rounds once, as `LmMoeLocalWeightedFinalizeKernel`.
- The end takes the mean of the streams and the head norm with its FP32 gain
  (`prenormed_head`).

CUDA shards the latent attention by context (gathered queries, partial
records and an all-to-all merge); the Metal ranks keep the whole latent and
attend their own heads, as for GLM 5.3. The window layers' inverse
frequencies come from the host in double precision, where CUDA's
`LmRopePerHeadKernel` takes `__powf` per pair.

`tools/motif3_tp16_pack.py --tp-degree 8` writes the BF16 rank packs (about
74 GB each); `tools/motif3_reference.py` gives the gate prompts in Motif's
chat turn (`gate-prompts`) and the FP32 teacher:

    SPARK_GATE_TP=8 SPARK_GATE_STREAM=1 SPARK_GATE_DECODE=16 SPARK_GATE_GREEDY=8 \
        sh tools/metal_gates.sh motif-3 PACKS/motif3.tp8.rank00.pack REFERENCE_DIR OUT REVISION
