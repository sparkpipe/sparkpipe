# GLM-5.2 FP8 support tracking — 2026-07-03

This note tracks two separate queues:

1. Sparkpipe's current top performance items, ranked by estimated speed impact.
2. CUDA code still needed for complete FP8 support.

The attached vLLM tree was used as implementation guidance. The most relevant patterns are its SM120 blockwise FP8 scaled-MEMM kernels under `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/`, its FP8 per-token/group quantization kernels under `csrc/quantization/w8a8/fp8/`, and its FP8 KV-cache/attention dtype plumbing under `csrc/attention/dtype_fp8.cuh`.

## Current top 10 performance items, ranked by estimated speed gain

1. **DSpark speculative decode backend binding** — estimated 2.0x–2.4x for realtime and underfilled decode, based on the DSpark GLM-5.2 acceptance evidence. Runtime/scheduler plumbing exists; the 5-layer draft backend and multi-token verify execution remain.
2. **FlashAttention-style paged/chunked prefill** — estimated 2x+ on long uncached prompts once the runtime KV block tables feed a tiled prefill kernel. Current tiled path exists but still needs Spark2 tuning.
3. **Full FP8 regular linears plus FP8 MoE experts** — estimated 1.3x–2.0x on compute-heavy stages if Q/KV/O, dense FFN, and routed expert GEMMs all run on real FP8 tensor-core paths.
4. **Spark2 async JIT KV prefetch backend** — estimated large workload-level win when queues contain cold external KV entries; scheduler and hotset policy exist, but the real 13-lane Spark2/NVMe/fabric backend remains.
5. **Exact PP13 six-layer AOT stage replay** — estimated 1.05x–1.2x by replacing six per-layer graph replays/submits with one exact-stage replay where the previous validator still showed six replays.
6. **B12x deterministic route-slice finalize validation/tuning** — estimated accuracy-enabling first, then performance recovery versus the failed single-route wrapper. The route-slice source integration exists; Spark2 must regenerate AOT and measure.
7. **Persistent hidden transport on Spark2 fabric** — estimated 1.05x–1.2x when PP13 hidden handoff becomes a bottleneck. Execution callbacks exist; the actual fabric backend is still required.
8. **Native/SM121-supported Q/KV/O projection tuning** — estimated 1.1x–1.4x over the current conservative WMMA path once Codex dev confirms the best supported SM121 formulation.
9. **Adaptive partial-B64 decode bucket validation** — estimated 1.2x for active decode batches in the B17–B32 range if B64 remains faster on Spark2 partial-fill graphs.
10. **Trace-tuned Centaur/bulk scheduling policy** — estimated workload-dependent; now that priority, prefix families, JIT prefetch, and DSpark hooks exist, the remaining gain is from measured policy tuning.

## CUDA code needed for proper FP8 support

1. **FP8 activation quantization for dense regular linears** — implemented in this pass as `SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationQuantize`. It writes E4M3 activations plus per-row/per-block F32 dynamic scales.
2. **FP8 dense MLP plan binding** — implemented in this pass. The resident linear-plan binder can now create FP8 tensor-core plans for dense gate/up/down weights, not only Q/KV/O projections.
3. **FP8 KV-cache store/load helpers** — implemented as E4M3 store/load CUDA entry points with F32 scales and strict resident plan validation. The follow-up CUDA pass also adds slot-mapped current-token FP8 KV page writes from the BF16 shadow cache.
4. **SM120/SM121 CUTLASS or cuBLASLt FP8 regular-linear backend** — still needed for peak performance. Sparkpipe now has the plan surface and WMMA-supported path, but the best production path should mirror vLLM's supported scaled-MEMM approach instead of fragile inline PTX.
5. **FP8 MoE grouped GEMM backend** — still needed. GLM-5.2 is mostly routed; full FP8 is incomplete until expert FC1/FC2 use an FP8 grouped GEMM and deterministic route finalize. The resident FP8 MoE plan ABI now fails closed on the required GLM-5.2 FP8 metadata: up/gate ordering, expert-major row-major weights, expert-major row-block scales, E4M3 quant mode, scale block size 128, aligned payloads, aligned scales, aligned workspace, tensor-core capability, and validated latency.
6. **FP8 KV attention consumption** — decode attention and paged prefill attention now have strict FP8-KV kernels that read MLA/key-nope/value E4M3 pages plus F32 scale blocks when `SPARK_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FP8_KV_CACHE` is set. The remaining work is Spark2 measurement and deciding whether any long-prefill bucket should stage dequantized tiles instead of direct page reads.
7. **Checkpoint scale-layout loader normalization** — partly implemented for the FP8 resident MoE packer. The packer now has a strict runtime row-block-major F32 scale view, optional transposed-scale normalization for checkpoint variants, lazy dependency loading, and a layout unit test. Remaining work is wiring the generated pack metadata into the runtime C/CUDA FP8 MoE plan loader for every layer.
8. **Fused RMSNorm + FP8 activation quantization** — public CUDA helper added for BF16 RMSNorm output plus E4M3 dynamic-scale rows. The remaining work is wiring it into production GEMM staging and measuring whether it should replace the existing standalone norm in each graph bucket.
9. **Fused SiLU/gate activation + FP8 quantization for dense and MoE FC2 input** — public CUDA helper added for BF16 SiLU(gate)*up output plus E4M3 dynamic-scale rows. The remaining work is attaching the helper to the FP8 MoE/grouped-GEMM path and route finalize.
10. **Strict FP8 validation harness on Spark2** — partly implemented in C/runtime validation. FP8 KV and FP8 MoE plans now fail closed on missing payloads/scales, scale block size, layout metadata, dtype, backend capabilities, alignment, and validated latency. Remaining work is CUDA-device numerical validation on Spark2 across decode and prefill buckets.

## This pass

This pass and the follow-up CUDA pass add connected support for the first three CUDA items above, plus direct decode/paged-prefill FP8-KV consumption and fused activation helpers:

- Dense FFN gate/up/down FP8 weight-plan fields and binder support.
- CUDA E4M3 activation quantization helper with dynamic F32 scales.
- CUDA E4M3 KV cache store/load helpers, slot-mapped current-token FP8 page writes, and an execution flag requiring a validated FP8 KV cache plan.
- Firmware validation proving FP8 dense MLP plans can replace dense BF16 weights, and malformed dense FP8 scale metadata is rejected.
- Firmware validation proving production FP8 KV-cache mode rejects missing scale/payload buffers.
- Decode and paged-prefill attention kernels that consume FP8 MLA/key-nope/value KV pages directly under the strict FP8 KV execution flag.
- Fused RMSNorm+FP8 and SiLU-multiply+FP8 CUDA entry points for the next scaled-MEMM and MoE wiring pass.
- CUDA graph signatures now include FP8 KV plan fields and payload/scale pointers, preventing graph replay across incompatible FP8 cache layouts.

CUDA hardware compilation was not available in this container, so Spark2 still needs to compile/debug the `.cu` entry points.

## This iteration

This iteration hardens the FP8 production contract instead of adding another permissive launch path:

- Bumped the resident FP8 MoE plan ABI and added explicit GLM-5.2 FP8 metadata fields: up/gate order, weight layout, scale layout, quant mode, and scale block size.
- Added C/runtime validation and CUDA-side validation for FP8 MoE layout fields, E4M3 quant mode, scale block size 128, payload alignment, scale alignment, optional workspace alignment, required capabilities, and validated latency.
- Extended firmware tests so malformed FP8 MoE scale layout, scale block size, weight alignment, workspace alignment, and missing launch metadata are rejected.
- Added FP8 resident MoE packer scale-layout helpers that validate runtime row-block-major F32 scale bytes and can normalize transposed checkpoint scale bytes before pack writing.
- Made the FP8 packer lazy-load torch/safetensors so layout-only tests do not require heavyweight checkpoint dependencies.
- Added a Python layout test for FP8 pack row-block-major and transposed scale conversion.

## This CUDA iteration

This iteration adds a concrete debug-distance FP8 MoE execution path on top of the stricter FP8 production contract:

- Added `SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoeGroupedReference`, a deterministic CUDA reference path for routed FP8 experts. It consumes router logits, computes top-k routing, reads E4M3 expert W1/W2 payloads with F32 row-block scales, stages gate/up/intermediate BF16 buffers, writes BF16 route output, and then uses the existing residual path.
- Added `SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceBytes` and `SparkGlm52Sm121RequiredDecodeStageBindFp8MoeGroupedReferencePlan` so Codex can bind a complete FP8 MoE plan before replacing the reference kernels with grouped tensor-core GEMM.
- The reference path honors the strict GLM-5.2 metadata added in the previous iteration: up-then-gate row order, expert-major row-major weights, expert-major row-block-major scales, E4M3 quant mode, scale block size 128, aligned payloads/scales, aligned workspace, and required backend capabilities.
- CUDA graph signatures now mix the FP8 MoE plan metadata, launch function, payload pointers, scale pointers, workspace pointer, workspace size, and validated latency so graph replay cannot cross incompatible FP8 expert layouts.

This is intentionally not the final high-performance FP8 MoE backend. It is a CUDA-visible correctness/debug scaffold for Spark2 bring-up before Codex swaps W1/W2 with true grouped tensor-core scaled-MEMM.


## This CUDA iteration — staged FP8 activation paths

This iteration moves the regular-linear and MoE debug paths closer to the production FP8 execution shape without depending on unavailable Spark2 CUDA debugging in this container:

- Added an activation-staged FP8 regular-linear reference launch. BF16 activations are dynamically quantized to E4M3 with per-row/per-128 F32 scales, then consumed with E4M3 weights and F32 weight scales by an FP8xFP8 reference GEMM. This gives Codex a direct replacement target for cuBLASLt/CUTLASS scaled-MEMM.
- Added a public workspace calculator and launch entry point for the staged FP8 regular-linear path, and made Blackwell quantized FP8 linear plan binding require aligned activation-staging workspace.
- Updated the resident FP8 linear-plan binder so created FP8 plans allocate and carry their activation-staging workspace, rather than leaving the launch path to guess at transient storage.
- Updated the FP8 MoE grouped-reference path so W1 still stages gate/up deterministically, but the SiLU(gate)*up activation is immediately quantized to E4M3 with dynamic F32 row-block scales. W2 now consumes that FP8 intermediate directly.
- Kept the BF16 residual/output contract unchanged, so this remains safe as a debug-distance CUDA scaffold while still exercising the FP8 activation-scale layout required by production grouped expert GEMM.

This is still not the final tensor-core backend. Production FP8 still needs Spark2 compile/debug plus real scaled-MEMM regular linears and grouped expert GEMM, but the contracts and reference dataflow are now much closer to what those kernels should replace.

## This CUDA iteration — deterministic packed-route FP8 MoE scaffold

This iteration moves the FP8 MoE reference path closer to the layout that a production grouped scaled-GEMM backend should consume:

- Added a deterministic expert-major route packing workspace for FP8 MoE. The builder now creates expert route offsets, per-expert route counts, packed expert ids, packed source token ids, packed source route ids, token-route-to-packed-row mapping, packed route weights, and a device-side packed route count.
- Added public CUDA entry points for packed route workspace sizing, workspace resolution, and route-build launch so Codex can validate route packing independently of the full MoE path.
- Reworked the FP8 grouped-reference workspace to reserve the packed route workspace next to the FP8 intermediate activation arena. The plan now fails closed if the packed route metadata cannot be resolved inside the aligned workspace budget.
- Added packed-route W1 and W2 reference kernels. W1 now consumes expert-major packed rows instead of token-major top-k routes, and W2 uses the deterministic token-route-to-packed-row map when reducing the FP8 intermediate through W2.
- Kept the kernels intentionally conservative and graph-safe: no host reads, no device allocation, no atomics in the deterministic fill path, and no changes to the BF16 output/residual contract.

This is still a debug-distance scaffold rather than the final high-performance MoE backend. The important production-facing change is that the CUDA surface now has the same expert-major packed route shape that true grouped tensor-core W1/W2 kernels should replace.

## This CUDA iteration — shared dense FFN staging and packed FP8 MoE input activation

This iteration removes more debug-path distance between the current FP8 scaffolds and the eventual production scaled-GEMM kernels:

- Added a public FP8 activation workspace resolver so Codex can inspect the E4M3 payload, F32 scale, F32 amax, scale-block count, and exact workspace byte requirement for any staged linear activation buffer.
- Added a prepared-activation FP8 linear reference entry point. Callers that already quantized BF16 activation rows to E4M3 can now reuse that activation with multiple FP8 weight matrices without repeating the quantize pass.
- Added a shared-activation dual-weight FP8 linear reference entry point. Dense gate/up can now quantize the post-attention hidden activation once and feed both FP8 gate and FP8 up projections from the same E4M3 activation-scale view.
- Wired the dense FFN FP8 path to use the shared gate/up staging path when dense gate/up/down are bound as FP8 tensor-core plans. The path then uses the existing fused SiLU(gate)*up plus FP8 activation staging before dense down, preserving the BF16 residual/output contract.
- Added a packed-route hidden-activation FP8 quantizer for MoE. After deterministic route packing, the reference W1 path now consumes packed expert-major E4M3 hidden rows plus F32 scales instead of repeatedly reading BF16 token hidden rows.
- Expanded the FP8 MoE grouped-reference workspace layout to reserve packed hidden E4M3 payloads, hidden scales, and hidden amax values before gate/up, intermediate, and route metadata arenas.

This is still a reference/debug backend, not the final tensor-core implementation. The important production-facing change is that both dense FFN and routed MoE now exercise the same activation-side FP8 staging contracts that cuBLASLt/CUTLASS regular linears and grouped expert GEMMs should eventually consume.

## This CUDA iteration — prepared FP8 activations for dense FFN and MoE W1

This iteration removes two remaining debug-path mismatches between the FP8 reference CUDA scaffolds and the eventual production grouped/scaled-MEMM shape:

- Added an explicit prepared-activation workspace view for FP8 regular linears. Codex can now resolve the activation E4M3 payload, F32 scale blocks, and F32 amax buffers independently of launching the GEMM-like reference kernel.
- Added a prepared FP8 activation × FP8 weight reference launch. This lets a future cuBLASLt/CUTLASS scaled-MEMM backend consume the exact same prequantized activation/scale buffers instead of hiding activation quantization inside the launcher.
- Added a shared-activation dual-linear helper for the dense FFN gate/up pair. The post-attention normalized hidden row is quantized once, then reused for both FP8 gate and FP8 up projections.
- Wired the dense FP8 FFN path to use prepared staging when dense gate/up/down are all FP8 E4M3 tensor-core plans. The down projection now consumes the FP8 output of fused SiLU(gate) * up quantization directly, instead of letting the regular-linear launcher quantize the same intermediate again.
- Added packed-route hidden activation quantization for FP8 MoE W1. After deterministic route packing, hidden rows are copied into expert-major packed-route order and quantized to E4M3 with per-row/per-128 F32 scales; packed W1 now consumes FP8 activations plus FP8 weights.
- Extended FP8 MoE grouped-reference workspace layout to reserve packed hidden FP8 payload, hidden scales, and hidden amax buffers before gate/up/intermediate storage. The workspace calculator now reflects the full routed W1/W2 FP8 activation staging contract.

The kernels are still reference/debug scaffolds rather than final tensor-core kernels, but the dataflow is now much closer to production: dense gate/up share one staged activation, dense down consumes prepared fused activation, and MoE W1/W2 both exercise FP8 activation-scale contracts in expert-major route order.

## Final CUDA handoff iteration — external FP8 backend seams and debug probes

This iteration focuses on reducing the remaining pure CUDA work for Sparkring by turning the reference FP8 paths into explicit replacement seams for production kernels:

- Added an external FP8 E4M3 scaled-GEMM backend ABI for regular linears. The backend receives prepared activation E4M3 payloads, activation F32 scale/amax buffers, E4M3 row-major weights, row-block-major F32 inverse weight scales, output dtype, explicit scale-block counts, backend workspace, opaque backend state, and the stream.
- Added workspace sizing and binding helpers for external FP8 regular-linear backends. Bound FP8 tensor-core linear plans now reserve activation staging plus backend workspace, store the backend descriptor in the plan, and fail closed if alignment/capability/latency metadata is incomplete.
- Updated the Blackwell FP8 regular-linear launch path and dense FFN FP8 prepared-staging path so a bound backend replaces only the GEMM body. Activation quantization, shared dense gate/up staging, fused SiLU/down staging, output routing, and graph plumbing remain Sparkpipe-owned.
- Added a prepared-activation FP8 linear debug compare launch. Sparkring can launch a candidate scaled-GEMM backend and compare its BF16/F32 output against the reference FP8 activation × FP8 weight dot product on device without host readback during graph-safe debugging.
- Added an external packed-route FP8 MoE grouped backend ABI. The backend receives deterministic expert-major packed routes, packed FP8 hidden activations and scales, W1/W2 E4M3 expert weights and F32 row-block scales, explicit hidden/intermediate/output scale-block counts, backend workspace, opaque state, and stream.
- Added workspace sizing and binding helpers for external FP8 MoE grouped backends. The wrapper still performs router top-k, deterministic packed-route build, and packed hidden FP8 quantization before handing W1/W2 execution and deterministic finalize to the backend.
- Added a device-side packed-route debug check for FP8 MoE. It validates packed route counts, offsets, token/route/expert ranges, token-route-to-packed-row mapping, and finite route weights.
- Extended CUDA graph signatures for FP8 linears and FP8 MoE so backend descriptors, launch function pointers, opaque backend state, required workspace, and validated latency participate in graph compatibility checks.
- Fixed the FP8 linear debug reference scale indexing to match the runtime row-block-major weight-scale layout: output rows share scale rows by `output_index / scale_block_size`, not by individual output row.

The remaining production work should now be concentrated in backend launch functions: cuBLASLt/CUTLASS scaled-MEMM for regular linears and grouped tensor-core W1/W2 for packed-route MoE. The Sparkpipe-side FP8 staging, validation, routing, graph signatures, and debug comparison surfaces are in place for Sparkring bring-up.
