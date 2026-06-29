# GLM 5.2 MoE SOTA Research, 2026-06-29

This note records the internet-backed performance model for the GLM 5.2 MoE
kernel work on DGX Spark / GB10 / SM121. It is meant to prevent us from
mistaking scaffolding or scalar persistent kernels for a real SOTA path.

## Sources Checked

- NVIDIA TensorRT-LLM, MoE as Dense GEMM:
  https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html
- NVIDIA TensorRT-LLM, DeepSeek-R1 B200 latency optimization:
  https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html
- NVIDIA CUTLASS Blackwell functionality:
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- CUTLASS SM120/SM121 GeForce NVFP4 grouped GEMM example:
  https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu
- CUTLASS Blackwell MoE grouped GEMM example:
  https://github.com/NVIDIA/cutlass/blob/main/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped.cu
- DeepSeek DeepGEMM:
  https://github.com/deepseek-ai/DeepGEMM
- FlashInfer GEMM / fused-MoE API surface:
  https://docs.flashinfer.ai/api/gemm.html
- NVIDIA DGX Spark hardware guide:
  https://docs.nvidia.com/dgx/dgx-spark/hardware.html

## Direct Conclusion

The strict tiny-N overlay is the wrong algorithmic family.

It is not 10x slower because a launch parameter is bad. It is slower because it
turns the GLM MoE critical path into scalar/output-row dot products and does not
keep SM121 tensor cores fed. The public SOTA paths are tensor-core GEMM shaped:
grouped selected-expert GEMM, dense all-expert GEMM in the medium-token
latency band, fused Mega-MoE style kernels, or kernel families that preserve the
same math shape while reducing padding, dispatch, and combine overhead.

## What The Public Sources Say

NVIDIA's DenseGEMM note splits MoE by token regime, using a DeepSeek-style
`num_experts = 256`, `top_k = 8` shape:

- very small, `num_tokens <= 32`: many experts are empty and fixed overheads
  dominate. Loading every routed expert is too expensive.
- low latency, roughly `32-320`: most routed experts receive at least one route,
  but each expert still has tiny token counts, so routed FC1/FC2 is memory-bound.
- larger token counts: some experts become compute-bound and grouped GEMM
  utilization improves.

The same NVIDIA note reports DenseGEMM as best only for a middle band in its
B200 TP8 module benchmark, about `num_tokens = 64-208`. Below that, the
TRTLLM-Gen grouped-GEMM path is faster; above that, dense redundant compute
stops helping.

NVIDIA's DeepSeek-R1 latency note shows that SOTA was not reached by one magic
kernel. Their sequence includes CUDA graphs / PDL, multi-stream overlap, MLA
kernel work, TopK work, Fuse_A/RouterGEMM work, MTP, grouped GEMM, sparse expert
rebalance, and router changes. That is the right mental model for SparkPipe:
the MoE path must be a coordinated resident firmware path, not a pile of
standalone probes.

CUTLASS confirms that SM120/SM121 supports narrow-precision Blackwell GEMMs and
that valid `nv_float4_t` tile shapes are tensor-core scale, such as
`128x128x128` and `256x128x128`. The CUTLASS 79d example is the relevant public
SM120/SM121 GeForce NVFP4 grouped GEMM family. The CUTLASS 92 MoE example has
the attractive `tokens_per_expert` contract, but it is written as a Blackwell
SM100 MoE example; on our GB10/SM121 path, 79d is the proven compile/run family.

DeepGEMM and FlashInfer point in the same direction: grouped GEMM, masked
grouped GEMM for CUDA graph decode, fused MoE, fused dispatch/combine, and
FP4/FP8 tensor-core kernels. They do not support the idea that scalar per-row
dot-product workers are a production SOTA path.

## Our Measurements

Current local Spark measurements:

```text
file: qualification/glm52_cutlass_moe_sm121a_benchmark.latest.tsv
path: fixed-slot CUTLASS SM121a grouped NVFP4/BF16 MoE

tokens=32 active_routes=256 bound_experts=8   capacity=32 avg_us=955.482
tokens=32 active_routes=256 bound_experts=32  capacity=8  avg_us=3334.458
tokens=32 active_routes=256 bound_experts=64  capacity=4  avg_us=6638.285
tokens=32 active_routes=256 bound_experts=128 capacity=2  avg_us=12868.603
tokens=32 active_routes=256 bound_experts=256 capacity=1  avg_us=25436.607
```

```text
file: qualification/glm52_strict_tiny_n_moe_overlay_sm121a_benchmark.latest.tsv
path: strict custom tiny-N overlay

tokens=32 groups=8   capacity=32 tile=8 workers=256  avg_us=66632.641
tokens=32 groups=8   capacity=32 tile=8 workers=1024 avg_us=66746.656
tokens=32 groups=8   capacity=32 tile=4 workers=1024 avg_us=74035.766
tokens=32 groups=256 capacity=1  tile=1 workers=256  avg_us=167447.172
```

This rejects three hypotheses:

1. The strict tiny-N work-item design does not beat generic fixed-group CUTLASS
   for compact active experts.
2. More persistent worker blocks do not expose under-provisioning; latency is
   unchanged.
3. Smaller route tiles do not improve occupancy; they are slower.

The strict overlay should not be used as the production path. It is a useful
negative result because it proves that "custom tiny-N" must still be tensor-core
MMA shaped. Custom does not mean scalar.

## DGX Spark Roofline Reality

NVIDIA lists DGX Spark memory bandwidth as `273 GB/s`. For GLM 5.2 constants:

```text
hidden H = 6144
moe intermediate I = 2048
experts E = 256
top_k = 8
NVFP4 payload = 0.5 bytes per weight, before scales
```

Per expert payload lower bound:

```text
gate/up = (2 * I * H) / 2 = 12,582,912 bytes
down    = (H * I) / 2     =  6,291,456 bytes
total   =                  18,874,368 bytes
```

All 256 experts:

```text
256 * 18,874,368 = 4,831,838,208 bytes
4.832 GB / 273 GB/s = 17.7 ms lower bound
```

This lower bound excludes scale-factor traffic, activation reads/writes,
packing, combine, route metadata, and launch overhead. Therefore, a path that
touches all 256 experts for one route slot each cannot be judged against an
8-expert compact path. It is mostly a memory-bandwidth problem on GB10.

The fixed CUTLASS 256-expert result:

```text
4.832 GB / 25.437 ms = about 190 GB/s payload bandwidth
```

That is not SOTA, but it is in the right physical category. The strict overlay:

```text
4.832 GB / 167.447 ms = about 28.9 GB/s payload bandwidth
```

That is not close to the hardware. The problem is the kernel shape.

## Correct SOTA Shape For SparkPipe

SparkPipe needs three production MoE modes and a real router histogram to choose
between them.

### Mode A: Tiny/latency sparse selected experts

Use for `B1-B32` or when active experts are compact.

Required kernel family:

- tensor-core grouped selected-expert GEMM, not scalar workers.
- compact active experts, not blind 256-group fixed shape.
- route coalescing when latency budget permits, so hot experts see useful `N`.
- fused gate/up, SiLU, requant, down, and weighted combine where possible.

The current CUTLASS 79d-derived path is the best public SM121 baseline we have,
but its 256-group result proves that blind fixed groups are too expensive for
uniform one-route-per-expert decode.

### Mode B: Medium-token dense alpha MoE

Use for `B64-B208` candidates only after measuring on GB10.

Required kernel family:

- dense all-expert routed FC1/FC2 formulation.
- alpha mask or equivalent per-token selected-expert weighting.
- tensor-core large GEMMs.
- fused alpha/SwiGLU/down/combine path.

NVIDIA's public result says this is the sweet spot on B200 for DeepSeek-style
MoE. GB10 has much less memory bandwidth, so we must measure it before
publishing. But this is the most credible way to remove the group-count cliff
for `B64/B96/B128`.

### Mode C: Larger batch sparse grouped experts

Use when per-expert token counts are high enough that selected-expert grouped
GEMM approaches normal GEMM efficiency.

Required kernel family:

- active-expert grouped tensor-core GEMM.
- no all-expert dense waste once redundant compute dominates.
- CUDA graph capture for stable physical profiles.

## Immediate Engineering Plan

1. Add a GLM router histogram benchmark.

   For `B=1,8,32,64,96,128`, record:

   ```text
   active expert count
   min/mean/max routes per active expert
   top hot experts
   empty expert count
   selected route count
   route entropy
   ```

   Without this, every kernel choice is guesswork.

2. Stop investing in the strict scalar tiny-N overlay.

   Preserve its benchmark artifact as a negative result, but do not tune it.
   Its internal dot product must be replaced by SM121 tensor-core MMA tiles or
   by a known tensor-core grouped/dense backend.

3. Implement the medium-token DenseGEMM experiment.

   This is the highest-leverage code path for the current 10x gap because
   NVIDIA's public work says dense all-expert MoE can beat grouped GEMM in the
   `64-208` token low-latency band. For SparkPipe this should be a GLM firmware
   mode, not a SparkPipe runtime abstraction.

4. Implement an active-expert compact grouped path.

   For tiny batches, do not launch 256 fixed groups when only a small hot subset
   has meaningful work. Use compact active experts, fixed resident workspaces,
   and graph-captured profiles.

5. Decide by measured regime, not ideology.

   Bench matrix:

   ```text
   B = 1, 8, 32, 64, 96, 128
   real router traces and synthetic uniform worst case
   compact grouped selected experts
   fixed 256-group CUTLASS baseline
   dense-alpha all-expert path
   strict overlay only as rejected baseline
   ```

   Required counters:

   ```text
   total latency
   gate/up latency
   activation/requant latency
   down latency
   combine latency
   bytes read/written
   achieved GB/s
   achieved TFLOP/s
   active expert count
   average N per active expert
   graph replay time
   ```

## What To Ask The Advisor For

The useful request is not "make tiny-N faster" in general. The specific missing
CUDA modules are:

1. SM121 tensor-core active-expert grouped NVFP4 MoE.

   It must consume compact active experts and use MMA tiles for gate/up and
   down. It must not compute one output row per CTA with scalar loops.

2. SM121 dense-alpha GLM52 MoE for `B64/B96/B128`.

   This should test NVIDIA's DenseGEMM regime on GB10. It can read all resident
   local experts if the large GEMM shape recovers enough tensor-core efficiency.

3. GLM52 router histogram capture and replay harness.

   This should feed real and replayable `top_k` distributions into both sparse
   and dense MoE paths. No more uniform-only intuition.

4. Fused MoE critical path.

   The target is:

   ```text
   route/topk -> pack/quant -> gate/up -> SiLU/requant -> down -> weighted combine
   ```

   Fusing or graph-capturing this matters because NVIDIA's SOTA stack combines
   kernel-level improvements with CUDA graphs, PDL, and overlap.

## Current Decision

The next performance branch should not continue with the strict tiny-N overlay.

The next branch should implement and benchmark either:

1. dense-alpha GLM52 MoE for `B64/B96/B128`, or
2. compact active-expert tensor-core grouped MoE for `B1/B8/B32`.

Given the current 10x complaint, the dense-alpha experiment is the cleanest way
to test the public SOTA claim directly. If it loses on GB10, the negative result
will still correct the model: GB10 memory bandwidth makes all-expert dense MoE
too expensive, and the only viable path is compact active-expert tensor-core
grouped MoE plus coalescing.
