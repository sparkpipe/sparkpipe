# CUDA-KERNELS — top-speed assessment + TOP-10

Area: the kernel surface as of `docs/KERNEL_PLAYBOOK.md` (tricks + receipts +
registry) and `docs/KERNEL_CONTRACT_CARDS.md` (18 speculation cards). Metric:
maximize Solutions / (code size²); DRY wins first, then performance LEVELS
`1 < 2 (80% SOTA) < 3 (90%) < 4 (match) < 5 (exceed)` (`METRIC.md:2-12`).

## Where we stand (verified)

- **dsv4-flash driver already matches/exceeds its vLLM reference.** 40.4553
  tok/s TP4 B1 (`PERFORMANCE_STATUS.md:159-161`) vs plain-vLLM 37.7854
  (`PERFORMANCE_STATUS.md:287`) = **LEVEL 4** on that workload; the 50 tok/s
  gate is unmet (`PERFORMANCE_STATUS.md:276-278`).
- **Every speculation kernel is built and NOT_MEASURED.** All 18 cards
  (`KERNEL_CONTRACT_CARDS.md`) have zero receipts; every DSV4/GLM52 run is
  explicitly "no speculation" (`PERFORMANCE_STATUS.md:102,581,586`). The DSV4
  dspark path is already wired — `dspark_verify` flag + attention/markov/argmax
  launches (`spark_dsv4_resident_decode_stage_module.c:3763,4248,4255,4414`) —
  it is a measurement, not a build.
- **The dense/projection GEMV path is near its bandwidth floor.** CUPTI puts
  dense/projection at 38.65% and routed experts 21.08%
  (`PERFORMANCE_STATUS.md:230-233`); the two further-fusion probes were
  bit-exact yet **rejected** end-to-end (persistent bundle -0.78%,
  `PERFORMANCE_STATUS.md:254-263`; readahead queue -33%,
  `PERFORMANCE_STATUS.md:265-274`). More B1 kernel fusion is a rejected-probe
  repeat, not a level.
- **glm52 single-stream is bandwidth-bound, not kernel-bound.** 6.91 tok/s vs
  the ~80 ms 273 GB/s floor (~15.7 GB FP8 + ~6 GB BF16/token),
  `PERFORMANCE_STATUS.md:594-599`. Its next level is batching (already lifts
  43.46→75.55) or trained draft weights — neither is a kernel change.

## The cheapest next LEVEL (headline)

**Enable + measure DSV4 Flash DSpark speculation** — cards `D4-D-01..05` +
`V-01`/V-02. The whole kernel surface exists and is wired; speculation is the
standard way to move past a non-speculative SOTA reference, so this is the
dsv4-flash **level 4→5** buy at ~0 new kernel code. After that, the K3 and
DSV4-Pro drivers (today level 1, no speculation) get their level from the
§0-pending drafter kernels — which is why filling those "0 (not declared)"
fields is the real gate (`spark_dspark_drafter.h:41-86`).

## TOP-10 (ranked; Δ = code-size estimate)

| # | What | Why right (level / DRY) | Δ | Owner | First step |
|---|---|---|---|---|---|
| 1 | Measure DSV4 Flash DSpark spec (cards D4-D-01..05, V-01) | LEVEL 4→5 for dsv4-flash; code exists, unmeasured | ~0 | dsv4-flash + speculation | TP4 B1 O128 run, dspark on; record tok/s + 128-token hash parity |
| 2 | DRY: fold GLM52 dspark norm/swiglu/add onto shared primitives | 3 hand-rolled kernels (`spark_glm52_dspark_draft_backend.cu:323-406`) duplicate `norm.cuh` LmFusedResidualRmsNorm/LmAddRows/LmSiluMul + `LmBlockSum` (`norm.cuh:27-46,91,154,358`); hand reduction is barrier-per-step vs warp-shuffle | ~-110 | cuda-kernels + glm52 | Preserve the 2 bf16 rounding points (`spark_glm52_dspark_draft_backend.cu:361-367`) while swapping the reduction |
| 3 | DRY: verify + delete `LmGatherRowsKernel` | Indirect gather was deleted (`route.cuh:24`); kernel still defined (`norm.cuh:562`) + instantiated (`inference/llms/kimi_k3/unity.cu:127`) and tests now FORBID it (`test_cuda_performance_contracts.py:280,355`) | ~-15 | cuda-kernels + k3 | Confirm the `unity.cu:127` instantiation is unlinked, then delete def + instantiation |
| 4 | DRY: emit one shared bf16 block-reduction path | `LmBlockSum`/`LmBlockMax` (`norm.cuh:27-67`) already exist; GLM52 dspark + older copies re-roll it (`spark_glm52_dspark_draft_backend.cu:347-357`); the tree warns against exactly this (`linear_attn.cuh:168-171`) | ~-60 | cuda-kernels | Move GLM52 dspark reductions onto `LmBlockSum` (same as #2, listed separately for the reduction class) |
| 5 | Pin the DSV4 B1 dense-GEMV weight stream (measure, don't fuse) | Turns "38.65%" into a proven floor; prevents re-proposing rejected fusions (PERF 254-263) | ~0 | cuda-kernels | nsight/compute over `SparkLmSm121FusedDenseW13GemvKernel` (`spark_lm_kernels.cuh:3016`) at B1 |
| 6 | K3 drafter kernels (block 7, 64q/16kv, intermediate 14336, markov 256) | LEVEL 1→2/3 for k3 (no speculation today); shapes already pinned (`spark_dspark_drafter.h:41-63`, `inference/llms/kimi_k3/dspark.h:17-83`) | +~1200 | k3 session → cuda-kernels | K3 fills §0 card; port GLM52 backend, swapping the drafter table |
| 7 | DSV4-Pro drafter kernels (markov 512, taps {58,59,60}) | LEVEL 1→2/3 for dsv4-pro; heads/intermediate are "0 (not declared)" (`spark_dspark_drafter.h:64-86`) | +~1200 | dsv4-pro session → cuda-kernels | Pro pins heads/intermediate; then reuse #6's backend |
| 8 | GLM52 MTP verify loop (6 kernels, `speculate.cuh:6`) | LEVEL for glm52 single-stream if draft weights ship; dspark path is blocked ("base checkpoint ships none", `PERFORMANCE_STATUS.md:600-601`) | ~0 (measure) | glm52 + speculation | Confirm MTP layer weights exist in checkpoint; measure if present |
| 9 | Per-card receipt discipline | Turns NOT_MEASURED → measured incrementally; each landed number is a free Solutions entry | ~0 | cuda-kernels | Add a receipt template; require one per landed kernel |
| 10 | Weight-read-ahead for expert weights, receipt-gated | Only if it survives the rejected-queue lesson (no CTA steal, `PERFORMANCE_STATUS.md:265-274`) | +~80 | cuda-kernels + dsv4 | Prototype overlap of the 60 us collective windows with expert prefetch; kill on regression |

**Out of lane (flag, don't build):** the ~4.47 ms/model-step collective-shaped
gap (73% of exposed idle, `PERFORMANCE_STATUS.md:224-228`) is transport, not
kernel surface.
