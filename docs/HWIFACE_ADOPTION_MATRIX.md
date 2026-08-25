# HWIFACE adoption matrix — all model drivers to hardware-agnostic (CUDA + ROCm)

Status: ANALYSIS (architecture input; no code modified). Written against the
frozen `hwiface_v1.md` contract (REV2) and unified head, 2026-08-25.
Terminology reuses `docs/coord/plan_amd_gfx950_mi350p.md` as-is: islands
E0/L1–L5/F1, targets `cuda.sm121.gb10` / `rocm.gfx950.mi350p`, archives
`dsv4_core` / `dsv4_cuda_sm121_gb10` / `dsv4_rocm_gfx950_mi350p`, steps
S1–S8, gates C1/C2/C3/P1, landmine rule, workstreams A–D. That plan owns
MXFP4 layout decisions (B1–B5), RCCL mapping detail, and MI350P bring-up;
this document does not duplicate them — it extends coverage from DSV4 to
**every** model driver and scores what must change.

Evidence convention: counts of `cuda[A-Z][A-Za-z]+` tokens include enum/error
literals (`cudaSuccess`, `cudaMemcpyHostToDevice`) — treat them as coupling
mass, not call-site count. Line numbers drift; symbols are the citation.

---

## 1. Per-driver coupling scorecard

Coupling classes: **K1** runtime/driver API outside the future hwiface surface ·
**K2** SM-arch conditionals / capability guards · **K3** inline PTX /
arch-specific asm · **K4** vendor library deps · **K5** CUDA-graph usage ·
**K6** transport/collective coupling.

### 1.1 Driver summary

| Driver (module) | Host `.c` K1 tokens | Device layer | `<<<` launches | K2 | K3 | K4 | K5 | Port weight |
|---|---|---|---|---|---|---|---|---|
| dsv4_resident_decode_stage | 489 (`…module.c`) | `…_cuda.cu` 224 tokens, 61 extern decls / 57 `SparkDsv4Launch*` defs | 42 in `.cu`; **0 in host module** *(contract-measured)* | guard `SparkDsv4RequireNativeSm121` `.cu:139/:161/:2785`; PTX image trap via `LM_SM121_NATIVE_COMPUTE_PTX` (`mma.cuh:42–46`) | LDS opt-in `.cu:2743–2773`; devattr MP count `.cu:2736` | none (no cuBLAS/cuDNN anywhere) | graph islands PROJECTION/ATTENTION/FFN/FINAL `module.c:275–278`, capture `:4179–4195`, replay `:4120` | HIGH mass, LOW risk — already island-shaped |
| qwen36_resident_decode_stage | 328 (`…module.c`) | `…_cuda.cu` 140 tokens, 47 launches; + `spark_qwen36_dspark_cuda.cuh`, `spark_qwen36_native_ws.cuh` | 47 | `__CUDA_ARCH__` host/device split `.cu:89` | named barriers `bar.sync 3,128/1,896/2,896` `.cu:1531–1578`; `cp.async.*` pipeline `native_ws.cuh:17–28`; more `bar.sync` `:147–247` | none | capture/replay in host `module.c:3872–4031` | HIGHEST device-code rework (wave64 barrier redesign) |
| qwen38_resident_decode_stage | 131 (`…module.c`) | `…_cuda.cu` 83 tokens, 22 launches | 22 | includes landmine header `spark_lm_kernels.cuh` (`.cu:5`) — bound by §5 rule | LDS opt-in `.cu:1695–1725` | none | none in module (uncaptured) | LOW–MEDIUM; shares route/expert machinery with DSV4 |
| glm52_resident_decode_stage | 52 (`…module.c`, `cudaLaunchHostFunc :1997`) | `source/cuda/{layer.cuh,unity.cu}` via `…_cuda.cu` 47 tokens, 7 launches | 7 | TMA-swizzle static assert `unity.cu:56` | none inline | none | `LmGraphCache` over shared `graph.cuh` `unity.cu:219–254` | LOWEST — smallest surface, dense REV1.1 island set |
| glm52_dspark_draft_backend | n/a (single TU) | standalone `.cu`: 87 tokens, 12 launches | 12 | includes `cuda_bf16.h` | none | **zero cuBLAS today** — removed; only a past-tense comment `.cu:1251/:2102` | none | LOW; rides with glm52 (REV1.3 keeps drafter inside F1) |
| k3_resident_decode_stage | **0 CUDA tokens in `module.c`** (cleanest host) | `runner.cu` 128 tokens, 6 launches; `…_cuda.cu` 30 tokens; `inference/llms/kimi_k3/{layer,slice}.cuh` (1083+570 ln) | 6 | bespoke graph cache duplicating `graph.cuh` semantics `runner.cu:238–340` | none inline | none | own `cudaGraphExec_t` cache `runner.cu:285–340` | MEDIUM mass — biggest kernel surface, duplicated mechanisms |

Shared per-family device layers (`inference/llms/*`) are thin extern-C launch
surfaces already shaped like islands (`Mimo25LayerAttentionFullFp8`
`mimo_2_5/unity.cu:102–131`, `K3HeadFullVocab` `kimi_k3/unity.cu:119`) but are
format-multiplexed rather than island-named — renaming is cheap, porting their
kernel bodies is where the cost lives. Note: `mimo_2_5` has **no decode-stage
module yet**; its port obligation defers until one exists.

### 1.2 Shared kernel layer (`inference/kernels/*.cuh`) — the real port surface

| Header | Coupling | Evidence | gfx950 disposition |
|---|---|---|---|
| `mma.cuh` | K2+K3, deepest | `__CUDA_ARCH__==1210` gate `:42–46` deliberately traps non-SM121 passes; block-scaled/mixed-input MMA PTX `:101,:130,:156,:237,:249,:273,:281,:298,:306,:440` | target-private compute layer (MFMA); trap philosophy mirrors fail-closed guard |
| `tma.cuh` | K3 | mbarrier + `cp.async.bulk.tensor` PTX `:63–188` | no ROCm analog; replaced by target-private async-copy/LDS scheme below identical island entries |
| `tensor_map.cuh` + `runtime/tensor_map.h` | K1 (driver API) | `cuTensorMapEncodeTiled` `runtime/tensor_map.h:84` with `#include <cuda.h> :15` | descriptor build becomes target-private; consumers keep opaque handles |
| `dtype.cuh` | K3 | FP8/UE8M0 pack PTX `cvt.*` `:97–180` | HIP conversion intrinsics; bit-exactness required by C2/C3 field tables |
| `graph.cuh` | K5 | `cudaGraphExec_t` cache, relaxed capture `:62,:120–171`, `#include <cuda_runtime.h> :34` | maps 1:1 onto HIP graphs → first candidate to sit on `spark_hw_graph_*` |
| `norm/activation/head/speculate/topk/linear_attn/kv/route/gqa/scale/layout` | near-neutral | warp-shuffle reductions keyed to `LM_WARP_LANES 32u` (`mma.cuh:32`; loops e.g. `norm.cuh:30–33`); traps at dead fallbacks `kv.cuh:205`, `activation.cuh:71`, `tile.cuh:334` | portable after wave-size parameterization; C3 reduction order fixed per bucket internally to the target |
| `formats/{bf16,fp8,int6,int7,int8,nvfp4}.cuh` | transitive K2/K3 | all include `mma/tensor_map/tma` headers | split neutral packing math from arch compute (same surgery as the landmine split) |

Vendor libraries: **no cuBLAS or cuDNN dependency exists in the tree**
(the last one, `cublasGemmEx` in the glm52 drafter, was already folded into
the shared `runtime/gemm.cuh` stack — comment `spark_glm52_dspark_draft_backend.cu:1251`).
NCCL appears only behind the collective seam (§3).

### 1.3 Landmine exposure (contract §5)

`SPARK_LM_SM121_*` selectors live in
`model-families/common/include/sparkpipe/spark_lm_kernels.cuh`. Measured
includers among drivers: dsv4 `.cu:1`, qwen36 `.cu:5` (+ `dspark_cuda.cuh:20`,
`native_ws.cuh:12`), qwen38 `.cu:5`. k3 and glm52 do not include it yet and
are bound the day they do. The header split is a hard precondition for any
ROCm build of qwen36/qwen38 (and for step-7 families).

---

## 2. Gap matrix vs the hwiface_v1 surface

**No `spark_hw_iface.h` exists anywhere in the tree** (`include/sparkpipe/`
verified); there are no per-target primitive archives. Every cell below is
therefore *missing by construction*; "nearest analog" names what the wrapper
mechanically replaces first.

| Contract family (§4.0–§4.2) | Functions | CUDA realization today (nearest analog) | `spark_hw_cuda_*` stub | `spark_hw_rocm_*` stub |
|---|---|---|---|---|
| Status model | `SparkHwStatus` 6 codes | ad-hoc `cudaError_t` plumbing in every module host (e.g. dsv4 `module.c` 141 `cudaError` tokens) | missing | missing |
| Memory pool | pool_alloc/free | direct `cudaMalloc`/free in hosts (qwen38 `module.c` ×2) | missing | missing (plan §3.2: wrap `hipMalloc`) |
| Host pinned | pinned_alloc/free, host_device_pointer | `cudaHostAllocPortable|Mapped` dsv4 `module.c:708–712`, glm52/qwen38 mapped tiers | missing | missing |
| Copy / memset | copy_async (H2D/D2D/D2H), memset_async | 23× `cudaMemcpyAsync` + memsets in dsv4 host *(contract-measured)*; qwen36 29×; qwen38 13×; glm52 5× | missing | missing |
| Read-ahead | read_ahead (A11 three-clause) | weight-read-ahead kick inside dsv4 device path | missing | missing |
| Queue | create/destroy/query/synchronize/enqueue_host_callback | `cudaStreamCreate` etc.; completion = `cudaLaunchHostFunc` dsv4 `module.c:5410`, glm52 `:1997`, rdma `rdma.cu:3959,:4098` | missing | missing (`hipLaunchHostFunc`; stress-loop proof per plan §3.2) |
| Event | create/destroy/record/wait (timing disabled) | `cudaStreamWaitEvent` transport `tp_device_collective.c:1826` | missing | missing |
| Graph | capture begin/end, instantiate, launch, destroy ×2 | three separate implementations: dsv4 host islands `:4179–4195`, qwen36 host `:3872–4031`, shared `graph.cuh` cache, plus k3's bespoke cache `runner.cu:310–340` | missing | missing (HIP graphs 1:1) |
| Target descriptor | `SparkHwTarget` | scattered: MP-count query dsv4 `.cu:2736`, LDS limit opt-ins `.cu:2743–2773`, qwen38 `.cu:1695–1725`, qwen36 `.cu:1303–1313`, `runtime/gemm.cuh:54–85` | missing | missing |
| Island entries E0/L1–L5/F1 | extern per island | DSV4: named graph islands `module.c:275–278` ≈ §3.2 mapping; other drivers: unity/bind externs not island-named | present-by-analogy (DSV4 only) | absent |
| Collective seam (§4.4) | existing `spark_tp_device_collective.h` ABI 12 | NCCL dynamically loaded `tp_device_collective_nccl.c:531,:539–546`; stream handles cast throughout `tp_device_collective.c:110,:1169,:1826,:3106` | exists (CUDA/NCCL) | absent — RCCL backend per plan workstream D |

Gap reading: the primitive gap is **one mechanical wrapper away** for CUDA
(rule R2: direct calls, static link), which is exactly frozen step S1. The
island-entry gap is already closed for DSV4 and is a naming exercise for
glm52/qwen38; the real cross-target work concentrates in the K2/K3 rows of §1.2.

---

## 3. Transport seam status (K6)

The collective header is already neutral (opaque `void *cuda_stream`,
backend enum `_HIDDEN_TRANSPORT|_NCCL`). Two coupling pockets remain for a
second target: `tp_device_collective.c` hardcodes `cudaStream_t` casts and
stream/event calls (sites above), and `ring/transport/rdma.cu` carries 51
CUDA tokens incl. `cudaLaunchHostFunc :3959,:4098`. Both resolve under the
same S1 pattern (opaque queue handles + `spark_hw_*` primitives) without
touching the ABI; RCCL realization details stay owned by plan workstream D.

---

## 4. Adoption order recommendation (cheapest-first to one dual-backend model)

The frozen sequence **already is** the cheapest path to the goal state:
S1–S3 make DSV4 target-split with C1 byte-identity + P1 no-regression,
S4–S6 stand up `rocm.gfx950.mi350p` (workstreams A/B), S7/S8 prove one layer
then full attention/MoE/TP4 (workstreams C/D). DSV4 is both the contractual
first (R7) and empirically the best-positioned driver: zero `<<<` launches in
its host module, everything funnels through 61 externs, and its graph topology
already speaks island names. Nothing pre-empts it.

Pre-S1 obligations (frozen): P1 baseline capture via
`tools/devcycle/bbench.sh B`, buckets {1, 8, 1024}, before any refactor.
Recommended alongside (shrinks every later port): TOP10 items #2/#4 (fold
GLM52 dspark hand-rolled norm/swiglu/add onto `norm.cuh` primitives) and #3
(delete the dead `LmGatherRowsKernel`) — pure DRY, no behavior change.

**Order for the remaining drivers after DSV4 goes dual-backend** (each gated
on the landmine split having landed, which S6 requires anyway):

| Order | Driver | Why this position | Main port cost | Effort (analytical, 1 eng, post-DSV4 patterns) |
|---|---|---|---|---|
| 1st model (frozen) | DSV4 | see above; contract steps S1–S8 | runtime primitives + island naming + MXFP4/RCCL per plan A–D | owned by plan (not re-estimated here) |
| 2nd | glm52 (+ dspark backend) | smallest host coupling (52 tokens); dense single-island FFN instantiation already defined (REV1.1); no current landmine exposure; `graph.cuh`→HIP graphs 1:1; completion callback pattern already in place `:1997` | TMA-swizzle tiles + tensor-map descriptors become target-private staging | 2–3 wk + validation |
| 3rd | qwen38 | grouped-MoE machinery overlaps the S8 ROCm L4 work (reuse); moderate surface (22 launches) | requires landed landmine split (it includes `spark_lm_kernels.cuh` today) | 2–3 wk + validation |
| 4th | qwen36 | deepest device-specific code: named-barrier producer/consumer pipelines (896-thread `bar.sync`) and `cp.async` staging must be redesigned for wave64; GDN hybrid-recurrence adds an L3 state channel + field table (REV1.2) | kernel-body rework, not plumbing | 4–6 wk + validation |
| 5th | kimi_k3 (+ mimo when its module exists) | largest kernel surface (1083-line layer.cuh + slice.cuh, int7/nvfp4 formats); bespoke runner duplicates graph-cache logic that should consolidate first | consolidation prerequisite below, then widest validation set | 5–8 wk combined + validation |

Cross-cutting prerequisites worth doing once, early: wave-size-parameterized
reduction paths (`LM_WARP_LANES`), dtype-conversion intrinsics abstraction
(`dtype.cuh`), and the graph-cache unification below — each amortizes across
every driver row above.

---

## 5. Shared-kernel consolidation opportunities (do before/with the ports)

1. **One graph mechanism, four implementations today.** dsv4 host island
   graphs (`module.c:4179–4195`), qwen36 host graphs (`:3872–4031`),
   `graph.cuh` `LmGraphCache` (glm52), and k3's private cache
   (`runner.cu:238–340`) re-implement key/record/instantiate/replay.
   Consolidate onto the `spark_hw_graph_*` family + one bucket-key cache;
   ROCm then inherits capture/replay for free (topology stays target-internal
   per §3.3).
2. **Wave-lane parameterization.** `LM_WARP_LANES 32u` is baked into
   `mma.cuh:32` and every reduction loop; move to target-selected launch
   configuration per shape bucket (descriptor stays informational, A12).
3. **Compute-mechanism vs format-math split in `inference/kernels`.**
   `mma/tma/dtype/tensor_map` are target-private compute layers hiding inside
   shared headers; `norm/activation/topk/head/route/gqa` are genuinely
   portable. Splitting them is the same surgery as the mandated
   `spark_lm_kernels.cuh` landmine split — do them together so neutral
   machinery stays shared while arch selection moves under target control
   (§5 sweep grant).
4. **Copy/pinned/completion boilerplate.** Every module host + serving
   adapter + `qwen36_tp.c` re-rolls memcpy/pinned-alloc/callback patterns
   (~1.9k CUDA tokens total across driver sources). The S1 primitives delete
   most of it per driver; do not write new copies in remaining drivers.
5. **Dead/duplicated kernels.** `LmGatherRowsKernel` definition +
   instantiation (`kimi_k3/unity.cu:127`) is test-forbidden (TOP10 #3);
   GLM52 dspark's hand-rolled reductions duplicate `LmBlockSum`
   (TOP10 #2/#4). Deleting before porting shrinks the audited surface.
6. **gemm stacks.** `runtime/gemm.cuh` launcher + descriptor cache vs
   `inference/kernels/gemm.cuh` tile bodies vs expert GEMMs in the landmine
   header: keep one launch policy, move tile selection fully target-side
   (post-split), so ROCm MFMA and CUDA SM121 paths stop sharing selectors.

---

## 6. Notes & risks

- Estimates are analytical, assume the plan §2 preconditions (landed
  `spark_hw_iface.h`, link-unit split, split landmine header, provisioned
  MI350P node) and exclude per-target perf qualification time.
- The k3 host module being CUDA-free shows the target end-state is reachable
  per driver without weakening R2; dsv4's host is the counter-example to fix
  first because it carries the most mass (489 tokens).
- No mixed-vendor collectives at any step (R6/R7); each new driver adopts
  hardware-agnosticism as its own homogeneous deployment.
