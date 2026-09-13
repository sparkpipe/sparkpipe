# SparkPipe Kernel Playbook

Owned by the CUDA-KERNELS agent (charter: `.agents/AGENT_CHARTER.md:17-20`). This is
the single living source of every kernel trick, performance receipt, and kernel
inventory in the tree. Model and subsystem agents request kernels by filling in
**Part 1 (the contract)**; they do not re-derive anything in Parts 2-3. The
CUDA-KERNELS agent maintains Parts 2-3 and extends Part 3 every time a kernel
lands or a measurement changes.

All claims below are cited `file:line` against the unified-branch clone at
`.agents/cuda-kernels/`. Read `PERFORMANCE_STATUS.md` for the canonical
performance ledger (`PERFORMANCE_STATUS.md:1-6`); this playbook distills it,
does not replace it.

---

# Part 1 — Kernel contract template (requestors MUST fill)

Copy this block into the request. Every field is required unless marked
optional. A contract with missing shapes, dtypes, or a reference is sent back.

```
## Kernel contract

requestor:      <agent name + lane (MODEL/SUBSYSTEM)>
op:             <one line: what the kernel computes, e.g. "grouped expert w1/w3
                (SiLU gate * up) GEMM", "RMSNorm + RoPE + FP8 QDQ fusion">
source:         <file:line of the caller / driver launch site, if any>

## Shapes
rows:           <batch rows, and whether B1/B8/B1024 decode or prefill>
input_dimension:<K, the contraction length; must state K-alignment>
output_dimension:<N for the GEMM; or per-op dims>
top_k:          <routed experts per token, 0 if dense>
experts:        <expert count, 0 if dense>
extra dims:     <heads, KV heads, head_dim, value_dim, rank, window, vocab, ...>

## dtypes
activation:     <BF16 | FP8 E4M3 (UE8M0, group __) | MXFP4 | ... ; stored bits>
weight:         <BF16 | FP8 E4M3 | MXFP4 E2M1 | NVFP4 | INT4/6/7/8; stored bits>
scale:          <none | UE8M0 | UE4M3 | F32-per-B128 | E8M0 ; group size>
accumulator:    <F32 | S32 (integer GEMM only)>
output:         <BF16 | FP8 codes+scale | ...>

## precision route (exact target string)
                e.g. cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.
                     expert_mxfp4.kv_bf16   (see PERFORMANCE_STATUS.md:334)
rounding:       <where BF16 rounding is preserved; where fp32 is held>
bias/scale:     <which biases/scales fold in, and the exact fold order>
codec:          <ACTIVATION_CODEC or NONE; group size; QDQ in-place or not>

## memory / layout
strides:        <row strides, column offsets, group strides; explicit, no default>
layout:         <row/col-major, interleaved cell grid, expert-major, KV slot layout>
indirection:    <source_row_map / route_source_token? scale rows follow source? (yes/no)>
alignment:      <K % 128 == 0? width % TILE_N == 0? row pitch % 16 == 0?>

## target number
target:         <the number to beat, with unit: tok/s, us/layer, GB/s, ms/step>
baseline:       <measured baseline this must exceed, if known>
constraints:    <static-shared <= 48 KB, occupancy, one-launch, DRY (no model name)>

## reference
ref:            <torch / flashinfer / vLLM / modeling file; exact identity>
tolerance:      <bit-exact, byte-exact, or RMS % + at what precision>
test gate:      <which test / GA validator / O128 exact token hash must pass>
```

Notes the implementer applies (all verified in-tree):

- **The precision route is a string, not adjectives.** The merged-main DSV4 Flash
  target is `linear_fp8.expert_mxfp4.kv_bf16` (`PERFORMANCE_STATUS.md:334`,
  `428`). GLM 5.2 is `bf16.expert_fp8` (`PERFORMANCE_STATUS.md:580`). Name it.
- **K-alignment is a hard fork in the dispatcher.** The tile path requires
  `input_dimension % TILE_K == 0`; non-aligned widths silently route to the
  scalar path (`spark_lm_kernels.cuh:5018-5025`, `4986-4990`). State it up
  front or the wrong path runs.
- **Row stride has no default.** `LmFusedResidualRmsNormKernel` takes an explicit
  `row_stride` because a self-filling stride was the "correct at rows==1, corrupt
  above" bug class (`norm.cuh:81-90`). Requestors state strides explicitly.
- **The DRY law:** shared kernels never name a model (`.agents/AGENT_CHARTER.md:34`).
  Facts (dims, gates, activation choices) go in tables; the kernel takes them as
  parameters or template constants. `SparkLmSm121NativeDecodeShape` is the
  qualified-batch-shape table pattern (`spark_lm_kernels.cuh:2313-2318`).

---

# Part 2 — Tricks and receipts (distilled)

## 2.1 Hardware + ptxas capability facts (the ground truth)

| Fact | Value | Source |
| --- | --- | --- |
| Compute | `compute_121a` / `sm_121a`, 48 SM, GB10 | receipts (`compute_capability: 12.1`); `docs/archive/PERF_ROADMAP_2026-08-01.md:29` |
| DRAM | 273 GB/s LPDDR5x unified, 128 GB | `PERFORMANCE_STATUS.md:597`; `docs/archive/PERF_ROADMAP_2026-08-01.md:29` |
| Linear path | ~6.5 TFLOP/s, ~30 FLOP/byte before compute binds | `project.cuh:550-556` |
| wgmma / tcgen05 | **NOT available** on sm_121a | `mma.cuh:21-25`; `docs/archive/HANDOFF_KV_SHARING_AND_KERNELS.md:240-241` |
| NVFP4 scale_vec::4X + ue8m0 | **ptxas rejects** on sm_121a (sm_100 capability) | `mma.cuh:229-234` |
| static shared ceiling | 48 KB/block (not the SM's 128 KB); ptxas errors "0xc000 max" | `tile.cuh:40-48`; `layout.cuh:17-21` |
| capability gate | `tests/test_ptx_capability_gate.py` fails the build if a required PTX form stops assembling, or if wgmma/tcgen05 appears | `tma.cuh:19-20`; `docs/archive/HANDOFF_KV_SHARING_AND_KERNELS.md:243-245` |

**The SM121 native-compute gate.** Block-scaled MXF8/MXF4 mma atoms compile to a
`trap` outside `__CUDA_ARCH__ == 1210` — never a silent BF16-dequant fallback
that could be mistaken for the qualified route (`mma.cuh:34-46`, `129-145`).

## 2.2 TMA + mbarrier pipeline (`inference/kernels/tma.cuh`, `tile.cuh`)

- **Why TMA:** `cp.async` is per-thread and burns issue slots per tile byte; TMA
  moves a whole tile with one instruction from one elected thread
  (`tma.cuh:3-16`).
- **Elect thread 0, never `elect.sync`:** warp-scoped elect duplicates TMA
  transactions and barrier arrivals in a multi-warp CTA (`tma.cuh:57-59`).
- **Arrive/expect/parity protocol.** Declare the byte total with
  `LmMbarrierArriveExpect`; it must equal the sum of every box issued into the
  stage or the barrier never flips (deadlock) (`tma.cuh:82-92`). Byte totals are
  always derived via `LmTileBytes`, never literal (`tile.cuh:137-148`).
- **`try_wait` returns a predicate, not a block:** the spin lives in C++, avoiding
  inline-asm label collisions when the function is inlined twice (`tma.cuh:101-111`).
- **2D box is bounds-checked and zero-fills** out-of-range elements — a ragged group
  tail is safe with no branch and no epilogue (`tma.cuh:119-131`).
- **3D box = expert-major weights.** Coordinate 2 selects the expert, so one
  descriptor covers all experts; the grouped dispatch never rebuilds a tensor map
  per group (`tma.cuh:133-142`; `tile.cuh:144-148`).
- **`LmTmaLoadBulk1d`** is the linear-address variant for indirect-A grouped GEMM
  (a per-row index the affine box cannot express); 16-byte swizzle chunk, **no
  hardware bounds check**, caller clamps the index or a wild copy faults
  (`tma.cuh:31-36`, `144-158`).
- **Store side:** `bulk_group` completion + `wait_group.read` (cheaper .read form,
  only guarantees the staging buffer is reusable) (`tma.cuh:160-182`).
- **Two fences:** `fence.proxy.async.shared::cta` orders the generic-proxy barrier
  init vs the async-proxy completion write, and shared writes vs the TMA store
  read (`tma.cuh:73-80`, `184-189`).

**The staged-tile pipeline (`tile.cuh`):** one implementation shared by every GEMM
(replacing four divergent copies, three of which never ran) (`tile.cuh:3-9`).

- **Schedule:** stage `s` holds K tile `s, s+STAGES, ...`; before consuming tile
  `t` the loop issues `t+STAGES-1` — a transfer is always in flight across the
  wait. The CTA-wide `__syncthreads` each stage already needs is what proves the
  refilled stage was consumed; one mbarrier per stage, nothing else
  (`tile.cuh:11-16`).
- **Depth = 2 stages, lookahead 1.** Little's Law vs 218 GB/s and 400-600 ns latency
  wants ~2.3 KB in flight per SM. Six stages of a 16-row NVFP4 tile is 110 KB
  (one CTA/SM); two stages is 37 KB (three CTAs/SM) (`tile.cuh:18-24`,
  `50-51`). The latency figure has never been measured — that is the one number
  that could move this.
- **48 KB static ceiling is the real occupancy limiter.** Exceeding it needs dynamic
  shared + `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` (`tile.cuh:40-48`).
- **Per-stored-width tile table** (16/8/7/6/4 bits → TILE_K, pitch, span, static-fit)
  (`tile.cuh:107-117`). Sub-byte non-power-of-two widths (INT7: 224 B row) cannot
  take a 128-B span; a 32-B span still removes most bank conflict (`mma.cuh:455-463`).

**Staging variants (the interleaved weight staging is the headline trick):**

- **Interleaved weight staging (pack V2 `mxfp4_ws_interleaved_v1`).** B is a cell
  grid, not a `[neuron,k]` plane: per expert and per 128-element pack k-tile, each
  16-neuron cell is 17 rows of 64 B — 16 payload rows (one neuron's 128 4-bit
  elements) + one row of sixteen 4-byte E8M0 group scales. One rank-3 UINT8 tensor
  map covers the whole expert operand; the staged layout IS the cell layout
  (`tile.cuh:171-203`).
- **K coordinate is in BYTES per operand** (descriptors are UINT8), so a BF16 A
  against a 4-bit B advances four times as far per K tile (`tile.cuh:222-228`).
- **Two-block A stage** when interleaved TILE_K=128: a 256-byte BF16 row cannot be
  one box, so blocks hold k 0..63 and k 64..127, with the `r % 8` selector shared
  by both blocks (`tile.cuh:237-258`).
- **Indirect-A staging = the MoE gather deletion.** When `LmGemmArguments` carries a
  `source_row_map`, packed row `p` is staged from the UN-gathered source tensor
  row `route_source_token[p]` — one `cp.async.bulk` per 16-B swizzle chunk; no
  packed activation tensor is written, no host sees the route (`tile.cuh:262-363`;
  `route.cuh:18-28`). Chunks are issued round-robin so the global side coalesces;
  expect-vs-copy order does not matter because the tx count is signed
  (`tile.cuh:279-283`). Ragged tail rows clamp to `row_base` (dead traffic, never
  wrong); a mapped row past `source_row_count` traps (`tile.cuh:285-292`,
  `333-334`).
- **Grouped tile scheduling is binary search O(log G)** over a device prefix; the
  prefix holds the true tile count so no host-side average launches empty groups
  (`tile.cuh:418-448`).

## 2.3 MMA atoms and fragment mapping (`inference/kernels/mma.cuh`)

- **The register↔matrix mapping is the silently-wrong part of mma.sync.** Every
  mapping is transcribed from CUTLASS `MMA_Traits` and checked element-by-element
  (bijection + negative control) by `tests/test_mma_fragment_mapping.c`
  (`mma.cuh:3-16`). Sources listed at `mma.cuh:12-16`.
- **sm_121a selects `SM120_16x8x32_TN`** (byte-identical layout to the SM89 atom it
  inherits; the verifier asserts it) (`mma.cuh:18-19`).
- **8-bit atom m16n8k32:** four contiguous bytes per register → plain aligned 32-bit
  loads, **not ldmatrix** (a second address derivation is a second chance to be
  silently wrong) (`mma.cuh:63-73`).
- **4-bit atom m16n8k64:** half the width, twice the depth, identical register
  footprint as the 8-bit atom (`mma.cuh:174-183`).
- **BF16 atom m16n8k16:** each register holds TWO consecutive k values — the whole
  decode path rests on one aligned 32-bit read serving a register, and a packed
  format serving the same register by extracting two adjacent codes (`mma.cuh:312-331`).
- **Free BF16 dequant (Marlin / Kim et al. trick):** a ≤7-bit signed code is turned
  into BF16 by OR-ing into mantissa of `0x4300` (value 128) — `128+m = 128*(1+m/128)`;
  the bias subtraction folds into the scale multiply as one fma. Saves two
  conversion instructions per value (`mma.cuh:354-372`). **BITS must be ≤7** or the
  code overflows into the exponent and doubles — the static_assert and the reason
  INT7 is the sweet spot (`mma.cuh:374-393`).
- **`LmStoreCodeOctet`** writes 8 codes into a thread-exclusive byte-aligned block;
  the old pair writer updated overlapping 32-bit words and adjacent threads lost
  each other's bits (`mma.cuh:404-431`).
- **Swizzle** is applied in exactly one place (`LmSwizzledOffset`); 16-B chunk xor
  the row's 128-B sector selector, justified by count (without it a 32-lane load
  puts 16 lanes on one bank; with it, 4) (`mma.cuh:446-467`).

## 2.4 Top-k, route, speculative-decode, GQA, projection tricks

**Top-k (`topk.cuh`)** — one algorithm, three shapes, was written four times
(`topk.cuh:3-20`):

- k=8/256 → bitonic sort in registers; k=2048/128k → radix; k=1/154880 → max
  reduction ("radix with one bucket") (`topk.cuh:14-19`).
- **Radix on float bits:** IEEE bits are monotonic for non-negatives; flip sign +
  invert negatives makes them monotonic everywhere → one pass over 8 bits narrows
  128k candidates, a second finishes (`topk.cuh:22-27`, `57-67`).
- **Bias selects, it does not weigh.** The router adds a correction bias to pick
  top-k but gathers mixture weights from the UNBIASED scores (KDA report); folding
  the bias into the sorted key leaks it into every mixture weight (`topk.cuh:91-99`).
- **Sigmoid in-pass, not a separate pass** (two trips over experts×rows floats per
  layer otherwise) (`topk.cuh:99-104`).
- **Grouped selection ranks by top-2 SUM** (recovered from keys, not summed as keys)
  (`topk.cuh:110-135`); **rank by counting, not by selecting TOP_GROUPS times**
  (was serial quadratic) (`topk.cuh:138-157`).
- Renormalise uses a **block reduction, not one thread** walking K (`topk.cuh:198-226`).

**Route build (`route.cuh`)** — a counting sort over 896 buckets: count, serial
prefix, scatter in shared; 896 additions is not a scan problem (`route.cuh:6-17`).
The **row-indirection consumer contract** is stated exactly once (`route.cuh:18-69`):
order within an expert is atomic-arrival order (no stability/sortedness
assumptions); **scale rows follow the source** — an indirect A-read must index
`scale_a` by `route_source_token[p]`, not `p`, or another token's scale is
applied with no fault (`route.cuh:56-62`).

**Speculative decode (`speculate.cuh`)** — MTP and DSpark are one algorithm; the
drafter is a policy, the verifier is a kernel (`speculate.cuh:3-13`):

- **Greedy acceptance is a comparison; sampled acceptance is the modified
  rejection rule** that preserves the target distribution. Greedy is NOT a special
  case of sampled — pretending it is is how bias is introduced (`speculate.cuh:15-28`).
- **Bonus token:** write the target's own argmax at the divergence point, so a fully
  rejected draft still advances by one and speculation never loses (`speculate.cuh:55-59`).
- **Cache rollback is the part that bites:** a rejected draft token's KV is already
  written; context length must be set from the accepted count and nothing else
  (`speculate.cuh:23-28`, `60-63`).
- Sampled rejection resamples from the positive part of `(target - draft)`, not the
  raw target (raw target over-proposes drafter-preferred tokens) (`speculate.cuh:105-107`).

**GQA (`gqa.cuh`)** — the KV slot layout is a contract, stated once:
`[key: KV_HEADS×HEAD_DIM][value: KV_HEADS×VALUE_DIM]` bf16, static_asserted at both
store and decode (`gqa.cuh:16-27`, `104-105`, `149-150`). Decode is online softmax
over the cache once, with the accumulator **exactly sized** (VALUE_DIM is a template
arg, not a fixed "8 slots" that silently drops wider tails) (`gqa.cuh:124-147`).

**Projections (`project.cuh`)** — raw vs absorbed low-rank projection; absorbed is
strictly better at decode (saves materialising per-head K/V) and strictly worse at
prefill (`project.cuh:12-30`). **Per-head block-diagonal value projection stays
unabsorbed** because elementwise output gating does not commute with the fold, and
on GB10 absorbing the value half inflates the output projection 8.30→19.55 GB
(55 ms/token) to save a 302 MFLOP kernel (46 us) — three orders of magnitude the
wrong way (`project.cuh:538-556`).

## 2.5 Norm / linear-attention / MoE-finalize tricks (`norm.cuh`, `linear_attn.cuh`)

- **Every output path is a Format trait:** a norm feeding BF16 writes BF16; feeding a
  quantised GEMM writes packed codes + block scale — same kernel, different Format
  (the old tree had four near-identical fusions) (`norm.cuh:15-19`).
- **Fused residual+RMSNorm** reads the residual once and writes the normed + residual
  in one pass (`norm.cuh:69-78`). **Fused norm+quantise** saves 225 MB/token at 6144
  hidden / B128 (3% of the 5.3 GB weight stream) for zero extra work
  (`norm.cuh:368-377`).
- **`LmQuantiseRowsKernel` static_asserts `kScaleGroup > 0`** — the general form of
  the divide-by-zero bug that shipped when an unquantised BF16 format reached the
  quantiser (`norm.cuh:267-278`).
- **MoE finalize** folds top_k packed rows back to token-major with the router gate
  weights; omitting it looks like a routing bug (fluent, wrong) (`norm.cuh:323-334`).
- **`LmBlockSum` / `LmBlockMax`:** warp shuffles first, one round through shared
  (shared traffic is warps, not threads) (`norm.cuh:27-67`).
- **Delta rule** updates the state with the prediction error `S = α·S + β·(v−Ŝᵀk)·kᵀ`;
  state is key-major so the outer-product update is contiguous (`linear_attn.cuh:16-39`).
- **Bounded decay `g_min=-5`** keeps the chunkwise reciprocal inside BF16 range; a
  "plausible" softplus-for-sigmoid substitution reintroduces the overflow the
  architecture removed (`linear_attn.cuh:45-65`).
- **ReplaySSM stores the inputs, not the state** (~1 KB vs 3 MB/step/layer); the fold
  must **not recompute the gate** — SGLang's drift story (`linear_attn.cuh:190-210`,
  `76-93`). Fold byte-exactness vs serial decode: same strided partials + same
  `LmBlockSum` at the same THREADS, and the bf16 store rounds once per committed
  step via `LmStoreState` (`linear_attn.cuh:219-248`, `291-299`).
- **One kernel for decode, prefill, verify** (delta rule + causal conv): a run of T
  is bit-identical to T decode calls because the state lives in shared at slot width;
  verify runs `commit==0` and abandons state (`linear_attn.cuh:341-356`,
  `528-536`). **Two reduction buffers, not one** — LmBlockSum's last read of
  `shared[0]` is after its final barrier; a second call can overwrite before another
  warp reads (`linear_attn.cuh:379-383`).

## 2.6 `spark_lm_kernels.cuh` — expert tiles, pair policies, Mloop

Header extracted verbatim from the audited K3 resident decode stage, renamed
`SparkLm*`; each driver module includes it into its own TU → one source, zero ABI
coupling, zero runtime cost; MXFP4 group size is a template parameter so a
disagreeing module fails to build (`spark_lm_kernels.cuh:14-28`). Consumers:
qwen38_27b, dsv4, mimo25 (`spark_lm_kernels.cuh:25-27`).

**Expert-tile bodies (two live schedules, `SPARK_LM_EXPERT_TILE_POLICY`):**

- **All-warps:** 16-slot × 128-neuron accumulator, K staged 64 wide; each warp owns a
  16-neuron column slice, weight stage decodes each neuron's K-run ONCE into shared
  bf16 (thread t owns 32 contiguous elements of neuron t mod 128), reused by all 16
  gathered rows; missing rows/neurons stage zeros, stores guarded
  (`spark_lm_kernels.cuh:3585-3706`).
- **Software-pipelined:** double-buffered tile, warps 0-3 compute while warps 4-7
  stage the next K strip (producer warps split into two neuron halves)
  (`spark_lm_kernels.cuh:3708-3889`). **AUTOMATIC** selects all-warps for
  `input_dimension <= TILE_K`, pipelined otherwise — both are live, neither removable
  without a measurement (`spark_lm_kernels.cuh:4278-4349`).
- **One-row grouped expert tile** (scalar, B1): the tensor-core path is M=16, so B1
  pads 15 rows; the grouped-scalar path keeps one CTA per 128-neuron tile and stages
  the source activation once, walking neurons with 8 warps (`spark_lm_kernels.cuh:4357-4365`,
  `4380-4440`).
- **All-expert tile** makes the routed w1/w3/w2 phase ONE launch with no host
  knowledge of grouping; gridDim.z spans the expert table, empty tiles exit in a
  few cycles (`spark_lm_kernels.cuh:4442-4465`).

**SM121 native block-scaled MXF8/MXF4 expert kernels** (`spark_lm_kernels.cuh:2350-3359`):

- `SparkLmSm121StageMxf8`: a warp owns a row, scale is the exact row-block amax,
  payload written is the E4M3 byte form mma consumes; indirect routed rows follow
  `source_row_map`, ragged rows become numeric zero (`spark_lm_kernels.cuh:2350-2407`).
- **E2M1 must sit in register bits [5:2]** of each 8-bit container for
  `kind::mxf8f6f4` — the checkpoint stays nibble-packed until the two padding bits
  per side are inserted (`spark_lm_kernels.cuh:2460-2467`).
- **B1 gets a GEMV, not a padded M16 tensor-core atom:** "True B1 is a matrix-vector
  product; padding it to the M16 atom multiplies activation work by sixteen and is
  slower on GB10" (`spark_lm_kernels.cuh:5159-5164`). Qualified native shapes are
  rows ∈ {1,5,7,8,16,32,64,1024} (`spark_lm_kernels.cuh:2313-2318`).

**Pair policies** (two GEMMs sharing one staged activation) (`spark_lm_kernels.cuh:40-43`,
`5070-5086`): `FLAT_8` (8 warps) vs `FLAT_16` (16 lanes flat) selected on
`row_count==1` and `input×output` work vs a wide threshold
(`SPARK_LM_FP8_PAIR_WIDE_MINIMUM_WORK 1572864`, `SPARK_LM_BF16_PAIR_WIDE_MINIMUM_WORK 524288`).

**Mloop launchers (dense + grouped)** (`spark_lm_kernels.cuh:3891-4276`,
`5088-5111`):

- **Dense M-group loop:** one CTA owns M_GROUP=8 m-tiles of one n-tile and stages
  each k-stage's weight strip ONCE, reusing it for every m-tile's MMA. The plain
  grid re-reads a dense weight 16× at B=256; Mloop cuts amplification to
  ceil(m_blocks/M_GROUP). BF16 only, ≥2 m-tiles, K past one tile (`spark_lm_kernels.cuh:3891-3898`,
  `5088-5099`).
- **Grouped expert m-loop:** grid.x is ONE (n-tiles × experts), each CTA walks its
  expert's row group in chunks of M_GROUP m-tiles. The plain grid launched ~122K
  empty CTAs at B=256 at ~1.25 us each — the measured 233 GB/s collapse
  (`spark_lm_kernels.cuh:4084-4090`). Expert-stride CTA budget: 64 CTAs walk up to
  expert_count/64 experts; empty experts cost a few cycles, not a launch
  (`spark_lm_kernels.cuh:4094-4098`, `5101-5111`).

**Weight read-ahead** (`spark_lm_kernels.cuh:71-100`): immutable weights read on an
auxiliary stream during exposed collective windows to warm cache; every thread
publishes a checksum into private scratch to stop the compiler deleting the loads
while keeping inference outputs disjoint; one uint4 load warms each 32-B GB10 cache
sector.

**Fused head (`spark_lm_kernels.cuh:4638-4688`):** matvec against hidden + running
argmax, no logits tensor materialised; each warp owns a candidate stripe, reduces
bests through shared. **Certified FP8 head screen** (`spark_lm_kernels.cuh:1500-1680`):
FP8 used only to form a conservative candidate set; every survivor is rescored
against the untouched BF16 head (`PERFORMANCE_STATUS.md:196-200`).

## 2.7 Measured receipts (isolated + end-to-end)

**End-to-end TP4 B1 decode (canonical ledger, `PERFORMANCE_STATUS.md`):**

| Milestone | tok/s | Source |
| --- | ---: | --- |
| Selected exact TP4 B1 lean stack (retained) | **40.4553** | `PERFORMANCE_STATUS.md:97-166` |
| pre-post stack vs resident chain | +5.9266% (40.2541) | `PERFORMANCE_STATUS.md:108` |
| combined + LinearPair (branch, da7f) | 39.5202 (+4.0561%) | `PERFORMANCE_STATUS.md:168-184` |
| resident decode chain (branch, a14c2e1) | 38.1059 (+11.95%) | `PERFORMANCE_STATUS.md:280-297` |
| device-predicated compressor (branch) | 33.9911 | `PERFORMANCE_STATUS.md:384-397` |
| merged-main ed1d731d | 33.5505 | `PERFORMANCE_STATUS.md:322-355` |
| scratch (7bf94d8) | 32.5674 | `PERFORMANCE_STATUS.md:418-451` |
| full-graph regression (07696e0, removed) | 27.7883 (−17.46%) | `PERFORMANCE_STATUS.md:499-525` |

All DSV4 runs emit the exact 128-token vector, SHA-256
`211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083`
(`PERFORMANCE_STATUS.md:130-132`).

**GLM 5.2 TP8 (bandwidth-bound, different profile):** B1 single-stream 6.91 tok/s
(~15.7 GB FP8 experts + ~6 GB BF16 spine per token at 273 GB/s); B8 aggregate
43.46 tok/s; B16 aggregate 75.55 tok/s (`PERFORMANCE_STATUS.md:567-628`).

**Isolated kernel fusion probes (NOT end-to-end; each is a bit/byte-exact A/B):**

| Fusion | control → candidate | gain | Receipt |
| --- | --- | ---: | --- |
| KV RMSNorm+RoPE+FP8-QDQ-sim | 0.485 → 0.177 ms | **+63.52%** | `qualification/.../kv_post_fusion_probe_da7f/receipt.json` |
| indexer RoPE+Hadamard+FP4-QDQ | 0.259 → 0.130 ms | **+49.90%** | `qualification/.../indexer_post_fusion_probe_da7f/receipt.json` |
| query RMSNorm+RoPE | 0.330 → 0.177 ms | **+46.42%** | `qualification/.../query_rms_rope_fusion_probe_da7f/receipt.json` |
| Hc residual copy fused into pre-reduce | 7.649 → 7.632 ms | +0.22% | `qualification/.../hc_residual_fusion_probe_da7f/receipt.json` |
| Hc row-adaptive boundary (pre-reduce) | 6.154 → 2.058 us | — | `PERFORMANCE_STATUS.md:68-81` |
| Hc row-adaptive boundary (post) | 8.199 → 3.305 us | — | `PERFORMANCE_STATUS.md:68-81` |

**CUPTI profile of a 39.0094 tok/s candidate** (category shares of summed kernel
time; can overlap; `PERFORMANCE_STATUS.md:219-241`): dense/projection GEMVs
38.65%, routed experts 21.08%, attention/KV 10.81%, weight read-ahead 9.48%,
routing/gate 7.93%, compressor/Hc 5.11%, output head 3.04%, collective local
combine/pack 2.21%, norm/embedding 1.55%, setup 0.14%.

**Transport (measured, `PERFORMANCE_STATUS.md:7-93`):** eight pairwise direct links
91.669–105.907 Gb/s/direction, 1.643 Tb/s combined; TP4 two-port ceiling 213.687
Gb/s (90.328% util at 14 MiB); crossover — direct all-to-all ≤80 KiB, recursive
doubling 80–640 KiB, counter-rotating split ring ≥640 KiB. TP4 thresholds are not
universal (TP8/TP16 need their own profiles, `PERFORMANCE_STATUS.md:92-93`).

**Rejected-at-production-boundary optimisations (measured, bit-exact, but regressed
end-to-end — do not re-propose without new evidence):** routed-only cooperative
W13→W2 (regressed 0.83%), persistent dense/projection bundle (−0.78%),
predeclared weight-read-ahead queue (−33.2%), projection-precollective overlap
(−0.58%), cooperative Hc finalize (−0.33%), Query fusion (−0.72%)
(`PERFORMANCE_STATUS.md:111-115`, `144-156`, `243-274`).

---

# Part 3 — Kernel registry (inventory)

Status convention: **measured** = a number exists in `PERFORMANCE_STATUS.md` or a
qualification receipt; **NOT_MEASURED** = no number yet (assembler/host-tested only).
The CUDA-KERNELS agent updates this table whenever a kernel lands or a number changes.

## 3.1 Shared GEMM / staging / MMA infrastructure (`inference/kernels/`)

| Kernel / primitive | File:line | What it is | Number |
| --- | --- | --- | --- |
| `LmTmaLoad2d/3d/Bulk1d`, `LmTmaStore2d` | `tma.cuh:125-168` | TMA box + bulk staging, mbarrier complete_tx | NOT_MEASURED |
| `LmMbarrier*` set | `tma.cuh:61-117` | mbarrier init/arrive/expect/try_wait | NOT_MEASURED |
| `LmPipelineProduce*` | `tile.cuh:149-416` | staged-tile produce (dense/grouped/interleaved/indirect/codec) | NOT_MEASURED |
| `LmGroupOfTile` / `LmTotalTiles` | `tile.cuh:431-448` | O(log G) tile→group | NOT_MEASURED |
| mma atoms (E4M3, Mxf8Mxf4/8, Nvfp4, Mxfp4, S8, S4, E3m2/E2m3, BF16) | `mma.cuh:99-444` | fragment-mapped mma.sync atoms | fragment-mapping host-tested; perf NOT_MEASURED |
| `LmCodeToBf16Bits` ≤7-bit dequant | `mma.cuh:379-393` | free BF16 dequant | NOT_MEASURED |

## 3.2 Elementwise / norm / top-k / route / attention / linear-attn / spec

| Kernel | File:line | What it is | Number |
| --- | --- | --- | --- |
| `LmFusedResidualRmsNormKernel` | `norm.cuh:91` | residual+RMSNorm, explicit stride | NOT_MEASURED |
| `LmLayerNormKernel` | `norm.cuh:121` | LayerNorm (DSA index keys) | NOT_MEASURED |
| `LmSiluMulKernel` / `LmSituMulKernel` | `norm.cuh:154`, `191` | SiLU·up / SiTU·up | NOT_MEASURED |
| `LmQuantiseRowsKernel` / `LmFusedNormQuantiseKernel` | `norm.cuh:265`, `385` | absmax QDQ / fused norm+QDQ | NOT_MEASURED |
| `LmMoeFinalizeKernel` | `norm.cuh:337` | weighted fold of top_k rows | NOT_MEASURED |
| `LmAttnResKernel` | `norm.cuh:490` | attention residual bank softmax | NOT_MEASURED |
| `LmTopkSmallKernel` (bitonic) | `topk.cuh:82` | router top-k (k=8/256) | NOT_MEASURED |
| `LmTopkHistogramKernel` / `LmTopkGatherKernel` | `topk.cuh:248`, `283` | radix top-k (k=2048/128k) | NOT_MEASURED |
| `LmRouteBuildKernel` | `route.cuh:95` | 896-bucket counting sort | NOT_MEASURED |
| `LmGqaKvStoreKernel` / `LmGqaAttentionDecodeKernel` | `gqa.cuh:102`, `136` | per-head KV store / GQA decode | NOT_MEASURED |
| `LmBoundedDecayKernel` / `LmGdnGateKernel` | `linear_attn.cuh:106`, `135` | retention/gate producers | NOT_MEASURED |
| `LmReplayFoldKernel` | `linear_attn.cuh:251` | rewindable-state fold | NOT_MEASURED |
| `LmDeltaRuleKernel` | `linear_attn.cuh:339` | gated delta rule decode/prefill/verify | NOT_MEASURED (decode 113→423 tok/s claimed via ReplaySSM, `linear_attn.cuh:192-194`) |
| `LmCausalConvKernel` | `linear_attn.cuh:526` | 4-tap causal conv | NOT_MEASURED |
| `LmLowRankProject` / `LmAbsorbedProject` | `project.cuh:154`, `342` | raw/absorbed MLA projection | NOT_MEASURED |
| `LmSplitQkvKernel` / `LmPerHeadProjectKernel` | `project.cuh:471`, `567` | fused-QKV split / per-head project | NOT_MEASURED |
| `LmSpeculativeVerifyGreedyKernel` / `SampledKernel` | `speculate.cuh:44`, `78` | speculative accept + KV rollback | NOT_MEASURED |

## 3.3 `spark_lm_kernels.cuh` expert/linear/head/MoE kernels

| Kernel | File:line | What it is | Number |
| --- | --- | --- | --- |
| `SparkLmWeightReadAheadKernel` | `spark_lm_kernels.cuh:79` | auxiliary-stream cache warm | measured share 9.48% of kernel time (`PERFORMANCE_STATUS.md:235`) |
| `SparkLmSm121NativeLinearKernel` | `spark_lm_kernels.cuh:2599` | MXF8/MXF4 tensor-core linear (B8+) | NOT_MEASURED standalone |
| `SparkLmSm121FusedDenseW13Kernel` / `FusedDenseW13GemvKernel` | `spark_lm_kernels.cuh:2670`, `3016` | fused SiLU·up (dense) | NOT_MEASURED standalone |
| `SparkLmSm121FusedExpertW13Kernel` / `B1ExpertW13Kernel` | `spark_lm_kernels.cuh:2747`, `2918` | fused expert w1/w3 (routed) | measured share "routed experts 21.08%" (`PERFORMANCE_STATUS.md:233`) |
| `SparkLmSm121ExpertW2Kernel` / `B1ExpertW2Kernel` | `spark_lm_kernels.cuh:3064`, `2988` | expert w2 (routed) | measured share "routed experts 21.08%" |
| `SparkLmExpertTileBodyAllWarps` / `SoftwarePipelined` | `spark_lm_kernels.cuh:3595`, `3709` | row-tiled expert GEMM (BF16) | NOT_MEASURED standalone |
| `SparkLmExpertTileMloopKernel` | `spark_lm_kernels.cuh:3900` | dense M-group (qwen38 spine) | measured collapse fixed: plain grid 233 GB/s → Mloop (`spark_lm_kernels.cuh:4088`) |
| `SparkLmExpertTileAllMloopKernel` | `spark_lm_kernels.cuh:4092` | grouped expert M-loop (qwen38 FP8) | same collapse receipt |
| `SparkLmGroupedScalarLinearKernel` | `spark_lm_kernels.cuh:4381` | one-row grouped scalar expert | NOT_MEASURED standalone |
| `SparkLmMoeGroupKernel` | `spark_lm_kernels.cuh:4477` | device histogram+prefix+scatter | NOT_MEASURED standalone |
| `SparkLmMoePairReduceKernel` (+Strided/Overwrite) | `spark_lm_kernels.cuh:4514`, `4548`, `4592` | inverse-map weighted fold | NOT_MEASURED standalone |
| `SparkLmHeadArgmaxKernel` | `spark_lm_kernels.cuh:4646` | fused matvec + argmax head | measured share "output head 3.04%" (`PERFORMANCE_STATUS.md:238`) |
| `SparkLmHeadCertifiedFp8*` | `spark_lm_kernels.cuh:1500-1680` | certified FP8 head screen + rescore | measured share "output head 3.04%" |

## 3.4 Where numbers go next

When a kernel lands, add a row here with `measured: <value> <unit> @ <shape/dtype>
(<receipt path or NOT_MEASURED>)`. Never record a projected figure as measured —
`PERFORMANCE_STATUS.md:1-5` and the 50 tok/s gate discipline
(`PERFORMANCE_STATUS.md:276-278`, `545-556`). The planning projections (TP8 60
tok/s, TP16 83 tok/s) are projections, not measurements (`PERFORMANCE_STATUS.md:527-543`).

---

# Appendix — how to implement against this playbook

1. **Read the contract, not the caller.** Shapes/dtypes/precision-route/target/reference
   come from Part 1. If the requestor skipped a field, ask — do not infer (the
   "correct at rows==1" bug class is exactly what inference produces, `norm.cuh:81-90`).
2. **Reuse the staged pipeline** (`tile.cuh`) and mma atoms (`mma.cuh`) — never
   hand-roll a second mbarrier protocol or a second fragment mapping (both are the
   two documented ways to be silently wrong: `tile.cuh:3-9`, `mma.cuh:3-16`).
3. **Respect the hard constraints:** static shared ≤48 KB, K % 128 == 0 for the
   native MXF path, sm_121a has no wgmma/tcgen05/4X+ue8m0, trap outside
   `__CUDA_ARCH__==1210`, no model names in shared code.
4. **Measure at the right boundary.** Isolated kernel fusion probes are the cheap
   first gate (see §2.7 receipt shape); end-to-end TP4 B1 O128 with exact token
   parity is the acceptance gate. A positive isolated PoC that regresses end-to-end
   is a rejection (all of §2.7's rejected list).
5. **Record it here** (Part 3) and, if it moves the ledger, in `PERFORMANCE_STATUS.md`.
