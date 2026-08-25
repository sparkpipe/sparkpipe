# Kernel Next Steps — no-spec Flash decode toward the DRAM roofline

Owner: GPU kernel-review agent (unified @ c834436). Static analysis only — no
builds, no new measurements; every number is cited to an existing receipt.
Companion docs: `KERNEL_PLAYBOOK.md`, `TOP10_cuda-kernels.md`,
`PERFORMANCE_STATUS.md` (ledger), `tools/devcycle/README.md` (fail-fast loop),
`docs/coord/plan_amd_gfx950_mi350p.md` (AMD plan).

---

## 1. Roofline justification

**Where we are:** retained no-spec Flash decode = **40.4553 tok/s TP4 B1**
(`PERFORMANCE_STATUS.md:97-166`) = **24.72 ms/token**, exact 128-token hash on
every run (`PERFORMANCE_STATUS.md:130-132`).

**The roofline:** per-rank mandatory stream ~**5.3 GB/token** (the playbook's
own accounting: fused norm+quantise saves 225 MB/token = "3% of the 5.3 GB
weight stream", `norm.cuh:368-377`). Against the measured GB10 ceilings:

| Ceiling | Floor/token | Ceiling tok/s |
|---|---:|---:|
| 273 GB/s LPDDR5x (`PERFORMANCE_STATUS.md:597`) | 19.4 ms | **51.5** |
| 250 GB/s (conservative measured stream rate) | 21.2 ms | **47.2** |

The retained 24.72 ms/token carries **~3.5-5.3 ms/token headroom**, and the 50
tok/s program target sits *inside* the roofline band — reachable only by
removing bytes or hiding latency, never by adding compute.

**Where the headroom lives** (CUPTI profile, 39.0094-tok/s candidate,
`PERFORMANCE_STATUS.md:219-241`): busy-union 22.06 ms vs exposed idle
5.79 ms/token, of which **~4.47 ms is collective-shaped transitions (73% of
exposed idle — transport lane, flagged out-of-lane below)**. Kernel shares:
dense/projection GEMVs 38.65% (10.202 ms), routed experts 21.08% (5.564 ms),
attention/KV 10.81% (2.854 ms), weight read-ahead 9.48% (2.504 ms),
routing/gate 7.93%. Summed kernel time (26.4 ms) exceeds busy-union (22.1 ms):
stream overlap already exists. Remaining kernel-side wins are (a) SFU /
instruction overhead inside the streaming loop, (b) DRAM efficiency of fixed
streams, (c) address-stream count, (d) serialization points between phases.

**Fence posts — do not re-propose** (`PERFORMANCE_STATUS.md:111-115,144-156,
243-274`): persistent dense/projection bundle -0.78%, cooperative W13-to-W2
-0.83%, predeclared read-ahead queue -33.2%, projection-precollective overlap
-0.58%, cooperative Hc finalize -0.33%, Query RMS+RoPE fusion -0.72%. Sibling
path: `cp.async.bulk` measured 115 GB/s vs 180 for 16B cp.async on GB10 (fenced
null, `DFLASH2_HANDOFF.md:236-238`); CUDA graphs self-hide when frames are
memory-bound (+1.6% only, `DFLASH2_HANDOFF.md` section 0f).

**PR #678 lessons (WS kernel staging, qwen36 DFlash2)** (commit `a97c63a`,
merge `76a92f8`; bench `tools/qwen36_native_warp_specialized_bench.cu`, kernel
`modules/qwen36_resident_decode_stage/source/spark_qwen36_native_ws.cuh`):

1. Plain uint4 load+store B-staging beat cp.async 16B by **+10% isolated**
   (195-205 vs 176-181 GB/s): the async engine itself was the isolated-kernel
   bottleneck on GB10.
2. **In-situ neutral** (20.90 vs 20.91 s wall): the isolated win did not
   survive the production boundary — verify every candidate in situ (playbook
   Appendix rule 4).
3. The timing change exposed a latent shared raw-A ring race (fence-resistant,
   root cause open); production reads A from global — treat A-ring staging as
   quarantined.
4. Their stated next lever is **in-situ locality (L2 persistence for e8m0
   scales, launch coalescing), not kernel micro-optimization** — this document
   adopts the same ordering.

---

## 2. Ranked candidates (no-spec Flash decode)

| # | Candidate | Category attacked | Expected effect | Risk | Acceptance gate |
|---|---|---|---|---|---|
| 1 | Share bit-exact E8M0 fast decode into MXFP4 format path | Routed experts (21.1%) | ~0.2-0.45 ms/token (SFU removal) | Very low | Bit-exact all-codes probe + O24/O128 hash parity + >= +0.5% tok/s x3 pairs |
| 2 | 16-byte vectorized weight loads, B1 dense/projection GEMVs | Dense/projection (38.65%) | 0.5-1.0 ms/token if DRAM efficiency +5-10% | Medium-low (documented counter-case) | Isolated A/B at B1 shapes, bit-exact, then devcycle spot >= +0.5%; kill on flat |
| 3 | Adopt `mxfp4_ws_interleaved_v1` expert tensors into DSV4 stagepack | Routed experts (21.1%) | 0.11-0.28 ms/token (one address stream fewer; net bytes unchanged) | Medium (pack regen + loader wave) | Pack layout tests pass for DSV4 shapes; bit-exact O24/O128; spot delta > noise |
| 4 | De-alias `ffn_accum_bf16` (WAR hazard) | Projection/MoE seam | Removes one serialization point; magnitude unknown; cost ~8 KB | Very low (one-line) | Bit-exact + repeatable spot delta beyond +/-0.5% noise |
| 5 | Fold MoE pair-reduce into W2/HcPost epilogue via existing `accumulate_bf16` | Routed experts + Hc (26.2%) | ~7.4 MB/token bytes + ~58 launch gaps = 0.15-0.33 ms/token upper bound | Medium-high (bad priors) | Mandatory bitwise PoC; production-overlap A/B x3; reject on ANY regression |
| 6 | Weight read-ahead scope audit (measure-only first) | Read-ahead (9.48%) | Data-dependent; trims bandwidth competition with consumers | Low to measure; high to act | CUPTI L2-hit receipt BEFORE any change; queue/persistent variants stay fenced |
| — | Collective transitions (~4.47 ms/token) | Exposed idle (73%) | Largest single number on the board | Out of lane | Flag only — transport surface, not kernels (`TOP10_cuda-kernels.md` closing note) |

### R1 — Bit-exact E8M0 fast decode, shared back into the format path

STATUS (2026-08-25): host-side source edit landed; hardware gates open — see
Appendix A.

The fix exists and is measured, but only on one consumer.
`SparkLmDotRowFp8E8m0` uses `__uint_as_float(code << 23)` ("IEEE exponent bits
are (e+127)<<23 ... one instruction vs the exp2f transcendental",
`spark_lm_kernels.cuh:698-702`; standalone 228 -> 248 GB/s, commit `5b27165`).
The shared MXFP4 decoder still pays the transcendental on the expert hot loop:

- `LmE8m0ToFloat` = `exp2f((float)code - 127.0f)` (`formats/mxfp4.cuh:33-36`),
- consumed per staged scale byte by the tile-pipeline GEMM (`gemm.cuh:266`),
- again in `SparkLmSm121E8m0ScaleValue` (`spark_lm_kernels.cuh:2390-2393`).

Applying the same one-instruction form attacks the routed-expert category
(21.08%, 5.564 ms summed). If half the qwen36 GEMV's +8.8% standalone gain
transfers: ~**0.2-0.45 ms/token**.

*Risk:* very low, one sharp edge — the bit trick equals `exp2f(code-127)` bit
-for-bit for codes 1..254; code 0 differs (subnormal vs +0.0). The encoder
clamps to [0,254] (`spark_lm_kernels.cuh:2385-2388`); either assert qualified
packs never emit 0 or keep an explicit code==0 guard. The 0xff NaN branch stays.

*Gate:* (a) host probe over all 256 codes — bit-equal except the documented 0;
(b) devcycle spot: exact O24+O128 token hashes; (c) >= +0.5% tok/s mean over 3
alternating pairs; (d) update Part 3 registry rows.

### R2 — 16-byte vectorized GEMV weight loads (dense/projection)

Largest category (38.65%, 10.202 ms summed) streams FP8/BF16 weights at
byte/2-byte granularity in the B1 GEMVs (`tools/devcycle/README.md` candidate
#1). Vectorizing to 16-byte loads raises transactions-in-flight on the fixed
stream — pure DRAM-efficiency play, zero bytes changed. +5-10% efficiency on
this category is **0.5-1.0 ms/token**: the largest kernel-side prize left.

*Risk:* medium-low — the sibling path holds a counter-receipt: in the qwen36
multi-row dot bench, 4-byte loads and __ldg-4B were *slower* than byte-wise
__ldg (83.7 / 91.7 vs 125.6 GB/s — "the outstanding byte transactions ARE the
memory-level parallelism", `DFLASH2_HANDOFF.md:95-104`). That pattern was M=8,
MMA-adjacent; the B1 scalar GEMV is plausibly latency-bound instead. The PoC
decides which regime this kernel occupies — that is the whole experiment.

*Gate:* isolated exact A/B on `SparkLmSm121FusedDenseW13GemvKernel`
(`spark_lm_kernels.cuh:3016`) at true B1 shapes; bit-exact; then devcycle spot
>= +0.5%. Kill on flat — no tuning past the first null.

### R3 — Interleaved ws expert tensors for DSV4 (K3 pack-V2 geometry)

K3's `mxfp4_ws_interleaved_v1` co-tiles each stage's E8M0 scales INTO the
weight box (17-row cells; one rank-3 UINT8 tensor map; "the V1 far LDG scale
stream is gone"; net DRAM bytes identical, one address stream fewer,
`K3_PACK_FORMAT_V2.md:118-193`). The DSV4 stagepack still ships separate scale
planes (`scale_offset`, `spark_dsv4_stagepack_format.h:150,329-330`). Adopting
the geometry deletes the second stream from the expert inner loop (21.08%) and
its per-stage fetch interleave: expected **0.11-0.28 ms/token** (2-5% of expert
time) plus sequential-prefetch locality not yet priced on this workload.

*Risk:* medium — pack regeneration + loader/shard wave for one module.
Mitigants: geometry, closure assertion, and layout tests exist and are proven
for K3; the consumption contract already reads interleaved scale rows through
the swizzle (`gemm.cuh:240-273`), so the kernel side largely exists.

*Gate:* regenerated pack passes validate_layout-class assertions; loader layout
test for DSV4 shapes; exact O24/O128; spot delta beyond noise; pack-hash
identity change recorded per SPEC section 2 discipline.

### R4 — De-alias `ffn_accum_bf16`

One buffer serves the projection-shard unpack (`module.c:3122`) and the MoE/ffn
output path (`module.c:3068,3228`) — a WAR serialization point between two
otherwise independent device phases. A second scratch slot (~8 KB class at B1)
lets them overlap (`tools/devcycle/README.md` candidate #2). Effect size unknown
by construction; the cheapest legitimate latency-hiding candidate on the board.

*Risk:* very low (allocation + pointer swap). *Gate:* bit-exact O24/O128 +
repeatable positive delta beyond the +/-0.5% run-to-run noise band; drop
without ceremony if flat (the fail-fast loop exists for exactly this).

### R5 — MoE pair-reduce folded into the epilogue (`accumulate_bf16`)

The mechanism landed (`bc7b53c`): `LmGemmArguments.accumulate_bf16` makes the
GEMM epilogue ADD into a second buffer — "one fewer kernel, one fewer
full-width read, per fused add" (`gemm.cuh:99-106`). Folding
`SparkLmMoePairReduceKernel`'s weighted fold into W2's store saves, at B1, the
top_k x hidden accumulator read-modify-write (~7.4 MB/token across 58 MoE
layers) plus ~58 launch gaps: **<= 0.15-0.33 ms/token upper bound**, mostly gap
recovery.

*Risk:* medium-high. This family owns three rejection receipts (cooperative
W13-to-W2 -0.83% at the production overlap boundary; cooperative Hc finalize
-0.33%; Hc residual copy fusion worth only +0.22%). Arithmetic reordering risks
the exact-FP32 accumulation contract, so the bitwise PoC is not optional.
Proceed only because the mechanism differs from every rejected variant (epilogue
add, not a schedule change) and the devcycle list carries it (candidate #3).

*Gate:* bitwise PoC vs the unfused pair at B1; then the full-ring
production-overlap A/B (three independent interleaved pairs — the standard that
rejected its cousins); any regression kills it permanently.

### R6 — Weight read-ahead: measure before touching

Read-ahead consumes 2.504 ms/token (9.48%) of DRAM bandwidth warming cache
during collective windows (`spark_lm_kernels.cuh:71-100`). On a
bandwidth-saturated machine this pays only while its L2 hits repay more than
they steal from consumers. The aggressive variant is fenced (predeclared queue
-33.2%: worker CTAs stole capacity, `PERFORMANCE_STATUS.md:265-274`). The move
is measurement: CUPTI hit/miss attribution per read-ahead tensor class; trim
scope to tensors consumed by the immediately following window only if the data
says so.

*Gate:* hit-rate receipt FIRST; any subsequent edit passes the standard spot
gate; no persistent/queued variants ever.

---

## 3. Out-of-lane flags (kernel-review lane, do not build here)

1. **Collective-shaped transitions, ~4.47 ms/token (73% of exposed idle)**:
   transport surface, not kernels (`TOP10_cuda-kernels.md` closing note).
2. **Speculation (DSpark cards D4-D-01..05 + V-01/V-02)**: whole kernel surface
   built, wired, NOT_MEASURED — the level 4-to-5 buy for dsv4-flash at ~0 new
   kernel code (`TOP10_cuda-kernels.md` headline). Out of scope for the no-spec
   roofline, but it dominates any speculative-path decision.
3. **TMA/cp.async.bulk staging for B1 GEMVs**: fenced by measurement on GB10
   (115 vs 180 GB/s; PR #678's plain-B win was in-situ neutral). Re-open only
   with new hardware evidence.

---

## 4. AMD side (MI350P / gfx950 plan mapping)

Plan of record: `docs/coord/plan_amd_gfx950_mi350p.md` (status PLAN; D1 done,
D2-D6 blocked on MI350P access + WS-A splits, `WBS.md:35-39`). Scope is DSV4
islands E0/L1-L5/F1 behind the frozen seven-island ABI, target string
`rocm.gfx950.mi350p`.

### 4.1 Candidates/kernels with planned ROCm counterparts

| Candidate / kernel class | AMD counterpart status |
|---|---|
| Routed-expert GEMMs (R1/R3/R5 consumers, `expert_mxfp4`) | **Planned, upgraded**: Workstream B targets native block-scaled MXFP4 MFMA — expert weights may feed MFMA with NO dequant-to-BF16 staging at all, deleting both the decode ALU cost and this entire optimization class on AMD (plan section 4.2, decisions B1-B5; zero-repack goal, load-time repack fallback keyed by pack hash) |
| Dense/projection linears + norms + residual adds (L1/L4/L5 islands) | Planned in Workstream C bring-up (S7): E0+L5 first, then F1 skeleton, L1, L4, L2/L3 last — uncaptured first, HIP graphs after correctness (plan 5.2) |
| Attention/KV + paged-cache ring emission (L2/L3 islands) | Planned at Workstream C step 4; KV payload quantization decided by the L3 emitted-field table at recipe time, not preempted by the weight layout (plan 4.4) |
| Output head / certified FP8 screen / argmax (F1 island) | Planned: lowest-index tie-break must survive wave64 reductions; engineered tie case is an S7 checklist item, not an observation wait (plan 5.3) |
| Weight read-ahead (R6) | Planned as a runtime primitive, not a kernel: `spark_hw_read_ahead` non-blocking kick, contents target-defined (plan 3.2) |
| Collectives (out-of-lane item) | Planned: NCCL-kind backend over RCCL at S8, algorithm-mask honesty (never silently substitute), recursive TP4 tree preserved as reference semantics (plan section 6) |

### 4.2 NVIDIA-intrinsic mechanisms needing different AMD equivalents

| NVIDIA mechanism | Why intrinsic | AMD equivalent direction |
|---|---|---|
| TMA boxes / tensor maps + mbarrier transaction-count pipeline (`tma.cuh`; K3 cell-grid ws staging) | gfx950 has no tensor-memory accelerator | Buffer/LDS async-copy staging or DMA engines; the 64B-swizzle cell grid must be re-derived against LDS bank behavior, and the MFMA operand-layout decision (B1/B2) decides whether the interleave survives at all (plan section 8 risk row) |
| `mma.sync` SM120_16x8x32_TN fragment mappings + SM121-native block-scaled PTX behind the `__CUDA_ARCH__==1210` trap gate (`mma.cuh:18-46`) | sm_121a-specific instruction forms | gfx950 MFMA atoms; fragment mappings and test_mma_fragment_mapping-class bijection proofs re-derived for wave64 lane ownership; fail-closed gfx950 guard replaces the PTX-trap discipline |
| wgmma / tcgen05 | Absent on sm_121a anyway (capability-gated out, playbook 2.1) | Not needed — MFMA + native MXFP4 is the corresponding capability and is *stronger* than what GB10 offers |
| 16B cp.async vs plain-B uint4 staging split (PR #678's +10%) | Measured property of GB10's async engine — does NOT transfer | gfx950 async-copy semantics (s_waitcnt model) differ; re-measure from zero on MI350P before importing any PR #678 conclusion |
| cudaAccessPolicyWindow-class L2 persistence (R6-adjacent) | CUDA-specific carve-out API | No direct HIP analog; drop, or re-express via AMD cache-op/buffer primitives only if profiling justifies |
| E8M0 fast decode `__uint_as_float(code << 23)` (R1) | None — pure IEEE binary32 bit arithmetic | Portable as-is to HIP; keep it out of any ROCm TU's reach of `SPARK_LM_SM121_*` selectors per freeze F4 (landmine rule), i.e. land it in the shared format layer, not the selector |

**Sequencing note:** R1 and R3 are the two candidates that also simplify the
AMD port — a shared, selector-free E8M0 decode shrinks the landmine surface,
and adopting the interleaved ws layout for DSV4 gives both vendors one weight
layout story instead of two (Workstream B's zero-repack goal consumes exactly
that geometry).

---

## Appendix A — R1 host-side landing (2026-08-25, kernel agent, tree @ unified c834436)

**Change note (commit-message style).** Share the proven bit-exact E8M0 fast
decode — `__uint_as_float(code << 23)`, the `SparkLmDotRowFp8E8m0` receipt
(standalone 228 -> 248 GB/s, commit `5b27165`) — into the two remaining
`exp2f()` sites: `LmE8m0ToFloat` (`formats/mxfp4.cuh`, consumed per staged
scale byte by the tile-pipeline GEMM via `ScaleDecode`, `gemm.cuh:266`) and
`SparkLmSm121E8m0ScaleValue` (`spark_lm_kernels.cuh`, the routed-expert hot
loop = 21% of decode kernel time, plus the qwen36 ws/bench consumers). The
`code==0` guard mirrors `SparkLmDecodeE8m0`
(`spark_lm_kernels.cuh:160-167`) verbatim — `__uint_as_float(0x00400000u)` =
the 2^-127 subnormal, exactly what `exp2f(-127.0f)` returns — so all 256 codes
decode bit-identically to the exp2f forms, denormal/zero cases included. Each
site's existing edge handling is untouched: mxfp4 keeps 0xff -> NaN
(0x7fffffff); `SparkLmSm121E8m0ScaleValue` keeps no branch (code 255 is +inf
in both forms). Accumulator semantics and interfaces unchanged.

**Files changed**

- `inference/kernels/formats/mxfp4.cuh` — `LmE8m0ToFloat` bit-shift form + guard
- `model-families/common/include/sparkpipe/spark_lm_kernels.cuh` — `SparkLmSm121E8m0ScaleValue` bit-shift form + guard
- `docs/KERNEL_NEXT_STEPS.md` — R1 status line + this appendix

**Gates run (host only) and results**

| Gate | Result |
|---|---|
| Host all-codes probe: both new forms vs exp2f originals, codes 0..255 | PASS (bit-identical on all 256; raw shift's code-0 divergence +0.0 vs 0x00400000 confirmed and restored by the guard) |
| `python3 tests/test_dsv4_contracts.py` | PASS |
| `python3 tests/test_cuda_math_policy.py` | PASS — no test update needed or made: the policy greps BUILD files for `--use_fast_math` only and asserts nothing about `exp2f` usage; the bit-shift form is pure IEEE bit arithmetic (an exponent-field move), not a fast-math substitution, so it does not touch this policy |
| `make -C modules/dsv4_resident_decode_stage contract` | PASS |
| `make -C modules/dsv4_resident_decode_stage -f Makefile.pro contract` | FAIL as literally invoked, PRE-EXISTING at c834436 (reproduced on a pristine HEAD worktree): under `-DSPARK_DSV4_PRO_BUILD=1` the model header includes the Pro aliases and never defines `SPARK_DSV4_MODEL_DSPARK_SPEC_STEP`, while Makefile.pro lacks the plain Makefile's `DSPARK_SPEC_STEP` plumbing (`Makefile:28-30`). With that one define supplied on the command line (`MODULE_COMPILE_FLAGS=... -DSPARK_DSV4_MODEL_DSPARK_SPEC_STEP=7u`): PASS at HEAD AND with this change |
| `./build/test_dsv4_pro_dspark_drafter_pin` (rebuilt locally, host cc) | PASS (exit 0) |
| `python3 tests/test_kernel_algorithms.py` | PASS |
| `python3 tests/test_dsv4_module_host_syntax.py` | PASS |
| `tests/test_dsv4_stage_source.py`, `test_dsv4_driver_source_contracts.py`, `test_k3_pack_layout.py` (kernel-source greps) | PASS |
| Pre-existing failures at c834436, identical on pristine HEAD, unrelated to E8M0 decode: `test_cuda_performance_contracts.py` (GLM host-module sync-count assertion), `test_code_size.py` (already 903 lines over ceiling before this change's +16 net; delta here is concurrent untracked `tools/fleet/` growth plus this edit), `test_dsv4_native_compute_source.py` (B1024 routed W13 M64 tile), `test_kernel_launches.py` | FAIL (pre-existing; left for their owners) |

**Hardware acceptance gate that REMAINS (spark4-7 — not attempted here)**

1. Device all-codes bit probe over the 256 E8M0 byte values: new decode ==
   exp2f(code-127) bit-for-bit everywhere, including code 0 (subnormal) and
   255 (+inf).
2. Exact O24/O128 token hash parity on the dspark serving path.
3. >= +0.5% tok/s mean over 3 interleaved A/B pairs.
4. On pass: update Part 3 registry rows / PERFORMANCE_STATUS ledger.
