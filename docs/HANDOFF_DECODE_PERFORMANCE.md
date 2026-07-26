# Handoff: decode-path performance, verification without a GPU, and what to measure first

Supersedes nothing; read alongside `HANDOFF_KV_SHARING_AND_KERNELS.md`, which
covers the KV subsystem specifically. This one covers the decode kernel path,
the performance model that should direct effort, and the tooling that now makes
CUDA-level claims checkable without hardware.

---

## 1. METHOD — read this first

Five conclusions in one session were confidently wrong, and every one came from
a crude instrument reported as a finding. They are listed with what the wrong
instrument was, because the pattern matters more than the individual errors.

| claim | instrument | reality |
| --- | --- | --- |
| `kv_dedup.c` has a live OOB write and a refcount leak | reading code paths in isolation | both unreachable; an entry is erased at refcount zero, and the reinsert always runs with a free slot |
| the admission pass is redundant with the acquire guards | reasoning about which checks overlap | acquire fails **late**, after mutating, and reports `BUSY` not `CAPACITY_EXCEEDED`; a test asserted fail-before-mutate |
| `pp13_service_backend.c` (3862 lines) is dead | grep for link-time references | it is a `dlopen` plugin exporting one symbol resolved by `dlsym` |
| top-k is implemented four times | counting keyword occurrences per family | the primitives are already shared; two independent selection kernels |
| dspark is 877 lines with one glm52 constant, easy to promote | grepping for model constants | 103 `SparkGlm52Dspark*` symbols across 18 files plus its own module |

Rules that follow:

- **A keyword count tells you where to look, never what is there.** Every
  conclusion drawn from one and not checked against function bodies was reversed.
- **Grep for callers is not a reachability test.** Plugins, `dlsym`, and build
  rules defeat it. Check `sources.mk` and the Makefile too.
- **Search for the capability by name before building it** — "prefix", "reuse",
  "dedup", "share", "async". Checking the component you already decided to
  modify tells you nothing.
- **A passing test proves nothing until you break the thing it covers.** Two
  tests written this session passed against unpatched code; both were measuring
  nothing until deliberately broken to confirm they had teeth.

The sparse-attention and MLA "implemented twice" figures in the other handoff
came from the same keyword instrument and are **still unverified**. Do not plan
around them without diffing function bodies.

---

## 2. ENVIRONMENT — what can and cannot be verified here

Verified this session, not assumed:

- **No `nvcc`, no `clang`.** The distribution CUDA toolkit 404s on a driver
  dependency. `pip install nvidia-cuda-nvcc-cu12` ships **`ptxas` only**, which
  assembles PTX and cannot compile CUDA C++.
- **`ptxas` 12.9 does target `sm_121a`**, with `.version 8.8`. 8.5 and below
  reject the target; 9.0 is newer than the assembler.
- Host C builds and runs. `make -j4 test` is the gate.

Two techniques make CUDA changes checkable anyway, and both are in the tree:

**PTX assembly gate** — `tests/test_ptx_capability_gate.py` assembles 20 forms
against `sm_121a` on every `make test`. It is symmetric: a required form that
stops assembling fails, and `tcgen05` or `wgmma` *starting* to assemble also
fails, because that means the target changed and the kernel strategy should be
revisited. Where `ptxas` is absent it skips rather than failing.

**Syntax check via keyword shim** — define `__device__`, `__forceinline__`,
`__shared__`, `__align__`, `threadIdx`, `blockDim`, `__nv_fp8_storage_t` and
`__cvta_generic_to_shared` away, then `g++ -std=c++17 -fsyntax-only` with every
template instantiation forced. Filter the x86 asm-constraint noise. This catches
typos, template errors and `static_assert` violations. It does not catch
anything about kernel correctness.

### Verified capability matrix, sm_121a

| form | available |
| --- | --- |
| `cp.async.ca`, `cp.async.cg` (4/8/16 B) | yes |
| `cp.async` with src-size operand (hardware zero-fill) | yes |
| `cp.async.bulk` | yes |
| `cp.async.bulk.tensor` 1D / 2D / 3D (TMA) | yes |
| `mbarrier.expect_tx` | yes |
| `barrier.cluster.*`, `mapa.shared::cluster` | yes |
| `mma.sync.m16n8k16.bf16` | yes |
| `cp.async.cg` at 4 bytes | **no** — 16 B only |
| `tcgen05.*` | **no** |
| `wgmma.*` | **no** |

The prose in the previous handoff was correct on all three of TMA, clusters and
tcgen05. It is now assembled rather than asserted.

---

## 3. PERFORMANCE MODEL — where the time actually goes

Constants from `GB10_CUDA_COST_MODEL_CALIBRATION.md` and the diagnostics it
cites. These are the numbers that should direct effort.

- Peak memory bandwidth **273 GB/s**, LPDDR5x unified, not HBM.
- Measured memory-path efficiency **eta_bw ~= 0.80**, three independent
  derivations agreeing, so **~218 GB/s effective**.
- Peak FP8 tensor **~250 TFLOP/s dense**. 48 SM at up to 2.55 GHz, 128 KB
  L1/shared per SM.
- **Balance point ~915 FLOP/byte.** Anything below is memory bound.
- Measured QKVO/linear WMMA: **6.5 TFLOP/s, 2.6% of FP8 peak.**
- Byte budget at B128: attention 165 MB + shared expert 38 MB + routed MoE
  **5.3 GB** = 5.5 GB. **The weight stream is 96% of all traffic.**

### The central arithmetic

The fp8 tile is `TILE_N=64`, `TILE_K=64`, `MMA_M=16`. Per CTA iteration:

- weight bytes: 64 x 64 = **4096 B**
- FLOP: 2 x 16 x 64 x 64 = **131,072**
- **arithmetic intensity = 32 FLOP/byte**

Against 218 GB/s effective, intensity 32 gives a ceiling of **7.0 TFLOP/s**.
Measured is 6.5. **The kernel is at ~93% of its own bandwidth roof.**

This is the finding that should redirect effort. The 2.6%-of-peak figure is not
inefficiency to be tuned away — it is the roof being low because the intensity
is low. Staging, stalls and pipelining are worth at most the remaining 7%.

**Intensity at decode is `2 x rows_per_weight_read`.** With `MMA_M=16`, each
weight tile serves 16 rows, so at B128 every weight byte is read **8 times** and
at B256, **16 times**.

### Latency and pipeline depth

Memory latency is **not measured anywhere in this repo** — that is a gap worth
closing with one microbenchmark. Estimating 350-600 ns for LPDDR5x unified.

Little's Law against 218 GB/s effective:

| latency | bytes in flight needed | per SM (48) |
| --- | --- | --- |
| 400 ns | 87 KB | 1.8 KB |
| 600 ns | 131 KB | 2.7 KB |

The 3-stage weight pipeline now in the tree keeps **2 tiles x 4096 B = 8 KB in
flight per CTA**, three to four times the requirement at one CTA/SM.
**Three stages is right-sized; four would buy nothing.** Do not spend the extra
4 KB of shared memory without a measurement that contradicts this.

---

## 4. WHAT LANDED

**Merged:** #507 (rebased, one commit skipped — see below), #509 KV block
identity, #510 fixing a red `main`, #511 decode-batch DRY.

**Open, #513:**

- `tests/test_ptx_capability_gate.py` — 20 probes, described above.
- `model-families/common/include/sparkpipe/spark_lm_async_copy.cuh` — sized
  `cp.async` at 4/8/16 B under either cache policy, a bounded form whose tail
  the hardware zero-fills, commit/wait, and a cooperative `StageBytes`. No
  pre-Ampere fallback by design: a fallback silently turns an async pipeline
  synchronous.
- FP8 weight staging converted to `cp.async`, prologue and pipeline.
- **3-stage weight pipeline** — issues two tiles ahead, retires exactly one group
  with `wait_group 1`, so a transfer is always in flight. The contract now
  **forbids `wait_all` in the K pipeline outright**, because a drain there idles
  the bus at every iteration boundary and is exactly what a future simplification
  would reintroduce.
- Hidden-transport send path: two synchronous default-stream `cudaMemcpy` became
  two async copies on the packet stream plus one `cudaStreamSynchronize`.
- Prefix-cache walk unified behind an iterator, with the coverage it never had.
- Tools file reader 227 lines -> 54. Scheduler prefill retirement unified.

### Preconditions discovered while doing it

- `cp.async` requires the **shared destination aligned to the transfer width**.
  `__nv_fp8_storage_t` is a byte type, so the tiles were 1-byte aligned and are
  now `__align__(16)`. Without this the instruction is malformed at runtime.
- A plain `if` on a template parameter **emits both `asm` strings** into the PTX;
  only optimisation removes the dead one. At `-O0` that emitted `cp.async.cg`
  with a 4-byte size, which `ptxas` rejects. Use `if constexpr`.
- `__syncthreads` does **not** wait on `cp.async`. Every staging site needs an
  explicit retire.
- The transport slot is published to the sender thread three lines after the
  staging copies. Async copies there without an explicit stream sync transmit
  the staging buffer's **previous contents** — wrong bytes on the wire, no error
  anywhere.

---

## 5. THE NEXT CHANGE: raise the M-tile

This is the first-order optimisation and its payoff is calculable in advance,
unlike everything else queued.

`m16n8k32` is fixed by hardware. The lever is **register-blocking over M**: one
CTA covers `R x 16` rows instead of 16, the grid shrinks by `R`, and each weight
tile is fetched `R` times less from DRAM. Intensity goes 32 -> `32R` FLOP/byte;
at R=4 the roof rises from 7.0 to 28 TFLOP/s.

Eleven coordinated edits, all in or around `spark_lm_fp8_tile.cuh`:

1. `SPARK_LM_FP8_TILE_ROW_BLOCKS` constant.
2. `tile_input` first dimension x R.
3. `row_scale` and `row_inverse_scale` x R.
4. `tile_output` first dimension x R.
5. `accumulator[R][4]`, `block_accumulator[R][4]`.
6. Every accumulator init site becomes nested.
7. Row absmax and scale computation over `R x 16` rows.
8. `SparkLmFp8StageInput` staging `R x 16` rows.
9. **Hoist `LoadFragB` out of the row loop in both MMA phases** — load the weight
   fragment once, issue R MMAs against it.
10. The block-scale `fmaf` accumulation and `StoreAccumulator`, per row block.
11. The calling kernel's `blockIdx` -> `slot_base` mapping and the host
    launcher's grid computation; then the ping-pong contract.

**Do it with R defined as 1 first.** R=1 is provably the current code path, so
the whole eleven-site restructure is verifiable against the existing tests
before it changes any behaviour. Only then flip to 2, check shared memory
(`tile_input` 2 KB -> 4 KB, `tile_output` grows too) and register pressure
against the 255/thread limit, then 4.

**The measurement that gates how far to push R:** whether the current re-reads
hit L2 or go to DRAM. With 5.3 GB of routed MoE weights L2 cannot hold much, but
if they are hitting L2 the DRAM saving is smaller than the arithmetic suggests
and R=2 may be sufficient. One Nsight counter answers it.

---

## 6. QUEUED FOR THE RING

In the order they should be run, because earlier answers change later plans.

1. **`CROSS_SEQUENCE_PREFIX_REUSE`.** One line in `pp13_service_backend.c:2352`
   masks a flag that is part of `DEFAULT_FLAGS` and honoured by two other call
   paths. It was masked because the path had not been validated, not because it
   broke. If it works, ~1300 lines of `WorkControlKvState` become deletable
   along with the mechanism #509 merged.
2. **Memory latency microbenchmark.** Nothing in the repo measures it, and it
   sets pipeline depth for every kernel. Cheapest measurement with the widest
   consequence.
3. **L2 vs DRAM on the weight re-reads.** Gates the M-tile depth above.
4. **3-stage vs 2-stage weight pipeline**, against current tile time.
5. **Send path** against the 29 us/hop floor.
6. **`a3a10fe`**, the FP8 per-K-tile activation scale skipped when #507 was
   rebased. It conflicts with `main`'s corrected `SparkLmFp8LoadFragA` fragment
   mapping and needs re-deriving on top of it. A wrong mma fragment mapping
   assembles cleanly and renders silently wrong, so this one must not merge on
   review alone.

---

## 7. OPEN, UNRESOLVED

- **`third_party` is 2048 of the repo's 5045 tracked files, 41%.** Our code
  consumes exactly four headers from it, whose transitive closure is 474 files,
  via one Makefile target. A submodule or fetched dependency removes 2048 files
  without touching a line of logic. Largest codesize lever in the repository and
  a packaging decision, not a refactor.
- **`SparkLmFp8StageInput` is the only staging that still stalls.** It is a bf16
  to fp8 transform, not a copy, so `cp.async` cannot express it. Overlapping it
  needs raw bf16 staged into a second shared buffer and converted in a separate
  pass — real shared memory cost, competing directly with raising R. The weight
  stream dominates, so R wins unless a profile says otherwise.
- **glm52 makes zero calls into the shared `SparkLm*` kernel library**; the other
  four families make 38-84 each. Structural overlap between glm52's kernels and
  the library measured **zero** — glm52 is not a copy, it solves the same
  problems differently. Whether the library should absorb the general algorithms
  is a real question; it is not a duplication cleanup.
- **Four `KvCacheArena` functions duplicate a validation preamble** when
  `SparkGlm52KvCacheArenaResolveBlock` already exists as a public API doing it.
  Small, and the same shape as the four tool file readers: an API that exists and
  is not reached for.
