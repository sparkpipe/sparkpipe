# EVIDENCE — L2 island audit R1 + indexer top-k landing

Date: audit iteration R1 of the AMD MI350P kernel hill-climb.
Territory: `modules/dsv4_resident_decode_stage/source/rocm/`.
Reference: `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu`
(CUDA, SM121) read line-by-line against each ROCm port.

## Why L2

Per the island status ledger, L2 was a partial island (QKV projection only)
with the largest unauthored surface: indexer score + top-k + sparse attention.
L4 had already had an audit pass; L2 had none.

## Defects found (audit) and fixed in this iteration

### D-L2-1 — `inverse` rope flag silently dropped (correctness surface narrowing)

`SparkDsv4RocmQueryHeadRmsRopeKernel` and `SparkDsv4RocmKvPostKernel` hard-pinned
the CUDA `inverse` argument (`:329`, `:345`) to `0u` while the TU header claimed
"exact bodies" from the reference. The reference uses `inverse != 0u` for the
attention-output de-rotation (`-sinf`). Fixed: both kernels carry the flag
again; the island entry passes `0u` with an ownership note (de-rotation belongs
to the not-yet-landed sparse-attention slice). No behavior change today; the
kernel surface no longer lies about what it can express.

### D-L2-2 — missing gridDim.y bound on wo projections (validation gap)

`wo_a_rows_out`/`wo_b_rows_out` were validated non-zero but not against the
65535 gridDim.y ceiling; an oversized request would die inside
`hipGetLastError` as an opaque launch error instead of a clean
SPARK_HW_INVALID at validation. Fixed in `SparkDsv4RocmL2Validate`.

### D-COMMON-1 — dead locals in the shared GEMV row kernel

`lane`/`wave` computed then `(void)`-discarded in `SparkDsv4RocmLinearRowsKernel`.
Removed; no semantic change.

### D-E0-1 — E0 stream-expand kernel could never have compiled (compile-readiness,
found by the new full-TU proof)

`SparkDsv4RocmStreamExpandKernel` subscripted its `void *boundary_packet_bf16`
output directly AND assigned a float right-hand side into it:
`boundary_packet_bf16[idx] = RocmBf16ToFloat(...)`. Invalid C++ (void*
subscript), wrong element type even under GNU extensions; hipcc would reject
the TU. Fixed via the shared helpers:
`RocmFloatToBf16(out, idx, RocmBf16ToFloat(in, src))`. The bf16->f32->bf16
roundtrip is bit-exact (widening is exact; RNE of an exactly representable
value restores the original bits including NaN payload bits that fit in bf16),
so the copy stays inside the C2 integer-pure class claimed by the in-source
comment.

### D-L5-1 — L5 called an undefined function (compile-readiness, found by proof)

`RocmDotRowFp8Block128` called `RocmDecodeE8m0(...)` which is neither declared
nor defined anywhere in the TU (the forward declaration at the top of file is
for `RocmFloatToBf16Bits`, a different function). hipcc would reject the unit.
Fixed: verbatim port of `SparkLmDecodeE8m0` (spark_lm_kernels.cuh :159):
pure power-of-two, bias 127, code 0xff (NaN encoding) decodes to 0.0f, code 0
decodes to 2^-127 (`__uint_as_float(0x00400000)`).

## New authored slice: L2 indexer top-k (S8 scope first landing)

Port of CUDA `SparkDsv4OrderedTopKKey` (:1657), `SparkDsv4TopKKernel` (:1668)
into `spark_dsv4_rocm_islands_l2_attention_qkv.hip`:

- Entry: `SparkDsv4RocmIslandL2IndexerTopK(queue, io)` with its own Io struct
  and cap bit `SPARK_DSV4_ROCM_L2_CAP_INDEXER_TOPK (0x2u)`. Existing L2Io ABI
  untouched. Caps contract tightened for this entry: the bit is REQUIRED
  (INVALID without it), unknown bits UNSUPPORTED (never silent partial).
- Key packing identical: ordered score (sign-flip monotone map) << 32 |
  (~slot); sentinel `score <= -3.0e38f` or NaN maps to key 0; -0.0 folded to
  +0.0 before the bit map exactly like the reference.
- Both reference paths kept: direct bitonic path when bounded_slot_count <=
  topk, else eight MSB-first byte-radix passes building prefix/prefix_mask,
  threshold collection, bitonic finish over next-pow2(topk), ordered emission
  with -1 padding beyond selected_count.
- Edge parity verified against the reference logic: empty row (all -1 out);
  all-sentinel row (valid_count = 0 -> all -1); valid_count < topk (pad -1);
  topk == 1 (safe WITHOUT the sort because after 8 passes prefix_mask covers
  all 64 bits so exactly one unique key matches the threshold — keys are
  unique since the low word carries the slot index); duplicate scores
  (canonical lower-slot tie break through the ~slot low word).

### Declared deviation (integer-equivalence argument)

CUDA aggregates the per-pass histogram with `__match_any_sync`/`__ffs`/
`__popc` leader election over 32-lane warps. That is shared-atomic traffic
optimization, not semantics. This port issues one plain shared atomicAdd per
matching thread. Histogram contents are identical integers, every downstream
decision (prefix walk, threshold, selection, order) is integer-determined, and
the code has NO intra-wave group intrinsics left — nothing can drift between
wave32/wave64 builds. This is what makes the slice C2 (integer-pure,
bit-exact across wave sizes): scores are consumed only through their IEEE-754
bit patterns; all comparisons after key construction are uint64.

### Fixed trees recorded (C3/C2 law: one path per bucket)

Bitonic network sizes 2..sort_width with the verbatim reference comparator;
radix pass order 56..0 step 8; thread-count loops stride blockDim.x (=256 at
launch, 4 waves x 64 lanes). Static LDS budget: histogram 1024 B +
selected_keys 4096 B + 7 scalars ~= 5.15 KB per CTA — far under the gfx950
LDS limit, two CTAs/CU co-resident feasible.

## Verification (honest classification)

Ran here (measured):
- NEW tier-1.5 full-TU host proof (`rocm_tu_host_proof.sh` +
  `rocm_tu_host_proof_shim.h`): device annotations inert, triple-angle
  launches rewritten into a variadic no-op call, so EVERY kernel signature and
  EVERY launch argument list type-checks under Apple clang g++ frontend
  (-std=c++17 -fsyntax-only -Wall -Wextra). ALL EIGHT TUs PASS, zero warnings.
  Before this iteration NO .hip TU had any static proof (only common.hiph);
  the new proof immediately caught D-E0-1 and D-L5-1, both real
  compile-breakers in landed code.
- Legacy tier-1 (`validate_rocm_syntax.sh`): tier1 OK after edits.
- S6 backend regression untouched-green: test_spark_hw_rocm_pure PASS;
  mi350p island_compile_check vendor mode COMPILE_CHECK_OK.

Unproven here (unchanged limitation, now smaller):
- Device compilation `hipcc --offload-arch=gfx950 -fsyntax-only` for every TU
  (no ROCm toolchain on this box; script prints exact commands).
- Any hardware execution/perf claim. None made.

## Self-audit of this iteration's own changes

- Re-read the full top-k port fresh after writing it; found and removed one
  dead local (`lane`) left from the dropped match_any approach.
- Re-checked barrier structure against the reference line-for-line: init ->
  sync -> count -> sync -> radix loop {clear -> sync -> histogram -> sync ->
  thread0 update -> sync} -> threshold/cursor init -> clear keys -> sync ->
  collect -> sync -> bitonic (sync per stride stage) -> emission. Identical.
- Confirmed intra-stage bitonic race-freedom: each unordered pair is touched
  only by the thread holding its lower slot, so no two threads write one
  element inside a stage.
- Checked launch signatures arg-by-arg (9-arg query kernel, 11-arg kvpost) —
  additionally enforced mechanically by the tier-1.5 deduction.
- Adjacent-code sweep: no other void* subscripting or undefined-symbol calls
  remain in the directory (proof covers all TUs mechanically now).

## Next island candidates (for the next iteration)

1. L3 fused KvEmission pipeline (deferred behind requested_caps) or
2. L2 indexer score kernel (feeds this top-k; completes the C2 selection
   chain end-to-end) or
3. F1 markov bias + DSpark drafter heads.
