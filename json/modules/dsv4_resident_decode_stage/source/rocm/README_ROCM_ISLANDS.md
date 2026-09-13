# DSV4 ROCm island sources — status note

Date: written 2026-08-25 against unified tip `eb4cf58`. Audience: hwiface
coordinator + AMD implementer. This directory is the first AMD source to
land in-tree.

## Record correction (read first)

An earlier task description said "after your E0+L5 landed". **Nothing AMD
had landed**: an E0/F1 authoring turn was interrupted by other work before
any file was written, and a repo-wide check confirms no prior .hip sources.
These L1/L2 translation units are therefore the FIRST entries of the ROCm
target archive, not additions to existing ones. E0/F1 remain unauthored.

## Files

| File | What it is |
|---|---|
| `spark_dsv4_rocm_kernels_common.hiph` | bf16 access, fixed wave64 reduction trees, bf16-weight GEMV row kernel, island Io structs + provisional entry declarations |
| `spark_dsv4_rocm_islands_l1_boundary_norm_project.hip` | hcPost fold (line-faithful port), RMSNorm, Q/KV projections, frozen TP-shard interleave pack |
| `spark_dsv4_rocm_islands_l2_attention_qkv.hip` | query-head RMS+YaRN rope fused, KV post (norm/clamp/rope on an 8-wave CTA), wo_a/wo_b projections |
| `rocm_syntax_shim.h` / `validate_rocm_syntax.sh` | local tier-1 proof tooling (never compiled into any image) |

## Contract position

- Landmine rule still ACTIVE: `spark_lm_kernels.cuh` carries 90
  `SPARK_LM_SM121_*` selectors at tip, so every TU here is deliberately
  self-contained and includes only the REAL landed
  `include/sparkpipe/spark_hw_iface.h` (frozen §F2 primitives).
- Entry spellings are PROVISIONAL (declared-surface-shaped) until the
  interface session pins island prototypes against dsv4_core call sites.
  Names follow freeze §F1; semantics follow the declared inputs/outputs.
- Geometry constants are pinned flash-model facts (hidden 4096, hc 4,
  q-lora 1024, head dim 512/rope 64, index head 128, kv block 64,
  eps 1e-6) validated per entry argument.

## Honest deviations / gates

1. **L2 is a partial island by construction**: indexer core + top-k and
   sparse paged-attention score/AV kernels are S8 scope.
   `requested_caps` makes that checkable — requesting unprovided slices
   returns `SPARK_HW_UNSUPPORTED`, never silent partial execution.
2. **KV post quant is range-clamp, not CUDA's group-absmax QuantSimGroup**
   yet — safe subset (cannot overflow E4M3), does not match CUDA
   bit-for-bit on quantized KV payload bytes. C2 halves unaffected
   (counters/addresses untouched). Exact scaling lands with S8 state
   comparison under its C3 tolerance owner.
3. **KvPost runs 512-thread CTAs** (8 waves × 64 lanes) to preserve CUDA's
   warp partitioning (7 quant groups + rope warp) instead of silently
   re-partitioning onto 4 waves. Fixed shape, recorded in-source.
4. Reduction trees are FIXED (butterfly deltas 32..1; block = 4 or 8 wave
   partials) — one code path per kernel, satisfying the per-bucket
   fixed-tree requirement for C3 comparison.

## Validation state (honest classification)

- **Ran here (measured):** tier-1 shimmed `g++ -fsyntax-only` over the
  common header — PASSES clean (`validate_rocm_syntax.sh`).
- **Unproven here:** both `.hip` TUs need
  `hipcc --offload-arch=gfx950 -fsyntax-only -I<repo>/include -I<dir> <tu>`
  on any ROCm machine; script prints exact commands when hipcc is absent.
  Nothing has executed on gfx950 hardware; no performance claims.
