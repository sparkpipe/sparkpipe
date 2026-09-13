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
| `spark_dsv4_rocm_islands_e0_embed.hip` | embedding gather (verbatim port), stage-0 four-stream expand broadcast, simplified read-ahead kick (declared deviation) |
| `spark_dsv4_rocm_islands_l3_cache_transition.hip` | compressor step (state staging, boundary detection, plain + overlap softmax pooling, emitted flags) + cache scatter with exact ring-slot arithmetic — the C2 core, tree-free parity. Partial: fused KvEmission norm/rope/quant pipeline reserved behind requested_caps (S8) |
| `spark_dsv4_rocm_islands_l4_moe_routed.hip` | grouped MXFP4 gate/up activation + W2 down over stacked [experts x …] views driven by sealed-batch route maps, MoE pair reduce (weighted scatter-accumulate). PARTIAL BY GATE: route realization itself (GateRoute C2 pipeline) reserved behind requested_caps pending a line-exact port pass — never silently partial |
| `spark_dsv4_rocm_islands_l5_moe_shared.hip` | fused shared-expert W13 act (fp8-e4m3 weights, block-128 scales, limit-clamped swish-gate x up) + W2 down accumulate. Deviation flagged: activation-input UE8M0 qdq not yet bit-exact — first suspect if L5 C3 tolerances fail high |
| `spark_dsv4_rocm_islands_l1_boundary_norm_project.hip` | hcPost fold (line-faithful port), RMSNorm, Q/KV projections, frozen TP-shard interleave pack |
| `spark_dsv4_rocm_islands_l2_attention_qkv.hip` | query-head RMS+YaRN rope fused, KV post (norm/clamp/rope on an 8-wave CTA), wo_a/wo_b projections, PLUS the indexer top-k slice (AUDIT R1): byte-radix MSB top-k with canonical lower-slot tie break over per-row scores — C2 integer-pure, wave-size-independent histogram (see in-TU deviation note). Indexer score kernel + sparse attention remain S8 |
| `spark_dsv4_rocm_islands_f1_head.hip` | hc head reduce, MXFP4 shadow quantize (E2M1/E8M0 + certified error norm), screen two-pass bound, exact rescore argmax fused with ordered maxloc packing, maxloc unpack, resident token feedback, accum-u64-max. Partial island: markov bias term + DSpark drafter heads (freeze Q3 keeps them inside F1) remain unported behind the requested_caps guard |
| `rocm_syntax_shim.h` / `validate_rocm_syntax.sh` | local tier-1 proof tooling (never compiled into any image) |
| `rocm_tu_host_proof_shim.h` / `rocm_tu_host_proof.sh` | tier-1.5 FULL-TU host proof (AUDIT R1): device annotations inert, triple-angle launches rewritten into a variadic no-op so every kernel signature and every launch argument list type-checks under g++ -fsyntax-only. Static proof only — device compile stays with hipcc on ROCm |

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
