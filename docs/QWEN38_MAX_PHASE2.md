# Qwen 3.8 Max phase 2 - activate the sharded KV path and finish the tier

Phase 1 (merged via #653) landed the bring-up, the JIT KV tier wiring, and
the head-parallel attention kernels behind a fail-closed gate. This branch
activates them and finishes the remaining KV-path work.

## Scope

1. Activate TP-sharded attention (the KV 4x-capacity fix)
   - Wire SparkTpDeviceCollective into the resident stage module: the
     deployment stage config already declares the hidden_transport backend,
     ports 66620+, and the recursive-doubling / counter-rotating split-ring
     algorithms; dsv4's module is the reference integration.
   - Slice the attention linear views per rank: query/key/value row slices
     and o_proj row-parallel, so each rank projects its 16 Q heads and its
     one KV head only.
   - One fused residual all-reduce per attention layer (16 KB per row) -
     the T3 item from docs/QWEN38_MAX_PERF.md; drop the initialize refusal
     for SPARK_QWEN38_STAGE_TP_DEGREE > 1 once the combine is in place.
   - Verify: tp_degree 1 byte-identical, tp_degree 4 on the fleet ring.

2. Prefill attention kernels (the paged cache takes prompt chunks)
   - Chunked causal GQA over [base, base + token_count): K/V blocks written
     through the paged table and per-position head outputs, then module
     prefill-frame acceptance (the PrefillFrameView contract is already
     declared in the firmware header).
   - The adapter already advertises CAPABILITY_PREFILL; the module refuses
     it today.

3. GDN state paging through the tier
   - Fix the lane GDN-state record size (stage-local 128 x 128 x 128 fp32
     per GDN layer, ~8 MiB per layer per lane) and page it with the KV
     blocks; the work-control gdn record slot is reserved and dormant.

4. Fleet verification pass
   - TP4xPP4 end-to-end run on the 16-spark ring: exact reference vectors,
     the paged tier under a real provider, and the residual all-reduce
     measured against the split-ring budget in docs/QWEN38_MAX_PERF.md.

## Non-goals
- MXFP4 requantization of the vendor FP8 release (quality-first stance;
  revisit only with an explicit capacity decision).
- Expert-parallel MoE sharding (separate track, T2 in the perf doc).
