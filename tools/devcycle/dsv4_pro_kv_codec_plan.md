# DSV4 Pro KV cache codec selectability (BF16 -> FP8-E4M3)

## Current state (audited)

- The KV cache codec is a compile-time constant: SPARK_DSV4_PRO_KV_CACHE_CODEC
  = BF16. The pack header carries a kv_cache_codec id (1 = bf16) that is part
  of the module's pack-geometry compare; the adapter descriptor and the
  generic JIT budget calculator both already know FP8-E4M3 layouts
  (SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE_FP8_E4M3 with
  fp8_scale_block_size).
- The kernel scaffolding for quantization ALREADY EXISTS: the compressor
  emission path (SparkDsv4KvEmissionKernel) runs SparkDsv4QuantSimGroup -
  an in-place quantize->dequantize SIMULATION (E4M3 with per-64 block
  power-of-two scales, rope tail excluded, FP4 for the rotated variant) -
  so the cache currently stores BF16 with the quantization error simulated,
  not real FP8 bytes. SPARK_DSV4_MODEL_KV_QUANT_BLOCK = 64 matches the
  reference act_quant(kv[..., :-rd], 64, e4m3, ue8m0) exactly.
- The reference model itself quantizes KV activations to E4M3+UE8M0 for
  both the backbone and the DSpark draft attention - FP8 KV is
  reference-consistent, not an off-recipe approximation.

## What real FP8 KV needs

Layout per cache entry (512-dim KV row): E4M3 payload for the first 448
elements (7 x 64 blocks) + 7 UE8M0 scale bytes + the 64-element rope tail
kept BF16 = 448 + 7 + 128 = 583 bytes vs 1024 bytes today - a 1.75x
capacity gain per entry (the window, compressed history, and index entries
all shrink; the index cache is 128-dim and fully quantizable).

Changes:
1. Write side: CacheScatter (window) + KvEmission (compressed) + the index
   emission write E4M3+scales instead of BF16; keep the sim's amax/scale
   computation and emit the real bytes.
2. Read side: the sparse attention kernel, the indexer score kernel, and
   the DSpark draft attention dequant on read (E4M3 x 2^scale per block,
   rope tail read as BF16) - the dot loops swap SparkLmBf16ToFloat for a
   block-scaled LmE4m3ToFloat.
3. The window decision: the module currently keeps the sliding window
   unquantized (quality) and sims the compressed history. The reference
   quantizes everything. Start with compressed-history-only FP8 (matches
   the existing sim boundary, lower risk), then measure the window.
4. Plumbing: DONE - PRO_KV_CODEC knob in Makefile.pro (kv_bf16/kv_fp8
   module id fragments + -DSPARK_DSV4_PRO_KV_CODEC_FP8_E4M3), the pro model
   header codec override, and --kv-codec on the packer (header field only;
   the cache is runtime data so the pack payloads are unchanged).

## The test gate (we cannot decide without testing)

1. Correctness/invariants (available NOW): build the kv_fp8 module variant
   and run the existing single-spark GPU validation on sparkb against the
   val4 slice (0+4 has no MTP/head records, so the GA module loads the
   preview val4 pack with a --kv-codec fp8_e4m3 header). Pass = the
   quantized KV path runs end-to-end on hardware (nonzero_hidden invariant).
2. Token-exactness (the real decision): once the GA packs land, run the
   valtail slice (57+4, real head) with BOTH codecs and compare the output
   tokens and head scores. The quality gate: identical token streams (or a
   measured, bounded divergence in scores) between kv_bf16 and kv_fp8 -
   only then does the 1.75x capacity become a decision, not a gamble.
3. Long-context quality: repeat the comparison at a longer prompt (the
   128-token O128 batch is too short to exercise the compressed history
   meaningfully) before enabling FP8 KV for large-capacity deployments.

## Status

- Plumbing landed + verified (header field 1<->5, module ids kv_bf16/kv_fp8,
  defaults byte-identical to the deployed strings).
- Kernel conversion (write/read) is the next implementation step; the sim
  scaffolding bounds it to the two write sites and three read sites above.
