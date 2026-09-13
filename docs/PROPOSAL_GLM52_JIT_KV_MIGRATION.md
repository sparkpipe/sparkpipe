# GLM52 JIT-KV onto the common seam — status + exact-field questions

Backlog item "GLM52 JIT-KV onto common core". Landed seam: SparkKvModelTable +
SparkKvBackendInitialize (include/sparkpipe/spark_kv_model_table.h,
cache/kv_model_table.c, HEAD 3b7a768). This doc records what is landed, what
remains, and the exact-field questions that block a correct module rewrite.

## What is already landed this turn

- `model-families/glm52/include/sparkpipe/spark_glm52_kv_geometry.h` — the GLM52
  geometry fill `SparkGlm52KvFillCapacityRequest` (mirrors `spark_k3_kv_geometry.h`):
  layout COMPRESSED_KEY_VALUE (BF16 today), 78 layers, 64 heads, qk_nope 192,
  value head 256, compressed 576 (latent 512 + rope 64), position 64,
  bytes_per_scalar 2, fp8 scale block 128, block 64 tokens, index-key 0.

## What remains (the module rewrite, blocked on the questions below)

The resident stage's `SparkGlm52AllocateCaches` (`spark_glm52_resident_decode_stage_module.c:641`)
owns a raw KV: a device `kv_cache` buffer (layer-major, 64 tokens/page x 576
elements x bf16), a redundant identity `page_table`
(`SparkGlm52BuildPageTable` :615), and a separate `index_cache`. The kernel
reads `wave->kv_cache + layer*stride + page_table[slot*pages+page]*page_bytes`
(`spark_glm52_resident_decode_stage_cuda.cu:263,369`). The migration replaces
this with SparkKvBackendInitialize + a SparkAdmissionPolicyTable.predicate tail
(the admit path at `spark_glm52_resident_decode_stage_module.c:1566` builds the
table; the predicate hook runs before the BUSY check per spark_admission.h:173).

## Exact-field questions for the seam owner (kv-cache agent)

Q1. **Layout enum.** GLM52's KV is BF16 today: the stagepack header check
    requires `kv_cache_codec == SPARK_WEIGHT_CODEC_BF16` (`...module.c:219`) and
    the alloc uses `sizeof(uint16_t)` (:655). The proposal's per-model table
    says COMPRESSED_KEY_VALUE_FP8. Do I fill layout = COMPRESSED_KEY_VALUE (BF16,
    matches today) or COMPRESSED_KEY_VALUE_FP8_E4M3 (target), and with which
    bytes_per_scalar (2 vs 1)?

Q2. **block_token_count 64 vs 128.** The proposal table says "64/128"; the
    current page is 64 tokens (`SparkGlm52BuildPageTable` :621). Canonical value?

Q3. **Arena layer striding.** GLM52's raw `kv_cache` is layer-major
    (`kv_cache + layer*kv_layer_stride_bytes + page*main_page_bytes`, :657,:369),
    with `key_block_stride_bytes` = one 64-token block (73728 B) and the layer
    term OUTSIDE the arena. SparkKvCacheArenaResolveBlock computes
    `key_device_base + key_block_stride_bytes*resident_slot_index` with no layer
    term (cache/kv_cache.c:1023). How do I express 78 layers: key_device_base at
    layer 0 and the driver adds `layer*kv_layer_stride_bytes` before calling the
    arena, or `key_block_stride_bytes = layer_count*73728` with layer_count=1?

Q4. **Value side.** GLM52 stores one fused compressed KV_A; the value is
    reconstructed from the latent. Is `value_device_base`/\`value_block_stride_bytes`
    = 0 (key-only), and does the copy primitive move only the key bytes
    (64*576*bytes_per_scalar)?

Q5. **Copy primitive page span.** Is one "page" one 64-token block across ALL 78
    layers (page_bytes = 78*73728) or per-layer (page_bytes = 73728)? This drives
    `page_store_config.page_bytes` and the copy function's `bytes`.

Q6. **Predicate prepare/commit/abort split.** On a CACHE_RELEASE frame the
    predicate should ReleaseLane + accept (short-circuit before BUSY). On a
    normal frame, is "prepare" = SparkKvPageCachePrepareLane (in the predicate,
    with predicate_context = module state) and "commit"/"abort" =
    SparkKvPageCacheCompleteLane / RollbackLaneTransaction driven by the frame's
    completion path (NOT the predicate)? Please confirm the exact call sequence.

## Landing plan once answered

1. Add `SparkGlm52KvFillModelTable` (arena_configuration: key_device_base =
   `state->kv_cache`, blocks[], resident_slot_logical_block_indices[];
   page_store_config: copy_function = SparkGlm52PageCopy, page_bytes per Q5;
   4 page-directory arrays: entries/sequences/hash_bucket_heads/
   entry_indices_by_logical_page) and call SparkKvBackendInitialize in
   SparkGlm52InitializeState (replacing SparkGlm52BuildPageTable).
2. Add SparkGlm52PageCopy (cudaMemcpyAsync device<->host of the page bytes).
3. Add SparkGlm52AdmissionPredicate (Q6) and set table.predicate/
   predicate_context in SparkGlm52ResidentDecodeStageAdmit.
4. Rewire the kernel page resolution to the arena resolve (drop the identity
   page_table) — the net-negative that offsets the new fill/predicate lines.

Blocked on Q1-Q6; the geometry header above is the safe, correct prefix.
