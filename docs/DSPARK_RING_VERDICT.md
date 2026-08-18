# RING VERDICT (spec vs lean, token-223 frame, pos 84)

## Method
- Spec side: dspark-k7-ring driver (bucket 8, current source + ring_anchor
  dump), k7 runtime (v4 pack), SPARK_DSPARK_DUMP=1 - 128 tokens, 516
  ring_anchor dumps (43 layers x 12 frames). Evidence: /tmp/spec_dumps_ring.txt
- Lean side: the PINNED LEAN DEPLOYMENT (the 3d962820 runtime, v3 pack) with
  the ef8fa302-era driver + the same ring_anchor dump (built at
  /tmp/sparkpipe-leandump - the ef8fa302 commit + the dump helpers + the
  __attribute__((unused)) fix + the -Werror drop in the rules file).
  The lean ran 128 tokens (the full O128 stream), 731 ring_anchor dumps
  (43 layers x 17 frames). Evidence: /tmp/lean_dumps_ring.txt

## THE VERDICT
ring_anchor per layer (the anchor's ring-slot KV the sparse attention reads):

- Layers 0, 1 (SWA, ratio 0): SAME (bit-identical fnv).
- Layer 2 (the first CSA, ratio 4): SAME.
- LAYER 3 (the FIRST HCA, ratio 128): FIRST DIFF - spec fnv 8b46d2c871 vs
  lean 4abb4eea6a (fp16-noise-level value diffs: v1=3bf0 vs 3d80, v7=3de0
  vs 3d30, ...). Every layer 3..41 then DIFFs (the state propagates).

The compression-ratio table (spark_dsv4_model.h SparkDsv4ModelCompressionRatios):
  layer 0,1 = 0 (SWA); layer 2 = 4 (CSA); layer 3 = 128 (HCA); alternating
  4/128 through layer 41; the tail = the MTP zeros.

=> The ring content is CLEAN through the SWA and CSA layers; the divergence
enters at the FIRST HCA LAYER (the 128:1 high-compression attention). The
HCA path = the indexer (index compressor Bf16LinearPair + IndexerPost with
RoPE/quantization + the HCA cache emission). The next bisect: the
Bf16LinearPair FLAT_8 (rows>1) vs FLAT_16 (rows==1) dispatch
(SparkLmHostLaunchBf16LinearPair / SparkLmSm121B1Bf16LinearPairPolicy) and
the IndexerPost quantization - the same exact-per-row pattern as the
pair/strided/expert/head fixes.

## Credit-flow bug (separate, logged for later - NOT needed for the ring)
The b1-in-k7 continuation path stalls: the rank-0 coordinator's completion
stuck in SparkModelResidentdPostTransport -> SparkHiddenTransportSend BUSY
(the output ring credit never returned). See DSPARK_VERIFY_HANDOFF_SESSION5B.md.
