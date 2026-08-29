# Qwen3.8-27B TP4 (qwen38_27b) phase 2 performance plan

Measured on the spark0-3 TP4 band (4 x GB10, hidden-transport verbs backend).
Quality gate on every number: Qwen38-TP4-E2E-PASS (per-rank agreement plus
64-token reference comparison, bit-identical tokens). Fresh re-verification:
B1 13.2 tok/s (75.6 ms) with speculation D=2, e2e PASS, 0 mismatches.

**Accounting discipline:** TP4 shards columns, not the batch - every spark
processes every token through its shard, so the band tok/s IS the per-spark
tok/s. With N sparks the band must reach N x the single-spark tok/s to hold
per-spark throughput; anything less is scaling loss, anything claimed as a
multiple of the spark count is marketing. Comparison basis is community
spark releases only (same hardware class), at the same precision and batch.

## Where we sit against the practical BF16 ceiling (B1, per spark)

Per spark, per token, the weights streamed are 13.5 GB (the rank shard of
all 64 layers plus head plus MTP). Measured device read bandwidth is
225.9 GB/s, so the pure weight-stream floor is 59.8 ms.

- plain B1: 83.0 ms = 72% of the weight floor.
- spec D=2: 74.8 ms = 80% of the weight floor.
- The step's other serialized work (129 collectives ~3.5 ms, head ~2.8 ms,
  GDN core + attention + norms + ~1000 launches ~10 ms) puts the practical
  BF16 ceiling near 76-78 ms; against that, plain is ~92% and spec is ~98%.

So: we are near the practical BF16 ceiling, not 80% of it. The remaining
BF16-only headroom is ~1-3 ms (head shadow path, GDN small-kernel fusions)
- a +25% at BF16 requires shrinking the bytes, not tuning.

## BF16 weight lossless-compression analysis (measured on the real weights)

Analyzed all 617 BF16 2-D tensors: 27.78B values, 55.55 GB.

- zero fraction: 0.000000 (nothing to skip)
- order-0 entropy of the 16-bit stream: **10.52 bits** -> 1.52x lossless
  (a single ANS/Huffman table over the bf16 codes)
- mantissa entropy 6.97 bits, exponent entropy 2.60 bits -> a two-stream
  codec (entropy-coded exponent + raw 7-bit mantissa) bounds at 9.57 bits
  = 1.67x lossless
- per-tensor distinct exponents: 18-33 of 256 - exponent locality is high,
  which is why the exponent stream compresses so well

Plan for the +25% at identical quality:
1. ANS codec on the weight stream, decompress inside the small-batch GEMM
   kernels' tile staging (decode to shared memory, compute bf16 - bit-
   identical results). Per-spark stream falls 13.5 GB -> 8.9 GB -> weight
   time 59.8 -> ~39.5 ms, so B1 lands ~63 ms (~15.9 tok/s, +31%) before the
   head-shadow and GDN trims, which push toward ~16-17 tok/s.
2. Fallback if the fused decode under-performs: keep the tiled kernels
   reading compressed tiles from L2 via a separate decode kernel.

## Phase 2 targets (expected value order)

1. Lossless weight-stream codec (measured 1.52x headroom) + decompress-in-
   GEMM: the same-quality +25-30% at B1/B8.
2. Sharded-delta B64 collective (1280 of 5120 columns shipped instead of
   the full 655,360 bytes): B64 step ~404 -> ~150 ms.
3. B1 head shadow path (rows=1 full-vocab read 2.8 ms -> screened ~1.5 ms).
4. B8 GDN branch overheads (~470 us/layer above the ~215 us GEMM floor).
5. Collective per-op latency (~27 us x 129/step): revisit multi-outstanding
   stream-ordered submissions once the hidden-transport backend's
   multi-outstanding failure is root-caused.
6. N-spark scaling validation (TP8/TP16): verify per-spark throughput holds
   as the shard shrinks - the N x single-spark rule is the acceptance test.

Verification gate stays: module GPU validator (TP4 standalone) plus
Qwen38-TP4-E2E-PASS on every landed change.
