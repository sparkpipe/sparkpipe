# K3 TP16 expert repack — TILE_K=32 INTERLEAVED_B design record

Why TP16 needs 32-element tiles (the arithmetic, settled by audit):

- The expert w1 is split on BOTH axes: the rank owns its k-tile range (the
  latent slice: 3584/16 = 224 columns = 7 x 32) AND its gate|up cell ranges
  (the intermediate slice: 6144/16 = 384 = 192 gate + 192 up elements).
  The rank's shard is the DIAGONAL subgrid (its tiles x its cells); the
  cross subgrids are read by no GEMM and live in no pack.
- The expert w2 k-splits on the SiTU intermediate, which is CONTIGUOUS per
  rank (the gate|up halves share cell offsets): 3072/16 = 192 = 6 x 32.
- 128-element tiles divide neither 224 nor 192; 64 divides neither (224/64
  = 3.5); 32 divides both. The packer's interleave_geometry already closes
  at tile_k 32 (16-byte payload row = 16 rows x 1 scale byte; the E2M1
  group is 32, so one scale byte per neuron per tile).
- TP 1/2/4/8 keep the 128-element tiles (28 and 24 tiles divide evenly);
  only TP16 packs switch. The manifest carries tile_k per tensor, so both
  formats coexist and the serving tier picks per pack.

Work items, in order:

1. LANDED: tools/k3_pack.py takes expert_tile_k (CLI arg 6);
   interleave_geometry already closes at 32.
2. LANDED: tools/k3_shard.py slices w1 as the diagonal subgrid (k-tiles +
   cells) and w2 on whole tiles, both at the pack's tile_k, repricing
   carries tile_k; the refusal names the tile size. tests/test_k3_shard.py
   proves the diagonal at 128- and 32-tile packs and the refusal.
3. The GEMM wave (runtime/gemm.cuh + inference/kernels/gemm.cuh +
   inference/kernels/tile.cuh):
   - the interleaved launcher template gains TILE_K (currently forced 128);
   - LmGemmSharedBytes: b_bytes = 17 x (TILE_N/16) x (TILE_K/2);
   - the INTERLEAVED_B staging at TILE_K 32: 16-byte cell rows -> the B
     tensor map uses SWIZZLE_NONE (a 16-byte box inner dim cannot carry the
     64-byte swizzle); the scale read is byte r16 + 0 of row c*17 + 16,
     linear (no swizzle XOR, one group per tile);
   - the A side at TILE_K 32 stays the single-block stage (32 BF16 = 64
     bytes, the ordinary 64-byte swizzle) - the two-block hack exists only
     for the 128-byte rows;
   - static_assert TILE_K == 128 becomes TILE_K == 128 || TILE_K == 32;
   - runtime/gemm.cuh's interleaved encode path takes the swizzle from the
     variant.
4. The dispatch + layer: a per-tensor tile_k field (filled from the
   manifest's interleave.tile_k) selects the 128- or 32-tile instantiation
   at the w1/w2 launch sites; unity.cu gains the TILE_K=32 explicit
   instantiations.
5. tests/test_k3_interleave_gemm.cu: a 32-tile numerical case against the
   128-tile reference (same payload, different tiling -> bit-identical
   products).
6. Then TP16 deployment: pack the full model with expert_tile_k 32, slice
   16 ways, and the existing TP16 configs/NCCL degree 16 carry it.

NOT part of the fix (rejected): replicated w2 + intermediate all-gather -
16x the down-GEMM FLOPs and a third collective per layer.
