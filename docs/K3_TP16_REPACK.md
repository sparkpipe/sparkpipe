# K3 TP16 expert repack — TILE_K=32 INTERLEAVED_B design record

Why TP16 needs 32-element tiles (the arithmetic, settled by audit):

- The SHIPPED scheme (input-split w1 / output-split w2, the numerics fix
  that retired the diagonal subgrid) splits ONE axis per expert tensor.
  Expert w1 input-splits on whole k-tiles - the rank's latent slice (3584/16
  = 224 columns = 7 x 32) addresses only its own tiles - and keeps the
  gate|up output FULL: sum(SiTU(p)) != SiTU(sum(p)), so the full-width
  partial must all-reduce BEFORE the SiTU, which is exactly what the
  diagonal form (its tiles x its cells) got wrong - it applied SiTU to a
  rank-sliced partial. Expert w2 output-splits on whole 16-neuron cells
  (3584/16 = 224 cells) with the intermediate input FULL: after that
  all-reduce every rank holds the whole SiTU intermediate, so a rank's
  latent cells read the whole k axis (the input-split form would pair its
  cells with its tiles alone and drop the cross terms).
- One split axis each makes the tile arithmetic w1's alone: the sliced
  extent is the k-tile count, latent/tile_k. At TP16 the rank's 224-column
  slice must be whole k-tiles: 128 gives 28 tiles over 16 ranks - not
  whole; 64 gives 56 - still not whole (56 % 16 = 8); 32 gives 112 = 7 per
  rank. The packer's interleave_geometry already closes at tile_k 32
  (16-byte payload row = 16 rows x 1 scale byte; the E2M1 group is 32, so
  one scale byte per neuron per tile).
- TP 1/2/4 keep the 128-element tiles (w1's 28 k-tiles divide by 1, 2 and
  4; w2's 224 cells divide every power of two up to 16). TP8 does NOT:
  28 % 8 = 4, so the w1 k-tile range does not split — TP8 packs switch to
  tile_k 32 just like TP16 (there the w1 slice is 224 = 7 x 32-tiles,
  whole; w2's cell axis is tile-independent and whole at both). The
  manifest carries tile_k per tensor, so both formats coexist and the
  serving tier picks per pack.

Work items, in order:

1. LANDED: tools/k3_pack.py takes expert_tile_k (CLI arg 6);
   interleave_geometry already closes at 32.
2. LANDED, then superseded by the SiTU numerics fix: tools/k3_shard.py first
   sliced w1 as the diagonal subgrid (k-tiles + cells); that applied SiTU to
   a rank-sliced partial, so the shipped form input-splits w1 on whole
   k-tiles (gate|up output full, all-reduced before the SiTU) and
   output-splits w2 on whole cells (input full), both at the pack's tile_k;
   repricing carries tile_k and the refusal names the tile size.
   tests/test_k3_shard.py proves the shipped splits at 128- and 32-tile
   packs and the refusal.
3. LANDED: the GEMM wave (runtime/gemm.cuh + inference/kernels/gemm.cuh +
   inference/kernels/tile.cuh). The interleaved launchers take TILE_K (128
   default, 32 for the TP16 grid); LmGemmSharedBytes and the kernel's
   b_stride scale with TILE_K/2; the B map at 32 carries 16-byte rows and
   LmTensorMapBoxSwizzleBytes maps them to SWIZZLE_NONE automatically; the
   consume's scale read is linear at 32 (one E8M0 byte per neuron) and the
   64B-swizzle XOR at 128; the two-block A stage is gated on TILE_K == 128
   (32-element BF16 rows are 64 bytes and take the ordinary single box);
   the runtime encode takes tile_k; the static asserts admit 128 | 32.
   The adapter .so compile-gate PASSES on sparka (sm_121a).
4. LANDED: the dispatch + layer. K3LayerWeights/K3LayerBuffers carry
   expert_tile_k; SparkK3PackLoadInterleaveTileK reads the manifest's
   per-tensor interleave.tile_k; the dispatch fills it and the w1/w2
   launch sites select the 32- or 128-tile instantiation. unity.cu gained
   the six TILE_K=32 instantiations. Also fixed on the way: the pack
   loader's uint64 offset fix read the unwritten uint32_t value - the
   assignments now use wide_value.
5. LANDED: tests/test_k3_interleave_gemm.cu gained the TILE_K=32
   numerical case (same logical weights on the 16-byte-row grid, same
   expected values; the encode diagnostic covers both tile sizes). GATE
   PASS on sparka: direct + indirect + tile_k 32, and the single-spark
   step gate re-PASSES bit-deterministic (the 128-tile paths are
   regression-clean).
6. The extra 16-byte-row couplings the gate flushed out, all fixed:
   LmTensorMapPlanBuild rejected SWIZZLE_NONE for non-INT tensors (now
   allowed for 16-byte boxes); LmLaunchPlanBuild priced the cell rows at
   64 bytes and rejected the 16-byte pitch (now TILE_K/2 and explicit);
   the runtime's staged B geometry hardcoded depth 64 (the mbarrier
   expect was 4x the arriving bytes - a hang, fixed to TILE_K/2);
   LmMxfp4::Fragment read through the swizzle xor (span 0 on 16-byte
   rows - reads a linear byte instead).
6. Then TP16 deployment: pack the full model with expert_tile_k 32, slice
   16 ways, and the existing TP16 configs/NCCL degree 16 carry it.

NOT part of the fix (rejected): replicated w2 + intermediate all-gather -
16x the down-GEMM FLOPs and a third collective per layer.
