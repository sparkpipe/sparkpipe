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
