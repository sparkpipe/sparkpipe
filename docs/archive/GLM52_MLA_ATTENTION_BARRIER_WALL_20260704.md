# MLA Attention: the Barrier Wall, and the 2-Pass Unification - 2026-07-04

## What the PR87 numbers established

The projection rewrite worked: QKVO + output projection went 73.9 ms ->
36.4 ms at B256 (17.1 + 19.3) and now scale sub-linearly (1.4x / 1.15x
across B128->B256) - weight amortization is finally real on that bucket.
But MLA attention did not move at all (36.8 -> 36.5 ms at B128), which is
itself the diagnosis: the uint4 dot and the expf hoist cut ALU and load
instructions substantially, so the kernel is not ALU-bound. Phase 5 was
verified to contain only the decode attention kernels (DSA selection and
rope/KV-write are separate phases), so the cost is inside the attention
kernel proper.

## Root cause: 768 block barriers per (sequence, head)

The attention execution enum has exactly one valid mode
(ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX), so AttentionOnlineKernel was
the production decode kernel. Its structure: 2048 candidates / 8 warps =
256 sequential rounds, each with (now three) __syncthreads plus a
dependent warp-reduce chain - 768 block-wide barriers per (seq, head)
block, 8192 blocks per layer at B128. Per-round latency times 256 rounds
times the block waves per SM lands in the measured range; the work I
removed in PR87 was hidden under the barrier chain, which is why the
phase time was unchanged.

The base AttentionKernel already has the right structure: score all 2048
candidates with warps striding freely (no per-round barrier), one block
max-reduce, one exp+sum pass, one normalized AV walk - about 4 barriers
total instead of 768, computing the identical exact softmax (online
rescaling is an incremental reformulation of the same result; deltas are
fp-association-level, same class PR87's token-match already absorbed).

## Change

Both attention execution branches (bf16 and fp8-KV) now launch the
2-pass kernels. Deleted: AttentionOnlineKernel, AttentionFp8KvOnlineKernel,
the never-launched PagedPrefillAttentionOnlineKernel, and the orphaned
WarpResolvePagedPrefillCacheSlot helper (-678 lines). The fp8 base kernel
received the same normalize-once treatment as the bf16 one (probabilities
divided by the block sum once in shared, removing a division per
dim x 2048 candidates). The bf16 base kernel already carries the uint4
WarpDotProduct from PR87. Launch site if/else collapsed since both
branches were identical. Net: 19 insertions, 741 deletions.

If the barrier theory is right, the MLA attention phase drops materially
at every bucket. If it instead lands at a floor, that floor is the KV
byte traffic - see below.

## The endgame for MLA attention scaling: absorbed decode

Decode currently reads per-head keys and values: per (sequence, slot)
that is 64 heads x (192 nope + 256 value) x 2 B ~ 57 KB of unique bytes,
irreducibly linear in batch. The MLA absorbed formulation reads only the
latent cache row (576 x 2 B = 1.15 KB per slot, shared by all 64 heads):
score[h] = (W_UK[h]^T q_nope[h]) . latent[0:512] + q_rope[h] .
latent[512:576], AV accumulates the 512-dim latent, and W_UV applies per
head afterward. ~50x fewer KV bytes for ~30 MFLOP/token of extra small
GEMMs (q absorption 64x192x512, W_UV apply 64x512x256). Everything
needed is already in mla_cache_bf16; the kv_b weights contain W_UK/W_UV
as row ranges. Required: split views of kv_b, two small batch GEMMs, one
new attention kernel over the latent (heads share the slot row -
smem-tile it per block), host plumbing behind a new execution mode.
Propose as the next PR, gated on two numbers from spark2: the sweep's
decode context length (grounds the current traffic math) and the
post-this-PR MLA phase time.

## LOCAL_MOE: grounded so far, one artifact needed

Phase 8 contains the router top-k kernels, the FlashInfer b12x launch
(generated GEMM + its prepare/finalize), and the residual add. The
launcher's per-launch memsets are all expert-count int32 arrays and
scalars - not a traffic source. Against the earlier Nsight numbers, the
phase carries ~14 ms at B128 beyond generated-GEMM + fc2-finalize, and
the generated GEMM itself grows 1.76x across B128->B256 with weight
coverage ~flat. Both need the same artifact before any code moves: a
per-kernel-name time split of the LOCAL_MOE phase from the existing
build/glm52_pr87_nsys_node_stage18_b{64,128,256}.nsys-rep reports
(nsys stats --report cuda_gpu_kern_sum scoped to the phase range works).
No blind codegen or wrapper changes until that lands.

## Validation on spark2

1. Full token-match at B64/B128/B256 (deltas are fp-association-level).
2. tests/test_glm52_exact_pp13_prefill_hidden.py.
3. Stage sweep with the phase split; MLA attention is the number to watch.
4. FP8 path inherits both changes (kernel swap + normalize-once) - the
   pending FP8 sweeps cover it.
