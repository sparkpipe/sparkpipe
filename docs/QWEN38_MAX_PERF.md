# Qwen3.8-2.4T-A95B performance plan

Baselines are measured on a single GB10 spark with the real FP8 packs
(timing harness: N decode steps x B sequences through the module).
Correctness and capacity findings live in QWEN38_MAX_AUDIT.md.

## 1. Where the time goes today (measured)

| slice | B | ms/step | tok/s |
|---|---|---|---|
| 1 GDN layer | 1 | 8.45 | 118 |
| 1 GDN layer | 8 | 36.99 | 216 |
| 1 GDN layer | 16 (tile path) | 26.93 | 594 |
| 1 GDN layer | 32 (tile path) | 39.85 | 803 |

- B=1: scalar MoE streams ~500 MB of expert weights per token per layer
  from LPDDR5X; 8.4 ms/layer, ~6 tok/s for the full model single-stream.
- The NEW tensor-core tile path (this branch) switches the grouped MoE to
  SparkLmExpertTileAllKernel at B >= 16: FP8_E4M3 block-128 decodes to
  BF16 fragments under wmma mma_sync. 2.75x throughput at B=16 and a much
  flatter per-step curve (26.9 -> 39.9 ms over B=16..32 vs the scalar
  4.4x growth over B=1..8). Sanitizer-clean on the real pack.
- Remaining levers, in order:
  1. B1 latency: the sm121 native B1 expert kernels are MXFP4-only; an FP8
     B1 kernel (or MXFP4 experts, see the capacity decision) is the next
     5-10x at B=1.
  2. Multi-slot, sync-free dispatch: the per-frame cudaStreamSynchronize
     serializes the pipeline; async completions let the next token's GDN
     overlap this token's MoE.
  3. Dense linears already take the tensor-core path at B >= 16 through
     SparkLmHostLaunchBatchedLinear; no work needed there.

## 1b0. Prefill + output performance estimates (16-spark ring, rev 2)

Rev 2 corrects rev 1's full-expert-set assumption: the grouped tile path
reads ONLY the experts a batch touches, so per-step expert bytes follow
E_touched x 83.8 MB (per expert: w1+w3 = 67 MB + w2 = 16.8 MB FP8), not the
whole set. Model anchors, both measured on spark4: replicated B=16 touches
~137 experts -> 11.5 GB in 14.3 ms = 804 GB/s effective; replicated B=256
touches ~508 -> 42.6 GB in 182.5 ms = 233 GB/s. Efficiency tracks the CTA
count per layer (2192 vs 8128), and TP16 keeps every batch in the low-CTA
high-efficiency regime (138 CTAs/layer at B=16 up to 512 at B=256), so
~800 GB/s applies across TP16 batches; the occupancy fix aims for ~900.

| Path | Aggregate | Per-sequence (B=1) | Notes |
|---|---|---|---|
| OUTPUT today (TP4xPP4 replicated, B=256) | ~39 tok/s | 0.78 s/token | measured anchors; 23-layer stage 6.0 s + screened head ~0.5 s |
| OUTPUT TP16, B=16 | ~190 tok/s | ~55 ms/token (~18 tok/s) | 721 MB experts/layer -> 0.90 ms -> 83 ms/step |
| OUTPUT TP16, B=32 | ~220 tok/s | ~55 ms/token | 1.25 GB -> 1.56 ms -> 144 ms/step |
| OUTPUT TP16, B=64 | ~290 tok/s | - | 1.91 GB -> 2.39 ms -> 220 ms/step |
| OUTPUT TP16, B=256 | ~450-500 tok/s | - | 2.68 GB -> 3.35 ms -> 308 ms/step; capped by the screened head (~500) |
| OUTPUT TP16 full-context ceiling (ctx 262144) | ~130 tok/s ANY B | ~7.5 ms/token | attention-bound: ctx x 1 KB per attn layer x 23 = 6.0 GB/token per rank |
| OUTPUT TP16 ctx 32768 | ~1.0K tok/s ceiling | ~0.9 ms/token | attention per token 0.75 GB |
| PREFILL today | ~1.3 prompt-tok/s | prompt = N decode steps | prefill frames refused |
| PREFILL after phase-2 kernels, replicated PP4 | ~1K tok/s | - | 52 GB per 64-chunk (experts 30.6 GB dominate) ~ 65 ms |
| PREFILL after phase-2 kernels, TP16, 4K prompt | ~5K tok/s | - | ~10.4 GB/chunk ~ 13 ms; experts 1.91 GB + attention 3.0 GB + dense 5.1 GB |
| PREFILL TP16, 32K prompt | ~1.6K tok/s | - | attention-bound: ~24.1 GB of the 31.5 GB per chunk |

Prefill notes: chunk-64 batches sit in the high-efficiency CTA regime;
weights amortize across the chunk (dense 5.1 GB once per chunk), so prefill
is NOT B=1-shaped. The quadratic causal-attention reads (per rank, one KV
head: L^2/2 x 512 B per attention layer) overtake the expert traffic past
~16K prompts, capping long-prompt prefill at ~1.6K tok/s at TP16. The
last-stage screened head adds ~2 ms/token (B>=2) and caps the output at
~500 tok/s until the head goes vocab-parallel. All TP16 rows assume the
collective wiring + head-sliced projections land; the residual all-reduce
is 16 KB/row (0.5% of the weight traffic) and does not move these numbers.

## 1b. Batch saturation measured (the honest single-node ceiling)

| B | full step (1 GDN layer) | GDN only (MoE disabled) | MoE share |
|---|---|---|---|
| 16 | 26.9 ms | 12.6 ms | 14.3 ms |
| 256 | 261.6 ms | 79.1 ms | 182.5 ms |

- Aggregate throughput saturates at ~10.5 tok/s (B=256, 92-layer
  extrapolation). The bisect plus one experiment isolate why:
  1. **The MoE tile path is CTA-latency bound, not weight-bandwidth
     bound.** At B=256 each expert has ~5 rows (one M-tile), so the
     25.8 GB is read once; the 182.5 ms comes from ~65K CTAs whose
     serial K-loops (decode + MMA per k-stage) leave the SM under-
     occupied. An M-loop variant (one CTA per (expert, N-tile), all
     M-tiles inside) was implemented and MEASURED: it REGRESSED
     (276.9 ms) - fewer, longer CTAs starve parallelism. The fix is
     more resident CTAs per SM (deeper software pipelining / smaller
     shared footprint), a common-kernel optimization.
  2. The GDN path costs ~0.3 ms/token from per-(row,head) kernel launch
     overhead (GdnStep/GatedNorm/conv: ~330 small blocks per token) -
     the CUDA-graph capture that removes it entirely.
- With BOTH fixed the single-node ceiling rises toward the weight-stream
  floor (~27 ms/layer), i.e. ~B/2.5s tok/s -> ~100 tok/s at B=256.
  Expert-parallel TP16 (below) reaches the same aggregate at B=16-32
  because each rank streams only 1.6 GB per layer - and its per-rank
  CTA counts shrink 16x, which also cures defect 1.

## 2. TP16 vs TP4xPP4: the collective schedule that makes all-reduce cheap

Per layer, per token, the tensors that cross TP ranks:

| tensor | bytes | required collective today |
|---|---|---|
| router gate logits | 2 KB (512 f32) | all-reduce |
| expert w2 partial sums | 16 KB (8192 bf16) | all-reduce |
| attention/GDN output residual | 16 KB | all-reduce |

The four techniques, in the order they are applied:

### T1. Replicate the router gate (removes one collective entirely)
The gate weight is 512 x 8192 BF16 = 8 MB per rank. Replicating it means
every rank computes the FULL top-10 locally - no gate all-reduce, ever.
The 8 MB is noise against 148 GB of experts. (TP16: 1 MB per rank if the
gate is itself column-parallel + all-gathered - but replication is
strictly better: zero traffic.)

### T2. Head-parallel attention (removes all attention collectives)
64 query heads over 16 ranks = 4 heads per rank, with the KV cache
head-local per rank. Q/K/V/O projections are head-parallel, softmax is
head-local, and the output residual is the ONLY thing that crosses. Same
for the GDN recurrence: 128 value heads / 16 ranks = 8 heads per rank,
and the delta-rule state is already per-head - the recurrence becomes
fully local with no collectives at all. The in/out projections for both
become column/row-parallel with the residual as the single shared tensor.

### T3. One fused collective per layer: reduce-scatter + all-gather = all-reduce
After T1/T2 the only tensor every rank needs in full is the residual
stream (8192 bf16 = 16 KB). The schedule per layer is:

    compute on the rank-local slice
        -> reduce-scatter the w2 + output partials (each rank ends with
           its own 512-wide slice)          [15 KB/rank ring traffic]
    next layer: all-gather the residual
        -> full 16 KB on every rank        [15 KB/rank ring traffic]

reduce-scatter followed by all-gather of the SAME tensor is algebraically
one all-reduce: 2(n-1)/n x 16 KB = 30 KB per rank per layer either way.
Two kernels, one fused collective buffer, one fence - no extra copies.
The RMSNorm partial sums (4 bytes per norm) ride the same buffer.

### T4. Overlap and latency hiding
- Dual-rail counter-rotating split-ring (the transport already lists it):
  layer i+1's gather starts on rail B while layer i's scatter still runs
  on rail A, hiding roughly half the ring latency.
- Async completions (transport completion queues) replace the adapter's
  per-send cudaStreamSynchronize, so no rank stalls on the fabric.
- Layer + collective captured in one CUDA graph island to amortize launch
  overhead.
- GPUDirect RDMA spec (dsv4 has one; qwen38's spec is host-rdma) removes
  the host bounce per collective.

### Cost after T1-T4

TP16 (92 layers, 1 collective/layer): 92 x ~10-17 us (16-node ring,
dual-rail) = 0.9-1.6 ms/token serialized; with T4's overlap it sits
entirely under compute (8.4 ms/layer today, ~0.3-1 ms/layer after the B1
tensor-core work).

TP4xPP4 (23 layers/stage, 1 collective/layer): 23 x ~6-10 us (4-node
ring) = 0.15-0.25 ms/token/stage, plus 3 PP handoffs of 16 KB each -
with async sends those overlap the next token's compute too.

Batching amortizes further: at B=16 the collective payload is 256 KB per
layer and the per-token cost drops to ~1 us.

### What is NOT on the table
- Quantizing the residual on the wire (bf16 -> fp8) saves 8 KB/layer but
  costs a second quantize pass and breaks the quality-first stance for a
  ~1% end-to-end gain. Revisit only if the fabric becomes the measured
  bottleneck.
- Expert-parallel all-to-all instead of the all-gathered schedule: 512
  experts need 10x16 KB = 160 KB/token/layer of all-to-all traffic vs
  30 KB for the residual all-reduce - ten times more fabric for nothing,
  because the all-gather already puts the full hidden on every rank.

## 3. TP16 readiness

- The adapter now takes "tp_degree" from the stage config at runtime
  (4 -> TP4xPP4 with {23,23,23,23} layer counts; 16 -> TP16xPP1 with
  {92}); deployment files for both topologies are committed
  (qwen38_fp8_tp16_host_rdma.spec.json / qwen38_fp8_tp16_stage.json).
- TP16 capacity reminder: 2.31 TiB / 16 = 148 GB/rank - same as TP4xPP4,
  and still above the ~107 GB usable on a GB10. The MXFP4 option in the
  audit doc resolves capacity AND unlocks the sm121 B1 expert kernels.
- TP16 removes all PP handoffs (one stage) and shrinks nothing else: the
  schedule above is the same 16-node ring either way.

## 4. The path to 100+ tok/s (measured, concrete)

Two complementary routes, both grounded in the measurements above:

A. Single node, batch serving (no fleet needed):
   - Fix the MoE tile M-loop re-decode (common kernel) + CUDA-graph the
     GDN path => ~27 ms/layer floor => B=256 gives ~100 tok/s aggregate.

B. Fleet, TP16 expert-parallel (the per-sequence win):
   - Each rank holds 32 experts = 1.61 GB/layer => ~1.8 ms/layer per rank
     => 92 layers = 165 ms/step => 6 tok/s at B=1, 97 tok/s at B=16,
     194 tok/s at B=32 - with the T1-T4 collective schedule the
     all-reduce sits fully under that compute.

## 5. Implementation order

1. (done) Grouped FP8 tensor-core tile path at B >= 16 - measured.
2. (done) Batch saturation bisect: MoE tile M-loop re-decode and GDN
   launch overhead are the two B-scaling defects.
3. Capacity decision: MXFP4 experts vs more nodes vs expert paging
   (148 GB/rank at FP8 does not fit; MXFP4 fits AND unlocks the sm121
   B1 kernels).
4. Common kernel: M-loop inside SparkLmExpertTileAllKernel (one weight
   decode per (expert, N-tile)) - coordinated with the other sessions.
5. CUDA graphs for the GDN layer sequence; multi-slot async dispatch;
   drop the per-frame stream sync.
6. FP8 B1 expert kernel (or adopt the sm121 MXFP4 ones) - the B=1 lever.
7. Replicate the router gate in the packer (8 MB duplicate, trivial).
8. TP rank-local packs (expert shards + column slices) + T2/T3 collective
   schedule + T4 (split-ring overlap, async sends, gpudirect spec).
9. Fleet window: measure, then tune the ring/rail split against reality.
