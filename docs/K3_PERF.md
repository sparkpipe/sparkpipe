# K3 serving-path performance audit and improvement plan

Measured on sparka, real rank pack, stage 0 (24 layers), 1 token:

- Cold first step: ~2.5 s - one-time JIT, tensor-map encodes, shared-memory
  opt-ins. Amortises to nothing in steady state.
- Warm step: **55.5 ms** = 2.3 ms/layer. The slice issues ~35 kernels per
  layer (~840 per step, ~66 us of host enqueue each - the HOST spent ~55 ms
  enqueueing each step).
- Graph replay of the captured slice: **54.2 ms** (step 3 in the gate) -
  bit-identical output. The kernels themselves take ~54 ms, so the path is
  MEMORY-BANDWIDTH-BOUND on weight streaming, not launch-bound: each layer
  streams ~0.8-1 GB of BF16 spine + MXFP4 expert weights per token through
  the persistent GEMMs (LPDDR5X ~273 GB/s => ~2.3 ms/layer). The capture's
  win is real but wall-clock-side: it removes the ~55 ms serialized host
  enqueue from every submit, roughly halving the per-step wall time at B1
  and freeing the host thread for the residentd loop and the other ranks.
  Steady-state decode: ~55 ms per stage per token (the PP4 stages
  pipeline), i.e. ~18 tokens/s for the 16-spark K3.

## Improvements, in dependency order

1. **CUDA-graph capture** (the slice's own capture audit: legal once no
   host traffic remains on the layer path). The per-step dense-offset H2D
   is now a device-side kernel (K3RunnerDenseOffsetsKernel); the remaining
   host traffic is the collective tier's staging. Target: the ~3.3k
   launches/token become one graph replay per shape.
2. **The fused per-layer collective (LANDED)**: attention_out | hidden |
   shared_out pack into contiguous 14 KB segments instead of three
   separate exchanges - the two-phase split (item 5) ships attention_out
   alone before the MLP-side retrieval and hidden | shared_out after the
   MLP, so each exchange carries the largest frame its phase allows and
   the wire utilization stays high. The host-TCP tier implements it; the
   device tier replaces the whole block with stream-ordered combines (no
   sync at all).
3. **The device-direct tier (LANDED)**: the adapter parses a
   `device_collective` object (backend nccl | hidden_transport, peer
   hosts, ports, timeouts), applies the topology, and hands the runner a
   completed config. The runner creates the collective with the K3 combine
   kernels (hidden transport) or plain NCCL, and the per-layer hook issues
   ONE stream-ordered all-reduce of the fused 3x7168 buffer with a
   completion that folds the summed segments into the AttnRes partial on
   the same stream. No sync, no host staging, no H2D on the hot path.
   NCCL's unique id bootstraps through the shared control port (64620);
   the host TCP tier stays available as the TP4 fallback and is skipped
   for TP16 (its 4-rank cap).
4. **TP16-ready geometry (LANDED)**: the adapter derives the PP stage
   split from `world_size / tp_degree` (16/16 -> PP1), the module takes
   slice bounds from the pack manifest when the runner passes
   SPARK_K3_MODULE_DERIVE_SLICE, the bound-layer cap is 93, and the
   generator emits TP16 configs (16 peer hosts, no host tier). The degree
   divisibility audit below holds. The expert w2 remains the ONE TP16
   blocker: its 24 k-tiles cannot split 16 ways in whole tiles, and an
   unbalanced whole-tile split CANNOT match the gate|up output split (a
   rank's down k-slice must equal the intermediate its w1 slice computed -
   192 = 96+96 elements at TP16 = 1.5 tiles) - the sharder now REFUSES
   TP16 loudly instead of emitting inconsistent packs, and the fix is the
   64-element half-tile repack (pack V3 + a TILE_K=64 INTERLEAVED_B
   variant). Everything else (configs, NCCL degree 16, geometry, pools)
   is staged.
5. **Two-phase layer collective (LANDED)**: the hook now fires after the
   attention half (phase 0) AND after the MLP half (phase 1). Phase 0 is
   required for correctness, not just speed: the MLP-side AttnRes retrieval
   is documented to read the POST-attention partial, and the old single
   post-MLP hook let the sharded path retrieve a partial missing the
   attention contribution. Each phase packs and all-reduces its own
   segment(s) with the fold landing on the submission's stream (no legacy
   default stream), and at tp_degree 1 the hook no-ops because the layer
   folds its own projections.
6. **Per-shape CUDA-graph capture (LANDED, gated)**: the runner captures
   the dense-offset kernel + the whole slice into a cudaGraphExec_t keyed
   by rows (warm-direct first submit, capture on the second, replay after),
   gated on a non-default stream and a capture-safe tier (NCCL device
   collective or tp_degree 1; the host tiers' syncs/staging are not
   replayable and self-disable the path). The slice audit (in slice.cuh)
   held: no host traffic on the layer path, host-resolved branches bake
   in, tensor-map encodes are steady-state cache hits. The single-spark
   gate now replays the captured slice and compares it against the direct
   launch (0 mismatches beyond the ULP limits).
7. **Overlap** (revised): the per-layer ARs sit on the critical path for
   B1 decode by construction - the MLP-side retrieval needs the summed
   attention output and the next layer needs the summed MLP outputs - so
   there is no compute to hide them behind. The AR-overhead reductions are
   the fused messages, the device-direct tier, and the graph capture
   above. Remaining ideas: a per-submission payload width (the collective
   currently fixes local_hidden_dimension at create time, so a 1-segment
   phase still ships the 3-segment frame) and the balanced TP16 w2
   half-tile repack.

## TP16 readiness audit (degree 16 divisibility)

| tensor | full | /16 | status |
| --- | --- | --- | --- |
| heads (KDA + MLA) | 96 | 6 | ok |
| vocab (embed/lm_head) | 163840 | 10240 | ok |
| kda_out input | 128 | 8 | ok |
| mla_out input | 12288 | 768 | ok |
| routed latent | 3584 | 224 | ok |
| expert w1 output (cells) | 6144 (384) | 384 (24) | ok |
| expert w2 k-tiles | 24 | 1.5 | was REFUSED - now UNBALANCED SPLIT |
| shared w1 halves | 1024 | 64 | ok |
| shared w2 input | 2048 | 128 | ok |
| dense gate_up halves | 16896 | 1056 | ok |
| dense down input | 33792 | 2112 | ok |

The sharder now splits the w2 across TP16 UNBALANCED (the first 8 ranks
take 2 of the 24 k-tiles, the rest 1) - whole tiles, scales intact, the
per-rank dims the layer already reads make it explicit. The balanced
follow-up is the 64-element half-tile repack (a pack-format change the
GEMM would take as an INTERLEAVED_B TILE_K=64 variant with 32-byte cell
rows).
