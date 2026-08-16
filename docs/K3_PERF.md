# K3 serving-path performance audit and improvement plan

Measured on sparka, real rank pack, stage 0 (24 layers), 1 token:

- Cold first step: ~2.5 s - one-time JIT, tensor-map encodes, shared-memory
  opt-ins. Amortises to nothing in steady state.
- Warm step: **55.5 ms** = 2.3 ms/layer. The slice issues ~35 kernels per
  layer (~840 per step), so ~66 us per launch - the path is LAUNCH-BOUND,
  not compute-bound (the GEMM ridge analysis says compute is ~60x spare).

## Improvements, in dependency order

1. **CUDA-graph capture** (the slice's own capture audit: legal once no
   host traffic remains on the layer path). The per-step dense-offset H2D
   is now a device-side kernel (K3RunnerDenseOffsetsKernel); the remaining
   host traffic is the collective tier's staging. Target: the ~3.3k
   launches/token become one graph replay per shape.
2. **The fused per-layer collective (LANDED)**: attention_out | hidden |
   shared_out pack into ONE 43 KB message instead of three 14 KB ones -
   three syncs become one and the wire utilization triples. The host-TCP
   tier implements it; the device tier replaces the whole block with one
   stream-ordered combine (no sync at all).
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
   divisibility audit below holds; the w2 k-tile split stays unbalanced
   until the TILE_K=64 variant lands.
5. **Overlap**: the next layer's norm/qkv projections do not depend on the
   partial; splitting the slice's per-layer sequence around the collective
   (aux stream + events) hides the AR behind the next layer's prologue.
   Best done together with the graph capture (the finite shape set).

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
